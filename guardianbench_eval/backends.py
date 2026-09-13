from __future__ import annotations

import base64
import copy
import io
import math
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .data import ModelInput


class BackendError(RuntimeError):


    def __init__(self, message: str, *, error_type: str = "BackendError", status_code: int | None = None):
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code


@dataclass
class Prediction:
    completion: str
    metadata: dict[str, Any] = field(default_factory=dict)


IMAGE_PREFIX = "<image>"

_API_RESERVED = {
    "model", "messages", "stream", "n", "temperature", "top_p", "seed",
    "max_tokens", "max_completion_tokens", "api_key", "effort", "reasoning_effort",
}
_TEMPLATE_RESERVED = {
    "conversation", "messages", "tokenize", "return_dict", "return_tensors",
    "add_generation_prompt", "continue_final_message", "text", "images",
    "videos", "audio",
}
_PROCESSOR_RESERVED = {
    "pretrained_model_name_or_path", "revision", "trust_remote_code",
    "local_files_only", "token", "use_auth_token",
}


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError(f"{name} must be an object with string keys")
    return copy.deepcopy(value)


def _reject_keys(value: dict[str, Any], reserved: set[str], name: str) -> None:
    if value.keys() & reserved:
        raise ValueError(f"{name} cannot override reserved inference parameters")


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, minimum: float, maximum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{name} is outside its supported numeric range")
    return float(value)


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def validate_backend_config(backend_name: str, config: dict[str, Any]) -> dict[str, Any]:

    config = _mapping(config, backend_name)
    if backend_name == "openai":
        if "api_key" in config:
            raise ValueError("Set credentials through api_key_env, not the configuration file")
        config["model"] = _nonempty_string(config.get("model"), "model")
        config["api_key_env"] = _nonempty_string(config.get("api_key_env", "OPENAI_API_KEY"), "api_key_env")
        key = config.get("max_tokens_parameter", "max_completion_tokens")
        if key not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("max_tokens_parameter must be max_tokens or max_completion_tokens")
        config["max_tokens_parameter"] = key
        config["max_tokens"] = _integer(config.get("max_tokens", 1024), "max_tokens", 1)
        for name, upper in (("temperature", 2.0), ("top_p", 1.0)):
            if config.get(name) is not None:
                config[name] = _number(config[name], name, 0.0, upper)
        effort = config.get("effort")
        config["effort"] = _nonempty_string(effort, "effort").strip() if effort is not None else None
        config["extra_body"] = _mapping(config.get("extra_body", {}), "extra_body")
        _reject_keys(config["extra_body"], _API_RESERVED, "extra_body")
        config["timeout"] = _number(config.get("timeout", 120.0), "timeout", 0.001)
        config["max_retries"] = _integer(config.get("max_retries", 2), "max_retries", 0)


        config["base_url"] = _nonempty_string(config.get("base_url", "https://api.openai.com/v1"), "base_url")
    elif backend_name == "local":
        config["model_name_or_path"] = _nonempty_string(config.get("model_name_or_path"), "model_name_or_path")
        dtype = config.get("dtype", "auto")
        if dtype not in ("auto", "float16", "bfloat16", "float32"):
            raise ValueError("dtype must be auto, float16, bfloat16, or float32")
        config["dtype"] = dtype
        config["temperature"] = _number(config.get("temperature", 0.0), "temperature", 0.0)
        config["top_p"] = _number(config.get("top_p", 1.0), "top_p", 0.0, 1.0)
        if config["top_p"] == 0:
            raise ValueError("top_p must be greater than zero")
        config["max_new_tokens"] = _integer(config.get("max_new_tokens", 1024), "max_new_tokens", 1)
        for name, reserved in (
            ("processor_kwargs", _PROCESSOR_RESERVED),
            ("chat_template_kwargs", _TEMPLATE_RESERVED),
        ):
            config[name] = _mapping(config.get(name, {}), name)
            _reject_keys(config[name], reserved, name)
        for name in ("trust_remote_code", "local_files_only"):
            config[name] = config.get(name, name == "local_files_only")
            if not isinstance(config[name], bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("revision", "attn_implementation"):
            if config.get(name) is not None:
                config[name] = _nonempty_string(config[name], name)
    else:
        raise ValueError("backend_name must be openai or local")
    if config.get("seed") is not None:
        config["seed"] = _integer(config["seed"], "seed", 0)
    return config


def _completion(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BackendError("Model returned an empty or non-text completion")
    return value


def _image_data_url(raw: bytes) -> str:
    from PIL import Image

    try:
        with Image.open(io.BytesIO(raw)) as image:
            mime = Image.MIME.get(image.format)
            image.load()
    except Exception:
        raise BackendError("Unable to decode an image input") from None
    if mime is None or not mime.startswith("image/"):
        raise BackendError("Image input has no recognized image MIME type")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _rejects_max_completion_tokens(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) in (400, 422) and "max_completion_tokens" in str(exc)


def _usage_metadata(usage: Any) -> dict[str, Any]:

    result: dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if isinstance(value, int) and not isinstance(value, bool):
            result[key] = value
    for group, keys in {
        "prompt_tokens_details": ("cached_tokens", "audio_tokens"),
        "completion_tokens_details": (
            "reasoning_tokens", "audio_tokens", "accepted_prediction_tokens",
            "rejected_prediction_tokens",
        ),
    }.items():
        details = getattr(usage, group, None)
        values = {
            key: getattr(details, key)
            for key in keys
            if isinstance(getattr(details, key, None), int)
            and not isinstance(getattr(details, key, None), bool)
        }
        if values:
            result[group] = values
    return result


class OpenAIBackend:


    def __init__(self, config: dict[str, Any], system_prompt: str):
        config = validate_backend_config("openai", config)
        self.system_prompt = _nonempty_string(system_prompt, "system_prompt")
        api_key = os.environ.get(config["api_key_env"])
        if not api_key or not api_key.strip():
            raise ValueError("The environment variable selected by api_key_env is unset or empty")
        self.request: dict[str, Any] = {
            "model": config["model"],
            "stream": False,
            "n": 1,
            config["max_tokens_parameter"]: config["max_tokens"],
        }
        for key in ("temperature", "top_p", "seed"):
            if config.get(key) is not None:
                self.request[key] = config[key]
        if config["effort"] is not None:
            self.request["reasoning_effort"] = config["effort"]
        if config["extra_body"]:
            self.request["extra_body"] = config["extra_body"]
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": config["timeout"],
            "max_retries": config["max_retries"],
        }
        if config.get("base_url") is not None:
            client_kwargs["base_url"] = config["base_url"]
        try:
            from openai import OpenAI
        except ImportError:
            raise BackendError("The openai Python package is required") from None
        try:
            self.client = OpenAI(**client_kwargs)
        except Exception as exc:
            raise BackendError(
                f"OpenAI client initialization failed ({type(exc).__name__})",
                error_type=type(exc).__name__,
            ) from None

    def generate(self, model_input: ModelInput) -> Prediction:
        content = [
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(raw)},
            }
            for raw in model_input.images
        ]
        content.append({"type": "text", "text": IMAGE_PREFIX + model_input.instruction})
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]
        try:
            response = self._create(messages)
        except Exception as exc:

            status = getattr(exc, "status_code", None)
            status = status if isinstance(status, int) and not isinstance(status, bool) else None
            status_text = f", HTTP {status}" if status is not None else ""
            raise BackendError(
                f"API request failed ({type(exc).__name__}{status_text})",
                error_type=type(exc).__name__,
                status_code=status,
            ) from None
        choices = getattr(response, "choices", None)
        if not choices:
            raise BackendError("API response contained no completion choices")
        choice = choices[0]
        completion = _completion(getattr(getattr(choice, "message", None), "content", None))
        metadata: dict[str, Any] = {}
        for key, source in (("model", response), ("id", response), ("finish_reason", choice)):
            value = getattr(source, key, None)
            if isinstance(value, str):
                metadata[key] = value
        usage = _usage_metadata(getattr(response, "usage", None))
        if usage:
            metadata["usage"] = usage
        metadata["max_tokens_parameter"] = "max_tokens" if "max_tokens" in self.request else "max_completion_tokens"
        return Prediction(completion=completion, metadata=metadata)

    def _create(self, messages: list[dict[str, Any]]) -> Any:
        try:
            return self.client.chat.completions.create(messages=messages, **self.request)
        except Exception as exc:
            if "max_completion_tokens" not in self.request or not _rejects_max_completion_tokens(exc):
                raise
            # Endpoints that predate max_completion_tokens: retry with max_tokens and keep it
            # for the rest of the run only once a request succeeds with it.
            request = {key: value for key, value in self.request.items() if key != "max_completion_tokens"}
            request["max_tokens"] = self.request["max_completion_tokens"]
            try:
                response = self.client.chat.completions.create(messages=messages, **request)
            except Exception:
                raise exc from None
            self.request = request
            print("The endpoint rejected max_completion_tokens; using max_tokens for the remaining requests.",
                  file=sys.stderr, flush=True)
            return response

    def close(self) -> None:
        self.client.close()


class LocalBackend:


    def __init__(self, config: dict[str, Any], system_prompt: str):
        config = validate_backend_config("local", config)
        self.system_prompt = _nonempty_string(system_prompt, "system_prompt")
        self.model_name = config["model_name_or_path"]
        dtype = config["dtype"]
        temperature = config["temperature"]
        self.generation_kwargs: dict[str, Any] = {
            "max_new_tokens": config["max_new_tokens"],
            "do_sample": temperature > 0,
            "num_beams": 1,
            "num_return_sequences": 1,
            "return_dict_in_generate": False,
        }
        if temperature > 0:
            self.generation_kwargs.update(temperature=temperature, top_p=config["top_p"])
        self.template_kwargs = config["chat_template_kwargs"]
        common_kwargs = {
            "trust_remote_code": config["trust_remote_code"],
            "local_files_only": config["local_files_only"],
        }
        if config.get("revision") is not None:
            common_kwargs["revision"] = config["revision"]
        seed = config.get("seed")
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed
        except ImportError:
            raise BackendError("The torch, transformers and accelerate Python packages are required") from None
        self.torch = torch
        if seed is not None:
            set_seed(seed)
        model_kwargs: dict[str, Any] = {
            **common_kwargs,
            "device_map": config.get("device_map", "auto"),
            "dtype": "auto" if dtype == "auto" else getattr(torch, dtype),
        }
        if config.get("attn_implementation") is not None:
            model_kwargs["attn_implementation"] = config["attn_implementation"]
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, **common_kwargs, **config["processor_kwargs"]
        )
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()

    def generate(self, model_input: ModelInput) -> Prediction:
        from PIL import Image

        images = []
        try:
            for raw in model_input.images:
                with Image.open(io.BytesIO(raw)) as image:
                    images.append(image.convert("RGB"))
        except Exception:
            for image in images:
                image.close()
            raise BackendError("Unable to decode an image input") from None
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": IMAGE_PREFIX + model_input.instruction})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": content},
        ]
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **self.template_kwargs,
            )
            inputs = inputs.to(self.model.device)
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, **self.generation_kwargs)
            if not getattr(self.model.config, "is_encoder_decoder", False):
                output = output[:, inputs["input_ids"].shape[-1]:]
            text = self.processor.batch_decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if len(text) != 1:
                raise BackendError("Local model returned an unexpected number of completions")
            completion = _completion(text[0])
            return Prediction(
                completion=completion,
                metadata={"model": self.model_name, "generated_tokens": int(output.shape[-1])},
            )
        finally:
            for image in images:
                image.close()

    def close(self) -> None:
        self.model = None
        self.processor = None
