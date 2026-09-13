import base64
import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from guardianbench_eval.backends import (
    BackendError,
    LocalBackend,
    OpenAIBackend,
    validate_backend_config,
)
from guardianbench_eval.data import ModelInput


def image_bytes(format="PNG", mode="RGB"):
    output = io.BytesIO()
    with Image.new(mode, (3, 2), color=42) as image:
        image.save(output, format=format)
    return output.getvalue()


@pytest.fixture
def api(monkeypatch):
    response = SimpleNamespace(
        model="served-model",
        id="response-123",
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="Judgment: Safe\nNo hazard.", refusal=None),
            finish_reason="stop",
        )],
        usage=SimpleNamespace(
            prompt_tokens=9,
            completion_tokens=5,
            total_tokens=14,
            prompt_tokens_details=SimpleNamespace(cached_tokens=2),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=1),
            provider_extra="do-not-save-provider-payload",
        ),
        secret="do-not-save-raw-response",
    )
    client = Mock()
    client.chat.completions.create.return_value = response
    factory = Mock(return_value=client)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=factory))
    monkeypatch.setenv("TEST_EVAL_API_KEY", "test-secret-from-environment")
    config = {
        "model": "requested-model",
        "base_url": "http://localhost:8000/v1",
        "api_key_env": "TEST_EVAL_API_KEY",
        "timeout": 17,
        "max_retries": 1,
        "max_tokens": 91,
    }
    return SimpleNamespace(client=client, factory=factory, response=response, config=config)


def test_api_exact_messages_mime_and_no_annotation_or_credential_leak(api):
    png, jpeg = image_bytes(), image_bytes("JPEG")

    model_input = SimpleNamespace(
        instruction="Check this action.", images=(png, jpeg),
        label="hidden-gold", reference="hidden-reference",
        messages=[{"role": "assistant", "content": "hidden-answer"}],
    )
    backend = OpenAIBackend(api.config, "Judge visible hazards.")
    result = backend.generate(model_input)
    request = api.client.chat.completions.create.call_args.kwargs
    assert request["messages"] == [
        {"role": "system", "content": "Judge visible hazards."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(png).decode(),
            }},
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
            }},
            {"type": "text", "text": "<image>Check this action."},
        ]},
    ]
    assert request["model"] == "requested-model"
    assert request["max_completion_tokens"] == 91
    assert "max_tokens" not in request
    assert request["stream"] is False
    assert request["n"] == 1
    assert "effort" not in request
    assert "reasoning_effort" not in request
    assert api.factory.call_args.kwargs == {
        "api_key": "test-secret-from-environment", "base_url": "http://localhost:8000/v1",
        "timeout": 17.0, "max_retries": 1,
    }
    serialized = json.dumps({"request": request, "metadata": result.metadata})
    for forbidden in ("hidden-gold", "hidden-reference", "hidden-answer", "test-secret-from-environment",
                      "do-not-save-provider-payload", "do-not-save-raw-response"):
        assert forbidden not in serialized
    assert result.metadata == {
        "model": "served-model", "id": "response-123", "finish_reason": "stop",
        "usage": {
            "prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
        "max_tokens_parameter": "max_completion_tokens",
    }
    assert result.completion == "Judgment: Safe\nNo hazard."
    backend.close()
    api.client.close.assert_called_once_with()


def test_api_nulls_omit_optional_parameters_and_forward_provider_options(api):
    backend = OpenAIBackend({
        **api.config, "max_tokens_parameter": "max_completion_tokens",
        "temperature": None, "top_p": None, "seed": None, "effort": None,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }, "System")
    backend.generate(ModelInput("Action", (image_bytes(),)))
    request = api.client.chat.completions.create.call_args.kwargs
    assert request["max_completion_tokens"] == 91
    for name in ("max_tokens", "temperature", "top_p", "seed", "effort", "reasoning_effort"):
        assert name not in request
    assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_api_explicit_sampling_parameters_are_forwarded(api):
    backend = OpenAIBackend({**api.config, "temperature": 0, "top_p": 0.8, "seed": 42}, "System")
    backend.generate(ModelInput("Action", (image_bytes(),)))
    request = api.client.chat.completions.create.call_args.kwargs
    assert {key: request[key] for key in ("temperature", "top_p", "seed")} == {
        "temperature": 0.0, "top_p": 0.8, "seed": 42,
    }


@pytest.mark.parametrize("effort,expected", [
    ("high", "high"), ("  medium \n", "medium"), ("provider-specific-level", "provider-specific-level"),
])
def test_api_effort_is_forwarded_as_reasoning_effort(api, effort, expected):
    backend = OpenAIBackend({**api.config, "effort": effort}, "System")
    backend.generate(ModelInput("Action", (image_bytes(),)))
    request = api.client.chat.completions.create.call_args.kwargs
    assert request["reasoning_effort"] == expected
    assert "effort" not in request


@pytest.mark.parametrize("reserved", [
    "model", "messages", "stream", "n", "temperature", "top_p", "seed",
    "max_tokens", "max_completion_tokens", "api_key", "effort", "reasoning_effort",
])
def test_api_extra_body_cannot_override_request_contract(api, reserved):
    with pytest.raises(ValueError, match="reserved"):
        OpenAIBackend({**api.config, "extra_body": {reserved: "not-allowed"}}, "System")
    api.factory.assert_not_called()


@pytest.mark.parametrize("content", [None, "", "  \n ", [{"type": "text", "text": "Safe"}]])
def test_api_empty_or_nontext_completions_are_errors(api, content):
    api.response.choices[0].message.content = content
    backend = OpenAIBackend(api.config, "System")
    with pytest.raises(BackendError, match="empty or non-text"):
        backend.generate(ModelInput("Action", (image_bytes(),)))


def test_api_rejects_missing_choices_and_invalid_image_before_network(api):
    backend = OpenAIBackend(api.config, "System")
    with pytest.raises(BackendError, match="decode"):
        backend.generate(ModelInput("Action", (b"broken-image",)))
    api.client.chat.completions.create.assert_not_called()
    api.response.choices = []
    with pytest.raises(BackendError, match="no completion choices"):
        backend.generate(ModelInput("Action", (image_bytes(),)))


def test_api_failure_preserves_status_without_server_body_or_outer_retry(api):
    class RateLimitError(Exception):
        status_code = 429

    api.client.chat.completions.create.side_effect = RateLimitError(
        "Authorization: test-secret-from-environment; private-request-and-server-body"
    )
    backend = OpenAIBackend(api.config, "System")
    with pytest.raises(BackendError) as caught:
        backend.generate(ModelInput("Action", (image_bytes(),)))
    assert str(caught.value) == "API request failed (RateLimitError, HTTP 429)"
    assert caught.value.error_type == "RateLimitError"
    assert caught.value.status_code == 429
    assert caught.value.__suppress_context__
    api.client.chat.completions.create.assert_called_once()


class BadRequestError(Exception):
    status_code = 400


def token_parameters(api):
    return [
        [key for key in ("max_completion_tokens", "max_tokens") if key in call.kwargs]
        for call in api.client.chat.completions.create.call_args_list
    ]


def test_api_falls_back_to_max_tokens_and_keeps_it_after_success(api, capsys):
    api.client.chat.completions.create.side_effect = [
        BadRequestError("Unrecognized request argument supplied: max_completion_tokens"),
        api.response,
        api.response,
    ]
    backend = OpenAIBackend(api.config, "System")
    first = backend.generate(ModelInput("Action", (image_bytes(),)))
    second = backend.generate(ModelInput("Action", (image_bytes(),)))
    assert token_parameters(api) == [["max_completion_tokens"], ["max_tokens"], ["max_tokens"]]
    assert api.client.chat.completions.create.call_args.kwargs["max_tokens"] == 91
    assert first.metadata["max_tokens_parameter"] == second.metadata["max_tokens_parameter"] == "max_tokens"
    assert capsys.readouterr().err.count("using max_tokens") == 1


def test_api_failed_fallback_reports_original_error_and_keeps_max_completion_tokens(api):
    class RetryError(Exception):
        status_code = 503

    api.client.chat.completions.create.side_effect = [
        BadRequestError("'max_completion_tokens' is not supported"), RetryError("unavailable"), api.response,
    ]
    backend = OpenAIBackend(api.config, "System")
    with pytest.raises(BackendError) as caught:
        backend.generate(ModelInput("Action", (image_bytes(),)))
    assert str(caught.value) == "API request failed (BadRequestError, HTTP 400)"
    assert caught.value.__suppress_context__
    result = backend.generate(ModelInput("Action", (image_bytes(),)))
    assert token_parameters(api) == [["max_completion_tokens"], ["max_tokens"], ["max_completion_tokens"]]
    assert result.metadata["max_tokens_parameter"] == "max_completion_tokens"


@pytest.mark.parametrize("error", [
    BadRequestError("Invalid image input"),
    type("RateLimitError", (Exception,), {"status_code": 429})("max_completion_tokens rate limited"),
])
def test_api_other_errors_do_not_trigger_the_fallback(api, error):
    api.client.chat.completions.create.side_effect = error
    backend = OpenAIBackend(api.config, "System")
    with pytest.raises(BackendError):
        backend.generate(ModelInput("Action", (image_bytes(),)))
    api.client.chat.completions.create.assert_called_once()


def test_api_explicit_max_tokens_is_sent_without_fallback(api):
    api.client.chat.completions.create.side_effect = BadRequestError("Use 'max_completion_tokens' instead.")
    backend = OpenAIBackend({**api.config, "max_tokens_parameter": "max_tokens"}, "System")
    with pytest.raises(BackendError, match="HTTP 400"):
        backend.generate(ModelInput("Action", (image_bytes(),)))
    assert token_parameters(api) == [["max_tokens"]]


def test_api_key_is_required_even_for_local_server(api, monkeypatch):
    monkeypatch.delenv("TEST_EVAL_API_KEY")
    with pytest.raises(ValueError, match="unset or empty"):
        OpenAIBackend(api.config, "System")
    api.factory.assert_not_called()


class FakeTensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, key):
        row_slice, column_slice = key
        return FakeTensor([row[column_slice] for row in self.rows[row_slice]])


class FakeInputs(dict):
    def to(self, device):
        self.device = device
        return self


@pytest.fixture
def local(monkeypatch):
    state = SimpleNamespace(inference=False, snapshots=[])

    class InferenceMode:
        def __enter__(self):
            state.inference = True

        def __exit__(self, *_):
            state.inference = False

    torch = SimpleNamespace(
        inference_mode=InferenceMode, float16=object(), bfloat16=object(), float32=object(),
    )
    processor = Mock()
    inputs = FakeInputs(input_ids=FakeTensor([[101, 102, 103]]), pixel_values="image-tensor")

    def apply_chat_template(messages, **kwargs):

        snapshot = []
        for message in messages:
            content = []
            for item in message["content"]:
                if item["type"] == "image":
                    image = item["image"]
                    content.append({"type": "image", "mode": image.mode, "size": image.size})
                else:
                    content.append(dict(item))
            snapshot.append({"role": message["role"], "content": content})
        state.snapshots.append(snapshot)
        return inputs

    processor.apply_chat_template.side_effect = apply_chat_template
    processor.batch_decode.return_value = ["Judgment: Unsafe"]
    model = Mock(device="cuda:0", config=SimpleNamespace(is_encoder_decoder=False))

    def generate(**kwargs):
        assert state.inference, "Generation must run inside torch.inference_mode()"
        return FakeTensor([[101, 102, 103, 401, 402]])

    model.generate.side_effect = generate
    model_factory = Mock(return_value=model)
    processor_factory = Mock(return_value=processor)
    set_seed = Mock()
    transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=processor_factory),
        AutoModelForImageTextToText=SimpleNamespace(from_pretrained=model_factory),
        set_seed=set_seed,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return SimpleNamespace(
        torch=torch, model=model, processor=processor, state=state, inputs=inputs,
        model_factory=model_factory, processor_factory=processor_factory, set_seed=set_seed,
    )


def test_local_native_template_rgb_greedy_trim_and_parameter_forwarding(local):
    backend = LocalBackend({
        "model_name_or_path": "example/native-vlm",
        "revision": "pinned-revision", "device_map": "auto", "dtype": "bfloat16",
        "local_files_only": True, "trust_remote_code": False, "attn_implementation": "sdpa",
        "temperature": 0, "top_p": 0.5, "max_new_tokens": 37, "seed": 19,
        "processor_kwargs": {"max_pixels": 313600},
        "chat_template_kwargs": {"enable_thinking": False},
    }, "System contract")
    model_input = SimpleNamespace(
        instruction="Action to assess", images=(image_bytes(mode="L"),),
        label="hidden-label", reference="hidden-reference", messages="hidden-messages",
    )
    result = backend.generate(model_input)
    assert local.state.snapshots == [[
        {"role": "system", "content": [{"type": "text", "text": "System contract"}]},
        {"role": "user", "content": [
            {"type": "image", "mode": "RGB", "size": (3, 2)},
            {"type": "text", "text": "<image>Action to assess"},
        ]},
    ]]
    assert local.processor.apply_chat_template.call_args.kwargs == {
        "add_generation_prompt": True, "tokenize": True, "return_dict": True,
        "return_tensors": "pt", "enable_thinking": False,
    }
    assert local.model_factory.call_args.args == ("example/native-vlm",)
    assert local.model_factory.call_args.kwargs == {
        "revision": "pinned-revision", "device_map": "auto", "dtype": local.torch.bfloat16,
        "local_files_only": True, "trust_remote_code": False, "attn_implementation": "sdpa",
    }
    assert local.processor_factory.call_args.kwargs == {
        "revision": "pinned-revision", "local_files_only": True,
        "trust_remote_code": False, "max_pixels": 313600,
    }
    generation = local.model.generate.call_args.kwargs
    assert generation["max_new_tokens"] == 37
    assert generation["do_sample"] is False
    assert generation["num_beams"] == 1
    assert generation["num_return_sequences"] == 1
    assert generation["return_dict_in_generate"] is False
    assert "temperature" not in generation
    assert "top_p" not in generation
    assert generation["pixel_values"] == "image-tensor"
    assert local.inputs.device == "cuda:0"
    assert local.processor.batch_decode.call_args.args[0].rows == [[401, 402]]
    assert local.processor.batch_decode.call_args.kwargs == {
        "skip_special_tokens": True, "clean_up_tokenization_spaces": False,
    }
    assert result.completion == "Judgment: Unsafe"
    assert result.metadata["generated_tokens"] == 2
    local.model.eval.assert_called_once_with()
    backend.generate(model_input)
    local.set_seed.assert_called_once_with(19)
    backend.close()
    assert backend.model is None
    assert backend.processor is None


def test_local_encoder_decoder_output_not_trimmed_and_sampling_enabled(local):
    local.model.config.is_encoder_decoder = True
    local.model.generate.side_effect = lambda **kwargs: FakeTensor([[401, 402]])
    backend = LocalBackend({
        "model_name_or_path": "example/encoder-decoder", "temperature": 0.7, "top_p": 0.9,
    }, "System")
    backend.generate(ModelInput("Action", (image_bytes(),)))
    assert local.processor.batch_decode.call_args.args[0].rows == [[401, 402]]
    generation = local.model.generate.call_args.kwargs
    assert {key: generation[key] for key in ("do_sample", "temperature", "top_p")} == {
        "do_sample": True, "temperature": 0.7, "top_p": 0.9,
    }
    local.set_seed.assert_not_called()


def test_local_rejects_empty_output(local):
    local.processor.batch_decode.return_value = ["  "]
    backend = LocalBackend({"model_name_or_path": "example/model"}, "System")
    with pytest.raises(BackendError, match="empty or non-text"):
        backend.generate(ModelInput("Action", (image_bytes(),)))


def test_validation_needs_neither_optional_libraries_nor_api_key(monkeypatch):
    for name in ("torch", "transformers", "openai"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    api = validate_backend_config("openai", {"model": "example/model"})
    local = validate_backend_config("local", {"model_name_or_path": "example/model"})
    assert api["max_tokens"] == local["max_new_tokens"] == 1024
    assert api["max_tokens_parameter"] == "max_completion_tokens"
    assert api["effort"] is None
    assert local["trust_remote_code"] is False
    assert local["temperature"] == 0


@pytest.mark.parametrize("backend,field,value", [
    ("openai", "max_retries", -1), ("openai", "max_tokens", True),
    ("openai", "temperature", float("nan")), ("openai", "timeout", 0),
    ("openai", "top_p", 1.1), ("openai", "seed", 1.2),
    ("openai", "extra_body", []),
    ("openai", "effort", ""), ("openai", "effort", " \n "),
    ("openai", "effort", 3), ("openai", "effort", True),
    ("openai", "effort", []), ("openai", "effort", {}),
    ("openai", "max_tokens_parameter", "max_output_tokens"),
    ("local", "temperature", -0.1), ("local", "top_p", 0),
    ("local", "dtype", "int8"), ("local", "max_new_tokens", 0),
    ("local", "trust_remote_code", "false"),
    ("local", "processor_kwargs", {"trust_remote_code": True}),
    ("local", "chat_template_kwargs", {"tokenize": False}),
    ("local", "chat_template_kwargs", {"images": "hidden-image"}),
])
def test_invalid_backend_config_rejected_before_inference(backend, field, value):
    name = "model" if backend == "openai" else "model_name_or_path"
    with pytest.raises(ValueError):
        validate_backend_config(backend, {name: "example/model", field: value})
