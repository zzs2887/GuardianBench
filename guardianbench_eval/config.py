from __future__ import annotations

import json
import math
from pathlib import Path
from urllib.parse import urlsplit

import yaml

ROOT = Path(__file__).resolve().parent.parent
COMMON = {"data_path", "output_dir", "system_prompt_path", "limit_pairs", "local", "openai"}
LOCAL = {
    "model_name_or_path", "revision", "device_map", "dtype", "local_files_only",
    "trust_remote_code", "attn_implementation", "temperature", "top_p",
    "max_new_tokens", "seed", "processor_kwargs", "chat_template_kwargs",
}
OPENAI = {
    "model", "base_url", "api_key_env", "timeout", "max_retries", "max_tokens",
    "max_tokens_parameter", "temperature", "top_p", "seed", "effort", "extra_body",
}


def reject_credentials(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("Configuration keys must be strings")
            if key.lower() in {"api_key", "apikey", "authorization", "password", "token", "access_token", "secret"}:
                raise ValueError("Store credentials in environment variables, not YAML")
            reject_credentials(item)
    elif isinstance(value, list):
        for item in value:
            reject_credentials(item)


def load_config(path: Path, backend: str, *, data=None, output=None, limit_pairs=None) -> dict:
    path = path.resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    reject_credentials(config)

    json.dumps(config, allow_nan=False)
    unknown = set(config) - COMMON
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {sorted(unknown)}")
    other = "local" if backend == "openai" else "openai"
    if other in config:
        raise ValueError(f"Use a separate configuration for the {other} entry point")
    section = config.get(backend)
    if not isinstance(section, dict):
        raise ValueError(f"Configuration must contain a {backend} mapping")
    unknown = set(section) - (OPENAI if backend == "openai" else LOCAL)
    if unknown:
        raise ValueError(f"Unknown {backend} configuration keys: {sorted(unknown)}")
    for field, override, default in (
        ("data_path", data, "../data/test.parquet"),
        ("output_dir", output, f"../outputs/{backend}"),
        ("system_prompt_path", None, str(ROOT / "prompts/test.txt")),
    ):
        value = override if override is not None else config.get(field, default)
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"{field} must be a nonempty path")
        resolved = Path(value).expanduser()
        if not resolved.is_absolute():
            resolved = (Path.cwd() if override is not None else path.parent) / resolved
        config[field] = str(resolved.resolve())
    if limit_pairs is not None:
        config["limit_pairs"] = limit_pairs
    limit = config.get("limit_pairs")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit_pairs must be a positive integer or null")
    if backend == "openai":
        url = section.get("base_url", "https://api.openai.com/v1")
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must be an HTTP(S) base URL without embedded credentials, query, or fragment")
    for name in ("temperature", "top_p", "timeout"):
        value = section.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError(f"{name} must be a finite number or null")
    return config
