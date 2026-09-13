import base64
import io
import json

import httpx
import openai
from PIL import Image

from guardianbench_eval.backends import OpenAIBackend, validate_backend_config
from guardianbench_eval.data import ModelInput, Sample
from guardianbench_eval.runner import evaluate


def test_endpoint_default_is_explicit_even_with_sdk_environment_override(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://unexpected.example.invalid/v1")
    config = validate_backend_config("openai", {"model": "mock"})
    assert config["base_url"] == "https://api.openai.com/v1"


def test_sdk_wire_format_and_scoring(monkeypatch, tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "blue").save(buffer, format="PNG")
    pixels = buffer.getvalue()
    seen = []

    def serve(request):
        assert request.url == "https://vision.example.invalid/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-only-dummy"
        body = json.loads(request.content)
        assert body["model"] == "mock-vision"
        assert body["temperature"] == 0
        assert body["max_completion_tokens"] == 1024
        assert "max_tokens" not in body and "top_p" not in body
        assert body["reasoning_effort"] == "high"
        assert "effort" not in body
        assert body["messages"][0] == {"role": "system", "content": "Fixed public prompt"}
        user = body["messages"][1]
        assert user["role"] == "user"
        image = user["content"][0]["image_url"]
        assert image["url"] == "data:image/png;base64," + base64.b64encode(pixels).decode()
        assert set(image) == {"url"}
        assert user["content"][1] == {"type": "text", "text": "<image>Synthetic instruction"}
        assert len(body["messages"]) == 2
        seen.append(body)
        verdict = "Safe" if len(seen) == 1 else "Unsafe"
        return httpx.Response(200, json={
            "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
            "model": "mock-vision-version",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": f"[Safety: {verdict}]",
            }}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        })

    original = openai.OpenAI
    client = httpx.Client(transport=httpx.MockTransport(serve))
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: original(http_client=client, **kwargs))
    monkeypatch.setenv("TEST_VISION_KEY", "test-only-dummy")
    backend = OpenAIBackend({
        "model": "mock-vision", "base_url": "https://vision.example.invalid/v1",
        "api_key_env": "TEST_VISION_KEY", "max_tokens_parameter": "max_completion_tokens",
        "max_tokens": 1024, "temperature": 0, "top_p": None, "max_retries": 0,
        "effort": "high",
    }, "Fixed public prompt")
    model_input = ModelInput("Synthetic instruction", (pixels,))
    samples = [Sample("1", "pair", "Safe", model_input), Sample("2", "pair", "Unsafe", model_input)]
    try:
        metrics = evaluate(samples, backend, tmp_path)
    finally:
        backend.close()
    assert len(seen) == 2
    assert metrics["accuracy"] == metrics["pair_accuracy"] == 1
    record = json.loads((tmp_path / "predictions.jsonl").read_text().splitlines()[0])
    assert record["metadata"]["usage"]["total_tokens"] == 16
    assert record["metadata"]["model"] == "mock-vision-version"
    assert "test-only-dummy" not in (tmp_path / "predictions.jsonl").read_text()


def test_sdk_auth_failure_stays_in_denominator_without_response_body(monkeypatch, tmp_path):
    raw = io.BytesIO()
    Image.new("RGB", (2, 2)).save(raw, "PNG")
    original = openai.OpenAI
    requests = []

    def reject(request):
        requests.append(request)
        body = json.loads(request.content)
        assert "reasoning_effort" not in body and "effort" not in body
        assert set(body["messages"][1]["content"][0]["image_url"]) == {"url"}
        return httpx.Response(401, json={"error": {"message": "PRIVATE-ERROR-PAYLOAD", "type": "auth_error"}})

    client = httpx.Client(transport=httpx.MockTransport(reject))
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: original(http_client=client, **kwargs))
    monkeypatch.setenv("TEST_VISION_KEY", "test-only-dummy")
    backend = OpenAIBackend({
        "model": "mock", "base_url": "https://vision.example.invalid/v1",
        "api_key_env": "TEST_VISION_KEY", "max_retries": 0, "effort": None,
    }, "Fixed prompt")
    sample_input = ModelInput("Synthetic instruction", (raw.getvalue(),))
    try:
        summary = evaluate([Sample("1", "p", "Safe", sample_input), Sample("2", "p", "Unsafe", sample_input)], backend, tmp_path)
    finally:
        backend.close()
    assert len(requests) == 2
    assert summary["status"] == "incomplete"
    assert summary["counts"]["rows"] == 2 and summary["counts"]["api_or_runtime_errors"] == 2
    ledger = (tmp_path / "predictions.jsonl").read_text()
    assert "PRIVATE-ERROR-PAYLOAD" not in ledger and "test-only-dummy" not in ledger
    for line in ledger.splitlines():
        assert json.loads(line)["error"]["http_status"] == 401


def test_sdk_endpoint_rejecting_max_completion_tokens_falls_back_to_max_tokens(monkeypatch, tmp_path):
    raw = io.BytesIO()
    Image.new("RGB", (2, 2)).save(raw, "PNG")
    original = openai.OpenAI
    seen = []

    def serve(request):
        body = json.loads(request.content)
        seen.append([key for key in ("max_completion_tokens", "max_tokens") if key in body])
        if "max_completion_tokens" in body:
            return httpx.Response(400, json={"error": {
                "message": "Unrecognized request argument supplied: max_completion_tokens",
                "type": "invalid_request_error", "param": None, "code": None,
            }})
        assert body["max_tokens"] == 1024
        return httpx.Response(200, json={
            "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "legacy-vision",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "[Safety: Unsafe]",
            }}],
        })

    client = httpx.Client(transport=httpx.MockTransport(serve))
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: original(http_client=client, **kwargs))
    monkeypatch.setenv("TEST_VISION_KEY", "test-only-dummy")
    backend = OpenAIBackend({
        "model": "legacy-vision", "base_url": "https://legacy.example.invalid/v1",
        "api_key_env": "TEST_VISION_KEY", "max_retries": 0,
    }, "Fixed prompt")
    sample_input = ModelInput("Synthetic instruction", (raw.getvalue(),))
    try:
        summary = evaluate([Sample("1", "p", "Safe", sample_input), Sample("2", "p", "Unsafe", sample_input)], backend, tmp_path)
    finally:
        backend.close()
    assert seen == [["max_completion_tokens"], ["max_tokens"], ["max_tokens"]]
    assert summary["status"] == "complete"
    records = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert [record["metadata"]["max_tokens_parameter"] for record in records] == ["max_tokens", "max_tokens"]
