"""Unit tests for the LLM transports + dispatcher (no live servers)."""
from __future__ import annotations

import httpx
import pytest

from app.services import crypto, llm, mistral, ollama
from app.services.llm import LLMConfig


def test_crypto_roundtrip():
    token = crypto.encrypt("sk-secret-123")
    assert token != "sk-secret-123"
    assert crypto.decrypt(token) == "sk-secret-123"
    assert crypto.decrypt("not-a-valid-token") is None


def _mock_httpx(monkeypatch, handler):
    """Patch httpx.AsyncClient so transports hit an in-process handler."""
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        return real(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(httpx, "AsyncClient", factory)


async def test_mistral_generate_parses_content(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer sk-test"
        body = request.read().decode()
        assert "mistral-small-latest" in body
        return httpx.Response(200, json={"choices": [{"message": {"content": "Hello coach"}}]})

    _mock_httpx(monkeypatch, handler)
    out = await mistral.generate(
        "prompt", "system", model="mistral-small-latest", api_key="sk-test"
    )
    assert out == "Hello coach"


async def test_mistral_generate_requires_key():
    with pytest.raises(mistral.MistralError):
        await mistral.generate("p", model="mistral-small-latest", api_key="")


async def test_mistral_list_models(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "mistral-large-latest"}, {"id": "mistral-small-latest"}]})

    _mock_httpx(monkeypatch, handler)
    models = await mistral.list_models("sk-test")
    assert models == ["mistral-large-latest", "mistral-small-latest"]


async def test_llm_dispatch_routes_to_mistral(monkeypatch):
    captured = {}

    async def fake_mgen(prompt, system=None, *, model, api_key, max_tokens):
        captured.update(model=model, api_key=api_key, provider="mistral")
        return "M"

    monkeypatch.setattr(mistral, "generate", fake_mgen)
    out = await llm.generate(
        LLMConfig(provider="mistral", model="mistral-small-latest", api_key="k"), "p", "s"
    )
    assert out == "M"
    assert captured == {"model": "mistral-small-latest", "api_key": "k", "provider": "mistral"}


async def test_llm_dispatch_routes_to_ollama(monkeypatch):
    captured = {}

    async def fake_ogen(prompt, system=None, *, model, num_predict):
        captured.update(model=model, provider="ollama")
        return "O"

    monkeypatch.setattr(ollama, "generate", fake_ogen)
    out = await llm.generate(LLMConfig(provider="ollama", model="qwen3.5:4b"), "p", "s")
    assert out == "O"
    assert captured == {"model": "qwen3.5:4b", "provider": "ollama"}


async def test_llm_error_wraps_provider_errors(monkeypatch):
    async def boom(prompt, system=None, *, model, num_predict):
        raise ollama.OllamaError("down")

    monkeypatch.setattr(ollama, "generate", boom)
    with pytest.raises(llm.LLMError):
        await llm.generate(LLMConfig(provider="ollama", model="qwen3.5:4b"), "p")


# ── OpenAI-compatible transport ───────────────────────────────────────────────
from app.services import openai_compat  # noqa: E402


def test_openai_compat_normalize_base_url():
    assert openai_compat.normalize_base_url(" https://openrouter.ai/api/v1/ ") == "https://openrouter.ai/api/v1"
    assert (
        openai_compat.normalize_base_url("https://openrouter.ai/api/v1/chat/completions")
        == "https://openrouter.ai/api/v1"
    )


def test_openai_compat_validate_rejects_bad_scheme():
    with pytest.raises(openai_compat.OpenAICompatConfigError):
        openai_compat.validate_base_url("ftp://example.com/v1")
    with pytest.raises(openai_compat.OpenAICompatConfigError):
        openai_compat.validate_base_url("")


def test_openai_compat_blocks_private_hosts_when_disabled(monkeypatch):
    monkeypatch.setattr(openai_compat._settings, "openai_compat_allow_private_hosts", False)
    with pytest.raises(openai_compat.OpenAICompatConfigError):
        openai_compat.validate_base_url("http://127.0.0.1:1234/v1")


async def test_openai_compat_generate(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "openrouter.ai"
        assert request.url.path == "/api/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer sk-or-test"
        body = request.read().decode()
        assert "meta-llama/llama-3.3-70b-instruct" in body
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "<think>hmm</think> Keep going!"}}]}
        )

    _mock_httpx(monkeypatch, handler)
    out = await openai_compat.generate(
        "p", "s", base_url="https://openrouter.ai/api/v1",
        model="meta-llama/llama-3.3-70b-instruct", api_key="sk-or-test",
    )
    assert out == "Keep going!"  # reasoning block stripped


async def test_openai_compat_generate_without_key_sends_no_auth(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    _mock_httpx(monkeypatch, handler)
    out = await openai_compat.generate(
        "p", base_url="http://localhost:1234/v1", model="local-model"
    )
    assert out == "ok"


async def test_openai_compat_401_is_config_error(monkeypatch):
    _mock_httpx(monkeypatch, lambda r: httpx.Response(401, json={"error": "bad key"}))
    cfg = LLMConfig(
        provider=llm.OPENAI_COMPAT, model="m", api_key="x", base_url="https://openrouter.ai/api/v1"
    )
    with pytest.raises(llm.LLMConfigError):
        await llm.generate(cfg, "p")


async def test_openai_compat_missing_model_is_config_error():
    cfg = LLMConfig(provider=llm.OPENAI_COMPAT, model="", base_url="https://openrouter.ai/api/v1")
    with pytest.raises(llm.LLMConfigError):
        await llm.generate(cfg, "p")


async def test_openai_compat_list_models_and_status(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/models"
        return httpx.Response(200, json={"data": [{"id": "b/model"}, {"id": "a/model"}]})

    _mock_httpx(monkeypatch, handler)
    cfg = LLMConfig(provider=llm.OPENAI_COMPAT, model="a/model", base_url="https://openrouter.ai/api/v1")
    listed = await llm.list_models(cfg)
    assert listed == {"models": ["a/model", "b/model"], "installed": [], "reachable": True}
    st = await llm.status(cfg)
    assert st["provider"] == llm.OPENAI_COMPAT
    assert st["reachable"] is True and st["model_ready"] is True


async def test_openai_compat_error_never_echoes_response_body(monkeypatch):
    secret = "INTERNAL-ADMIN-PANEL-SECRET"
    _mock_httpx(monkeypatch, lambda r: httpx.Response(500, text=secret))
    with pytest.raises(openai_compat.OpenAICompatError) as ei:
        await openai_compat.generate(
            "p", base_url="http://internal.example/v1", model="m"
        )
    assert secret not in str(ei.value)
    assert "500" in str(ei.value)

    _mock_httpx(monkeypatch, lambda r: httpx.Response(200, json={"error": secret}))
    with pytest.raises(openai_compat.OpenAICompatError) as ei:
        await openai_compat.generate("p", base_url="http://internal.example/v1", model="m")
    assert secret not in str(ei.value)

    _mock_httpx(monkeypatch, lambda r: httpx.Response(200, text=secret))
    with pytest.raises(openai_compat.OpenAICompatError) as ei:
        await openai_compat.list_models("http://internal.example/v1")
    assert secret not in str(ei.value)
