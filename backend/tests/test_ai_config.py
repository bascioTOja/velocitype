"""AI provider config endpoint tests."""
from __future__ import annotations

from app.services import llm, mistral, ollama


async def _register(client, user):
    resp = await client.post("/api/auth/register", json=user)
    assert resp.status_code == 201


async def test_config_defaults(client, unique_user):
    await _register(client, unique_user)
    resp = await client.get("/api/coach/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "ollama"
    assert body["mistral_key_set"] is False
    assert body["ollama_model"] == body["ollama_default"]
    assert body["mistral_model"] == body["mistral_default"]


async def test_set_provider_and_key_never_returns_key(client, unique_user):
    await _register(client, unique_user)
    saved = await client.put(
        "/api/coach/config",
        json={
            "provider": "mistral",
            "mistral_model": "mistral-large-latest",
            "mistral_api_key": "sk-super-secret",
        },
    )
    assert saved.status_code == 200
    body = saved.json()
    assert body["provider"] == "mistral"
    assert body["mistral_model"] == "mistral-large-latest"
    assert body["mistral_key_set"] is True
    # The key itself must never be echoed back anywhere in the payload.
    assert "sk-super-secret" not in saved.text


async def test_key_unchanged_when_null_then_cleared_with_empty(client, unique_user):
    await _register(client, unique_user)
    await client.put("/api/coach/config", json={"provider": "mistral", "mistral_api_key": "sk-a"})
    # null → leave unchanged
    r1 = await client.put("/api/coach/config", json={"mistral_model": "mistral-small-latest"})
    assert r1.json()["mistral_key_set"] is True
    # "" → clear
    r2 = await client.put("/api/coach/config", json={"mistral_api_key": ""})
    assert r2.json()["mistral_key_set"] is False


async def test_analyze_uses_selected_provider(client, unique_user, monkeypatch):
    await _register(client, unique_user)
    await client.put(
        "/api/coach/config",
        json={"provider": "mistral", "mistral_api_key": "sk-live"},
    )
    captured = {}

    async def fake_mgen(prompt, system=None, *, model, api_key, max_tokens):
        captured.update(model=model, api_key=api_key)
        return "Mistral says: nice work."

    monkeypatch.setattr(mistral, "generate", fake_mgen)
    resp = await client.post("/api/coach/analyze")
    assert resp.status_code == 200
    body = resp.json()
    assert body["analysis"] == "Mistral says: nice work."
    assert body["model"] == "mistral-small-latest"  # default model applied
    assert captured["api_key"] == "sk-live"  # decrypted key threaded through


async def test_analyze_missing_mistral_key_returns_422(client, unique_user):
    await _register(client, unique_user)
    await client.put("/api/coach/config", json={"provider": "mistral"})  # no key
    resp = await client.post("/api/coach/analyze")
    assert resp.status_code == 422
    assert resp.headers["content-type"].startswith("application/problem+json")


async def test_models_list_ollama(client, unique_user, monkeypatch):
    await _register(client, unique_user)

    async def fake_list():
        return ["qwen3.5:4b", "llama3.2:3b"]

    monkeypatch.setattr(ollama, "list_models", fake_list)
    resp = await client.get("/api/coach/models?provider=ollama")
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "ollama"
    assert body["reachable"] is True
    assert "qwen3.5:4b" in body["installed"]


async def test_model_pull_kicks_off_and_status(client, unique_user, monkeypatch):
    await _register(client, unique_user)

    async def fake_pull(name):
        ollama._PULLS[name] = {
            "name": name, "status": "success", "completed": True, "percent": 100.0, "error": None,
        }

    monkeypatch.setattr(ollama, "pull_model", fake_pull)
    started = await client.post("/api/coach/models/pull", json={"name": "llama3.2:3b"})
    assert started.status_code == 202
    assert started.json()["name"] == "llama3.2:3b"

    status = await client.get("/api/coach/models/pull?name=llama3.2:3b")
    assert status.status_code == 200


async def test_ai_config_requires_auth(client):
    assert (await client.get("/api/coach/config")).status_code == 401
    assert (await client.get("/api/coach/models")).status_code == 401
    assert (await client.post("/api/coach/models/pull", json={"name": "x"})).status_code == 401


async def test_openai_compat_config_roundtrip(client, unique_user):
    await _register(client, unique_user)
    body = (await client.get("/api/coach/config")).json()
    assert body["openai_key_set"] is False
    assert body["openai_base_url"] == body["openai_default_base_url"]

    saved = await client.put(
        "/api/coach/config",
        json={
            "provider": "openai_compat",
            "openai_base_url": "https://openrouter.ai/api/v1/chat/completions/",
            "openai_model": "meta-llama/llama-3.3-70b-instruct",
            "openai_api_key": "sk-or-secret",
        },
    )
    assert saved.status_code == 200
    out = saved.json()
    assert out["provider"] == "openai_compat"
    assert out["openai_base_url"] == "https://openrouter.ai/api/v1"
    assert out["openai_model"] == "meta-llama/llama-3.3-70b-instruct"
    assert out["openai_key_set"] is True
    assert "sk-or-secret" not in saved.text

    cleared = await client.put("/api/coach/config", json={"openai_api_key": ""})
    assert cleared.json()["openai_key_set"] is False


async def test_openai_compat_rejects_invalid_base_url(client, unique_user):
    await _register(client, unique_user)
    resp = await client.put("/api/coach/config", json={"openai_base_url": "file:///etc/passwd"})
    assert resp.status_code == 422


async def test_analyze_uses_openai_compat(client, unique_user, monkeypatch):
    from app.services import openai_compat

    await _register(client, unique_user)
    await client.put(
        "/api/coach/config",
        json={
            "provider": "openai_compat",
            "openai_base_url": "https://openrouter.ai/api/v1",
            "openai_model": "openai/gpt-4o-mini",
            "openai_api_key": "sk-or-live",
        },
    )
    captured = {}

    async def fake_gen(prompt, system=None, *, base_url, model, api_key, max_tokens):
        captured.update(base_url=base_url, model=model, api_key=api_key)
        return "Router says: nice work."

    monkeypatch.setattr(openai_compat, "generate", fake_gen)
    resp = await client.post("/api/coach/analyze")
    assert resp.status_code == 200
    assert resp.json()["analysis"] == "Router says: nice work."
    assert resp.json()["model"] == "openai/gpt-4o-mini"
    assert captured == {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o-mini",
        "api_key": "sk-or-live",
    }


async def test_analyze_openai_compat_without_model_returns_422(client, unique_user):
    await _register(client, unique_user)
    await client.put("/api/coach/config", json={"provider": "openai_compat"})
    resp = await client.post("/api/coach/analyze")
    assert resp.status_code == 422
