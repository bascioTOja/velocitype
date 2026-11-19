"""Provider-agnostic LLM facade.

Resolves a per-user :class:`LLMConfig` and dispatches generation / status / model
listing to the local Ollama transport, the Mistral (EU cloud) transport, or a
generic OpenAI-compatible endpoint (OpenRouter, LM Studio, vLLM …). Coach code
depends only on this module, never on a specific provider.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.services import mistral, ollama, openai_compat

OLLAMA = "ollama"
MISTRAL = "mistral"
OPENAI_COMPAT = "openai_compat"
PROVIDERS = (OLLAMA, MISTRAL, OPENAI_COMPAT)


class LLMError(RuntimeError):
    """Provider-neutral failure (unreachable, missing key, bad output)."""


class LLMConfigError(LLMError):
    """The user's provider config is incomplete or invalid (missing key, base
    URL or model) — fixable in Settings, not a transient outage."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str | None = None  # only meaningful for cloud providers
    base_url: str | None = None  # only meaningful for OPENAI_COMPAT


async def generate(
    cfg: LLMConfig, prompt: str, system: str | None = None, *, max_tokens: int = 300
) -> str:
    try:
        if cfg.provider == MISTRAL:
            if not cfg.api_key:
                raise LLMConfigError("No Mistral API key set.")
            return await mistral.generate(
                prompt, system, model=cfg.model, api_key=cfg.api_key, max_tokens=max_tokens
            )
        if cfg.provider == OPENAI_COMPAT:
            return await openai_compat.generate(
                prompt,
                system,
                base_url=cfg.base_url or "",
                model=cfg.model,
                api_key=cfg.api_key,
                max_tokens=max_tokens,
            )
        return await ollama.generate(prompt, system, model=cfg.model, num_predict=max_tokens)
    except openai_compat.OpenAICompatConfigError as exc:
        raise LLMConfigError(str(exc)) from exc
    except (ollama.OllamaError, mistral.MistralError, openai_compat.OpenAICompatError) as exc:
        raise LLMError(str(exc)) from exc


async def status(cfg: LLMConfig) -> dict:
    if cfg.provider == MISTRAL:
        st = await mistral.status(cfg.api_key or "")
        return {"provider": MISTRAL, "model": cfg.model, **st}
    if cfg.provider == OPENAI_COMPAT:
        st = await openai_compat.status(cfg.base_url or "", cfg.model, cfg.api_key)
        return {"provider": OPENAI_COMPAT, "model": cfg.model, **st}
    st = await ollama.status(cfg.model)
    return {"provider": OLLAMA, **st}


async def list_models(cfg: LLMConfig) -> dict:
    """Return {models, installed, reachable} for the config's provider."""
    if cfg.provider == MISTRAL:
        try:
            models = await mistral.list_models(cfg.api_key or "")
            return {"models": models, "installed": [], "reachable": True}
        except mistral.MistralError:
            return {"models": [], "installed": [], "reachable": False}
    if cfg.provider == OPENAI_COMPAT:
        try:
            models = await openai_compat.list_models(cfg.base_url or "", cfg.api_key)
            return {"models": models, "installed": [], "reachable": True}
        except openai_compat.OpenAICompatError:
            return {"models": [], "installed": [], "reachable": False}
    try:
        installed = await ollama.list_models()
        return {"models": installed, "installed": installed, "reachable": True}
    except ollama.OllamaError:
        return {"models": [], "installed": [], "reachable": False}
