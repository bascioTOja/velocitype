"""Thin async client for any OpenAI-compatible chat-completions API.

Covers OpenRouter, Together, Groq, DeepSeek, OpenAI itself, and self-hosted
servers that speak the same protocol (LM Studio, vLLM, llama.cpp server, LocalAI,
LiteLLM proxy …). The user supplies the base URL in the standard OpenAI-SDK form
(``https://openrouter.ai/api/v1``) — ``/chat/completions`` and ``/models`` are
appended to it. The API key is optional (many local servers need none), per-user,
stored encrypted, and passed here at call time. All requests originate from the
API server, never the browser.

Error messages raised here reach the end user (via the coach 422/503 detail), so
they never include the endpoint's response body — only the HTTP status and a
generic hint. Otherwise a user could point the base URL at an internal service
and read its responses back (SSRF read). Bodies go to the server log instead.
"""
from __future__ import annotations

import ipaddress
import logging
import re
import socket
from urllib.parse import urlparse

import httpx

from app.config import get_settings

_settings = get_settings()
_log = logging.getLogger(__name__)

# User-facing hints per status (OpenAI / OpenRouter semantics). Deliberately
# generic: the remote body is logged, never echoed.
_STATUS_HINTS: dict[int, str] = {
    400: "the request was rejected - check the model id",
    402: "insufficient credits on the provider account",
    404: "not found - check the base URL and model id",
    408: "the endpoint timed out",
    413: "the prompt is too large for this model",
    422: "the request was rejected - check the model id",
    429: "rate limited - try again shortly",
}

# Reasoning models (DeepSeek-R1, qwen3 …) served via some gateways leak a
# <think>…</think> block into the content; strip it like the Ollama client does.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OpenAICompatError(RuntimeError):
    """Raised when the endpoint is unreachable, misconfigured or returns an error."""


class OpenAICompatConfigError(OpenAICompatError):
    """A user-config problem (no base URL / model / disallowed host), not an outage."""


def normalize_base_url(url: str) -> str:
    """Trim whitespace and trailing slashes; tolerate users pasting the full
    ``…/chat/completions`` endpoint instead of the base."""
    url = (url or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/models"):
        if url.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
    return url


def _is_private_host(host: str) -> bool:
    """True if ``host`` is (or resolves to) a loopback/private/link-local address."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False  # unresolvable → the request itself will fail
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return True
    return False


def validate_base_url(url: str) -> str:
    """Normalise and validate a user-supplied base URL. Raises
    OpenAICompatConfigError if it is unusable or (when the deployment forbids it)
    points at a private network."""
    url = normalize_base_url(url)
    if not url:
        raise OpenAICompatConfigError("No base URL set for the OpenAI-compatible provider.")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise OpenAICompatConfigError("Base URL must be an http(s) URL, e.g. https://openrouter.ai/api/v1")
    if not _settings.openai_compat_allow_private_hosts and _is_private_host(parsed.hostname):
        raise OpenAICompatConfigError(
            "Base URL points to a private network address, which this deployment does not allow."
        )
    return url


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # OpenRouter attribution headers; harmless for every other provider.
    if _settings.openai_compat_app_title:
        headers["X-Title"] = _settings.openai_compat_app_title
    return headers


def _unreachable(exc: httpx.HTTPError) -> OpenAICompatError:
    _log.warning("OpenAI-compatible endpoint unreachable: %r", exc)
    if isinstance(exc, httpx.TimeoutException):
        return OpenAICompatError("Endpoint timed out.")
    return OpenAICompatError("Endpoint unreachable.")


def _http_error(resp: httpx.Response) -> OpenAICompatError:
    """Map a non-200 response to a user-safe error; log the body server-side."""
    _log.warning(
        "OpenAI-compatible endpoint %s returned %s: %s",
        resp.request.url, resp.status_code, resp.text[:500],
    )
    if resp.status_code in (401, 403):
        return OpenAICompatConfigError(
            f"Endpoint rejected the API key ({resp.status_code}). Check your API key."
        )
    hint = _STATUS_HINTS.get(resp.status_code) or (
        "provider error" if resp.status_code >= 500 else "unexpected status"
    )
    return OpenAICompatError(f"Endpoint returned {resp.status_code}: {hint}.")


def _clean(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


async def generate(
    prompt: str,
    system: str | None = None,
    *,
    base_url: str,
    model: str,
    api_key: str | None = None,
    max_tokens: int = 400,
) -> str:
    """Run a single chat completion. Raises OpenAICompatError on failure."""
    url = validate_base_url(base_url)
    if not model:
        raise OpenAICompatConfigError("No model set for the OpenAI-compatible provider.")
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=_settings.openai_compat_timeout_s) as client:
            resp = await client.post(
                f"{url}/chat/completions", json=payload, headers=_headers(api_key)
            )
    except httpx.HTTPError as exc:
        raise _unreachable(exc) from exc

    if resp.status_code != 200:
        raise _http_error(resp)

    try:
        data = resp.json()
        if data.get("error"):  # some gateways return 200 with an error body
            _log.warning("OpenAI-compatible endpoint returned an error body: %s", resp.text[:500])
            raise OpenAICompatError("Endpoint returned an error — see server logs.")
        content = data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, ValueError, TypeError, AttributeError) as exc:
        _log.warning("Unexpected OpenAI-compatible response: %s", resp.text[:500])
        raise OpenAICompatError("Endpoint returned a response that isn't OpenAI-compatible.") from exc
    text = _clean(content if isinstance(content, str) else "")
    if not text:
        raise OpenAICompatError("Endpoint returned an empty response.")
    return text


async def list_models(base_url: str, api_key: str | None = None) -> list[str]:
    """IDs from ``GET {base}/models``. Raises OpenAICompatError on failure."""
    url = validate_base_url(base_url)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{url}/models", headers=_headers(api_key))
    except httpx.HTTPError as exc:
        raise _unreachable(exc) from exc
    if resp.status_code != 200:
        raise _http_error(resp)
    try:
        data = resp.json()
        items = data.get("data", []) if isinstance(data, dict) else data
        ids = [m.get("id", "") for m in items if isinstance(m, dict) and m.get("id")]
    except (ValueError, AttributeError, TypeError) as exc:
        _log.warning("Unexpected OpenAI-compatible /models response: %s", resp.text[:500])
        raise OpenAICompatError("Endpoint returned a response that isn't OpenAI-compatible.") from exc
    return sorted(ids)


async def status(base_url: str, model: str, api_key: str | None = None) -> dict:
    """Whether the endpoint is reachable and the chosen model is listed.

    Not every compatible server implements ``/models``; if listing fails but the
    URL is configured we still report unreachable so the UI can warn."""
    result = {"reachable": False, "model_ready": False}
    if not normalize_base_url(base_url) or not model:
        return result
    try:
        models = await list_models(base_url, api_key)
    except OpenAICompatError:
        return result
    result["reachable"] = True
    # An empty listing (some proxies) shouldn't block usage of a typed model id.
    result["model_ready"] = not models or model in models
    return result
