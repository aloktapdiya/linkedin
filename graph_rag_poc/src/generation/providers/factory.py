from __future__ import annotations

import os

import requests

from .base import LLMProvider


def is_ollama_running(base_url: str, timeout: float = 2.0) -> bool:
    """Return True if an Ollama instance is reachable at *base_url*."""
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def create_provider(config) -> LLMProvider:
    """
    Auto-detect available inference engines and return the highest-priority one.

    Priority order
    --------------
    1. Ollama  (local, zero marginal cost)   <-- default
    2. Gemini  (remote, GEMINI_API_KEY)      <-- first remote fallback
    3. Claude  (remote, ANTHROPIC_API_KEY)   <-- last resort
    """
    from .ollama import OllamaProvider
    from .gemini import GeminiProvider
    from .claude import ClaudeProvider

    reasons: list[str] = []

    # ------------------------------------------------------------------ #
    # 1. Ollama
    # ------------------------------------------------------------------ #
    if is_ollama_running(config.ollama_base_url):
        print(
            f"[Provider] Ollama detected at {config.ollama_base_url} — "
            f"using model '{config.ollama_model}'"
        )
        return OllamaProvider(config)
    reasons.append(f"Ollama: not reachable at {config.ollama_base_url}")

    # ------------------------------------------------------------------ #
    # 2. Gemini
    # ------------------------------------------------------------------ #
    gemini_key = (
        config.gemini_api_key
        or os.getenv("GEMINI_API_KEY", "")
        or os.getenv("GOOGLE_API_KEY", "")
    )
    if gemini_key:
        config.gemini_api_key = gemini_key
        try:
            provider = GeminiProvider(config)
            print(
                f"[Provider] Ollama unavailable — falling back to Gemini "
                f"(model '{config.gemini_model}')"
            )
            return provider
        except Exception as exc:
            reasons.append(f"Gemini: init failed ({exc})")
    else:
        reasons.append("Gemini: no API key (set GEMINI_API_KEY or GOOGLE_API_KEY)")

    # ------------------------------------------------------------------ #
    # 3. Claude
    # ------------------------------------------------------------------ #
    anthropic_key = config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        config.anthropic_api_key = anthropic_key
        print(
            f"[Provider] Gemini unavailable — falling back to Claude "
            f"(model '{config.claude_model}')"
        )
        return ClaudeProvider(config)
    reasons.append("Claude: no API key (set ANTHROPIC_API_KEY)")

    raise RuntimeError(
        "No inference engine available. Tried:\n"
        + "\n".join(f"  ✗ {r}" for r in reasons)
        + "\n\nStart Ollama locally, or set GEMINI_API_KEY / ANTHROPIC_API_KEY."
    )
