from __future__ import annotations

import requests

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    """Local Ollama inference via its HTTP REST API."""

    def __init__(self, config) -> None:
        self.base_url = config.ollama_base_url.rstrip("/")
        self.model = config.ollama_model
        self._session = requests.Session()

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        resp = self._session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def list_models(self) -> list[str]:
        resp = self._session.get(f"{self.base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
