from __future__ import annotations

import os

import anthropic

from .base import LLMProvider


class ClaudeProvider(LLMProvider):
    """Anthropic Claude inference via the official SDK."""

    def __init__(self, config) -> None:
        api_key = config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model_name = config.claude_model
        self._client = anthropic.Anthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return f"claude/{self.model_name}"

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        message = self._client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
