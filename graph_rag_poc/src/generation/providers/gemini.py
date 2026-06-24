from __future__ import annotations

import google.generativeai as genai

from .base import LLMProvider


class GeminiProvider(LLMProvider):
    """Google Gemini inference via the generativeai SDK."""

    def __init__(self, config) -> None:
        self.model_name = config.gemini_model
        genai.configure(api_key=config.gemini_api_key)
        self._model = genai.GenerativeModel(self.model_name)

    @property
    def name(self) -> str:
        return f"gemini/{self.model_name}"

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        gen_cfg = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        response = self._model.generate_content(prompt, generation_config=gen_cfg)
        return response.text.strip()
