from __future__ import annotations

import os
from typing import Any, Dict, List

import anthropic


class Generator:
    """Answer generator backed by Anthropic Claude."""

    def __init__(self, config) -> None:
        self.config = config
        self._client: anthropic.Anthropic | None = None

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            api_key = self.config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def generate(self, query: str, context_passages: List[Dict[str, Any]]) -> str:
        context = self._format_context(context_passages)
        prompt = (
            "You are a precise question-answering assistant.\n"
            "Use ONLY the provided context to answer the question.\n"
            "Give a concise answer — a short phrase or a few words when possible.\n"
            "If the answer is not in the context, say: 'Answer not found in context.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        message = self._get_client().messages.create(
            model=self.config.model_id,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    def _format_context(self, passages: List[Dict[str, Any]]) -> str:
        if not passages:
            return "No context available."
        parts = []
        for i, p in enumerate(passages, 1):
            title = p.get("title") or f"Passage {i}"
            parts.append(f"[{i}] {title}\n{p.get('text', '')}")
        return "\n\n".join(parts)
