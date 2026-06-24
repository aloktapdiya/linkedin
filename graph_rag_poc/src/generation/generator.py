from __future__ import annotations

from typing import Any, Dict, List

from .providers.base import LLMProvider
from .providers.factory import create_provider


class Generator:
    """LLM answer generator — provider selected automatically at first use."""

    _QA_PROMPT = (
        "You are a precise question-answering assistant.\n"
        "Use ONLY the provided context to answer the question.\n"
        "Give a concise answer — a short phrase or a few words when possible.\n"
        "If the answer is not in the context, respond with: "
        "'Answer not found in context.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def __init__(self, config) -> None:
        self.config = config
        self._provider: LLMProvider | None = None

    def _get_provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = create_provider(self.config)
        return self._provider

    def generate(self, query: str, context_passages: List[Dict[str, Any]]) -> str:
        context = self._format_context(context_passages)
        prompt = self._QA_PROMPT.format(context=context, question=query)
        provider = self._get_provider()
        return provider.generate(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

    def _format_context(self, passages: List[Dict[str, Any]]) -> str:
        if not passages:
            return "No context available."
        parts = []
        for i, p in enumerate(passages, 1):
            title = p.get("title") or f"Passage {i}"
            parts.append(f"[{i}] {title}\n{p.get('text', '')}")
        return "\n\n".join(parts)
