from __future__ import annotations

import re

from .base import LLMProvider


class MockProvider(LLMProvider):
    """
    Offline demo provider — answers from context using keyword matching.
    Used when no Ollama/Gemini/Claude is available (CI, offline demo, testing).
    """

    @property
    def name(self) -> str:
        return "mock/keyword-demo"

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        q_match = re.search(r"Question:\s*(.+?)\n", prompt)
        question = q_match.group(1).strip().lower() if q_match else ""

        ctx_match = re.search(r"Context:\n(.+?)\nQuestion:", prompt, re.DOTALL)
        context = ctx_match.group(1).strip() if ctx_match else ""

        # NER extraction prompt
        if "named entities" in prompt.lower() or "entities:" in prompt.lower():
            text_match = re.search(r"Text:\s*(.+?)\n\nEntities:", prompt, re.DOTALL)
            text = text_match.group(1).strip() if text_match else prompt
            pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
            entities = list({m for m in re.findall(pattern, text) if len(m) > 2})
            return ", ".join(entities[:10]) if entities else "No entities found"

        # QA: find most relevant sentence and extract a short answer
        keywords = re.findall(r"\b\w{4,}\b", question)
        best_sentence, best_score = "", 0
        for sent in re.split(r"[.\n]", context):
            score = sum(1 for kw in keywords if kw in sent.lower())
            if score > best_score:
                best_score, best_sentence = score, sent.strip()

        if best_sentence:
            short = re.search(
                r"\b[A-Z][a-zA-Z\s]+\b|\b\d{4}\b|\byes\b|\bno\b", best_sentence
            )
            return short.group(0).strip() if short else best_sentence[:80]

        return "Answer not found in context."
