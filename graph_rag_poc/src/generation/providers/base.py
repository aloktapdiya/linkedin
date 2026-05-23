from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base for all inference backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider/model identifier."""
        ...

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        """Generate a completion for *prompt* and return the text."""
        ...
