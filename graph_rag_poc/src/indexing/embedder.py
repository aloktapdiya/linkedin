from __future__ import annotations

from typing import List

import numpy as np


class Embedder:
    """Lazy-loading sentence embedding wrapper."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        self._load()
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
