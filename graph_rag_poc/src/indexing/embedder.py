from __future__ import annotations

from typing import List

import numpy as np


class Embedder:
    """
    Sentence embedder with two backends:
      1. sentence-transformers (preferred, requires internet on first use)
      2. TF-IDF (offline fallback — no download needed)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._backend: str = "unknown"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._backend = "sentence-transformers"
            print(f"[Embedder] Using sentence-transformers ({self.model_name})")
        except Exception:
            self._model = _TFIDFEmbedder()
            self._backend = "tfidf"
            print("[Embedder] HuggingFace unavailable — using TF-IDF offline fallback")

    def embed(self, texts: List[str]) -> np.ndarray:
        self._load()
        if self._backend == "tfidf":
            return self._model.embed(texts)
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class _TFIDFEmbedder:
    """
    Offline TF-IDF embedder using sklearn.
    Produces sparse vectors projected to a fixed dense dimension via SVD.
    No internet required.
    """

    DIM = 384

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self._vectorizer = TfidfVectorizer(
            max_features=4096, sublinear_tf=True, strip_accents="unicode"
        )
        self._svd = TruncatedSVD(n_components=self.DIM, random_state=42)
        self._fitted = False
        self._corpus: List[str] = []

    def _fit_or_partial(self, texts: List[str]) -> None:
        self._corpus.extend(texts)
        tfidf = self._vectorizer.fit_transform(self._corpus)
        n_components = min(self.DIM, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        self._svd.n_components = max(1, n_components)
        self._svd.fit(tfidf)
        self._fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        self._fit_or_partial(texts)
        tfidf = self._vectorizer.transform(texts)
        vecs = self._svd.transform(tfidf).astype(np.float32)
        if vecs.shape[1] < self.DIM:
            pad = np.zeros((vecs.shape[0], self.DIM - vecs.shape[1]), dtype=np.float32)
            vecs = np.hstack([vecs, pad])
        return vecs
