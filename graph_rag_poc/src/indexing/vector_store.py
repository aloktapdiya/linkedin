from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


class VectorStore:
    """Cosine-similarity vector store backed by FAISS (numpy fallback if FAISS absent)."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.id_map: List[str] = []
        self.metadata: Dict[str, Any] = {}
        if _FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(dim)
            self._use_faiss = True
        else:
            self._vectors: List[np.ndarray] = []
            self._use_faiss = False

    def add(
        self,
        node_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        vec = _normalize(embedding)
        if self._use_faiss:
            self._index.add(vec.reshape(1, -1).astype(np.float32))
        else:
            self._vectors.append(vec.astype(np.float32))
        self.id_map.append(node_id)
        if metadata:
            self.metadata[node_id] = metadata

    def search(
        self, query: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        if not self.id_map:
            return []
        k = min(top_k, len(self.id_map))
        q = _normalize(query)
        if self._use_faiss:
            scores, indices = self._index.search(
                q.reshape(1, -1).astype(np.float32), k
            )
            return [
                (self.id_map[idx], float(scores[0][i]))
                for i, idx in enumerate(indices[0])
                if 0 <= idx < len(self.id_map)
            ]
        matrix = np.stack(self._vectors)
        sims = matrix @ q.astype(np.float32)
        top_idx = np.argsort(sims)[::-1][:k]
        return [(self.id_map[i], float(sims[i])) for i in top_idx]

    def __len__(self) -> int:
        return len(self.id_map)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v
