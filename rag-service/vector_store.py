from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors


class InMemoryVectorStore:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None
        self.nn: Optional[NearestNeighbors] = None

    def add(self, vectors: np.ndarray, texts: List[str], metadata: Optional[Dict[str, Any]] = None) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError("Invalid vector shape for this store")
        if vectors.shape[0] != len(texts):
            raise ValueError("Number of vectors must match number of texts")

        for text in texts:
            self.texts.append(text)
            self.metadata.append(metadata or {})

        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack((self.vectors, vectors))

        self._reindex()

    def _reindex(self) -> None:
        if self.vectors is None or len(self.vectors) == 0:
            self.nn = None
            return
        n_neighbors = min(5, len(self.vectors))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        self.nn.fit(self.vectors)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.nn is None:
            return []
        n = min(top_k, len(self.texts))
        distances, indices = self.nn.kneighbors(query_vector.reshape(1, -1), n_neighbors=n)
        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append(
                {
                    "text": self.texts[int(idx)],
                    "score": 1.0 - float(dist),  # cosine similarity approximation
                    "metadata": self.metadata[int(idx)],
                    "index": int(idx),
                }
            )
        results.sort(key=lambda r: r["score"], reverse=True)
        return results


