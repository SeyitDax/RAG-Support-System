from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from vector_store import InMemoryVectorStore


class SimpleEmbedder:
    """Deterministic, lightweight embedder for demo purposes.

    Uses a seeded normal generator based on the text hash to produce a stable
    pseudo-embedding. Not suitable for production.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for text in texts:
            seed = abs(hash(text)) % (2**32)
            rng = np.random.default_rng(seed)
            vectors.append(rng.standard_normal(self.dimension))
        return np.vstack(vectors) if vectors else np.zeros((0, self.dimension))


class RAGEngine:
    def __init__(self, dimension: int = 384) -> None:
        self.embedder = SimpleEmbedder(dimension=dimension)
        self.store = InMemoryVectorStore(dimension=dimension)

    def ingest(self, documents: List[str], metadata: Dict[str, Any] | None = None) -> int:
        if not documents:
            return 0
        vectors = self.embedder.embed(documents)
        self.store.add(vectors=vectors, texts=documents, metadata=metadata or {})
        return len(documents)

    def query(self, question: str, top_k: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
        query_vec = self.embedder.embed([question])
        results = self.store.search(query_vec[0], top_k=top_k)
        answer = results[0]["text"] if results else ""
        return answer, results


