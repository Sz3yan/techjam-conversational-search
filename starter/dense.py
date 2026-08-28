"""Optional dense retrieval route.

Embeddings are precomputed once and cached to disk, so scoring runs need only
the encoder for the short query string. Every import is guarded: if numpy or
sentence-transformers is unavailable the agent silently falls back to the
stdlib sparse path, which is the configuration the submission guarantees.
"""

from __future__ import annotations

import json
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_VECTORS = Path("data/embeddings.npy")
CACHE_IDS = Path("data/embeddings_ids.json")
MAX_CHARS = 900

try:  # pragma: no cover - availability depends on the host environment
    import numpy as _np
except ImportError:
    _np = None


def embedding_text(product: dict) -> str:
    """Compact semantic summary of a product: what it is, not every bullet."""
    parts = [str(product.get("title") or "")]
    categories = product.get("categories") or []
    if isinstance(categories, list):
        parts.append(" ".join(str(value) for value in categories))
    features = product.get("features") or []
    if isinstance(features, list):
        parts.append(" ".join(str(value) for value in features[:4]))
    store = product.get("store")
    if store:
        parts.append(str(store))
    return " ".join(part for part in parts if part)[:MAX_CHARS]


class DenseIndex:
    """Cosine similarity over cached product embeddings. Optional by design."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.available = False
        self.ids: list[str] = []
        self.vectors = None
        self._encoder = None
        if _np is None or not (CACHE_VECTORS.exists() and CACHE_IDS.exists()):
            return
        try:
            self.vectors = _np.load(CACHE_VECTORS)
            self.ids = json.loads(CACHE_IDS.read_text(encoding="utf-8"))
            self.position = {asin: i for i, asin in enumerate(self.ids)}
            self.available = len(self.ids) == self.vectors.shape[0]
        except Exception:
            self.available = False

    def _encode(self, text: str):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(MODEL_NAME)
        vector = self._encoder.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return vector[0]

    def similarity(self, query: str, candidates: list[str]):
        """Cosine similarity of each candidate to the query, in [0, 1]."""
        if not self.available or not candidates or not query.strip():
            return {}
        try:
            vector = self._encode(query)
        except Exception:
            return {}
        rows, keep = [], []
        for asin in candidates:
            index = self.position.get(asin)
            if index is not None:
                rows.append(index)
                keep.append(asin)
        if not rows:
            return {}
        scores = self.vectors[rows] @ vector
        return {asin: float((score + 1.0) / 2.0) for asin, score in zip(keep, scores)}

    def search(self, query: str, limit: int) -> list[str]:
        """Top-`limit` catalog entries by semantic similarity."""
        if not self.available or not query.strip():
            return []
        try:
            vector = self._encode(query)
        except Exception:
            return []
        scores = self.vectors @ vector
        top = _np.argpartition(-scores, min(limit, len(scores) - 1))[:limit]
        top = top[_np.argsort(-scores[top])]
        return [self.ids[i] for i in top]
