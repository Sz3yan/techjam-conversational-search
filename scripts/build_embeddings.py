"""Precompute and cache catalog embeddings for the optional dense route.

Not required: the agent scores identically without it (W_DENSE is 0.0, disabled
by measurement -- see SOLUTION.md). Kept so the negative result is reproducible.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/build_embeddings.py
"""

from __future__ import annotations

import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from starter.dense import CACHE_IDS, CACHE_VECTORS, MODEL_NAME, embedding_text

CATALOG = "data/catalog.jsonl"
BATCH_SIZE = 256


def main() -> None:
    ids: list[str] = []
    texts: list[str] = []
    with open(CATALOG, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            ids.append(str(product["parent_asin"]))
            texts.append(embedding_text(product))

    print(f"encoding {len(texts)} products with {MODEL_NAME}", flush=True)
    model = SentenceTransformer(MODEL_NAME)
    started = time.time()
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype("float32")
    print(f"encoded in {time.time() - started:.1f}s  shape={vectors.shape}", flush=True)

    np.save(CACHE_VECTORS, vectors)
    CACHE_IDS.write_text(json.dumps(ids), encoding="utf-8")
    print(f"cached -> {CACHE_VECTORS} ({vectors.nbytes / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
