"""Build the dense product embedding cache.

One-off, offline, ~6 minutes on an M-series laptop. The agent never calls this
at scoring time -- it loads the resulting file if present, and runs without the
dense channel if not, so a submission bundle without the cache still scores.

Vectors are truncated to 256 dimensions (embeddinggemma is Matryoshka-trained,
so a prefix of the vector is itself a usable embedding) and stored as float16.
That takes the cache from 154 MB to roughly 25 MB, for a cost in retrieval
quality that the ablation table prices rather than assumes.

Requires numpy. Nothing else in the agent does.

    python3 -m scripts.build_embeddings
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

HOST = "http://127.0.0.1:11434"
DIMS = 256
BATCH = 128


def document(product: dict) -> str:
    """What we embed: identity plus the attributes a shopper would name."""
    parts = [str(product.get("title") or "")]
    parts.extend(str(f) for f in (product.get("features") or [])[:4])
    categories = product.get("categories") or []
    if categories:
        parts.append(" ".join(str(c) for c in categories[-3:]))
    return ". ".join(p for p in parts if p)[:900]


def embed(texts: list[str], model: str) -> list[list[float]]:
    body = json.dumps({"model": model, "input": texts, "truncate": True}).encode("utf-8")
    request = urllib.request.Request(
        f"{HOST}/api/embed", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))["embeddings"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--model", default="embeddinggemma:300m")
    parser.add_argument("--out", default=".cache/embeddings.npz")
    args = parser.parse_args()

    products = [json.loads(line) for line in Path(args.catalog).open(encoding="utf-8")]
    texts = [document(p) for p in products]
    asins = [str(p["parent_asin"]) for p in products]

    vectors = np.zeros((len(texts), DIMS), dtype=np.float16)
    started = time.time()
    for start in range(0, len(texts), BATCH):
        chunk = texts[start : start + BATCH]
        raw = np.asarray(embed(chunk, args.model), dtype=np.float32)[:, :DIMS]
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        vectors[start : start + len(chunk)] = (raw / norms).astype(np.float16)
        if start % (BATCH * 40) == 0:
            done = start + len(chunk)
            rate = done / (time.time() - started)
            print(
                f"{done}/{len(texts)}  {rate:.0f}/s  eta {(len(texts) - done) / rate / 60:.1f} min",
                flush=True,
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, vectors=vectors, asins=np.array(asins))
    size = out.stat().st_size / 2**20
    print(f"wrote {out} ({size:.1f} MB) in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
