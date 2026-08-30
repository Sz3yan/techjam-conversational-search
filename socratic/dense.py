"""Dense retrieval as likelihood smoothing.

Pillar I calls for a vector similarity route. A previous attempt on this
problem built one, swept its weight, and measured a loss at every setting --
embeddings blur exactly the fine distinctions that separate the target from its
near neighbours, because the customer quotes catalogue metadata verbatim and
lexical matching is already the right operation.

We think the formulation was wrong rather than the model. That attempt *blended*
semantic similarity into the ranking score, where it competes with exact
matching on exact matching's strongest ground and can only add noise.

Here it enters as a **mixture component of the likelihood** instead:

    P(disclosure | product) = (1-e)*P_lexical + e*P_dense

In log space that is a soft maximum, not a sum -- so a strong lexical match is
untouched by a weak semantic one, and the dense term only asserts itself where
the lexical score has collapsed to nothing. Its job is not to rank; its job is
to stop the posterior zeroing out the true target when the customer's phrasing
shares no words with the catalogue.

That formulation makes a falsifiable prediction: this route should be worth
approximately nothing on the clean public split, and should matter under
paraphrase. The ablation table reports whether that happened.

Requires numpy and a built cache (`scripts/build_embeddings.py`). Absent
either, `available()` is False and the agent runs unchanged.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
HOST = _HOST if _HOST.startswith("http") else f"http://{_HOST}"
QUERY_TIMEOUT = 10.0
# Cosine below this is noise for this corpus; everything is at least loosely
# related because the whole catalogue is one department.
COS_FLOOR = 0.45


class DenseChannel:
    def __init__(self, index, model: str, path: str | Path = ".cache/embeddings.npz") -> None:
        self.index = index
        self.model = model
        self.vectors = None
        self._cache: dict[str, object] = {}
        try:
            import numpy as np
        except ImportError:
            self.np = None
            return
        self.np = np
        cache = Path(path)
        if not cache.exists():
            return
        try:
            with np.load(cache, allow_pickle=False) as data:
                vectors = data["vectors"]
                asins = [str(a) for a in data["asins"]]
        except (OSError, ValueError, KeyError):
            return
        if len(asins) != len(index.asins) or asins != list(index.asins):
            # Row order must match the index, or every score is attached to the
            # wrong product. Rebuild rather than guess.
            return
        self.vectors = vectors.astype(np.float32)

    def available(self) -> bool:
        return self.vectors is not None

    def embed_query(self, text: str):
        if not self.available():
            return None
        if text in self._cache:
            return self._cache[text]
        body = json.dumps(
            {"model": self.model, "input": [text], "truncate": True}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{HOST}/api/embed", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=QUERY_TIMEOUT) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            return None
        raw = payload.get("embeddings") or []
        if not raw:
            return None
        vector = self.np.asarray(raw[0], dtype=self.np.float32)[: self.vectors.shape[1]]
        norm = float(self.np.linalg.norm(vector))
        if norm == 0.0:
            return None
        vector /= norm
        self._cache[text] = vector
        return vector

    def similarity(self, text: str, weight: float):
        """Per-product dense evidence score, or None if unavailable.

        Returned on the same scale as the lexical channels so the caller can
        take a soft maximum rather than a sum.
        """
        vector = self.embed_query(text)
        if vector is None:
            return None
        cosine = self.vectors @ vector
        scaled = (cosine - COS_FLOOR) / (1.0 - COS_FLOOR)
        self.np.clip(scaled, 0.0, 1.0, out=scaled)
        scaled *= weight
        return scaled
