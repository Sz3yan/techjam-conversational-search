"""Rerank a BM25 candidate pool against accumulated session state."""

from __future__ import annotations

from starter.retrieve import CatalogIndex, normalize, tokenize
from starter.state import SessionState

# Tuned on the public set; see docs in the project README for the sweep.
W_BM25 = 1.0        # trust in the retriever's own ordering
W_CONSTRAINT = 4.0  # IDF-weighted coverage of disclosed constraints
W_PHRASE = 1.5      # verbatim clause match (public-set signal; deliberately bounded)
W_CATEGORY = 2.5    # coverage of the stated category
# Disabled by measurement, not by omission. A MiniLM dense route over the same
# candidate pool was built and swept (weights 1-8, and gated to only fire when
# few constraints were known, and on boundary sessions alone). It never beat
# sparse-only: 0.9327 -> 0.9285 at weight 3.0. The disclosed constraints are
# literal catalog metadata ("100% Leather", "Buckle closure"), so exact lexical
# matching is the correct operation; embeddings blur precisely the attribute
# distinctions that decide the target. See SOLUTION.md.
W_DENSE = 0.0
P_RULED_OUT = 100.0 # demotion for products already shown and thus not the target


def _weighted(index: CatalogIndex, tokens: list[str]) -> list[tuple[str, float]]:
    return [(token, index.idf(token)) for token in tokens]


def _coverage(index: CatalogIndex, parent_asin: str, weighted: list[tuple[str, float]]) -> float:
    """Fraction of a phrase's IDF mass present in the product text."""
    if not weighted:
        return 0.0
    total = sum(weight for _, weight in weighted)
    if total <= 0:
        return 0.0
    found = sum(weight for token, weight in weighted if index.contains(parent_asin, token))
    return found / total


def rerank(
    index: CatalogIndex,
    state: SessionState,
    candidates: list[str],
    phrase_weight: float = W_PHRASE,
    dense_scores: dict[str, float] | None = None,
) -> list[str]:
    """Order candidates best-first. Pure reordering: nothing is dropped."""
    if not candidates:
        return []

    constraints = [
        (_weighted(index, tokenize(constraint["text"])), normalize(constraint["text"]).strip())
        for constraint in state.constraints
    ]
    constraints = [(weighted, phrase) for weighted, phrase in constraints if weighted]
    category = _weighted(index, tokenize(state.category or ""))
    ruled_out = state.ruled_out()
    pool = float(len(candidates))

    def score(item: tuple[int, str]) -> float:
        rank, parent_asin = item
        value = W_BM25 * (1.0 - rank / pool)
        if constraints:
            hit = 0.0
            for weighted, phrase in constraints:
                hit += _coverage(index, parent_asin, weighted)
                if phrase and f" {phrase} " in index.blob.get(parent_asin, ""):
                    hit += phrase_weight
            value += W_CONSTRAINT * hit / len(constraints)
        if category:
            blob = index.category_blob.get(parent_asin, "")
            found = sum(w for token, w in category if f" {token} " in blob)
            total = sum(w for _, w in category)
            value += W_CATEGORY * (found / total if total else 0.0)
        if dense_scores:
            value += W_DENSE * dense_scores.get(parent_asin, 0.0)
        if parent_asin in ruled_out:
            value -= P_RULED_OUT
        return -value

    return [parent_asin for _, parent_asin in sorted(enumerate(candidates), key=score)]
