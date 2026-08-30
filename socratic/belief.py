"""Posterior over the catalogue, maintained across a conversation.

The hypothesis space is every product in the catalogue. Each customer
disclosure is evidence: a string the target product is likely to assert about
itself and other products are not. We score how well each product explains the
disclosure and treat that score as a log-likelihood, so the running state is a
calibrated distribution rather than a bag of matched keywords.

That distinction is the point of the module. A ranked list only has to be in
the right order; a distribution also has to say *how sure* it is, and the
policy layer spends that number -- on choosing which question to ask, and on
deciding whether to answer at all.

Evidence is append-only and each item stays individually addressable, so an
intent override retracts exactly the contradicted item and the posterior is
recomputed from what remains.
"""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass, field

from .index import CatalogIndex, clause_keys
from .text import classify, find_price, normalize, tokenize

# Weight of an exact clause match relative to token coverage. A clause match is
# strong evidence, but scaled by how rare the clause is: matching "Imported"
# (13,889 products) says almost nothing, matching a full fabric composition
# (3 products) says almost everything.
W_CLAUSE = 2.0
# Log-likelihood temperature. Higher makes the posterior commit harder.
ALPHA = 6.0
W_CATEGORY = 1.0
W_PRICE = 0.6


@dataclass
class Evidence:
    """One thing the customer told us."""

    text: str
    turn: int
    bucket: str
    kind: str = "constraint"          # constraint | category | price
    active: bool = True               # retracted evidence stays on the ledger
    scores: array | None = field(default=None, repr=False)


class Belief:
    def __init__(
        self,
        index: CatalogIndex,
        channels: frozenset[str] | None = None,
        dense=None,
        dense_weight: float = 1.2,
        dense_offset: float = 0.6,
        alpha: float = ALPHA,
        w_clause: float = W_CLAUSE,
    ) -> None:
        self.index = index
        self.dense = dense
        self.dense_weight = dense_weight
        self.dense_offset = dense_offset
        self.alpha = alpha
        self.w_clause = w_clause
        self.channels = channels if channels is not None else frozenset(
            {"clause", "token", "category", "price"}
        )
        self.n = len(index)
        self.ledger: list[Evidence] = []
        self.ruled_out: set[int] = set()
        self._logits: list[float] | None = None
        self._log_n = math.log(self.n)

    # ------------------------------------------------------------- evidence

    def add(self, text: str, turn: int, kind: str = "constraint") -> Evidence | None:
        norm = normalize(text)
        if not norm:
            return None
        if any(e.text == norm and e.kind == kind for e in self.ledger):
            return None
        item = Evidence(text=norm, turn=turn, bucket=classify(norm), kind=kind)
        item.scores = self._score(item)
        self.ledger.append(item)
        self._logits = None
        return item

    def retract(self, predicate) -> list[Evidence]:
        """Deactivate ledger items matching `predicate`. Returns what changed."""
        changed = [e for e in self.ledger if e.active and predicate(e)]
        for item in changed:
            item.active = False
        if changed:
            self._logits = None
        return changed

    def rule_out(self, product_ids) -> None:
        """Products shown on a previous turn that did not end the session."""
        before = len(self.ruled_out)
        self.ruled_out.update(product_ids)
        if len(self.ruled_out) != before:
            self._logits = None

    def readmit(self) -> None:
        """Undo eliminations (an override session cannot convert early, so
        products shown before the override were never really excluded)."""
        if self.ruled_out:
            self.ruled_out.clear()
            self._logits = None

    @property
    def disclosed(self) -> list[str]:
        return [e.text for e in self.ledger if e.active and e.kind == "constraint"]

    # -------------------------------------------------------------- scoring

    def _score(self, item: Evidence) -> array:
        scores = array("f", bytes(4 * self.n))
        if item.kind == "category" and "category" in self.channels:
            self._score_category(item.text, scores, W_CATEGORY)
            return scores
        if "token" in self.channels:
            self._score_tokens(item.text, scores, 1.0)
        if "clause" in self.channels:
            self._score_clause(item.text, scores)
        if "price" in self.channels:
            price = find_price(item.text)
            if price is not None:
                self._score_price(price, scores, W_PRICE)
        if "dense" in self.channels:
            self._mix_dense(item.text, scores)
        return scores

    def _mix_dense(self, text: str, scores: array) -> None:
        """Fold the dense channel in as a mixture, not as a sum.

        `(1-e)*P_lex + e*P_dense` is a soft maximum in log space. Taking the
        maximum directly is the sharp limit of that, and it is the behaviour we
        want: a confident lexical match is never diluted, and the semantic
        score only speaks where the lexical one is silent.
        """
        if self.dense is None or not self.dense.available():
            return
        scaled = self.dense.similarity(text, self.dense_weight)
        if scaled is None:
            return
        floor = self.dense_offset
        candidates = self.dense.np.nonzero(scaled > floor)[0]
        for pid in candidates.tolist():
            value = float(scaled[pid]) - floor
            if value > scores[pid]:
                scores[pid] = value

    def _score_tokens(self, text: str, scores: array, weight: float) -> None:
        terms = set(tokenize(text))
        if not terms:
            return
        total = sum(self.index.term_idf(t) for t in terms)
        if total <= 0:
            return
        postings = self.index.token_postings
        for term in terms:
            share = weight * self.index.term_idf(term) / total
            for pid in postings.get(term, ()):
                scores[pid] += share

    def _score_clause(self, text: str, scores: array) -> None:
        clause_index = self.index.clause_index
        for key in clause_keys(text):
            ids = clause_index.get(key)
            if not ids:
                continue
            # Rarity-scaled: a clause held by half the catalogue is not evidence.
            rarity = math.log(self.n / (len(ids) + 1.0)) / self._log_n
            if rarity <= 0.0:
                continue
            bonus = self.w_clause * rarity
            for pid in ids:
                scores[pid] += bonus

    def _score_category(self, text: str, scores: array, weight: float) -> None:
        terms = set(tokenize(text))
        if not terms:
            return
        for pid, cats in enumerate(self.index.categories):
            if not cats:
                continue
            cat_terms = set(tokenize(" ".join(cats)))
            if not cat_terms:
                continue
            overlap = len(terms & cat_terms) / len(terms)
            if overlap:
                scores[pid] += weight * overlap

    def _score_price(self, price: float, scores: array, weight: float) -> None:
        for pid, value in enumerate(self.index.prices):
            if value is None or value <= 0:
                continue
            ratio = abs(math.log(value / price))
            if ratio < 0.7:  # within roughly a factor of two
                scores[pid] += weight * (1.0 - ratio / 0.7)

    # ------------------------------------------------------------- posterior

    def logits(self) -> list[float]:
        if self._logits is not None:
            return self._logits
        combined = [0.0] * self.n
        for item in self.ledger:
            if not item.active or item.scores is None:
                continue
            row = item.scores
            for pid in range(self.n):
                value = row[pid]
                if value:
                    combined[pid] += value
        for pid in range(self.n):
            combined[pid] *= self.alpha
        for pid in self.ruled_out:
            combined[pid] -= 12.0
        self._logits = combined
        return combined

    def posterior(self, top: int = 200) -> tuple[list[int], list[float], float]:
        """Return the `top` most likely products, their probabilities, and the
        entropy of the full distribution in bits."""
        logits = self.logits()
        peak = max(logits)
        weights = [math.exp(value - peak) for value in logits]
        total = sum(weights)
        if total <= 0:
            uniform = 1.0 / self.n
            return list(range(top)), [uniform] * top, math.log2(self.n)
        entropy = 0.0
        for weight in weights:
            if weight > 0.0:
                probability = weight / total
                entropy -= probability * math.log2(probability)
        order = sorted(range(self.n), key=lambda pid: weights[pid], reverse=True)[:top]
        return order, [weights[pid] / total for pid in order], entropy
