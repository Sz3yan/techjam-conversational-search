"""Decision layer: which question to ask, and whether to answer yet.

Two decisions are made here, both by maximising expected score rather than by
a hand-tuned schedule.

**Which question.** Asking is free -- the API lets one turn carry both a
question and a recommendation list -- so the agent always asks. The choice of
`ask_attribute` is made by expected information gain: for every attribute we
predict what each plausible product's owner *would* say if asked, group the
candidates by the answer they'd give, and pick the attribute whose answer
splits the posterior most evenly. Products that would answer identically are
indistinguishable under that question, so the entropy that survives the answer
is exactly the entropy inside those groups.

**Whether to answer.** The session ends at the *first* hit, so showing the
target at rank 9 banks 0.3/9 and destroys every later chance of showing it
first. Recommending is therefore a commitment with a real cost, and the width
of the list is chosen by a short expectimax over the scoring function itself:

    session score = 0.5*hit + 0.3*(1/rank) + 0.2*(11 - turn)/10

This is Pillar II's "over-generality cutoff" in its decision-theoretic form.
When the posterior is flat the arithmetic says stay quiet and ask; as it
sharpens the same arithmetic opens the list up. Nothing about the widening
schedule is hardcoded -- it falls out.
"""

from __future__ import annotations

import math

# The attributes the API will accept in `ask_attribute`.
ALLOWED_ATTRIBUTES = (
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
)
MAX_TURNS = 10
TOP_K = 10

W_HIT = 0.50
W_MRR = 0.30
W_EFF = 0.20


def session_score(rank: int, turn: int) -> float:
    """What one session is worth if it converts now at this rank."""
    return W_HIT + W_MRR / rank + W_EFF * (MAX_TURNS + 1 - turn) / 10.0


def entropy(probabilities) -> float:
    total = 0.0
    for probability in probabilities:
        if probability > 0.0:
            total -= probability * math.log2(probability)
    return total


def answer_signature(clauses, buckets, attribute: str, disclosed: frozenset[str]) -> tuple[str, ...]:
    """What this product's owner would disclose if asked about `attribute`.

    `other` is modelled as an open question -- "anything else that matters?" --
    which a shopper answers with their most salient remaining requirement
    rather than with nothing. Specific attributes only unlock their own bucket.
    The customer volunteers at most two things per turn.
    """
    out: list[str] = []
    for clause, bucket in zip(clauses, buckets):
        if clause in disclosed:
            continue
        if attribute == "other" or bucket == attribute:
            out.append(clause)
            if len(out) == 2:
                break
    return tuple(out)


class Policy:
    def __init__(self, index, *, depth: int = 2, candidates: int = 120) -> None:
        self.index = index
        self.depth = depth
        self.candidates = candidates

    # ------------------------------------------------------------ questions

    def rank_questions(
        self,
        product_ids: list[int],
        probabilities: list[float],
        disclosed: frozenset[str],
        asked: frozenset[str],
        refused: frozenset[str],
    ) -> list[tuple[str, float, dict]]:
        """Expected information gain for each allowed attribute, best first."""
        total = sum(probabilities) or 1.0
        normalized = [p / total for p in probabilities]
        prior_entropy = entropy(normalized)
        scored: list[tuple[str, float, dict]] = []
        for attribute in ALLOWED_ATTRIBUTES:
            if attribute in refused:
                continue
            groups: dict[tuple[str, ...], list[float]] = {}
            for pid, probability in zip(product_ids, normalized):
                signature = answer_signature(
                    self.index.clauses[pid], self.index.clause_buckets[pid], attribute, disclosed
                )
                groups.setdefault(signature, []).append(probability)
            expected = 0.0
            for members in groups.values():
                mass = sum(members)
                if mass <= 0.0:
                    continue
                expected += mass * entropy([m / mass for m in members])
            gain = prior_entropy - expected
            # Asking the same attribute twice cannot unlock a new bucket.
            if attribute in asked:
                gain *= 0.25
            scored.append((attribute, gain, groups))
        scored.sort(key=lambda row: row[1], reverse=True)
        return scored

    # --------------------------------------------------------------- width

    def choose_width(
        self,
        probabilities: list[float],
        tail: float,
        turn: int,
        groups: dict[tuple[str, ...], list[float]] | None = None,
    ) -> tuple[int, float]:
        """Pick how many products to show.

        `probabilities` is the head of the posterior and `tail` is the mass
        sitting on every product outside it. The tail is carried explicitly
        rather than normalised away: conditioning on "the target was not in the
        list I just showed" has to dilute across the whole catalogue, and a
        head that has been renormalised in isolation looks confident exactly
        when the agent knows least.
        """
        best_width, best_value = 0, -1.0
        for width in range(0, TOP_K + 1):
            if width > len(probabilities):
                break
            value = self._value(probabilities, tail, turn, self.depth, groups, width)
            if value > best_value:
                best_width, best_value = width, value
        return best_width, best_value

    def value(self, probabilities, tail, turn, depth, groups=None) -> float:
        best = 0.0
        for width in range(0, TOP_K + 1):
            if width > len(probabilities):
                break
            candidate = self._value(probabilities, tail, turn, depth, groups, width)
            if candidate > best:
                best = candidate
        return best

    def _value(self, probabilities, tail, turn, depth, groups, width) -> float:
        if turn > MAX_TURNS:
            return 0.0
        immediate = 0.0
        shown = 0.0
        for rank in range(1, width + 1):
            probability = probabilities[rank - 1]
            immediate += probability * session_score(rank, turn)
            shown += probability
        survivor = 1.0 - shown
        if survivor <= 1e-9 or depth <= 0 or turn >= MAX_TURNS:
            return immediate
        rest = [p / survivor for p in probabilities[width:]]
        return immediate + survivor * self._after_question(
            rest, tail / survivor, turn + 1, depth - 1, groups
        )

    def _after_question(self, probabilities, tail, turn, depth, groups) -> float:
        """Value of the next turn once the pending question has been answered.

        With a model of the possible answers we average over them -- that
        expectation is what makes staying quiet worth something. Groups are
        modelled at the first level only; deeper levels reuse the distribution
        unconditioned, which understates the value of asking and so biases the
        agent towards answering rather than towards stalling.
        """
        if not groups:
            return self.value(probabilities, tail, turn, depth)
        head_mass = sum(probabilities)
        if head_mass <= 0.0:
            return self.value(probabilities, tail, turn, depth)
        expected = 0.0
        group_total = sum(sum(members) for members in groups.values()) or 1.0
        for members in groups.values():
            mass = sum(members)
            if mass <= 0.0:
                continue
            share = mass / group_total
            inner = sorted((m / mass for m in members), reverse=True)
            # Within an answer group the head keeps `head_mass` of the belief;
            # the unseen tail is unaffected by the answer.
            scaled = [p * head_mass for p in inner]
            expected += share * self.value(scaled, tail, turn, depth)
        return expected
