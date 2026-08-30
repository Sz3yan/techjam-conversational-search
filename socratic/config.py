"""Agent configuration.

Every design decision that could plausibly have gone the other way is a field
here, so `scripts/experiment.py` can price it instead of the report asserting
it. Defaults are the shipped configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentConfig:
    # Posterior temperature. This is a calibration parameter, not a ranking
    # one: the width policy compares probabilities against the scoring
    # function, so an over-confident posterior recommends too early and an
    # under-confident one stalls. Ranking is unchanged by it.
    alpha: float = 6.0
    # Weight of an exact clause match relative to token coverage.
    w_clause: float = 2.0

    # Likelihood channels folded into the posterior.
    channels: frozenset[str] = frozenset({"clause", "token", "category", "price"})

    # Question selection: "eig" estimates expected information gain,
    # "ladder" walks a fixed hand-authored priority order, "none" never asks.
    question_policy: str = "eig"

    # Recommendation width: "expected_score" runs the expectimax,
    # "fixed" always returns `fixed_width`, "schedule" mimics a hand-tuned ramp.
    width_policy: str = "expected_score"
    fixed_width: int = 10

    # Intent override. Pillar II asks for "slot erasure and rewriting" and all
    # three behaviours are implemented, but erasure is *measured* to lose here
    # and the default is the one that wins. See SOLUTION.md: in every override
    # session both the old and the new preference describe the same target
    # product, so the retracted evidence was still true of it.
    #   "keep"    -- retain all evidence (default; measured best)
    #   "retract" -- erase only the slot the new statement contradicts
    #   "erase"   -- wipe every prior constraint
    override_mode: str = "keep"

    # Products already shown and not accepted are demoted.
    eliminate_shown: bool = True

    # Search depth of the width expectimax, and how much of the posterior head
    # the question policy reasons over.
    depth: int = 2
    candidates: int = 120

    # Local model use. "auto" uses Ollama when the daemon answers a health
    # check and silently falls back otherwise; "off" never contacts it.
    llm_mode: str = "auto"
    llm_model: str = "qwen3.5:2b-mlx"
    embed_model: str = "embeddinggemma:300m"
    # "message" is a presentation feature, not a scoring one: it costs a
    # model call on every turn and cannot change the ranking. Off by default,
    # on for the demo script.
    llm_uses: frozenset[str] = frozenset({"segment", "rerank"})
    # Fire the semantic reranker only when the posterior cannot separate its
    # top candidates -- this ratio is the definition of "cannot separate".
    rerank_ratio: float = 1.5
    rerank_pool: int = 6
    # ...and only when the leader is a real contender. When the whole head is
    # flat the model has no more to go on than the arithmetic does.
    rerank_min_prob: float = 0.02

    # Dense channel. Off by default until the ablation says otherwise; it is a
    # mixture component, so `dense_offset` is the log-odds penalty that keeps a
    # semantic match from ever outbidding a real lexical one.
    dense_weight: float = 1.2
    dense_offset: float = 0.6

    def replace(self, **changes) -> "AgentConfig":
        from dataclasses import replace as _replace
        return _replace(self, **changes)
