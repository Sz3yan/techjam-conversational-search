"""Per-session conversational state."""

from __future__ import annotations

from dataclasses import dataclass, field

from .belief import Belief


@dataclass
class Session:
    belief: Belief
    profile: dict = field(default_factory=dict)
    category: str | None = None
    route: str = "unknown"          # buying | browsing
    asked: set[str] = field(default_factory=set)
    refused: set[str] = field(default_factory=set)
    shown: list[int] = field(default_factory=list)
    override_seen: bool = False
    last_question: str | None = None
    # Top few (attribute, expected information gain in bits) for this turn,
    # so a replay can show why one question beat the others.
    question_scores: list[tuple[str, float]] = field(default_factory=list)
    turns: int = 0
