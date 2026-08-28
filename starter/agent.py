"""Conversational shopping agent.

Pipeline per turn:

    observe -> route -> sparse retrieve -> rerank -> answer + clarify

Runs entirely on the Python standard library: the catalog lives in an in-memory
SQLite FTS5 index, and ranking is IDF-weighted constraint matching. No network
access, no credentials, no model download.
"""

from __future__ import annotations

from pathlib import Path

from starter.ask import compose, next_attribute
from starter.dense import DenseIndex
from starter.rerank import rerank
from starter.retrieve import CatalogIndex
from starter.state import SessionState

# Candidate pool handed to the reranker. Browsing sessions start vague, so they
# need a wider net; buying sessions arrive with a hard constraint and do better
# with a tight, high-precision pool.
POOL_BUYING = 100
POOL_BROWSING = 200

BROWSING_HINTS = ("still exploring", "not sure", "just looking", "browsing", "ideas")

# How many products to actually put in front of the customer.
#
# A session ends the moment the target appears in the scored list, which makes a
# premature wide guess expensive: surfacing the right product at rank 9 banks a
# reciprocal rank of 0.11 and forfeits every later chance to present it first.
# So we commit to a single confident pick while turns remain to learn more, then
# widen as the deadline approaches so a near-miss still lands in the window.
NARROW_UNTIL = 5   # turns 1-5: one best guess
SHORTLIST_UNTIL = 7  # turns 6-7: short list
SHORTLIST_WIDTH = 3


class Agent:
    """Required competition interface."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.index = CatalogIndex(catalog_path)
        # Optional. Absent cached embeddings or numpy, this stays disabled and
        # the agent runs as a pure-stdlib sparse system.
        self.dense = DenseIndex(catalog_path)
        self.sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = SessionState(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        # Never raise: the evaluator scores an exception as a lost session.
        state = self.sessions.get(session_id)
        if state is None:
            state = self.sessions[session_id] = SessionState(session_id, {})

        state.observe(user_message, turn)

        pool_size = POOL_BUYING if self._is_buying(state) else POOL_BROWSING
        candidates = self.index.search(state.query_terms(), pool_size)
        dense_scores = None
        if self.dense.available:
            dense_scores = self.dense.similarity(self._semantic_query(state), candidates)
        ordered = rerank(self.index, state, candidates, dense_scores=dense_scores)
        picks = ordered[:self._width(turn, top_k)]

        state.record_shown(picks, turn)
        attribute = next_attribute(state)
        state.asked.append(attribute)

        return {
            "message": compose(state, attribute, len(picks)),
            "ask_attribute": attribute,
            "recommendations": [{"parent_asin": parent_asin} for parent_asin in picks],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    @staticmethod
    def _semantic_query(state: SessionState) -> str:
        """Natural-language rendering of the session for the encoder."""
        parts = [state.category or ""]
        parts.extend(constraint["text"] for constraint in state.constraints)
        return ", ".join(part for part in parts if part)

    @staticmethod
    def _width(turn: int, top_k: int) -> int:
        """Recommendation count for this turn: narrow early, wide at the end."""
        if turn <= NARROW_UNTIL:
            return 1
        if turn <= SHORTLIST_UNTIL:
            return min(SHORTLIST_WIDTH, top_k)
        return top_k

    def _is_buying(self, state: SessionState) -> bool:
        """Buying discloses a hard constraint up front; browsing starts vague."""
        opening = state.history[0].lower() if state.history else ""
        if any(hint in opening for hint in BROWSING_HINTS):
            return False
        return bool(state.constraints)
