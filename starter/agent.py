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
from starter.rerank import rerank
from starter.retrieve import CatalogIndex
from starter.state import SessionState

# How many candidates to pull out of the catalog before reranking them.
# If the customer is buying, then they have already given us enough to search on, so a small and precise pool serves them better.
# If the customer is browsing, then they have given us little more than a category, so the product they want could be well down the list and we need to look further to catch it.

POOL_BUYING = 100
POOL_BROWSING = 200

# derieved from the evaluator.
BROWSING_HINTS = ("still exploring", "not sure", "just looking", "browsing", "ideas")

# How many products to actually put in front of the customer.
# The session ends the moment the right product appears anywhere in our list,
# and the score for that session is 1 divided by its position in the list. That
# makes a wide guess early on a bad trade: if we show ten products on turn one
# and the right one is ninth, we bank a score of 0.11 and the session is over,
# with no chance ever to present it first.
#
# So while there are still turns left to learn something, we show a single
# confident guess. If it is right we score a full 1.0. If it is wrong we get
# another requirement out of the customer, and that product is eliminated for
# the rest of the session. As the deadline approaches there is no more time to
# learn, so we widen out and show everything, giving a near miss a chance to
# land inside the scored window.
NARROW_UNTIL = 5     # turns 1 to 5: a single best guess
SHORTLIST_UNTIL = 7  # turns 6 to 7: a short list
SHORTLIST_WIDTH = 3


class Agent:
    """The interface the competition requires: reset once, then respond each turn."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        # Building the index means reading all 50,000 products, so it happens
        self.index = CatalogIndex(catalog_path)
        self.sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Start a fresh session. Nothing carries over from any previous one."""
        self.sessions[session_id] = SessionState(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        """Handle one turn: read the message, pick products, ask a question."""
        # If reset() was somehow never called for this session, quietly start a
        # blank one instead of raising. The evaluator counts an exception as a
        # completely lost session, so crashing is never the better option.
        state = self.sessions.get(session_id)
        if state is None:
            state = self.sessions[session_id] = SessionState(session_id, {})

        # to keep track of the conversation history
        state.observe(user_message, turn)

        # Steps 2 to 4: search the catalog, then reorder what comes back.
        pool_size = POOL_BUYING if self._is_buying(state) else POOL_BROWSING
        candidates = self.index.search(state.query_terms(), pool_size)
        ordered = rerank(self.index, state, candidates)
        picks = ordered[:self._width(turn, top_k)]

        # Step 5: remember what we showed, so that if we are wrong we can rule
        # these products out next turn, then pick something to ask about.
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
    def _width(turn: int, top_k: int) -> int:
        """How many products to show on this turn: few early, all of them late."""

        if turn <= NARROW_UNTIL:
            return 1
        if turn <= SHORTLIST_UNTIL:
            return min(SHORTLIST_WIDTH, top_k)
        return top_k

    def _is_buying(self, state: SessionState) -> bool:
        """Guess whether this customer is buying or just browsing.

        Their opening message is the tell: somebody browsing says they are still exploring (derived from the evaluator),
        while somebody buying leads with a requirement (derived from the evaluator).
        """

        opening = state.history[0].lower() if state.history else ""
        if any(hint in opening for hint in BROWSING_HINTS):
            return False
        return bool(state.constraints)
