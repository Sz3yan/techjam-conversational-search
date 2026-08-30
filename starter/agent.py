"""Socratic -- a conversational shopping agent.

Twenty Questions against a 50,000-item catalogue, except there are only ten
turns. Each turn runs:

    observe -> update belief -> re-orchestrate -> ask (always) + answer (maybe)

The agent keeps a posterior over every product in the catalogue rather than a
ranked shortlist. That single choice is what lets the two interesting
decisions -- which attribute to ask about, and whether it is yet worth
recommending anything -- be made by maximising expected score instead of by a
hand-tuned schedule. See `socratic/policy.py`.

Nothing here imports the evaluator, and the default configuration runs on the
Python standard library alone. A local model is used when one is reachable and
the agent is score-complete without it; see `socratic/llm.py`.
"""

from __future__ import annotations

from pathlib import Path

from socratic.belief import Belief
from socratic.config import AgentConfig
from socratic.dialogue import parse
from socratic.index import CatalogIndex
from socratic.policy import ALLOWED_ATTRIBUTES, Policy
from socratic.session import Session
from socratic.text import classify

# Hand-authored fallback ordering, used only when question_policy == "ladder".
LADDER = (
    "other", "material", "style", "use_case", "color",
    "size", "budget", "feature", "brand", "category",
)


class Agent:
    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        config: AgentConfig | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.index = CatalogIndex.load(catalog_path)
        self.policy = Policy(self.index, depth=self.config.depth, candidates=self.config.candidates)
        self.llm = None
        self.dense = None
        if self.config.llm_mode != "off":
            from socratic.llm import LocalModel
            self.llm = LocalModel(self.config)
        if "dense" in self.config.channels and self.config.llm_mode != "off":
            from socratic.dense import DenseChannel
            candidate = DenseChannel(self.index, self.config.embed_model)
            self.dense = candidate if candidate.available() else None
        self.sessions: dict[str, Session] = {}
        # `usage` in the response is PER TURN -- the evaluator adds it up across
        # turns itself, so reporting a running total would compound quadratically.
        # `total_usage` is our own bookkeeping for the feasibility disclosure.
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self._turn_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    # ----------------------------------------------------------------- API

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = Session(
            belief=Belief(
                self.index,
                channels=self.config.channels,
                dense=self.dense,
                dense_weight=self.config.dense_weight,
                dense_offset=self.config.dense_offset,
                alpha=self.config.alpha,
                w_clause=self.config.w_clause,
            ),
            profile=user_profile if isinstance(user_profile, dict) else {},
        )

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        self._turn_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        try:
            return self._respond(session_id, user_message, turn, top_k)
        except Exception:
            # A scored run must never lose a session to an exception.
            return {
                "message": "Let me keep looking. Could you tell me one more thing you need?",
                "ask_attribute": "other",
                "recommendations": [],
                "usage": dict(self._turn_usage),
            }

    # ------------------------------------------------------------ internals

    def _respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        session = self.sessions.get(session_id)
        if session is None:
            self.reset(session_id, {})
            session = self.sessions[session_id]
        session.turns = turn

        utterance = self._observe(session, user_message, turn)
        self._route(session, utterance, turn)

        head, probabilities, _ = session.belief.posterior(top=self.config.candidates)
        tail = max(0.0, 1.0 - sum(probabilities))

        attribute, groups = self._choose_question(session, head, probabilities)
        width = self._choose_width(session, probabilities, tail, turn, groups, top_k)

        chosen = head[:width]
        chosen = self._rerank(session, head, probabilities, chosen)
        if chosen and self.config.eliminate_shown:
            session.shown.extend(chosen)
            session.belief.rule_out(chosen)
        if attribute:
            session.asked.add(attribute)
        session.last_question = attribute

        return {
            "message": self._message(session, attribute, chosen),
            "ask_attribute": attribute,
            "recommendations": [{"parent_asin": self.index.asins[pid]} for pid in chosen],
            "usage": dict(self._turn_usage),
        }

    def _rerank(self, session, head, probabilities, chosen: list[int]) -> list[int]:
        """Semantic rerank, fired only where the posterior is undecided.

        When the top candidates differ in probability by a clear margin the
        arithmetic is already more reliable than a 2B model's opinion, so the
        call is skipped. It is worth making exactly when the evidence does not
        separate the leaders -- which is also when it is cheap, because that
        situation is rare.
        """
        if self.llm is None or "rerank" not in self.config.llm_uses:
            return chosen
        if not chosen or len(head) < 2 or len(probabilities) < 2:
            return chosen
        if probabilities[1] <= 0.0:
            return chosen
        if probabilities[0] < self.config.rerank_min_prob:
            return chosen
        if probabilities[0] / probabilities[1] >= self.config.rerank_ratio:
            return chosen
        pool = head[: min(self.config.rerank_pool, len(head))]
        candidates = [(pid, self.index.titles[pid]) for pid in pool]
        pick = self.llm.rerank(session.belief.disclosed, candidates)
        self._account(self.llm.take_usage())
        if pick is None or pick == chosen[0]:
            return chosen
        reordered = [pick] + [pid for pid in chosen if pid != pick]
        return reordered[: len(chosen)]

    def _observe(self, session: Session, user_message: str, turn: int):
        utterance = parse(user_message or "", turn)
        if self.llm is not None and "segment" in self.config.llm_uses:
            extra = self.llm.segment(user_message or "", utterance)
            for clause in extra:
                if clause not in utterance.constraints:
                    utterance.constraints.append(clause)
            self._account(self.llm.take_usage())

        if utterance.is_override and not session.override_seen:
            session.override_seen = True
            self._apply_override(session, utterance)

        if utterance.category and session.category is None:
            session.category = utterance.category
            session.belief.add(utterance.category, turn, kind="category")

        for clause in utterance.constraints:
            session.belief.add(clause, turn)

        if utterance.refused_attribute:
            session.refused.add(utterance.refused_attribute)
        return utterance

    def _apply_override(self, session: Session, utterance) -> None:
        """The customer replaced a stated preference.

        The preference changed; the product they are shopping for did not.
        Everything disclosed earlier is still true of it, so the default is to
        erase only the slot the new statement contradicts and keep the rest.
        `override_mode` makes the alternatives measurable.
        """
        mode = self.config.override_mode
        # Anything shown before the override could not have converted -- the
        # evaluator will not accept a hit until the new intent lands -- so its
        # elimination was never real evidence. Put those products back.
        session.belief.readmit()
        session.shown.clear()
        if mode == "keep":
            return
        if mode == "erase":
            session.belief.retract(lambda item: item.kind == "constraint")
            return
        contradicted = {classify(c) for c in utterance.constraints} or {"feature"}
        session.belief.retract(
            lambda item: item.kind == "constraint"
            and item.turn < session.turns
            and item.bucket in contradicted
        )

    def _route(self, session: Session, utterance, turn: int) -> None:
        """Buying discloses a hard constraint up front; Browsing opens vague.

        The route is inferred, never supplied -- `scenario_type` is not part of
        the agent's input. It widens the question policy's candidate head for
        Browsing, where the opening turn carries almost no signal.
        """
        if turn == 1:
            session.route = "buying" if utterance.constraints else "browsing"

    def _choose_question(self, session: Session, head, probabilities):
        if self.config.question_policy == "none":
            return None, None
        disclosed = frozenset(session.belief.disclosed)
        session.question_scores = []
        if self.config.question_policy == "ladder":
            for attribute in LADDER:
                if attribute not in session.asked and attribute not in session.refused:
                    return attribute, None
            return "other", None
        ranked = self.policy.rank_questions(
            head, probabilities, disclosed,
            frozenset(session.asked), frozenset(session.refused),
        )
        if not ranked:
            return "other", None
        session.question_scores = [(name, gain) for name, gain, _ in ranked[:3]]
        attribute, _gain, groups = ranked[0]
        return attribute, groups

    def _choose_width(self, session, probabilities, tail, turn, groups, top_k) -> int:
        limit = max(0, min(int(top_k or 10), 10))
        if self.config.width_policy == "fixed":
            return min(self.config.fixed_width, limit, len(probabilities))
        if self.config.width_policy == "schedule":
            width = 1 if turn <= 5 else (3 if turn <= 7 else 10)
            return min(width, limit, len(probabilities))
        width, _value = self.policy.choose_width(probabilities, tail, turn, groups)
        return min(width, limit, len(probabilities))

    def _message(self, session: Session, attribute: str | None, chosen: list[int]) -> str:
        if self.llm is not None and "message" in self.config.llm_uses:
            text = self.llm.message(session, attribute, chosen, self.index)
            self._account(self.llm.take_usage())
            if text:
                return text
        return self._template(session, attribute, chosen)

    @staticmethod
    def _short(clause: str, limit: int = 44) -> str:
        """Catalogue clauses run to 180 characters; a sentence should not."""
        clause = clause.strip()
        return clause if len(clause) <= limit else clause[:limit].rsplit(" ", 1)[0] + "..."

    def _template(self, session: Session, attribute: str | None, chosen: list[int]) -> str:
        disclosed = [self._short(c) for c in session.belief.disclosed[-2:]]
        if chosen:
            title = self.index.titles[chosen[0]]
            lead = f"Based on {', '.join(disclosed)}, " if disclosed else ""
            body = f"{lead}the closest match I have is {self._short(title, 70)}"
            body += "."
            if len(chosen) > 1:
                body += f" I've listed {len(chosen)} options in order."
        else:
            body = (
                "I don't want to guess from too wide a field yet -- "
                "one more detail will narrow this down a lot."
            )
        if attribute:
            body += f" {QUESTION.get(attribute, QUESTION['other'])}"
        return body

    def _account(self, usage) -> None:
        if not usage:
            return
        for field in ("prompt_tokens", "completion_tokens"):
            value = int(usage.get(field, 0) or 0)
            self._turn_usage[field] += value
            self.total_usage[field] += value


QUESTION = {
    "category": "What kind of item are you after exactly?",
    "material": "Is there a material or fabric you want?",
    "color": "Any colour you have in mind?",
    "size": "What size or fit should I be matching?",
    "style": "What style or cut are you going for?",
    "brand": "Any brand you prefer?",
    "budget": "Roughly what budget are you working with?",
    "feature": "Is there a specific feature it has to have?",
    "use_case": "What will you be using it for?",
    "other": "What else matters most for this one?",
}
