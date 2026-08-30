"""Customer-side stressors for robustness measurement.

The public simulator speaks in fixed phrasings and the specification reserves
the right to paraphrase the private split. These stressors rewrite what the
customer says while preserving exactly the information it conveys, so any drop
in score is attributable to language handling and not to a harder problem.

Three levels:

* `clean`      -- the simulator verbatim.
* `paraphrase` -- natural rewording; a shopper who does not talk like a form.
* `hostile`    -- every marker string the parser keys on is deliberately
                  avoided: no "looking for", no "what matters is:", no
                  semicolon lists, no "actually", no "I don't have a
                  preference". This is the worst case, not the likely one.
* `starved`    -- clean phrasing, but every second disclosure is withheld,
                  simulating an extractor that silently loses half its input.

They patch the evaluator's own customer functions rather than reimplementing
the protocol, so the session dynamics stay exactly as scored.
"""

from __future__ import annotations

import evaluator.local_evaluator as ev

ALLOWED = ev.ALLOWED_ATTRIBUTES


def _constraints_for(sample, attribute, disclosed):
    constraints = [
        *[str(v) for v in sample["intent_card"].get("hard_constraints", [])],
        *[str(v) for v in sample["intent_card"].get("soft_preferences", [])],
    ]
    return [
        value for value in constraints
        if value not in disclosed
        and (attribute == "other" or ev.classify_constraint(value) == attribute)
    ][:2]


def make_paraphrase(level: str):
    """Return (initial_message, customer_reply) replacements for `level`."""

    def initial_paraphrase(sample, category, disclosed):
        scenario = sample["scenario_type"]
        if scenario == "buying" and sample["intent_card"].get("hard_constraints"):
            constraint = str(sample["intent_card"]["hard_constraints"][0])
            disclosed.add(constraint)
            if level == "hostile":
                return f"{category} is the space I'm in. One thing is non-negotiable, {constraint}"
            return f"I could use {category}. It really has to be {constraint}, that part is firm."
        if scenario == "intent_override":
            old = str(sample["behavior"]["override"]["old_value"])
            if level == "hostile":
                return f"{category} is roughly the area. {old}"
            return f"I could use {category}. {old}"
        if level == "hostile":
            return f"{category} is roughly the area, no fixed plan on my end."
        return f"I could use {category}, though nothing is settled yet."

    def reply_paraphrase(sample, ask_attribute, disclosed, boundary_used):
        attribute = ask_attribute if isinstance(ask_attribute, str) else None
        if sample["scenario_type"] == "boundary" and not boundary_used and attribute:
            if level == "hostile":
                return f"{attribute.title()}, your call entirely.", True
            return f"Honestly {attribute} is up to you, I really don't mind.", True
        if not attribute:
            if level == "hostile":
                return "Still not it. Pick one thing and put it to me.", boundary_used
            return "Those aren't landing yet. Ask me about one thing in particular.", boundary_used
        if attribute not in ALLOWED:
            attribute = "other"
        matches = _constraints_for(sample, attribute, disclosed)
        if not matches:
            if level == "hostile":
                return f"Nothing more on {attribute} from me.", boundary_used
            return f"There's nothing else I can tell you about {attribute}.", boundary_used
        disclosed.update(matches)
        if level == "hostile":
            return ", plus ".join(matches), boundary_used
        joined = " and also ".join(matches)
        return f"Well, {joined}.", boundary_used

    return initial_paraphrase, reply_paraphrase


def make_starved():
    """Clean phrasing, half the disclosures silently dropped."""
    original_reply = ev.customer_reply
    state = {"count": 0}

    def reply(sample, ask_attribute, disclosed, boundary_used):
        text, used = original_reply(sample, ask_attribute, disclosed, boundary_used)
        state["count"] += 1
        if state["count"] % 2 == 0 and "what matters is" in text:
            head = text.split("what matters is: ", 1)[1].rstrip(".")
            parts = head.split("; ")
            if len(parts) > 1:
                return "For that, what matters is: " + parts[0] + ".", used
        return text, used

    return ev.initial_message, reply


class Stressor:
    """Context manager that swaps the simulator's customer functions."""

    def __init__(self, level: str) -> None:
        self.level = level
        self._saved = None

    def __enter__(self):
        self._saved = (ev.initial_message, ev.customer_reply)
        if self.level == "clean":
            return self
        if self.level == "starved":
            initial, reply = make_starved()
        else:
            initial, reply = make_paraphrase(self.level)
        ev.initial_message = initial
        ev.customer_reply = reply
        return self

    def __exit__(self, *exc):
        ev.initial_message, ev.customer_reply = self._saved
        return False
