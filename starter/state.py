"""Per-session conversation state: slots, disclosures, and turn bookkeeping.

The evaluator's simulated customer speaks in a small number of templates, so
constraint extraction is marker-driven with a free-text fallback. The fallback
matters: the organizer reserves the right to paraphrase the private sessions,
and an agent that only understands the public templates would collapse there.
"""

from __future__ import annotations

import re

from starter.retrieve import tokenize

# The attribute vocabulary the simulator accepts in `ask_attribute`.
ATTRIBUTES = (
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
)

MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange")

CATEGORY_RE = re.compile(r"looking for\s+(.+?)(?:[.,]|$)", re.I)
NO_ADDITIONAL_RE = re.compile(r"don't have an additional preference for\s+(\w+)", re.I)
NO_PREFERENCE_RE = re.compile(r"don't have a preference for\s+(\w+)", re.I)
OVERRIDE_RE = re.compile(r"ignore my earlier preference", re.I)
STALLED_RE = re.compile(r"not quite right yet", re.I)

# Sentence scaffolding left behind once the category clause is removed. Harmless
# for retrieval (these tokens are stopword-filtered) but it leaks into the
# customer-facing message, so strip it at the source.
LEADING_FILLER_RE = re.compile(
    r"^(?:i'?m|i am|i|and|also|plus|well|hmm|so|but|it'?s|the thing is|need|needs)\b[\s,.:;-]*",
    re.I,
)

# Phrases that introduce a disclosed constraint. Order matters only for stripping.
CONSTRAINT_MARKERS = (
    "A key requirement is:",
    "For that, what matters is:",
    "What I need is:",
)

# Removed before the free-text fallback treats a message as new information.
BOILERPLATE = (
    "but I'm still exploring",
    "please use your judgment",
    "Actually, ignore my earlier preference.",
    "Ask me about one specific attribute.",
    "Those options are not quite right yet.",
)


def classify(constraint: str) -> str:
    """Map a disclosed constraint onto the ask_attribute vocabulary."""
    lowered = constraint.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if "color" in lowered or any(color in lowered for color in COLORS):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


class SessionState:
    """Accumulated knowledge for one session."""

    def __init__(self, session_id: str, user_profile: dict | None = None) -> None:
        self.session_id = session_id
        self.profile = user_profile or {}
        self.history: list[str] = []
        self.constraints: list[dict] = []      # {"text", "turn", "attribute"}
        self.category: str | None = None
        self.asked: list[str] = []
        self.exhausted: set[str] = set()       # attribute answered "no additional preference"
        self.no_preference: set[str] = set()   # boundary-style refusal
        self.shown: list[tuple[int, list[str]]] = []
        self.override_turn: int | None = None
        self.stalled_turns = 0

    # ------------------------------------------------------------------ input

    def observe(self, message: str, turn: int) -> None:
        """Fold one customer utterance into the state."""
        text = (message or "").strip()
        self.history.append(text)
        if not text:
            return

        if OVERRIDE_RE.search(text):
            # The customer replaced a preference. Note it, but keep the prior
            # slots: the simulator's override does not change the target
            # product, so earlier disclosures remain true of it. Measured:
            # wiping state drops intent_override hit rate 0.867 -> 0.500.
            self.override_turn = turn

        if STALLED_RE.search(text):
            self.stalled_turns += 1

        match = NO_ADDITIONAL_RE.search(text)
        if match:
            self.exhausted.add(match.group(1).lower())
        else:
            match = NO_PREFERENCE_RE.search(text)
            if match:
                self.no_preference.add(match.group(1).lower())

        if self.category is None:
            match = CATEGORY_RE.search(text)
            if match:
                self.category = match.group(1).strip()

        for constraint in self._extract(text):
            self._add_constraint(constraint, turn)

    def _extract(self, text: str) -> list[str]:
        """Pull disclosed constraints out of an utterance."""
        found: list[str] = []
        for marker in CONSTRAINT_MARKERS:
            if marker in text:
                tail = text.split(marker, 1)[1]
                found.extend(part.strip(" .;") for part in tail.split(";"))
        if found:
            return [item for item in found if item]

        # Paraphrase fallback: strip everything we recognise as boilerplate and
        # treat any remaining substance as a disclosure.
        residue = text
        for phrase in BOILERPLATE:
            residue = residue.replace(phrase, " ")
        residue = CATEGORY_RE.sub(" ", residue, count=1)
        if NO_ADDITIONAL_RE.search(text) or NO_PREFERENCE_RE.search(text):
            return []
        residue = residue.strip(" .,;")
        return [residue] if len(residue) > 8 else []

    @staticmethod
    def _tidy(text: str) -> str:
        """Collapse whitespace and shave off leading sentence scaffolding."""
        cleaned = re.sub(r"\s+", " ", text).strip(" .,;:-")
        while True:
            trimmed = LEADING_FILLER_RE.sub("", cleaned).strip(" .,;:-")
            if trimmed == cleaned:
                return cleaned
            cleaned = trimmed

    def _add_constraint(self, text: str, turn: int) -> None:
        cleaned = self._tidy(text)
        if not cleaned or len(cleaned) < 3:
            return
        if any(existing["text"].lower() == cleaned.lower() for existing in self.constraints):
            return
        self.constraints.append({"text": cleaned, "turn": turn, "attribute": classify(cleaned)})

    # ----------------------------------------------------------------- output

    def query_terms(self) -> list[str]:
        """Terms for sparse retrieval, most informative first.

        Category and disclosed constraints lead; raw history backfills so a
        paraphrased session still produces a usable query.
        """
        terms: list[str] = []
        if self.category:
            terms.extend(tokenize(self.category))
        for constraint in self.constraints:
            terms.extend(tokenize(constraint["text"]))
        for message in self.history:
            terms.extend(tokenize(message))
        return list(dict.fromkeys(terms))

    def record_shown(self, parent_asins: list[str], turn: int) -> None:
        self.shown.append((turn, list(parent_asins)))

    def ruled_out(self) -> set[str]:
        """Products provably not the target.

        Anything scored in a previous turn without ending the session cannot be
        the target. The exception is an intent-override session before the
        override lands, where the evaluator refuses to convert at all; those
        turns are re-admitted once we learn the override turn.
        """
        floor = self.override_turn or 0
        return {
            parent_asin
            for turn, batch in self.shown if turn >= floor
            for parent_asin in batch
        }
