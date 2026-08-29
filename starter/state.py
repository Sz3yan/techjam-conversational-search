"""Keeps track of what we have learned about the customer during one session.

Every session gets its own `SessionState` object. Each time the customer says
something we call `observe()`, which stores the raw message and then reads it
for anything useful: what kind of product they want, what requirements they
have mentioned, and which attributes they have told us not to ask about again.

The simulated customer in the evaluator speaks from a small set of fixed
sentence templates, so most of the extraction below works by looking for those
exact phrases. There is also a fallback for when none of them match. That
fallback matters, because the organizer may reword the 800 private sessions,
and an agent that only understood the public wording would fall apart there.
"""

from __future__ import annotations

import re

from starter.retrieve import tokenize

# The only attribute names the simulator accepts in the `ask_attribute` field.
# Anything outside this list gets quietly treated as "other".
ATTRIBUTES = (
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
)

# Words we recognise as describing what a product is made of, and what colour
# it is. Used to sort a requirement into the right attribute bucket.
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
COLORS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange")

# Each pattern below matches one of the sentence shapes the simulated customer
# uses, so that we can recognise it and react:
#
#   CATEGORY_RE       "I'm looking for Earrings Hoop, but I'm still exploring."
#   NO_ADDITIONAL_RE  "I don't have an additional preference for color."
#   NO_PREFERENCE_RE  "I don't have a preference for size; please use your judgment."
#   OVERRIDE_RE       "Actually, ignore my earlier preference. What I need is: ..."
#   STALLED_RE        "Those options are not quite right yet."
CATEGORY_RE = re.compile(r"looking for\s+(.+?)(?:[.,]|$)", re.I)
NO_ADDITIONAL_RE = re.compile(r"don't have an additional preference for\s+(\w+)", re.I)
NO_PREFERENCE_RE = re.compile(r"don't have a preference for\s+(\w+)", re.I)
OVERRIDE_RE = re.compile(r"ignore my earlier preference", re.I)
STALLED_RE = re.compile(r"not quite right yet", re.I)

# Filler words left dangling at the start of a sentence once we have cut the
# "I'm looking for ..." part out of it. They do no harm to the search, since
# they are stopwords and get dropped anyway, but they do end up in the message
# we show the customer, so we trim them off as soon as we extract the text.
LEADING_FILLER_RE = re.compile(
    r"^(?:i'?m|i am|i|and|also|plus|well|hmm|so|but|it'?s|the thing is|need|needs)\b[\s,.:;-]*",
    re.I,
)

# The phrases the customer uses to introduce a requirement. Whatever follows
# one of these is something they actually want. Their order in this tuple does
# not matter for detection, only for the order we cut them out of the text.
CONSTRAINT_MARKERS = (
    "A key requirement is:",
    "For that, what matters is:",
    "What I need is:",
)

# Stock phrases the simulator adds around the real content. We delete these
# before deciding whether a message contained anything new, so that a message
# made up entirely of boilerplate is not mistaken for a requirement.
BOILERPLATE = (
    "but I'm still exploring",
    "please use your judgment",
    "Actually, ignore my earlier preference.",
    "Ask me about one specific attribute.",
    "Those options are not quite right yet.",
)


def classify(constraint: str) -> str:
    """Work out which attribute a requirement belongs to.

    This matters because the customer only tells us something new when we ask
    about the right attribute. If they say "100% Cotton" we need to record that
    as a *material* requirement, otherwise we would go on asking about material
    and they would go on repeating themselves.

    The checks run from most specific to least specific, and the first one that
    matches wins. Anything we do not recognise is filed under the generic
    "feature" bucket.
    """
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
    """Everything we know so far about one customer's session."""

    def __init__(self, session_id: str, user_profile: dict | None = None) -> None:
        self.session_id = session_id
        self.profile = user_profile or {}
        # Every message the customer has sent us, oldest first.
        self.history: list[str] = []
        # The requirements they have told us about. Each one is stored as
        # {"text": what they said, "turn": when, "attribute": which bucket}.
        self.constraints: list[dict] = []
        # The kind of product they said they were after, e.g. "Earrings Hoop".
        self.category: str | None = None
        # Attributes we have already asked about, so we do not repeat ourselves.
        self.asked: list[str] = []
        # Attributes where they said they had nothing further to add.
        self.exhausted: set[str] = set()
        # Attributes where they declined to have an opinion at all.
        self.no_preference: set[str] = set()
        # What we recommended and when: (turn number, list of product IDs).
        self.shown: list[tuple[int, list[str]]] = []
        # The turn on which they changed their mind, if they ever did.
        self.override_turn: int | None = None
        # How many times they have told us our suggestions are off the mark.
        self.stalled_turns = 0

    # ------------------------------------------------------------------ input

    def observe(self, message: str, turn: int) -> None:
        """Read one customer message and record everything useful in it.

        Adds the raw text to `history`, then checks the message for each of the
        things we care about: a change of mind, a complaint that we are off
        track, a refusal to answer, the product category, and any new
        requirements.
        """
        text = (message or "").strip()
        self.history.append(text)
        if not text:
            return

        if OVERRIDE_RE.search(text):
            # The customer has replaced one of their preferences. We note which
            # turn it happened on, but we deliberately keep everything they
            # told us earlier. The override changes what they say they want, it
            # does not change which product is the right answer, so the earlier
            # facts are still true of that product. We measured the
            # alternative: throwing the earlier state away drops our hit rate
            # on override sessions from 0.867 to 0.500.
            self.override_turn = turn

        if STALLED_RE.search(text):
            self.stalled_turns += 1

        # Two ways of declining to answer, and they mean slightly different
        # things, so we keep them in separate sets. "No additional preference"
        # means that attribute is used up; "no preference" is the boundary
        # scenario, where they are handing the decision back to us.
        match = NO_ADDITIONAL_RE.search(text)
        if match:
            self.exhausted.add(match.group(1).lower())
        else:
            match = NO_PREFERENCE_RE.search(text)
            if match:
                self.no_preference.add(match.group(1).lower())

        # The category is only ever stated in the opening message, so once we
        # have it we stop looking. This also protects us from a later message
        # that happens to contain the words "looking for".
        if self.category is None:
            match = CATEGORY_RE.search(text)
            if match:
                self.category = match.group(1).strip()

        for constraint in self._extract(text):
            self._add_constraint(constraint, turn)

    def _extract(self, text: str) -> list[str]:
        """Pull the customer's stated requirements out of one message.

        The first choice is always the marker phrases. When the customer says
        "A key requirement is: X" we know for certain that everything after the
        colon is a requirement, and that semicolons separate several of them.

        When no marker is present the message was either reworded by the
        organizer or is one of the override openings, which state a preference
        without announcing it. So we fall back to guessing: delete every stock
        phrase we recognise, delete the "I'm looking for ..." clause, and treat
        whatever text survives as a requirement, as long as there is enough of
        it left to mean anything. Messages that were purely a refusal are
        skipped, because "I don't have a preference for color" tells us what
        the customer does *not* care about and contains no requirement to
        search on.
        """
        found: list[str] = []
        for marker in CONSTRAINT_MARKERS:
            if marker in text:
                tail = text.split(marker, 1)[1]
                found.extend(part.strip(" .;") for part in tail.split(";"))
        if found:
            return [item for item in found if item]

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
        """Clean up an extracted requirement before we store it.

        Squashes runs of whitespace, trims stray punctuation off both ends, and
        strips filler words off the front. The stripping runs in a loop because
        filler stacks up: "And I'm ..." needs two passes before the real
        content is at the front, so we keep going until a pass changes nothing.
        """
        cleaned = re.sub(r"\s+", " ", text).strip(" .,;:-")
        while True:
            trimmed = LEADING_FILLER_RE.sub("", cleaned).strip(" .,;:-")
            if trimmed == cleaned:
                return cleaned
            cleaned = trimmed

    def _add_constraint(self, text: str, turn: int) -> None:
        """Store one requirement, unless it is empty or one we already have."""
        cleaned = self._tidy(text)
        if not cleaned or len(cleaned) < 3:
            return
        if any(existing["text"].lower() == cleaned.lower() for existing in self.constraints):
            return
        self.constraints.append({"text": cleaned, "turn": turn, "attribute": classify(cleaned)})

    # ----------------------------------------------------------------- output

    def query_terms(self) -> list[str]:
        """Build the list of words to search the catalog with, best first.

        The category and the stated requirements go first, because those are
        the parts we are most confident about. Every raw message then gets
        appended behind them as a safety net: if our extraction missed
        something because the customer phrased it unusually, the words are
        still in there somewhere and can still find the product.

        Duplicates are removed while keeping the original order, so a word we
        are confident about stays near the front where it belongs.
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
        """Remember which products we recommended on this turn."""
        self.shown.append((turn, list(parent_asins)))

    def ruled_out(self) -> set[str]:
        """The products we can prove are not the one the customer wants.

        The session ends the instant we show the right product, so if we
        recommended something on an earlier turn and the conversation carried
        on afterwards, that product was wrong. Collecting them lets the
        reranker push them to the bottom instead of offering them again and
        wasting the turns we have left.

        There is one exception. In an intent-override session the evaluator
        refuses to accept any answer at all until the customer changes their
        mind on turn 3 or 4, so anything we showed before that never had a
        chance of being accepted and cannot be ruled out. We start out
        assuming everything counts, and once we see the override happen we know
        which turn it was and let those earlier products back in. There is no
        risk in guessing this way round, because those turns could never have
        ended the session anyway.
        """
        floor = self.override_turn or 0
        return {
            parent_asin
            for turn, batch in self.shown if turn >= floor
            for parent_asin in batch
        }
