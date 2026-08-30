"""Turning what the customer said into evidence.

The public simulator emits fixed phrasings, and it is easy to key on them
exactly -- and brittle, because the specification reserves the right to
paraphrase the private split. So the marker patterns here are *hints that
improve segmentation*, never requirements: every utterance also goes through a
general path that strips conversational framing and splits what is left into
clauses. If the markers vanish the agent degrades, it does not fall over.

The one thing worth getting right is clause granularity. "For that, what
matters is: A; B" carries two independent constraints, and merging them into
one blob dilutes both -- a rare, decisive phrase ends up averaged with filler.
Segmenting is what keeps each constraint separately weighable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import normalize

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Framing that introduces a disclosure. Used to find where the content starts.
FRAMING = re.compile(
    r"^.{0,60}?\b(?:requirement|matters|need|want|looking for|prefer|prefers|"
    r"preference|must|should|has to|are|is|it'?s|i'?d like)\b[^:]{0,40}?[::]\s*",
    re.IGNORECASE,
)
LEAD_IN = re.compile(
    r"^\s*(?:i'?m|i am|i'?ve been)?\s*(?:looking for|shopping for|searching for|"
    r"after|browsing for|in the market for|trying to find|show me|find me|"
    r"give me|recommend|suggest)\s+",
    re.IGNORECASE,
)
TRAILING_HEDGE = re.compile(
    r"\b(?:but|though|although)?\s*(?:i'?m|i am)?\s*(?:still\s+)?"
    r"(?:just\s+)?(?:exploring|browsing|looking around|not sure|undecided)\b.*$",
    re.IGNORECASE,
)
REFUSAL = re.compile(
    r"\b(?:do\s?n[o']?t|don'?t|no)\b[^.]{0,40}?\bpreference\b|"
    r"\bno\s+(?:strong\s+)?(?:preference|opinion|view)\b|"
    r"\buse your judg(?:e)?ment\b|\bup to you\b|\bwhatever you (?:think|recommend)\b",
    re.IGNORECASE,
)
ADDITIONAL = re.compile(r"\badditional\b|\banything (?:else|more)\b|\bnothing (?:else|more)\b", re.IGNORECASE)
OVERRIDE = re.compile(
    r"\b(?:actually|instead|on second thought|scratch that|forget|ignore|"
    r"changed my mind|rather than|no longer|not anymore)\b",
    re.IGNORECASE,
)
STALL = re.compile(
    r"\bnot (?:quite )?right\b|\bnone of (?:these|those)\b|\bnot what i\b|"
    r"\btry again\b|\bask me about\b|\bkeep looking\b",
    re.IGNORECASE,
)
ATTRIBUTE_MENTION = re.compile(
    r"\b(category|material|colou?r|size|style|brand|budget|feature|use case|use_case)\b",
    re.IGNORECASE,
)
# Phrases that are conversation, not product attributes.
BOILERPLATE = re.compile(
    r"^(?:please|thanks?|thank you|ok(?:ay)?|sure|yes|no|hi|hello|hmm+|well)\b[\s,]*",
    re.IGNORECASE,
)
INFORMATIVE = re.compile(r"[a-z0-9]", re.IGNORECASE)
# Talk *about* the conversation rather than about a product. An override
# announcement carries no attribute of its own and must not become evidence.
META_ONLY = re.compile(
    r"^(?:actually|instead|well|so|ok(?:ay)?|right)?[\s,]*"
    r"(?:please\s+)?(?:just\s+)?(?:ignore|forget|drop|scratch|disregard|skip)?\s*"
    r"(?:that|this|it|my|the)?\s*(?:earlier|previous|first|last|original|old)?\s*"
    r"(?:preference|requirement|request|answer|note|point|comment|thing|one)?s?[\s.,!]*$",
    re.IGNORECASE,
)
LONG_FRAGMENT = 90


@dataclass
class Utterance:
    """What one customer turn told us."""

    category: str | None = None
    constraints: list[str] = field(default_factory=list)
    refused_attribute: str | None = None
    is_override: bool = False
    is_stall: bool = False
    exhausted: bool = False       # "no *additional* preference for X"
    marked: bool = False          # a known framing marker was found


def _attribute_in(text: str) -> str | None:
    match = ATTRIBUTE_MENTION.search(text)
    if not match:
        return None
    value = match.group(1).lower().replace(" ", "_")
    return "color" if value in ("colour", "color") else value


def _split_clauses(fragment: str) -> list[str]:
    """Break one disclosure into independently weighable constraints."""
    parts = [p for p in re.split(r"\s*;\s*", fragment) if p.strip()]
    out: list[str] = []
    for part in parts:
        # Only break on conjunctions when nothing stronger was available and the
        # fragment is long enough that merging would genuinely dilute it.
        if len(parts) == 1 and len(part) > LONG_FRAGMENT:
            pieces = re.split(r"\s+and\s+|\s*\|\s*|\s*•\s*", part)
            out.extend(p for p in pieces if p.strip())
        else:
            out.append(part)
    cleaned: list[str] = []
    for item in out:
        item = BOILERPLATE.sub("", item).strip()
        item = normalize(item)
        if item and INFORMATIVE.search(item):
            cleaned.append(item)
    return cleaned


def parse(message: str, turn: int) -> Utterance:
    result = Utterance()
    if not message or not message.strip():
        return result
    sentences = [s for s in SENTENCE_SPLIT.split(message.strip()) if s.strip()]
    for sentence in sentences:
        stripped = sentence.strip()
        if OVERRIDE.search(stripped):
            result.is_override = True
        if REFUSAL.search(stripped):
            result.refused_attribute = _attribute_in(stripped) or result.refused_attribute
            result.exhausted = bool(ADDITIONAL.search(stripped))
            # A refusal can still carry content after a marker ("no preference
            # on colour, but it must be waterproof"), so keep parsing it.
        if STALL.search(stripped):
            result.is_stall = True
            continue

        lead = LEAD_IN.search(stripped)
        if lead:
            result.marked = True
        if lead and result.category is None:
            remainder = stripped[lead.end():]
            remainder = TRAILING_HEDGE.sub("", remainder).strip(" .,")
            head = re.split(r"[.;]", remainder, maxsplit=1)[0]
            head = re.sub(r"\s*,?\s*(?:but|and)\b.*$", "", head, flags=re.IGNORECASE)
            if head.strip():
                result.category = normalize(head)
            continue

        framing = FRAMING.match(stripped)
        if framing:
            result.marked = True
        body = stripped[framing.end():] if framing else stripped
        if not framing and REFUSAL.search(stripped) and ":" not in stripped:
            continue
        if not framing:
            # No marker survived. Drop conversational framing heuristically and
            # keep whatever content remains.
            body = re.sub(
                r"^(?:for that,?|about that,?|on that,?|as for that,?)\s*", "", body, flags=re.IGNORECASE
            )
            body = re.sub(
                r"^(?:what|the thing|the part|all)\s+(?:i\s+)?(?:matters|care about|need|want)"
                r"(?:\s+is)?[,:]?\s*", "", body, flags=re.IGNORECASE
            )
        if META_ONLY.match(body.strip()):
            continue
        for clause in _split_clauses(body):
            if clause not in result.constraints:
                result.constraints.append(clause)
    return result
