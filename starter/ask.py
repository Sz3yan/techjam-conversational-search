"""Decides what to ask the customer next, and writes the sentence that asks it.

Every turn the agent sends back two related things: a question in plain English
for the person reading it, and an attribute name in the `ask_attribute` field
for the evaluator. Only the attribute name affects the score, because that is
what the evaluator reads to decide which requirement to reveal next. The
sentence is what a human judge sees during a demo, and what makes the exchange
look like a conversation rather than a search box.

Both are written here, in the same place, so that they can never drift apart
and leave us asking about colour in words while asking about material in the
structured field.
"""

from __future__ import annotations

from starter.state import ATTRIBUTES, SessionState

# The order we work through attributes in, running roughly from the ones whose
# answer narrows a clothing catalog the most down to the ones that narrow it
# the least.
PRIORITY = ("material", "style", "use_case", "color", "size", "budget", "feature", "brand", "category")

# The sentence we use for each attribute.
PROMPTS = {
    "material": "What material are you hoping for?",
    "style": "What style or fit works best for you?",
    "use_case": "What will you mainly be using it for?",
    "color": "Any colour you're set on?",
    "size": "What size or fit should I be matching?",
    "budget": "Roughly what budget are you working with?",
    "feature": "Is there a specific feature it needs to have?",
    "brand": "Any brand you lean towards?",
    "category": "Which kind of item are we narrowing to?",
    "other": "Anything else that matters for this one?",
}

OPENING = "Tell me a bit more and I'll narrow this down."


def next_attribute(state: SessionState) -> str:
    """Choose which attribute to ask the customer about this turn.

    We open with "other", which is the unscoped question -- "anything else that
    matters?" -- for two reasons. It lets the customer volunteer whatever they
    think is important, instead of marching them through our list of
    categories. And because it is unscoped, the evaluator will hand over any
    requirement it still has, whatever kind it is, so it is the question least
    likely to come back empty.

    Once that runs dry we walk down the priority list, skipping anything they
    have declined to answer, anything we have already asked, and anything they
    have already told us about.

    The two loops after that are fallbacks for a long session where the good
    options are used up. The first relaxes the rule about attributes we already
    know, and the second relaxes the rule about ones we already asked. Asking a
    repeat question is a poor turn, but it beats asking nothing at all, because
    a turn with no question gets us no new information whatsoever.
    """
    unavailable = state.exhausted | state.no_preference
    if "other" not in unavailable:
        return "other"

    covered = {constraint["attribute"] for constraint in state.constraints}
    for attribute in PRIORITY:
        if attribute in unavailable or attribute in state.asked or attribute in covered:
            continue
        return attribute
    for attribute in PRIORITY:
        if attribute not in unavailable and attribute not in state.asked:
            return attribute
    for attribute in ATTRIBUTES:
        if attribute not in unavailable:
            return attribute
    return "other"


def compose(state: SessionState, attribute: str, count: int) -> str:
    """Write the sentence the customer actually reads this turn.

    There are three situations. We might know nothing about them yet, we might
    know something but have found nothing worth showing, or we might have both.

    Whichever it is, the sentence always ends with the question belonging to
    `attribute`, which is how the words and the `ask_attribute` field stay in
    agreement. When we do have requirements we also read the last two back to
    them, so they can see what we understood and correct us if we got it wrong.
    """
    question = PROMPTS.get(attribute, PROMPTS["other"])
    if not state.constraints:
        return f"{OPENING} {question}"
    if count == 0:
        return f"I couldn't find a close match yet. {question}"
    known = ", ".join(constraint["text"] for constraint in state.constraints[-2:])
    return f"Going on {known}, here are the closest matches. {question}"
