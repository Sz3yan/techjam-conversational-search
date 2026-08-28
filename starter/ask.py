"""Clarification strategy: what to ask, and how to phrase it.

Only `ask_attribute` is machine-read by the simulator; `message` is for the
human on the other side. Both are produced here so the natural language and the
structured field can never drift apart.
"""

from __future__ import annotations

from starter.state import ATTRIBUTES, SessionState

# Ordered by how much a typical answer narrows an apparel catalog.
PRIORITY = ("material", "style", "use_case", "color", "size", "budget", "feature", "brand", "category")

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
    """Pick the most informative attribute still worth asking about.

    Open-ended first: an unscoped question ("anything else that matters?")
    surfaces whatever the customer considers important instead of forcing them
    through our taxonomy. Only once that dries up do we probe specific slots.
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
    """Customer-facing text for this turn."""
    question = PROMPTS.get(attribute, PROMPTS["other"])
    if not state.constraints:
        return f"{OPENING} {question}"
    if count == 0:
        return f"I couldn't find a close match yet. {question}"
    known = ", ".join(constraint["text"] for constraint in state.constraints[-2:])
    return f"Going on {known}, here are the closest matches. {question}"
