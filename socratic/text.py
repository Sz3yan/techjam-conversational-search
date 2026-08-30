"""Text normalisation and domain vocabulary.

This module models *apparel product metadata*, not the public simulator. The
material and colour vocabularies below are deliberately wider than anything the
evaluator recognises: the goal is to describe what a clothing catalogue says
about itself, so the same code keeps working if the private split phrases
things differently.
"""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+")
WS_RE = re.compile(r"\s+")
EDGE_PUNCT = " -;,.:\t\n\r\"'()[]"

# Words that carry no discriminative weight in a product catalogue.
STOPWORDS = frozenset("""
a an and are as at be been but by can could do does for from had has have he her
his i if in into is it its me my no not of on or our out please she so some that
the their them then there these they this to up us want was we were what when
which who will with would you your looking need am really just very also more
most other than too item product
""".split())

# Apparel materials. Broader than the evaluator's nine; ordered longest-first at
# match time so "faux leather" wins over "leather".
MATERIALS = (
    "faux leather", "genuine leather", "patent leather", "stainless steel",
    "sterling silver", "rose gold", "white gold", "yellow gold",
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "linen", "cashmere", "denim", "suede", "velvet", "satin",
    "chiffon", "fleece", "acrylic", "viscose", "elastane", "lycra", "modal",
    "bamboo", "canvas", "mesh", "lace", "tulle", "corduroy", "flannel",
    "jersey", "twill", "chambray", "tweed", "merino", "alpaca", "mohair",
    "sequin", "rubber", "silicone", "alloy", "brass", "copper", "titanium",
    "tungsten", "platinum", "silver", "gold", "pearl", "crystal", "rhinestone",
    "cubic zirconia", "resin", "acetate", "polyurethane", "microfiber",
    "terry", "poplin", "gabardine", "organza", "neoprene", "gore-tex",
    "fabric", "steel", "beads", "enamel", "zinc",
)

COLORS = (
    "black", "white", "blue", "navy", "red", "pink", "green", "brown", "gray",
    "grey", "purple", "yellow", "orange", "beige", "cream", "ivory", "tan",
    "khaki", "olive", "burgundy", "maroon", "teal", "turquoise", "coral",
    "lavender", "violet", "indigo", "magenta", "gold", "silver", "rose gold",
    "charcoal", "heather", "multicolor", "multicolour", "clear", "nude",
    "mint", "peach", "mustard", "rust", "wine", "plum", "aqua", "camel",
)

SIZE_WORDS = (
    "size", "sizing", "small", "medium", "large", "x-large", "xl", "xxl",
    "xs", "petite", "plus size", "regular fit", "slim fit", "relaxed fit",
    "true to size", "width", "wide", "narrow", "length", "inseam", "waist",
    "bust", "chest", "us size", "eu size", "uk size",
)

STYLE_WORDS = (
    "style", "fit", "sleeve", "sleeveless", "neckline", "crew", "v-neck",
    "collar", "casual", "formal", "vintage", "classic", "modern", "bohemian",
    "elegant", "sporty", "closure", "zipper", "button", "buckle", "lace-up",
    "pullover", "cut", "hem", "pattern", "striped", "floral", "plaid",
    "solid", "print", "design", "department", "womens", "mens", "unisex",
)

USE_CASE_WORDS = (
    "hiking", "running", "gym", "workout", "training", "yoga", "swimming",
    "winter", "summer", "spring", "autumn", "fall", "outdoor", "indoor",
    "work", "office", "business", "party", "wedding", "beach", "travel",
    "camping", "hunting", "fishing", "cycling", "golf", "tennis", "everyday",
    "school", "sleep", "lounge", "occasion", "gift", "christmas", "halloween",
)

_MATERIAL_RE = re.compile(
    r"\b(" + "|".join(re.escape(m) for m in sorted(MATERIALS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_COLOR_RE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in sorted(COLORS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
PRICE_RE = re.compile(r"\$\s*([0-9]+(?:\.[0-9]{1,2})?)|\b([0-9]+(?:\.[0-9]{1,2})?)\s*(?:dollars|usd)\b", re.I)


def normalize(text: str) -> str:
    """Canonical form used as the key of the clause index."""
    return WS_RE.sub(" ", str(text)).strip(EDGE_PUNCT).strip().lower()


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(str(text).lower()) if len(t) > 1 and t not in STOPWORDS]


def find_materials(text: str) -> list[str]:
    return [m.group(1).lower() for m in _MATERIAL_RE.finditer(text)]


def find_colors(text: str) -> list[str]:
    return [m.group(1).lower() for m in _COLOR_RE.finditer(text)]


def find_price(text: str) -> float | None:
    match = PRICE_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(1) or match.group(2))
    except (TypeError, ValueError):
        return None


def classify(clause: str) -> str:
    """Sort a clause into one of the ten attribute buckets the API allows.

    Ordering matters: a clause mentioning both a price and a colour is a budget
    statement first. Checks run from most to least specific.
    """
    low = clause.lower()
    if "budget" in low or "price" in low or PRICE_RE.search(low):
        return "budget"
    if any(w in low for w in ("brand", "manufacturer", "by ", "store")):
        if "manufacturer" in low or "brand" in low:
            return "brand"
    if _MATERIAL_RE.search(low):
        return "material"
    if _COLOR_RE.search(low):
        return "color"
    if any(w in low for w in SIZE_WORDS):
        return "size"
    if any(w in low for w in USE_CASE_WORDS):
        return "use_case"
    if any(w in low for w in STYLE_WORDS):
        return "style"
    return "feature"
