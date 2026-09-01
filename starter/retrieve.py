"""Loads the product catalog and searches it.

Three jobs happen in this file:

1. Read `catalog.jsonl` once when the agent starts and load all 50,000
   products into an in-memory SQLite table that supports full-text search.
   SQLite has a built-in search extension called FTS5, which gives us keyword
   ranking without installing any search library.
2. Count how many products contain each word. This lets us tell common words
   apart from rare ones later on: "leather" appears in thousands of products
   and barely narrows anything down, while an unusual phrase might appear in
   only one and settle the question outright.
3. Keep a cleaned-up copy of each product's text, so that later we can quickly
   ask "does this product mention the word 'cotton'?"
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

# Matches a run of letters and digits. Used to chop text into words.
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
# Matches anything that is not a letter or a digit. Used to turn punctuation
# into spaces.
PUNCTUATION_RE = re.compile(r"[^a-z0-9]+")

# Words that are far too common to tell us anything about which product the
# customer wants. We remove them before searching, because a query containing
# "the" would match essentially the whole catalog.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "have", "has", "was", "were", "will", "can", "do", "does", "not", "no",
    "am", "if", "so", "just", "very", "more", "most", "your", "our", "their",
}

# The catalog fields we search over. This tuple must stay in the same order as
# the columns in the CREATE VIRTUAL TABLE statement further down, because the
# weights on the next line are matched to the columns by position.
SEARCH_FIELDS = ("title", "categories", "features", "details", "store", "description")

# How much a match in each column counts for. Reading left to right these line
# up with parent_asin, title, categories, features, details, store and
# description. A word found in the title counts six times as much as the same
# word found in the description, because a title is short and every word in it
# was chosen deliberately, while a description is long and rambling and a word
# can appear in it by accident. The leading 0.0 switches off parent_asin, which
# is an ID and not worth searching. These numbers came with the organizer's
# original starter agent and we kept them: a full sweep of alternatives changed
# the final score by less than the measurement noise.
BM25_WEIGHTS = "0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0"

# How many products to insert into SQLite in one go. Inserting one row at a
# time would be slow, and holding all 50,000 rows in a list before inserting
# would waste memory, so we insert in chunks.
BATCH_SIZE = 1000


def flatten(value: object) -> str:
    """Turn one catalog field into a single line of plain text.

    Catalog fields do not all have the same shape. Some are ordinary strings,
    some are lists of bullet points, and some are dictionaries mapping a spec
    name to its value. Whichever shape a field arrives in, we want one flat
    string that we can search through.
    """
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase words, dropping stopwords and single letters."""
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def normalize(text: str) -> str:
    """Rewrite text so that checking for a whole word becomes a substring check.

    Everything is lowercased, every punctuation mark is replaced by a space,
    and one extra space is added to the front and to the back. So

        "100% Cotton, Imported."   becomes   " 100 cotton imported "

    Now every single word in the string has a space on both sides of it, which
    means we can test for a whole word just by searching for that word wrapped
    in spaces:

        " cotton " in blob     ->  True
        " cot " in blob        ->  False, which is what we want

    Two details are doing real work here. Wrapping the word in spaces is what
    stops a plain `"cotton" in blob` from also matching "cottonwood" and
    "cottontail". Padding the two ends is what stops us from missing a word
    that happens to sit at the very start or very end of the text, where it
    would otherwise have no space beside it.

    The obvious alternative is to split the text into a set of words and check
    membership in that. It gives the same answer, but it builds a brand new set
    every time we ask, and we end up asking thousands of times per turn.
    """
    return " " + PUNCTUATION_RE.sub(" ", text.lower()).strip() + " "


class CatalogIndex:
    """The searchable catalog. Built once when the agent starts, then read-only."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        # Product ID -> all of that product's text, normalized for word lookups.
        self.blob: dict[str, str] = {}
        # Product ID -> only that product's category text, normalized.
        self.category_blob: dict[str, str] = {}
        # Word -> how many products contain it anywhere in their text.
        self.document_frequency: Counter[str] = Counter()
        # How many products are in the catalog.
        self.size = 0
        self._build()

    def _build(self) -> None:
        """Read the catalog file and fill in the search table and the lookups."""
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, ...]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                columns = tuple(flatten(product.get(field)) for field in SEARCH_FIELDS)
                batch.append((parent_asin, *columns))

                # Alongside the row we hand to SQLite, keep our own copies: one
                # of the whole product for word lookups, one of just the
                # categories, and a tally of which words this product uses.
                # Counting a set of words rather than a list means a product
                # that says "cotton" ten times still only counts once.
                joined = " ".join(columns)
                self.blob[parent_asin] = normalize(joined)
                self.category_blob[parent_asin] = normalize(columns[1])
                self.document_frequency.update(set(tokenize(joined)))
                self.size += 1

                if len(batch) >= BATCH_SIZE:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        # The loop above only writes when a batch fills up, so whatever is left
        # over at the end still needs inserting.
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def idf(self, term: str) -> float:
        """How rare a word is, expressed as a number. Rarer words score higher.

        IDF is short for inverse document frequency. A word that appears in
        almost every product scores close to 1, and a word that appears in a
        single product scores around 11. We use this to work out which parts of
        what the customer said are actually worth matching on. A word we have
        never seen in the catalog is treated as being as rare as possible.
        """
        frequency = self.document_frequency.get(term, 0)
        return math.log((self.size + 1) / (frequency + 1)) + 1.0

    def contains(self, parent_asin: str, token: str) -> bool:
        """Does this product's text contain this exact word?"""
        return f" {token} " in self.blob.get(parent_asin, "")

    def search(self, terms: list[str], limit: int) -> list[str]:
        """Find the products that best match these words, best match first.

        Builds a query that means "contains any of these words", hands it to
        SQLite, and asks for the best `limit` results. The ranking comes from
        BM25, a standard keyword-relevance formula that rewards matching rare
        words and matching in the columns we weighted heavily.

        Two safeguards. We cap the query at 40 words, because a long
        conversation would otherwise build an enormous query mostly made of
        low-value terms. And if the query turns out to be malformed we return
        an empty list rather than letting the error escape, because the
        evaluator counts a crash as a completely lost session.
        """
        unique = list(dict.fromkeys(terms))[:40]
        if not unique:
            return []
        expression = " OR ".join(f'"{term}"' for term in unique)
        try:
            rows = self.connection.execute(
                f"SELECT parent_asin FROM products WHERE products MATCH ? "
                f"ORDER BY bm25(products, {BM25_WEIGHTS}) LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.Error:
            return []
        return [str(row[0]) for row in rows]
