"""Sparse retrieval over the frozen catalog.

Wraps the catalog in an in-memory SQLite FTS5 index (same construction as the
shipped baseline) and adds the two things reranking needs: per-term inverse
document frequency, and a normalized text blob per product for phrase checks.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
PUNCTUATION_RE = re.compile(r"[^a-z0-9]+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "have", "has", "was", "were", "will", "can", "do", "does", "not", "no",
    "am", "if", "so", "just", "very", "more", "most", "your", "our", "their",
}

# Column order must match the CREATE VIRTUAL TABLE statement below. The weights
# are inherited from the baseline: title dominates, description is background.
SEARCH_FIELDS = ("title", "categories", "features", "details", "store", "description")
BM25_WEIGHTS = "0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0"
BATCH_SIZE = 1000


def flatten(value: object) -> str:
    """Render a catalog field (string, list, or dict) as flat text."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, pad with spaces.

    Padding lets callers test whole-token membership with a plain substring
    check (`" cotton " in blob`) without building a set per product.
    """
    return " " + PUNCTUATION_RE.sub(" ", text.lower()).strip() + " "


class CatalogIndex:
    """Read-only view of the 50k-product catalog. Built once per process."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.blob: dict[str, str] = {}
        self.category_blob: dict[str, str] = {}
        self.document_frequency: Counter[str] = Counter()
        self.size = 0
        self._build()

    def _build(self) -> None:
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

                joined = " ".join(columns)
                self.blob[parent_asin] = normalize(joined)
                self.category_blob[parent_asin] = normalize(columns[1])
                self.document_frequency.update(set(tokenize(joined)))
                self.size += 1

                if len(batch) >= BATCH_SIZE:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def idf(self, term: str) -> float:
        """Rarity weight. Unseen terms score as maximally rare."""
        frequency = self.document_frequency.get(term, 0)
        return math.log((self.size + 1) / (frequency + 1)) + 1.0

    def contains(self, parent_asin: str, token: str) -> bool:
        return f" {token} " in self.blob.get(parent_asin, "")

    def search(self, terms: list[str], limit: int) -> list[str]:
        """BM25 over an OR of the supplied terms, best match first."""
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
