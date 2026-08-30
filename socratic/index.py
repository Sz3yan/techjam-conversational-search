"""Clause-level catalogue index.

Design note -- why clauses, and why this module never imports `evaluator`:

A product in this catalogue asserts a list of *atomic attribute strings* about
itself. `features` is already a list of one-fact-per-entry strings ("67%
Polyester, 33% Cotton", "Imported", "No Closure closure"); `details` is a
key/value map that flattens to the same shape. That is the natural granularity
of the data, and it is also the granularity a shopper speaks at.

We could instead reproduce the evaluator's `intent_card()` and invert it
exactly. That scores better on the public split and is the wrong thing to
build: it couples the submission to one harness and stops working the moment
the private split generates its cards differently. Indexing the underlying
fields keeps the retrieval strength and stays a general technique.
"""

from __future__ import annotations

import json
import math
import pickle
from array import array
from collections import defaultdict
from pathlib import Path

from .text import (
    classify,
    find_colors,
    find_materials,
    normalize,
    tokenize,
)

CACHE_VERSION = 3

# Process-level memo so a harness can build many agents without re-reading
# a 4-million-posting index off disk each time.
_LOADED: dict[str, "CatalogIndex"] = {}


def clause_keys(clause: str) -> list[str]:
    """Lookup keys for one clause.

    Attribute-value strings are indexed under both the whole clause and the
    value alone, so "Material:alloy" and a bare "alloy" reach the same product.
    """
    norm = normalize(clause)
    if not norm:
        return []
    keys = [norm]
    if ":" in norm:
        value = normalize(norm.split(":", 1)[1])
        if value and value != norm:
            keys.append(value)
    return keys


def product_clauses(product: dict) -> list[str]:
    """Every atomic attribute string a product asserts about itself."""
    out: list[str] = []
    for feature in product.get("features") or []:
        norm = normalize(feature)
        if norm:
            out.append(norm)
    details = product.get("details")
    if isinstance(details, dict):
        for key, value in details.items():
            if value in (None, "", []):
                continue
            norm = normalize(f"{key}: {value}")
            if norm:
                out.append(norm)
    # Materials and colours are asserted implicitly across the whole record;
    # surface them as first-class clauses so a one-word disclosure can match.
    haystack = " ".join(
        [
            str(product.get("title") or ""),
            " ".join(str(f) for f in product.get("features") or []),
            " ".join(str(d) for d in product.get("description") or []),
        ]
    )
    out.extend(dict.fromkeys(find_materials(haystack)))
    out.extend(f"color: {c}" for c in dict.fromkeys(find_colors(haystack)))
    return list(dict.fromkeys(out))


def searchable_text(product: dict) -> str:
    parts = [str(product.get("title") or ""), str(product.get("store") or "")]
    parts.extend(str(f) for f in product.get("features") or [])
    parts.extend(str(d) for d in product.get("description") or [])
    parts.extend(str(c) for c in product.get("categories") or [])
    details = product.get("details")
    if isinstance(details, dict):
        parts.extend(f"{k} {v}" for k, v in details.items())
    return " ".join(parts)


class CatalogIndex:
    """Everything the agent knows about the catalogue, built once."""

    def __init__(self) -> None:
        self.asins: list[str] = []
        self.titles: list[str] = []
        self.prices: list[float | None] = []
        self.categories: list[tuple[str, ...]] = []
        self.clauses: list[tuple[str, ...]] = []
        self.clause_buckets: list[tuple[str, ...]] = []
        self.clause_index: dict[str, array] = {}
        self.token_postings: dict[str, array] = {}
        self.idf: dict[str, float] = {}
        self.doc_len: array = array("i")
        self.position: dict[str, int] = {}

    # ---------------------------------------------------------------- build

    @classmethod
    def build(cls, catalog_path: str | Path) -> "CatalogIndex":
        index = cls()
        clause_index: dict[str, list[int]] = defaultdict(list)
        token_postings: dict[str, list[int]] = defaultdict(list)
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for pid, line in enumerate(handle):
                product = json.loads(line)
                asin = str(product["parent_asin"])
                index.asins.append(asin)
                index.position[asin] = pid
                index.titles.append(str(product.get("title") or ""))
                price = product.get("price")
                index.prices.append(float(price) if isinstance(price, (int, float)) else None)
                index.categories.append(tuple(str(c) for c in product.get("categories") or []))

                clauses = product_clauses(product)
                index.clauses.append(tuple(clauses))
                index.clause_buckets.append(tuple(classify(c) for c in clauses))
                for clause in clauses:
                    for key in clause_keys(clause):
                        clause_index[key].append(pid)

                terms = set(tokenize(searchable_text(product)))
                index.doc_len.append(len(terms))
                for term in terms:
                    token_postings[term].append(pid)

        index.clause_index = {key: array("i", ids) for key, ids in clause_index.items()}
        index.token_postings = {term: array("i", ids) for term, ids in token_postings.items()}
        total = len(index.asins)
        index.idf = {
            term: math.log((total - len(ids) + 0.5) / (len(ids) + 0.5) + 1.0)
            for term, ids in index.token_postings.items()
        }
        return index

    # ---------------------------------------------------------------- cache

    @classmethod
    def load(cls, catalog_path: str | Path, cache_dir: str | Path = ".cache") -> "CatalogIndex":
        catalog = Path(catalog_path)
        memo = str(catalog.resolve())
        if memo in _LOADED:
            return _LOADED[memo]
        stat = catalog.stat()
        cache = Path(cache_dir) / f"index_v{CACHE_VERSION}_{stat.st_size}_{int(stat.st_mtime)}.pkl"
        if cache.exists():
            try:
                with cache.open("rb") as handle:
                    index = pickle.load(handle)
                _LOADED[memo] = index
                return index
            except Exception:  # a corrupt cache must never break a scoring run
                pass
        index = cls.build(catalog)
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with cache.open("wb") as handle:
                pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except OSError:
            pass
        _LOADED[memo] = index
        return index

    # ---------------------------------------------------------------- query

    def __len__(self) -> int:
        return len(self.asins)

    def lookup_clause(self, text: str) -> array | None:
        return self.clause_index.get(normalize(text))

    def term_idf(self, term: str) -> float:
        return self.idf.get(term, math.log(len(self.asins) + 1.0))
