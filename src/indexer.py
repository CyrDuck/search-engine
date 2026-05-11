"""
indexer.py - Inverted Index with TF-IDF Ranking
XJCO3011 Coursework 2

Data structure:
  index: dict[term -> dict[url -> WordStats]]

WordStats stores:
  - frequency (term frequency in the document)
  - positions (list of character offsets / word positions)
  - tf_idf   (computed after all documents are indexed)

Design rationale:
  A nested dict gives O(1) average-case lookup by term and URL.
  Storing positions enables future phrase-proximity ranking.
  TF-IDF (Term Frequency × Inverse Document Frequency) is the
  industry-standard baseline relevance signal used by early
  commercial search engines (Salton & Buckley, 1988).

  TF  = freq / total_words_in_doc        (normalised term frequency)
  IDF = log(N / df + 1) + 1              (smoothed inverse doc frequency)
  Score = TF × IDF
"""

import json
import logging
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from crawler import PageData

logger = logging.getLogger(__name__)


@dataclass
class WordStats:
    """Per-document statistics for a single term."""
    frequency: int = 0
    positions: list[int] = field(default_factory=list)
    tf_idf: float = 0.0


# Type alias for the full index structure
Index = dict[str, dict[str, WordStats]]


def tokenise(text: str) -> list[str]:
    """
    Lowercase and split *text* into alphabetic tokens.

    Non-alphabetic characters (punctuation, digits) are treated as
    delimiters so 'it's' → ['it', 's'] and '2024' is ignored.
    Case-insensitive per assignment requirement.

    Time complexity: O(n) where n = len(text).
    """
    return re.findall(r"[a-z]+", text.lower())


class Indexer:
    """
    Builds and manages an inverted index over a corpus of web pages.

    The index maps each unique term to a dictionary of {url: WordStats}.
    After all documents are added, call compute_tf_idf() to populate
    the tf_idf field with relevance scores.
    """

    # Words so common they hurt retrieval precision
    STOP_WORDS: frozenset[str] = frozenset(
        "a an the and or but in on at to for of with by from is was are were "
        "be been being have has had do does did will would could should may "
        "might shall can this that these those it its".split()
    )

    def __init__(self) -> None:
        self._index: Index = {}
        self._doc_lengths: dict[str, int] = {}   # url → total word count
        self._page_titles: dict[str, str] = {}    # url → page title
        self._num_docs: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_page(self, page: PageData) -> None:
        """
        Index a single *page*.

        Tokenises its text, records per-position occurrences, and
        updates the inverted index.

        Time complexity: O(W) where W = number of words in the page.
        Space complexity: O(U × V) where U = unique terms, V = pages.
        """
        tokens = tokenise(page.text)
        self._doc_lengths[page.url] = len(tokens)
        self._page_titles[page.url] = page.title
        self._num_docs += 1

        for position, token in enumerate(tokens):
            if token in self.STOP_WORDS:
                continue
            if token not in self._index:
                self._index[token] = {}
            if page.url not in self._index[token]:
                self._index[token][page.url] = WordStats()

            stats = self._index[token][page.url]
            stats.frequency += 1
            stats.positions.append(position)

        logger.debug("Indexed %s (%d tokens)", page.url, len(tokens))

    def build(self, pages: list[PageData]) -> None:
        """Index all *pages* then compute TF-IDF scores."""
        for page in pages:
            self.add_page(page)
        self.compute_tf_idf()
        logger.info("Index built: %d terms across %d documents.", len(self._index), self._num_docs)

    def compute_tf_idf(self) -> None:
        """
        Compute TF-IDF scores for every (term, document) pair.

        Uses smoothed IDF = log(N / (df + 1)) + 1 to avoid division-by-zero
        and to dampen the effect of terms appearing in almost every document.

        Time complexity: O(U × D) where U = unique terms, D = documents.
        """
        N = self._num_docs
        for term, postings in self._index.items():
            df = len(postings)  # document frequency
            idf = math.log(N / (df + 1)) + 1
            for url, stats in postings.items():
                doc_len = self._doc_lengths.get(url, 1) or 1
                tf = stats.frequency / doc_len
                stats.tf_idf = tf * idf

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_postings(self, term: str) -> dict[str, WordStats]:
        """Return postings for *term* (case-insensitive). Empty dict if not found."""
        return self._index.get(term.lower(), {})

    def find(self, query: str) -> list[tuple[str, float]]:
        """
        Find pages containing ALL words in *query*.

        Returns a list of (url, score) tuples sorted by descending TF-IDF sum.
        Multi-word query uses set intersection (AND semantics).

        Time complexity: O(Q × D log D) where Q = query terms, D = matching docs.
        """
        terms = [t for t in tokenise(query) if t not in self.STOP_WORDS]
        if not terms:
            return []

        # Start with the postings for the rarest term (smallest set)
        sorted_terms = sorted(terms, key=lambda t: len(self._index.get(t, {})))
        candidate_urls: set[str] | None = None

        for term in sorted_terms:
            postings = self._index.get(term, {})
            if candidate_urls is None:
                candidate_urls = set(postings.keys())
            else:
                candidate_urls &= set(postings.keys())

        if not candidate_urls:
            return []

        # Score = sum of TF-IDF across query terms
        scored: list[tuple[str, float]] = []
        for url in candidate_urls:
            score = sum(
                self._index[t][url].tf_idf
                for t in sorted_terms
                if url in self._index.get(t, {})
            )
            scored.append((url, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Serialise the index to a JSON file at *path*.

        The JSON schema is intentionally human-readable so markers can
        inspect the index without running the tool.
        Time complexity: O(U × D) for serialisation.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "meta": {
                "num_docs": self._num_docs,
                "num_terms": len(self._index),
            },
            "doc_lengths": self._doc_lengths,
            "page_titles": self._page_titles,
            "index": {
                term: {
                    url: asdict(stats)
                    for url, stats in postings.items()
                }
                for term, postings in self._index.items()
            },
        }

        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        logger.info("Index saved to %s (%d terms).", path, len(self._index))

    def load(self, path: str | Path) -> None:
        """
        Deserialise the index from a JSON file at *path*.

        Raises FileNotFoundError if the index has not been built yet.
        Time complexity: O(U × D) for deserialisation.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Index file not found: {path}\n"
                "Run 'build' first to create the index."
            )

        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        self._num_docs = payload["meta"]["num_docs"]
        self._doc_lengths = payload["doc_lengths"]
        self._page_titles = payload["page_titles"]
        self._index = {
            term: {
                url: WordStats(**stats)
                for url, stats in postings.items()
            }
            for term, postings in payload["index"].items()
        }

        logger.info(
            "Index loaded from %s (%d terms, %d docs).",
            path,
            len(self._index),
            self._num_docs,
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def print_postings(self, term: str) -> str:
        """Return a human-readable string of postings for *term*."""
        postings = self.get_postings(term)
        if not postings:
            return f"No entries found for '{term}'."

        lines = [f"Inverted index for '{term}' ({len(postings)} document(s)):\n"]
        for url, stats in sorted(postings.items(), key=lambda x: x[1].tf_idf, reverse=True):
            title = self._page_titles.get(url, url)
            lines.append(
                f"  URL      : {url}\n"
                f"  Title    : {title}\n"
                f"  Frequency: {stats.frequency}\n"
                f"  TF-IDF   : {stats.tf_idf:.6f}\n"
                f"  Positions: {stats.positions[:10]}{'...' if len(stats.positions) > 10 else ''}\n"
            )
        return "\n".join(lines)

    @property
    def num_terms(self) -> int:
        return len(self._index)

    @property
    def num_docs(self) -> int:
        return self._num_docs

    def complexity_report(self) -> str:
        """Return a brief complexity analysis report string."""
        total_postings = sum(len(v) for v in self._index.values())
        avg_postings = total_postings / max(len(self._index), 1)
        return (
            f"=== Complexity Report ===\n"
            f"Documents indexed : {self._num_docs}\n"
            f"Unique terms      : {len(self._index)}\n"
            f"Total postings    : {total_postings}\n"
            f"Avg postings/term : {avg_postings:.2f}\n"
            f"Index density     : {total_postings / max(self._num_docs * len(self._index), 1):.6f}\n"
            f"\nLookup complexity : O(1) average (hash map)\n"
            f"Build complexity  : O(W) where W = total words in corpus\n"
            f"TF-IDF complexity : O(U × D) where U = unique terms, D = docs\n"
            f"Find complexity   : O(Q × D log D) where Q = query terms\n"
        )
