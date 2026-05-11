"""
indexer.py - Basic Inverted Index (frequency only)
XJCO3011 Coursework 2
"""

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class WordStats:
    frequency: int = 0
    positions: list[int] = field(default_factory=list)


def tokenise(text):
    return re.findall(r"[a-z]+", text.lower())


class Indexer:
    STOP_WORDS = frozenset("a an the and or but in on at to for of with by".split())

    def __init__(self):
        self._index = {}
        self._page_titles = {}
        self._num_docs = 0

    def add_page(self, page):
        tokens = tokenise(page.text)
        self._page_titles[page.url] = page.title
        self._num_docs += 1
        for position, token in enumerate(tokens):
            if token in self.STOP_WORDS:
                continue
            if token not in self._index:
                self._index[token] = {}
            if page.url not in self._index[token]:
                self._index[token][page.url] = WordStats()
            self._index[token][page.url].frequency += 1
            self._index[token][page.url].positions.append(position)

    def build(self, pages):
        for page in pages:
            self.add_page(page)
        print(f"Index built: {len(self._index)} terms, {self._num_docs} documents.")

    def get_postings(self, term):
        return self._index.get(term.lower(), {})

    def find(self, query):
        terms = [t for t in tokenise(query) if t not in self.STOP_WORDS]
        if not terms:
            return []
        candidate_urls = None
        for term in terms:
            postings = set(self._index.get(term, {}).keys())
            candidate_urls = postings if candidate_urls is None else candidate_urls & postings
        if not candidate_urls:
            return []
        return [
            (url, sum(self._index[t][url].frequency for t in terms if url in self._index.get(t, {})))
            for url in candidate_urls
        ]

    def print_postings(self, term):
        postings = self.get_postings(term)
        if not postings:
            return f"No entries found for '{term}'."
        lines = [f"Inverted index for '{term}':"]
        for url, stats in postings.items():
            lines.append(f"  URL: {url}\n  Frequency: {stats.frequency}\n  Positions: {stats.positions[:5]}")
        return "\n".join(lines)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_docs": self._num_docs,
            "page_titles": self._page_titles,
            "index": {
                term: {url: asdict(s) for url, s in postings.items()}
                for term, postings in self._index.items()
            }
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Index saved to {path}.")

    def load(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index not found: {path}. Run 'build' first.")
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self._num_docs = payload["num_docs"]
        self._page_titles = payload["page_titles"]
        self._index = {
            term: {url: WordStats(**s) for url, s in postings.items()}
            for term, postings in payload["index"].items()
        }
        print(f"Index loaded: {len(self._index)} terms, {self._num_docs} documents.")

    @property
    def num_terms(self):
        return len(self._index)

    @property
    def num_docs(self):
        return self._num_docs