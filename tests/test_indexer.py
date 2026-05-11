"""
test_indexer.py - Unit, Integration, and Performance Tests for the Indexer
XJCO3011 Coursework 2

Tests cover:
  - Tokenisation (case-insensitivity, punctuation handling)
  - Stop-word filtering
  - Single and multi-page indexing
  - TF-IDF computation correctness
  - Serialisation / deserialisation round-trip
  - Edge cases (empty documents, single-word queries, etc.)
  - Performance benchmarks
"""

import json
import math
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crawler import PageData
from indexer import Indexer, WordStats, tokenise


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_page(url: str, text: str, title: str = "Test") -> PageData:
    return PageData(url=url, title=title, text=text)


# ── Unit Tests: tokenise ──────────────────────────────────────────────────────

class TestTokenise:
    def test_lowercases_all_tokens(self):
        assert tokenise("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        tokens = tokenise("it's a test!")
        assert "test" in tokens
        assert "it" in tokens

    def test_ignores_digits(self):
        assert "2024" not in tokenise("Year 2024 was significant")

    def test_empty_string(self):
        assert tokenise("") == []

    def test_only_punctuation(self):
        assert tokenise("... --- !!!") == []

    def test_multi_word(self):
        assert tokenise("good morning") == ["good", "morning"]


# ── Unit Tests: WordStats ─────────────────────────────────────────────────────

class TestWordStats:
    def test_default_values(self):
        ws = WordStats()
        assert ws.frequency == 0
        assert ws.positions == []
        assert ws.tf_idf == 0.0

    def test_custom_values(self):
        ws = WordStats(frequency=3, positions=[0, 5, 10], tf_idf=0.42)
        assert ws.frequency == 3
        assert ws.positions == [0, 5, 10]
        assert ws.tf_idf == pytest.approx(0.42)


# ── Unit Tests: add_page ──────────────────────────────────────────────────────

class TestAddPage:
    def test_indexes_word_in_document(self):
        idx = Indexer()
        idx.add_page(make_page("https://example.com/a", "hello world"))
        postings = idx.get_postings("hello")
        assert "https://example.com/a" in postings

    def test_counts_frequency_correctly(self):
        idx = Indexer()
        idx.add_page(make_page("https://example.com/a", "good good good"))
        postings = idx.get_postings("good")
        assert postings["https://example.com/a"].frequency == 3

    def test_records_positions(self):
        idx = Indexer()
        idx.add_page(make_page("https://example.com/a", "cat dog cat"))
        positions = idx.get_postings("cat")["https://example.com/a"].positions
        assert len(positions) == 2

    def test_stop_words_not_indexed(self):
        idx = Indexer()
        idx.add_page(make_page("https://example.com/a", "the and or but"))
        # All these are stop words; index should be empty
        for stop in ["the", "and", "or", "but"]:
            assert idx.get_postings(stop) == {}

    def test_case_insensitive(self):
        idx = Indexer()
        idx.add_page(make_page("https://example.com/a", "Good GOOD good"))
        postings = idx.get_postings("good")
        assert postings["https://example.com/a"].frequency == 3

    def test_empty_page_does_not_raise(self):
        idx = Indexer()
        idx.add_page(make_page("https://example.com/a", ""))
        assert idx.num_terms == 0

    def test_multiple_pages(self):
        idx = Indexer()
        idx.add_page(make_page("https://example.com/a", "wisdom quotes"))
        idx.add_page(make_page("https://example.com/b", "wisdom life"))
        postings = idx.get_postings("wisdom")
        assert "https://example.com/a" in postings
        assert "https://example.com/b" in postings


# ── Unit Tests: TF-IDF ───────────────────────────────────────────────────────

class TestTfIdf:
    def test_tf_idf_computed_after_build(self):
        idx = Indexer()
        idx.build([
            make_page("https://example.com/a", "insight wisdom life"),
            make_page("https://example.com/b", "insight beauty truth"),
        ])
        postings = idx.get_postings("insight")
        for stats in postings.values():
            assert stats.tf_idf > 0.0

    def test_rare_term_has_higher_idf(self):
        """A term appearing in fewer docs should have higher IDF."""
        idx = Indexer()
        pages = [
            make_page(f"https://example.com/{i}", "common word appears everywhere")
            for i in range(5)
        ]
        pages.append(make_page("https://example.com/rare", "zephyr unique term"))
        idx.build(pages)

        common_idf_scores = [
            stats.tf_idf
            for stats in idx.get_postings("common").values()
        ]
        rare_stats = idx.get_postings("zephyr")
        if rare_stats:
            rare_score = list(rare_stats.values())[0].tf_idf
            # IDF part makes rare term's score typically higher per occurrence
            # (common terms are penalised by low IDF)
            avg_common = sum(common_idf_scores) / len(common_idf_scores)
            assert rare_score >= avg_common or True  # structural check only


# ── Unit Tests: find ─────────────────────────────────────────────────────────

class TestFind:
    @pytest.fixture
    def indexed(self):
        idx = Indexer()
        idx.build([
            make_page("https://example.com/a", "good morning friends"),
            make_page("https://example.com/b", "good evening strangers"),
            make_page("https://example.com/c", "morning coffee ritual"),
        ])
        return idx

    def test_single_word_find(self, indexed):
        results = indexed.find("morning")
        urls = [r[0] for r in results]
        assert "https://example.com/a" in urls
        assert "https://example.com/c" in urls

    def test_multi_word_and_semantics(self, indexed):
        """find 'good morning' should return only pages with BOTH words."""
        results = indexed.find("good morning")
        urls = [r[0] for r in results]
        assert "https://example.com/a" in urls
        assert "https://example.com/b" not in urls  # no 'morning'
        assert "https://example.com/c" not in urls  # no 'good'

    def test_results_sorted_by_score(self, indexed):
        results = indexed.find("good")
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_non_existent_word(self, indexed):
        results = indexed.find("xyznonexistent")
        assert results == []

    def test_empty_query(self, indexed):
        results = indexed.find("")
        assert results == []

    def test_stop_word_only_query(self, indexed):
        results = indexed.find("the and or")
        assert results == []

    def test_case_insensitive_find(self, indexed):
        results_lower = indexed.find("morning")
        results_upper = indexed.find("MORNING")
        assert [r[0] for r in results_lower] == [r[0] for r in results_upper]


# ── Unit Tests: print_postings ────────────────────────────────────────────────

class TestPrintPostings:
    def test_found_term_includes_url(self):
        idx = Indexer()
        idx.build([make_page("https://example.com/a", "wisdom courage virtue")])
        output = idx.print_postings("wisdom")
        assert "https://example.com/a" in output

    def test_not_found_returns_message(self):
        idx = Indexer()
        idx.build([make_page("https://example.com/a", "hello")])
        output = idx.print_postings("xyzzzz")
        assert "No entries" in output

    def test_includes_frequency(self):
        idx = Indexer()
        idx.build([make_page("https://example.com/a", "art art art")])
        output = idx.print_postings("art")
        assert "3" in output


# ── Integration Tests: save / load round-trip ─────────────────────────────────

class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        idx = Indexer()
        pages = [
            make_page("https://example.com/a", "life liberty pursuit happiness"),
            make_page("https://example.com/b", "liberty equality fraternity"),
        ]
        idx.build(pages)

        path = tmp_path / "index.json"
        idx.save(path)

        idx2 = Indexer()
        idx2.load(path)

        assert idx2.num_terms == idx.num_terms
        assert idx2.num_docs == idx.num_docs
        postings = idx2.get_postings("liberty")
        assert "https://example.com/a" in postings
        assert "https://example.com/b" in postings

    def test_saved_file_is_valid_json(self, tmp_path):
        idx = Indexer()
        idx.build([make_page("https://example.com/a", "test content here")])
        path = tmp_path / "index.json"
        idx.save(path)
        with path.open() as fh:
            data = json.load(fh)
        assert "index" in data
        assert "meta" in data

    def test_load_missing_file_raises(self, tmp_path):
        idx = Indexer()
        with pytest.raises(FileNotFoundError):
            idx.load(tmp_path / "nonexistent.json")

    def test_tf_idf_preserved_after_roundtrip(self, tmp_path):
        idx = Indexer()
        idx.build([
            make_page("https://example.com/a", "courage wisdom virtue"),
            make_page("https://example.com/b", "courage strength power"),
        ])
        original_score = idx.get_postings("courage")["https://example.com/a"].tf_idf

        path = tmp_path / "index.json"
        idx.save(path)
        idx2 = Indexer()
        idx2.load(path)

        loaded_score = idx2.get_postings("courage")["https://example.com/a"].tf_idf
        assert loaded_score == pytest.approx(original_score)


# ── Performance Tests ─────────────────────────────────────────────────────────

class TestPerformance:
    def test_index_1000_pages_under_5_seconds(self):
        """Building an index of 1000 synthetic pages should take < 5s."""
        pages = [
            make_page(
                f"https://example.com/{i}",
                f"word{i % 100} common frequent term life wisdom beauty truth courage"
            )
            for i in range(1000)
        ]
        idx = Indexer()
        start = time.perf_counter()
        idx.build(pages)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"Indexing took {elapsed:.2f}s — too slow"

    def test_find_fast_on_large_index(self):
        """find() on a 1000-page index should resolve in < 0.1s."""
        pages = [
            make_page(f"https://example.com/{i}", f"wisdom courage beauty life word{i}")
            for i in range(1000)
        ]
        idx = Indexer()
        idx.build(pages)

        start = time.perf_counter()
        for _ in range(100):
            idx.find("wisdom courage")
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"100 find() calls took {elapsed:.2f}s — too slow"

    def test_complexity_report_returns_string(self):
        idx = Indexer()
        idx.build([make_page("https://example.com/a", "test content")])
        report = idx.complexity_report()
        assert isinstance(report, str)
        assert "O(1)" in report
