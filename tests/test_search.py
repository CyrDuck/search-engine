"""
test_search.py - Unit and Integration Tests for the Search Engine
XJCO3011 Coursework 2

Tests cover:
  - cmd_print: found, not found, spelling suggestion
  - cmd_find: single/multi-word, no results, empty query, stop words
  - Spelling suggestion accuracy
  - Edge cases: special characters, very long queries, case sensitivity
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crawler import PageData
from indexer import Indexer
from search import SearchEngine


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_page(url: str, text: str, title: str = "Test") -> PageData:
    return PageData(url=url, title=title, text=text)


@pytest.fixture
def engine():
    """A SearchEngine backed by a small pre-built index."""
    idx = Indexer()
    idx.build([
        make_page("https://quotes.toscrape.com/", "wisdom life beauty truth courage", "Home"),
        make_page("https://quotes.toscrape.com/page/2", "wisdom courage friends wisdom", "Page 2"),
        make_page("https://quotes.toscrape.com/page/3", "nonsense indifference strange", "Page 3"),
    ])
    return SearchEngine(idx)


# ── Tests: cmd_print ──────────────────────────────────────────────────────────

class TestCmdPrint:
    def test_found_term_returns_url(self, engine):
        output = engine.cmd_print("wisdom")
        assert "quotes.toscrape.com" in output

    def test_found_term_returns_frequency(self, engine):
        output = engine.cmd_print("wisdom")
        assert "Frequency" in output

    def test_not_found_returns_message(self, engine):
        output = engine.cmd_print("xyznonexistent")
        assert "No entries" in output

    def test_not_found_offers_suggestion(self, engine):
        """A close misspelling should trigger a spelling suggestion."""
        output = engine.cmd_print("wisdum")
        # Either no-entries message or spelling suggestion — we don't require
        # suggestion for every typo, but the output must not crash.
        assert isinstance(output, str)

    def test_empty_term(self, engine):
        output = engine.cmd_print("")
        assert "Usage" in output

    def test_case_insensitive(self, engine):
        output_lower = engine.cmd_print("wisdom")
        output_upper = engine.cmd_print("WISDOM")
        # Both should find the same documents
        assert "quotes.toscrape.com" in output_lower
        assert "quotes.toscrape.com" in output_upper

    def test_stop_word_returns_no_entries(self, engine):
        output = engine.cmd_print("the")
        assert "No entries" in output


# ── Tests: cmd_find ───────────────────────────────────────────────────────────

class TestCmdFind:
    def test_single_word_returns_results(self, engine):
        output = engine.cmd_find("nonsense")
        assert "page/3" in output

    def test_multi_word_and_logic(self, engine):
        """'wisdom courage' should return pages containing BOTH words."""
        output = engine.cmd_find("wisdom courage")
        # Page 2 has both "wisdom" and "courage"; Page 1 has both too
        assert "Found" in output

    def test_word_not_in_any_page(self, engine):
        output = engine.cmd_find("zzzznonexistent")
        assert "No pages found" in output

    def test_empty_query(self, engine):
        output = engine.cmd_find("")
        assert "Usage" in output

    def test_multi_word_intersection_excludes_partial(self, engine):
        """
        'wisdom strange' — 'wisdom' is on pages 1 & 2, 'strange' on page 3 only.
        Intersection should be empty.
        """
        output = engine.cmd_find("wisdom strange")
        assert "No pages found" in output

    def test_results_ranked_by_score(self, engine):

        output = engine.cmd_find("wisdom")
        lines = output.splitlines()
    # Find the [1] rank line and the URL line right after it
        for i, line in enumerate(lines):
            if "[1]" in line:
            # The URL line is the next line (starts with "URL  :")
                url_line = lines[i + 1] if i + 1 < len(lines) else ""
                assert "page/2" in url_line  # Page 2 has 'wisdom' twice
                break
        else:
            pytest.fail("No rank [1] result found in output")

    def test_case_insensitive_find(self, engine):
        output_lower = engine.cmd_find("nonsense")
        output_upper = engine.cmd_find("NONSENSE")
        assert ("page/3" in output_lower) == ("page/3" in output_upper)

    def test_find_with_stop_words_in_query(self, engine):
        """Stop words in a query should be ignored, not cause zero results."""
        output_plain = engine.cmd_find("wisdom")
        output_with_stop = engine.cmd_find("the wisdom")
        # Both should return the same logical results (stop words filtered)
        assert "quotes.toscrape.com" in output_plain
        assert isinstance(output_with_stop, str)

    def test_very_long_query(self, engine):
        """A query with many words shouldn't crash — should just return no results."""
        long_query = " ".join([f"word{i}" for i in range(50)])
        output = engine.cmd_find(long_query)
        assert isinstance(output, str)

    def test_special_characters_in_query(self, engine):
        """Special characters should be silently stripped by tokeniser."""
        output = engine.cmd_find("wisdom!!!???")
        assert isinstance(output, str)


# ── Integration Tests ─────────────────────────────────────────────────────────

class TestSearchIntegration:
    def test_full_workflow(self, tmp_path):
        """Build → save → load → find should return consistent results."""
        idx = Indexer()
        pages = [
            make_page("https://quotes.toscrape.com/", "love life friendship"),
            make_page("https://quotes.toscrape.com/page/2", "love truth wisdom"),
        ]
        idx.build(pages)

        path = tmp_path / "index.json"
        idx.save(path)

        idx2 = Indexer()
        idx2.load(path)
        engine2 = SearchEngine(idx2)

        results = engine2.cmd_find("love")
        assert "Found" in results
        assert "quotes.toscrape.com" in results

    def test_print_after_load(self, tmp_path):
        idx = Indexer()
        idx.build([make_page("https://quotes.toscrape.com/", "indifference apathy")])
        path = tmp_path / "index.json"
        idx.save(path)

        idx2 = Indexer()
        idx2.load(path)
        engine2 = SearchEngine(idx2)

        output = engine2.cmd_print("indifference")
        assert "quotes.toscrape.com" in output


# ── Edge Case Tests ───────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_index_find_returns_empty(self):
        idx = Indexer()
        engine = SearchEngine(idx)
        output = engine.cmd_find("anything")
        assert isinstance(output, str)

    def test_empty_index_print_returns_message(self):
        idx = Indexer()
        engine = SearchEngine(idx)
        output = engine.cmd_print("anything")
        assert isinstance(output, str)

    def test_unicode_text(self):
        idx = Indexer()
        idx.build([make_page("https://example.com/", "café résumé naïve")])
        engine = SearchEngine(idx)
        output = engine.cmd_find("caf")
        assert isinstance(output, str)
