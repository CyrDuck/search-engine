"""
main.py - Command-Line Interface for the Search Engine Tool
XJCO3011 Coursework 2

Commands:
    build        Crawl website, build index, save to disk
    load         Load previously built index from disk
    print <word> Print inverted index entry for a word
    find <query> Find pages containing all words in query
    stats        Show index complexity report
    help         Show available commands
    exit / quit  Exit the shell

Usage:
    python main.py
"""

import sys
import logging
from pathlib import Path

# Ensure src/ is on the path when running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from crawler import Crawler
from indexer import Indexer
from search import SearchEngine

# ── Configuration ────────────────────────────────────────────────────────────
BASE_URL = "https://quotes.toscrape.com/"
INDEX_PATH = Path(__file__).parent.parent / "data" / "index.json"
POLITENESS_WINDOW = 6.0  # seconds between requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent.parent / "data" / "crawler.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

BANNER = """
╔══════════════════════════════════════════════════╗
║   XJCO3011 Search Engine  ·  Coursework 2        ║
║   Target: quotes.toscrape.com                    ║
╚══════════════════════════════════════════════════╝
Type 'help' for available commands.
"""

HELP_TEXT = """
Available commands:
  build          Crawl the website, build index, and save to disk
  load           Load an existing index from disk
  print <word>   Print the inverted index entry for <word>
  find <query>   Find pages containing all words in <query>
  stats          Show index statistics and complexity analysis
  help           Show this help message
  exit | quit    Exit the search engine
"""


def run_shell(indexer: Indexer, engine: SearchEngine) -> None:
    """Interactive REPL loop."""
    index_loaded = False
    print(BANNER)

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        parts = raw.split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1] if len(parts) > 1 else ""

        # ── build ──────────────────────────────────────────────────────
        if command == "build":
            print(f"Starting crawl of {BASE_URL} (politeness window: {POLITENESS_WINDOW}s)…")
            print("This may take several minutes. Please wait.\n")
            crawler = Crawler(base_url=BASE_URL, politeness_window=POLITENESS_WINDOW)
            pages = crawler.crawl()
            print(f"\nCrawled {len(pages)} pages. Building index…")
            indexer.build(pages)
            INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            indexer.save(INDEX_PATH)
            print(f"Index saved to: {INDEX_PATH}")
            print(f"Terms indexed: {indexer.num_terms} | Documents: {indexer.num_docs}")
            index_loaded = True

        # ── load ──────────────────────────────────────────────────────
        elif command == "load":
            try:
                indexer.load(INDEX_PATH)
                engine = SearchEngine(indexer)
                print(f"Index loaded: {indexer.num_terms} terms, {indexer.num_docs} documents.")
                index_loaded = True
            except FileNotFoundError as exc:
                print(f"Error: {exc}")

        # ── print ─────────────────────────────────────────────────────
        elif command == "print":
            if not index_loaded:
                print("No index loaded. Run 'build' or 'load' first.")
            elif not argument:
                print("Usage: print <word>")
            else:
                print(engine.cmd_print(argument))

        # ── find ──────────────────────────────────────────────────────
        elif command == "find":
            if not index_loaded:
                print("No index loaded. Run 'build' or 'load' first.")
            elif not argument:
                print("Usage: find <word> [word …]")
            else:
                print(engine.cmd_find(argument))

        # ── stats ─────────────────────────────────────────────────────
        elif command == "stats":
            if not index_loaded:
                print("No index loaded. Run 'build' or 'load' first.")
            else:
                print(indexer.complexity_report())

        # ── help ──────────────────────────────────────────────────────
        elif command in ("help", "?"):
            print(HELP_TEXT)

        # ── exit ──────────────────────────────────────────────────────
        elif command in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        else:
            print(f"Unknown command: '{command}'. Type 'help' for usage.")


def main() -> None:
    """Entry point — create shared objects and start the REPL."""
    # Ensure data directory exists before logging starts
    (Path(__file__).parent.parent / "data").mkdir(parents=True, exist_ok=True)

    indexer = Indexer()
    engine = SearchEngine(indexer)
    run_shell(indexer, engine)


if __name__ == "__main__":
    main()
