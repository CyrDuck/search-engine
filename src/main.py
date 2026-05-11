"""
main.py - Basic CLI with build and load commands
XJCO3011 Coursework 2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from crawler import Crawler
from indexer import Indexer

BASE_URL = "https://quotes.toscrape.com/"
INDEX_PATH = Path(__file__).parent.parent / "data" / "index.json"


def main():
    indexer = Indexer()
    index_loaded = False

    print("XJCO3011 Search Engine - Coursework 2")
    print("Commands: build, load, exit")

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        command = raw.split()[0].lower()

        if command == "build":
            print(f"Crawling {BASE_URL}...")
            crawler = Crawler(base_url=BASE_URL, politeness_window=6.0)
            pages = crawler.crawl()
            indexer.build(pages)
            INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            indexer.save(INDEX_PATH)
            index_loaded = True

        elif command == "load":
            try:
                indexer.load(INDEX_PATH)
                index_loaded = True
            except FileNotFoundError as e:
                print(f"Error: {e}")

        elif command in ("exit", "quit"):
            print("Goodbye!")
            break

        else:
            print(f"Unknown command: '{command}'. Commands: build, load, exit")


if __name__ == "__main__":
    main()