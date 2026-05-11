"""
main.py - CLI with all four commands: build, load, print, find
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
    print("Commands: build, load, print <word>, find <query>, exit")

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

        if command == "build":
            print(f"Crawling {BASE_URL} (politeness window: 6s)...")
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

        elif command == "print":
            if not index_loaded:
                print("No index loaded. Run 'build' or 'load' first.")
            elif not argument:
                print("Usage: print <word>")
            else:
                print(indexer.print_postings(argument.strip().lower()))

        elif command == "find":
            if not index_loaded:
                print("No index loaded. Run 'build' or 'load' first.")
            elif not argument:
                print("Usage: find <word> [word ...]")
            else:
                results = indexer.find(argument)
                if not results:
                    print(f"No pages found for '{argument}'.")
                else:
                    print(f"Found {len(results)} page(s) for '{argument}':")
                    for url, score in sorted(results, key=lambda x: x[1], reverse=True):
                        title = indexer._page_titles.get(url, url)
                        print(f"  {title}\n  URL: {url}\n  Score: {score}")

        elif command in ("exit", "quit"):
            print("Goodbye!")
            break

        else:
            print(f"Unknown command: '{command}'.")


if __name__ == "__main__":
    main()