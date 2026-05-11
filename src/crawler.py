"""
crawler.py - Web Crawler for Search Engine Tool
XJCO3011 Coursework 2

Crawls https://quotes.toscrape.com/ with a politeness window of 6 seconds.
Collects page content, metadata, and links for indexing.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PageData:
    """Represents the crawled data for a single web page."""
    url: str
    title: str
    text: str
    links: list[str] = field(default_factory=list)
    status_code: int = 200


class Crawler:
    """
    Web crawler that respects a politeness window between requests.

    Attributes:
        base_url: The starting URL for the crawl.
        politeness_window: Minimum seconds between HTTP requests (default 6).
        timeout: HTTP request timeout in seconds.
        max_retries: Number of retry attempts for failed requests.
    """

    def __init__(
        self,
        base_url: str = "https://quotes.toscrape.com/",
        politeness_window: float = 6.0,
        timeout: int = 15,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url
        self.politeness_window = politeness_window
        self.timeout = timeout
        self.max_retries = max_retries
        self._visited: set[str] = set()
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": "XJCO3011-SearchBot/1.0 (educational project)"}
        )

    def _normalise_url(self, url: str) -> str:
        """Strip fragments and trailing slashes for deduplication."""
        parsed = urlparse(url)
        normalised = parsed._replace(fragment="").geturl()
        return normalised.rstrip("/")

    def _is_same_domain(self, url: str) -> bool:
        """Return True if *url* belongs to the same domain as base_url."""
        base_host = urlparse(self.base_url).netloc
        target_host = urlparse(url).netloc
        return base_host == target_host

    def _fetch(self, url: str) -> Optional[requests.Response]:
        """
        Fetch *url* with retry logic.

        Returns the Response on success, or None after exhausting retries.
        Time complexity: O(max_retries) per URL call.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as exc:
                logger.warning("HTTP %s for %s (attempt %d)", exc.response.status_code, url, attempt)
                if exc.response.status_code in (403, 404, 410):
                    return None  # Do not retry permanent errors
            except requests.exceptions.ConnectionError:
                logger.warning("Connection error for %s (attempt %d/%d)", url, attempt, self.max_retries)
            except requests.exceptions.Timeout:
                logger.warning("Timeout for %s (attempt %d/%d)", url, attempt, self.max_retries)
            except requests.exceptions.RequestException as exc:
                logger.error("Request failed for %s: %s", url, exc)
                return None

            if attempt < self.max_retries:
                backoff = self.politeness_window * attempt
                logger.info("Backing off %.1f s before retry...", backoff)
                time.sleep(backoff)

        return None

    def _parse_page(self, url: str, html: str) -> PageData:
        """
        Extract title, visible text, and internal links from *html*.

        Text extraction strips script/style nodes so only human-readable
        content is indexed — this mirrors what a user would see in a browser.
        Time complexity: O(n) where n = number of DOM nodes.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url

        # Visible text: remove script, style, and nav boilerplate
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)

        # Internal links only
        links: list[str] = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            absolute = urljoin(url, href)
            normalised = self._normalise_url(absolute)
            if self._is_same_domain(normalised) and normalised not in self._visited:
                links.append(normalised)

        return PageData(url=url, title=title, text=text, links=links)

    def crawl(self) -> list[PageData]:
        """
        BFS crawl starting from base_url.

        Respects the politeness window between every HTTP request.
        Returns a list of PageData objects, one per successfully crawled page.

        Time complexity: O(P × (N + E)) where P = politeness_window,
        N = number of pages, E = number of links processed.
        Space complexity: O(N + E) for the visited set and queue.
        """
        pages: list[PageData] = []
        queue: list[str] = [self._normalise_url(self.base_url)]
        self._visited.add(queue[0])

        while queue:
            url = queue.pop(0)
            logger.info("Crawling: %s", url)

            response = self._fetch(url)
            if response is None:
                logger.warning("Skipping %s (failed to fetch)", url)
                time.sleep(self.politeness_window)
                continue

            page = self._parse_page(url, response.text)
            pages.append(page)

            # Enqueue newly discovered links
            for link in page.links:
                norm = self._normalise_url(link)
                if norm not in self._visited:
                    self._visited.add(norm)
                    queue.append(norm)

            logger.info(
                "  → found %d new links | total crawled: %d",
                len(page.links),
                len(pages),
            )

            # Politeness: always sleep before the next request
            if queue:
                logger.debug("Sleeping %.1f s (politeness window)...", self.politeness_window)
                time.sleep(self.politeness_window)

        logger.info("Crawl complete. %d pages collected.", len(pages))
        return pages

    @property
    def visited_urls(self) -> set[str]:
        """Return a copy of all URLs visited during the crawl."""
        return set(self._visited)
