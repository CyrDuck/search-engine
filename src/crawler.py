"""
crawler.py - Basic Web Crawler
XJCO3011 Coursework 2
"""

import time
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup


@dataclass
class PageData:
    url: str
    title: str
    text: str
    links: list[str] = field(default_factory=list)


class Crawler:
    def __init__(self, base_url="https://quotes.toscrape.com/", politeness_window=6.0):
        self.base_url = base_url
        self.politeness_window = politeness_window
        self._visited = set()

    def _is_same_domain(self, url):
        return urlparse(url).netloc == urlparse(self.base_url).netloc

    def _fetch(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def _parse_page(self, url, html):
        soup = BeautifulSoup(html, "html.parser")
        title = soup.find("title")
        title = title.get_text(strip=True) if title else url
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        links = []
        for a in soup.find_all("a", href=True):
            absolute = urljoin(url, a["href"])
            if self._is_same_domain(absolute) and absolute not in self._visited:
                links.append(absolute)
        return PageData(url=url, title=title, text=text, links=links)

    def crawl(self):
        pages = []
        queue = [self.base_url]
        self._visited.add(self.base_url)

        while queue:
            url = queue.pop(0)
            print(f"Crawling: {url}")
            response = self._fetch(url)
            if response is None:
                continue
            page = self._parse_page(url, response.text)
            pages.append(page)
            for link in page.links:
                if link not in self._visited:
                    self._visited.add(link)
                    queue.append(link)
            if queue:
                time.sleep(self.politeness_window)

        print(f"Done. {len(pages)} pages crawled.")
        return pages