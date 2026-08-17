"""
Temporal activities for scraping the IGN RSS feed and fetching full articles.

Two activities:
  fetch_rss_urls(feed_url)  → list of new article URLs
  scrape_article(url)       → ScrapedArticle (title, body, meta)

The BeautifulSoup selectors are specific to IGN's HTML structure.
To support a new source, add a new extractor and register it in
SOURCE_EXTRACTORS.
"""

import re

import structlog
from bs4 import BeautifulSoup

log = structlog.get_logger(__name__)


def _extract_ign(soup: BeautifulSoup, url: str) -> tuple[str, dict]:
    """
    Returns (body_text, meta_dict) for an IGN article.

    IGN stores the article content inside the <article> element,
    specifically within the div.article-page container.
    """
    meta: dict = {}

    # Attempt article-page div, fallback on tag
    container = soup.find("div", class_="article")
    if not container:
        container = soup.find("article")
    if not container:
        container = soup.find("main")

    if not container:
        return "", meta

    # Strip unwanted elements
    for tag in container.find_all(
        ["script", "style", "aside", "nav", "figure", "iframe", "noscript"]
    ):
        tag.decompose()

    body = container.get_text(separator="\n", strip=True)
    body = _clean_text(body)

    # Try to pull tags/categories
    tags_el = soup.find_all("a", {"data-cy": "BreadcrumbLink"})
    if tags_el:
        meta["tags"] = [t.get_text(strip=True) for t in tags_el]

    return body, meta


def _extract_generic(soup: BeautifulSoup, url: str) -> tuple[str, dict]:
    """Generic fallback extractor for most news websites.
    Attempts to extract content using standard HTML5 semantic elements."""
    for selector in [
        "article",
        "main",
        '[role="main"]',
        ".post-content",
        ".entry-content",
        ".content",
    ]:
        container = soup.select_one(selector)
        if container:
            for tag in container.find_all(["script", "style", "aside", "nav"]):
                tag.decompose()
            return _clean_text(container.get_text(separator="\n", strip=True)), {}

    # Last resort – whole body
    return _clean_text(soup.get_text(separator="\n", strip=True)), {}


SOURCE_EXTRACTORS = {
    "ign.com": _extract_ign,
}


def _get_extractor(url: str):
    for domain, fn in SOURCE_EXTRACTORS.items():
        if domain in url:
            return fn
    return _extract_generic


def _clean_text(text: str) -> str:
    """Uklanja visak whitespace-a i praznih linija."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    # Spoji vise uzastopnih praznih linija u jednu
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return cleaned.strip()


def _detect_source(url: str) -> str:
    if "ign.com" in url:
        return "ign"
    if "rockpapershotgun.com" in url:
        return "rock_paper_shotgun"
    return "unknown"
