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
from datetime import datetime

import feedparser
import httpx
import structlog
from bs4 import BeautifulSoup
from temporalio import activity

from retrieval_lab.core.exceptions import InvalidFeedFormatError
from retrieval_lab.ingestion.activities.dto import RSSEntry, ScrapedArticle

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
    for tag in container.find_all(["script", "style", "aside", "nav",
                                   "figure", "iframe", "noscript"]):
        tag.decompose()

    body = container.get_text(separator='\n', strip=True)
    body = _clean_text(body)

    # Try to pull tags/categories
    tags_el = soup.find_all('a', {'data-cy': "BreadcrumbLink"})
    if tags_el:
        meta['tags'] = [t.get_text(strip=True) for t in tags_el]

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


# ============== TEMPORAL ACTIVITIES ==============
@activity.defn
async def fetch_rss_urls(feed_url: str, limit: int = 20) -> list[RSSEntry]:
    """
    Parses an RSS/Atom feed and returns a list of RSSEntry objects.

    feedparser is synchronous, but it is acceptable for an
    activity because it does not block the Temporal event loop.

    Args:
        feed_url: URL of the RSS/Atom feed to parse.
        limit: Maximum number of entries to return.

    Returns:
        A list of RSSEntry objects containing information about feed items.
    """
    log.info(f"Fetching {feed_url}, limit: {limit}")

    feed = feedparser.parse(feed_url)

    if feed.bozo and not feed.entries:
        error_msg = str(feed.get("bozo_exception", "Malformed XML structure"))
        raise InvalidFeedFormatError(msg=error_msg)

    entries: list[RSSEntry] = []
    for entry in feed.entries[:limit]:
        url = entry.get("link", '')
        title = entry.get("title", 'Untitled')

        published_at = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published_at = datetime(*entry.published_parsed[:6])

        entries.append(RSSEntry(
            url=url,
            title=title,
            published_at=published_at,
        ))

    log.info(f"Fetched {len(entries)} entries")
    return entries


@activity.defn
async def scrape_article(rss_entry: RSSEntry) -> ScrapedArticle | None:
    """
    Fetches the full article page and extracts its clean text content.
    Returns None if the article cannot be fetched or parsed.

    Args:
        rss_entry: RSS feed entry containing the article URL and metadata.
    Returns:
        A ScrapedArticle object on success, or None if extraction fails.
    """
    url = rss_entry.url
    log.info(f'Scraping article {url}')

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; lore-bot/1.0; "
            "+https://github.com/konstantindesisti/retrieval-lab)"
        )
    }

    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            log.warning(f'Scrape HTTP error for url {url}: {e.response.status_code}')
            return None

        except httpx.RequestError as e:
            log.warning(f'Scrape request error for url {url}: {e}')
            return None

    soup = BeautifulSoup(response.text, "html.parser")

    title = rss_entry.title

    extractor = _get_extractor(url)
    body, meta = extractor(soup, url)

    if not body or len(body) < 200:
        log.warning(f'Scrape empty body error for url {url}, body length: {len(body)}')
        return None

    source = _detect_source(url)
    log.info(f'Scrape successful for url {url}, source: {source}, body length: {len(body)}')

    return ScrapedArticle(
        url=url,
        title=title,
        body=body,
        source=source,
        meta=meta,
    )