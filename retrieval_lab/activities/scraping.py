from datetime import datetime

import feedparser
import httpx
from bs4 import BeautifulSoup
from temporalio import activity

from retrieval_lab.core.exceptions import InvalidFeedFormatError
from retrieval_lab.ingestion.dto import RSSEntry, ScrapedArticle
from retrieval_lab.ingestion.scraper import log, _get_extractor, _detect_source


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
        url = entry.get("link", "")
        title = entry.get("title", "Untitled")

        published_at = None
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            published_at = datetime(*entry.published_parsed[:6])

        entries.append(
            RSSEntry(
                url=url,
                title=title,
                published_at=published_at,
            )
        )

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
    log.info(f"Scraping article {url}")

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
            log.warning(f"Scrape HTTP error for url {url}: {e.response.status_code}")
            return None

        except httpx.RequestError as e:
            log.warning(f"Scrape request error for url {url}: {e}")
            return None

    soup = BeautifulSoup(response.text, "html.parser")

    title = rss_entry.title

    extractor = _get_extractor(url)
    body, meta = extractor(soup, url)

    if not body or len(body) < 200:
        log.warning(f"Scrape empty body error for url {url}, body length: {len(body)}")
        return None

    source = _detect_source(url)
    log.info(
        f"Scrape successful for url {url}, source: {source}, body length: {len(body)}"
    )

    return ScrapedArticle(
        url=url,
        title=title,
        body=body,
        source=source,
        meta=meta,
    )
