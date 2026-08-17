"""
Temporal workflows for the ingestion pipeline.

Three workflows:

IngestArticleWorkflow   – processes a single article from URL to pgvector
IngestFeedWorkflow      – fetches RSS and starts a child workflow per article
ScheduledPollerWorkflow – long-running workflow that periodically polls RSS feeds

Execution order in IngestArticleWorkflow:
    fetch_rss_urls ──► [per entry]
                            │
                        scrape_article        ← HTTP, retry 3x
                            │
                        save_article          ← DB upsert, retry 5x
                            │
                        chunk_document        ← CPU, retry 3x
                            │
                        generate_embeddings   ← OpenAI + Redis cache, retry 5x
                            │
                        index_chunks          ← DB bulk insert, retry 5x

Status tracking:
Temporal automatically stores the status of each workflow(RUNNING/COMPLETED/FAILED).
The API can query it using:client.get_workflow_handle(wf_id).describe()
There is no need for manual Redis progress tracking.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

from retrieval_lab.ingestion.embedding.retry_policies import DB_RETRY, HTTP_RETRY

with workflow.unsafe.imports_passed_through():
    from retrieval_lab.config import settings
    from retrieval_lab.activities.scraping import scrape_article, fetch_rss_urls
    from retrieval_lab.activities.chunking import chunk_document
    from retrieval_lab.activities.indexer import StorageActivities
    from retrieval_lab.activities.embedding import EmbeddingActivities

if TYPE_CHECKING:
    from retrieval_lab.ingestion.dto import RSSEntry, ScrapedArticle


@workflow.defn
class IngestArticleWorkflow:
    """
    Processes a single RSS entry – from scraping to indexed chunks in pgvector.

    Workflow ID: ingest-article-{url_hash}

    → Temporal rejects duplicate workflow IDs, providing automatic deduplication.
    The same article will not be indexed twice if it appears again in a later
    feed run.

    If an activity fails after the maximum number of retries, the workflow enters
    the FAILED state. IngestFeedWorkflow catches the error and continues processing
    the remaining articles.
    """

    @workflow.run
    async def run(self, rss_entry: RSSEntry) -> dict:

        # 1. Scrape
        scraped_article: ScrapedArticle | None = await workflow.execute_activity(
            scrape_article,
            rss_entry,
            start_to_close_timeout=timedelta(minutes=30),
            retry_policy=HTTP_RETRY,
        )

        if scraped_article is None:
            workflow.logger.warning(f"Scraped article [{rss_entry}] failed")
            return {
                "status": "skipped",
                "url": rss_entry.url,
                "reason": "empty_content",
            }

        # 2. Save Article into DB before chunking
        # If the embedding API fails after this point, the Article remains with is_indexed=False.
        # The workflow retries only from generate_embeddings – no data is lost.

        article_id: int = await workflow.execute_activity(
            StorageActivities.save_article.__name__,
            args=[scraped_article],
            start_to_close_timeout=timedelta(minutes=15),
            retry_policy=DB_RETRY,
        )

        # 3. Chunking
        chunked = await workflow.execute_activity(
            chunk_document,
            args=[
                scraped_article,
                settings.chunker.strategy,
                settings.chunker.size,
                settings.chunker.overlap,
            ],
            start_to_close_timeout=timedelta(minutes=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        if not chunked.chunks:
            workflow.logger.warning(f"No chunks produced for url [{rss_entry.url}].")
            return {
                "status": "skipped",
                "url": rss_entry.url,
                "article_id": article_id,
                "reason": "no_chunks_produced",
            }

        # 4. Embedding (Embedder + Redis cache)
        embedded = await workflow.execute_activity(
            EmbeddingActivities.generate_embeddings,
            args=[chunked, scraped_article.body],
            start_to_close_timeout=timedelta(minutes=30),
            retry_policy=DB_RETRY,
        )

        # 5. Index into pgvector
        chunk_count: int = await workflow.execute_activity(
            StorageActivities.index_chunks.__name__,
            args=[embedded, article_id],
            start_to_close_timeout=timedelta(minutes=30),
            retry_policy=DB_RETRY,
        )

        workflow.logger.info(
            "article_ingested",
            url=rss_entry.url,
            article_id=article_id,
            chunks=chunk_count,
        )

        return {
            "status": "success",
            "url": rss_entry.url,
            "article_id": article_id,
            "chunks": chunk_count,
            "strategy": embedded.strategy,
            "model": embedded.embedding_model,
        }


@workflow.defn
class IngestFeedWorkflow:
    """
    Fetches an RSS feed and starts an IngestArticleWorkflow for each article.

    Child workflows are executed sequentially to avoid hammering IGN and the
    OpenAI API with too many simultaneous requests.

    For parallel processing (once rate limits are well understood):

    -> tasks = [workflow.execute_child_workflow(...) for entry in entries]
    -> results = await asyncio.gather(*tasks, return_exceptions=True)

    Args:
        feed_url: URL of the RSS/Atom feed to fetch.
        workflow_id: Unique identifier for this feed ingestion workflow.

    Returns:
        None.
    """

    @workflow.run
    async def run(self, feed_url: str, limit: int = 20) -> dict:
        workflow.logger.info("feed_ingestion_started", feed_url=feed_url, limit=limit)

        # Fetch RSS
        entries: list[RSSEntry] = await workflow.execute_activity(
            fetch_rss_urls.__name__,
            args=[feed_url, limit],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=HTTP_RETRY,
        )

        results = []
        errors = []

        for entry in entries:
            # Deterministic workflow ID = automatic deduplication
            wf_id = f"ingest-article-{hash(entry.url) & 0xFFFFFFFF}"

            try:
                result = await workflow.execute_child_workflow(
                    IngestArticleWorkflow.run,
                    entry,
                    id=wf_id,
                    # If exists (previous run), Teporal return
                    # existing result instead of throwing an error
                )
                results.append(result)

            except ActivityError as e:
                # Jedan clanak nije usao – loguj i nastavi sa ostalima
                workflow.logger.warning(
                    "article_ingestion_failed",
                    url=entry.url,
                    error=str(e),
                )
                errors.append({"url": entry.url, "error": str(e)})

        summary = {
            "feed_url": feed_url,
            "total": len(entries),
            "succeeded": len(results),
            "failed": len(errors),
            "errors": errors,
        }

        workflow.logger.info("feed_ingestion_complete", **summary)
        return summary


@workflow.defn
class ScheduledPollerWorkflow:
    """
    Long-running workflow that periodically triggers IngestFeedWorkflow.
    A classic Temporal pattern: using `workflow.sleep()` instead of cron jobs.

    Advantages over cron:
    * Temporal keeps the workflow history, allowing you to see when each run
      occurred and what happened.
    * If the server goes down during sleep, Temporal resumes execution from the
      same point.
    * The interval can be changed without restarting the service.

    Starting (once, during deployment):
        await client.start_workflow(
            ScheduledPollerWorkflow.run,
            args=[IGN_FEEDS, 6],
            id="scheduled-poller",
            task_queue=settings.temporal_task_queue
        )
    """

    @workflow.run
    async def run(self, feed_urls: list[str], interval_hours: int = 6) -> None:
        while True:
            workflow.logger.info(
                "poller_cycle_started",
                feeds=len(feed_urls),
                interval_hours=interval_hours,
            )

            for feed_url in feed_urls:
                await workflow.execute_child_workflow(
                    IngestFeedWorkflow.run,
                    args=[feed_url, 20],
                    id=f"feed-{hash(feed_url) & 0xFFFFFFFF}-{workflow.now().timestamp():.0f}",
                )

            workflow.logger.info(
                "poller_cycle_complete",
                next_run_hours=interval_hours,
            )

            await workflow.sleep(timedelta(hours=interval_hours))
