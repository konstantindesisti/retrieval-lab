"""
Temporal worker entrypoint.

Registers all Workflows and Activities, then blocks waiting for tasks from the
task queue.

Runs as a separate Docker service (see docker-compose.yml).

Task queue "lore-ingestion" is a logical group:

* Workflow server sends tasks to this queue.
* Workers receive and execute them.
* Multiple worker instances can listen to the same queue, enabling
  horizontal scaling.
"""

import asyncio

import structlog
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.exceptions import WorkflowAlreadyStartedError

from retrieval_lab.config import settings
from retrieval_lab.db.connection import session_factory
from retrieval_lab.cache.redis import RedisClient

from retrieval_lab.activities.chunking import chunk_document
from retrieval_lab.activities.embedding import EmbeddingActivities
from retrieval_lab.activities.indexer import StorageActivities
from retrieval_lab.activities.scraping import fetch_rss_urls, scrape_article
from retrieval_lab.ingestion.workflows import (
    IngestArticleWorkflow,
    IngestFeedWorkflow,
    ScheduledPollerWorkflow,
)

log = structlog.get_logger(__name__)


IGN_FEEDS = [
    "https://feeds.feedburner.com/ign/all-articles",
    "https://feeds.feedburner.com/ign/reviews",
]


async def main() -> None:
    log.info("worker_starting", temporal_host=settings.temporal.host)

    redis_client = RedisClient(url=settings.redis.url)
    storage_acts = StorageActivities(session_factory=session_factory)
    embedding_acts = EmbeddingActivities(redis_client=redis_client)

    temporal_client = await Client.connect(
        settings.temporal.host,
        namespace=settings.temporal.namespace,
    )

    async with Worker(
        temporal_client,
        task_queue=settings.temporal.task_queue,
        workflows=[
            IngestArticleWorkflow,
            IngestFeedWorkflow,
            ScheduledPollerWorkflow,
        ],
        activities=[
            fetch_rss_urls,
            scrape_article,
            storage_acts.save_article,
            chunk_document,
            embedding_acts.generate_embeddings,
            storage_acts.index_chunks,
        ],
    ):
        log.info(
            "worker_ready",
            task_queue=settings.temporal.task_queue,
            workflows=3,
            activities=6,
        )
        # Start the scheduled poller on the first worker startup.
        # The workflow_id is fixed – Temporal ignores duplicate starts,
        # which means restarting the worker will not create another poller.
        try:
            await temporal_client.start_workflow(
                ScheduledPollerWorkflow.run,
                args=[IGN_FEEDS, 6],
                id="scheduled-poller",
                task_queue=settings.temporal.task_queue,
            )
            log.info("scheduled_poller_started")
        except WorkflowAlreadyStartedError:
            log.info("scheduled_poller_already_running")

            # Blocks until workflow turns off
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
