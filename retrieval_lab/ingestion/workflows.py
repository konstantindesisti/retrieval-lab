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
from datetime import datetime

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError