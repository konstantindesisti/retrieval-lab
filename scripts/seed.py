"""
Initial seed script – triggers IngestFeedWorkflow for IGN RSS feeds.

Usage:
  python scripts/seed.py               # default feeds, limit 20
  python scripts/seed.py --limit 50    # more articles
  python scripts/seed.py --wait        # wait for completion and print results

Run this once after running `docker-compose up` to populate the database.
Make sure the worker service is running (refer to docker-compose.yml).
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from temporalio.client import Client

from retrieval_lab.config import settings
from retrieval_lab.ingestion.workflows import IngestFeedWorkflow

IGN_FEEDS = [
    "https://feeds.feedburner.com/ign/all-articles",
    "https://feeds.feedburner.com/ign/reviews",
]


async def main(limit: int, wait: bool) -> None:
    print(f"Connecting to Temporal at {settings.temporal_host}...")
    client = await Client.connect(settings.temporal_host)

    for feed_url in IGN_FEEDS:
        wf_id = f"seed-{feed_url.split('/')[-1]}"

        handle = await client.start_workflow(
            IngestFeedWorkflow.run,
            args=[feed_url, limit],
            id=wf_id,
            task_queue=settings.temporal_task_queue,
        )

        print(f"  Started: {wf_id}  (feed: {feed_url})")

        if wait:
            print("  Waiting for completion...")
            result = await handle.result()
            print(f"  ✓  {result['succeeded']}/{result['total']} articles indexed")
            if result.get("errors"):
                for err in result["errors"]:
                    print(f"  ✗  {err['url'][:60]}  → {err['error'][:60]}")

    if not wait:
        print()
        print("Workflows started. Track progress at http://localhost:8080 (Temporal UI)")
        print("Or poll: GET http://localhost:8000/ingest/{workflow_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed lore database from IGN RSS feeds")
    parser.add_argument("--limit", type=int, default=20, help="Articles per feed")
    parser.add_argument("--wait", action="store_true", help="Wait for workflow completion")
    args = parser.parse_args()

    asyncio.run(main(limit=args.limit, wait=args.wait))