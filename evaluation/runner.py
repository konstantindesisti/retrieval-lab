"""
Evaluation runner – runs all search modes against the golden dataset
and prints a comparative summary table of retrieval metrics.

Usage:
  python scripts/run_eval.py [--k 5] [--verbose]

Example Output:
  ══════════════════════════════════════════════════════════
   LORE – Retrieval Evaluation  |  k=5  |  20 queries
  ══════════════════════════════════════════════════════════
   Mode                    R@5    P@5    MRR    NDCG@5
  ──────────────────────────────────────────────────────────
   vector                  0.612  0.240  0.534  0.621
   keyword                 0.445  0.175  0.389  0.452
   hybrid                  0.734  0.290  0.667  0.751
   hybrid+rerank           0.812  0.320  0.778  0.834   ← BEST
  ══════════════════════════════════════════════════════════
   Δ rerank vs vector     +0.200 +0.080 +0.244 +0.213

This output serves as a clear benchmark demonstrating the quantitative
improvements introduced by each layer of the search pipeline.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import json
from pathlib import Path
from typing import Any

import structlog

from evaluation.metrics import EvalMetrics, QueryResult, compute_metrics
from retrieval_lab.cache.redis import RedisClient
from retrieval_lab.config import settings
from retrieval_lab.db.connection import session_factory
from retrieval_lab.ingestion.embedding.fastembed_impl import FastEmbedder

from retrieval_lab.search.hybrid import hybrid_search
from retrieval_lab.db.repository import keyword_search, vector_search
from retrieval_lab.search.reranker import rerank
from retrieval_lab.search.vector import embed_query

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

log = structlog.get_logger(__name__)


GOLDEN_DATASET_PATH = Path(__file__).parent / "test_queries.json"
VECTOR_CANDIDATE_LIMIT = 20  # Number of candidates retrieved before reranking

class EvalRunner:
    """
    Encapsulates the evaluation pipeline, managing shared resources
    like the Embedder and Redis client across different search modes.
    """

    def __init__(
        self, embedder: Any | None = None, redis_client: RedisClient | None = None
    ) -> None:
        # Initialize heavy singletons or accept them via dependency injection (for testing)
        self.embedder = embedder if embedder is not None else FastEmbedder(model_name=settings.embedding_providers.fastembed.model_name)
        self.redis_client = (
            redis_client
            if redis_client is not None
            else RedisClient(url=settings.redis.url)
        )

    async def _run_vector(
        self, session: AsyncSession, query: str, k: int
    ) -> list[str]:
        """Executes pure vector search."""
        embedding = await embed_query(
            query, embedder=self.embedder, redis_client=self.redis_client
        )  # pass Embedder if function supports
        chunks = await vector_search(session=session, query_embedding=embedding, limit=k)
        return [c.meta.get("url", "") for c in chunks]

    async def _run_keyword(
        self, session: AsyncSession, query: str, k: int
    ) -> list[str]:
        """Executes full-text keyword search."""
        chunks = await keyword_search(session=session, query=query, limit=k)
        return [c.meta.get("url", "") for c in chunks]

    async def _run_hybrid(
        self, session: AsyncSession, query: str, k: int
    ) -> list[str]:
        """Executes hybrid search (vector + keyword fusion)."""
        embedding = await embed_query(query=query, embedder=self.embedder, redis_client=self.redis_client)
        chunks = await hybrid_search(
            session, query, embedding, limit=VECTOR_CANDIDATE_LIMIT
        )
        return [c.meta.get("url", "") for c in chunks[:k]]

    async def _run_hybrid_rerank(
        self, session: AsyncSession, query: str, k: int
    ) -> list[str]:
        """Executes hybrid search followed by cross-encoder reranking."""
        embedding = await embed_query(query, embedder=self.embedder, redis_client=self.redis_client)
        candidates = await hybrid_search(
            session, query, embedding, limit=VECTOR_CANDIDATE_LIMIT
        )
        reranked = await rerank(query, candidates, top_k=k)
        return [c.meta.get("url", "") for c in reranked]

    def get_modes(self):
        """Returns a mapping of search mode names to their respective handler methods."""
        return {
            "vector": self._run_vector,
            "keyword": self._run_keyword,
            "hybrid": self._run_hybrid,
            "hybrid+rerank": self._run_hybrid_rerank,
        }

    async def run(self, k: int = 5) -> list[EvalMetrics]:
        """
        Runs the full evaluation suite across all defined search modes.
        """
        dataset = self._load_golden_dataset()
        if not dataset:
            print("❌ No labeled queries found in test_queries.json.")
            return []

        all_metrics: list[EvalMetrics] = []
        modes = self.get_modes()

        # Open a single database session for the entire evaluation run
        async with session_factory() as session:
            for mode_name, search_fn in modes.items():
                log.info("eval_mode_started", mode=mode_name, queries=len(dataset))
                query_results: list[QueryResult] = []

                for item in dataset:
                    query = item["query"]
                    relevant_urls = set(item["relevant_urls"])

                    try:
                        retrieved_urls = await search_fn(session, query, k)
                    except Exception as e:
                        log.warning(
                            "eval_query_failed",
                            query=query[:50],
                            mode=mode_name,
                            error=str(e),
                        )
                        retrieved_urls = []

                    query_results.append(
                        QueryResult(
                            query=query,
                            relevant_urls=relevant_urls,
                            retrieved_urls=retrieved_urls,
                        )
                    )

                metrics = compute_metrics(mode_name, query_results, k)
                all_metrics.append(metrics)
                log.info(
                    "eval_mode_complete",
                    mode=mode_name,
                    ndcg=round(metrics.ndcg, 3),
                )

        # Ensure redis connections are properly cleaned up after eval run
        await self.redis_client.close()
        return all_metrics

    @staticmethod
    def _load_golden_dataset(
        path: Path = GOLDEN_DATASET_PATH,
    ) -> list[dict[str, Any]]:
        """Loads and validates the golden dataset from `test_queries.json`."""
        if not path.exists():
            return []
        with open(path, mode="r", encoding="utf-8") as f:
            data = json.load(f)

        labeled = [q for q in data if q.get("relevant_urls")]
        skipped = len(data) - len(labeled)
        if skipped:
            log.warning("eval_unlabeled_queries_skipped", count=skipped)

        log.info("golden_dataset_loaded", total=len(data), labeled=len(labeled))
        return labeled


def print_results(all_metrics: list[EvalMetrics], verbose: bool = False) -> None:
    """Prints a formatted evaluation table with metrics and comparative deltas."""
    if not all_metrics:
        return

    k = all_metrics[0].k
    n = all_metrics[0].num_queries
    width = 62

    print()
    print("═" * width)
    print(f"  LORE – Retrieval Evaluation  |  k={k}  |  {n} queries")
    print("═" * width)
    print(f"  {'Mode':<22} {'R@'+str(k):<8} {'P@'+str(k):<8} {'MRR':<8} {'NDCG@'+str(k):<8}")
    print("─" * width)

    best_ndcg = max(m.ndcg for m in all_metrics)

    for m in all_metrics:
        marker = "   ← BEST" if abs(m.ndcg - best_ndcg) < 1e-6 else ""
        print(
            f"  {m.mode:<22} "
            f"{m.recall:.3f}    "
            f"{m.precision:.3f}    "
            f"{m.mrr:.3f}    "
            f"{m.ndcg:.3f}"
            f"{marker}"
        )

    print("═" * width)

    # Calculate and display improvement delta: hybrid+rerank vs pure vector
    if len(all_metrics) >= 4:
        base = all_metrics[0]   # pure vector
        best = all_metrics[-1]  # hybrid+rerank

        def diff(b: EvalMetrics, a: EvalMetrics, attr: str) -> str:
            val_b = getattr(b, attr)
            val_a = getattr(a, attr)
            return f"{val_b - val_a:+.3f}"

        print(
            f"  {'Δ rerank vs vector':<22} "
            f"{diff(best, base, 'recall')}    "
            f"{diff(best, base, 'precision')}    "
            f"{diff(best, base, 'mrr')}    "
            f"{diff(best, base, 'ndcg')}"
        )
        print("═" * width)

    # Detailed per-query breakdown if verbose flag is enabled
    if verbose:
        for m in all_metrics:
            print(f"\n  [{m.mode}] Per-query breakdown:")
            for pq in m.per_query:
                hits_str = f"{pq['hits']}/{pq['relevant_count']}"
                print(
                    f"    {pq['query'][:45]:<46} "
                    f"hits={hits_str:<5} "
                    f"RR={pq['rr']:.2f}  "
                    f"NDCG={pq['ndcg']:.3f}"
                )

    print()
