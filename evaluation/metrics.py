"""
Retrieval evaluation metrics – pure functions, no external dependencies.

Each core function accepts:
  relevant  : set[str]  – A set of relevant URLs for a given query
  retrieved : list[str] – Ranked URLs returned by the search system
  k         : int       – Cutoff (only evaluate the top-k results)

Metrics:

  Recall@k
    How many of the relevant documents were found in the top-k?
    → "How much of the desired content did we cover?"
    Formula: |relevant ∩ retrieved[:k]| / |relevant|
    Range: 0.0 (none) – 1.0 (all found)

  Precision@k
    How many of the top-k retrieved results are actually relevant?
    → "How accurate are the returned results?"
    Formula: |relevant ∩ retrieved[:k]| / k
    Range: 0.0 – 1.0

  MRR – Mean Reciprocal Rank
    Where is the FIRST relevant document located?
    → "How fast does the user get a relevant result?"
    Formula: 1 / rank_of_first_relevant (0 if none found)
    MRR is the average of Reciprocal Ranks across all queries.
    Range: 0.0 – 1.0

  NDCG@k – Normalized Discounted Cumulative Gain
    Accounts for both rank AND position – highly ranked relevant docs
    yield more points than lower-ranked ones.
    → The most comprehensive metric, standard in IR literature.
    Formula: DCG@k / IDCG@k
      DCG@k  = Σ rel_i / log2(i+1)  for i=1..k
      IDCG@k = DCG when ALL relevant docs are at the top (ideal ranking)
    Range: 0.0 – 1.0

Why multiple metrics:
  - Recall@k + Precision@k together show the Recall-Precision tradeoff.
  - MRR is excellent for use cases where the user only needs ONE good result.
  - NDCG is excellent for use cases where the user scans a ranked list.
  - Combining them provides a complete picture of system quality.
"""

import math
from dataclasses import dataclass, field
from typing import Any


# ── Per-query Metrics ──────────────────────────────────────────────────────────

def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Proportion of relevant documents found in the top-k results."""
    if not relevant:
        return 0.0
    hits = len(relevant & set(retrieved[:k]))
    return hits / len(relevant)


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Proportion of the top-k results that are actually relevant."""
    if k == 0:
        return 0.0
    hits = sum(1 for url in retrieved[:k] if url in relevant)
    return hits / k


def reciprocal_rank(relevant: set[str], retrieved: list[str]) -> float:
    """
    1 / position of the first relevant result.
    Returns 0.0 if there are no relevant results in the list.
    """
    for i, url in enumerate(retrieved, start=1):
        if url in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k.

    Uses binary relevance (relevant=1, non-relevant=0).
    For graded relevance (e.g., 0/1/2), expand to use a relevance dictionary.
    """
    def _dcg(items: list[str], cutoff: int) -> float:
        score = 0.0
        for i, url in enumerate(items[:cutoff], start=1):
            if url in relevant:
                score += 1.0 / math.log2(i + 1)
        return score

    actual_dcg = _dcg(retrieved, k)

    # Ideal ordering: all relevant documents placed at the very top
    ideal_retrieved = list(relevant) + ["__pad__"] * k
    ideal_dcg = _dcg(ideal_retrieved, k)

    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg


# ── Aggregated Metrics (across a set of queries) ───────────────────────────────

@dataclass
class QueryResult:
    """The search result of a single query for a specific search mode."""
    query: str
    relevant_urls: set[str]
    retrieved_urls: list[str]  # ranked, top-k

    @property
    def k(self) -> int:
        return len(self.retrieved_urls)


@dataclass
class EvalMetrics:
    """
    Aggregated metrics for the entire evaluation set under a single search mode.
    All values are macro-averages (each query contributes equally).
    """
    mode: str
    k: int
    num_queries: int

    recall: float = 0.0
    precision: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0

    # Per-query breakdown for detailed debugging
    per_query: list[dict[str, Any]] = field(default_factory=list)

    def summary_row(self) -> str:
        """Returns a formatted string summary of the metrics."""
        return (
            f"  {self.mode:<22} "
            f"R@{self.k}={self.recall:.3f}  "
            f"P@{self.k}={self.precision:.3f}  "
            f"MRR={self.mrr:.3f}  "
            f"NDCG@{self.k}={self.ndcg:.3f}"
        )


def compute_metrics(
    mode: str,
    results: list[QueryResult],
    k: int,
) -> EvalMetrics:
    """
    Calculates aggregated metrics for a given list of QueryResult objects.

    Macro-average: Each query carries an equal weight coefficient
    (unlike a micro-average which would favor queries with more relevant documents).
    """
    if not results:
        return EvalMetrics(mode=mode, k=k, num_queries=0)

    recalls, precisions, rrs, ndcgs = [], [], [], []
    per_query = []

    for qr in results:
        r = recall_at_k(qr.relevant_urls, qr.retrieved_urls, k)
        p = precision_at_k(qr.relevant_urls, qr.retrieved_urls, k)
        rr = reciprocal_rank(qr.relevant_urls, qr.retrieved_urls)
        n = ndcg_at_k(qr.relevant_urls, qr.retrieved_urls, k)

        recalls.append(r)
        precisions.append(p)
        rrs.append(rr)
        ndcgs.append(n)

        per_query.append({
            "query": qr.query,
            "recall": round(r, 4),
            "precision": round(p, 4),
            "rr": round(rr, 4),
            "ndcg": round(n, 4),
            "relevant_count": len(qr.relevant_urls),
            "retrieved_count": len(qr.retrieved_urls),
            "hits": len(qr.relevant_urls & set(qr.retrieved_urls[:k])),
        })

    num_results = len(results)
    return EvalMetrics(
        mode=mode,
        k=k,
        num_queries=num_results,
        recall=sum(recalls) / num_results,
        precision=sum(precisions) / num_results,
        mrr=sum(rrs) / num_results,
        ndcg=sum(ndcgs) / num_results,
        per_query=per_query,
    )