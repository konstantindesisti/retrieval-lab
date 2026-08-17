"""
Hybrid search module using Reciprocal Rank Fusion (RRF).

RRF formula:
  score(doc) = Σ  1 / (k + rank_i(doc))
              per retriever i

  k = 60 (empirically optimal value from Cormack et al., 2009)

Why RRF outperforms simple score fusion:
  - Vector search scores (cosine distance) and keyword scores (ts_rank) are not on the same scale.
  - Normalization (min-max, z-score) is fragile and highly dependent on the database distribution.
  - RRF operates strictly on RANKS, not absolute scores -> extremely robust against varying score distributions.

Example (k=60):
  doc A: vector rank 1, keyword rank 3
    score = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226

  doc B: vector rank 2, keyword rank 1
    score = 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252  <- winner

  doc C: only in vector, rank 5
    score = 1/(60+5) + 0 = 0.01538  <- penalized for absence in keyword search

Documents appearing in BOTH retrievers receive a natural combinatorial boost.
This is the core intuition and power behind hybrid search.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import asyncio

import structlog

from retrieval_lab.search.dto import ScoredChunk, SearchFilters
from retrieval_lab.db.repository import keyword_search, vector_search

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
log = structlog.get_logger(__name__)

RRF_K = 60


def reciprocal_rank_fusion(
    *result_lists: list[ScoredChunk],
    k: int = RRF_K,
) -> list[ScoredChunk]:
    """
    Takes N lists of ranked results and returns a single fused list.

    Accepts 2+ lists: hybrid_search passes [vector_results, keyword_results],
    but it could easily accept a third retriever (e.g., BM25 from Elasticsearch)
    without changing the API.
    """
    # chunk_id -> accumulated RRF score
    rrf_scores: dict[int, float] = {}
    # chunk_id -> ScoredChunk (stored for final output)
    chunk_by_id: dict[int, ScoredChunk] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                k + rank
            )
            # Store the chunk (the last list "wins" in case of metadata duplicates)
            chunk_by_id[chunk.chunk_id] = chunk

    # Sort descending by the newly calculated RRF score
    fused = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    result = [
        ScoredChunk(
            chunk_id=cid,
            content=chunk_by_id[cid].content,
            score=rrf_scores[cid],
            meta=chunk_by_id[cid].meta,
            retriever="hybrid",
        )
        for cid in fused
    ]

    log.info(
        "rrf_fusion_complete",
        input_lists=len(result_lists),
        input_total=sum(len(r) for r in result_lists),
        output_unique=len(result),
    )

    return result


async def hybrid_search(
    session: AsyncSession,
    query: str,
    query_embedding: list[float],
    limit: int = 20,
    filters: SearchFilters | None = None,
) -> list[ScoredChunk]:
    """
    Executes vector and keyword searches concurrently using a shared database session,
    then fuses their results using Reciprocal Rank Fusion (RRF).

    Args:
        session: Active SQLAlchemy AsyncSession.
        query: Raw search query text for keyword search.
        query_embedding: Pre-calculated vector embedding for vector search.
        limit: Maximum number of results to return per retriever before fusion.
        filters: Optional search criteria (source, genre, platform, year).

    Returns:
        List of fused and re-ranked ScoredChunk objects.
    """
    vector_results, keyword_results = await asyncio.gather(
        vector_search(
            session=session,
            query_embedding=query_embedding,
            limit=limit,
            filters=filters,
        ),
        keyword_search(session=session, query=query, limit=limit, filters=filters),
    )

    log.info(
        "hybrid_search_inputs",
        vector_count=len(vector_results),
        keyword_count=len(keyword_results),
    )

    return reciprocal_rank_fusion(vector_results, keyword_results)
