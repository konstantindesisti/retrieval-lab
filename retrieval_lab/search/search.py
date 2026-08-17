"""
POST /search endpoint module.

Request -> Response flow:
 1. Check Redis query cache (hash of query + params).
  2. Cache miss:
     a. embed_query()      -> float vector (uses FastEmbed via DI)
     b. vector / hybrid()  -> top candidates (using shared DB session)
     c. rerank()           -> (optional) re-ranks candidates to top K
     d. generate_answer()  -> LLM answer with source attribution (Ollama/OpenAI compatible)
  3. Write to Redis query cache (TTL 1h).
  4. Log to SearchLog table (async background task, non-blocking).
  5. Return SearchResponse.
SearchMode enum (defined in DTO):
  vector  - ANN cosine search only
  keyword - Full Text Search (FTS) only
  hybrid  - RRF(vector, keyword) -> Recommended
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends
from openai import AsyncOpenAI


from retrieval_lab.config import settings
from retrieval_lab.db.connection import session_factory
from retrieval_lab.db.models import SearchLog
from retrieval_lab.db.repository import keyword_search, vector_search
from retrieval_lab.search.dto import (
    ScoredChunk,
    SearchMode,
    SearchRequest,
    SearchResponse,
    SourceAttribution,
)
from retrieval_lab.search.hybrid import hybrid_search
from retrieval_lab.search.reranker import rerank
from retrieval_lab.search.vector import embed_query
from services.api.depencencies import get_db_session, get_embedder, get_redis_client

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from retrieval_lab.cache.redis import RedisClient
    from retrieval_lab.ingestion.embedding.base import BaseEmbedder
    from retrieval_lab.search.dto import SearchFilters


log = structlog.get_logger(__name__)

router = APIRouter(prefix="/search", tags=["search"])

QUERY_CACHE_TTL = 60 * 60  # 1 hour in seconds


def get_llm_client() -> AsyncOpenAI:
    """
    Creates an AsyncOpenAI client compatible with both OpenAI and Ollama.

    For Ollama usage, set the following in environment:
    LLM_BASE_URL="http://localhost:11434/v1"
    LLM_MODEL="llama3" (or any downloaded local model)
    LLM_API_KEY="ollama" (dummy value required by the client)
    """
    return AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key or "dummy-key-for-local-llm",
    )


def _query_cache_key(req: SearchRequest) -> str:
    """Generates a deterministic Redis cache key based on search parameters."""
    payload = f"{req.query}:{req.mode}:{req.reranked}:{req.top_k}:{req.filters}"
    return f"query:{hashlib.sha256(payload.encode()).hexdigest()}"


async def _generate_answer(
    query: str, chunks: list[ScoredChunk], llm_client: AsyncOpenAI
) -> str:
    """
    RAG: Constructs the final answer using an LLM + retrieved context.

    Accepts an injected `llm_client` (Ollama or OpenAI compatible).
    The prompt enforces strict source attribution based ONLY on retrieved chunks.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk.meta.get("title", "Unknown")
        url = chunk.meta.get("url", "")
        context_blocks.append(f"[{i}] {title}\n{chunk.content}\nSource: {url}")

    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are a gaming expert assistant. Answer the user's question using ONLY "
        "the provided context articles. Cite sources by number [1], [2], etc. "
        "If the context doesn't contain enough information, say so clearly. "
        "Keep your answer concise and factual."
    )

    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    response = await llm_client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,  # Lower temperature for deterministic and factual answers
        max_tokens=512,
    )

    return response.choices[0].message.content or ""


@router.post("", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session),
    redis_client: RedisClient = Depends(get_redis_client),
    embedder: BaseEmbedder = Depends(get_embedder),
    llm_client: AsyncOpenAI = Depends(get_llm_client),
) -> SearchResponse:
    start_time = time.perf_counter()

    # ── 1. Query Cache ─────────────────────────────────────────────────────────
    cache_key = _query_cache_key(req)
    cached = await redis_client.redis.get(cache_key)

    if cached:
        log.info("query_cache_hit", query=req.query[:50])
        response_data = json.loads(cached)
        return SearchResponse(**response_data)

    # ── 2. Retrieval ───────────────────────────────────────────────────────────
    filters = SearchFilters(**req.filters) if req.filters else SearchFilters()
    candidates: list[ScoredChunk] = []

    if req.mode == SearchMode.VECTOR:
        query_embedding = await embed_query(req.query, embedder, redis_client)
        candidates = await vector_search(
            query_embedding=query_embedding,
            limit=settings.vector_search_limit,
            filters=filters,
            session=session,
        )

    elif req.mode == SearchMode.KEYWORD:
        candidates = await keyword_search(
            query=req.query,
            limit=settings.vector_search_limit,
            filters=filters,
            session=session,
        )

    else:  # HYBRID (default)
        query_embedding = await embed_query(req.query, embedder, redis_client)
        candidates = await hybrid_search(
            session=session,
            query=req.query,
            query_embedding=query_embedding,
            limit=settings.vector_search_limit,
            filters=filters,
        )

    # ── 3. Reranking ───────────────────────────────────────────────────────────
    if req.reranked and candidates:
        final_chunks = await rerank(req.query, candidates, top_k=req.top_k)
    else:
        final_chunks = candidates[: req.top_k]

    # ── 4. LLM Answer ─────────────────────────────────────────────────────────
    answer = (
        await _generate_answer(req.query, final_chunks, llm_client)
        if final_chunks
        else ("No relevant articles found for your query.")
    )

    # ── 5. Prepare Response ────────────────────────────────────────────────────
    latency_ms = round((time.perf_counter() - start_time) * 1000, 1)

    sources = [
        SourceAttribution(
            title=chunk.meta.get("title", "Unknown"),
            url=chunk.meta.get("url", ""),
            score=round(chunk.score, 4),
        )
        for chunk in final_chunks
    ]

    mode_label = f"{req.mode}+rerank" if req.reranked else req.mode

    response = SearchResponse(
        answer=answer,
        sources=sources,
        mode=mode_label,
        reranked=req.reranked,
        latency_ms=latency_ms,
    )

    # ── 6. Cache Response ──────────────────────────────────────────────────────
    await redis_client.redis.set(
        cache_key, response.model_dump_json(), ex=QUERY_CACHE_TTL
    )

    # ── 7. Async Logging ───────────────────────────────────────────────────────
    background_tasks.add_task(_log_search, req, len(final_chunks), latency_ms, sources)

    log.info(
        "search_complete",
        query=req.query[:50],
        mode=mode_label,
        results=len(final_chunks),
        latency_ms=latency_ms,
    )

    return response


async def _log_search(
    req: SearchRequest,
    results_count: int,
    latency_ms: float,
    sources: list[SourceAttribution],
) -> None:
    """
    Fire-and-forget background logging to the SearchLog table.

    Must use its own session factory because the request's dependency session
    is closed immediately after the HTTP response is sent.
    """
    try:
        async with session_factory() as session:
            entry = SearchLog(
                query=req.query,
                mode=req.mode,
                reranked=req.reranked,
                top_k=req.top_k,
                results_count=results_count,
                latency_ms=latency_ms,
                results_snapshot=[s.model_dump() for s in sources],
            )
            session.add(entry)
            await session.commit()
    except Exception as e:
        log.warning("search_log_failed", error=str(e))
