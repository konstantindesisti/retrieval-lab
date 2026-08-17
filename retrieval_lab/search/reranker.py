"""
Cross-encoder reranker.

Two-stage retrieval pipeline:

    Stage 1 – Bi-encoder (embedding): fast, O(1) per query, coarse filtering
        → returns top 20 candidates

    Stage 2 – Cross-encoder (reranker): slower, O(N) per query, precise ranking
        → scores each (query, chunk) pair together
        → returns top K re-ranked results

Why a cross-encoder is better than a bi-encoder for ranking:
    A bi-encoder embeds the query and document SEPARATELY.
    A cross-encoder sees the query AND document TOGETHER in a single forward pass.

    → It can model fine-grained interactions between query terms and content.
    → Typically achieves +5-15% NDCG improvement compared to a pure bi-encoder.

Model: `BAAI/bge-reranker-base` (via FastEmbed `TextCrossEncoder`)

    * ~280MB ONNX model weights, runs locally (no external API cost or network latency)
    * Multilingual support: excellent performance on Serbian/xBCS and English (unlike MS-MARCO)
    * Powered by FastEmbed & ONNX Runtime: C++ inference backend that bypasses PyTorch overhead
    * Highly optimized CPU latency: ~15-30ms for 20 candidate pairs (2-4x faster than PyTorch)
    * Superior ranking quality: state-of-the-art MRR/NDCG benchmark scores among base rerankers

Alternative: Cohere Rerank API (higher quality, pay-per-use, ~$1/1000 queries)
"""

import asyncio
from functools import lru_cache

import structlog
from fastembed.rerank.cross_encoder import TextCrossEncoder
from retrieval_lab.search.dto import ScoredChunk

log = structlog.get_logger()

RERANKER_MODEL = "BAAI/bge-reranker-base"


@lru_cache(maxsize=1)
def _get_reranker() -> TextCrossEncoder:
    """
    Lazy loads the FastEmbed ONNX Cross-Encoder model.

    `lru_cache(maxsize=1)` ensures that the ONNX model is loaded into memory
    only once and reused for all subsequent requests.
    """
    log.info("loading_reranker_model", model=RERANKER_MODEL)
    model = TextCrossEncoder(model_name=RERANKER_MODEL)
    log.info("reranker_model_loaded")
    return model


async def rerank(
    query: str,
    chunks: list[ScoredChunk],
    top_k: int = 5,
) -> list[ScoredChunk]:
    """
    Re-ranks chunks using the FastEmbed ONNX cross-encoder.

    FastEmbed `rerank()` is a synchronous CPU/ONNX operation.
    `asyncio.to_thread()` runs it in a separate thread to avoid blocking the
    FastAPI async event loop.
    """

    if not chunks:
        return []

    documents = [chunk.content for chunk in chunks]

    def _predict() -> list[float]:
        model = _get_reranker()
        # model.rerank returns  generator float scores for passed documents
        scores = list(model.rerank(query=query, documents=documents))
        return [float(s) for s in scores]

    log.info("reranking_started", candidates=len(chunks), top_k=top_k)

    scores: list[float] = await asyncio.to_thread(_predict)

    # Connectin original chunks with their new reranker scores
    scored = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    result = [
        ScoredChunk(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            score=round(score, 4),
            meta={
                **chunk.meta,
                "initial_score": chunk.score,
            },  # Save original hybrid score
            retriever="reranked",
        )
        for chunk, score in scored[:top_k]
    ]

    log.info(
        "reranking_complete",
        input=len(chunks),
        output=len(result),
        top_score=result[0].score if result else None,
    )

    return result
