from __future__ import annotations

from temporalio import activity

from retrieval_lab.ingestion.chunker import (
    log,
    _fixed_size_chunk,
    _sliding_window_chunk,
    _build_chunk_data,
)
from retrieval_lab.ingestion.dto import ScrapedArticle, ChunkedDocument


@activity.defn
async def chunk_document(
    article: ScrapedArticle,
    strategy: str = "fixed",
    chunk_size: int = 512,
    overlap: int = 64,
) -> ChunkedDocument:
    """
    Splits a ScrapedArticle into chunks and returns a ChunkedDocument.

    Args:
        article: ScrapedArticle object containing the article content
            and metadata.
        strategy: Chunking strategy to use. Supported values:
            "fixed" | "sliding".
        chunk_size: Number of words per chunk.
        overlap: Number of overlapping words between consecutive chunks.

    Temporal passes all parameters from the workflow, making it easy
    to change the strategy without redeploying workers (only the workflow
    call needs to be updated).

    Returns:
        A ChunkedDocument containing the generated chunks and metadata.
    """
    log.info(
        "chunking_document",
        url=article.url,
        strategy=strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        body_len=len(article.body),
    )

    if strategy == "fixed":
        raw_chunks = _fixed_size_chunk(article.body, chunk_size, overlap)
    elif strategy == "sliding":
        step = chunk_size - overlap
        raw_chunks = _sliding_window_chunk(article.body, chunk_size, step)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")

    # Ukloni prekratke chunkove (ostaci od headers/nav teksta)
    raw_chunks = [c for c in raw_chunks if len(c.split()) >= 20]

    chunk_data = _build_chunk_data(raw_chunks, article, strategy)

    log.info(
        "chunking_complete",
        url=article.url,
        strategy=strategy,
        num_chunks=len(chunk_data),
    )

    return ChunkedDocument(
        article_url=article.url,
        article_title=article.title,
        source=article.source,
        chunks=chunk_data,
        strategy=strategy,
    )
