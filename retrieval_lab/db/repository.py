from __future__ import annotations
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import delete, update, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from retrieval_lab.db.models import Article, Chunk
from retrieval_lab.search.dto import ScoredChunk, SearchFilters

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from retrieval_lab.ingestion.dto import ScrapedArticle, EmbeddedDocument


log = structlog.get_logger(__name__)


# ============== PUT ==============
async def save_article(*, session: AsyncSession, article: ScrapedArticle) -> int:
    """
    Upserts an Article into the database and returns its article_id.

    Uses ON CONFLICT (url) DO UPDATE to:

    Update the title and body if the article has changed.
    Reset is_indexed to False so the workflow re-indexes the article.
    Preserve the original scraped_at timestamp from the initial scrape.

    Args:
        article: ScrapedArticle containing the article content and metadata.

    Returns:
        The ID of the inserted or updated Article.
    """

    stmt = (
        pg_insert(Article)
        .values(
            url=article.url,
            title=article.title,
            body=article.body,
            source=article.source,
            is_indexed=False,
            meta=article.meta or {},
        )
        .on_conflict_do_update(
            index_elements=["url"],
            set_={
                "title": pg_insert(Article).excluded.title,
                "body": pg_insert(Article).excluded.body,
                "is_indexed": False,
                "meta": pg_insert(Article).excluded.meta,
            },
        )
        .returning(Article.id)
    )

    result = await session.execute(stmt)
    article_id: int = result.scalar_one()
    await session.commit()

    log.info(f"Article saved, url: [{article.url}], article_id: [{article_id}] ")
    return article_id


async def index_chunks(
    *, session: AsyncSession, doc: EmbeddedDocument, article_id: int
) -> int:
    """
    Deletes existing chunks for the specified article and inserts the new ones.
    When indexing completes successfully, sets Article.is_indexed = True.

    Args:
        doc: EmbeddedDocument containing chunk text, embeddings, and metadata.
        article_id: ID of the Article to associate the chunks with.

    Returns:
        The number of chunks inserted into the database.
    """

    # 1. Removing old chunks
    # Clear existing chunks before insertion to support re-indexing when the
    # chunking strategy or embedding model changes.
    del_result = await session.execute(
        delete(Chunk).where(Chunk.article_id == article_id)
    )
    deleted_count = del_result.rowcount

    if deleted_count:
        log.info("chunks_deleted", article_id=article_id, count=deleted_count)

    # 2. Bulk insert new chunks
    # Use core insert (not ORM) for better performance on bulk write
    chunk_rows = [
        {
            "article_id": article_id,
            "content": chunk.content,
            "embedding": chunk.embedding,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "embedding_model": doc.embedding_model,
            "meta": chunk.meta,
        }
        for chunk in doc.chunks
    ]

    await session.execute(pg_insert(Chunk), chunk_rows)

    # 3. Mark Article as indexed
    await session.execute(
        update(Article).where(Article.id == article_id).values(is_indexed=True)
    )

    await session.commit()

    log.info(
        "chunks_indexed",
        article_id=article_id,
        url=doc.article_url,
        count=len(doc.chunks),
        model=doc.embedding_model,
        strategy=doc.strategy,
    )

    return len(doc.chunks)


# ============== GET ==============
async def keyword_search(
    *,
    session: AsyncSession,
    query: str,
    limit: int = 20,
    filters: SearchFilters | None = None,
):
    """
    Performs PostgreSQL Full-Text Search (FTS) using `ts_rank` for scoring.

    `ts_rank` returns a floating-point score, but its distribution is not
    uniform—scores rarely exceed 0.3 for typical queries.

    This is not an issue because `hybrid.py` does not compare raw scores.
    Instead, it combines results using Reciprocal Rank Fusion (RRF), which
    operates on ranks rather than scores.

    Args:
    query: Search query in natural language.
    limit: Maximum number of matching chunks to return.

    Returns:
    A list of ScoredChunk objects ranked by `ts_rank`.
    """
    f = filters or SearchFilters()

    where_clauses = [
        "to_tsvector('english', c.content) @@ plainto_tsquery('english', :query)"
    ]
    params: dict = {"query": query, "limit": limit}

    if f.source:
        where_clauses.append("c.meta->>'source' = :source")
        params["source"] = f.source

    if f.genre:
        where_clauses.append("c.meta->'genre' ? :genre")
        params["genre"] = f.genre

    if f.platform:
        where_clauses.append("c.meta->'platform' ? :platform")
        params["platform"] = f.platform

    if f.year:
        where_clauses.append("(c.meta->>'year')::int = :year")
        params["year"] = f.year

    where_sql = " AND ".join(where_clauses)

    sql = text(f"""
        SELECT
            c.id      AS chunk_id,
            c.content,
            c.meta,
            ts_rank(
                to_tsvector('english', c.content),
                plainto_tsquery('english', :query)
            ) AS score
        FROM chunks c
        WHERE {where_sql}
        ORDER BY score DESC
        LIMIT :limit
    """)
    result = await session.execute(sql, params=params)
    rows = result.mappings().all()

    chunks = [
        ScoredChunk(
            chunk_id=row["chunk_id"],
            content=row["content"],
            score=float(row["score"]),
            meta=row["meta"] or {},
            retriever="keyword",
        )
        for row in rows
    ]

    log.info("keyword_search_complete", query=query[:50], results=len(chunks))
    return chunks


async def vector_search(
    *,
    session: AsyncSession,
    query_embedding: list[float],
    limit: int = 20,
    filters: SearchFilters | None = None,
) -> list[ScoredChunk]:
    """
    Executes an Approximate Nearest Neighbor (ANN) search over the pgvector index.

    Uses raw SQL because the SQLAlchemy ORM lacks native support for pgvector operators.
    The text() construct is used safely with bind parameters to prevent SQL injection.

    Metadata filtering operates on PostgreSQL JSONB columns:
      meta->>'source' = 'ign'
      meta->'genre' ? 'RPG'       (JSONB array contains)
    PostgreSQL can efficiently index JSONB structures for rapid filtering.
    """
    f = filters or SearchFilters()

    # Dynamically build WHERE clauses based on provided filters
    where_clauses = ["c.embedding IS NOT NULL"]
    params: dict = {"embedding": str(query_embedding), "limit": limit}

    if f.source:
        where_clauses.append("c.meta->>'source' = :source")
        params["source"] = f.source

    if f.genre:
        where_clauses.append("c.meta->'genre' ? :genre")
        params["genre"] = f.genre

    if f.platform:
        where_clauses.append("c.meta->'platform' ? :platform")
        params["platform"] = f.platform

    if f.year:
        where_clauses.append("(c.meta->>'year')::int = :year")
        params["year"] = f.year

    where_sql = " AND ".join(where_clauses)

    sql = text("""
        SELECT
            c.id         AS chunk_id,
            c.content,
            c.meta,
            1 - (c.embedding <=> (:embedding)::vector) AS score
        FROM chunks c
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> (:embedding)::vector
        LIMIT :limit
    """)

    result = await session.execute(sql, params)
    rows = result.mappings().all()

    chunks = [
        ScoredChunk(
            chunk_id=row["chunk_id"],
            content=row["content"],
            score=float(row["score"]),
            meta=row["meta"] or {},
            retriever="vector",
        )
        for row in rows
    ]

    log.info("vector_search_complete", results=len(chunks), limit=limit)
    return chunks
