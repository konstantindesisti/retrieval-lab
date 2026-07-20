"""
Temporal activities for persisting data to the database.

Two activities with clearly separated responsibilities:

save_article(scraped)   → upserts an Article and returns its `article_id` (int)
Called immediately after scraping, before chunking.
This ensures the Article already exists in the database
when indexing begins.

index_chunks(doc, id)   → deletes existing chunks and inserts new ones with
embeddings
Sets `Article.is_indexed = True` when indexing completes.

Why separate these activities:
If the embedding API fails, the workflow retries only from
`generate_embeddings`.

The Article remains in the database (`is_indexed=False`), so no data is lost.

The `is_indexed` flag always indicates which articles have been fully indexed.
"""
import structlog
from sqlalchemy import delete, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from temporalio import activity

from retrieval_lab.db.models import Article, Chunk
from retrieval_lab.db.connection import session_factory
from retrieval_lab.ingestion.activities.dto import EmbeddedDocument, ScrapedArticle

log = structlog.get_logger(__name__)


@activity.defn
async def save_article(article: ScrapedArticle) -> int:
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
    async with session_factory() as session:
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
                index_elements=['url'],
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

@activity.defn
async def index_chunks(doc: EmbeddedDocument, article_id: int) -> int:
    """
    Deletes existing chunks for the specified article and inserts the new ones.
    When indexing completes successfully, sets Article.is_indexed = True.

    Args:
        doc: EmbeddedDocument containing chunk text, embeddings, and metadata.
        article_id: ID of the Article to associate the chunks with.

    Returns:
        The number of chunks inserted into the database.
    """
    async with session_factory() as session:
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
