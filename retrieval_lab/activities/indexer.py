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

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, AsyncContextManager

import structlog
from temporalio import activity

from retrieval_lab.db.repository import save_article, index_chunks

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from retrieval_lab.ingestion.dto import EmbeddedDocument, ScrapedArticle

log = structlog.get_logger(__name__)


class StorageActivities:
    """
    Temporal activities responsible for persisting and indexing data in the database.

    This class manages database connections and performs transactional operations
    for storing articles and their corresponding vector records.
    """

    def __init__(
        self, session_factory: Callable[[], AsyncContextManager[AsyncSession]]
    ):
        """
        Initializes StorageActivities with a database session factory.

        Args:
        session_factory: SQLAlchemy async session factory used to create
        new database transactions during activity execution.
        """
        self.session_factory = session_factory

    @activity.defn
    async def save_article(
        self,
        article: ScrapedArticle,
    ) -> int:
        """Stores a scraped article in the database and returns its assigned ID.

        Args:
            article: Scraped article containing title, URL, body, and metadata.

        Returns:
            int: The unique database ID of the newly saved article.

        Raises:
            Exception: Any database error triggers a transaction rollback and
                is re-raised so Temporal can retry the activity.
        """
        async with self.session_factory() as session:
            return await save_article(session=session, article=article)

    @activity.defn
    async def index_chunks(
        self,
        doc: EmbeddedDocument,
        article_id: int,
    ) -> int:
        """Indexes vector chunks for a specific article in the database.

        Args:
            doc: Object containing the generated embeddings and text chunks.
            article_id: The database ID of the parent article to link chunks with.

        Returns:
            int: The total count of successfully indexed vector chunks.

        Raises:
            Exception: Any database error triggers a transaction rollback and
                is re-raised so Temporal can retry the activity.
        """
        async with self.session_factory() as session:
            return await index_chunks(
                session=session,
                doc=doc,
                article_id=article_id,
            )
