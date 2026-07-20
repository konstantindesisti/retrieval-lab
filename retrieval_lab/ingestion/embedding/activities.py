from __future__ import annotations
from typing import TYPE_CHECKING

from temporalio import activity


from retrieval_lab.config import settings
from retrieval_lab.ingestion.activities.dto import EmbeddedDocument, EmbeddedChunk, ChunkedDocument
from retrieval_lab.ingestion.embedding.factory import get_embedder
if TYPE_CHECKING:
    from retrieval_lab.cache.client import RedisClient


class EmbeddingActivities:
    def __init__(self, redis_client: RedisClient) -> None:
        self.redis_client = redis_client

    @activity.defn
    async def generate_embeddings(
            self,
            doc: ChunkedDocument,
            article_body: str,
    ) -> EmbeddedDocument:
        """
        Converts a ChunkedDocument into an EmbeddedDocument by generating
        embeddings for all chunks.

        The original article body is passed separately because the chunker does not
        retain the full text, while the indexer requires it for the Article.body
        column.

        Args:
            doc: ChunkedDocument containing the text chunks to embed.
            article_body: Original article text to be stored in the database.

        Returns:
            An EmbeddedDocument containing the generated embedding vectors.
        """
        # 1. Extract content from the chunks
        contents = [chunk.content for chunk in doc.chunks]

        # 2. Initialize embedder
        embedder = get_embedder(redis_client=self.redis_client)

        embeddings = await embedder.embed(contents)

        embedded_chunks = [
            EmbeddedChunk(
                content=chunk.content,
                embedding=vec,
                chunk_index=chunk.chunk_index,
                total_chunks=chunk.total_chunks,
                meta=chunk.meta,
            )
            for chunk, vec in zip(doc.chunks, embeddings)
        ]

        return EmbeddedDocument(
            article_url=doc.article_url,
            article_title=doc.article_title,
            article_body=article_body,
            source=doc.source,
            chunks=embedded_chunks,
            strategy=doc.strategy,
            embedding_provider=embedder.provider_name,
            embedding_model=embedder.model_name,
        )