"""
Vector search – pgvector cosine similarity (ANN).

Query flow:
    1. embed_query(text)        → float vector (with Redis cache)
    2. vector_search(embedding) → top-N ScoredChunk results

The `<=>` operator represents cosine distance
(0 = identical, 2 = opposite).

The distance is converted into a score:
score = 1 - distance

This makes the value more intuitive:
1 = perfect match.

Index that should be created in a migration (not automatically):

    CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

    HNSW vs IVFFlat:
        HNSW     – better recall, consistently fast, higher memory usage (~2x)
        IVFFlat  – lower memory usage, recall decreases in edge cases

For a project of this scale, HNSW is the recommended choice.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import structlog


if TYPE_CHECKING:
    from retrieval_lab.ingestion.embedding.base import BaseEmbedder
    from retrieval_lab.cache.redis import RedisClient
log = structlog.get_logger()

QUERY_EMBEDDING_TTL = 60 * 60 * 24  # 24h – querys are repeating


async def embed_query(
    query: str,
    embedder: BaseEmbedder,
    redis_client: RedisClient,
) -> list[float]:
    """
    Embeds the query text using the provided embedder, backed by a Redis cache.

    By depending on the BaseEmbedder interface, this function remains agnostic
    to the underlying provider (FastEmbed, Ollama, OpenAI, etc.).

    Args:
        query (str): The raw text query from the user to be vectorized.
        embedder (BaseEmbedder): An implementation of the embedding provider interface
            responsible for generating vector embeddings.
        redis_client (RedisClient): Async Redis wrapper used to read and store
            cached query embeddings.

    Returns:
        list[float]: A list of floating-point numbers representing the vector
            embedding of the query.

    Raises:
        RedisError: If communication with the Redis cache fails (handled upstream).
        EmbeddingProviderError: If the underlying embedder fails to generate vectors.
    """
    model_name = embedder.model_name

    # 1. Look up in Redis cache first (Read-Through Cache Pattern)
    cached_vector = await redis_client.get_query_embedding(
        model_name=model_name, query=query
    )

    if cached_vector is not None:
        return cached_vector

    # 2. Compute embedding using the injected provider
    # The embed() method expects a list of strings and returns a list of vectors
    embeddings = await embedder.embed([query])
    vector = embeddings[0]

    log.debug(
        "query_embedding_generated", query=query[:50], provider=embedder.provider_name
    )

    # 3. Store the new vector in cache (e.g., TTL of 24 hours)
    await redis_client.set_query_embedding(
        model_name=model_name,
        query=query,
        embedding=vector,
        ttl_seconds=86400,
    )

    return vector
