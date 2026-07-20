"""
Temporal activity for generating embeddings with Redis caching.

Flow for each chunk:
  1. cache_key = sha256(model + ":" + content)
  2. Redis GET → cache hit: deserialize and return immediately
  3. Redis MISS → collect into a batch and call the OpenAI API
  4. Store results in Redis (pipeline, single round-trip)
  5. Return EmbeddedDocument

Why use Redis pipeline for cache writes:
  N chunks → N SET commands → one TCP round-trip instead of N.

Note about Temporal payload size:
  1536 floats * ~17 chars (JSON) * 50 chunks ≈ 1.3MB per workflow step.
  Temporal's default limit is 4MB, so this is acceptable for typical articles.

  For larger documents (books, documentation), consider having the embedder
  write embeddings directly to the database and return only chunk IDs.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any
import hashlib
import json
from abc import ABC, abstractmethod

import structlog
from temporalio import activity

from retrieval_lab.config import settings
from retrieval_lab.ingestion.activities.chunker import ChunkedDocument
from retrieval_lab.ingestion.activities.dto import EmbeddedChunk, EmbeddedDocument

if TYPE_CHECKING:
    from retrieval_lab.cache.client import RedisClient

log = structlog.get_logger(__name__)


class BaseEmbedder(ABC):
    def __init__(self, **kwargs: Any):
        super().__init__()

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the embedding model (used for cache key)"""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the embedding provider (e.g., 'fastembed', 'ollama', 'openai')"""
        pass

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Receives a list of texts and returns a list of embedding vectors."""
        pass

class CachedEmbedder(BaseEmbedder):
    def __init__(self, base_embedder: BaseEmbedder, redis_client: RedisClient, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.base_embedder = base_embedder
        self.redis = redis_client
        self.ttl = settings.redis.embedding_cache_ttl

    @property
    def model_name(self) -> str:
        return self.base_embedder.model_name

    @property
    def provider_name(self) -> str:
        return self.base_embedder.provider_name
    
    def _cache_key(self, text: str) -> str:
        """
        Generates a deterministic cache key: the same text + the same model always
        produce the same key.

        If the model changes, the keys will be different → automatic cache invalidation.

        Args:
            text: Input text used to generate the embedding.

        Returns:
            A unique cache key based on the model and text hash.
        """
        digest = hashlib.sha256(f"{self.model_name}:{text}".encode()).hexdigest()
        return f"emb:{digest}"
    
    async def _batch_cache_get(
            self, 
            keys: list[str]
    ) -> list[list[float] | None]:
        """
        Batch Redis GET using a pipeline for all keys – a single round-trip.

        Args:
            keys: List of Redis cache keys to retrieve.

        Returns:
            A list of cached embedding vectors. Returns None for keys that
            are missing from the cache.
        """
        redis = await self.redis.get()

        async with redis.pipeline(transaction=False) as pipe:
            for key in keys:
                await pipe.get(key)
            results = await pipe.execute()

        return [json.loads(r) if r is not None else None for r in results]
    
    async def _batch_cache_set(
            self, 
            key_vector_pairs: list[tuple[str, list[float]]],
    ) -> None:
        """
        Batch Redis SET using a pipeline for multiple embeddings in a single
        round-trip.

        Args:
            key_vector_pairs: List of (cache_key, embedding_vector) tuples to
            store in the cache.

        Returns:
            None.
        """
        redis = await self.redis.get()

        async with redis.pipeline(transaction=False) as pipe:
            for key, vec in key_vector_pairs:
                await pipe.set(key, json.dumps(vec), ex=self.ttl)
            await pipe.execute()


    async def embed(
            self, 
            texts: list[str]
    ) -> list[list[float]]:
        # 1. Prepare keys
        if not texts:
            return []

        # 1. Generate keys for all texts
        keys = [self._cache_key(text) for text in texts]

        # 2. Pull all from Redis at once (1 round-trip)
        cached_embeddings = await self._batch_cache_get(keys)

        # 3. Find missing
        missing_indexes = [i for i, emb in enumerate(cached_embeddings) if emb is None]

        # 4. If something is missing, call embedder only for those texts
        if missing_indexes:
            missing_texts = [texts[i] for i in missing_indexes]

            new_embeddings = await self.base_embedder.embed(missing_texts)

            # 5. Save new vector into Redis
            pairs_to_cache = [
                (keys[i], vec)
                for i, vec in zip(missing_indexes, new_embeddings)
            ]
            await self._batch_cache_set(pairs_to_cache)

            # 6. Return new vectors back into original
            new_vec_iter = iter(new_embeddings)
            for i in missing_indexes:
                cached_embeddings[i] = next(new_vec_iter)

        # 7. Now when all places are filled in original arrangement
        return cached_embeddings
