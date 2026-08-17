"""
Redis client – singleton per worker process.

Three roles in the project:

1. Embedding cache  – emb:{sha256(model:text)} → JSON vector, TTL 7 days
2. Query cache      – query:{sha256(query:params)} → JSON results, TTL 1 hour
3. Job progress     – job:{workflow_id} → status string, TTL 24 hours

decode_responses=False because binary JSON payloads are stored for embeddings.
"""

import hashlib
import json

import structlog
from redis.asyncio import Redis

log = structlog.get_logger(__name__)


class RedisClient:
    """
    Async Redis client manager.

    Provides lazy initialization and lifecycle management of a Redis connection.
    The Redis connection is created only when it is first requested and reused
    for subsequent calls.

    Attributes:
        url (str): Redis connection URL.
        redis (Redis | None): Internal Redis client instance. It is initialized
            lazily on the first call to `get()`.
    """

    def __init__(self, url: str):
        """
        Args:
            url (str): Redis connection URL used to create the Redis client.
                Example:
                    redis://localhost:6379/0
        """
        self.url = url
        self.redis: Redis = Redis.from_url(url, decode_responses=True)

    async def get(self) -> Redis:
        """
        Get the Redis client instance.

        Creates a new Redis client on the first call and returns the same
        instance for all subsequent calls.

        Returns:
            Redis: Initialized async Redis client.
        """
        if self.redis is None:
            self.redis = Redis.from_url(self.url, decode_responses=False)
        return self.redis

    async def close(self):
        """
        Close the Redis connection.

        Releases resources held by the Redis connection pool and resets the
        internal client reference.
        """
        if self.redis:
            await self.redis.aclose()

    @staticmethod
    def _generate_query_key(model_name: str, query: str) -> str:
        """
        Helper method for creating a consistent hash key for query embeddings.

        The query is normalized using .strip().lower() so that values like
        "Beograd " and "beograd" share the same cache entry.
        """
        raw_identifier = f"{model_name}:{query.strip().lower()}"
        hashed = hashlib.sha256(raw_identifier.encode("utf-8")).hexdigest()
        return f"qemb:{hashed}"

    async def get_query_embedding(
        self,
        model_name: str,
        query: str,
    ) -> list[float] | None:
        """Gets cached embedding vector for given query and model."""
        key = self._generate_query_key(model_name, query)
        cached_data = await self.redis.get(key)

        if cached_data:
            log.debug("query_embedding_cache_hit", query=query[:50])
            return json.loads(cached_data)

        log.debug("query_embedding_cache_miss", query=query[:50])
        return None

    async def set_query_embedding(
        self,
        model_name: str,
        query: str,
        embedding: list[float],
        ttl_seconds: int = 604800,  # Default cached 7 days (7 * 24 * 3600)
    ) -> None:
        """Stores embedding vector into cache with TTL expiration time."""
        key = self._generate_query_key(model_name, query)
        await self.redis.set(
            name=key,
            value=json.dumps(embedding),
            ex=ttl_seconds,  # ex = expiration in seconds
        )
