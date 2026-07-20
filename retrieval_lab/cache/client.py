"""
Redis client – singleton per worker process.

Three roles in the project:

1. Embedding cache  – emb:{sha256(model:text)} → JSON vector, TTL 7 days
2. Query cache      – query:{sha256(query:params)} → JSON results, TTL 1 hour
3. Job progress     – job:{workflow_id} → status string, TTL 24 hours

decode_responses=False because binary JSON payloads are stored for embeddings.
"""
from redis.asyncio import Redis

class RedisClient:
    """
    Async Redis client manager.

    Provides lazy initialization and lifecycle management of a Redis connection.
    The Redis connection is created only when it is first requested and reused
    for subsequent calls.

    Attributes:
        url (str): Redis connection URL.
        _client (Redis | None): Internal Redis client instance. It is initialized
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
        self._client: Redis | None = None

    async def get(self) -> Redis:
        """
        Get the Redis client instance.

        Creates a new Redis client on the first call and returns the same
        instance for all subsequent calls.

        Returns:
            Redis: Initialized async Redis client.
        """
        if self._client is None:
            self._client = Redis.from_url(self.url, decode_responses=False)

        assert self._client is not None
        return self._client


    async def close(self):
        """
        Close the Redis connection.

        Releases resources held by the Redis connection pool and resets the
        internal client reference.
        """
        if self._client:
            await self._client.aclose()


