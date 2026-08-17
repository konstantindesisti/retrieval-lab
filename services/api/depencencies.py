"""
FastAPI dependency providers for Database sessions, Redis clients, and Embedders.
"""

from typing import AsyncGenerator
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from retrieval_lab.cache.redis import RedisClient
from retrieval_lab.db.connection import session_factory
from retrieval_lab.ingestion.embedding.base import BaseEmbedder


# 1. Database Session Dependency
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session from session_factory.
    Reuses existing error handling and rollback logic.
    """
    async with session_factory() as session:
        yield session


# 2. Redis Client Dependency
async def get_redis_client(request: Request) -> RedisClient:
    """
    Retrieves the global RedisClient singleton instance stored in FastAPI app state.
    """
    return request.app.state.redis_client


# 3. Embedder Dependency
def get_embedder(request: Request) -> BaseEmbedder:
    """
    Retrieves the global BaseEmbedder singleton instance stored in FastAPI app state.
    """
    return request.app.state.embedder
