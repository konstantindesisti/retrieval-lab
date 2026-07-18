from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from retrieval_lab.config.settings import settings
from retrieval_lab.core.exceptions import DBError

async_engine = create_async_engine(settings.DB_URL, pool_pre_ping=True, echo=False)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

@asynccontextmanager
async def session_factory() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except SQLAlchemyError as e:
            await session.rollback()
            raise DBError(message="Session error") from e
