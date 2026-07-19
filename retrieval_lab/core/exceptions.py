from __future__ import annotations

class AppError(Exception):
    """Base exception for application errors."""

    def __init__(self, *, msg: str, stage: str = "GENERAL") -> None:
        self.msg = msg
        self.stage = stage
        super().__init__(self.msg)

# =====================================================================
# 1. DATABASE & VECTOR STORE (Stage: DATABASE)
# =====================================================================

class DBError(AppError):
    """Base exception for database errors."""
    def __init__(self, *, msg: str) -> None:
        super().__init__(msg=msg, stage="DATABASE")

class DBConnectionError(DBError):
    """Raised when a connection to the database cannot be established."""
    def __init__(self, *, msg: str) -> None:
        super().__init__(msg=msg)

class VectorStoreError(DBError):
    """Raised when pgvector index creation, upsert, or similarity search fails."""
    def __init__(self, *, msg: str) -> None:
        super().__init__(msg=msg)

# =====================================================================
# 2. CACHE & REDIS (Stage: CACHE)
# =====================================================================

class CacheError(AppError):
    """Base exception for caching and Redis-related errors."""
    def __init__(self, *, msg: str) -> None:
        super().__init__(msg=msg, stage="CACHE")

class CacheConnectionError(CacheError):
    """Raised when the connection to the Redis server fails."""
    def __init__(self, *, msg: str) -> None:
        super().__init__(msg=msg)

class CacheOperationError(CacheError):
    """Raised when caching operations (get, set, invalidation) fail."""
    def __init__(self, *, msg: str) -> None:
        super().__init__(msg=msg)

# =====================================================================
# 3. INGESTION & PIPELINE (Stage: INGESTION)
# =====================================================================

class IngestionError(AppError):
    """Base exception for data ingestion pipeline errors."""
    def __init__(self, *, msg: str) -> None:
        super().__init__(msg=msg, stage="INGESTION")

class ScraperError(IngestionError):
    """Raised when scraping external sources or parsing RSS feeds fails."""
    def __init__(self, *, msg: str) -> None:
        super().__init__(msg=msg)

class InvalidFeedFormatError(ScraperError):
    """Raised when the RSS/Atom feed content is malformed or not XML."""
    def __init__(self, *, msg: str) -> None:
        super().__init__(msg=msg)

class RateLimitError(ScraperError):
    """Raised when external APIs (e.g., IGDB, OpenAI) return rate limit (HTTP 429)."""
    def __init__(self, *, msg: str) -> None:
        super().__init__(msg=msg)

class ChunkingError(IngestionError):
    """Raised when text-splitting or semantic chunking strategies fail."""
    def __init__(self, *, msg: str) -> None:
        super().__init__(msg=msg)

class EmbeddingError(IngestionError):
    """Raised when generating vector embeddings via the API fails."""
    def __init__(self, *, msg: str) -> None:
        super().__init__(msg=msg)

# =====================================================================
# 4. SEARCH & RETRIEVAL (Stage: SEARCH)
# =====================================================================

class SearchError(AppError):
    """Base exception for search, query processing, and retrieval errors."""
    def __init__(self, *, msg: str) -> None:
        super().__init__(msg=msg, stage="SEARCH")

class InvalidQueryError(SearchError):
    """Raised when the user's search query is invalid or syntactically malformed."""
    def __init__(self, *, msg: str) -> None:
        super().__init__(msg=msg)

class RerankingError(SearchError):
    """Raised when the Cross-Encoder reranking process fails."""
    def __init__(self, *, msg: str) -> None:
        super().__init__(msg=msg)

# =====================================================================
# 5. WORKFLOW & TEMPORAL (Stage: WORKFLOW)
# =====================================================================

class WorkflowError(AppError):
    """Base exception for Temporal orchestration and worker errors."""
    def __init__(self, *, msg: str) -> None:
        super().__init__(msg=msg, stage="WORKFLOW")

class ActivityExecutionError(WorkflowError):
    """Raised when a specific Temporal activity fails inside a workflow."""
    def __init__(self, *, msg: str) -> None:
        super().__init__(msg=msg)
