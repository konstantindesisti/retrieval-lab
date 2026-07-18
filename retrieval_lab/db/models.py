"""
SQLAlchemy 2.0 models for the Lore project.

Tables:
  Article    – scraped article (one per URL)
  Chunk      – article chunk with a pgvector embedding
  SearchLog  – log of every search query (for evaluation and analytics)

The Chunk.meta JSONB column is essential for metadata filtering during search:
  {
    "title": "Elden Ring Review",
    "source": "ign",
    "game": "Elden Ring",
    "genre": ["RPG", "Action"],
    "platform": ["PC", "PS5"],
    "year": 2022
  }
"""
from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from retrieval_lab.db.base import Base


class Article(Base):
    """
    Represents a raw scraped document from external sources (e.g., IGN, Rock Paper Shotgun).

    This table acts as the source of truth for raw text. It stores global metadata,
    source tracking, and serves as the parent record for split text chunks.

    Key Features for Production:
      - `url` is unique and indexed to prevent duplicate scraping during ingestion.
      - `is_indexed` acts as a state flag used by Temporal workflows to coordinate
        scraping, chunking, and embedding processes.
      - `meta` stores rich, raw domain-specific metadata (e.g., IGDB API integration).
    """
    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String(2048), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)

    # "ign" | "rock_paper_shotgun" | ...
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # False dok Temporal workflow nije zavrsio ingestion
    is_indexed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Slobodan prostor za game tags, genre, platform i sl. koji dolaze iz IGDB API-ja
    meta: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    scraped_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk", back_populates="article", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Article id={self.id} source={self.source!r} title={self.title[:40]!r}>"
        )


class Chunk(Base):
    """
    Represents an atomic fragment of an Article, processed and prepared for Vector Search.

    Key Features for Search & Retrieval:
      - `embedding`: Stores high-dimensional dense vector embeddings (e.g., 1536 dimensions
        for OpenAI's text-embedding-3-small).
      - `chunk_index` & `total_chunks`: Retains structural positioning of the chunk, which
        enables reconstructing surrounding context (parent retrieval / windowing).
      - `embedding_model`: Keeps track of the generator model. If we upgrade the embedding
        model, this allows us to easily invalidate older embeddings and re-index selectively.
      - `meta` (Denormalized JSONB): Duplicates key metadata fields from the Article table.
        This is a critical architectural pattern for pgvector; it enables single-stage
        pre-filtering (e.g., WHERE meta->'genre' ? 'RPG') directly inside the vector search
        index without costly SQL joins.
    """
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    article_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Dimenzija mora da odgovara embedding modelu iz settings.py (1536 za text-embedding-3-small)
    # nullable=True jer se chunk prvo insertuje, pa se embedding popunjava
    embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(1536), nullable=True
    )

    # Pozicija chunka unutar roditelja (0-indexed)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False)

    # Cuva koji model je generisao embedding – bitno za invalidaciju cache-a
    embedding_model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    # Denormalizovani metadata za search filtering bez JOIN-a:
    # { "title": ..., "url": ..., "source": ..., "game": ...,
    #   "genre": [...], "platform": [...], "year": ... }
    meta: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    article: Mapped["Article"] = relationship("Article", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<Chunk id={self.id} article_id={self.article_id} idx={self.chunk_index}/{self.total_chunks}>"


class SearchLog(Base):
    """
    Stores a detailed telemetry trail of every search request executed against the API.

    This table is a cornerstone of the search evaluation framework. It records system performance
    metrics and search behavior to drive data-driven tuning of the ranking algorithms.

    Key Features for Analytics & Evaluation:
      - `mode`: Tracks which retrieval engine served the request ("vector", "keyword", or "hybrid").
      - `latency_ms`: Monitors performance bottlenecks (especially crucial during heavy Reranking runs).
      - `results_snapshot`: Saves a static copy of the returned document IDs and their relative
        relevance scores. This is essential for calculating offline retrieval quality metrics
        (like NDCG or Mean Reciprocal Rank) and evaluating ranking degradation over time.
    """

    __tablename__ = "search_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)

    # "vector" | "keyword" | "hybrid"
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    reranked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    results_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Ukupno vreme od prijema zahteva do odgovora (ms)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Sacuvani rezultati (title, url, score) za naknadnu analizu
    results_snapshot: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    def __repr__(self) -> str:
        return f"<SearchLog id={self.id} mode={self.mode!r} query={self.query[:40]!r}>"


