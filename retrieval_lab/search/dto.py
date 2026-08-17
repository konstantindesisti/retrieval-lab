from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field


@dataclass
class ScoredChunk:
    chunk_id: int  # ovo je Chunk.id, ali nije FK – nema baze
    content: str
    score: float
    meta: dict
    retriever: str

class SearchMode(StrEnum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass
class SearchFilters:
    source: str | None = None
    genre: str | None = None
    platform: str | None = None
    year: int | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    mode: SearchMode = SearchMode.HYBRID
    reranked: bool = True
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict = Field(default_factory=dict)  # source, genre, platform, year


class SourceAttribution(BaseModel):
    title: str
    url: str
    score: float


class SearchResponse(BaseModel):
    answer: str
    sources: list[SourceAttribution]
    mode: str
    reranked: bool
    latency_ms: float
