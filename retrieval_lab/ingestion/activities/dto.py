import hashlib
from dataclasses import dataclass, field
from datetime import datetime


# ============== SCRAPER ==============
@dataclass
class RSSEntry:
    url: str
    title: str
    published_at: datetime | None = None


@dataclass
class ScrapedArticle:
    url: str
    title: str
    body: str
    source: str
    meta: dict = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash and self.body:
            self.content_hash = hashlib.sha256(self.body.encode()).hexdigest()


# ============== CHUNKER ==============
@dataclass
class ChunkData:
    content: str
    chunk_index: int
    total_chunks: int  # popunjava se naknadno (po zavretku chunking-a)
    meta: dict = field(default_factory=dict)  # nasledjen + pozicioni metadata


@dataclass
class ChunkedDocument:
    article_url: str
    article_title: str
    source: str
    chunks: list[ChunkData]
    strategy: str

# ============== EMBEDDER ==============
@dataclass
class EmbeddedChunk:
    content: str
    embedding: list[float]
    chunk_index: int
    total_chunks: int
    meta: dict = field(default_factory=dict)

@dataclass
class EmbeddedDocument:
    article_url: str
    article_title: str
    article_body: str  # needed for indexer for Article table
    source: str
    chunks: list[EmbeddedChunk]
    strategy: str
    embedding_provider: str
    embedding_model: str


