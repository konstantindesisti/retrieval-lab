from __future__ import annotations
from typing import Callable, TYPE_CHECKING

from retrieval_lab.config import settings

if TYPE_CHECKING:
    from retrieval_lab.cache.client import RedisClient
    from retrieval_lab.ingestion.embedding.base import BaseEmbedder


# 1. Define functions that performs import only when triggered
def _get_fastembed_class():
    from retrieval_lab.ingestion.embedding.fastembed_impl import FastEmbedder
    return FastEmbedder


def _get_ollama_class():
    from retrieval_lab.ingestion.embedding.ollama_impl import OllamaEmbedder
    return OllamaEmbedder


def _get_cohere_class():
    from retrieval_lab.ingestion.embedding.cohere_impl import CohereEmbedder
    return CohereEmbedder


EMBEDDER_REGISTRY: dict[str, Callable[[], type[BaseEmbedder]]] = {
    "fastembed": _get_fastembed_class,
    "ollama": _get_ollama_class,
    "cohere": _get_cohere_class,
}


def get_embedder(redis_client: RedisClient) -> BaseEmbedder:
    """
    Dynamically imports and creates the embedder without loading unnecessary
    heavy libraries during application startup.
    """
    from retrieval_lab.ingestion.embedding.base import CachedEmbedder

    provider_name = settings.embedding_option.provider

    if provider_name not in EMBEDDER_REGISTRY:
        raise ValueError(f"Unknown embedder provider: {provider_name}. "
                         f"Available providers: {list(EMBEDDER_REGISTRY.keys())}")

    embedder_class = EMBEDDER_REGISTRY[provider_name]()

    provider_config = getattr(settings.embedding_providers, provider_name).model_dump()

    base_embedder = embedder_class(**provider_config)

    return CachedEmbedder(base_embedder, redis_client)
