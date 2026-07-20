import asyncio
from typing import Callable, Any
from retrieval_lab.config import settings
from retrieval_lab.ingestion.embedding.base import BaseEmbedder
from fastembed import TextEmbedding

class FastEmbedder(BaseEmbedder):
    def __init__(self, model_name, **kwargs: Any):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Name of the embedding provider (e.g., 'fastembed', 'ollama', 'openai')"""
        return 'fastembed'

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # FastEmbed returns generator which we convert into a list
        # It is CPY bound, so we put it into a thread
        def _get_embeddings():
            return [vec.tolist() for vec in self._model.embed(texts)]
        return await asyncio.to_thread(_get_embeddings)

