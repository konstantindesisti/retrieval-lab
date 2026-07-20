from typing import Any

from ollama import AsyncClient
from retrieval_lab.ingestion.embedding.base import BaseEmbedder
from retrieval_lab.config.settings import settings


class OllamaEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, **kwargs: Any):
        super().__init__(**kwargs)
        self._model_name = model_name
        self.host = kwargs.get('host') or settings.embedding_providers.ollama.host
        self.client = AsyncClient(host=self.host)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Name of the embedding provider (e.g., 'fastembed', 'ollama', 'openai')"""
        return 'fastembed'

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = await self.client.embed(
            model=self._model_name,
            input=texts)

        return [list(vec) for vec in response.embeddings]