from typing import Any
from retrieval_lab.ingestion.embedding.base import BaseEmbedder


class CohereEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, **kwargs: Any):
        super().__init__(**kwargs)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Name of the embedding provider (e.g., 'fastembed', 'ollama', 'openai')"""
        return "cohere"

    def embed(self, texts: list[str]) -> list[list[float]]:
        pass
