import asyncio
from typing import Callable
from retrieval_lab.ingestion.embedding.base import BaseEmbedder
from retrieval_lab.config import settings


class CohereEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        self.model = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        pass