import os
import pathlib
from typing import Any

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource, DotEnvSettingsSource, \
    PydanticBaseSettingsSource

_CURRENT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = _CURRENT_DIR.parent.parent

class RedisConfig(BaseSettings):
    url: str
    embedding_cache_ttl: int = 60 * 60 * 24 * 7  # default 7 days

class FastEmbedConfig(BaseSettings):
    model_name: str= "BAAI/bge-small-en-v1.5"
    
class OllamaConfig(BaseSettings):
    model_name: str
    host: str = "http://localhost:11434"

class CohereConfig(BaseSettings):
    model_name: str

class EmbeddingOption(BaseSettings):
    provider: str

class EmbeddingProviders(BaseSettings):
    fastembed: FastEmbedConfig
    ollama: OllamaConfig
    cohere: CohereConfig


class Settings(BaseSettings):

    # DATABASE
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int

    @computed_field
    @property
    def DB_URL(self) -> str:
        """
        Constructs a PostgreSQL connection URL.

        Format:
            postgresql://<user>:<password>@<host>:<port>/<database>

        Components:
        - user (DB_USER): Username used to authenticate with the database.
        - password (DB_PASSWORD): Password for the database user.
        - host (DB_HOST): Address of the database server.
            - Use "postgres" when running inside Docker (service name).
            - Use "localhost" when connecting to a local database.
        - port (DB_PORT): Port on which the database is exposed (default: 5432).
        - database (DB_NAME): Name of the target database.

        Example:
            postgresql://sandbox:secret@postgres:5432/mydb
        """
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # AI
    GEMINI_API_KEY: str
    COHERE_API_KEY: str
    EMBEDDING_MODEL: str
    OLLAMA_HOST: str

    # YAML Settings
    redis: RedisConfig
    embedding_option: EmbeddingOption
    embedding_providers: EmbeddingProviders

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        yaml_file= _CURRENT_DIR / 'config.yaml',
        yaml_file_encoding="utf-8",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            DotEnvSettingsSource(settings_cls),
            YamlConfigSettingsSource(settings_cls),
            env_settings,
        )

settings = Settings()

if __name__ == '__main__':
    print(settings.embedding_providers.fastembed.model)
    print(settings.redis.url)