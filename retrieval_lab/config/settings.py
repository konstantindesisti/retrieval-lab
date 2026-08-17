import pathlib
from typing import Optional

from pydantic import computed_field, Field, ConfigDict
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    YamlConfigSettingsSource,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
)

_CURRENT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = _CURRENT_DIR.parent.parent


class RedisConfig(BaseSettings):
    url: str
    embedding_cache_ttl: int = 60 * 60 * 24 * 7  # default 7 days


class EmbeddingProviderConfig(BaseSettings):
    model_config = ConfigDict(
        extra="allow"
    )  # Allows 'host', 'api_key' itd. without defining

    model_name: str
    dimension: int

    # Optional fields (can be filled from .env)
    host: Optional[str] = None
    api_key: Optional[str] = None


class EmbeddingOption(BaseSettings):
    provider: str = 'fastembed'


class ChunkConfig(BaseSettings):
    strategy: str = "fixed"
    size: int = 512
    overlap: int = 64


class TemporalConfig(BaseSettings):
    host: str = "localhost:7233"
    namespace: str = "default"
    task_queue: str = "ingestion_queue"


# ============== MAIN SETTINGS CLASS ==============
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
    redis: RedisConfig = Field(default_factory=RedisConfig)
    embedding_option: EmbeddingOption = Field(default_factory=EmbeddingOption)
    embedding_providers: dict[str, EmbeddingProviderConfig] = Field(default_factory=dict)
    chunker: ChunkConfig = Field(default=ChunkConfig())
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)

    @property
    def active_embedding_provider(self) -> EmbeddingProviderConfig:
        """Automatically returns configuration of the active embedding provider."""
        provider_name = self.embedding_option.provider

        if provider_name not in self.embedding_providers:
            raise ValueError(
                f"Provider '{provider_name}' is not configured or is turned off. "
                f"Available providers: {list(self.embedding_providers.keys())}"
            )

        return self.embedding_providers[provider_name]

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        yaml_file=_CURRENT_DIR / "config.yaml",
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
