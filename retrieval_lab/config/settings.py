import os
import pathlib

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


_CURRENT_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = _CURRENT_DIR.parent.parent

class Settings(BaseSettings):
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int

#    OPEN_AI_KEY: str

    EMBEDDING_MODEL: str
    OLLAMA_HOST: str


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


    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
    )

settings = Settings()