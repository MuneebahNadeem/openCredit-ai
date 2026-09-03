"""Backend configuration — all values overridable via environment / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Comma-separated list of allowed browser origins for the dev frontend.
    backend_cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173"
    )

    # Where completed investigation records are persisted (JSON files).
    investigations_dir: str = "data/investigations"

    # Where uploaded document binaries are stored.
    documents_dir: str = "data/documents"

    # How many investigations may run at the same time.
    max_concurrent_investigations: int = 2

    # Conversational "Ask OpenCredit" endpoint settings.
    ask_model: str = "gpt-4o-mini"
    ask_temperature: float = 0.2

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.backend_cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
