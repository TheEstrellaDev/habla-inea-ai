from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración global de la aplicación usando Pydantic Settings."""

    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-3.1-flash-lite"
    FALLBACK_MODELS: List[str] = ["gemini-flash-lite-latest"]
    APP_NAME: str = "PrimarIA INEA AI"
    DATA_DIR: Path = Path("data")
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 2.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
