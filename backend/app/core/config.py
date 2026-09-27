from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Environment-backed application configuration.

    The dataset path is configuration only: no participant files are opened or read here.
    """

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    process2_dataset_path: Path | None = None

    def process2_dataset_exists(self) -> bool:
        """Check only whether the configured dataset directory exists."""
        return self.process2_dataset_path is not None and self.process2_dataset_path.exists()


@lru_cache
def get_settings() -> Settings:
    return Settings()
