from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    """Environment-backed application configuration.

    The dataset path is configuration only: no participant files are opened or read here.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    process2_dataset_path: Path | None = None
    screening_max_upload_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    screening_max_duration_seconds: float = Field(default=180.0, gt=0)
    screening_target_sample_rate: int = Field(default=16_000, gt=0)

    def process2_dataset_exists(self) -> bool:
        """Check only whether the configured dataset directory exists."""
        return self.process2_dataset_path is not None and self.process2_dataset_path.exists()


@lru_cache
def get_settings() -> Settings:
    return Settings()
