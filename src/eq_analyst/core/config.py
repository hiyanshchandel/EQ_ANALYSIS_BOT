"""Application settings, loaded from environment variables (and a local .env file).

Every component reads configuration from one typed `Settings` object instead of
calling `os.getenv` all over the codebase. Missing or malformed values fail fast
at startup with a clear validation error rather than deep inside a request.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM -----------------------------------------------------------------
    google_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.8-flash"
    llm_temperature: float = 0.2
    llm_timeout_seconds: float = 60.0

    # --- MCP market-data server ---------------------------------------------
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8000
    mcp_server_url: str = "http://127.0.0.1:8000/mcp"

    # Yahoo exchange codes, most preferred first (NSI = NSE India, BSE = Bombay).
    # NoDecode: read the raw string and let `_split_csv` parse it (instead of JSON).
    preferred_exchanges: Annotated[list[str], NoDecode] = Field(
        default=["NSI", "BSE", "NMS", "NYQ"]
    )

    # --- Telegram ------------------------------------------------------------
    telegram_bot_token: SecretStr | None = None
    # Empty = anyone may use the bot. Otherwise only these chat ids (cost control).
    allowed_chat_ids: Annotated[list[int], NoDecode] = Field(default_factory=list)

    # --- Persistence ---------------------------------------------------------
    checkpoint_db_path: Path = Path("data/checkpoints.sqlite")

    # --- Observability -------------------------------------------------------
    # Optional: when both keys are set, traces, scores and prompts go to Langfuse.
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_environment: str = "development"
    prompt_label: str = "production"

    log_level: str = "INFO"
    log_json: bool = False

    @field_validator("preferred_exchanges", "allowed_chat_ids", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Allow comma-separated env values, e.g. PREFERRED_EXCHANGES=NSI,BSE."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, int):
            return [value]
        return value

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings (parsed once, then cached)."""
    return Settings()
