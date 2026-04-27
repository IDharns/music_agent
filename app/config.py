from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

load_dotenv(BASE_DIR / ".env")


def default_db_path() -> Path:
    env_path = os.getenv("MUSIC_DB_PATH")
    if env_path:
        return Path(env_path)

    primary = DATA_DIR / "music.db"
    legacy = DATA_DIR / "music_v2.db"
    if primary.exists() or not legacy.exists():
        return primary
    return legacy


class Settings:
    APP_NAME = "Music Agent API"
    APP_VERSION = "0.6.0"

    BASE_DIR = BASE_DIR
    DATA_DIR = DATA_DIR

    DB_PATH = default_db_path()
    INDEX_PATH = Path(os.getenv("MUSIC_INDEX_PATH", DATA_DIR / "faiss.index"))
    IDS_PATH = Path(os.getenv("MUSIC_IDS_PATH", DATA_DIR / "ids.npy"))

    OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")
    OPENROUTER_BASE_URL: str = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )

    ENABLE_LLM_QUERY_REWRITE: bool = os.getenv("ENABLE_LLM_QUERY_REWRITE", "1") == "1"
    ENABLE_LLM_RERANK: bool = os.getenv("ENABLE_LLM_RERANK", "1") == "1"

    LASTFM_API_KEY: str | None = os.getenv("LASTFM_API_KEY")

    @classmethod
    def validate_runtime_files(cls) -> None:
        missing = [
            str(path)
            for path in (cls.DB_PATH, cls.INDEX_PATH, cls.IDS_PATH)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing required runtime files:\n" + "\n".join(missing)
            )


settings = Settings()
