from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

load_dotenv(BASE_DIR / ".env")


class Settings:
    APP_NAME = "Music Agent API"
    APP_VERSION = "0.5.0"

    BASE_DIR = BASE_DIR
    DATA_DIR = DATA_DIR

    DB_PATH = DATA_DIR / "music.db"
    INDEX_PATH = DATA_DIR / "faiss.index"
    IDS_PATH = DATA_DIR / "ids.npy"

    OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
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