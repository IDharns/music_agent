from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


DB_PATH = Path("data/music_v2.db")


def has_json_items(raw: Any) -> bool:
    if raw is None:
        return False
    try:
        value = json.loads(str(raw))
    except Exception:
        return False
    return isinstance(value, list) and len(value) > 0


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    print("=" * 80)
    print(f"tracks: {total}")

    scalar_columns = [
        "vocal_type",
        "language",
        "popularity_proxy",
        "popularity_bucket",
        "lastfm_artist_listeners",
        "lastfm_artist_playcount",
        "embedding_text",
    ]
    for column in scalar_columns:
        count = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM tracks
            WHERE {column} IS NOT NULL
              AND trim(cast({column} AS TEXT)) != ''
            """
        ).fetchone()[0]
        print(f"{column}: {count}")

    json_columns = [
        "primary_artists_json",
        "featured_artists_json",
        "all_contributors_json",
        "raw_artist_tags_json",
        "raw_album_tags_json",
        "norm_style_tags_json",
        "mood_anchors_json",
    ]
    rows = conn.execute(
        "SELECT " + ", ".join(json_columns) + " FROM tracks"
    ).fetchall()

    print("=" * 80)
    print("JSON coverage")
    for column in json_columns:
        count = sum(1 for row in rows if has_json_items(row[column]))
        print(f"{column}: {count}")

    print("=" * 80)
    print("Contributor caches")
    for table in ("contributor_enrich_cache", "album_enrich_cache", "contributors"):
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {count}")

    conn.close()


if __name__ == "__main__":
    main()
