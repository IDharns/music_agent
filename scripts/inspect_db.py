from __future__ import annotations

import sqlite3
from pathlib import Path


DB_PATH = Path("data/music_v2.db")


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("=" * 80)
    print("tables")
    for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ):
        print(row["name"])

    print("=" * 80)
    print("counts")
    for table in ("tracks", "contributors", "contributor_enrich_cache", "album_enrich_cache"):
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {count}")

    print("=" * 80)
    print("sample tracks")
    rows = conn.execute(
        """
        SELECT
            id,
            title,
            artist_name,
            album_name,
            release_year,
            vocal_type,
            popularity_bucket,
            norm_style_tags_json,
            mood_anchors_json
        FROM tracks
        ORDER BY id
        LIMIT 10
        """
    ).fetchall()
    for row in rows:
        print(dict(row))

    conn.close()


if __name__ == "__main__":
    main()
