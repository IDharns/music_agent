from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.rebuild_music_library import (
    DEFAULT_BATCH_SIZE,
    build_index,
    derive_mood_anchors,
    infer_language_from_tags,
    infer_vocal_type_from_tags,
    json_dumps,
    make_embedding_text,
    normalize_style_tags,
    parse_json_tags,
)


def parse_json_list(raw: Any) -> list[str]:
    return parse_json_tags(raw, max_items=100)


def refresh_contributors(conn: sqlite3.Connection) -> tuple[int, int]:
    rows = conn.execute(
        """
        SELECT contributor_key, raw_tags_json
        FROM contributors
        """
    ).fetchall()

    changed = 0
    for contributor_key, raw_tags_json in rows:
        raw_tags = parse_json_tags(raw_tags_json, max_items=50)
        next_style = normalize_style_tags(raw_tags, None, None)
        next_style_json = json_dumps(next_style)

        current = conn.execute(
            """
            SELECT norm_style_tags_json
            FROM contributors
            WHERE contributor_key = ?
            """,
            (contributor_key,),
        ).fetchone()

        if current and current[0] != next_style_json:
            conn.execute(
                """
                UPDATE contributors
                SET norm_style_tags_json = ?
                WHERE contributor_key = ?
                """,
                (next_style_json, contributor_key),
            )
            changed += 1

    return len(rows), changed


def refresh_tracks(conn: sqlite3.Connection, commit_every: int) -> tuple[int, int]:
    conn.row_factory = sqlite3.Row
    total = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]

    changed = 0
    processed = 0
    last_id = 0

    while True:
        rows = conn.execute(
            """
            SELECT
                id,
                title,
                artist_name,
                album_name,
                release_year,
                language,
                vocal_type,
                popularity_bucket,
                primary_artists_json,
                featured_artists_json,
                raw_artist_tags_json,
                raw_album_tags_json,
                genre_text,
                style_text,
                mood_text,
                norm_style_tags_json,
                mood_anchors_json,
                mood_confidence,
                embedding_text
            FROM tracks
            WHERE id > ?
            ORDER BY id
            LIMIT ?
            """,
            (last_id, commit_every),
        ).fetchall()

        if not rows:
            break

        for row in rows:
            last_id = int(row["id"])
            processed += 1

            primary_artists = parse_json_list(row["primary_artists_json"])
            featured_artists = parse_json_list(row["featured_artists_json"])
            raw_artist_tags = parse_json_tags(row["raw_artist_tags_json"], max_items=50)
            raw_album_tags = parse_json_tags(row["raw_album_tags_json"], max_items=50)

            style_tags = normalize_style_tags(
                raw_tags=raw_artist_tags + raw_album_tags,
                genre_text=row["genre_text"],
                style_text=row["style_text"],
            )
            mood_anchors, mood_confidence = derive_mood_anchors(
                raw_tags=raw_artist_tags + raw_album_tags,
                mood_text=row["mood_text"],
                style_tags=style_tags,
            )
            language = infer_language_from_tags(raw_artist_tags + raw_album_tags, row["language"])
            vocal_type = infer_vocal_type_from_tags(raw_artist_tags + raw_album_tags, row["vocal_type"])

            embedding_text = make_embedding_text(
                title=row["title"],
                artist_name=row["artist_name"],
                album_name=row["album_name"],
                release_year=row["release_year"],
                language=language,
                vocal_type=vocal_type,
                genre_text=row["genre_text"],
                style_text=row["style_text"],
                mood_text=row["mood_text"],
                primary_artists=primary_artists,
                featured_artists=featured_artists,
                style_tags=style_tags,
                mood_anchors=mood_anchors,
                raw_artist_tags=raw_artist_tags,
                raw_album_tags=raw_album_tags,
                pop_bucket=row["popularity_bucket"],
            )

            next_values = (
                language,
                vocal_type,
                json_dumps(style_tags),
                json_dumps(mood_anchors),
                mood_confidence,
                embedding_text,
            )
            current_values = (
                row["language"],
                row["vocal_type"],
                row["norm_style_tags_json"],
                row["mood_anchors_json"],
                row["mood_confidence"],
                row["embedding_text"],
            )

            if next_values != current_values:
                conn.execute(
                    """
                    UPDATE tracks
                    SET
                        language = ?,
                        vocal_type = ?,
                        norm_style_tags_json = ?,
                        mood_anchors_json = ?,
                        mood_confidence = ?,
                        embedding_text = ?
                    WHERE id = ?
                    """,
                    (*next_values, row["id"]),
                )
                changed += 1

        conn.commit()
        print(f"processed tracks: {processed}/{total} changed={changed}")

    conn.commit()
    return total, changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh derived fields in music_v2.db without calling Last.fm."
    )
    parser.add_argument("--db-path", type=Path, default=Path("data/music_v2.db"))
    parser.add_argument("--commit-every", type=int, default=50000)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--emb-path", type=Path, default=Path("data/embeddings.npy"))
    parser.add_argument("--ids-path", type=Path, default=Path("data/ids.npy"))
    parser.add_argument("--index-path", type=Path, default=Path("data/faiss.index"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    if not args.db_path.exists():
        raise FileNotFoundError(args.db_path)

    conn = sqlite3.connect(args.db_path)
    try:
        print("=" * 80)
        print("Step 1/2: refresh contributor derived styles")
        contributor_total, contributor_changed = refresh_contributors(conn)
        conn.commit()
        print(f"contributors checked: {contributor_total}")
        print(f"contributors changed: {contributor_changed}")

        print("=" * 80)
        print("Step 2/2: refresh track derived fields")
        track_total, track_changed = refresh_tracks(conn, commit_every=args.commit_every)
        print(f"tracks checked: {track_total}")
        print(f"tracks changed: {track_changed}")
    finally:
        conn.close()

    if args.rebuild_index:
        print("=" * 80)
        print("Rebuilding embeddings + FAISS index")
        build_index(
            dst_db=args.db_path,
            emb_path=args.emb_path,
            ids_path=args.ids_path,
            index_path=args.index_path,
            batch_size=args.batch_size,
        )
    else:
        print("=" * 80)
        print("Skipped embeddings + FAISS rebuild.")
        print("Run again with --rebuild-index before using this DB for API retrieval.")


if __name__ == "__main__":
    main()
