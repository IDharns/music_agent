import os
import json
import sqlite3
from typing import Any, List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


DB_PATH = "data/music.db"
EMB_PATH = "data/embeddings.npy"
IDS_PATH = "data/ids.npy"
INDEX_PATH = "data/faiss.index"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256


def safe_strip(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def parse_tag_json(raw: Any, max_items: int = 10) -> List[str]:
    """
    支持几种常见格式：
    1. '[{"tag":"dream pop","count":100}, ...]'
    2. '["dream pop", "shoegaze"]'
    3. None / 非法 JSON
    """
    if raw is None:
        return []

    raw = safe_strip(raw)
    if not raw:
        return []

    try:
        data = json.loads(raw)
    except Exception:
        return []

    tags: List[str] = []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                tag = safe_strip(item.get("tag"))
                if tag:
                    tags.append(tag)
            elif isinstance(item, str):
                tag = safe_strip(item)
                if tag:
                    tags.append(tag)

    # 去重，保序
    seen = set()
    out = []
    for t in tags:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            out.append(t)
        if len(out) >= max_items:
            break

    return out


def year_to_era(year: Any) -> str | None:
    if year is None:
        return None
    try:
        y = int(year)
    except Exception:
        return None
    if y < 1000 or y > 3000:
        return None
    return f"{(y // 10) * 10}s"


def join_nonempty(parts: List[str], sep: str = " | ") -> str:
    return sep.join([p for p in parts if safe_strip(p)])


def build_fallback_doc_text(row: Tuple[Any, ...]) -> str:
    (
        song_id,
        title,
        artist_name,
        album_name,
        release_year,
        popularity,
        popularity_bucket,
        language,
        vocal_type,
        genre_text,
        style_text,
        mood_text,
        tags_json,
        artist_tags_json,
        embedding_text,
    ) = row

    # 1) 优先使用库里现成的 embedding_text
    if safe_strip(embedding_text):
        return embedding_text.strip()

    # 2) 否则从 richer metadata 兜底构造
    track_tags = parse_tag_json(tags_json, max_items=10)
    artist_tags = parse_tag_json(artist_tags_json, max_items=10)

    parts: List[str] = []

    if safe_strip(title):
        parts.append(f"Title: {title}")
    if safe_strip(artist_name):
        parts.append(f"Artist: {artist_name}")
    if safe_strip(album_name):
        parts.append(f"Album: {album_name}")
    if release_year is not None:
        parts.append(f"Year: {release_year}")
        era = year_to_era(release_year)
        if era:
            parts.append(f"Era: {era}")

    if safe_strip(genre_text):
        parts.append(f"Genres: {genre_text}")
    if safe_strip(style_text):
        parts.append(f"Styles: {style_text}")
    if safe_strip(mood_text):
        parts.append(f"Mood: {mood_text}")

    if safe_strip(vocal_type):
        parts.append(f"Vocal: {vocal_type}")
    if safe_strip(language):
        parts.append(f"Language: {language}")

    if safe_strip(popularity_bucket):
        parts.append(f"Popularity: {popularity_bucket}")
    elif popularity is not None:
        # 只有在 bucket 缺失时才退回原始值
        parts.append(f"Popularity score: {popularity}")

    if track_tags:
        parts.append("Track tags: " + ", ".join(track_tags))
    if artist_tags:
        parts.append("Artist tags: " + ", ".join(artist_tags))

    return join_nonempty(parts, sep=" | ")


def load_songs_from_db(db_path: str) -> tuple[np.ndarray, List[str]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
                SELECT
                    id,
                    title,
                    artist_name,
                    album_name,
                    release_year,
                    popularity,
                    popularity_bucket,
                    language,
                    vocal_type,
                    genre_text,
                    style_text,
                    mood_text,
                    tags_json,
                    artist_tags_json,
                    embedding_text
                FROM tracks
                WHERE title IS NOT NULL
                """)

    rows = cur.fetchall()
    conn.close()

    song_ids: List[int] = []
    texts: List[str] = []

    used_embedding_text = 0
    used_fallback = 0
    empty_after_build = 0

    for row in rows:
        song_id = row[0]
        embedding_text = row[-1]

        text = build_fallback_doc_text(row)

        if not safe_strip(text):
            empty_after_build += 1
            continue

        if safe_strip(embedding_text):
            used_embedding_text += 1
        else:
            used_fallback += 1

        song_ids.append(song_id)
        texts.append(text)

    print("Text build stats:")
    print(f"  used existing embedding_text: {used_embedding_text}")
    print(f"  used fallback-built text:     {used_fallback}")
    print(f"  skipped empty docs:           {empty_after_build}")

    return np.array(song_ids, dtype=np.int64), texts


def main():
    os.makedirs("data", exist_ok=True)

    print("Loading songs from database...")
    song_ids, texts = load_songs_from_db(DB_PATH)
    print(f"Loaded {len(song_ids)} songs for embedding.")

    if len(song_ids) == 0:
        raise RuntimeError("No songs available to encode. Check your database content.")

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding song texts...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if embeddings.ndim != 2 or embeddings.shape[0] != len(song_ids):
        raise RuntimeError(
            f"Unexpected embedding shape: {embeddings.shape}, expected ({len(song_ids)}, dim)"
        )

    print("Saving embeddings...")
    np.save(EMB_PATH, embeddings.astype(np.float32))
    np.save(IDS_PATH, song_ids)

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, INDEX_PATH)

    print("Done.")
    print(f"embeddings saved to: {EMB_PATH}")
    print(f"ids saved to: {IDS_PATH}")
    print(f"faiss index saved to: {INDEX_PATH}")
    print(f"embedding matrix shape: {embeddings.shape}")


if __name__ == "__main__":
    main()