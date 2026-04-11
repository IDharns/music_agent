import json
import re
import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd

from init_db import get_connection, create_tables

DB_PATH = "../data/music.db"
MSD_CSV_PATH = "../data/msd_tracks.csv"

COLUMN_MAP_CANDIDATES = {
    "source_track_id": ["track_id"],
    "title": ["title"],
    "artist_name": ["artist_name"],
    "album_name": ["album_name"],
    "release_year": ["year"],
    "duration_ms": ["duration", "duration_ms"],
}


def find_existing_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_text(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    return re.sub(r"\s+", " ", s)


def normalize_title(title: Optional[str]) -> Optional[str]:
    if not title:
        return None
    s = title.lower().strip()
    s = re.sub(r"\(.*?remaster.*?\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\(.*?live.*?\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\(.*?version.*?\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\[.*?\]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def normalize_artist(artist: Optional[str]) -> Optional[str]:
    if not artist:
        return None
    s = artist.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s or None


def to_year(x) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    m = re.search(r"(19|20)\d{2}", s)
    if not m:
        return None
    y = int(m.group())
    return y if 1900 <= y <= 2100 else None


def to_duration_ms(x) -> Optional[int]:
    if pd.isna(x):
        return None
    try:
        v = float(x)
    except Exception:
        return None

    if v <= 0:
        return None

    if v < 10000:
        return int(v * 1000)
    return int(v)


def year_to_era(year: Optional[int]) -> Optional[str]:
    if year is None:
        return None
    decade = (year // 10) * 10
    return f"{decade}s"


def popularity_to_bucket(popularity: Optional[float]) -> Optional[str]:
    if popularity is None:
        return None
    if popularity < 30:
        return "low"
    if popularity < 70:
        return "mid"
    return "high"


def detect_vocal_type(title: Optional[str]) -> Optional[str]:
    # 这里只是占位，后续可接外部 enrichment
    return None


def detect_language(title: Optional[str], artist_name: Optional[str]) -> Optional[str]:
    # 暂时不做瞎猜
    return None


def build_track_internal_id(source: str, source_track_id: Optional[str], title: Optional[str], artist_name: Optional[str]) -> str:
    base = f"{source}::{source_track_id}" if source_track_id else f"{source}::{title or ''}::{artist_name or ''}"
    return "trk_" + hashlib.md5(base.encode("utf-8")).hexdigest()


def build_artist_internal_id(source: str, artist_name: str) -> str:
    base = f"{source}::artist::{artist_name}"
    return "art_" + hashlib.md5(base.encode("utf-8")).hexdigest()


def build_duplicate_group_key(normalized_title: Optional[str], normalized_artist: Optional[str], release_year: Optional[int]) -> Optional[str]:
    if not normalized_title and not normalized_artist:
        return None
    return f"{normalized_title or ''}__{normalized_artist or ''}__{release_year or ''}"


def join_nonempty(parts: list[Optional[str]], sep: str = " | ") -> str:
    return sep.join([p for p in parts if p])


def build_search_text(row: pd.Series) -> str:
    parts = [
        row.get("title"),
        row.get("artist_name"),
        row.get("album_name"),
        row.get("genre_text"),
        row.get("style_text"),
        row.get("mood_text"),
        row.get("vocal_type"),
        row.get("language"),
    ]

    if row.get("release_year"):
        parts.append(str(row["release_year"]))
    if row.get("era_text"):
        parts.append(row["era_text"])

    tags = row.get("tags_list") or []
    if tags:
        parts.append(" ".join(tags[:12]))

    artist_tags = row.get("artist_tags_list") or []
    if artist_tags:
        parts.append(" ".join(artist_tags[:12]))

    return join_nonempty(parts, sep=" | ")


def build_embedding_text(row: pd.Series) -> str:
    parts = []

    if row.get("title"):
        parts.append(f"Title: {row['title']}")
    if row.get("artist_name"):
        parts.append(f"Artist: {row['artist_name']}")
    if row.get("album_name"):
        parts.append(f"Album: {row['album_name']}")
    if row.get("release_year"):
        parts.append(f"Year: {row['release_year']}")
    if row.get("era_text"):
        parts.append(f"Era: {row['era_text']}")

    if row.get("genre_text"):
        parts.append(f"Genres: {row['genre_text']}")
    if row.get("style_text"):
        parts.append(f"Styles: {row['style_text']}")
    if row.get("mood_text"):
        parts.append(f"Mood: {row['mood_text']}")

    if row.get("vocal_type"):
        parts.append(f"Vocal: {row['vocal_type']}")
    if row.get("language"):
        parts.append(f"Language: {row['language']}")
    if row.get("popularity_bucket"):
        parts.append(f"Popularity: {row['popularity_bucket']}")

    tags = row.get("tags_list") or []
    if tags:
        parts.append("Tags: " + ", ".join(tags[:10]))

    artist_tags = row.get("artist_tags_list") or []
    if artist_tags:
        parts.append("Artist tags: " + ", ".join(artist_tags[:10]))

    return join_nonempty(parts, sep=" | ")


def load_and_standardize(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    out = pd.DataFrame()
    for std_col, candidates in COLUMN_MAP_CANDIDATES.items():
        real_col = find_existing_column(df, candidates)
        out[std_col] = df[real_col] if real_col else None

    out["_raw_json"] = df.to_dict(orient="records")
    return out


def clean_msd(df: pd.DataFrame) -> pd.DataFrame:
    df["source"] = "msd"

    for col in ["source_track_id", "title", "artist_name", "album_name"]:
        df[col] = df[col].apply(normalize_text)

    df["release_year"] = df["release_year"].apply(to_year)
    df["duration_ms"] = df["duration_ms"].apply(to_duration_ms)

    df["normalized_title"] = df["title"].apply(normalize_title)
    df["normalized_artist"] = df["artist_name"].apply(normalize_artist)

    df["duplicate_group_key"] = df.apply(
        lambda r: build_duplicate_group_key(r["normalized_title"], r["normalized_artist"], r["release_year"]),
        axis=1
    )

    df["track_id_internal"] = df.apply(
        lambda r: build_track_internal_id(r["source"], r["source_track_id"], r["title"], r["artist_name"]),
        axis=1
    )

    df["popularity"] = None
    df["popularity_bucket"] = None

    df["language"] = df.apply(lambda r: detect_language(r["title"], r["artist_name"]), axis=1)
    df["vocal_type"] = df["title"].apply(detect_vocal_type)
    df["is_instrumental"] = 0
    df["explicit_flag"] = 0

    df["genre_text"] = None
    df["style_text"] = None
    df["mood_text"] = None

    df["lyrics_text"] = None
    df["tags_list"] = [[] for _ in range(len(df))]
    df["artist_tags_list"] = [[] for _ in range(len(df))]
    df["tags_json"] = df["tags_list"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    df["artist_tags_json"] = df["artist_tags_list"].apply(lambda x: json.dumps(x, ensure_ascii=False))

    df["era_text"] = df["release_year"].apply(year_to_era)

    df["search_text"] = df.apply(build_search_text, axis=1)
    df["embedding_text"] = df.apply(build_embedding_text, axis=1)

    df["metadata_source_json"] = json.dumps({
        "source": "msd",
        "genre_text": None,
        "style_text": None,
        "mood_text": None,
        "language": None,
        "vocal_type": None,
        "tags": None,
        "artist_tags": None,
        "popularity": None,
    }, ensure_ascii=False)

    df["extra_json"] = df["_raw_json"].apply(lambda x: json.dumps(x, ensure_ascii=False))

    df = df[~(df["title"].isna() & df["artist_name"].isna())].copy()

    has_source_id = df["source_track_id"].notna()
    df_with_id = df[has_source_id].drop_duplicates(subset=["source", "source_track_id"], keep="first")
    df_without_id = df[~has_source_id].drop_duplicates(subset=["duplicate_group_key"], keep="first")
    df = pd.concat([df_with_id, df_without_id], ignore_index=True)

    return df[[
        "track_id_internal", "source", "source_track_id", "title", "artist_name",
        "album_name", "release_year", "duration_ms",
        "popularity", "popularity_bucket",
        "language", "vocal_type", "is_instrumental", "explicit_flag",
        "genre_text", "style_text", "mood_text",
        "lyrics_text",
        "tags_json", "artist_tags_json",
        "search_text", "embedding_text",
        "normalized_title", "normalized_artist", "duplicate_group_key",
        "metadata_source_json", "extra_json"
    ]].copy()


def upsert_artist(conn, source: str, artist_name: str) -> str:
    artist_name = normalize_text(artist_name)
    artist_id_internal = build_artist_internal_id(source, artist_name)
    normalized = normalize_artist(artist_name)

    conn.execute("""
                 INSERT OR IGNORE INTO artists (
        artist_id_internal, source, source_artist_id, artist_name, normalized_artist,
        popularity, popularity_bucket, language, vocal_type,
        genre_text, style_text, artist_tags_json, embedding_text, extra_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                 """, (
                     artist_id_internal, source, None, artist_name, normalized,
                     None, None, None, None,
                     None, None, json.dumps([], ensure_ascii=False), None, None
                 ))

    return artist_id_internal


def insert_tracks_and_artists(conn, df: pd.DataFrame) -> None:
    cur = conn.cursor()

    for _, row in df.iterrows():
        cur.execute("""
                    INSERT OR IGNORE INTO tracks (
            track_id_internal, source, source_track_id, title, artist_name, album_name,
            release_year, duration_ms,
            popularity, popularity_bucket,
            language, vocal_type, is_instrumental, explicit_flag,
            genre_text, style_text, mood_text,
            lyrics_text,
            tags_json, artist_tags_json,
            search_text, embedding_text,
            normalized_title, normalized_artist, duplicate_group_key,
            metadata_source_json, extra_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row["track_id_internal"], row["source"], row["source_track_id"],
                        row["title"], row["artist_name"], row["album_name"],
                        row["release_year"], row["duration_ms"],
                        row["popularity"], row["popularity_bucket"],
                        row["language"], row["vocal_type"], row["is_instrumental"], row["explicit_flag"],
                        row["genre_text"], row["style_text"], row["mood_text"],
                        row["lyrics_text"],
                        row["tags_json"], row["artist_tags_json"],
                        row["search_text"], row["embedding_text"],
                        row["normalized_title"], row["normalized_artist"], row["duplicate_group_key"],
                        row["metadata_source_json"], row["extra_json"]
                    ))

        if row["artist_name"]:
            artist_id_internal = upsert_artist(conn, row["source"], row["artist_name"])
            cur.execute("""
                        INSERT OR IGNORE INTO track_artists (
                track_id_internal, artist_id_internal, artist_order, role
            ) VALUES (?, ?, ?, ?)
                        """, (row["track_id_internal"], artist_id_internal, 0, "primary"))

    conn.commit()


def main():
    if not Path(MSD_CSV_PATH).exists():
        raise FileNotFoundError(f"找不到文件: {MSD_CSV_PATH}")

    conn = get_connection(DB_PATH)
    create_tables(conn)

    df_raw = load_and_standardize(MSD_CSV_PATH)
    print(f"原始记录数: {len(df_raw)}")

    df_clean = clean_msd(df_raw)
    print(f"清洗后记录数: {len(df_clean)}")

    insert_tracks_and_artists(conn, df_clean)
    conn.close()

    print("导入完成。")


if __name__ == "__main__":
    main()