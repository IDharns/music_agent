# scripts/rebuild_music_library.py
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import faiss
from app.config import Settings

# =========================
# Config
# =========================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LASTFM_BASE_URL = "https://ws.audioscrobbler.com/2.0/"
USER_AGENT = "music-agent/0.1"
HTTP_TIMEOUT = 20
LASTFM_SLEEP_SEC = 0.12  # 温和一点，别打太猛
DEFAULT_BATCH_SIZE = 256

# 只做“风格主信号”，避免把一堆主观情绪硬塞进去
STYLE_VOCAB: dict[str, list[str]] = {
    "dream_pop": ["dream pop", "ethereal", "shoegaze", "lush", "atmospheric"],
    "indie_pop": ["indie pop", "indietronica", "bedroom pop", "chamber pop"],
    "singer_songwriter": ["singer-songwriter", "songwriter", "acoustic pop"],
    "female_vocal": ["female vocalists", "female vocalist", "female vocal", "female singer", "women vocals"],
    "male_vocal": ["male vocalists", "male vocalist", "male vocal", "male singer"],
    "electronic": ["electronic", "electronica", "synthpop", "electropop", "dance"],
    "indie_rock": ["indie rock", "alternative rock", "alt rock", "alt-rock"],
    "folk": ["folk", "indie folk", "folk pop"],
    "rnb": ["rnb", "r&b", "soul", "neo soul"],
    "hip_hop": ["hip hop", "hip-hop", "rap", "trap"],
    "j_pop": ["j-pop", "jpop"],
    "c_pop": ["c-pop", "cpop", "mandopop", "cantopop"],
    "k_pop": ["k-pop", "kpop"],
    "ambient": ["ambient", "downtempo", "chillout"],
    "country_pop": ["country pop", "country"],
    "pop": ["pop", "dance pop", "soft pop"],
}

# mood 只保留很窄的 anchor，做弱信号
MOOD_ANCHOR_MAP: dict[str, list[str]] = {
    "dreamy": ["dream pop", "dreamy", "ethereal", "lush", "atmospheric"],
    "calm": ["ambient", "downtempo", "chillout", "soft", "gentle", "peaceful"],
    "dark": ["darkwave", "gothic", "brooding", "dark"],
    "melancholic": ["melancholy", "melancholic", "wistful", "heartbreak", "sad"],
    "energetic": ["upbeat", "energetic", "driving", "dance"],
    "uplifting": ["uplifting", "hopeful", "bright", "inspiring"],
}

# 这些 style -> mood 可以窄传播
STYLE_TO_MOOD: dict[str, list[str]] = {
    "dream_pop": ["dreamy"],
    "ambient": ["calm"],
}

STRONG_MOOD_KEYS = {
    "dream pop",
    "dreamy",
    "ethereal",
    "ambient",
    "darkwave",
    "gothic",
    "melancholy",
    "melancholic",
    "upbeat",
    "uplifting",
}

ARTIST_SPLIT_PATTERNS = [
    r"\s+feat\.\s+",
    r"\s+featuring\s+",
    r"\s+ft\.\s+",
    r"\s+with\s+",
    r"\s+x\s+",
    r"\s+vs\.\s+",
    r"\s+and\s+",
    r"\s*&\s*",
    r"\s*,\s*",
    r"\s*;\s*",
    r"\s*/\s*",
]


# =========================
# Small utils
# =========================

def safe_str(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def safe_lower(x: Any) -> str:
    s = safe_str(x)
    return s.lower() if s else ""


def uniq_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        k = x.lower().strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x.strip())
    return out


def join_nonempty(parts: list[str], sep: str = " | ") -> str:
    return sep.join([p for p in parts if safe_str(p)])


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def parse_json_tags(raw: Any, max_items: int = 20) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        data = raw
    else:
        text = safe_str(raw)
        if not text:
            return []
        try:
            data = json.loads(text)
        except Exception:
            return []

    tags: list[str] = []
    if isinstance(data, list):
        for item in data:
            tag = None
            if isinstance(item, dict):
                tag = safe_str(item.get("tag") or item.get("name"))
            elif isinstance(item, str):
                tag = safe_str(item)
            if tag:
                tags.append(tag)

    return uniq_keep_order(tags)[:max_items]


def normalize_name(name: str | None) -> str:
    if not name:
        return ""
    s = name.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[‐-–—]", "-", s)
    return s


def normalize_album(title: str | None) -> str:
    return normalize_name(title)


def year_to_era(year: int | None) -> str | None:
    if year is None or year < 1000 or year > 3000:
        return None
    return f"{(year // 10) * 10}s"


def text_blob(*parts: Any) -> str:
    xs = [safe_str(p) for p in parts]
    return " | ".join([x for x in xs if x]).lower()


def contains_term(text: str, term: str) -> bool:
    if not text or not term:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


def split_artists(artist_name: str | None) -> tuple[list[str], list[str]]:
    """
    返回 (primary_artists, featured_artists)
    规则不可能完美，但比整串 artist_name 直接查 API 稳得多。
    """
    artist = safe_str(artist_name)
    if not artist:
        return [], []

    original = artist

    # 先拆 feat/ft/featuring/with
    feature_split_regex = r"\s+(feat\.|featuring|ft\.|with)\s+"
    feat_parts = re.split(feature_split_regex, original, flags=re.IGNORECASE)

    if len(feat_parts) >= 3:
        primary_raw = feat_parts[0]
        featured_raw = feat_parts[-1]
    else:
        primary_raw = original
        featured_raw = ""

    def split_multi(s: str) -> list[str]:
        if not s.strip():
            return []
        out = [s]
        for pat in [
            r"\s*&\s*",
            r"\s+x\s+",
            r"\s+and\s+",
            r"\s*,\s*",
            r"\s*/\s*",
            r"\s*;\s*",
        ]:
            next_out: list[str] = []
            for piece in out:
                next_out.extend(re.split(pat, piece, flags=re.IGNORECASE))
            out = next_out
        return uniq_keep_order([p.strip() for p in out if p.strip()])

    primary = split_multi(primary_raw)
    featured = split_multi(featured_raw)

    if not primary and artist:
        primary = [artist]

    return primary, featured


def infer_language_from_tags(tags: list[str], existing: str | None) -> str | None:
    if safe_str(existing):
        return existing
    blob = " ".join(tags).lower()
    if "mandarin" in blob or "chinese" in blob or "mandopop" in blob or "cantopop" in blob:
        return "zh"
    if "japanese" in blob or "j-pop" in blob or "jpop" in blob:
        return "ja"
    if "korean" in blob or "k-pop" in blob or "kpop" in blob:
        return "ko"
    if blob:
        return "unknown"
    return None


def infer_vocal_type_from_tags(tags: list[str], existing: str | None) -> str | None:
    if safe_str(existing):
        return existing
    blob = " ".join(tags).lower()
    if "female vocal" in blob or "female vocalists" in blob or "female singer" in blob:
        return "female vocal"
    if "male vocal" in blob or "male vocalists" in blob or "male singer" in blob:
        return "male vocal"
    return None


def compute_popularity_proxy(popularity: float | None, listeners: float | None, playcount: float | None) -> float | None:
    vals: list[float] = []

    if popularity is not None:
        vals.append(max(0.0, min(100.0, popularity)) / 100.0)

    if listeners is not None and listeners > 0:
        vals.append(min(math.log10(listeners + 1) / 7.0, 1.0))

    if playcount is not None and playcount > 0:
        vals.append(min(math.log10(playcount + 1) / 8.0, 1.0))

    if not vals:
        return None

    return round(sum(vals) / len(vals), 4)


def popularity_bucket(proxy: float | None) -> str | None:
    if proxy is None:
        return None
    if proxy >= 0.75:
        return "high"
    if proxy >= 0.45:
        return "medium"
    return "low"


# =========================
# API client
# =========================

class LastFMClient:
    def __init__(self, api_key: str | None):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _get(self, params: dict[str, Any]) -> dict[str, Any] | None:
        if not self.api_key:
            return None
        full_params = {
            "api_key": self.api_key,
            "format": "json",
            **params,
        }
        try:
            resp = self.session.get(
                LASTFM_BASE_URL,
                params=full_params,
                timeout=HTTP_TIMEOUT,
            )
            if resp.status_code != 200:
                return None
            time.sleep(LASTFM_SLEEP_SEC)
            return resp.json()
        except Exception:
            return None

    def artist_top_tags(self, artist_name: str) -> list[dict[str, Any]]:
        data = self._get({"method": "artist.getTopTags", "artist": artist_name})
        if not data:
            return []
        raw = data.get("toptags", {}).get("tag", [])
        if isinstance(raw, dict):
            raw = [raw]
        out = []
        for item in raw:
            name = safe_str(item.get("name"))
            count = item.get("count")
            try:
                count = int(count) if count is not None else None
            except Exception:
                count = None
            if name:
                out.append({"tag": name, "count": count})
        return out

    def artist_info(self, artist_name: str) -> dict[str, Any]:
        data = self._get({"method": "artist.getInfo", "artist": artist_name})
        if not data:
            return {}
        artist = data.get("artist", {}) if isinstance(data, dict) else {}
        stats = artist.get("stats", {}) if isinstance(artist, dict) else {}
        listeners = stats.get("listeners")
        playcount = stats.get("playcount")
        try:
            listeners = float(listeners) if listeners is not None else None
        except Exception:
            listeners = None
        try:
            playcount = float(playcount) if playcount is not None else None
        except Exception:
            playcount = None
        return {
            "listeners": listeners,
            "playcount": playcount,
        }

    def album_top_tags(self, artist_name: str, album_name: str) -> list[dict[str, Any]]:
        data = self._get({"method": "album.getTopTags", "artist": artist_name, "album": album_name})
        if not data:
            return []
        raw = data.get("toptags", {}).get("tag", [])
        if isinstance(raw, dict):
            raw = [raw]
        out = []
        for item in raw:
            name = safe_str(item.get("name"))
            count = item.get("count")
            try:
                count = int(count) if count is not None else None
            except Exception:
                count = None
            if name:
                out.append({"tag": name, "count": count})
        return out


# =========================
# DB schema
# =========================

def create_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript(
        """
        DROP TABLE IF EXISTS contributor_enrich_cache;
        DROP TABLE IF EXISTS album_enrich_cache;
        DROP TABLE IF EXISTS contributors;
        DROP TABLE IF EXISTS tracks;

        CREATE TABLE contributor_enrich_cache (
                                                  contributor_key TEXT PRIMARY KEY,
                                                  contributor_name TEXT NOT NULL,
                                                  raw_tags_json TEXT NOT NULL,
                                                  listeners REAL,
                                                  playcount REAL,
                                                  fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE album_enrich_cache (
                                            album_key TEXT PRIMARY KEY,
                                            artist_name TEXT NOT NULL,
                                            album_name TEXT NOT NULL,
                                            raw_tags_json TEXT NOT NULL,
                                            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE contributors (
                                      contributor_key TEXT PRIMARY KEY,
                                      contributor_name TEXT NOT NULL,
                                      raw_tags_json TEXT NOT NULL,
                                      norm_style_tags_json TEXT NOT NULL,
                                      listeners REAL,
                                      playcount REAL
        );

        CREATE TABLE tracks (
                                id INTEGER PRIMARY KEY,
                                title TEXT,
                                artist_name TEXT,
                                album_name TEXT,
                                release_year INTEGER,
                                language TEXT,
                                vocal_type TEXT,

                                popularity REAL,
                                lastfm_artist_listeners REAL,
                                lastfm_artist_playcount REAL,
                                popularity_proxy REAL,
                                popularity_bucket TEXT,

                                primary_artists_json TEXT NOT NULL,
                                featured_artists_json TEXT NOT NULL,
                                all_contributors_json TEXT NOT NULL,

                                raw_artist_tags_json TEXT NOT NULL,
                                raw_album_tags_json TEXT NOT NULL,

                                norm_style_tags_json TEXT NOT NULL,
                                mood_anchors_json TEXT NOT NULL,
                                mood_confidence REAL NOT NULL,

                                genre_text TEXT,
                                style_text TEXT,
                                mood_text TEXT,

                                embedding_text TEXT NOT NULL
        );

        CREATE INDEX idx_tracks_artist_name ON tracks(artist_name);
        CREATE INDEX idx_tracks_release_year ON tracks(release_year);
        CREATE INDEX idx_tracks_popularity_proxy ON tracks(popularity_proxy);
        """
    )
    conn.commit()


def table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {r[1] for r in rows}


def pick(row: sqlite3.Row, cols: set[str], *candidates: str) -> Any:
    for c in candidates:
        if c in cols:
            return row[c]
    return None


# =========================
# Read source rows
# =========================

def load_source_rows(src_conn: sqlite3.Connection) -> tuple[list[sqlite3.Row], set[str]]:
    src_conn.row_factory = sqlite3.Row
    # Support both "tracks" and "songs" table names
    table_name = "tracks"
    existing_tables = [r[0] for r in src_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    if "tracks" not in existing_tables and "songs" in existing_tables:
        table_name = "songs"
    cols = table_columns(src_conn, table_name)
    rows = src_conn.execute(f"SELECT * FROM {table_name} WHERE title IS NOT NULL").fetchall()
    return rows, cols


# =========================
# Enrich helpers
# =========================

@dataclass
class ContributorEnrich:
    name: str
    raw_tags: list[str]
    listeners: float | None
    playcount: float | None


@dataclass
class AlbumEnrich:
    artist_name: str
    album_name: str
    raw_tags: list[str]


def contributor_key(name: str) -> str:
    return normalize_name(name)


def album_key(artist_name: str, album_name: str) -> str:
    return f"{normalize_name(artist_name)}||{normalize_album(album_name)}"


def save_contributor_cache(conn: sqlite3.Connection, key: str, data: ContributorEnrich) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO contributor_enrich_cache
        (contributor_key, contributor_name, raw_tags_json, listeners, playcount)
        VALUES (?, ?, ?, ?, ?)
        """,
        (key, data.name, json_dumps(data.raw_tags), data.listeners, data.playcount),
    )
    conn.commit()


def get_contributor_cache(conn: sqlite3.Connection, key: str) -> ContributorEnrich | None:
    row = conn.execute(
        """
        SELECT contributor_name, raw_tags_json, listeners, playcount
        FROM contributor_enrich_cache
        WHERE contributor_key = ?
        """,
        (key,),
    ).fetchone()
    if not row:
        return None
    return ContributorEnrich(
        name=row[0],
        raw_tags=parse_json_tags(row[1], max_items=50),
        listeners=row[2],
        playcount=row[3],
    )


def save_album_cache(conn: sqlite3.Connection, key: str, data: AlbumEnrich) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO album_enrich_cache
        (album_key, artist_name, album_name, raw_tags_json)
        VALUES (?, ?, ?, ?)
        """,
        (key, data.artist_name, data.album_name, json_dumps(data.raw_tags)),
    )
    conn.commit()


def get_album_cache(conn: sqlite3.Connection, key: str) -> AlbumEnrich | None:
    row = conn.execute(
        """
        SELECT artist_name, album_name, raw_tags_json
        FROM album_enrich_cache
        WHERE album_key = ?
        """,
        (key,),
    ).fetchone()
    if not row:
        return None
    return AlbumEnrich(
        artist_name=row[0],
        album_name=row[1],
        raw_tags=parse_json_tags(row[2], max_items=50),
    )


def fetch_all_contributors(rows: list[sqlite3.Row], cols: set[str]) -> dict[str, str]:
    all_contributors: dict[str, str] = {}
    for row in rows:
        artist_name = safe_str(pick(row, cols, "artist_name", "artist"))
        primary, featured = split_artists(artist_name)
        for name in primary + featured:
            key = contributor_key(name)
            if key and key not in all_contributors:
                all_contributors[key] = name
    return all_contributors


def fetch_all_albums(rows: list[sqlite3.Row], cols: set[str]) -> dict[str, tuple[str, str]]:
    albums: dict[str, tuple[str, str]] = {}
    for row in rows:
        artist_name = safe_str(pick(row, cols, "artist_name", "artist"))
        album_name = safe_str(pick(row, cols, "album_name", "album"))
        if not artist_name or not album_name:
            continue
        k = album_key(artist_name, album_name)
        if k not in albums:
            albums[k] = (artist_name, album_name)
    return albums


def enrich_contributors(
        dst_conn: sqlite3.Connection,
        lastfm: LastFMClient,
        contributors: dict[str, str],
        max_count: int | None = None,
) -> None:
    n = 0
    for key, name in contributors.items():
        if max_count is not None and n >= max_count:
            break
        n += 1

        cached = get_contributor_cache(dst_conn, key)
        if cached is not None:
            if n % 500 == 0:
                print(f"[contributors] cache hit {n}/{len(contributors)}")
            continue

        tags_raw = lastfm.artist_top_tags(name)
        info = lastfm.artist_info(name)

        tags = []
        for item in tags_raw[:20]:
            tag = safe_str(item.get("tag"))
            if tag:
                tags.append(tag)

        data = ContributorEnrich(
            name=name,
            raw_tags=uniq_keep_order(tags),
            listeners=info.get("listeners"),
            playcount=info.get("playcount"),
        )
        save_contributor_cache(dst_conn, key, data)

        if n % 100 == 0:
            print(f"[contributors] fetched {n}/{len(contributors)}")


def enrich_albums(
        dst_conn: sqlite3.Connection,
        lastfm: LastFMClient,
        albums: dict[str, tuple[str, str]],
        enabled: bool,
        max_count: int | None = None,
) -> None:
    if not enabled:
        print("[albums] skipped")
        return

    n = 0
    for key, (artist_name, album_name) in albums.items():
        if max_count is not None and n >= max_count:
            break
        n += 1

        cached = get_album_cache(dst_conn, key)
        if cached is not None:
            if n % 1000 == 0:
                print(f"[albums] cache hit {n}/{len(albums)}")
            continue

        tags_raw = lastfm.album_top_tags(artist_name, album_name)
        tags = []
        for item in tags_raw[:15]:
            tag = safe_str(item.get("tag"))
            if tag:
                tags.append(tag)

        data = AlbumEnrich(
            artist_name=artist_name,
            album_name=album_name,
            raw_tags=uniq_keep_order(tags),
        )
        save_album_cache(dst_conn, key, data)

        if n % 200 == 0:
            print(f"[albums] fetched {n}/{len(albums)}")


# =========================
# Style / mood derive
# =========================

def normalize_style_tags(raw_tags: list[str], genre_text: str | None, style_text: str | None) -> list[str]:
    tag_blob = text_blob(" | ".join(raw_tags))
    metadata_blob = text_blob(genre_text, style_text)
    out: list[str] = []
    for canon, keys in STYLE_VOCAB.items():
        if any(contains_term(tag_blob, k) or contains_term(metadata_blob, k) for k in keys):
            out.append(canon)
    return uniq_keep_order(out)


def derive_mood_anchors(raw_tags: list[str], mood_text: str | None, style_tags: list[str]) -> tuple[list[str], float]:
    blob = text_blob(" ".join(raw_tags), mood_text)
    hits: list[str] = []

    for canon, keys in MOOD_ANCHOR_MAP.items():
        if any(k in blob for k in keys):
            hits.append(canon)

    for s in style_tags:
        hits.extend(STYLE_TO_MOOD.get(s, []))

    hits = uniq_keep_order(hits)

    strong_count = sum(1 for k in STRONG_MOOD_KEYS if k in blob)
    confidence = 0.0
    if hits:
        confidence = min(0.45 + 0.12 * strong_count, 0.95)
        if "dream_pop" in style_tags and "dreamy" in hits:
            confidence = max(confidence, 0.72)
        if "ambient" in style_tags and "calm" in hits:
            confidence = max(confidence, 0.70)

    if confidence < 0.65:
        return [], confidence
    return hits, confidence


def weighted_merge_tags(
        primary_tags: list[str],
        featured_tags: list[str],
        album_tags: list[str],
) -> list[str]:
    """
    这里不是精确概率，只是人为权重排序：
    primary > featured > album
    """
    scores: dict[str, float] = {}

    def add(tags: list[str], weight: float) -> None:
        for t in tags:
            k = t.lower().strip()
            if not k:
                continue
            scores[k] = scores.get(k, 0.0) + weight

    add(primary_tags, 1.0)
    add(featured_tags, 0.4)
    add(album_tags, 0.6)

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, _ in ranked]


def make_embedding_text(
        title: str | None,
        artist_name: str | None,
        album_name: str | None,
        release_year: int | None,
        language: str | None,
        vocal_type: str | None,
        genre_text: str | None,
        style_text: str | None,
        mood_text: str | None,
        primary_artists: list[str],
        featured_artists: list[str],
        style_tags: list[str],
        mood_anchors: list[str],
        raw_artist_tags: list[str],
        raw_album_tags: list[str],
        pop_bucket: str | None,
) -> str:
    parts: list[str] = []

    if safe_str(title):
        parts.append(f"Title: {title}")
    if safe_str(artist_name):
        parts.append(f"Artist: {artist_name}")
    if primary_artists:
        parts.append("Primary artists: " + ", ".join(primary_artists))
    if featured_artists:
        parts.append("Featured artists: " + ", ".join(featured_artists))
    if safe_str(album_name):
        parts.append(f"Album: {album_name}")
    if release_year is not None:
        parts.append(f"Year: {release_year}")
        era = year_to_era(release_year)
        if era:
            parts.append(f"Era: {era}")
    if safe_str(language):
        parts.append(f"Language: {language}")
    if safe_str(vocal_type):
        parts.append(f"Vocal: {vocal_type}")
    if safe_str(genre_text):
        parts.append(f"Genres: {genre_text}")
    if safe_str(style_text):
        parts.append(f"Styles: {style_text}")
    if safe_str(mood_text):
        parts.append(f"Mood text: {mood_text}")
    if style_tags:
        parts.append("Normalized styles: " + ", ".join(style_tags))
    if mood_anchors:
        parts.append("Mood anchors: " + ", ".join(mood_anchors))
    if raw_artist_tags:
        parts.append("Artist tags: " + ", ".join(raw_artist_tags[:12]))
    if raw_album_tags:
        parts.append("Album tags: " + ", ".join(raw_album_tags[:10]))
    if pop_bucket:
        parts.append(f"Popularity: {pop_bucket}")

    return join_nonempty(parts, " | ")


# =========================
# Build clean rows
# =========================

@dataclass
class CleanTrack:
    track_id: int
    title: str | None
    artist_name: str | None
    album_name: str | None
    release_year: int | None
    language: str | None
    vocal_type: str | None
    popularity: float | None
    lastfm_artist_listeners: float | None
    lastfm_artist_playcount: float | None
    popularity_proxy: float | None
    popularity_bucket: str | None
    primary_artists: list[str]
    featured_artists: list[str]
    all_contributors: list[str]
    raw_artist_tags: list[str]
    raw_album_tags: list[str]
    norm_style_tags: list[str]
    mood_anchors: list[str]
    mood_confidence: float
    genre_text: str | None
    style_text: str | None
    mood_text: str | None
    embedding_text: str


def build_contributor_profile_tables(dst_conn: sqlite3.Connection) -> None:
    dst_conn.execute("DELETE FROM contributors")

    rows = dst_conn.execute(
        """
        SELECT contributor_key, contributor_name, raw_tags_json, listeners, playcount
        FROM contributor_enrich_cache
        """
    ).fetchall()

    for key, name, raw_tags_json, listeners, playcount in rows:
        raw_tags = parse_json_tags(raw_tags_json, max_items=30)
        norm_style = normalize_style_tags(raw_tags, None, None)
        dst_conn.execute(
            """
            INSERT INTO contributors
            (contributor_key, contributor_name, raw_tags_json, norm_style_tags_json, listeners, playcount)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (key, name, json_dumps(raw_tags), json_dumps(norm_style), listeners, playcount),
        )
    dst_conn.commit()


def build_clean_track(
        row: sqlite3.Row,
        cols: set[str],
        dst_conn: sqlite3.Connection,
        track_id_override: int | None = None,
) -> CleanTrack:
    # Use provided track_id or generate from numeric field
    if track_id_override is not None:
        track_id = track_id_override
    else:
        numeric_id = pick(row, cols, "track_7digitalid", "id")
        if numeric_id is None:
            track_id_str = pick(row, cols, "track_id", "song_id")
            if track_id_str:
                numeric_id = int(hashlib.md5(str(track_id_str).encode()).hexdigest()[:8], 16)
            else:
                numeric_id = 0
        track_id = int(numeric_id) if numeric_id else 0
    title = safe_str(pick(row, cols, "title"))
    artist_name = safe_str(pick(row, cols, "artist_name", "artist"))
    album_name = safe_str(pick(row, cols, "album_name", "album", "release"))

    year_raw = pick(row, cols, "release_year", "year")
    try:
        release_year = int(year_raw) if year_raw is not None else None
    except Exception:
        release_year = None

    popularity_raw = pick(row, cols, "popularity")
    try:
        popularity = float(popularity_raw) if popularity_raw is not None else None
    except Exception:
        popularity = None

    genre_text = safe_str(pick(row, cols, "genre_text", "genre"))
    style_text = safe_str(pick(row, cols, "style_text", "style"))
    mood_text = safe_str(pick(row, cols, "mood_text", "mood"))
    language = safe_str(pick(row, cols, "language"))
    vocal_type = safe_str(pick(row, cols, "vocal_type"))

    primary_artists, featured_artists = split_artists(artist_name)
    all_contributors = uniq_keep_order(primary_artists + featured_artists)

    primary_artist_tags: list[str] = []
    featured_artist_tags: list[str] = []
    listeners_pool: list[float] = []
    playcount_pool: list[float] = []

    for name in primary_artists:
        key = contributor_key(name)
        cached = get_contributor_cache(dst_conn, key)
        if cached:
            primary_artist_tags.extend(cached.raw_tags)
            if cached.listeners is not None:
                listeners_pool.append(cached.listeners)
            if cached.playcount is not None:
                playcount_pool.append(cached.playcount)

    for name in featured_artists:
        key = contributor_key(name)
        cached = get_contributor_cache(dst_conn, key)
        if cached:
            featured_artist_tags.extend(cached.raw_tags)
            if cached.listeners is not None:
                listeners_pool.append(cached.listeners * 0.5)
            if cached.playcount is not None:
                playcount_pool.append(cached.playcount * 0.5)

    raw_album_tags: list[str] = []
    if artist_name and album_name:
        akey = album_key(artist_name, album_name)
        ac = get_album_cache(dst_conn, akey)
        if ac:
            raw_album_tags = ac.raw_tags

    raw_artist_tags = weighted_merge_tags(
        primary_tags=uniq_keep_order(primary_artist_tags),
        featured_tags=uniq_keep_order(featured_artist_tags),
        album_tags=uniq_keep_order(raw_album_tags),
    )

    style_tags = normalize_style_tags(raw_artist_tags, genre_text, style_text)
    mood_anchors, mood_conf = derive_mood_anchors(raw_artist_tags, mood_text, style_tags)

    language = infer_language_from_tags(raw_artist_tags + raw_album_tags, language)
    vocal_type = infer_vocal_type_from_tags(raw_artist_tags + raw_album_tags, vocal_type)

    avg_listeners = round(sum(listeners_pool) / len(listeners_pool), 2) if listeners_pool else None
    avg_playcount = round(sum(playcount_pool) / len(playcount_pool), 2) if playcount_pool else None

    pop_proxy = compute_popularity_proxy(popularity, avg_listeners, avg_playcount)
    pop_bucket = popularity_bucket(pop_proxy)

    emb_text = make_embedding_text(
        title=title,
        artist_name=artist_name,
        album_name=album_name,
        release_year=release_year,
        language=language,
        vocal_type=vocal_type,
        genre_text=genre_text,
        style_text=style_text,
        mood_text=mood_text,
        primary_artists=primary_artists,
        featured_artists=featured_artists,
        style_tags=style_tags,
        mood_anchors=mood_anchors,
        raw_artist_tags=raw_artist_tags,
        raw_album_tags=raw_album_tags,
        pop_bucket=pop_bucket,
    )

    return CleanTrack(
        track_id=track_id,
        title=title,
        artist_name=artist_name,
        album_name=album_name,
        release_year=release_year,
        language=language,
        vocal_type=vocal_type,
        popularity=popularity,
        lastfm_artist_listeners=avg_listeners,
        lastfm_artist_playcount=avg_playcount,
        popularity_proxy=pop_proxy,
        popularity_bucket=pop_bucket,
        primary_artists=primary_artists,
        featured_artists=featured_artists,
        all_contributors=all_contributors,
        raw_artist_tags=raw_artist_tags,
        raw_album_tags=raw_album_tags,
        norm_style_tags=style_tags,
        mood_anchors=mood_anchors,
        mood_confidence=mood_conf,
        genre_text=genre_text,
        style_text=style_text,
        mood_text=mood_text,
        embedding_text=emb_text,
    )


def write_tracks(dst_conn: sqlite3.Connection, clean_tracks: list[CleanTrack]) -> None:
    cur = dst_conn.cursor()
    cur.execute("DELETE FROM tracks")

    for t in clean_tracks:
        cur.execute(
            """
            INSERT INTO tracks (
                id, title, artist_name, album_name, release_year, language, vocal_type,
                popularity, lastfm_artist_listeners, lastfm_artist_playcount,
                popularity_proxy, popularity_bucket,
                primary_artists_json, featured_artists_json, all_contributors_json,
                raw_artist_tags_json, raw_album_tags_json,
                norm_style_tags_json, mood_anchors_json, mood_confidence,
                genre_text, style_text, mood_text, embedding_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                t.track_id,
                t.title,
                t.artist_name,
                t.album_name,
                t.release_year,
                t.language,
                t.vocal_type,
                t.popularity,
                t.lastfm_artist_listeners,
                t.lastfm_artist_playcount,
                t.popularity_proxy,
                t.popularity_bucket,
                json_dumps(t.primary_artists),
                json_dumps(t.featured_artists),
                json_dumps(t.all_contributors),
                json_dumps(t.raw_artist_tags),
                json_dumps(t.raw_album_tags),
                json_dumps(t.norm_style_tags),
                json_dumps(t.mood_anchors),
                t.mood_confidence,
                t.genre_text,
                t.style_text,
                t.mood_text,
                t.embedding_text,
            ),
        )

    dst_conn.commit()


# =========================
# Build embeddings / index
# =========================

def build_index(
        dst_db: Path,
        emb_path: Path,
        ids_path: Path,
        index_path: Path,
        batch_size: int,
) -> None:
    conn = sqlite3.connect(dst_db)
    rows = conn.execute(
        """
        SELECT id, embedding_text
        FROM tracks
        WHERE embedding_text IS NOT NULL AND embedding_text != ''
        ORDER BY id
        """
    ).fetchall()
    conn.close()

    if not rows:
        raise RuntimeError("No rows with embedding_text found in target DB.")

    ids = np.array([r[0] for r in rows], dtype=np.int64)
    texts = [r[1] for r in rows]

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    emb_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(emb_path, vecs)
    np.save(ids_path, ids)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(index_path))


# =========================
# Main pipeline
# =========================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-db", type=Path, required=True, help="旧库，必须包含 tracks 表")
    parser.add_argument("--dst-db", type=Path, default=Path("data/music_clean.db"))
    parser.add_argument("--emb-path", type=Path, default=Path("data/embeddings.npy"))
    parser.add_argument("--ids-path", type=Path, default=Path("data/ids.npy"))
    parser.add_argument("--index-path", type=Path, default=Path("data/faiss.index"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    parser.add_argument("--enable-lastfm", action="store_true")
    parser.add_argument("--enable-album-enrich", action="store_true")
    parser.add_argument("--lastfm-api-key", type=str, default=Settings.LASTFM_API_KEY)

    parser.add_argument("--max-contributors", type=int, default=None)
    parser.add_argument("--max-albums", type=int, default=None)

    args = parser.parse_args()

    if args.enable_lastfm and not args.lastfm_api_key:
        raise ValueError("enable-lastfm 已开启，但没有 LASTFM_API_KEY。")

    args.dst_db.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Step 1/6: read source rows")
    src_conn = sqlite3.connect(args.src_db)
    rows, cols = load_source_rows(src_conn)
    src_conn.close()
    print(f"source rows: {len(rows)}")

    print("=" * 80)
    print("Step 2/6: create target schema")
    dst_conn = sqlite3.connect(args.dst_db)
    create_schema(dst_conn)

    print("=" * 80)
    print("Step 3/6: collect contributors / albums")
    contributors = fetch_all_contributors(rows, cols)
    albums = fetch_all_albums(rows, cols)
    print(f"unique contributors: {len(contributors)}")
    print(f"unique albums: {len(albums)}")

    print("=" * 80)
    print("Step 4/6: enrich caches")
    if args.enable_lastfm:
        lastfm = LastFMClient(args.lastfm_api_key)
        enrich_contributors(
            dst_conn=dst_conn,
            lastfm=lastfm,
            contributors=contributors,
            max_count=args.max_contributors,
        )
        enrich_albums(
            dst_conn=dst_conn,
            lastfm=lastfm,
            albums=albums,
            enabled=args.enable_album_enrich,
            max_count=args.max_albums,
        )
    else:
        print("[lastfm] disabled")

    print("=" * 80)
    print("Step 5/6: build clean tracks")
    build_contributor_profile_tables(dst_conn)

    clean_tracks: list[CleanTrack] = []
    for i, row in enumerate(rows, start=1):
        clean_tracks.append(build_clean_track(row, cols, dst_conn, track_id_override=i))
        if i % 100000 == 0:
            print(f"built clean track docs: {i}/{len(rows)}")

    write_tracks(dst_conn, clean_tracks)
    dst_conn.close()

    style_nonempty = sum(1 for x in clean_tracks if x.norm_style_tags)
    mood_nonempty = sum(1 for x in clean_tracks if x.mood_anchors)
    pop_nonempty = sum(1 for x in clean_tracks if x.popularity_proxy is not None)
    multi_artist = sum(1 for x in clean_tracks if len(x.all_contributors) > 1)

    print("=" * 80)
    print("Step 6/6: build embeddings + faiss")
    build_index(
        dst_db=args.dst_db,
        emb_path=args.emb_path,
        ids_path=args.ids_path,
        index_path=args.index_path,
        batch_size=args.batch_size,
    )

    print("=" * 80)
    print(f"tracks written: {len(clean_tracks)}")
    print(f"multi-artist tracks: {multi_artist}")
    print(f"style-covered: {style_nonempty}")
    print(f"mood-anchor-covered: {mood_nonempty}")
    print(f"popularity_proxy-covered: {pop_nonempty}")
    print(f"target db: {args.dst_db}")
    print(f"embeddings: {args.emb_path}")
    print(f"ids: {args.ids_path}")
    print(f"faiss: {args.index_path}")


if __name__ == "__main__":
    main()
