import json
import re
import sqlite3
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


TRACK_SELECT_COLUMNS = """
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
    normalized_title,
    normalized_artist
"""


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_tag_json(raw: Any, max_items: int = 12) -> List[str]:
    if raw is None:
        return []

    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except Exception:
            return []
    else:
        data = raw

    tags: List[str] = []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                tag = item.get("tag")
                if tag:
                    tags.append(str(tag).strip())
            elif isinstance(item, str):
                t = item.strip()
                if t:
                    tags.append(t)

    # 去重保序
    seen = set()
    out = []
    for t in tags:
        nt = normalize_text(t)
        if not nt or nt in seen:
            continue
        seen.add(nt)
        out.append(t)
        if len(out) >= max_items:
            break

    return out


class MusicRetriever:
    def __init__(
            self,
            db_path: str,
            index_path: str,
            ids_path: str,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.db_path = db_path
        self.index = faiss.read_index(index_path)
        self.song_ids = np.load(ids_path)
        self.model = SentenceTransformer(model_name)

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _row_to_track(self, row) -> Dict[str, Any]:
        tags = parse_tag_json(row[12])
        artist_tags = parse_tag_json(row[13])

        return {
            "id": row[0],
            "title": row[1],
            "artist": row[2],
            "album": row[3],
            "release_year": row[4],
            "popularity": row[5],
            "popularity_bucket": row[6],
            "language": row[7],
            "vocal_type": row[8],
            "genre_text": row[9],
            "style_text": row[10],
            "mood_text": row[11],
            "tags_json": row[12],
            "artist_tags_json": row[13],
            "tags": tags,
            "artist_tags": artist_tags,
            "normalized_title": row[14],
            "normalized_artist": row[15],
        }

    def _fetch_track_by_id(self, song_id: int) -> Dict[str, Any] | None:
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute(
            f"""
            SELECT
                {TRACK_SELECT_COLUMNS}
            FROM tracks
            WHERE id = ?
            """,
            (int(song_id),),
        )
        row = cur.fetchone()
        conn.close()

        if row is None:
            return None

        return self._row_to_track(row)

    def _fetch_tracks_by_ids(self, song_ids: List[int]) -> List[Dict[str, Any]]:
        if not song_ids:
            return []

        conn = self._get_connection()
        cur = conn.cursor()

        placeholders = ",".join(["?"] * len(song_ids))
        cur.execute(
            f"""
            SELECT
                {TRACK_SELECT_COLUMNS}
            FROM tracks
            WHERE id IN ({placeholders})
            """,
            tuple(int(x) for x in song_ids),
        )
        rows = cur.fetchall()
        conn.close()

        track_map = {row[0]: self._row_to_track(row) for row in rows}
        return [track_map[sid] for sid in song_ids if sid in track_map]

    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        query = query.strip()
        if not query:
            return []

        query_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(query_vec, top_k)

        ordered_song_ids: List[int] = []
        ordered_scores: List[float] = []

        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            ordered_song_ids.append(int(self.song_ids[idx]))
            ordered_scores.append(float(score))

        tracks = self._fetch_tracks_by_ids(ordered_song_ids)

        score_map = {sid: score for sid, score in zip(ordered_song_ids, ordered_scores)}
        for track in tracks:
            track["score"] = score_map.get(track["id"], 0.0)
            track["match_type"] = "semantic"

        return tracks

    def search_by_artist(self, artist_query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """
        只给纯 artist query 用。
        命中则按 popularity 和年份做简单排序。
        """
        raw_query = artist_query.strip()
        norm_query = normalize_text(raw_query)
        if not norm_query:
            return []

        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute(
            f"""
            SELECT
                {TRACK_SELECT_COLUMNS}
            FROM tracks
            WHERE normalized_artist = ?
            ORDER BY
                CASE WHEN popularity IS NULL THEN 1 ELSE 0 END,
                popularity DESC,
                CASE WHEN release_year IS NULL THEN 1 ELSE 0 END,
                release_year DESC,
                id ASC
            LIMIT ?
            """,
            (norm_query, top_k),
        )
        rows = cur.fetchall()

        if not rows:
            cur.execute(
                f"""
                SELECT
                    {TRACK_SELECT_COLUMNS}
                FROM tracks
                WHERE LOWER(artist_name) = LOWER(?)
                ORDER BY
                    CASE WHEN popularity IS NULL THEN 1 ELSE 0 END,
                    popularity DESC,
                    CASE WHEN release_year IS NULL THEN 1 ELSE 0 END,
                    release_year DESC,
                    id ASC
                LIMIT ?
                """,
                (raw_query, top_k),
            )
            rows = cur.fetchall()

        conn.close()

        results = []
        for i, row in enumerate(rows):
            track = self._row_to_track(row)
            # 仅供 artist 精确匹配路径内部展示排序使用
            track["score"] = 1.0 - i * 0.001
            track["match_type"] = "artist_exact"
            results.append(track)

        return results