from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorRetriever:
    """
    Retrieve music candidates from:
    1. FAISS semantic search
    2. SQLite artist lookup
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
            self,
            db_path: str,
            index_path: str,
            ids_path: str,
            model_name: str | None = None,
    ) -> None:
        self.db_path = str(db_path)
        self.index_path = str(index_path)
        self.ids_path = str(ids_path)
        self.model_name = model_name or self.MODEL_NAME

        self._validate_runtime_files()

        self.model = SentenceTransformer(self.model_name)
        self.index = faiss.read_index(self.index_path)
        self.ids = np.load(self.ids_path)

        if self.ids.ndim != 1:
            raise ValueError(f"ids.npy must be 1-D, got shape={self.ids.shape}")

        if self.index.ntotal != len(self.ids):
            raise ValueError(
                "FAISS index size does not match ids.npy length: "
                f"index.ntotal={self.index.ntotal}, ids={len(self.ids)}"
            )

    # =========================
    # Public APIs
    # =========================

    def search_semantic(self, text: str, top_k: int = 50) -> list[dict[str, Any]]:
        """
        Semantic retrieval via sentence-transformers + FAISS.
        """
        query = self._normalize_text(text)
        if not query:
            return []

        top_k = self._clamp_top_k(top_k)

        query_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(query_vec, top_k)

        pairs: list[tuple[int, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            if idx >= len(self.ids):
                continue

            song_id = int(self.ids[idx])
            pairs.append((song_id, float(score)))

        return self._hydrate_candidates(
            id_score_pairs=pairs,
            match_type="semantic",
        )

    def search_by_artist(self, artist_name: str, top_k: int = 50) -> list[dict[str, Any]]:
        """
        Artist lookup:
        1. Exact normalized match
        2. Prefix/LIKE fallback
        """
        artist_query = self._normalize_text(artist_name)
        if not artist_query:
            return []

        top_k = self._clamp_top_k(top_k)

        exact_rows = self._search_artist_exact(artist_query, top_k=top_k)
        if exact_rows:
            return exact_rows

        return self._search_artist_like(artist_query, top_k=top_k)

    # =========================
    # Artist lookup
    # =========================

    def _search_artist_exact(self, artist_name: str, top_k: int) -> list[dict[str, Any]]:
        sql = """
              SELECT
                  id,
                  title,
                  artist_name AS artist,
                  album_name AS album,
                  release_year,
                  popularity,
                  popularity_bucket,
                  language,
                  vocal_type,
                  genre_text,
                  style_text,
                  mood_text,
                  tags_json,
                  artist_tags_json
              FROM tracks
              WHERE lower(trim(artist_name)) = ?
              ORDER BY
                  popularity DESC NULLS LAST,
                  release_year DESC NULLS LAST,
                  id ASC
                  LIMIT ? \
              """

        rows = self._fetch_rows(
            sql=sql,
            params=(artist_name, top_k),
        )

        results: list[dict[str, Any]] = []
        for rank, row in enumerate(rows):
            # 给 exact artist 一个稳定高分，方便上层识别
            score = 1.0 - rank * 0.001
            results.append(
                self._row_to_candidate(
                    row=row,
                    score=score,
                    match_type="artist_exact",
                )
            )
        return results

    def _search_artist_like(self, artist_name: str, top_k: int) -> list[dict[str, Any]]:
        like_query = f"%{artist_name}%"

        sql = """
              SELECT
                  id,
                  title,
                  artist_name AS artist,
                  album_name AS album,
                  release_year,
                  popularity,
                  popularity_bucket,
                  language,
                  vocal_type,
                  genre_text,
                  style_text,
                  mood_text,
                  tags_json,
                  artist_tags_json
              FROM tracks
              WHERE lower(artist_name) LIKE ?
              ORDER BY
                  CASE
                  WHEN lower(trim(artist_name)) = ? THEN 0
                  WHEN lower(artist_name) LIKE ? THEN 1
                  ELSE 2
              END,
                popularity DESC NULLS LAST,
                release_year DESC NULLS LAST,
                id ASC
            LIMIT ? \
              """

        prefix_query = f"{artist_name}%"
        rows = self._fetch_rows(
            sql=sql,
            params=(like_query, artist_name, prefix_query, top_k),
        )

        results: list[dict[str, Any]] = []
        for rank, row in enumerate(rows):
            # like match 给一个略低于 exact 的分数
            score = 0.92 - rank * 0.001
            results.append(
                self._row_to_candidate(
                    row=row,
                    score=score,
                    match_type="artist_like",
                )
            )
        return results

    # =========================
    # Candidate hydration
    # =========================

    def _hydrate_candidates(
            self,
            id_score_pairs: list[tuple[int, float]],
            match_type: str,
    ) -> list[dict[str, Any]]:
        if not id_score_pairs:
            return []

        ids = [song_id for song_id, _ in id_score_pairs]
        score_map = {song_id: score for song_id, score in id_score_pairs}

        placeholders = ",".join("?" for _ in ids)
        sql = f"""
            SELECT
                id,
                title,
                artist_name AS artist,
                album_name AS album,
                release_year,
                popularity,
                popularity_bucket,
                language,
                vocal_type,
                genre_text,
                style_text,
                mood_text,
                tags_json,
                artist_tags_json
            FROM tracks
            WHERE id IN ({placeholders})
        """

        rows = self._fetch_rows(sql=sql, params=tuple(ids))
        row_map = {int(row["id"]): row for row in rows}

        out: list[dict[str, Any]] = []
        for song_id, score in id_score_pairs:
            row = row_map.get(int(song_id))
            if row is None:
                continue

            out.append(
                self._row_to_candidate(
                    row=row,
                    score=score_map[song_id],
                    match_type=match_type,
                )
            )

        return out

    def _row_to_candidate(
            self,
            row: sqlite3.Row,
            score: float,
            match_type: str,
    ) -> dict[str, Any]:
        track_tags = self._parse_tag_json(row["tags_json"])
        artist_tags = self._parse_tag_json(row["artist_tags_json"])

        merged_tags: list[str] = []
        seen: set[str] = set()
        for tag in track_tags + artist_tags:
            key = tag.lower()
            if key not in seen:
                seen.add(key)
                merged_tags.append(tag)

        return {
            "id": row["id"],
            "title": row["title"],
            "artist": row["artist"],
            "album": row["album"],
            "release_year": row["release_year"],
            "popularity": row["popularity"],
            "popularity_bucket": row["popularity_bucket"],
            "language": row["language"],
            "vocal_type": row["vocal_type"],
            "genre_text": row["genre_text"],
            "style_text": row["style_text"],
            "mood_text": row["mood_text"],
            "tags": merged_tags,
            "score": float(score),
            "match_type": match_type,
        }

    # =========================
    # DB helpers
    # =========================

    def _fetch_rows(
            self,
            sql: str,
            params: tuple[Any, ...] = (),
    ) -> list[sqlite3.Row]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            return cur.fetchall()
        finally:
            conn.close()

    # =========================
    # Utility helpers
    # =========================

    def _validate_runtime_files(self) -> None:
        required = [
            Path(self.db_path),
            Path(self.index_path),
            Path(self.ids_path),
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required retrieval files:\n" + "\n".join(missing)
            )

    def _clamp_top_k(self, top_k: int) -> int:
        if top_k <= 0:
            return 1
        return min(int(top_k), len(self.ids))

    def _normalize_text(self, value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).strip().lower().split())

    def _parse_tag_json(self, raw: Any) -> list[str]:
        if raw is None:
            return []

        if isinstance(raw, list):
            return self._dedupe_preserve_order(
                [str(x).strip() for x in raw if str(x).strip()]
            )

        text = str(raw).strip()
        if not text:
            return []

        try:
            parsed = json.loads(text)
        except Exception:
            return [text]

        tags: list[str] = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str):
                    tag = item.strip()
                    if tag:
                        tags.append(tag)
                elif isinstance(item, dict):
                    tag = str(item.get("tag", "")).strip()
                    if tag:
                        tags.append(tag)

        return self._dedupe_preserve_order(tags)

    def _dedupe_preserve_order(self, values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()

        for value in values:
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(value)

        return out