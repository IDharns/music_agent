from __future__ import annotations

import json
import re
from typing import Any


class Reranker:
    DREAMY_TERMS = {
        "dreamy",
        "dream pop",
        "ethereal",
        "shoegaze",
        "ambient",
        "atmospheric",
        "lush",
        "hazy",
        "floaty",
        "soft",
        "chill",
        "mellow",
    }

    SAD_TERMS = {
        "sad",
        "melancholy",
        "melancholic",
        "sorrow",
        "heartbreak",
        "heartbroken",
        "emo",
        "tearful",
        "blue",
        "despair",
        "lonely",
    }

    ACOUSTIC_PROXY_TERMS = {
        "folk",
        "singer songwriter",
        "singer-songwriter",
        "country folk",
        "indie folk",
        "americana",
    }

    LIVE_PATTERNS = (
        " live ",
        "(live",
        "[live",
        "live version",
        "live at",
        "session",
        "hotel cafe",
        "premiere performance",
        "background vocals",
    )

    REMIX_PATTERNS = (
        " remix",
        "(remix",
        "[remix",
        " vocal mix",
        " dub mix",
        " extended mix",
        " club mix",
        " rework",
        " edit)",
    )

    ACOUSTIC_PATTERNS = (
        " acoustic",
        "(acoustic",
        "[acoustic",
        " unplugged",
        " acoustic version",
    )

    INSTRUMENTAL_PATTERNS = (
        " instrumental",
        "(instrumental",
        "[instrumental",
    )

    TITLE_NOISE_PATTERNS = (
        "made famous by",
        "karaoke",
        "tribute",
        "cover version",
        "performance track",
        "originally performed by",
    )

    def rerank(
            self,
            raw_candidates: list[dict[str, Any]],
            parsed_query: dict[str, Any],
            final_k: int = 10,
            max_per_artist: int = 3,
    ) -> list[dict[str, Any]]:
        rescored: list[dict[str, Any]] = []

        for item in raw_candidates:
            result = self._score_one(item, parsed_query)
            if result is None:
                continue

            new_score, similarity, tag_overlap = result
            out = dict(item)
            out["score"] = round(float(new_score), 6)
            out["similarity"] = round(float(similarity), 6)
            out["tag_overlap"] = round(float(tag_overlap), 6)
            rescored.append(out)

        rescored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return self._dedup_and_diversify(rescored, final_k, max_per_artist)

    def _score_one(
            self,
            item: dict[str, Any],
            parsed_query: dict[str, Any],
    ) -> tuple[float, float, float] | None:
        """Returns (final_score, similarity, tag_overlap) or None if filtered."""
        title = self._norm(item.get("title"))
        artist = self._norm(item.get("artist"))
        album = self._norm(item.get("album"))
        genre_text = self._norm(item.get("genre_text"))
        style_text = self._norm(item.get("style_text"))
        mood_text = self._norm(item.get("mood_text"))
        style_tags_blob = self._tags_blob(item.get("style_tags")).replace("_", " ")
        mood_anchors_blob = self._tags_blob(item.get("mood_anchors")).replace("_", " ")
        artist_tags_blob = self._tags_blob(item.get("artist_tags")).replace("_", " ")
        album_tags_blob = self._tags_blob(item.get("album_tags")).replace("_", " ")
        vocal_type = self._norm(item.get("vocal_type"))
        popularity = item.get("popularity")
        popularity_proxy = item.get("popularity_proxy")

        tags_blob = self._tags_blob(item.get("tags"))

        meta_blob = " | ".join(
            x for x in [
                artist,
                album,
                genre_text,
                style_text,
                mood_text,
                style_tags_blob,
                mood_anchors_blob,
                vocal_type,
                artist_tags_blob,
                album_tags_blob,
                tags_blob,
            ]
            if x
        )
        full_blob = " | ".join(x for x in [title, meta_blob] if x)

        query_type = parsed_query.get("query_type", "semantic")
        genres = {self._norm(x) for x in parsed_query.get("genres", []) if self._norm(x)}
        moods = {self._norm(x) for x in parsed_query.get("moods", []) if self._norm(x)}
        includes = {self._norm(x) for x in parsed_query.get("include", []) if self._norm(x)}
        excludes = {self._norm(x) for x in parsed_query.get("exclude", []) if self._norm(x)}
        artist_seeds = {self._norm(x) for x in parsed_query.get("artist_seeds", []) if self._norm(x)}

        vocal_pref = self._norm(parsed_query.get("vocal"))
        popularity_pref = self._norm(parsed_query.get("popularity_preference"))

        semantic_terms = self._semantic_terms(parsed_query)
        similarity = float(item.get("score", 0.0))
        tag_overlap = self._tag_overlap(
            semantic_terms=semantic_terms,
            style_tags_blob=style_tags_blob,
            mood_anchors_blob=mood_anchors_blob,
            artist_tags_blob=artist_tags_blob,
            album_tags_blob=album_tags_blob,
        )
        score = 0.7 * similarity + 0.3 * tag_overlap
        has_structured_constraints = bool(
            genres
            or moods
            or includes
            or artist_seeds
            or (vocal_pref and vocal_pref != "unknown")
            or (popularity_pref and popularity_pref != "unknown")
        )

        if self._contains_any(title, self.TITLE_NOISE_PATTERNS):
            return None

        # 默认轻惩罚
        if self._contains_any(full_blob, self.LIVE_PATTERNS):
            score -= 0.42
        if self._contains_any(full_blob, self.REMIX_PATTERNS):
            score -= 0.42

        # 显式排除
        if "live" in excludes and self._contains_any(full_blob, self.LIVE_PATTERNS):
            return None
        if "remix" in excludes and self._contains_any(full_blob, self.REMIX_PATTERNS):
            return None
        if "acoustic" in excludes and self._contains_any(full_blob, self.ACOUSTIC_PATTERNS):
            return None
        if "classical" in excludes and (
                "classical" in meta_blob or "neoclassical" in meta_blob or "neo classical" in meta_blob
        ):
            return None
        if "instrumental" in excludes and (
                "instrumental" in vocal_type or self._contains_any(full_blob, self.INSTRUMENTAL_PATTERNS)
        ):
            return None

        # mixed query 里 seed artist 本人不该霸榜
        if query_type == "mixed" and artist in artist_seeds:
            score -= 0.35

        title_hits = sum(1 for term in semantic_terms if term in title)
        meta_hits = sum(1 for term in semantic_terms if term in meta_blob)

        if title_hits >= 1 and meta_hits == 0:
            score -= 0.35 if has_structured_constraints else 0.22

        if has_structured_constraints and title_hits >= 1 and meta_hits == 0 and tag_overlap == 0:
            return None

        dreamy_required = bool(moods & {"dreamy", "ethereal", "soft", "mellow"})
        dreamy_ok = self._contains_any(meta_blob, self.DREAMY_TERMS) or "dreamy" in mood_anchors_blob

        if dreamy_required and dreamy_ok:
            score += 0.22
        elif dreamy_required and title_hits >= 1 and not dreamy_ok:
            score -= 0.15

        sad_required = "sad" in moods
        sad_ok = self._contains_any(meta_blob, self.SAD_TERMS) or "melancholic" in mood_anchors_blob

        if sad_required and sad_ok:
            score += 0.18
        elif sad_required and title_hits >= 1 and not sad_ok:
            score -= 0.10

        style_hit_count = 0
        for term in semantic_terms:
            if term in meta_blob:
                style_hit_count += 1
        if style_hit_count:
            score += min(style_hit_count, 3) * 0.12
        elif has_structured_constraints and title_hits == 0 and tag_overlap == 0 and similarity < 0.35:
            return None

        if vocal_pref == "female vocal":
            if vocal_type == "female vocal" or "female vocal" in style_tags_blob:
                score += 0.14
            elif vocal_type == "male vocal" or "male vocal" in style_tags_blob:
                score -= 0.20
            elif "instrumental" in vocal_type:
                score -= 0.16

        elif vocal_pref == "male vocal":
            if vocal_type == "male vocal" or "male vocal" in style_tags_blob:
                score += 0.14
            elif vocal_type == "female vocal" or "female vocal" in style_tags_blob:
                score -= 0.20
            elif "instrumental" in vocal_type:
                score -= 0.16

        wants_pop_family = bool(genres & {"pop", "indie pop", "dream pop", "country pop"})
        if wants_pop_family and (
                "hip-hop" in meta_blob or "hip hop" in meta_blob or "rap" in meta_blob
        ):
            score -= 0.28

        if popularity_pref == "less_popular":
            if popularity is not None:
                try:
                    p = float(popularity)
                    if p >= 70:
                        score -= 0.20
                    elif p <= 40:
                        score += 0.10
                except Exception:
                    pass
            elif popularity_proxy is not None:
                try:
                    pp = float(popularity_proxy)
                    if pp >= 0.8:
                        score -= 0.16
                    elif pp <= 0.3:
                        score += 0.08
                except Exception:
                    pass

        if "acoustic" in includes:
            if self._contains_any(full_blob, self.ACOUSTIC_PATTERNS) or "acoustic" in meta_blob:
                score += 0.18
            elif self._contains_any(meta_blob, self.ACOUSTIC_PROXY_TERMS):
                score += 0.10

        return score, similarity, tag_overlap

    def _tag_overlap(
            self,
            semantic_terms: set[str],
            style_tags_blob: str,
            mood_anchors_blob: str,
            artist_tags_blob: str,
            album_tags_blob: str,
    ) -> float:
        """Fraction of semantic query terms that appear in the track's tag fields."""
        if not semantic_terms:
            return 0.0
        tag_blob = " | ".join(
            x for x in [style_tags_blob, mood_anchors_blob, artist_tags_blob, album_tags_blob]
            if x
        )
        hits = sum(1 for term in semantic_terms if term in tag_blob)
        return hits / len(semantic_terms)

    def _semantic_terms(self, parsed_query: dict[str, Any]) -> set[str]:
        terms: set[str] = set()

        for key in ("genres", "moods", "include"):
            for value in parsed_query.get(key, []) or []:
                v = self._norm(value)
                if v and len(v) >= 3:
                    terms.add(v)

        vocal = self._norm(parsed_query.get("vocal"))
        if vocal and vocal != "unknown":
            terms.add(vocal)

        energy = self._norm(parsed_query.get("energy"))
        if energy and energy != "unknown":
            terms.add(energy)

        pop_pref = self._norm(parsed_query.get("popularity_preference"))
        if pop_pref == "less_popular":
            terms.update({"underrated", "less popular"})

        return terms

    def _tags_blob(self, value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, list):
            return " ".join(self._norm(v) for v in value if self._norm(v))

        if isinstance(value, str):
            s = value.strip()
            if not s:
                return ""
            try:
                parsed = json.loads(s)
            except Exception:
                return self._norm(s)

            out: list[str] = []
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, str):
                        n = self._norm(item)
                        if n:
                            out.append(n)
                    elif isinstance(item, dict):
                        tag = self._norm(item.get("tag"))
                        if tag:
                            out.append(tag)
            return " ".join(out)

        return ""

    def _contains_any(self, text: str, patterns: set[str] | tuple[str, ...]) -> bool:
        if not text:
            return False
        return any(p in text for p in patterns)

    def _dedup_and_diversify(
            self,
            items: list[dict[str, Any]],
            final_k: int,
            max_per_artist: int,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        artist_counts: dict[str, int] = {}

        for item in items:
            title = self._norm(item.get("title"))
            artist = self._norm(item.get("artist"))
            if not title or not artist:
                continue

            key = (title, artist)
            if key in seen:
                continue

            if artist_counts.get(artist, 0) >= max_per_artist:
                continue

            seen.add(key)
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
            out.append(item)

            if len(out) >= final_k:
                break

        return out

    def _norm(self, value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).strip().lower().split())
