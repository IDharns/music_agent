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
        "soft",
        "lush",
        "atmospheric",
        "chill",
        "mellow",
        "floaty",
        "hazy",
    }

    SAD_TERMS = {
        "sad",
        "melancholy",
        "melancholic",
        "heartbreak",
        "heartbroken",
        "tearful",
        "blue",
        "sorrow",
        "emotional",
    }

    SYNTHPOP_TERMS = {
        "synthpop",
        "synth-pop",
        "new wave",
        "electropop",
        "electro pop",
        "synth",
        "synthesizer",
    }

    LIVE_PATTERNS = (
        " live ",
        "(live",
        "[live",
        "live version",
        "live at",
        "concert",
        "session",
        "peel session",
        "premiere performance",
        "mtv unplugged",
    )

    REMIX_PATTERNS = (
        " remix",
        "(remix",
        "[remix",
        " mix)",
        " mix ]",
        " remastered",
        " edit)",
        " dub mix",
        " extended mix",
        " club mix",
    )

    ACOUSTIC_PATTERNS = (
        " acoustic",
        "(acoustic",
        "[acoustic",
        " unplugged",
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
        "originally by",
        "sound-alike",
    )

    BAD_MIXED_PATTERNS = (
        "interview",
        "podcast",
        "commentary",
        "spoken word",
    )

    FEMALE_HINTS = (
        "female vocal",
        "female vocals",
        "female vocalist",
        "female vocalists",
        "女声",
    )

    MALE_HINTS = (
        "male vocal",
        "male vocals",
        "male vocalist",
        "male vocalists",
        "男声",
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
            new_score = self._score_one(item, parsed_query)
            if new_score is None:
                continue

            out = dict(item)
            out["score"] = round(float(new_score), 6)
            rescored.append(out)

        rescored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return self._dedup_and_diversify(
            rescored,
            final_k=final_k,
            max_per_artist=max_per_artist,
        )

    def _score_one(
            self,
            item: dict[str, Any],
            parsed_query: dict[str, Any],
    ) -> float | None:
        score = float(item.get("score", 0.0))

        title = self._norm(item.get("title"))
        artist = self._norm(item.get("artist"))
        album = self._norm(item.get("album"))
        genre_text = self._norm(item.get("genre_text"))
        style_text = self._norm(item.get("style_text"))
        mood_text = self._norm(item.get("mood_text"))
        vocal_type = self._norm(item.get("vocal_type"))
        language = self._norm(item.get("language"))
        popularity_bucket = self._norm(item.get("popularity_bucket"))
        match_type = self._norm(item.get("match_type"))
        popularity = item.get("popularity")
        release_year = item.get("release_year")

        tags = self._tags_blob(item.get("tags"))

        meta_blob = " | ".join(
            x for x in [
                artist,
                album,
                genre_text,
                style_text,
                mood_text,
                vocal_type,
                language,
                popularity_bucket,
                tags,
            ]
            if x
        )
        full_blob = " | ".join(x for x in [title, meta_blob] if x)

        query_type = parsed_query.get("query_type", "semantic")
        artist_seeds = {
            self._norm(x)
            for x in (parsed_query.get("artist_seeds") or [])
            if self._norm(x)
        }
        genres = {
            self._norm(x)
            for x in (parsed_query.get("genres") or [])
            if self._norm(x)
        }
        moods = {
            self._norm(x)
            for x in (parsed_query.get("moods") or [])
            if self._norm(x)
        }
        includes = {
            self._norm(x)
            for x in (parsed_query.get("include") or [])
            if self._norm(x)
        }
        excludes = {
            self._norm(x)
            for x in (parsed_query.get("exclude") or [])
            if self._norm(x)
        }

        vocal_pref = self._norm(parsed_query.get("vocal"))
        energy_pref = self._norm(parsed_query.get("energy"))
        era_pref = parsed_query.get("era")
        popularity_pref = self._norm(parsed_query.get("popularity_preference"))

        # 0) 明显噪声内容直接丢掉
        if self._contains_any(title, self.TITLE_NOISE_PATTERNS) or self._contains_any(artist, self.TITLE_NOISE_PATTERNS):
            return None

        if query_type == "mixed" and (
                self._contains_any(title, self.BAD_MIXED_PATTERNS)
                or self._contains_any(artist, self.BAD_MIXED_PATTERNS)
        ):
            return None

        # 1) mixed query 不允许 seed artist 本人霸榜
        if query_type == "mixed":
            if match_type == "artist_exact":
                score -= 0.35

            if artist and artist in artist_seeds:
                score -= 0.35

            for seed in artist_seeds:
                if seed and seed in title and artist != seed:
                    score -= 0.18
                if seed and seed in album and artist != seed:
                    score -= 0.10

        # 2) exclude 规则先处理
        if "live" in excludes and self._contains_any(full_blob, self.LIVE_PATTERNS):
            return None
        if "remix" in excludes and self._contains_any(full_blob, self.REMIX_PATTERNS):
            return None
        if "acoustic" in excludes and self._contains_any(full_blob, self.ACOUSTIC_PATTERNS):
            return None
        if "instrumental" in excludes and (
                "instrumental" in vocal_type or self._contains_any(full_blob, self.INSTRUMENTAL_PATTERNS)
        ):
            return None

        # 3) title-only 命中惩罚
        semantic_terms = self._semantic_terms(parsed_query)
        title_hits = sum(1 for term in semantic_terms if term in title)
        meta_hits = sum(1 for term in semantic_terms if term in meta_blob)

        if title_hits >= 1 and meta_hits == 0:
            score -= 0.22

        # 4) 风格/标签真实命中加分
        style_hit_count = 0
        for term in semantic_terms:
            if term in meta_blob:
                style_hit_count += 1
        if style_hit_count:
            score += min(style_hit_count, 3) * 0.12

        # 5) 梦幻语义强化
        dreamy_required = bool(moods & {"dreamy", "ethereal", "soft", "mellow"})
        dreamy_ok = self._contains_any(meta_blob, self.DREAMY_TERMS)

        if dreamy_required and dreamy_ok:
            score += 0.18
        if dreamy_required and not dreamy_ok and title_hits >= 1:
            score -= 0.15

        # 6) sad 语义强化
        sad_required = "sad" in moods
        sad_ok = self._contains_any(meta_blob, self.SAD_TERMS)
        if sad_required and sad_ok:
            score += 0.15
        elif sad_required and title_hits >= 1 and not sad_ok:
            score -= 0.10

        # 7) synthpop / 80s 语义强化
        synth_required = "synthpop" in genres or self._contains_any(" ".join(genres), self.SYNTHPOP_TERMS)
        synth_ok = self._contains_any(meta_blob, self.SYNTHPOP_TERMS)
        if synth_required and synth_ok:
            score += 0.18
        elif synth_required and not synth_ok:
            score -= 0.08

        if self._era_matches(release_year, era_pref):
            score += 0.10
        elif era_pref:
            score -= 0.04

        # 8) vocal 偏好
        if vocal_pref and vocal_pref != "unknown":
            if "female" in vocal_pref:
                if "female" in vocal_type or self._contains_any(meta_blob, self.FEMALE_HINTS):
                    score += 0.14
                elif "male" in vocal_type:
                    score -= 0.08
                elif "instrumental" in vocal_type:
                    score -= 0.12

            elif "male" in vocal_pref:
                if "male" in vocal_type or self._contains_any(meta_blob, self.MALE_HINTS):
                    score += 0.14
                elif "female" in vocal_type:
                    score -= 0.08
                elif "instrumental" in vocal_type:
                    score -= 0.12

            elif "instrumental" in vocal_pref:
                if "instrumental" in vocal_type or self._contains_any(full_blob, self.INSTRUMENTAL_PATTERNS):
                    score += 0.14
                else:
                    score -= 0.08

        # 9) include 规则
        if "acoustic" in includes:
            if self._contains_any(full_blob, self.ACOUSTIC_PATTERNS) or "acoustic" in meta_blob:
                score += 0.10
            else:
                score -= 0.04

        if "ambient" in includes and "ambient" in meta_blob:
            score += 0.10

        if "singer songwriter" in includes and (
                "singer songwriter" in meta_blob
                or "songwriter" in meta_blob
        ):
            score += 0.10

        # 10) popularity 偏好
        if popularity_pref == "less_popular":
            if popularity is None:
                score -= 0.03
            else:
                try:
                    p = float(popularity)
                    if p >= 70:
                        score -= 0.20
                    elif p <= 40:
                        score += 0.10
                except Exception:
                    pass

            if popularity_bucket:
                if "high" in popularity_bucket or "popular" in popularity_bucket:
                    score -= 0.12
                if "low" in popularity_bucket or "obscure" in popularity_bucket:
                    score += 0.08

        elif popularity_pref == "more_popular":
            if popularity is not None:
                try:
                    p = float(popularity)
                    if p >= 70:
                        score += 0.10
                except Exception:
                    pass

        # 11) 轻量 energy 处理
        if energy_pref and energy_pref != "unknown":
            if energy_pref == "high":
                if self._contains_any(meta_blob, ("energetic", "upbeat", "fast", "dance")):
                    score += 0.08
            elif energy_pref == "low":
                if self._contains_any(meta_blob, ("calm", "soft", "mellow", "chill", "ambient")):
                    score += 0.08

        # 12) mixed 查询的 seed-style 偏好加分
        if query_type == "mixed":
            if "taylor swift" in artist_seeds:
                if "female" in vocal_type:
                    score += 0.06
                if "pop" in genre_text or "pop" in style_text:
                    score += 0.06
                if "country pop" in meta_blob:
                    score += 0.10
                if "singer songwriter" in meta_blob or "songwriter" in meta_blob:
                    score += 0.08
                if dreamy_ok:
                    score += 0.06

        return score

    def _dedup_and_diversify(
            self,
            items: list[dict[str, Any]],
            final_k: int,
            max_per_artist: int,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen_tracks: set[tuple[str, str]] = set()
        artist_counts: dict[str, int] = {}

        for item in items:
            title = self._norm(item.get("title"))
            artist = self._norm(item.get("artist"))

            if not title or not artist:
                continue

            track_key = (title, artist)
            if track_key in seen_tracks:
                continue

            count = artist_counts.get(artist, 0)
            if count >= max_per_artist:
                continue

            seen_tracks.add(track_key)
            artist_counts[artist] = count + 1
            out.append(item)

            if len(out) >= final_k:
                break

        return out

    def _semantic_terms(self, parsed_query: dict[str, Any]) -> set[str]:
        terms: set[str] = set()

        for key in ("genres", "moods", "include"):
            values = parsed_query.get(key) or []
            if isinstance(values, list):
                for value in values:
                    v = self._norm(value)
                    if v and len(v) >= 3:
                        terms.add(v)

        vocal = self._norm(parsed_query.get("vocal"))
        if vocal and vocal != "unknown":
            terms.add(vocal)

        energy = self._norm(parsed_query.get("energy"))
        if energy and energy != "unknown":
            terms.add(energy)

        era = self._norm(parsed_query.get("era"))
        if era:
            terms.add(era)

        popularity_pref = self._norm(parsed_query.get("popularity_preference"))
        if popularity_pref == "less_popular":
            terms.update({"underrated", "less popular"})
        elif popularity_pref == "more_popular":
            terms.add("popular")

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

    def _era_matches(self, release_year: Any, era_pref: Any) -> bool:
        if not era_pref:
            return False

        try:
            year = int(release_year)
        except Exception:
            return False

        era_text = self._norm(era_pref)
        m = re.fullmatch(r"(19|20)\d0s", era_text)
        if not m:
            return False

        decade = int(era_text[:4])
        return decade <= year <= decade + 9

    def _contains_any(self, text: str, patterns: set[str] | tuple[str, ...]) -> bool:
        if not text:
            return False
        return any(p in text for p in patterns)

    def _norm(self, value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).strip().lower().split())