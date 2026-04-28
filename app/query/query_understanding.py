from __future__ import annotations

import re
from typing import Any


class QueryUnderstandingModule:
    SEED_STYLE_HINTS: dict[str, list[str]] = {
        "taylor swift": [
            "female vocal",
            "pop",
            "country pop",
            "singer songwriter",
            "acoustic",
        ],
        "adele": [
            "female vocal",
            "pop",
            "soul",
            "ballad",
            "singer songwriter",
        ],
        "jay chou": [
            "male vocal",
            "mandopop",
            "pop",
            "rnb",
        ],
        "周杰伦": [
            "male vocal",
            "mandopop",
            "pop",
            "rnb",
        ],
        "radiohead": [
            "alternative rock",
            "indie rock",
            "electronic",
            "art rock",
            "dark",
        ],
    }

    ARTIST_LIKE_TRIGGERS = (
        "类似", "像", "接近", "参考", "跟", "像是",
        "like", "similar to", "sounds like",
    )

    NON_ARTIST_HINTS = (
        "pop", "rock", "indie", "electronic", "jazz", "folk", "hip hop", "hiphop",
        "rnb", "r&b", "synthpop", "dream pop", "shoegaze", "classical",
        "sad", "dreamy", "soft", "mellow", "energetic", "chill", "dark", "ethereal",
        "female", "male", "instrumental",
        "热门", "冷门", "小众", "popular", "underrated",
        "live", "remix", "acoustic",
        "80s", "90s", "2000s", "2010s", "年代",
    )

    def __init__(self, text_processor: Any | None = None) -> None:
        self.text_processor = text_processor

    def understand(self, raw_query: str) -> dict[str, Any]:
        raw_query = (raw_query or "").strip()
        normalized_query = self._normalize_query(raw_query)

        artist_seeds = self._extract_artist_seeds(raw_query)
        genres = self._extract_genres(normalized_query)
        moods = self._extract_moods(normalized_query)
        vocal = self._extract_vocal(normalized_query)
        energy = self._extract_energy(normalized_query)
        era = self._extract_era(normalized_query)
        popularity_preference = self._extract_popularity_preference(normalized_query)
        result_limit = self._extract_result_limit(normalized_query)
        include = self._extract_include(normalized_query)
        exclude = self._extract_exclude(normalized_query)
        if exclude:
            excluded = set(exclude)
            genres = [genre for genre in genres if genre not in excluded]
            include = [value for value in include if value not in excluded]

        # 单独一个艺人名时，直接当 artist query
        if not artist_seeds and self._looks_like_artist_query(raw_query, normalized_query):
            artist_seeds = [raw_query.strip()]

        query_type = self._decide_query_type(
            artist_seeds=artist_seeds,
            genres=genres,
            moods=moods,
            vocal=vocal,
            energy=energy,
            era=era,
            popularity_preference=popularity_preference,
            include=include,
            exclude=exclude,
        )

        return {
            "query_type": query_type,
            "artist_seeds": artist_seeds,
            "genres": genres,
            "moods": moods,
            "vocal": vocal,
            "energy": energy,
            "era": era,
            "popularity_preference": popularity_preference,
            "result_limit": result_limit,
            "include": include,
            "exclude": exclude,
            "normalized_query": normalized_query,
            "raw_query": raw_query,
        }

    def build_semantic_query(self, parsed_query: dict[str, Any]) -> str:
        query_type = parsed_query.get("query_type", "semantic")

        if query_type == "artist":
            return parsed_query.get("normalized_query", "") or parsed_query.get("raw_query", "")

        parts: list[str] = []

        parts.extend(parsed_query.get("genres", []))
        parts.extend(parsed_query.get("moods", []))
        parts.extend(parsed_query.get("include", []))

        vocal = parsed_query.get("vocal")
        if vocal and vocal != "unknown":
            parts.append(vocal)

        energy = parsed_query.get("energy")
        if energy and energy != "unknown":
            parts.append(energy)

        era = parsed_query.get("era")
        if era:
            parts.append(str(era))
            # 给 embedding 一个更常见的 decade 表达
            if era == "1980s":
                parts.append("80s")
            elif era == "1990s":
                parts.append("90s")
            elif era == "2000s":
                parts.append("2000s")
            elif era == "2010s":
                parts.append("2010s")

        pop_pref = parsed_query.get("popularity_preference")
        if pop_pref == "less_popular":
            parts.extend(["less popular", "underrated"])
        elif pop_pref == "more_popular":
            parts.append("popular")

        artist_seeds = parsed_query.get("artist_seeds", [])
        if query_type == "mixed" and artist_seeds:
            parts.extend(self._expand_seed_style_terms(artist_seeds))

        deduped: list[str] = []
        seen: set[str] = set()
        for part in parts:
            p = " ".join(str(part).strip().lower().split())
            if not p:
                continue
            if p not in seen:
                seen.add(p)
                deduped.append(p)

        if deduped:
            return " ".join(deduped)

        return parsed_query.get("normalized_query", "") or parsed_query.get("raw_query", "")

    def _decide_query_type(
            self,
            artist_seeds: list[str],
            genres: list[str],
            moods: list[str],
            vocal: str,
            energy: str,
            era: str | None,
            popularity_preference: str | None,
            include: list[str],
            exclude: list[str],
    ) -> str:
        has_seed = bool(artist_seeds)
        has_other_constraints = bool(
            genres
            or moods
            or include
            or exclude
            or era
            or popularity_preference
            or (vocal and vocal != "unknown")
            or (energy and energy != "unknown")
        )

        if has_seed and not has_other_constraints:
            return "artist"
        if has_seed and has_other_constraints:
            return "mixed"
        return "semantic"

    def _extract_artist_seeds(self, raw_query: str) -> list[str]:
        candidates: list[str] = []

        patterns = [
            r"(?:类似|像|偏向|接近|参考|跟)\s*([A-Za-z][A-Za-z0-9&' .\-]{1,80})",
            r"(?:like|similar to|sounds like)\s+([A-Za-z][A-Za-z0-9&' .\-]{1,80})",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, raw_query, flags=re.IGNORECASE):
                seed = self._cleanup_artist_seed(match.group(1))
                if seed:
                    candidates.append(seed)

        trigger = any(t.lower() in raw_query.lower() for t in self.ARTIST_LIKE_TRIGGERS)
        if trigger and not candidates:
            for match in re.finditer(r"[A-Za-z][A-Za-z0-9&' .\-]{2,80}", raw_query):
                seed = self._cleanup_artist_seed(match.group(0))
                if seed and len(seed.split()) <= 5:
                    candidates.append(seed)

        return self._dedupe_preserve_order(candidates)

    def _cleanup_artist_seed(self, value: str) -> str:
        value = value.strip()
        value = re.split(
            r"(?:，|,|。|！|!|？|\?|但不要|但是|不过|更|而且|然后|并且|不要|别太|一点|一些|but|more|less)",
            value,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        value = value.strip(" -_.,，。!！？?()[]{}\"'")
        value = " ".join(value.split())
        return value

    def _expand_seed_style_terms(self, artist_seeds: list[str]) -> list[str]:
        out: list[str] = []

        for seed in artist_seeds:
            key = seed.lower()
            hints = self.SEED_STYLE_HINTS.get(key)
            if hints:
                out.extend(hints)

        if not out and artist_seeds:
            out.extend(["similar style"])

        return out

    def _looks_like_artist_query(self, raw_query: str, normalized_query: str) -> bool:
        q = normalized_query.strip()
        if not q:
            return False

        # 明显带条件词/风格词，就不是纯 artist query
        if any(self._keyword_in_text(q, token) for token in self.NON_ARTIST_HINTS):
            return False

        # 太长通常不是单纯艺人名
        if len(q.split()) > 4:
            return False

        # 纯中文名：2~8 个汉字
        if re.fullmatch(r"[\u4e00-\u9fff]{2,8}", q):
            return True

        # 英文艺人名：1~4 个词，每个词首字母大写或全大写
        words = raw_query.strip().split()
        if 1 <= len(words) <= 4:
            ok = True
            for w in words:
                clean = w.strip(".-'&")
                if not clean:
                    ok = False
                    break
                if not (clean[0].isupper() or clean.isupper()):
                    ok = False
                    break
            if ok:
                return True

        # 单个词且像专有名词，例如 Adele / Drake
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9'&.\-]{1,30}", q):
            return True

        return False

    def _extract_genres(self, text: str) -> list[str]:
        mapping = {
            "synthpop": ["synthpop", "synth-pop"],
            "dream pop": ["dream pop"],
            "shoegaze": ["shoegaze"],
            "indie pop": ["indie pop"],
            "country pop": ["country pop"],
            "mandopop": ["mandopop", "华语流行"],
            "hip hop": ["hip hop", "hiphop", "说唱", "嘻哈"],
            "rnb": ["rnb", "r&b"],
            "electronic": ["electronic", "电子", "electronica"],
            "indie": ["indie", "独立"],
            "rock": ["rock", "摇滚"],
            "pop": ["pop", "流行"],
            "folk": ["folk", "民谣"],
            "jazz": ["jazz", "爵士"],
            "classical": ["classical", "古典"],
            "alternative": ["alternative"],
            "new wave": ["new wave"],
            "soul": ["soul"],
        }
        return self._extract_multi_label(text, mapping)

    def _extract_moods(self, text: str) -> list[str]:
        mapping = {
            "dreamy": ["dreamy", "梦幻"],
            "sad": ["sad", "伤感", "难过", "emo"],
            "soft": ["soft", "柔和", "轻柔"],
            "mellow": ["mellow", "舒缓"],
            "energetic": ["energetic", "有活力", "燃", "upbeat"],
            "chill": ["chill", "放松", "轻松"],
            "dark": ["dark", "黑暗", "阴郁", "冷一点", "冷感", "冷"],
            "ethereal": ["ethereal", "空灵"],
        }
        return self._extract_multi_label(text, mapping)

    def _extract_vocal(self, text: str) -> str:
        if any(k in text for k in [
            "female vocal", "female vocals", "female singer", "female singers",
            "girl", "girl music", "girly", "women", "woman", "女声", "女生", "female pop",
        ]):
            return "female vocal"
        if any(k in text for k in [
            "male vocal", "male vocals", "male singer", "male singers",
            "boy", "boy music", "men", "man", "男声", "男生", "male pop",
        ]):
            return "male vocal"
        if "female" in text and "instrumental" not in text:
            return "female vocal"
        if "male" in text and "instrumental" not in text:
            return "male vocal"
        if "instrumental" in text or "纯音乐" in text:
            return "instrumental"
        return "unknown"

    def _extract_energy(self, text: str) -> str:
        if any(k in text for k in ["high energy", "energetic", "高能量", "激烈", "upbeat"]):
            return "high"
        if any(k in text for k in ["low energy", "低能量", "平静", "calm"]):
            return "low"
        return "unknown"

    def _extract_era(self, text: str) -> str | None:
        if re.search(r"\b80s\b", text):
            return "1980s"
        if re.search(r"\b90s\b", text):
            return "1990s"
        if re.search(r"\b2000s\b|\b00s\b", text):
            return "2000s"
        if re.search(r"\b2010s\b|\b10s\b", text):
            return "2010s"

        match = re.search(r"\b(19\d0|20\d0)s\b", text)
        if match:
            return match.group(0)

        cn_map = {
            "80年代": "1980s",
            "90年代": "1990s",
            "00年代": "2000s",
            "10年代": "2010s",
        }
        for key, value in cn_map.items():
            if key in text:
                return value

        return None

    def _extract_popularity_preference(self, text: str) -> str | None:
        less_popular_markers = [
            "不要太热门", "不太热门", "别太热门", "冷门", "小众",
            "underrated", "less popular", "not too popular",
        ]
        more_popular_markers = [
            "热门", "流行一点", "popular", "mainstream",
        ]

        if any(k in text for k in less_popular_markers):
            return "less_popular"
        if any(k in text for k in more_popular_markers):
            return "more_popular"
        return None

    def _extract_result_limit(self, text: str) -> int | None:
        patterns = [
            r"\btop\s+(\d{1,2})\b",
            r"\b(\d{1,2})\s+(?:songs?|tracks?|results?|recs?|recommendations?)\b",
            r"\b(?:give me|show me|need|want)\s+(\d{1,2})\b",
            r"(\d{1,2})\s*(?:首|个|條|条)",
        ]
        word_based_limits = [
            (r"\ba couple of\b", 2),
            (r"\ba couple\b", 2),
            (r"\ba few\b", 3),
            (r"\bsome\b", 5),
            (r"几首", 3),
            (r"几个", 3),
            (r"一些", 5),
            (r"来几首", 3),
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            try:
                value = int(match.group(1))
            except Exception:
                continue
            if 1 <= value <= 50:
                return value

        for pattern, value in word_based_limits:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return value
        return None

    def _extract_include(self, text: str) -> list[str]:
        mapping = {
            "acoustic": ["acoustic", "木吉他", "不插电"],
            "singer songwriter": ["singer songwriter", "singer-songwriter", "创作歌手", "songwriter"],
            "ambient": ["ambient", "氛围"],
        }
        return self._extract_multi_label(text, mapping)

    def _extract_exclude(self, text: str) -> list[str]:
        mapping = {
            "live": ["不要live", "不要现场", "not live", "live版不要", "but not live", "no live"],
            "remix": ["不要remix", "不要混音", "not remix", "no remix"],
            "instrumental": ["不要纯音乐", "not instrumental", "no instrumental"],
            "acoustic": ["不要acoustic", "not acoustic", "no acoustic"],
            "classical": ["不要classical", "不要古典", "not classical", "no classical"],
        }
        return self._extract_multi_label(text, mapping)

    def _extract_multi_label(self, text: str, mapping: dict[str, list[str]]) -> list[str]:
        out: list[str] = []
        for label, keywords in mapping.items():
            if any(self._keyword_in_text(text, keyword) for keyword in keywords):
                out.append(label)
        return out

    def _keyword_in_text(self, text: str, keyword: str) -> bool:
        keyword = keyword.strip().lower()
        if not keyword:
            return False

        text = text.lower()
        if re.fullmatch(r"[a-z0-9][a-z0-9&' .+\-]*", keyword):
            return bool(re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text))
        return keyword in text

    def _normalize_query(self, text: str) -> str:
        text = (text or "").strip().lower()
        text = text.replace("，", ",")
        text = text.replace("。", ".")
        text = re.sub(r"\s+", " ", text)
        return text

    def _dedupe_preserve_order(self, items: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()

        for item in items:
            key = item.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item.strip())

        return out
