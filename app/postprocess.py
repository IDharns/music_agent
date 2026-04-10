from typing import List, Dict, Any

from app.query_parser import ParsedQuery


def _safe_norm(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().lower()


def _year_matches_era(year: Any, era: str | None) -> bool:
    if not era or year is None:
        return False

    year_str = str(year).strip()
    if not year_str:
        return False

    if era == "recent":
        if year_str[:4].isdigit():
            return int(year_str[:4]) >= 2015
        return False

    if era.endswith("s") and len(era) >= 4:
        prefix = era[:4]
        return year_str.startswith(prefix)

    return False


def _text_blob(item: Dict[str, Any]) -> str:
    parts = [
        item.get("title"),
        item.get("artist"),
        item.get("album"),
        item.get("tags"),
        item.get("genre"),
        item.get("genres"),
        item.get("mood"),
        item.get("moods"),
    ]
    return " ".join(str(x) for x in parts if x is not None).lower()


def apply_constraints(
    candidates: List[Dict[str, Any]],
    parsed: ParsedQuery,
) -> List[Dict[str, Any]]:
    rescored: List[Dict[str, Any]] = []

    for item in candidates:
        score = float(item.get("score", 0.0))
        title = _safe_norm(item.get("title"))
        blob = _text_blob(item)
        popularity = item.get("popularity", 0) or 0

        # exclude rules
        if "live" in parsed.exclude and "live" in title:
            continue

        if "remix" in parsed.exclude and "remix" in title:
            continue

        if "instrumental" in parsed.exclude and "instrumental" in blob:
            continue

        if "explicit" in parsed.exclude and "explicit" in blob:
            continue

        if "loud" in parsed.exclude:
            score -= 0.15

        # popularity preference
        if parsed.popularity_preference == "less_popular":
            score -= float(popularity) * 0.001
        elif parsed.popularity_preference == "more_popular":
            score += float(popularity) * 0.001

        # era
        if parsed.era:
            if _year_matches_era(item.get("release_year"), parsed.era):
                score += 0.20
            else:
                score -= 0.08

        # vocal
        if parsed.vocal == "female":
            if "female" in blob:
                score += 0.10
            else:
                score -= 0.03
        elif parsed.vocal == "male":
            if "male" in blob:
                score += 0.10
        elif parsed.vocal == "instrumental":
            if "instrumental" in blob:
                score += 0.10

        # genres
        if parsed.genres:
            matched_genre = any(g.lower() in blob for g in parsed.genres)
            if matched_genre:
                score += 0.20
            else:
                score -= 0.05

        # moods
        if parsed.moods:
            matched_mood = any(m.lower() in blob for m in parsed.moods)
            if matched_mood:
                score += 0.15

        # include terms
        if parsed.include:
            include_hits = sum(1 for term in parsed.include if term.lower() in blob)
            score += 0.05 * include_hits

        updated = dict(item)
        updated["score"] = score
        rescored.append(updated)

    rescored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return rescored


def dedup_and_diversify(
    candidates: List[Dict[str, Any]],
    final_k: int = 10,
    max_per_artist: int = 3,
) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen_song_keys = set()
    artist_counts: dict[str, int] = {}

    for item in candidates:
        norm_title = _safe_norm(item.get("normalized_title")) or _safe_norm(item.get("title"))
        norm_artist = _safe_norm(item.get("normalized_artist")) or _safe_norm(item.get("artist"))

        song_key = (norm_title, norm_artist)
        if song_key in seen_song_keys:
            continue

        current_artist_count = artist_counts.get(norm_artist, 0)
        if current_artist_count >= max_per_artist:
            continue

        seen_song_keys.add(song_key)
        artist_counts[norm_artist] = current_artist_count + 1
        deduped.append(item)

        if len(deduped) >= final_k:
            break

    return deduped


def apply_constraints_and_rerank(
    candidates: List[Dict[str, Any]],
    parsed: ParsedQuery,
    final_k: int = 10,
    max_per_artist: int = 3,
) -> List[Dict[str, Any]]:
    rescored = apply_constraints(candidates, parsed)
    return dedup_and_diversify(
        rescored,
        final_k=final_k,
        max_per_artist=max_per_artist,
    )