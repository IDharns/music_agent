from collections import Counter
from pathlib import Path
from typing import Any, Dict, List
import re

from fastapi import FastAPI, Query

from app.recommender import MusicRetriever, normalize_text
from app.postprocess import dedup_and_diversify
from app.query_router import classify_query


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "music.db"
INDEX_PATH = BASE_DIR / "data" / "faiss.index"
IDS_PATH = BASE_DIR / "data" / "ids.npy"

app = FastAPI(title="Music Agent API", version="0.3.0")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = MusicRetriever(
    db_path=str(DB_PATH),
    index_path=str(INDEX_PATH),
    ids_path=str(IDS_PATH),
)


def build_reason(item: Dict[str, Any], query_type: str, fallback_used: bool) -> str:
    match_type = item.get("match_type", "unknown")

    if match_type == "artist_exact":
        return "Matched artist name exactly."
    if match_type == "semantic" and fallback_used and query_type == "artist":
        return "Artist lookup missed, so semantic retrieval was used as fallback."
    if query_type == "mixed":
        return "Retrieved by semantic similarity, then adjusted by mixed-query rules."
    return "Retrieved by semantic similarity against the query."


def build_response(
        query: str,
        route: Dict[str, Any],
        fallback_used: bool,
        results: List[Dict[str, Any]],
        semantic_query_used: str | None = None,
) -> Dict[str, Any]:
    query_type = route.get("query_type", "semantic")

    output_results = []
    for item in results:
        output_results.append(
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "artist": item.get("artist"),
                "album": item.get("album"),
                "release_year": item.get("release_year"),
                "popularity": item.get("popularity"),
                "popularity_bucket": item.get("popularity_bucket"),
                "language": item.get("language"),
                "vocal_type": item.get("vocal_type"),
                "genre_text": item.get("genre_text"),
                "style_text": item.get("style_text"),
                "mood_text": item.get("mood_text"),
                "tags": item.get("tags", []),
                "artist_tags": item.get("artist_tags", []),
                "score": item.get("score"),
                "match_type": item.get("match_type"),
                "reason": build_reason(item, query_type, fallback_used),
            }
        )

    return {
        "query": query,
        "parsed_query": route,
        "query_type": query_type,
        "fallback_used": fallback_used,
        "semantic_query_used": semantic_query_used,
        "result_count": len(output_results),
        "results": output_results,
    }


def _norm_list(values: Any) -> List[str]:
    if not values:
        return []
    if isinstance(values, list):
        return [str(x).strip() for x in values if str(x).strip()]
    return [str(values).strip()]


def _split_multi_value(text: Any) -> List[str]:
    if not text:
        return []
    raw = str(text).strip()
    if not raw:
        return []
    parts = re.split(r"[,/|]+", raw)
    return [p.strip() for p in parts if p.strip()]


def _era_to_year_range(era: Any) -> tuple[int, int] | None:
    if era is None:
        return None

    text = str(era).strip().lower()

    decade_map = {
        "70s": (1970, 1979),
        "80s": (1980, 1989),
        "90s": (1990, 1999),
        "00s": (2000, 2009),
        "10s": (2010, 2019),
    }
    if text in decade_map:
        return decade_map[text]

    m = re.fullmatch(r"(19\d0|20\d0)s", text)
    if m:
        start = int(m.group(1))
        return start, start + 9

    m = re.fullmatch(r"(19\d{2}|20\d{2})", text)
    if m:
        year = int(m.group(1))
        return year, year

    return None


def _build_blob(item: Dict[str, Any]) -> str:
    parts = [
        item.get("title") or "",
        item.get("artist") or "",
        item.get("album") or "",
        item.get("vocal_type") or "",
        item.get("genre_text") or "",
        item.get("style_text") or "",
        item.get("mood_text") or "",
        item.get("language") or "",
        " ".join(item.get("tags", []) or []),
        " ".join(item.get("artist_tags", []) or []),
        ]
    return normalize_text(" | ".join(parts))


def _has_any(blob: str, terms: List[str]) -> bool:
    for t in terms:
        nt = normalize_text(t)
        if nt and nt in blob:
            return True
    return False


def _match_count(blob: str, terms: List[str]) -> int:
    count = 0
    for t in terms:
        nt = normalize_text(t)
        if nt and nt in blob:
            count += 1
    return count


def _extract_seed_style_terms(seed_name: str, limit: int = 5) -> List[str]:
    """
    不把 seed artist 原名放进 query。
    改为从该 artist 的已有 metadata 中抽弱风格锚点。
    """
    tracks = retriever.search_by_artist(seed_name, top_k=20)
    if not tracks:
        return []

    counter: Counter[str] = Counter()
    seed_norm = normalize_text(seed_name)
    seed_words = set(seed_norm.split())

    bad_terms = {
        "", "all", "spotify", "american", "british", "swedish", "german",
        "japanese", "chinese", "mandarin", "cantonese", "taiwan", "taiwanese",
        "malaysia", "malaysian", "portugal", "portuguese", "norway", "norwegian",
        "usa", "uk", "england", "female", "male",
    }

    def add_term(term: str, weight: int) -> None:
        norm = normalize_text(term)
        if not norm:
            return
        if norm in bad_terms:
            return
        if norm in seed_words:
            return
        if seed_norm and norm in seed_norm:
            return
        if any(w in seed_words for w in norm.split()):
            return
        counter[norm] += weight

    for track in tracks:
        vocal_type = str(track.get("vocal_type") or "").strip().lower()
        if "female" in vocal_type:
            add_term("female vocal", 4)
        elif "male" in vocal_type:
            add_term("male vocal", 4)

        for x in _split_multi_value(track.get("genre_text")):
            add_term(x, 3)

        for x in _split_multi_value(track.get("style_text")):
            add_term(x, 3)

        for x in track.get("artist_tags", []) or []:
            add_term(str(x), 1)

        for x in track.get("tags", []) or []:
            add_term(str(x), 1)

    prioritized = []
    for term, _ in counter.most_common():
        if term in {
            "female vocalists", "female vocalist", "female vocals"
        }:
            term = "female vocal"
        if term in {
            "male vocalists", "male vocalist", "male vocals"
        }:
            term = "male vocal"
        prioritized.append(term)

    # 尽量保留对风格更有用的词
    good_order = []
    preferred_patterns = [
        "female vocal",
        "male vocal",
        "pop",
        "indie pop",
        "dream pop",
        "country pop",
        "singer songwriter",
        "singer-songwriter",
        "acoustic",
        "soft rock",
        "folk",
        "electronic",
        "shoegaze",
        "ethereal",
    ]

    for p in preferred_patterns:
        for t in prioritized:
            if t == p and t not in good_order:
                good_order.append(t)

    for t in prioritized:
        if t not in good_order:
            good_order.append(t)

    return good_order[:limit]


def build_mixed_semantic_query(route: Dict[str, Any]) -> str:
    """
    不直接放 artist 名字，但保留从 seed artist metadata 推出来的弱风格锚点。
    """
    parts: List[str] = []

    moods = _norm_list(route.get("moods"))
    genres = _norm_list(route.get("genres"))
    artist_seeds = _norm_list(route.get("artist_seeds"))

    for m in moods:
        parts.append(m)

    for g in genres:
        parts.append(g)

    vocal = str(route.get("vocal", "")).strip().lower()
    if vocal and vocal != "unknown":
        if vocal == "female":
            parts.append("female vocal")
        elif vocal == "male":
            parts.append("male vocal")
        else:
            parts.append(vocal)

    energy = str(route.get("energy", "")).strip().lower()
    if energy and energy != "unknown":
        parts.append(energy)

    era = route.get("era")
    if era:
        parts.append(str(era))

    pop_pref = str(route.get("popularity_preference", "")).strip().lower()
    if pop_pref == "less_popular":
        parts.extend(["less popular", "underrated"])
    elif pop_pref == "more_popular":
        parts.append("popular")

    for x in _norm_list(route.get("include")):
        parts.append(x)

    for x in _norm_list(route.get("exclude")):
        parts.append(f"not {x}")

    # 从 seed artist 提取弱风格锚点
    seed_style_terms: List[str] = []
    for seed in artist_seeds[:2]:
        seed_style_terms.extend(_extract_seed_style_terms(seed, limit=5))

    # 去重保序
    deduped: List[str] = []
    seen = set()
    for t in parts + seed_style_terms:
        nt = normalize_text(t)
        if not nt or nt in seen:
            continue
        seen.add(nt)
        deduped.append(t)

    final_query = " ".join(deduped).strip()
    if not final_query:
        final_query = str(route.get("raw_query", "")).strip()

    return final_query


def should_drop_for_mixed(item: Dict[str, Any]) -> bool:
    title = normalize_text(item.get("title"))
    artist = normalize_text(item.get("artist"))
    album = normalize_text(item.get("album"))

    bad_patterns = [
        "made famous by",
        "karaoke",
        "tribute",
        "interview",
        "dumped",
        "parody",
        "cover band",
    ]

    if any(p in title for p in bad_patterns):
        return True
    if any(p in artist for p in bad_patterns):
        return True
    if any(p in album for p in bad_patterns):
        return True

    return False


def _apply_constraint_rerank(
        raw_candidates: List[Dict[str, Any]],
        route: Dict[str, Any],
        *,
        is_mixed: bool,
) -> List[Dict[str, Any]]:
    artist_seeds = {normalize_text(x) for x in _norm_list(route.get("artist_seeds"))}
    moods = {normalize_text(x) for x in _norm_list(route.get("moods"))}
    genres = {normalize_text(x) for x in _norm_list(route.get("genres"))}
    include_terms = {normalize_text(x) for x in _norm_list(route.get("include"))}
    exclude_terms = {normalize_text(x) for x in _norm_list(route.get("exclude"))}

    pop_pref = str(route.get("popularity_preference", "")).strip().lower()
    vocal_pref = str(route.get("vocal", "")).strip().lower()
    era_range = _era_to_year_range(route.get("era"))

    seed_style_terms: List[str] = []
    if is_mixed:
        for seed in _norm_list(route.get("artist_seeds"))[:2]:
            seed_style_terms.extend(_extract_seed_style_terms(seed, limit=5))

    dreamy_terms = ["dreamy", "dream pop", "ethereal", "shoegaze", "soft", "lush", "ambient"]
    sad_terms = ["sad", "melancholy", "melancholic", "heartbreak", "slow", "emo"]

    for item in raw_candidates:
        score = float(item.get("score", 0.0))

        blob = _build_blob(item)
        title = normalize_text(item.get("title"))
        artist = normalize_text(item.get("artist"))
        album = normalize_text(item.get("album"))
        popularity = item.get("popularity")
        release_year = item.get("release_year")
        vocal_type = normalize_text(item.get("vocal_type"))
        genre_text = normalize_text(item.get("genre_text"))
        style_text = normalize_text(item.get("style_text"))
        artist_tags_blob = normalize_text(" ".join(item.get("artist_tags", []) or []))
        tags_blob = normalize_text(" ".join(item.get("tags", []) or []))

        if is_mixed:
            if artist in artist_seeds:
                score -= 0.35

            for seed in artist_seeds:
                if seed and seed in title and artist != seed:
                    score -= 0.20
                if seed and seed in album and artist != seed:
                    score -= 0.10

            style_hit_count = _match_count(blob, seed_style_terms)
            score += min(style_hit_count, 3) * 0.06

        vocal_required = vocal_pref in {"female", "male"}
        genre_required = len(genres) > 0
        mood_required = len(moods) > 0
        era_required = era_range is not None

        vocal_ok = False
        genre_ok = False
        mood_ok = False
        mood_title_only = False
        year_ok = False

        # ---------- vocal ----------
        if vocal_pref == "female":
            vocal_ok = (
                    "female" in vocal_type
                    or "female vocalists" in blob
                    or "female vocalist" in blob
            )
            if vocal_ok:
                score += 0.18
            elif "instrumental" in vocal_type:
                score -= 0.22
            else:
                score -= 0.12

        elif vocal_pref == "male":
            vocal_ok = (
                    "male" in vocal_type
                    or "male vocalists" in blob
                    or "male vocalist" in blob
            )
            if vocal_ok:
                score += 0.18
            elif "instrumental" in vocal_type:
                score -= 0.22
            else:
                score -= 0.12

        # ---------- genre ----------
        genre_hits = _match_count(blob, list(genres))
        if genre_hits > 0:
            genre_ok = True
            score += min(genre_hits, 2) * 0.10

        # ---------- mood ----------
        for m in moods:
            if m == "dreamy":
                dream_in_meta = (
                        _has_any(genre_text, ["dream pop", "shoegaze"]) or
                        _has_any(style_text, ["dream pop", "shoegaze"]) or
                        _has_any(artist_tags_blob, ["dream pop", "shoegaze", "ethereal", "ambient"]) or
                        _has_any(tags_blob, ["dream pop", "shoegaze", "ethereal", "ambient"])
                )
                dream_in_title = any(tok in title for tok in ["dream", "dreamy", "dreaming", "dreams"])

                if dream_in_meta:
                    mood_ok = True
                    score += 0.20
                elif dream_in_title:
                    mood_title_only = True
                    score -= 0.06

            elif m == "sad":
                sad_in_meta = _has_any(blob, sad_terms)
                sad_in_title = "sad" in title

                if sad_in_meta:
                    mood_ok = True
                    score += 0.20
                elif sad_in_title:
                    mood_title_only = True
                    score -= 0.06

            elif m and m in blob:
                mood_ok = True
                score += 0.10

        # ---------- include / exclude ----------
        include_hits = _match_count(blob, list(include_terms))
        exclude_hits = _match_count(blob, list(exclude_terms))
        score += include_hits * 0.05
        score -= exclude_hits * 0.10

        # ---------- era ----------
        if era_range:
            if release_year is None:
                score -= 0.12
            else:
                try:
                    y = int(release_year)
                    if era_range[0] <= y <= era_range[1]:
                        year_ok = True
                        score += 0.24
                    else:
                        score -= 0.14
                except Exception:
                    score -= 0.10

        # ---------- popularity ----------
        if pop_pref == "less_popular":
            if popularity is None:
                score -= 0.01
            elif popularity >= 75:
                score -= 0.12
            elif popularity <= 40:
                score += 0.06
        elif pop_pref == "more_popular":
            if popularity is not None and popularity >= 70:
                score += 0.05

        # ---------- 标题词面污染 ----------
        query_words = [
            normalize_text(x) for x in (
                    list(moods) + list(genres) + list(include_terms) +
                    ([vocal_pref] if vocal_pref != "unknown" else [])
            )
        ]
        title_hits = sum(1 for w in query_words if w and w in title)

        meta_hit = 0
        if genre_ok:
            meta_hit += 1
        if mood_ok:
            meta_hit += 1
        if vocal_required and vocal_ok:
            meta_hit += 1
        if year_ok:
            meta_hit += 1

        if title_hits >= 1 and meta_hit == 0:
            score -= 0.10

        # ---------- 核心 facet 满足度 ----------
        required_count = 0
        matched_count = 0

        if vocal_required:
            required_count += 1
            if vocal_ok:
                matched_count += 1

        if genre_required:
            required_count += 1
            if genre_ok:
                matched_count += 1

        if mood_required:
            required_count += 1
            if mood_ok:
                matched_count += 1

        if era_required:
            required_count += 1
            if year_ok:
                matched_count += 1

        score += matched_count * 0.04

        if required_count >= 3 and matched_count <= 1:
            score -= 0.22
        elif required_count == 2 and matched_count == 0:
            score -= 0.18

        if vocal_required and not vocal_ok:
            score -= 0.10

        # ---------- 新增：mood 缺失单独处罚 ----------
        # 用户明确写了 dreamy / sad，这不该只是可有可无的 hint
        if mood_required and not mood_ok:
            score -= 0.18

            # 只有标题带 dream / sad，但 metadata 没支撑，再多罚一点
            if mood_title_only:
                score -= 0.06

        # ---------- 新增：query 有 4 个 facet 时，缺 mood 的歌不能霸榜 ----------
        if required_count >= 4 and not mood_ok:
            score -= 0.08

        # ---------- 新增：四个 facet 全中时，给组合奖励 ----------
        if required_count >= 4 and matched_count >= 4:
            score += 0.08
        elif required_count >= 3 and matched_count >= 3:
            score += 0.04

        item["score"] = score

    raw_candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return raw_candidates

def _candidate_facets(item: Dict[str, Any], route: Dict[str, Any]) -> Dict[str, bool]:
    moods = {normalize_text(x) for x in _norm_list(route.get("moods"))}
    genres = {normalize_text(x) for x in _norm_list(route.get("genres"))}
    vocal_pref = str(route.get("vocal", "")).strip().lower()
    era_range = _era_to_year_range(route.get("era"))

    title = normalize_text(item.get("title"))
    vocal_type = normalize_text(item.get("vocal_type"))
    genre_text = normalize_text(item.get("genre_text"))
    style_text = normalize_text(item.get("style_text"))
    artist_tags_blob = normalize_text(" ".join(item.get("artist_tags", []) or []))
    tags_blob = normalize_text(" ".join(item.get("tags", []) or []))
    blob = _build_blob(item)

    dreamy_meta = (
            _has_any(genre_text, ["dream pop", "shoegaze"]) or
            _has_any(style_text, ["dream pop", "shoegaze"]) or
            _has_any(artist_tags_blob, ["dream pop", "shoegaze", "ethereal", "ambient"]) or
            _has_any(tags_blob, ["dream pop", "shoegaze", "ethereal", "ambient"])
    )
    sad_meta = _has_any(blob, ["sad", "melancholy", "melancholic", "heartbreak", "slow", "emo"])

    vocal_ok = False
    if vocal_pref == "female":
        vocal_ok = (
                "female" in vocal_type
                or "female vocalists" in blob
                or "female vocalist" in blob
        )
    elif vocal_pref == "male":
        vocal_ok = (
                "male" in vocal_type
                or "male vocalists" in blob
                or "male vocalist" in blob
        )
    else:
        vocal_ok = True

    genre_ok = True if not genres else _match_count(blob, list(genres)) > 0

    mood_ok = True
    if moods:
        mood_ok = False
        for m in moods:
            if m == "dreamy" and dreamy_meta:
                mood_ok = True
            elif m == "sad" and sad_meta:
                mood_ok = True
            elif m and m in blob:
                mood_ok = True

    year_ok = True
    if era_range:
        year_ok = False
        release_year = item.get("release_year")
        if release_year is not None:
            try:
                y = int(release_year)
                year_ok = era_range[0] <= y <= era_range[1]
            except Exception:
                year_ok = False

    # 仅标题含 dream / sad，不算真正 mood 命中
    title_only_mood = False
    if moods and not mood_ok:
        if "dreamy" in moods and any(tok in title for tok in ["dream", "dreamy", "dreaming", "dreams"]):
            title_only_mood = True
        if "sad" in moods and "sad" in title:
            title_only_mood = True

    matched_count = sum([
        1 if vocal_ok else 0,
        1 if genre_ok else 0,
        1 if mood_ok else 0,
        1 if year_ok else 0,
    ])

    return {
        "vocal_ok": vocal_ok,
        "genre_ok": genre_ok,
        "mood_ok": mood_ok,
        "year_ok": year_ok,
        "title_only_mood": title_only_mood,
        "matched_count": matched_count,
    }


def _semantic_filter_priority(item: Dict[str, Any], route: Dict[str, Any]) -> tuple:
    f = _candidate_facets(item, route)

    # 排序优先级：
    # 1. 年代命中
    # 2. 声线命中
    # 3. mood 真命中
    # 4. genre 命中
    # 5. 总 facet 数
    # 6. rerank 后 score
    return (
        1 if f["year_ok"] else 0,
        1 if f["vocal_ok"] else 0,
        1 if f["mood_ok"] else 0,
        1 if f["genre_ok"] else 0,
        f["matched_count"],
        float(item.get("score", 0.0)),
    )


def postfilter_semantic_candidates(
        raw_candidates: List[Dict[str, Any]],
        route: Dict[str, Any],
        final_k: int,
) -> List[Dict[str, Any]]:
    """
    对 semantic query 做一层比 rerank 更硬的后筛。
    尤其针对:
    - era
    - vocal
    - mood
    - genre
    这些明确约束。
    """
    if not raw_candidates:
        return []

    moods = {normalize_text(x) for x in _norm_list(route.get("moods"))}
    genres = {normalize_text(x) for x in _norm_list(route.get("genres"))}
    vocal_pref = str(route.get("vocal", "")).strip().lower()
    era_range = _era_to_year_range(route.get("era"))

    has_constraints = bool(moods or genres or era_range or vocal_pref in {"female", "male"})
    if not has_constraints:
        return raw_candidates

    enriched = []
    for item in raw_candidates:
        f = _candidate_facets(item, route)
        enriched.append((item, f))

    # 先按 facet 优先级重新排一次
    enriched.sort(
        key=lambda x: _semantic_filter_priority(x[0], route),
        reverse=True,
    )

    # tier 1: 年代 + 声线 + genre 命中，且 mood 真命中
    tier1 = [
        item for item, f in enriched
        if f["year_ok"] and f["vocal_ok"] and f["genre_ok"] and f["mood_ok"]
    ]

    # tier 2: 年代 + 声线 + genre 命中
    tier2 = [
        item for item, f in enriched
        if f["year_ok"] and f["vocal_ok"] and f["genre_ok"]
    ]

    # tier 3: 年代 + 声线 命中
    tier3 = [
        item for item, f in enriched
        if f["year_ok"] and f["vocal_ok"]
    ]

    # tier 4: 至少 3 个 facet 命中，且不是标题骗分
    tier4 = [
        item for item, f in enriched
        if f["matched_count"] >= 3 and not f["title_only_mood"]
    ]

    # tier 5: 至少 2 个 facet 命中
    tier5 = [
        item for item, f in enriched
        if f["matched_count"] >= 2
    ]

    merged: List[Dict[str, Any]] = []
    seen = set()

    for tier in [tier1, tier2, tier3, tier4, tier5, [item for item, _ in enriched]]:
        for item in tier:
            sid = item.get("id")
            if sid in seen:
                continue
            seen.add(sid)
            merged.append(item)

    return merged

def rerank_mixed_candidates(
        raw_candidates: List[Dict[str, Any]],
        route: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return _apply_constraint_rerank(raw_candidates, route, is_mixed=True)


def rerank_semantic_candidates(
        raw_candidates: List[Dict[str, Any]],
        route: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return _apply_constraint_rerank(raw_candidates, route, is_mixed=False)


def run_query(query: str, final_k: int = 10, max_per_artist: int = 2) -> Dict[str, Any]:
    route = classify_query(query)
    query_type = route.get("query_type", "semantic")
    fallback_used = False
    semantic_query_used: str | None = None

    if query_type == "artist":
        raw_candidates = retriever.search_by_artist(
            route.get("normalized_query") or query,
            top_k=80,
            )
        if not raw_candidates:
            fallback_used = True
            semantic_query_used = query
            raw_candidates = retriever.search(query, top_k=80)

    elif query_type == "mixed":
        semantic_query_used = build_mixed_semantic_query(route)
        raw_candidates = retriever.search(semantic_query_used, top_k=150)
        raw_candidates = [x for x in raw_candidates if not should_drop_for_mixed(x)]
        raw_candidates = rerank_mixed_candidates(raw_candidates, route)

    else:
        semantic_query_used = query
        raw_candidates = retriever.search(query, top_k=300)
        raw_candidates = rerank_semantic_candidates(raw_candidates, route)
        raw_candidates = postfilter_semantic_candidates(
            raw_candidates,
            route,
            final_k=final_k,
        )

    final_results = dedup_and_diversify(
        raw_candidates,
        final_k=final_k,
        max_per_artist=max_per_artist,
    )

    return build_response(
        query=query,
        route=route,
        fallback_used=fallback_used,
        results=final_results,
        semantic_query_used=semantic_query_used,
    )


@app.get("/")
def root():
    return {
        "name": "Music Agent API",
        "version": "0.3.0",
        "endpoints": ["/health", "/search"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search")
def search(
        query: str = Query(..., min_length=1, description="User query"),
        final_k: int = Query(10, ge=1, le=50),
        max_per_artist: int = Query(2, ge=1, le=20),
):
    return run_query(
        query=query,
        final_k=final_k,
        max_per_artist=max_per_artist,
    )