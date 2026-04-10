import re
from typing import Any, Dict, List


CJK_PUNCT_TRANSLATION = str.maketrans({
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "：": ":",
    "；": ";",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
})


EN_MOOD_MAP = {
    "sad": "sad",
    "happy": "happy",
    "dreamy": "dreamy",
    "soft": "soft",
    "chill": "chill",
    "relaxing": "chill",
    "relax": "chill",
    "study": "study",
    "night": "night",
    "romantic": "romantic",
    "dark": "dark",
    "calm": "calm",
    "ethereal": "dreamy",
}

CN_MOOD_MAP = {
    "伤感": "sad",
    "悲伤": "sad",
    "难过": "sad",
    "开心": "happy",
    "快乐": "happy",
    "梦幻": "dreamy",
    "空灵": "dreamy",
    "柔和": "soft",
    "温柔": "soft",
    "轻盈": "soft",
    "放松": "chill",
    "学习": "study",
    "深夜": "night",
    "浪漫": "romantic",
    "黑暗": "dark",
    "平静": "calm",
}

EN_GENRE_MAP = {
    "electronic": "electronic",
    "pop": "pop",
    "rock": "rock",
    "jazz": "jazz",
    "ambient": "ambient",
    "acoustic": "acoustic",
    "dance": "dance",
    "indie": "indie",
    "folk": "folk",
    "rnb": "r&b",
    "rb": "r&b",
    "shoegaze": "shoegaze",
    "dream pop": "dream pop",
}

CN_GENRE_MAP = {
    "电子": "electronic",
    "流行": "pop",
    "摇滚": "rock",
    "爵士": "jazz",
    "氛围": "ambient",
    "民谣": "folk",
    "独立": "indie",
    "舞曲": "dance",
    "梦泡": "dream pop",
}

MIXED_MARKER_PATTERNS = [
    r"(?:类似|像|风格像|有点像|接近)\s*",
    r"(?:similar to|like|in the style of)\s+",
]

LESS_POPULAR_PATTERNS = [
    r"不要太热门",
    r"别太热门",
    r"不太热门",
    r"冷门一点",
    r"小众一点",
    r"less popular",
    r"not too popular",
    r"underrated",
]

MORE_POPULAR_PATTERNS = [
    r"更热门",
    r"热门一点",
    r"流行一点",
    r"more popular",
    r"mainstream",
]

EXCLUDE_PATTERNS = {
    "live": [r"不要live", r"别live", r"不要现场", r"别现场", r"\bno live\b"],
    "remix": [r"不要remix", r"别remix", r"不要混音版", r"别混音版", r"\bno remix\b"],
}

INCLUDE_PATTERNS = {
    "live": [r"live", r"现场"],
    "remix": [r"remix", r"混音"],
}

ARTIST_SEED_PATTERNS = [
    # 中文
    r"(?:类似|像|风格像|有点像|接近)\s*([A-Za-z0-9\u4e00-\u9fff&'._\- ]{1,40}?)(?=(?:但|但是|不过|而且|并且|不要|别|更|再|然后|,|\.|!|\?|;|:|$))",
    # 英文
    r"(?:similar to|like|in the style of)\s+([A-Za-z0-9\u4e00-\u9fff&'._\- ]{1,40}?)(?=(?:\s+but|\s+with|\s+less|\s+more|,|\.|!|\?|;|:|$))",
]


def normalize_query_text(text: str) -> str:
    text = text.translate(CJK_PUNCT_TRANSLATION)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_match_text(text: str) -> str:
    text = normalize_query_text(text).lower()
    return text


def unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        k = x.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x.strip())
    return out


def contains_any_pattern(text: str, patterns: List[str]) -> bool:
    for p in patterns:
        if re.search(p, text, flags=re.IGNORECASE):
            return True
    return False


def tokenize_query(text: str) -> List[str]:
    return re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+(?:[&'._\-][A-Za-z0-9]+)*", text)


def extract_artist_seeds(query: str) -> List[str]:
    seeds = []

    for pattern in ARTIST_SEED_PATTERNS:
        for m in re.finditer(pattern, query, flags=re.IGNORECASE):
            seed = m.group(1).strip(" -_.,:;!?\"'()[]")
            seed = re.sub(r"\s+", " ", seed).strip()
            if seed:
                seeds.append(seed)

    return unique_keep_order(seeds)


def extract_moods(query: str) -> List[str]:
    q = normalize_match_text(query)
    moods = []

    for k, v in EN_MOOD_MAP.items():
        if re.search(rf"\b{re.escape(k)}\b", q):
            moods.append(v)

    for k, v in CN_MOOD_MAP.items():
        if k in q:
            moods.append(v)

    return unique_keep_order(moods)


def extract_genres(query: str) -> List[str]:
    q = normalize_match_text(query)
    genres = []

    for k, v in EN_GENRE_MAP.items():
        if " " in k:
            if k in q:
                genres.append(v)
        else:
            if re.search(rf"\b{re.escape(k)}\b", q):
                genres.append(v)

    for k, v in CN_GENRE_MAP.items():
        if k in q:
            genres.append(v)

    return unique_keep_order(genres)


def extract_vocal(query: str) -> str:
    q = normalize_match_text(query)

    female_patterns = [r"\bfemale\b", r"女声", r"女生", r"女嗓", r"female vocal"]
    male_patterns = [r"\bmale\b", r"男声", r"男生", r"男嗓", r"male vocal"]

    if contains_any_pattern(q, female_patterns):
        return "female"
    if contains_any_pattern(q, male_patterns):
        return "male"
    return "unknown"


def extract_energy(query: str) -> str:
    q = normalize_match_text(query)

    energetic_patterns = [r"energetic", r"upbeat", r"high energy", r"有活力", r"亢奋", r"热烈"]
    calm_patterns = [r"calm", r"soft", r"gentle", r"轻柔", r"平静", r"舒缓"]

    if contains_any_pattern(q, energetic_patterns):
        return "energetic"
    if contains_any_pattern(q, calm_patterns):
        return "calm"
    return "unknown"


def extract_popularity_preference(query: str) -> str | None:
    q = normalize_match_text(query)

    if contains_any_pattern(q, LESS_POPULAR_PATTERNS):
        return "less_popular"
    if contains_any_pattern(q, MORE_POPULAR_PATTERNS):
        return "more_popular"
    return None


def extract_era(query: str) -> str | None:
    q = normalize_match_text(query)

    patterns = [
        (r"\b80s\b|80年代", "80s"),
        (r"\b90s\b|90年代", "90s"),
        (r"\b00s\b|00年代|2000s", "00s"),
        (r"\b10s\b|10年代|2010s", "10s"),
        (r"\b70s\b|70年代", "70s"),
    ]

    for pattern, era in patterns:
        if re.search(pattern, q, flags=re.IGNORECASE):
            return era

    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", q)
    if year_match:
        return year_match.group(1)

    return None


def extract_include_terms(query: str) -> List[str]:
    q = normalize_match_text(query)
    out = []

    for term, patterns in INCLUDE_PATTERNS.items():
        if contains_any_pattern(q, patterns):
            out.append(term)

    return unique_keep_order(out)


def extract_exclude_terms(query: str) -> List[str]:
    q = normalize_match_text(query)
    out = []

    for term, patterns in EXCLUDE_PATTERNS.items():
        if contains_any_pattern(q, patterns):
            out.append(term)

    return unique_keep_order(out)


def has_mixed_marker(query: str) -> bool:
    q = normalize_match_text(query)
    return contains_any_pattern(q, MIXED_MARKER_PATTERNS)


def has_descriptive_signals(query: str) -> bool:
    moods = extract_moods(query)
    genres = extract_genres(query)
    vocal = extract_vocal(query)
    energy = extract_energy(query)
    era = extract_era(query)
    pop_pref = extract_popularity_preference(query)
    include = extract_include_terms(query)
    exclude = extract_exclude_terms(query)

    return any([
        bool(moods),
        bool(genres),
        vocal != "unknown",
        energy != "unknown",
        era is not None,
        pop_pref is not None,
        bool(include),
        bool(exclude),
        ])


def looks_like_artist_query(query: str) -> bool:
    """
    只在“很像纯艺人名”的时候才判 artist。
    """
    q = normalize_query_text(query)
    if not q:
        return False

    # 有明显混合/比较语义，直接不是纯 artist
    if has_mixed_marker(q):
        return False

    # 有明显描述性约束，也不是纯 artist
    if has_descriptive_signals(q):
        return False

    # 中英文标点都算
    if any(ch in q for ch in [",", ".", ";", "?", "!", ":", "(", ")"]):
        return False

    tokens = tokenize_query(q)
    if not (1 <= len(tokens) <= 4):
        return False

    return True


def classify_query(query: str) -> Dict[str, Any]:
    q = normalize_query_text(query)

    artist_seeds = extract_artist_seeds(q)
    moods = extract_moods(q)
    genres = extract_genres(q)
    vocal = extract_vocal(q)
    energy = extract_energy(q)
    era = extract_era(q)
    popularity_preference = extract_popularity_preference(q)
    include = extract_include_terms(q)
    exclude = extract_exclude_terms(q)

    mixed = bool(artist_seeds) and (
            has_mixed_marker(q)
            or bool(moods)
            or bool(genres)
            or vocal != "unknown"
            or energy != "unknown"
            or era is not None
            or popularity_preference is not None
            or bool(include)
            or bool(exclude)
    )

    if mixed:
        return {
            "query_type": "mixed",
            "normalized_query": q,
            "artist_seeds": artist_seeds,
            "genres": genres,
            "moods": moods,
            "vocal": vocal,
            "energy": energy,
            "era": era,
            "popularity_preference": popularity_preference,
            "include": include,
            "exclude": exclude,
            "raw_query": q,
        }

    if looks_like_artist_query(q):
        return {
            "query_type": "artist",
            "normalized_query": q,
            "artist_seeds": [q],
            "genres": [],
            "moods": [],
            "vocal": "unknown",
            "energy": "unknown",
            "era": None,
            "popularity_preference": None,
            "include": [],
            "exclude": [],
            "raw_query": q,
        }

    return {
        "query_type": "semantic",
        "normalized_query": q,
        "artist_seeds": [],
        "genres": genres,
        "moods": moods,
        "vocal": vocal,
        "energy": energy,
        "era": era,
        "popularity_preference": popularity_preference,
        "include": include,
        "exclude": exclude,
        "raw_query": q,
    }