import time
import json
import requests
import sqlite3
from app.config import Settings

BASE_URL = "http://ws.audioscrobbler.com/2.0/"

DB_PATH = "../data/music.db"

SLEEP_SEC = 0.2   # 防止被限速


# ====== tag 分类规则（第一版够用） ======

GENRE_STYLE_WHITELIST = {
    "pop", "indie pop", "dream pop", "shoegaze", "synthpop",
    "electronic", "hip-hop", "rap", "r&b", "folk", "indie folk",
    "rock", "alternative", "alternative rock", "j-pop", "k-pop",
    "country", "country pop"
}

MOOD_WHITELIST = {
    "dreamy", "melancholic", "chill", "soft", "romantic",
    "dark", "energetic", "uplifting", "ethereal", "calm"
}

VOCAL_TAGS = {
    "female vocalists": "female vocal",
    "male vocalists": "male vocal",
    "instrumental": "instrumental"
}

DROP_TAGS = {
    "favorites", "favourite", "seen live", "awesome", "beautiful",
    "my favorite", "love", "good"
}


def normalize_tag(tag: str) -> str:
    return tag.strip().lower()


def fetch_artist_tags(artist_name: str):
    params = {
        "method": "artist.getTopTags",
        "artist": artist_name,
        "api_key": Settings.LASTFM_API_KEY,
        "format": "json"
    }

    try:
        r = requests.get(BASE_URL, params=params, timeout=10)
        data = r.json()
    except Exception as e:
        print(f"[ERROR] {artist_name}: {e}")
        return []

    tags = data.get("toptags", {}).get("tag", [])
    results = []

    for t in tags:
        name = normalize_tag(t.get("name", ""))
        count = t.get("count", 0)

        if not name or name in DROP_TAGS:
            continue

        results.append((name, count))

    # 按权重排序
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def classify_tags(tag_list):
    genres = []
    moods = []
    vocal = None

    for tag, _ in tag_list:
        if tag in GENRE_STYLE_WHITELIST:
            genres.append(tag)

        if tag in MOOD_WHITELIST:
            moods.append(tag)

        if tag in VOCAL_TAGS:
            vocal = VOCAL_TAGS[tag]

    return genres, moods, vocal


def build_embedding_text(artist_name, genres, moods, vocal):
    parts = []

    parts.append(f"Artist: {artist_name}")

    if genres:
        parts.append("Genres: " + ", ".join(genres[:5]))

    if moods:
        parts.append("Mood: " + ", ".join(moods[:5]))

    if vocal:
        parts.append(f"Vocal: {vocal}")

    return " | ".join(parts)


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
                SELECT id, artist_name
                FROM artists
                """)
    rows = cur.fetchall()

    print(f"Total artists: {len(rows)}")

    for i, (artist_id, artist_name) in enumerate(rows):
        if not artist_name:
            continue

        print(f"[{i}] Processing: {artist_name}")

        tag_list = fetch_artist_tags(artist_name)

        genres, moods, vocal = classify_tags(tag_list)

        tags_json = json.dumps(
            [{"tag": t, "count": c} for t, c in tag_list[:20]],
            ensure_ascii=False
        )

        embedding_text = build_embedding_text(
            artist_name, genres, moods, vocal
        )

        cur.execute("""
                    UPDATE artists
                    SET artist_tags_json = ?,
                        genre_text = ?,
                        style_text = ?,
                        vocal_type = ?,
                        embedding_text = ?
                    WHERE id = ?
                    """, (
                        tags_json,
                        ", ".join(genres[:5]) if genres else None,
                        ", ".join(genres[:5]) if genres else None,
                        vocal,
                        embedding_text,
                        artist_id
                    ))

        conn.commit()

        time.sleep(SLEEP_SEC)

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()