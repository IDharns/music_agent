import time
import json
import requests
import sqlite3

API_KEY = "YOUR_LASTFM_API_KEY"
BASE_URL = "http://ws.audioscrobbler.com/2.0/"
DB_PATH = "../data/music.db"

SLEEP_SEC = 0.2
LIMIT = None   # 可以先设成 5000 调试


GENRE_STYLE_WHITELIST = {
    "pop", "indie pop", "dream pop", "shoegaze", "synthpop",
    "electronic", "hip-hop", "rap", "r&b", "folk", "indie folk",
    "rock", "alternative", "alternative rock", "j-pop", "k-pop",
    "country", "country pop", "ambient", "dance", "indie rock",
    "jazz", "classical", "metal", "punk", "blues", "soul"
}

MOOD_WHITELIST = {
    "dreamy", "melancholic", "chill", "soft", "romantic",
    "dark", "energetic", "uplifting", "ethereal", "calm",
    "sad", "happy", "mellow", "atmospheric", "beautiful"
}

VOCAL_TAGS = {
    "female vocalists": "female vocal",
    "male vocalists": "male vocal",
    "instrumental": "instrumental"
}

DROP_TAGS = {
    "favorites", "favourite", "seen live", "awesome", "beautiful songs",
    "my favorite", "love", "good", "under 2000 listeners", "00s", "90s", "80s"
}


def normalize_tag(tag: str) -> str:
    return tag.strip().lower()


def fetch_lastfm(method: str, **kwargs):
    params = {
        "method": method,
        "api_key": API_KEY,
        "format": "json",
    }
    params.update(kwargs)

    try:
        r = requests.get(BASE_URL, params=params, timeout=10)
        data = r.json()
        return data
    except Exception as e:
        return {"_error": str(e)}


def parse_tag_response(data):
    if "_error" in data:
        return []

    tags = data.get("toptags", {}).get("tag", [])
    out = []
    for t in tags:
        name = normalize_tag(t.get("name", ""))
        count = t.get("count", 0)
        if not name or name in DROP_TAGS:
            continue
        out.append((name, count))

    out.sort(key=lambda x: x[1], reverse=True)
    return out


def fetch_track_tags(artist_name: str, track_name: str):
    data = fetch_lastfm(
        "track.getTopTags",
        artist=artist_name,
        track=track_name,
        autocorrect=1,
    )
    return parse_tag_response(data)


def fetch_album_tags(artist_name: str, album_name: str):
    data = fetch_lastfm(
        "album.getTopTags",
        artist=artist_name,
        album=album_name,
        autocorrect=1,
    )
    return parse_tag_response(data)


def classify_tags(tag_list):
    genres = []
    moods = []
    vocal = None

    for tag, _ in tag_list:
        if tag in GENRE_STYLE_WHITELIST and tag not in genres:
            genres.append(tag)
        if tag in MOOD_WHITELIST and tag not in moods:
            moods.append(tag)
        if tag in VOCAL_TAGS and vocal is None:
            vocal = VOCAL_TAGS[tag]

    return genres, moods, vocal


def merge_unique(primary, fallback, max_items=10):
    out = []
    seen = set()

    for x in primary + fallback:
        if x not in seen:
            seen.add(x)
            out.append(x)
        if len(out) >= max_items:
            break

    return out


def build_track_embedding_text(
        title,
        artist_name,
        album_name,
        release_year,
        genre_text,
        style_text,
        mood_text,
        vocal_type,
        language,
        popularity_bucket,
        track_tags,
        artist_tags,
):
    parts = []

    if title:
        parts.append(f"Title: {title}")
    if artist_name:
        parts.append(f"Artist: {artist_name}")
    if album_name:
        parts.append(f"Album: {album_name}")
    if release_year:
        parts.append(f"Year: {release_year}")
        parts.append(f"Era: {(release_year // 10) * 10}s")

    if genre_text:
        parts.append(f"Genres: {genre_text}")
    if style_text:
        parts.append(f"Styles: {style_text}")
    if mood_text:
        parts.append(f"Mood: {mood_text}")
    if vocal_type:
        parts.append(f"Vocal: {vocal_type}")
    if language:
        parts.append(f"Language: {language}")
    if popularity_bucket:
        parts.append(f"Popularity: {popularity_bucket}")

    if track_tags:
        parts.append("Track tags: " + ", ".join(track_tags[:10]))
    if artist_tags:
        parts.append("Artist tags: " + ", ".join(artist_tags[:10]))

    return " | ".join(parts)


def load_artist_tags_json(s: str):
    if not s:
        return []
    try:
        arr = json.loads(s)
        out = []
        for x in arr:
            if isinstance(x, dict) and "tag" in x:
                tag = normalize_tag(x["tag"])
                if tag and tag not in DROP_TAGS:
                    out.append(tag)
            elif isinstance(x, str):
                tag = normalize_tag(x)
                if tag and tag not in DROP_TAGS:
                    out.append(tag)
        return out
    except Exception:
        return []


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    sql = """
          SELECT
              id,
              title,
              artist_name,
              album_name,
              release_year,
              language,
              vocal_type,
              popularity_bucket,
              artist_tags_json
          FROM tracks
          WHERE title IS NOT NULL
            AND artist_name IS NOT NULL \
          """

    if LIMIT:
        sql += f" LIMIT {LIMIT}"

    cur.execute(sql)
    rows = cur.fetchall()

    print(f"Total tracks to process: {len(rows)}")

    for i, row in enumerate(rows):
        (
            track_id,
            title,
            artist_name,
            album_name,
            release_year,
            language,
            old_vocal_type,
            popularity_bucket,
            artist_tags_json,
        ) = row

        print(f"[{i}] {artist_name} - {title}")

        artist_tags = load_artist_tags_json(artist_tags_json)

        metadata_source = {
            "track_tags": None,
            "album_tags": None,
            "artist_tags": "existing_track_field" if artist_tags else None,
        }

        track_tag_pairs = fetch_track_tags(artist_name, title)
        time.sleep(SLEEP_SEC)

        source_level = None

        if track_tag_pairs:
            source_level = "lastfm_track"
            metadata_source["track_tags"] = "lastfm"
        else:
            track_tag_pairs = []

            if album_name:
                album_tag_pairs = fetch_album_tags(artist_name, album_name)
                time.sleep(SLEEP_SEC)
            else:
                album_tag_pairs = []

            if album_tag_pairs:
                track_tag_pairs = album_tag_pairs
                source_level = "lastfm_album"
                metadata_source["album_tags"] = "lastfm"

        track_tags = [t for t, _ in track_tag_pairs[:10]]

        genres, moods, vocal_from_track = classify_tags(track_tag_pairs)
        merged_tags = merge_unique(track_tags, artist_tags, max_items=12)

        # style_text 第一版先和 genre_text 共用，后面再细分
        genre_text = ", ".join(genres[:5]) if genres else None
        style_text = ", ".join(genres[:5]) if genres else None
        mood_text = ", ".join(moods[:5]) if moods else None
        vocal_type = vocal_from_track or old_vocal_type

        tags_json = json.dumps(
            [{"tag": t, "count": c} for t, c in track_tag_pairs[:20]],
            ensure_ascii=False
        ) if track_tag_pairs else None

        embedding_text = build_track_embedding_text(
            title=title,
            artist_name=artist_name,
            album_name=album_name,
            release_year=release_year,
            genre_text=genre_text,
            style_text=style_text,
            mood_text=mood_text,
            vocal_type=vocal_type,
            language=language,
            popularity_bucket=popularity_bucket,
            track_tags=track_tags,
            artist_tags=artist_tags,
        )

        cur.execute("""
                    UPDATE tracks
                    SET
                        tags_json = COALESCE(?, tags_json),
                        genre_text = COALESCE(?, genre_text),
                        style_text = COALESCE(?, style_text),
                        mood_text = COALESCE(?, mood_text),
                        vocal_type = COALESCE(?, vocal_type),
                        metadata_source_json = ?,
                        embedding_text = ?
                    WHERE id = ?
                    """, (
                        tags_json,
                        genre_text,
                        style_text,
                        mood_text,
                        vocal_type,
                        json.dumps(metadata_source, ensure_ascii=False),
                        embedding_text,
                        track_id,
                    ))

        if i % 100 == 0:
            conn.commit()

    conn.commit()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()