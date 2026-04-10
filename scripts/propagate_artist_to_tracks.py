import sqlite3

DB_PATH = "../data/music.db"


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 用 normalized_artist 对齐，比直接 artist_name 更稳
    cur.execute("""
                UPDATE tracks
                SET
                    artist_tags_json = (
                        SELECT a.artist_tags_json
                        FROM artists a
                        WHERE a.normalized_artist = tracks.normalized_artist
                    LIMIT 1
                    ),
                    genre_text = COALESCE(
                    genre_text,
                  (
                    SELECT a.genre_text
                    FROM artists a
                    WHERE a.normalized_artist = tracks.normalized_artist
                    LIMIT 1
                    )
                    ),
                    style_text = COALESCE(
                    style_text,
                  (
                    SELECT a.style_text
                    FROM artists a
                    WHERE a.normalized_artist = tracks.normalized_artist
                    LIMIT 1
                    )
                    ),
                    vocal_type = COALESCE(
                    vocal_type,
                  (
                    SELECT a.vocal_type
                    FROM artists a
                    WHERE a.normalized_artist = tracks.normalized_artist
                    LIMIT 1
                    )
                    ),
                    language = COALESCE(
                    language,
                  (
                    SELECT a.language
                    FROM artists a
                    WHERE a.normalized_artist = tracks.normalized_artist
                    LIMIT 1
                    )
                    ),
                    popularity = COALESCE(
                    popularity,
                  (
                    SELECT a.popularity
                    FROM artists a
                    WHERE a.normalized_artist = tracks.normalized_artist
                    LIMIT 1
                    )
                    ),
                    popularity_bucket = COALESCE(
                    popularity_bucket,
                  (
                    SELECT a.popularity_bucket
                    FROM artists a
                    WHERE a.normalized_artist = tracks.normalized_artist
                    LIMIT 1
                    )
                    )
                WHERE normalized_artist IS NOT NULL
                """)

    conn.commit()

    cur.execute("SELECT COUNT(*) FROM tracks WHERE artist_tags_json IS NOT NULL")
    n = cur.fetchone()[0]
    print(f"tracks with artist_tags_json: {n}")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()