import sqlite3


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute("""
                CREATE TABLE IF NOT EXISTS tracks (
                                                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                      track_id_internal TEXT UNIQUE,
                                                      source TEXT NOT NULL,
                                                      source_track_id TEXT,

                                                      title TEXT,
                                                      artist_name TEXT,
                                                      album_name TEXT,
                                                      release_year INTEGER,
                                                      duration_ms INTEGER,

                                                      popularity REAL,
                                                      popularity_bucket TEXT,

                                                      language TEXT,
                                                      vocal_type TEXT,
                                                      is_instrumental INTEGER DEFAULT 0,
                                                      explicit_flag INTEGER DEFAULT 0,

                                                      genre_text TEXT,
                                                      style_text TEXT,
                                                      mood_text TEXT,

                                                      lyrics_text TEXT,

                                                      tags_json TEXT,
                                                      artist_tags_json TEXT,

                                                      search_text TEXT,
                                                      embedding_text TEXT,

                                                      normalized_title TEXT,
                                                      normalized_artist TEXT,
                                                      duplicate_group_key TEXT,

                                                      metadata_source_json TEXT,
                                                      extra_json TEXT,

                                                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                                      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)

    cur.execute("""
                CREATE TABLE IF NOT EXISTS artists (
                                                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                       artist_id_internal TEXT UNIQUE,
                                                       source TEXT NOT NULL,
                                                       source_artist_id TEXT,

                                                       artist_name TEXT NOT NULL,
                                                       normalized_artist TEXT,

                                                       popularity REAL,
                                                       popularity_bucket TEXT,
                                                       language TEXT,
                                                       vocal_type TEXT,

                                                       genre_text TEXT,
                                                       style_text TEXT,

                                                       artist_tags_json TEXT,

                                                       embedding_text TEXT,
                                                       extra_json TEXT,

                                                       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)

    cur.execute("""
                CREATE TABLE IF NOT EXISTS track_artists (
                                                             track_id_internal TEXT NOT NULL,
                                                             artist_id_internal TEXT NOT NULL,
                                                             artist_order INTEGER DEFAULT 0,
                                                             role TEXT DEFAULT 'primary',
                                                             PRIMARY KEY (track_id_internal, artist_id_internal)
                    )
                """)

    cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tracks_source_track_id
                    ON tracks(source, source_track_id)
                """)

    cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tracks_norm
                    ON tracks(normalized_title, normalized_artist)
                """)

    cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tracks_artist_name
                    ON tracks(artist_name)
                """)

    cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tracks_release_year
                    ON tracks(release_year)
                """)

    cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tracks_pop_bucket
                    ON tracks(popularity_bucket)
                """)

    cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tracks_language
                    ON tracks(language)
                """)

    cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tracks_vocal_type
                    ON tracks(vocal_type)
                """)

    cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_artists_norm
                    ON artists(normalized_artist)
                """)

    conn.commit()


if __name__ == "__main__":
    conn = get_connection("data/music.db")
    create_tables(conn)
    conn.close()
    print("Schema created.")