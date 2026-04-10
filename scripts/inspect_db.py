# inspect_db.py
import sqlite3
import pandas as pd

conn = sqlite3.connect("../data/music.db")

print(pd.read_sql_query("SELECT COUNT(*) AS n FROM tracks", conn))
print(pd.read_sql_query("SELECT COUNT(*) AS n FROM artists", conn))
print(pd.read_sql_query("""
    SELECT title, artist_name, album_name, release_year, duration_ms
    FROM tracks
    LIMIT 10
""", conn))

conn.close()