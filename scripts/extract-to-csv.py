# extract_msd_to_csv.py
import sqlite3
from pathlib import Path
import pandas as pd

MSD_DB_PATH = "../data/track_metadata.db"  # 改成你的实际路径
OUTPUT_CSV_PATH = "../data/msd_tracks.csv"


def main():
    Path("../data").mkdir(exist_ok=True)

    conn = sqlite3.connect(MSD_DB_PATH)

    # 先看看有哪些表
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;",
        conn
    )
    print("Tables:")
    print(tables)

    # MSD 常见主表是 songs
    query = """
    SELECT
        track_id,
        title,
        artist_name,
        release AS album_name,
        year,
        duration
    FROM songs
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"导出前记录数: {len(df)}")
    print(df.head(10))

    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"已导出到: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()