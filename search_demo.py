import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


DB_PATH = "data/music.db"
IDS_PATH = "data/ids.npy"
INDEX_PATH = "data/faiss.index"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 10


def fetch_song_meta_by_id(db_path: str, song_id: int):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        SELECT id, title, artist_name, album_name
        FROM tracks
        WHERE id = ?
    """, (int(song_id),))

    row = cur.fetchone()
    conn.close()
    return row


def main():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading FAISS index and ids...")
    index = faiss.read_index(INDEX_PATH)
    song_ids = np.load(IDS_PATH)

    while True:
        query = input("\nEnter your query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        query_vec = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = index.search(query_vec, TOP_K)

        print("\nTop results:")
        rank = 1
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue

            song_id = int(song_ids[idx])
            row = fetch_song_meta_by_id(DB_PATH, song_id)
            if row is None:
                continue

            sid, title, artist_name, album_name = row
            print(f"{rank:2d}. score={score:.4f} | {title} - {artist_name} | album={album_name} | id={sid}")
            rank += 1


if __name__ == "__main__":
    main()