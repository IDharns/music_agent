import requests


BASE_URL = "http://127.0.0.1:8000/search"

QUERIES = [
    "Adele",
    "sad female pop",
    "类似Taylor Swift但不要太热门，更梦幻一点",
    "80s synthpop but not live",
    "dreamy indie pop female vocal",
]


def main() -> None:
    for query in QUERIES:
        resp = requests.get(
            BASE_URL,
            params={"query": query, "final_k": 8, "max_per_artist": 2},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        print("=" * 100)
        print("query:", data["query"])
        print("query_type:", data.get("query_type"))
        print("semantic_query_used:", data.get("semantic_query_used"))
        print("fallback_used:", data.get("fallback_used"))
        print("result_count:", data.get("result_count"))
        print("parsed_query:", data.get("parsed_query"))
        print("-" * 100)

        for i, item in enumerate(data.get("results", []), start=1):
            print(
                f"{i:>2}. score={float(item.get('score', 0.0)):.4f} | "
                f"{item.get('title')} - {item.get('artist')} | "
                f"match={item.get('match_type')} | "
                f"genre={item.get('genre_text')} | "
                f"mood={item.get('mood_text')} | "
                f"vocal={item.get('vocal_type')} | "
                f"pop={item.get('popularity')}"
            )


if __name__ == "__main__":
    main()