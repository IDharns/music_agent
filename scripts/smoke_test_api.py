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
            params={
                "query": query,
                "final_k": 8,
                "max_per_artist": 2,
                "include_debug": True,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        print("=" * 120)
        print("query:", data["query"])
        print("query_type:", data.get("query_type"))
        print("semantic_query_base:", data.get("semantic_query_base"))
        print("semantic_query_llm:", data.get("semantic_query_llm"))
        print("semantic_query_used:", data.get("semantic_query_used"))
        print("fallback_used:", data.get("fallback_used"))
        print("result_count:", data.get("result_count"))
        print("parsed_query:", data.get("parsed_query"))
        print("-" * 120)

        for i, item in enumerate(data.get("results", []), start=1):
            evidence = item.get("match_evidence") or {}
            print(
                f"{i:>2}. score={item.get('score')} | "
                f"heuristic={item.get('heuristic_score')} | "
                f"llm={item.get('llm_score')} | "
                f"{item.get('title')} - {item.get('artist')} | "
                f"style_tags={item.get('style_tags')} | "
                f"mood_anchors={item.get('mood_anchors')} | "
                f"vocal={item.get('vocal_type')} | "
                f"popularity_bucket={item.get('popularity_bucket')} | "
                f"evidence={evidence} | "
                f"reason={item.get('reason')}"
            )


if __name__ == "__main__":
    main()
