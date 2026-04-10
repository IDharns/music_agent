from pathlib import Path
from typing import Any, Dict, List

from app.recommender import MusicRetriever
from app.postprocess import dedup_and_diversify
from app.query_router import classify_query


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "music.db"
INDEX_PATH = BASE_DIR / "data" / "faiss.index"
IDS_PATH = BASE_DIR / "data" / "ids.npy"


def build_reason(item: Dict[str, Any], query_type: str, fallback_used: bool) -> str:
    match_type = item.get("match_type", "unknown")

    if match_type == "artist_exact":
        return "Matched artist name exactly."
    if match_type == "semantic" and fallback_used and query_type == "artist":
        return "Artist lookup missed, so semantic retrieval was used as fallback."
    if match_type == "semantic":
        return "Retrieved by semantic similarity against the query."
    return "Retrieved by the system."


def build_response(
    query: str,
    route: Dict[str, Any],
    fallback_used: bool,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    query_type = route["query_type"]

    output_results = []
    for item in results:
        output_results.append(
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "artist": item.get("artist"),
                "album": item.get("album"),
                "release_year": item.get("release_year"),
                "popularity": item.get("popularity"),
                "language": item.get("language"),
                "score": item.get("score"),
                "match_type": item.get("match_type"),
                "reason": build_reason(item, query_type, fallback_used),
            }
        )

    return {
        "query": query,
        "query_type": query_type,
        "fallback_used": fallback_used,
        "result_count": len(output_results),
        "results": output_results,
    }


def print_response(response: Dict[str, Any]) -> None:
    print("\n=== RESPONSE ===")
    print(f"query: {response['query']}")
    print(f"query_type: {response['query_type']}")
    print(f"fallback_used: {response['fallback_used']}")
    print(f"result_count: {response['result_count']}")

    if not response["results"]:
        print("\nNo results.")
        return

    print("\nFinal recommendations:")
    for i, item in enumerate(response["results"], start=1):
        print(
            f"{i:2d}. "
            f"score={item['score']:.4f} | "
            f"{item['title']} - {item['artist']} | "
            f"album={item['album']} | "
            f"year={item['release_year']} | "
            f"id={item['id']} | "
            f"match={item['match_type']} | "
            f"reason={item['reason']}"
        )


def run_query(retriever: MusicRetriever, query: str) -> Dict[str, Any]:
    route = classify_query(query)
    fallback_used = False

    if route["query_type"] == "artist":
        raw_candidates = retriever.search_by_artist(
            route["normalized_query"],
            top_k=50,
        )

        if not raw_candidates:
            fallback_used = True
            raw_candidates = retriever.search(query, top_k=50)
    else:
        raw_candidates = retriever.search(query, top_k=50)

    final_results = dedup_and_diversify(
        raw_candidates,
        final_k=10,
        max_per_artist=3,
    )

    response = build_response(
        query=query,
        route=route,
        fallback_used=fallback_used,
        results=final_results,
    )
    return response


def main():
    retriever = MusicRetriever(
        db_path=str(DB_PATH),
        index_path=str(INDEX_PATH),
        ids_path=str(IDS_PATH),
    )

    while True:
        query = input("\nEnter your query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        response = run_query(retriever, query)
        print_response(response)


if __name__ == "__main__":
    main()