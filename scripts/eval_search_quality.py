from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8000/search"
DEFAULT_QUERIES_PATH = Path(__file__).resolve().parents[1] / "eval_queries.json"


def normalize_term(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ")
    return " ".join(text.split())


def term_matches(term: str, haystack: str) -> bool:
    term = normalize_term(term)
    if not term:
        return False
    if term == "acoustic" and any(
        re.search(rf"(?<![a-z0-9]){re.escape(proxy)}(?![a-z0-9])", haystack)
        for proxy in ("acoustic", "unplugged", "folk", "singer songwriter", "indie folk")
    ):
        return True
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", haystack))


def item_haystack(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "title",
        "artist",
        "album",
        "language",
        "vocal_type",
        "popularity_bucket",
        "genre_text",
        "style_text",
        "mood_text",
    ):
        value = item.get(key)
        if value:
            parts.append(str(value))

    for key in (
        "style_tags",
        "mood_anchors",
        "artist_tags",
        "album_tags",
        "primary_artists",
        "featured_artists",
        "all_contributors",
    ):
        value = item.get(key)
        if isinstance(value, list):
            parts.extend(str(x) for x in value)

    evidence = item.get("match_evidence") or {}
    for key in ("style_hits", "mood_hits", "penalties"):
        value = evidence.get(key)
        if isinstance(value, list):
            parts.extend(str(x) for x in value)

    return normalize_term(" ".join(parts))


def has_any(item: dict[str, Any], terms: list[str]) -> bool:
    haystack = item_haystack(item)
    return any(term_matches(term, haystack) for term in terms)


def has_all(item: dict[str, Any], terms: list[str]) -> bool:
    haystack = item_haystack(item)
    return all(term_matches(term, haystack) for term in terms)


def has_violation(item: dict[str, Any], avoid_terms: list[str]) -> bool:
    haystack = item_haystack(item)
    title = normalize_term(item.get("title"))
    artist = normalize_term(item.get("artist"))
    compact = f"{title} {artist} {haystack}"
    return any(term_matches(term, compact) for term in avoid_terms)


def summarize_query(spec: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    results = data.get("results") or []
    must_have = [normalize_term(x) for x in spec.get("must_have", []) if normalize_term(x)]
    nice_to_have = [normalize_term(x) for x in spec.get("nice_to_have", []) if normalize_term(x)]
    avoid = [normalize_term(x) for x in spec.get("avoid", []) if normalize_term(x)]

    total = len(results)
    if total == 0:
        return {
            "query": spec.get("query"),
            "intent": spec.get("intent"),
            "result_count": 0,
            "must_have_rate": 0.0,
            "nice_to_have_rate": 0.0,
            "violation_rate": 0.0,
            "artist_diversity": 0,
            "needs_review": True,
            "top_results": [],
        }

    must_matches = sum(1 for item in results if has_all(item, must_have)) if must_have else total
    nice_matches = sum(1 for item in results if has_any(item, nice_to_have)) if nice_to_have else total
    violations = sum(1 for item in results if has_violation(item, avoid)) if avoid else 0
    artists = {
        normalize_term(item.get("artist"))
        for item in results
        if normalize_term(item.get("artist"))
    }

    top_results = [
        {
            "rank": idx,
            "title": item.get("title"),
            "artist": item.get("artist"),
            "score": item.get("score"),
            "popularity_bucket": item.get("popularity_bucket"),
            "style_tags": item.get("style_tags", []),
            "mood_anchors": item.get("mood_anchors", []),
            "evidence": item.get("match_evidence"),
            "reason": item.get("reason"),
        }
        for idx, item in enumerate(results[:5], start=1)
    ]

    must_rate = must_matches / total
    nice_rate = nice_matches / total
    violation_rate = violations / total
    needs_review = bool(
        total == 0
        or (must_have and must_rate < 0.6)
        or (nice_to_have and nice_rate < 0.4)
        or violation_rate > 0.0
    )

    return {
        "query": spec.get("query"),
        "intent": spec.get("intent"),
        "query_type": data.get("query_type"),
        "semantic_query_used": data.get("semantic_query_used"),
        "result_count": total,
        "must_have_rate": round(must_rate, 3),
        "nice_to_have_rate": round(nice_rate, 3),
        "violation_rate": round(violation_rate, 3),
        "artist_diversity": len(artists),
        "needs_review": needs_review,
        "top_results": top_results,
    }


def print_markdown(summaries: list[dict[str, Any]]) -> None:
    total = len(summaries)
    review = sum(1 for item in summaries if item["needs_review"])
    avg_must = sum(float(item["must_have_rate"]) for item in summaries) / max(total, 1)
    avg_nice = sum(float(item["nice_to_have_rate"]) for item in summaries) / max(total, 1)
    avg_violation = sum(float(item["violation_rate"]) for item in summaries) / max(total, 1)

    print("# Search Quality Eval")
    print()
    print(f"- queries: {total}")
    print(f"- needs_review: {review}")
    print(f"- avg_must_have_rate: {avg_must:.3f}")
    print(f"- avg_nice_to_have_rate: {avg_nice:.3f}")
    print(f"- avg_violation_rate: {avg_violation:.3f}")
    print()

    for summary in summaries:
        marker = "REVIEW" if summary["needs_review"] else "OK"
        print(f"## {marker}: {summary['query']}")
        print()
        print(f"- intent: {summary.get('intent') or '-'}")
        print(f"- query_type: {summary.get('query_type') or '-'}")
        print(f"- semantic_query_used: {summary.get('semantic_query_used') or '-'}")
        print(f"- result_count: {summary['result_count']}")
        print(f"- must_have_rate: {summary['must_have_rate']}")
        print(f"- nice_to_have_rate: {summary['nice_to_have_rate']}")
        print(f"- violation_rate: {summary['violation_rate']}")
        print(f"- artist_diversity: {summary['artist_diversity']}")
        print()
        for item in summary["top_results"]:
            styles = ", ".join(item.get("style_tags") or []) or "-"
            moods = ", ".join(item.get("mood_anchors") or []) or "-"
            print(
                f"{item['rank']}. {item.get('title') or '-'} - {item.get('artist') or '-'} "
                f"(score={item.get('score')}, pop={item.get('popularity_bucket')})"
            )
            print(f"   styles: {styles}")
            print(f"   moods: {moods}")
            print(f"   reason: {item.get('reason') or '-'}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate search quality against fixed query specs.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--max-per-artist", type=int, default=3)
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    specs = json.loads(args.queries.read_text(encoding="utf-8"))
    summaries: list[dict[str, Any]] = []

    for spec in specs:
        response = requests.get(
            args.base_url,
            params={
                "query": spec["query"],
                "final_k": args.final_k,
                "max_per_artist": args.max_per_artist,
                "include_debug": True,
            },
            timeout=120,
        )
        response.raise_for_status()
        summaries.append(summarize_query(spec, response.json()))

    if args.format == "json":
        output_text = json.dumps(summaries, ensure_ascii=False, indent=2)
    else:
        from io import StringIO
        import contextlib

        buffer = StringIO()
        with contextlib.redirect_stdout(buffer):
            print_markdown(summaries)
        output_text = buffer.getvalue()

    if args.output:
        args.output.write_text(output_text, encoding="utf-8")
    else:
        print(output_text, end="")


if __name__ == "__main__":
    main()
