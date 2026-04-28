from __future__ import annotations

from typing import Any

from app.models import Track, TrackResponse
from app.services.explanation_service import ExplanationService


class ResponseBuilder:
    """
    Converts internal pipeline results into the API response.

    RecommendationService owns orchestration; ResponseBuilder owns output shape.
    The canonical track schema is defined in app.models.TrackResponse.
    """

    def __init__(self, explanation_service: ExplanationService) -> None:
        self.explanation_service = explanation_service

    def build(self, payload: dict[str, Any], include_debug: bool = False) -> dict[str, Any]:
        output_results: list[TrackResponse] = []
        route = payload.get("parsed_query", {})
        fallback_used = bool(payload.get("fallback_used"))

        for item in payload.get("results", []):
            reason = item.get("reason") or self.explanation_service.build_listener_reason(
                item=item,
                route=route,
                fallback_used=fallback_used,
            )

            # Base fields — always present in every response.
            result: TrackResponse = {
                "id": item.get("id"),
                "title": item.get("title"),
                "artist": item.get("artist"),
                "album": item.get("album"),
                "release_year": item.get("release_year"),
                "popularity_bucket": item.get("popularity_bucket"),
                "language": item.get("language"),
                "score": item.get("score"),
                "similarity": item.get("similarity"),
                "tag_overlap": item.get("tag_overlap"),
                "match_type": item.get("match_type"),
                "reason": reason,
            }

            # Debug fields — only included when include_debug=True.
            if include_debug:
                result.update(
                    {
                        "popularity": item.get("popularity"),
                        "popularity_proxy": item.get("popularity_proxy"),
                        "vocal_type": item.get("vocal_type"),
                        "genre_text": item.get("genre_text"),
                        "style_text": item.get("style_text"),
                        "mood_text": item.get("mood_text"),
                        "primary_artists": item.get("primary_artists", []),
                        "featured_artists": item.get("featured_artists", []),
                        "all_contributors": item.get("all_contributors", []),
                        "style_tags": item.get("style_tags", []),
                        "mood_anchors": item.get("mood_anchors", []),
                        "artist_tags": item.get("artist_tags", []),
                        "album_tags": item.get("album_tags", []),
                        "mood_confidence": item.get("mood_confidence"),
                        "heuristic_score": item.get("heuristic_score"),
                        "llm_score": item.get("llm_score"),
                        "match_evidence": self.explanation_service.build_match_evidence(
                            item=item,
                            route=route,
                            fallback_used=fallback_used,
                        ),
                    }
                )

            output_results.append(result)

        out: dict[str, Any] = {
            "query": payload.get("query"),
            "query_type": payload.get("query_type"),
            "fallback_used": fallback_used,
        }

        if include_debug:
            out.update(
                {
                    "parsed_query": payload.get("parsed_query"),
                    "semantic_query_base": payload.get("semantic_query_base"),
                    "semantic_query_llm": payload.get("semantic_query_llm"),
                    "semantic_query_used": payload.get("semantic_query_used"),
                    "llm_rewrite": payload.get("llm_rewrite"),
                    "llm_rank_debug": payload.get("llm_rank_debug"),
                }
            )

        out["result_count"] = len(output_results)
        out["results"] = output_results
        return out
