from __future__ import annotations

from typing import Any

from app.services.explanation_service import ExplanationService


class ResponseBuilder:
    """
    统一负责把内部结果对象转成 API response。

    RecommendationService 负责流程调度；
    ResponseBuilder 负责输出结构。
    """

    def __init__(self, explanation_service: ExplanationService):
        self.explanation_service = explanation_service

    def build(self, payload: dict[str, Any], include_debug: bool = False) -> dict[str, Any]:
        output_results: list[dict[str, Any]] = []
        route = payload.get("parsed_query", {})
        fallback_used = bool(payload.get("fallback_used"))

        for item in payload.get("results", []):
            reason = item.get("reason") or self.explanation_service.build_listener_reason(
                item=item,
                route=route,
                fallback_used=fallback_used,
            )
            result = {
                "id": item.get("id"),
                "title": item.get("title"),
                "artist": item.get("artist"),
                "album": item.get("album"),
                "release_year": item.get("release_year"),
                "popularity_bucket": item.get("popularity_bucket"),
                "language": item.get("language"),
                "reason": reason,
            }

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
                        "score": item.get("score"),
                        "heuristic_score": item.get("heuristic_score"),
                        "llm_score": item.get("llm_score"),
                        "match_type": item.get("match_type"),
                        "match_evidence": self.explanation_service.build_match_evidence(
                            item=item,
                            route=route,
                            fallback_used=fallback_used,
                        ),
                    }
                )

            output_results.append(result)

        out = {
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

    def build_search_response(
            self,
            query: str,
            route: dict[str, Any],
            fallback_used: bool,
            results: list[dict[str, Any]],
            semantic_query_used: str | None = None,
    ) -> dict[str, Any]:
        query_type = route.get("query_type", "semantic")

        output_results: list[dict[str, Any]] = []

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
                    "vocal_type": item.get("vocal_type"),
                    "genre_text": item.get("genre_text"),
                    "style_text": item.get("style_text"),
                    "mood_text": item.get("mood_text"),
                    "score": item.get("score"),
                    "match_type": item.get("match_type"),
                    "reason": self.explanation_service.build_reason(
                        item=item,
                        route=route,
                        fallback_used=fallback_used,
                    ),
                }
            )

        return {
            "query": query,
            "parsed_query": route,
            "query_type": query_type,
            "semantic_query_used": semantic_query_used,
            "fallback_used": fallback_used,
            "result_count": len(output_results),
            "results": output_results,
        }
