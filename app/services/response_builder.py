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