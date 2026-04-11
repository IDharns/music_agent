from __future__ import annotations

from typing import Any


class ExplanationService:
    """
    统一负责结果解释/原因文案生成。

    当前先只收最基础的 reason 逻辑，不改变任何推荐行为。
    后续如果要接模板 explanation 或 LLM explanation，就继续往这里扩。
    """

    def build_reason(
            self,
            item: dict[str, Any],
            route: dict[str, Any],
            fallback_used: bool,
    ) -> str:
        query_type = route.get("query_type", "semantic")
        match_type = item.get("match_type", "unknown")

        if match_type == "artist_exact":
            return "Matched artist name exactly."

        if match_type == "semantic" and fallback_used and query_type == "artist":
            return "Artist lookup missed, so semantic retrieval was used as fallback."

        if match_type == "semantic" and query_type == "mixed":
            return "Retrieved by semantic similarity and adjusted for mixed-query constraints."

        if match_type == "semantic":
            return "Retrieved by semantic similarity against the query."

        return "Retrieved by the system."