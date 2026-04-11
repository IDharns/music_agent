from __future__ import annotations

from typing import Any


class RecommendationService:
    def __init__(
            self,
            query_module: Any,
            retriever: Any,
            reranker: Any,
            response_builder: Any | None = None,
    ) -> None:
        self.query_module = query_module
        self.retriever = retriever
        self.reranker = reranker
        self.response_builder = response_builder

    def run_query(
            self,
            query: str,
            final_k: int = 10,
            max_per_artist: int = 3,
    ) -> dict[str, Any]:
        parsed_query = self.query_module.understand(query)
        query_type = parsed_query.get("query_type", "semantic")

        fallback_used = False
        semantic_query_used = self._build_semantic_query(
            raw_query=query,
            parsed_query=parsed_query,
        )

        if query_type == "artist":
            raw_candidates = self.retriever.search_by_artist(
                artist_name=parsed_query.get("normalized_query") or query,
                top_k=max(50, final_k * 8),
            )
            if not raw_candidates:
                fallback_used = True
                raw_candidates = self.retriever.search_semantic(
                    text=query,
                    top_k=max(80, final_k * 10),
                )
                semantic_query_used = query
            else:
                semantic_query_used = None

        elif query_type == "mixed":
            raw_candidates = self.retriever.search_semantic(
                text=semantic_query_used,
                top_k=max(100, final_k * 12),
            )

        else:
            raw_candidates = self.retriever.search_semantic(
                text=semantic_query_used,
                top_k=max(80, final_k * 10),
            )

        final_results = self.reranker.rerank(
            raw_candidates=raw_candidates,
            parsed_query=parsed_query,
            final_k=final_k,
            max_per_artist=max_per_artist,
        )

        payload: dict[str, Any] = {
            "query": query,
            "query_type": query_type,
            "parsed_query": parsed_query,
            "semantic_query_used": semantic_query_used,
            "fallback_used": fallback_used,
            "result_count": len(final_results),
            "results": final_results,
        }

        if self.response_builder is not None:
            build_fn = getattr(self.response_builder, "build", None)
            if callable(build_fn):
                return build_fn(payload)

        return payload

    def _build_semantic_query(
            self,
            raw_query: str,
            parsed_query: dict[str, Any],
    ) -> str:
        build_fn = getattr(self.query_module, "build_semantic_query", None)
        if callable(build_fn):
            built = build_fn(parsed_query)
            if isinstance(built, str) and built.strip():
                return built.strip()

        query_type = parsed_query.get("query_type", "semantic")

        if query_type == "artist":
            return raw_query

        parts: list[str] = []

        for key in ("genres", "moods", "include", "exclude"):
            value = parsed_query.get(key) or []
            if isinstance(value, list):
                parts.extend(str(x).strip() for x in value if str(x).strip())

        vocal = parsed_query.get("vocal")
        if vocal and vocal != "unknown":
            parts.append(str(vocal).strip())

        energy = parsed_query.get("energy")
        if energy and energy != "unknown":
            parts.append(str(energy).strip())

        era = parsed_query.get("era")
        if era:
            parts.append(str(era).strip())

        pop_pref = parsed_query.get("popularity_preference")
        if pop_pref == "less_popular":
            parts.extend(["less popular", "underrated"])
        elif pop_pref == "more_popular":
            parts.append("popular")

        deduped: list[str] = []
        seen: set[str] = set()
        for part in parts:
            p = " ".join(part.split()).strip()
            if not p:
                continue
            k = p.lower()
            if k not in seen:
                seen.add(k)
                deduped.append(p)

        return " ".join(deduped) if deduped else raw_query