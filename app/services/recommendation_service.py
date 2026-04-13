from __future__ import annotations

from typing import Any


class RecommendationService:
    def __init__(
            self,
            query_module: Any,
            retriever: Any,
            reranker: Any,
            response_builder: Any | None = None,
            llm_query_rewriter: Any | None = None,
            llm_candidate_reranker: Any | None = None,
            enable_llm_query_rewrite: bool = False,
            enable_llm_rerank: bool = False,
    ) -> None:
        self.query_module = query_module
        self.retriever = retriever
        self.reranker = reranker
        self.response_builder = response_builder
        self.llm_query_rewriter = llm_query_rewriter
        self.llm_candidate_reranker = llm_candidate_reranker
        self.enable_llm_query_rewrite = enable_llm_query_rewrite
        self.enable_llm_rerank = enable_llm_rerank

    def run_query(
            self,
            query: str,
            final_k: int = 10,
            max_per_artist: int = 3,
            include_debug: bool = False,
    ) -> dict[str, Any]:
        parsed_query = self.query_module.understand(query)
        query_type = parsed_query.get("query_type", "semantic")

        fallback_used = False
        llm_rewrite: dict[str, Any] | None = None

        semantic_query_base = self._build_semantic_query(
            raw_query=query,
            parsed_query=parsed_query,
        )
        semantic_query_llm: str | None = None
        semantic_query_used = semantic_query_base

        if (
                self.enable_llm_query_rewrite
                and self.llm_query_rewriter is not None
                and query_type != "artist"
        ):
            try:
                llm_rewrite = self.llm_query_rewriter.rewrite(
                    raw_query=query,
                    parsed_query=parsed_query,
                )
                semantic_query_llm = str(llm_rewrite.get("semantic_query", "")).strip() or None
                if semantic_query_llm:
                    semantic_query_used = self._merge_semantic_queries(
                        base_query=semantic_query_base,
                        llm_query=semantic_query_llm,
                    )
            except Exception as e:
                llm_rewrite = {"error": str(e)}
                semantic_query_used = semantic_query_base

        if query_type == "artist":
            artist_name = (
                    (parsed_query.get("artist_seeds") or [None])[0]
                    or parsed_query.get("normalized_query")
                    or query
            )
            raw_candidates = self.retriever.search_by_artist(
                artist_name=artist_name,
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
                semantic_query_base = None
                semantic_query_llm = None
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

        heuristic_candidates = self.reranker.rerank(
            raw_candidates=raw_candidates,
            parsed_query=parsed_query,
            final_k=max(25, final_k * 3),
            max_per_artist=final_k if query_type == "artist" else max_per_artist,
        )

        final_results = heuristic_candidates[:final_k]
        llm_rank_debug: list[dict[str, Any]] = []

        if (
                self.enable_llm_rerank
                and self.llm_candidate_reranker is not None
                and heuristic_candidates
                and query_type != "artist"
        ):
            try:
                llm_rank_debug = self.llm_candidate_reranker.rerank(
                    raw_query=query,
                    parsed_query=parsed_query,
                    candidates=heuristic_candidates[:25],
                    top_k=final_k,
                    llm_hints=llm_rewrite,
                )
                merged = self._merge_llm_rerank(
                    heuristic_candidates=heuristic_candidates,
                    llm_ranked=llm_rank_debug,
                    final_k=final_k,
                )
                if merged:
                    final_results = merged
            except Exception as e:
                llm_rank_debug = [{"error": str(e)}]

        payload: dict[str, Any] = {
            "query": query,
            "query_type": query_type,
            "parsed_query": parsed_query,
            "semantic_query_base": semantic_query_base,
            "semantic_query_llm": semantic_query_llm,
            "semantic_query_used": semantic_query_used,
            "fallback_used": fallback_used,
            "llm_rewrite": llm_rewrite,
            "llm_rank_debug": llm_rank_debug,
            "result_count": len(final_results),
            "results": final_results,
        }

        if self.response_builder is not None:
            build_fn = getattr(self.response_builder, "build", None)
            if callable(build_fn):
                return build_fn(payload, include_debug=include_debug)

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
        return raw_query

    def _merge_semantic_queries(self, base_query: str, llm_query: str) -> str:
        base_terms = [x.strip().lower() for x in base_query.split() if x.strip()]
        llm_terms = [x.strip().lower() for x in llm_query.split() if x.strip()]

        out: list[str] = []
        seen: set[str] = set()

        for term in base_terms:
            if term not in seen:
                seen.add(term)
                out.append(term)

        for term in llm_terms:
            if term not in seen:
                seen.add(term)
                out.append(term)

        return " ".join(out)

    def _merge_llm_rerank(
            self,
            heuristic_candidates: list[dict[str, Any]],
            llm_ranked: list[dict[str, Any]],
            final_k: int,
    ) -> list[dict[str, Any]]:
        by_id = {
            int(item["id"]): dict(item)
            for item in heuristic_candidates
            if item.get("id") is not None
        }

        merged: list[dict[str, Any]] = []
        used_ids: set[int] = set()
        used_keys: set[tuple[str, str]] = set()

        for item in llm_ranked:
            song_id = item.get("id")
            if song_id is None:
                continue

            song_id = int(song_id)
            base = by_id.get(song_id)
            if base is None:
                continue
            key = self._dedupe_key(base)
            if key in used_keys:
                continue

            heuristic_score = base.get("score")
            llm_score = item.get("llm_score")

            base["heuristic_score"] = heuristic_score
            base["llm_score"] = llm_score
            if llm_score is not None:
                base["score"] = llm_score
            if item.get("reason"):
                base["reason"] = item["reason"]

            merged.append(base)
            used_ids.add(song_id)
            used_keys.add(key)

            if len(merged) >= final_k:
                return merged

        for item in heuristic_candidates:
            song_id = item.get("id")
            if song_id is None:
                continue

            song_id = int(song_id)
            if song_id in used_ids:
                continue
            key = self._dedupe_key(item)
            if key in used_keys:
                continue

            fallback_item = dict(item)
            fallback_item["heuristic_score"] = fallback_item.get("score")
            fallback_item["llm_score"] = None
            merged.append(fallback_item)
            used_keys.add(key)

            if len(merged) >= final_k:
                break

        return merged

    def _dedupe_key(self, item: dict[str, Any]) -> tuple[str, str]:
        title = " ".join(str(item.get("title") or "").strip().lower().split())
        artist = " ".join(str(item.get("artist") or "").strip().lower().split())
        return title, artist
