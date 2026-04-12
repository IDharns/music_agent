from __future__ import annotations

from typing import Any

from app.llm.openrouter_client import OpenRouterClient


class LLMCandidateReranker:
    def __init__(self, client: OpenRouterClient) -> None:
        self.client = client

    def rerank(
            self,
            raw_query: str,
            parsed_query: dict[str, Any],
            candidates: list[dict[str, Any]],
            top_k: int = 10,
            llm_hints: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        compact_candidates = []
        for item in candidates:
            compact_candidates.append(
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "artist": item.get("artist"),
                    "album": item.get("album"),
                    "release_year": item.get("release_year"),
                    "genre_text": item.get("genre_text"),
                    "style_text": item.get("style_text"),
                    "mood_text": item.get("mood_text"),
                    "style_tags": item.get("style_tags", []),
                    "mood_anchors": item.get("mood_anchors", []),
                    "vocal_type": item.get("vocal_type"),
                    "popularity": item.get("popularity"),
                    "popularity_proxy": item.get("popularity_proxy"),
                    "popularity_bucket": item.get("popularity_bucket"),
                    "primary_artists": item.get("primary_artists", []),
                    "featured_artists": item.get("featured_artists", []),
                    "tags": item.get("tags", [])[:12],
                    "base_score": item.get("score"),
                }
            )

        system_prompt = """
You are a music recommendation reranker.

You will receive:
- the raw user query
- parsed query fields
- optional query rewrite hints
- a candidate list already retrieved by vector search

Your job:
- rank the candidates for recommendation quality
- use ONLY the provided metadata
- do not invent facts
- punish obvious mismatches:
  - male vocal when female vocal is requested
  - live/remix when excluded or clearly undesirable
  - hip-hop/rap when user asks for dreamy indie pop / Taylor Swift-like pop
  - candidates that only match a title word but not the musical metadata
- reward genre/style/vocal/tag matches
- prefer explicit style_tags, mood_anchors, vocal_type, and popularity_bucket over outside knowledge
- write "reason" as listener-facing recommendation copy
- do not list database field names in "reason"
- do not invent specific instrumentation, lyrics, production details, or biography that are not supported by the metadata
- output JSON only

Schema:
{
  "ranked": [
    {
      "id": int,
      "llm_score": float,
      "reason": str
    }
  ]
}
""".strip()

        user_prompt = f"""
raw_query:
{raw_query}

parsed_query:
{parsed_query}

llm_hints:
{llm_hints or {}}

top_k:
{top_k}

candidates:
{compact_candidates}
""".strip()

        result = self.client.chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=1200,
        )

        ranked = result.get("ranked")
        if not isinstance(ranked, list):
            return []

        out: list[dict[str, Any]] = []
        for item in ranked:
            if not isinstance(item, dict):
                continue
            try:
                song_id = int(item.get("id"))
                llm_score = float(item.get("llm_score", 0.0))
            except Exception:
                continue

            reason = str(item.get("reason", "")).strip()
            out.append(
                {
                    "id": song_id,
                    "llm_score": llm_score,
                    "reason": reason,
                }
            )

        return out
