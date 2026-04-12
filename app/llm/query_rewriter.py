from __future__ import annotations

from typing import Any

from app.llm.openrouter_client import OpenRouterClient


class LLMQueryRewriter:
    def __init__(self, client: OpenRouterClient) -> None:
        self.client = client

    def rewrite(
            self,
            raw_query: str,
            parsed_query: dict[str, Any],
    ) -> dict[str, Any]:
        system_prompt = """
You are a music retrieval query rewriter.

Your task:
- Convert a user music request into a compact retrieval query for embedding search.
- Also produce structured constraints for reranking.
- Output JSON only.
- Do not recommend songs yet.
- Do not invent artists, genres, moods, languages, or years that are not grounded in the input.
- For mixed queries like "similar to Taylor Swift but dreamier and less popular":
  - DO NOT repeat the seed artist literal name in semantic_query.
  - Instead translate it into style anchors such as female vocal, pop, singer-songwriter, country pop, acoustic, etc.
- Keep semantic_query short and retrieval-friendly.
- Prefer descriptive musical terms over full natural language sentences.

Critical rule:
- You must preserve every explicit user constraint from the parsed query.
- Never remove moods, popularity preferences, exclusions, era constraints, or vocal constraints.
- You may only add helpful retrieval terms, not replace or weaken existing ones.

Required JSON schema:
{
  "semantic_query": str,
  "must_have": [str],
  "should_have": [str],
  "must_not": [str],
  "reasoning_focus": [str]
}
""".strip()

        user_prompt = f"""
raw_query:
{raw_query}

parsed_query:
{parsed_query}
""".strip()

        result = self.client.chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=400,
        )

        return {
            "semantic_query": str(result.get("semantic_query", "")).strip() or raw_query,
            "must_have": self._clean_list(result.get("must_have")),
            "should_have": self._clean_list(result.get("should_have")),
            "must_not": self._clean_list(result.get("must_not")),
            "reasoning_focus": self._clean_list(result.get("reasoning_focus")),
        }

    def _clean_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []

        out: list[str] = []
        seen: set[str] = set()

        for item in value:
            s = " ".join(str(item).strip().lower().split())
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)

        return out