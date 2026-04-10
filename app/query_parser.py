import json
import os
from typing import Any, Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


QueryType = Literal["artist", "semantic", "mixed", "unknown"]
VocalType = Literal["male", "female", "instrumental", "mixed", "unknown"]
EnergyType = Literal["low", "medium", "high", "unknown"]
PopularityType = Literal["more_popular", "less_popular", "unknown"]


class ParsedQuery(BaseModel):
    query_type: QueryType = "unknown"
    artist_seeds: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    moods: list[str] = Field(default_factory=list)
    vocal: VocalType = "unknown"
    energy: EnergyType = "unknown"
    era: Optional[str] = None
    popularity_preference: PopularityType = "unknown"
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    raw_query: str


SYSTEM_PROMPT = """
You are a strict music query parser.

Your task is to convert a user's music request into a compact JSON object.

Rules:
1. Output JSON only. No markdown. No explanation.
2. Use only the required schema.
3. If information is not clearly present, use:
   - empty list for list fields
   - "unknown" for enum fields
   - null for era when absent
4. Do not invent artist names, genres, moods, or constraints.
5. If the user mentions one or more specific artists, put them in artist_seeds.
6. query_type rules:
   - "artist" if the query is primarily an artist lookup with little or no extra semantic constraint
   - "semantic" if the query is primarily descriptive and does not rely on named artists
   - "mixed" if the query includes both artist seeds and semantic constraints
   - "unknown" otherwise
7. Normalize genre terms into concise lowercase labels.
8. Normalize mood terms into concise lowercase labels.
9. vocal must be one of: male, female, instrumental, mixed, unknown
10. energy must be one of: low, medium, high, unknown
11. popularity_preference must be one of: more_popular, less_popular, unknown
12. For decade-style time expressions, use values like: 1980s, 1990s, 2000s, 2010s, 2020s.
13. For exclusion constraints, prefer concise normalized values such as: loud, live, remix, instrumental, explicit.

Return exactly this JSON schema:
{
  "query_type": "artist | semantic | mixed | unknown",
  "artist_seeds": [],
  "genres": [],
  "moods": [],
  "vocal": "male | female | instrumental | mixed | unknown",
  "energy": "low | medium | high | unknown",
  "era": null,
  "popularity_preference": "more_popular | less_popular | unknown",
  "include": [],
  "exclude": [],
  "raw_query": ""
}
""".strip()


class LLMQueryParser:
    def __init__(
        self,
        model: str = "openai/gpt-4.1-mini",
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_tokens: int = 300,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url=base_url,
        )

    def parse(self, query: str) -> ParsedQuery:
        raw_text = self._call_llm(query)
        data = self._safe_load_json(raw_text, query)
        return self._validate_and_repair(data, query)

    def _call_llm(self, query: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Parse this music request into JSON:\n\n{query}",
                },
            ],
        )
        return completion.choices[0].message.content or ""

    def _safe_load_json(self, text: str, query: str) -> dict[str, Any]:
        text = text.strip()

        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        return self._default_payload(query)

    def _default_payload(self, query: str) -> dict[str, Any]:
        return {
            "query_type": "unknown",
            "artist_seeds": [],
            "genres": [],
            "moods": [],
            "vocal": "unknown",
            "energy": "unknown",
            "era": None,
            "popularity_preference": "unknown",
            "include": [],
            "exclude": [],
            "raw_query": query,
        }

    def _validate_and_repair(self, data: dict[str, Any], query: str) -> ParsedQuery:
        merged = {**self._default_payload(query), **data}
        merged["raw_query"] = query

        for key in ["artist_seeds", "genres", "moods", "include", "exclude"]:
            merged[key] = self._ensure_clean_str_list(merged.get(key, []))

        for key in ["query_type", "vocal", "energy", "popularity_preference"]:
            merged[key] = str(merged.get(key, "unknown")).strip().lower()

        era_value = merged.get("era")
        if era_value is not None:
            era_value = str(era_value).strip()
            merged["era"] = era_value if era_value else None

        merged["genres"] = self._normalize_genres(merged["genres"])
        merged["moods"] = self._normalize_moods(merged["moods"])
        merged["exclude"] = self._normalize_exclude_list(merged["exclude"])
        merged["include"] = self._normalize_include_list(merged["include"])
        merged["era"] = self._normalize_era(merged["era"])
        merged["artist_seeds"] = self._normalize_artist_seeds(merged["artist_seeds"])

        try:
            return ParsedQuery.model_validate(merged)
        except ValidationError:
            return ParsedQuery(raw_query=query)

    def _ensure_clean_str_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned = []
        for x in value:
            s = str(x).strip()
            if s:
                cleaned.append(s)
        return cleaned

    def _dedup_keep_order(self, items: list[str]) -> list[str]:
        seen = set()
        output = []
        for item in items:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                output.append(item)
        return output

    def _normalize_artist_seeds(self, items: list[str]) -> list[str]:
        result = []
        for item in items:
            s = " ".join(item.strip().split())
            if not s:
                continue
            s = s.title()
            result.append(s)
        return self._dedup_keep_order(result)

    def _normalize_genres(self, items: list[str]) -> list[str]:
        mapping = {
            "edm": "electronic",
            "electronica": "electronic",
            "electro": "electronic",
            "electronic music": "electronic",
            "dance-pop": "dance",
            "dance pop": "dance",
            "hip hop": "hip-hop",
            "rap": "hip-hop",
            "rnb": "r&b",
            "indie pop": "indie",
            "indie rock": "indie",
            "alt rock": "rock",
            "alternative rock": "rock",
        }

        result = []
        for item in items:
            x = item.strip().lower()
            x = mapping.get(x, x)
            result.append(x)

        return self._dedup_keep_order(result)

    def _normalize_moods(self, items: list[str]) -> list[str]:
        mapping = {
            "dream-like": "dreamy",
            "airy": "light",
            "gentle": "soft",
            "peaceful": "calm",
            "relaxing": "calm",
            "moody": "dark",
            "melancholy": "melancholic",
            "sad": "melancholic",
            "bright": "uplifting",
            "cozy": "warm",
            "study": "focused",
            "late night": "late-night",
            "night": "late-night",
            "exciting": "energetic",
        }

        result = []
        for item in items:
            x = item.strip().lower().replace("_", "-")
            x = mapping.get(x, x)
            result.append(x)

        return self._dedup_keep_order(result)

    def _normalize_exclude_list(self, items: list[str]) -> list[str]:
        result = []
        for item in items:
            x = item.strip().lower()

            if x in {"too loud", "loud", "noisy", "aggressive", "harsh"}:
                result.append("loud")
            elif x in {"live version", "live", "concert", "concert version"}:
                result.append("live")
            elif x in {"remix", "remixes", "remixed"}:
                result.append("remix")
            elif x in {"instrumental", "no vocal", "without vocals"}:
                result.append("instrumental")
            elif x in {"explicit", "dirty"}:
                result.append("explicit")
            else:
                result.append(x)

        return self._dedup_keep_order(result)

    def _normalize_include_list(self, items: list[str]) -> list[str]:
        result = []
        for item in items:
            x = item.strip().lower()

            if x in {"female vocal", "female vocals", "female voice"}:
                result.append("female vocal")
            elif x in {"male vocal", "male vocals", "male voice"}:
                result.append("male vocal")
            else:
                result.append(x)

        return self._dedup_keep_order(result)

    def _normalize_era(self, era: str | None) -> str | None:
        if era is None:
            return None

        e = era.strip().lower()

        mapping = {
            "80s": "1980s",
            "1980": "1980s",
            "1980s": "1980s",
            "90s": "1990s",
            "1990": "1990s",
            "1990s": "1990s",
            "00s": "2000s",
            "2000": "2000s",
            "2000s": "2000s",
            "10s": "2010s",
            "2010": "2010s",
            "2010s": "2010s",
            "20s": "2020s",
            "2020": "2020s",
            "2020s": "2020s",
            "recent": "recent",
            "new": "recent",
            "newer": "recent",
        }

        return mapping.get(e, era)