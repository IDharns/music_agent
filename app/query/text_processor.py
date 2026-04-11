from __future__ import annotations

import json
import re
import unicodedata
from typing import Any


class TextProcessor:
    _whitespace_re = re.compile(r"\s+")
    _quote_chars_re = re.compile(r"[“”‘’`]")
    _dash_chars_re = re.compile(r"[‐-–—]")

    def normalize_query(self, text: str) -> str:
        text = self.safe_str(text)
        text = unicodedata.normalize("NFKC", text)
        text = self._quote_chars_re.sub('"', text)
        text = self._dash_chars_re.sub("-", text)
        text = text.strip()
        text = self._whitespace_re.sub(" ", text)
        return text

    def normalize_artist_name(self, text: str) -> str:
        text = self.normalize_query(text)
        return text.casefold()

    def normalize_tag(self, text: str) -> str:
        text = self.normalize_query(text)
        return text.casefold()

    def safe_lower(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().casefold()

    def safe_str(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    def parse_tag_json(self, raw: str | None, max_items: int = 20) -> list[str]:
        if not raw:
            return []

        try:
            data = json.loads(raw)
        except Exception:
            return []

        tags: list[str] = []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    tag = item.get("tag")
                    if tag:
                        tags.append(self.normalize_query(str(tag)))
                elif isinstance(item, str):
                    tags.append(self.normalize_query(item))

        deduped: list[str] = []
        seen: set[str] = set()

        for tag in tags:
            key = tag.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(tag)
            if len(deduped) >= max_items:
                break

        return deduped

    def join_nonempty(self, parts: list[str], sep: str = " ") -> str:
        cleaned = [self.normalize_query(p) for p in parts if self.safe_str(p).strip()]
        return sep.join(cleaned).strip()