from __future__ import annotations

from typing import Any


class ExplanationService:
    """
    Build listener-facing recommendation copy and developer-facing match evidence.
    """

    def build_listener_reason(
            self,
            item: dict[str, Any],
            route: dict[str, Any],
            fallback_used: bool,
    ) -> str:
        query_type = route.get("query_type", "semantic")
        match_type = item.get("match_type", "unknown")
        style_tags = self._as_list(item.get("style_tags"))
        mood_anchors = self._as_list(item.get("mood_anchors"))
        vocal_type = str(item.get("vocal_type") or "").strip().lower()
        popularity_bucket = str(item.get("popularity_bucket") or "").strip().lower()

        if match_type == "artist_exact":
            artist = item.get("artist") or "this artist"
            return f"A direct pick from {artist} that gives you a clean starting point in their catalog."

        if match_type == "semantic" and fallback_used and query_type == "artist":
            return "A nearby pick that keeps close to the sound and context of your search."

        if match_type == "semantic" and query_type == "mixed":
            phrase = self._sound_phrase(style_tags, mood_anchors, vocal_type, popularity_bucket)
            return f"This moves away from the seed artist while keeping {phrase}."

        if match_type == "semantic":
            phrase = self._sound_phrase(style_tags, mood_anchors, vocal_type, popularity_bucket)
            return f"A strong fit for this request, with {phrase}."

        phrase = self._sound_phrase(style_tags, mood_anchors, vocal_type, popularity_bucket)
        return f"This sits naturally in the direction you asked for, with {phrase}."

    def build_match_evidence(
            self,
            item: dict[str, Any],
            route: dict[str, Any],
            fallback_used: bool,
    ) -> dict[str, Any]:
        style_tags = self._as_list(item.get("style_tags"))
        mood_anchors = self._as_list(item.get("mood_anchors"))
        artist_tags = self._as_list(item.get("artist_tags"))

        query_styles = self._query_style_terms(route)
        query_moods = {
            str(x).strip().lower()
            for x in route.get("moods", []) or []
            if str(x).strip()
        }

        style_hits = [
            tag
            for tag in style_tags
            if tag.replace("_", " ") in query_styles or tag in query_styles
        ]
        mood_hits = [
            anchor
            for anchor in mood_anchors
            if anchor in query_moods or self._mood_matches(anchor, query_moods)
        ]

        vocal_pref = str(route.get("vocal") or "").strip().lower()
        vocal_type = str(item.get("vocal_type") or "").strip().lower()
        vocal_match = bool(vocal_pref and vocal_pref != "unknown" and vocal_pref == vocal_type)

        penalties: list[str] = []
        title = str(item.get("title") or "").lower()
        tag_blob = " ".join(artist_tags).lower()
        excludes = {
            str(x).strip().lower()
            for x in route.get("exclude", []) or []
            if str(x).strip()
        }

        if "live" in excludes and ("live" in title or "live" in tag_blob):
            penalties.append("excluded_live")
        if "remix" in excludes and ("remix" in title or "remix" in tag_blob):
            penalties.append("excluded_remix")

        return {
            "match_type": item.get("match_type"),
            "style_hits": style_hits,
            "mood_hits": mood_hits,
            "vocal_match": vocal_match,
            "popularity_bucket": item.get("popularity_bucket"),
            "fallback_used": fallback_used,
            "penalties": penalties,
        }

    def build_reason(
            self,
            item: dict[str, Any],
            route: dict[str, Any],
            fallback_used: bool,
    ) -> str:
        return self.build_listener_reason(item, route, fallback_used)

    def _query_style_terms(self, route: dict[str, Any]) -> set[str]:
        terms: set[str] = set()
        for key in ("genres", "include"):
            for value in route.get(key, []) or []:
                text = str(value).strip().lower()
                if text:
                    terms.add(text)

        vocal = str(route.get("vocal") or "").strip().lower()
        if vocal and vocal != "unknown":
            terms.add(vocal)

        return terms

    def _mood_matches(self, anchor: str, query_moods: set[str]) -> bool:
        if anchor == "melancholic" and "sad" in query_moods:
            return True
        if anchor == "dreamy" and {"dreamy", "ethereal"} & query_moods:
            return True
        return False

    def _sound_phrase(
            self,
            style_tags: list[str],
            mood_anchors: list[str],
            vocal_type: str,
            popularity_bucket: str,
    ) -> str:
        style_words = [self._humanize_tag(tag) for tag in style_tags[:2]]
        mood_words = [self._humanize_tag(tag) for tag in mood_anchors[:1]]

        parts: list[str] = []
        if mood_words:
            parts.append(f"a {mood_words[0]} feel")
        if vocal_type in {"female vocal", "male vocal"}:
            parts.append(vocal_type.replace(" vocal", " vocals"))
        if style_words:
            parts.append(" / ".join(style_words))
        if popularity_bucket == "low":
            parts.append("a less mainstream profile")

        if not parts:
            return "a sound that lines up with the request"
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts[:-1]) + f", and {parts[-1]}"

    def _humanize_tag(self, value: str) -> str:
        return str(value).replace("_", " ").replace("-", " ").strip().lower()

    def _as_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(x) for x in value if str(x).strip()]
