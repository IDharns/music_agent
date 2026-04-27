from __future__ import annotations

from typing import Any, TypedDict


class _TrackCore(TypedDict):
    """
    Fields always present on a Track candidate produced by VectorRetriever.

    This is the canonical internal representation that flows through the entire
    pipeline (retrieval → reranking → response building).
    """

    # Identity
    id: int
    title: str
    artist: str
    album: str | None
    release_year: int | None

    # Popularity signals
    popularity: float | None
    popularity_bucket: str | None
    popularity_proxy: float | None

    # Metadata
    language: str | None
    vocal_type: str | None
    genre_text: str | None
    style_text: str | None
    mood_text: str | None

    # Artist / album participants
    primary_artists: list[str]
    featured_artists: list[str]
    all_contributors: list[str]

    # Tag vectors
    artist_tags: list[str]
    album_tags: list[str]
    style_tags: list[str]
    mood_anchors: list[str]
    mood_confidence: float | None

    # Combined deduped artist_tags + album_tags.
    # Used internally by LLMCandidateReranker; NOT exposed in the API response.
    tags: list[str]

    # Retrieval scores
    score: float
    match_type: str


class Track(_TrackCore, total=False):
    """
    Internal candidate dict.

    The pipeline may append these optional fields downstream:
      - heuristic_score: set by RecommendationService after LLM rerank merge
      - llm_score: set by LLMCandidateReranker
      - reason: set by LLMCandidateReranker or ResponseBuilder
    """

    heuristic_score: float | None
    llm_score: float | None
    reason: str | None


# ---------------------------------------------------------------------------
# API response shapes
# ---------------------------------------------------------------------------


class _TrackResponseBase(TypedDict):
    """
    Fields present in every API track result, regardless of include_debug.
    """

    id: int
    title: str
    artist: str
    album: str | None
    release_year: int | None
    popularity_bucket: str | None
    language: str | None
    score: float | None
    match_type: str | None
    reason: str | None


class TrackResponse(_TrackResponseBase, total=False):
    """
    API-facing track shape returned by /search.

    _TrackResponseBase fields are always present.
    The fields below are only included when include_debug=True.
    """

    popularity: float | None
    popularity_proxy: float | None
    vocal_type: str | None
    genre_text: str | None
    style_text: str | None
    mood_text: str | None
    primary_artists: list[str]
    featured_artists: list[str]
    all_contributors: list[str]
    style_tags: list[str]
    mood_anchors: list[str]
    artist_tags: list[str]
    album_tags: list[str]
    mood_confidence: float | None
    heuristic_score: float | None
    llm_score: float | None
    match_evidence: dict[str, Any] | None
