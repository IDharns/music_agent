# Feedback Roadmap

This document is intentionally separate from the main README. The current version is a baseline for local natural-language retrieval. The next iteration should focus on interactive feedback and discovery.

## Current Baseline

The current system supports:

- natural-language query parsing
- local SQLite + FAISS retrieval
- heuristic reranking
- optional OpenRouter query rewrite and candidate rerank
- a Next.js search UI
- fixed-query metadata consistency evaluation

It does not yet support:

- listening or preview playback
- like/dislike feedback
- long-running user profiles
- audio-content understanding
- personalized session memory

## Recommended Next Iteration

Build a swipe-style discovery interface. On desktop, the gestures can be simple buttons:

- Like
- Pass
- Love
- More like this
- Hide artist
- Too popular
- Wrong mood
- Wrong language

The goal is to turn recommendation into an iterative loop:

```text
initial query -> candidate pool -> one recommendation card -> feedback -> updated session profile -> next card
```

## Backend Shape

Keep the current `/search` endpoint as the baseline endpoint. Add a new discovery flow:

```text
POST /discover/start
GET  /discover/{session_id}/next
POST /discover/{session_id}/feedback
```

The session profile can start simple:

```json
{
  "liked_track_ids": [],
  "disliked_track_ids": [],
  "liked_artists": [],
  "hidden_artists": [],
  "liked_style_tags": {},
  "disliked_style_tags": {},
  "liked_mood_anchors": {},
  "disliked_mood_anchors": {},
  "language_preference": null,
  "popularity_preference": null
}
```

## Ranking Loop

For each feedback event:

1. Update the session profile.
2. Penalize rejected artists, styles, moods, and tracks.
3. Boost liked styles, moods, language, and popularity buckets.
4. Keep diversity constraints so one artist does not dominate.
5. Return the next best unseen candidate.

## Evaluation Upgrade

The current eval is a metadata consistency eval. It checks whether returned tracks match requested tags and avoid obvious violations. It is useful for regression testing, but it is not a listening-quality eval.

The feedback version should add:

- session-level metrics
- repeated-artist rate
- duplicate-track rate
- feedback acceptance rate
- average likes before first pass
- qualitative review logs for fixed discovery sessions

If preview audio becomes available, add listening-based feedback and measure like/pass after playback.
