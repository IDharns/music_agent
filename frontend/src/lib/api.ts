export type ParsedQuery = {
    query_type?: string;
    artist_seeds?: string[];
    genres?: string[];
    moods?: string[];
    vocal?: string;
    energy?: string;
    era?: string | number | null;
    popularity_preference?: string | null;
    include?: string[];
    exclude?: string[];
    raw_query?: string;
};

/**
 * Canonical track shape returned by /search.
 *
 * Base fields are always present (score and match_type are always emitted,
 * but may be null if the pipeline had no value).
 *
 * Debug-only fields (populated when include_debug=true) are marked optional.
 *
 * Mirrors app.models.TrackResponse on the Python side.
 */
export type SearchResultItem = {
    // --- Always present ---
    id: number | string;
    title: string;
    artist: string;
    album: string | null;
    release_year: number | null;
    popularity_bucket: string | null;
    language: string | null;
    score: number | null;
    similarity: number | null;
    tag_overlap: number | null;
    match_type: string | null;
    reason: string | null;

    // --- Debug-only (include_debug=true) ---
    popularity?: number | null;
    popularity_proxy?: number | null;
    vocal_type?: string | null;
    genre_text?: string | null;
    style_text?: string | null;
    mood_text?: string | null;
    primary_artists?: string[];
    featured_artists?: string[];
    all_contributors?: string[];
    style_tags?: string[];
    mood_anchors?: string[];
    artist_tags?: string[];
    album_tags?: string[];
    mood_confidence?: number | null;
    heuristic_score?: number | null;
    llm_score?: number | null;
    match_evidence?: {
        match_type?: string | null;
        style_hits?: string[];
        mood_hits?: string[];
        vocal_match?: boolean;
        popularity_bucket?: string | null;
        fallback_used?: boolean;
        penalties?: string[];
    } | null;
};

export type SearchResponse = {
    query: string;
    query_type?: string;
    fallback_used?: boolean;
    result_count?: number;
    // Debug-only envelope fields
    parsed_query?: ParsedQuery;
    semantic_query_base?: string | null;
    semantic_query_llm?: string | null;
    semantic_query_used?: string | null;
    llm_rewrite?: Record<string, unknown> | null;
    llm_rank_debug?: unknown[];
    results: SearchResultItem[];
};

export async function searchMusic(
    query: string,
    finalK = 10,
    maxPerArtist = 3,
    includeDebug = false
): Promise<SearchResponse> {
    const url = new URL("/api/search", window.location.origin);
    url.searchParams.set("query", query);
    url.searchParams.set("final_k", String(finalK));
    url.searchParams.set("max_per_artist", String(maxPerArtist));
    if (includeDebug) {
        url.searchParams.set("include_debug", "true");
    }

    const res = await fetch(url.toString(), {
        method: "GET",
        cache: "no-store",
    });

    if (!res.ok) {
        const text = await res.text();
        throw new Error(`API ${res.status}: ${text}`);
    }

    return res.json();
}
