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

export type SearchResultItem = {
    id: number | string;
    title: string;
    artist: string;
    album?: string | null;
    release_year?: number | null;
    popularity?: number | null;
    popularity_proxy?: number | null;
    popularity_bucket?: string | null;
    language?: string | null;
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
    score?: number | null;
    heuristic_score?: number | null;
    llm_score?: number | null;
    match_type?: string | null;
    reason?: string | null;
    match_evidence?: {
        match_type?: string | null;
        style_hits?: string[];
        mood_hits?: string[];
        vocal_match?: boolean;
        popularity_bucket?: string | null;
        fallback_used?: boolean;
        penalties?: string[];
    };
};

export type SearchResponse = {
    query: string;
    query_type?: string;
    fallback_used?: boolean;
    result_count?: number;
    parsed_query?: ParsedQuery;
    semantic_query_used?: string;
    results: SearchResultItem[];
};

const API_BASE =
    process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

export async function searchMusic(
    query: string,
    finalK = 10,
    maxPerArtist = 3,
    includeDebug = false
): Promise<SearchResponse> {
    const url = new URL("/search", API_BASE);
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
