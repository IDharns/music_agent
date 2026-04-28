"use client";

import Image from "next/image";
import { useEffect, useMemo, useRef, useState } from "react";
import { searchMusic, SearchResponse, SearchResultItem } from "@/lib/api";
import { fetchItunesMatch, ItunesMatch } from "@/lib/itunes";

const DEBUG_UI = process.env.NEXT_PUBLIC_DEBUG_UI === "1";

const EXAMPLE_QUERIES = [
  "sad female pop",
  "dreamy indie pop female vocal",
  "类似Taylor Swift但不要太热门，更梦幻一点",
];

const GENRE_OPTIONS = [
  "pop",
  "indie pop",
  "dream pop",
  "shoegaze",
  "rock",
  "indie",
  "electronic",
  "folk",
  "jazz",
  "rnb",
  "hip hop",
  "classical",
] as const;

const ERA_OPTIONS = [
  { value: "", label: "Any era" },
  { value: "1980s", label: "1980s" },
  { value: "1990s", label: "1990s" },
  { value: "2000s", label: "2000s" },
  { value: "2010s", label: "2010s" },
] as const;

const VOCAL_OPTIONS = [
  { value: "", label: "Any vocal" },
  { value: "female vocal", label: "Female vocal" },
  { value: "male vocal", label: "Male vocal" },
  { value: "instrumental", label: "Instrumental" },
] as const;

const POPULARITY_OPTIONS = [
  { value: "", label: "Any popularity" },
  { value: "less_popular", label: "Less popular" },
  { value: "more_popular", label: "More popular" },
] as const;

type SearchFilters = {
  artist: string;
  genre: string;
  era: string;
  vocal: string;
  popularity: string;
  language: string;
  resultCount: string;
  excludeLive: boolean;
  excludeRemix: boolean;
  excludeInstrumental: boolean;
};

type FilterKey = keyof SearchFilters;

const DEFAULT_FILTERS: SearchFilters = {
  artist: "",
  genre: "",
  era: "",
  vocal: "",
  popularity: "",
  language: "",
  resultCount: "10",
  excludeLive: false,
  excludeRemix: false,
  excludeInstrumental: false,
};

const FILTER_OPTIONS: Array<{ key: FilterKey; label: string }> = [
  { key: "artist", label: "Artist" },
  { key: "genre", label: "Genre" },
  { key: "era", label: "Era" },
  { key: "vocal", label: "Vocal" },
  { key: "popularity", label: "Popularity" },
  { key: "language", label: "Language" },
  { key: "resultCount", label: "Result count" },
  { key: "excludeLive", label: "Exclude live" },
  { key: "excludeRemix", label: "Exclude remix" },
  { key: "excludeInstrumental", label: "Exclude instrumental" },
];

function formatScore(score?: number | null) {
  if (score === null || score === undefined || Number.isNaN(score)) {
    return "-";
  }
  return score.toFixed(4);
}

function formatPopularity(bucket?: string | null) {
  if (bucket === "low") return "Less mainstream";
  if (bucket === "medium") return "Familiar";
  if (bucket === "high") return "Popular";
  return "Unknown reach";
}

function buildSearchQuery(query: string, filters: SearchFilters, activeFilters: FilterKey[]): string {
  const parts: string[] = [];
  const trimmedQuery = query.trim();
  const active = new Set(activeFilters);

  if (trimmedQuery) {
    parts.push(trimmedQuery);
  }
  if (active.has("artist") && filters.artist.trim()) {
    parts.push(`like ${filters.artist.trim()}`);
  }
  if (active.has("genre") && filters.genre) {
    parts.push(filters.genre);
  }
  if (active.has("era") && filters.era) {
    parts.push(filters.era);
  }
  if (active.has("vocal") && filters.vocal) {
    parts.push(filters.vocal);
  }
  if (active.has("popularity") && filters.popularity === "less_popular") {
    parts.push("not too popular");
  } else if (active.has("popularity") && filters.popularity === "more_popular") {
    parts.push("popular");
  }
  if (active.has("language") && filters.language.trim()) {
    parts.push(filters.language.trim());
  }
  if (active.has("excludeLive") && filters.excludeLive) {
    parts.push("not live");
  }
  if (active.has("excludeRemix") && filters.excludeRemix) {
    parts.push("not remix");
  }
  if (active.has("excludeInstrumental") && filters.excludeInstrumental) {
    parts.push("not instrumental");
  }

  return parts.join(" ").replace(/\s+/g, " ").trim();
}

function useItunesEnrich(title: string, artist: string): ItunesMatch | null {
  const key = `${title}||${artist}`;
  const [state, setState] = useState<{
    key: string;
    match: ItunesMatch | null;
  }>({ key: "", match: null });

  useEffect(() => {
    if (!title || !artist) return;

    let cancelled = false;

    fetchItunesMatch(title, artist).then((result) => {
      if (!cancelled) {
        setState({ key, match: result });
      }
    });

    return () => {
      cancelled = true;
    };
  }, [key, title, artist]);

  if (!title || !artist || state.key !== key) {
    return null;
  }

  return state.match;
}

function PreviewPlayer({ url }: { url: string }) {
  const [playing, setPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  function toggle() {
    const audio = audioRef.current;
    if (!audio) return;
    if (playing) {
      audio.pause();
      setPlaying(false);
    } else {
      audio.play();
      setPlaying(true);
    }
  }

  return (
    <div className="mt-3 flex items-center gap-2">
      <button
        onClick={toggle}
        className="flex items-center gap-1.5 rounded-lg border border-neutral-700 bg-neutral-900 px-3 py-1.5 text-xs text-neutral-300 transition hover:border-neutral-500 hover:text-white"
      >
        {playing ? (
          <>
            <span className="inline-block h-2.5 w-2.5">
              <svg viewBox="0 0 10 10" fill="currentColor"><rect x="1" y="0" width="3" height="10"/><rect x="6" y="0" width="3" height="10"/></svg>
            </span>
            Pause
          </>
        ) : (
          <>
            <span className="inline-block h-2.5 w-2.5">
              <svg viewBox="0 0 10 10" fill="currentColor"><polygon points="0,0 10,5 0,10"/></svg>
            </span>
            Preview
          </>
        )}
      </button>
      {playing && (
        <span className="text-xs text-neutral-500">30s sample</span>
      )}
      <audio
        ref={audioRef}
        src={url}
        onEnded={() => setPlaying(false)}
        preload="none"
      />
    </div>
  );
}

function ResultCard({ item, rank }: { item: SearchResultItem; rank: number }) {
  const itunes = useItunesEnrich(item.title ?? "", item.artist ?? "");

  return (
    <article className="rounded-lg border border-neutral-800 bg-neutral-950 p-4">
      <div className="flex items-start gap-3">
        {/* Album art or rank badge */}
        {itunes?.artworkUrl ? (
          <Image
            src={itunes.artworkUrl}
            alt={`${item.album ?? item.title} cover`}
            className="h-14 w-14 shrink-0 rounded-lg object-cover"
            width={56}
            height={56}
          />
        ) : (
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-white text-sm font-semibold text-black">
            {rank}
          </div>
        )}

        <div className="min-w-0 flex-1">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
            <div className="min-w-0">
              <h3 className="break-words text-base font-semibold text-white">
                {item.title || "Untitled"}
              </h3>
              <p className="mt-1 break-words text-sm text-neutral-300">
                {item.artist || "Unknown Artist"}
              </p>
            </div>

            <div className="shrink-0 rounded-lg border border-neutral-700 px-2.5 py-1 text-xs text-neutral-300">
              {formatPopularity(item.popularity_bucket)}
            </div>
          </div>

          <div className="mt-3 flex flex-wrap gap-2 text-xs text-neutral-400">
            {item.album ? (
              <span className="rounded-lg bg-neutral-900 px-2.5 py-1">{item.album}</span>
            ) : null}
            {item.release_year ? (
              <span className="rounded-lg bg-neutral-900 px-2.5 py-1">{item.release_year}</span>
            ) : null}
            {item.language ? (
              <span className="rounded-lg bg-neutral-900 px-2.5 py-1">{item.language}</span>
            ) : null}
          </div>

          {itunes?.previewUrl ? (
            <PreviewPlayer url={itunes.previewUrl} />
          ) : null}

          {item.similarity != null ? (
            <div className="mt-3 flex flex-wrap gap-3 text-xs text-neutral-500 font-mono">
              <span>sim <span className="text-neutral-300">{item.similarity.toFixed(4)}</span></span>
              <span>tag overlap <span className="text-neutral-300">{item.tag_overlap != null ? item.tag_overlap.toFixed(4) : "0.0000"}</span></span>
              <span>final <span className="text-neutral-300">{item.score != null ? item.score.toFixed(4) : "-"}</span></span>
            </div>
          ) : null}

          {item.reason ? (
            <p className="mt-4 text-sm leading-6 text-neutral-200">{item.reason}</p>
          ) : null}

          {DEBUG_UI ? (
            <div className="mt-4 rounded-lg border border-neutral-800 bg-black p-3 text-xs text-neutral-400">
              <div className="grid gap-1 sm:grid-cols-2">
                <p>score: {formatScore(item.score)}</p>
                <p>match: {item.match_type || "-"}</p>
                <p>vocal: {item.vocal_type || "-"}</p>
                <p>bucket: {item.popularity_bucket || "-"}</p>
              </div>
              <p className="mt-2">styles: {(item.style_tags || []).join(", ") || "-"}</p>
              <p className="mt-1">moods: {(item.mood_anchors || []).join(", ") || "-"}</p>
              <pre className="mt-2 max-h-32 overflow-auto whitespace-pre-wrap break-words">
                {JSON.stringify(item.match_evidence ?? null, null, 2)}
              </pre>
            </div>
          ) : null}
        </div>
      </div>
    </article>
  );
}

function AssistantIntro() {
  return (
    <div className="flex gap-3">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-emerald-400 text-sm font-semibold text-black">
        M
      </div>
      <div className="max-w-2xl rounded-lg border border-neutral-800 bg-neutral-950 px-4 py-3 text-sm leading-6 text-neutral-200">
        Tell me an artist, a mood, or a sound you want to move toward. I will return a short set of tracks with a listener-facing reason for each pick.
      </div>
    </div>
  );
}

function UserBubble({ query }: { query: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-2xl rounded-lg bg-white px-4 py-3 text-sm leading-6 text-black">
        {query}
      </div>
    </div>
  );
}

type SearchTurn = {
  id: string;
  query: string;
  loading: boolean;
  error: string;
  data: SearchResponse | null;
};

function SearchTurnView({ turn }: { turn: SearchTurn }) {
  const hasResults = !!turn.data && Array.isArray(turn.data.results) && turn.data.results.length > 0;

  return (
    <div className="space-y-4">
      <UserBubble query={turn.query} />

      {turn.loading ? (
        <div className="flex gap-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-emerald-400 text-sm font-semibold text-black">
            M
          </div>
          <div className="rounded-lg border border-neutral-800 bg-neutral-950 px-4 py-3 text-sm text-neutral-300">
            Listening through the catalog...
          </div>
        </div>
      ) : null}

      {turn.error ? (
        <div className="rounded-lg border border-red-900 bg-red-950 px-4 py-3 text-sm text-red-200">
          {turn.error}
        </div>
      ) : null}

      {hasResults ? (
        <div className="flex gap-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-emerald-400 text-sm font-semibold text-black">
            M
          </div>
          <div className="min-w-0 flex-1 space-y-3">
            <div className="rounded-lg border border-neutral-800 bg-neutral-950 px-4 py-3 text-sm text-neutral-200">
              I found {turn.data?.result_count ?? 0} tracks that fit this direction.
            </div>
            {turn.data!.results.map((item, index) => (
              <ResultCard key={`${turn.id}-${item.id}-${item.title}`} item={item} rank={index + 1} />
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default function Page() {
  const [query, setQuery] = useState("");
  const [filters, setFilters] = useState<SearchFilters>(DEFAULT_FILTERS);
  const [activeFilterKeys, setActiveFilterKeys] = useState<FilterKey[]>([]);
  const [pendingFilterKey, setPendingFilterKey] = useState("");
  const [turns, setTurns] = useState<SearchTurn[]>([]);
  const nextTurnId = useRef(1);
  const latestData = useMemo(() => {
    for (let i = turns.length - 1; i >= 0; i -= 1) {
      if (turns[i].data) return turns[i].data;
    }
    return null;
  }, [turns]);
  const isLoading = useMemo(() => turns.some((turn) => turn.loading), [turns]);
  const availableFilterOptions = useMemo(
    () => FILTER_OPTIONS.filter((option) => !activeFilterKeys.includes(option.key)),
    [activeFilterKeys]
  );

  async function runSearch(nextQuery?: string) {
    const baseQuery = (nextQuery ?? query).trim();
    const composedQuery = buildSearchQuery(baseQuery, filters, activeFilterKeys);
    if (!composedQuery) return;

    const turnId = String(nextTurnId.current++);
    setQuery("");
    setTurns((current) => [
      ...current,
      {
        id: turnId,
        query: composedQuery,
        loading: true,
        error: "",
        data: null,
      },
    ]);

    try {
      const requestedCount = activeFilterKeys.includes("resultCount")
        ? Number.parseInt(filters.resultCount, 10)
        : 10;
      const finalK = Number.isFinite(requestedCount) ? requestedCount : 10;
      const result = await searchMusic(composedQuery, finalK, 3, DEBUG_UI);
      setTurns((current) =>
        current.map((turn) =>
          turn.id === turnId
            ? { ...turn, loading: false, error: "", data: result }
            : turn
        )
      );
    } catch (err) {
      console.error(err);
      const message = err instanceof Error ? err.message : "Unknown error";
      setTurns((current) =>
        current.map((turn) =>
          turn.id === turnId
            ? { ...turn, loading: false, error: message, data: null }
            : turn
        )
      );
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      void runSearch();
    }
  }

  function addFilter(key: string) {
    if (!key) return;
    if (!FILTER_OPTIONS.some((option) => option.key === key)) return;
    setActiveFilterKeys((current) => (
      current.includes(key as FilterKey) ? current : [...current, key as FilterKey]
    ));
    setPendingFilterKey("");
  }

  function removeFilter(key: FilterKey) {
    setActiveFilterKeys((current) => current.filter((item) => item !== key));
    setFilters((current) => ({ ...current, [key]: DEFAULT_FILTERS[key] }));
  }

  return (
    <main className="h-screen overflow-hidden bg-black text-white">
      <div className="mx-auto flex h-screen max-w-5xl flex-col px-4 py-5 sm:px-6">
        <header className="flex items-center justify-between border-b border-neutral-900 pb-4">
          <div>
            <p className="text-xs uppercase tracking-[0.18em] text-neutral-500">Music Agent</p>
            <h1 className="mt-1 text-xl font-semibold">Ask for a sound</h1>
          </div>
          {DEBUG_UI ? (
            <div className="rounded-lg border border-amber-700 px-3 py-1 text-xs text-amber-300">
              Debug on
            </div>
          ) : null}
        </header>

        <section className="min-h-0 flex-1 space-y-6 overflow-y-auto py-6">
          <AssistantIntro />

          {turns.map((turn) => (
            <SearchTurnView key={turn.id} turn={turn} />
          ))}

          {!turns.length ? (
            <div className="space-y-3">
              <p className="text-sm text-neutral-500">Try one of these:</p>
              <div className="flex flex-wrap gap-2">
                {EXAMPLE_QUERIES.map((example) => (
                  <button
                    key={example}
                    onClick={() => void runSearch(example)}
                    className="rounded-lg border border-neutral-800 bg-neutral-950 px-3 py-2 text-sm text-neutral-300 transition hover:border-neutral-600 hover:text-white"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          {DEBUG_UI ? (
            <aside className="rounded-lg border border-neutral-800 bg-neutral-950 p-4">
              <h2 className="text-sm font-semibold text-neutral-200">Debug</h2>
              <div className="mt-3 grid gap-3 text-xs text-neutral-400 lg:grid-cols-2">
                <div>
                  <p className="text-neutral-500">query_type</p>
                  <p className="mt-1 text-neutral-200">{latestData?.query_type || "-"}</p>
                </div>
                <div>
                  <p className="text-neutral-500">semantic_query_used</p>
                  <p className="mt-1 break-words text-neutral-200">{latestData?.semantic_query_used || "-"}</p>
                </div>
              </div>
              <pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap break-words text-xs text-neutral-400">
                {JSON.stringify(latestData, null, 2)}
              </pre>
            </aside>
          ) : null}
        </section>

        <footer className="-mx-4 sticky bottom-0 z-10 border-t border-neutral-900 bg-black/95 px-4 pt-4 pb-4 backdrop-blur sm:-mx-6 sm:px-6">
          <div className="space-y-3">
            <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_220px_auto]">
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Try: dreamy indie pop female vocal"
                className="min-w-0 rounded-lg border border-neutral-800 bg-neutral-950 px-4 py-3 text-white outline-none placeholder:text-neutral-600 focus:border-neutral-500"
              />
              <div className="flex gap-3">
                <select
                  value={pendingFilterKey}
                  onChange={(e) => addFilter(e.target.value)}
                  className="min-w-0 flex-1 rounded-lg border border-neutral-800 bg-neutral-950 px-3 py-3 text-sm text-white outline-none focus:border-neutral-500"
                >
                  <option value="">Add filter</option>
                  {availableFilterOptions.map((option) => (
                    <option key={option.key} value={option.key}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={() => {
                    setFilters(DEFAULT_FILTERS);
                    setActiveFilterKeys([]);
                    setPendingFilterKey("");
                  }}
                  className="rounded-lg border border-neutral-800 bg-neutral-950 px-4 py-3 text-sm text-neutral-300 transition hover:border-neutral-600 hover:text-white"
                >
                  Reset
                </button>
              </div>
              <button
                onClick={() => void runSearch()}
                disabled={isLoading}
                className="rounded-lg bg-white px-5 py-3 font-medium text-black transition hover:bg-neutral-200 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isLoading ? "Searching" : "Send"}
              </button>
            </div>

            {activeFilterKeys.length ? (
              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                {activeFilterKeys.map((filterKey) => (
                  <div
                    key={filterKey}
                    className="rounded-lg border border-neutral-800 bg-neutral-950 p-3"
                  >
                    <div className="mb-2 flex items-center justify-between gap-3">
                      <span className="text-xs font-medium uppercase tracking-[0.14em] text-neutral-500">
                        {FILTER_OPTIONS.find((option) => option.key === filterKey)?.label}
                      </span>
                      <button
                        type="button"
                        onClick={() => removeFilter(filterKey)}
                        className="text-xs text-neutral-500 transition hover:text-white"
                      >
                        Remove
                      </button>
                    </div>

                    {filterKey === "artist" ? (
                      <input
                        value={filters.artist}
                        onChange={(e) => setFilters((current) => ({ ...current, artist: e.target.value }))}
                        placeholder="Artist"
                        className="w-full min-w-0 rounded-lg border border-neutral-800 bg-black px-3 py-2.5 text-sm text-white outline-none placeholder:text-neutral-600 focus:border-neutral-500"
                      />
                    ) : null}

                    {filterKey === "genre" ? (
                      <select
                        value={filters.genre}
                        onChange={(e) => setFilters((current) => ({ ...current, genre: e.target.value }))}
                        className="w-full min-w-0 rounded-lg border border-neutral-800 bg-black px-3 py-2.5 text-sm text-white outline-none focus:border-neutral-500"
                      >
                        <option value="">Any genre</option>
                        {GENRE_OPTIONS.map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    ) : null}

                    {filterKey === "era" ? (
                      <select
                        value={filters.era}
                        onChange={(e) => setFilters((current) => ({ ...current, era: e.target.value }))}
                        className="w-full min-w-0 rounded-lg border border-neutral-800 bg-black px-3 py-2.5 text-sm text-white outline-none focus:border-neutral-500"
                      >
                        {ERA_OPTIONS.map((option) => (
                          <option key={option.label} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    ) : null}

                    {filterKey === "vocal" ? (
                      <select
                        value={filters.vocal}
                        onChange={(e) => setFilters((current) => ({ ...current, vocal: e.target.value }))}
                        className="w-full min-w-0 rounded-lg border border-neutral-800 bg-black px-3 py-2.5 text-sm text-white outline-none focus:border-neutral-500"
                      >
                        {VOCAL_OPTIONS.map((option) => (
                          <option key={option.label} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    ) : null}

                    {filterKey === "popularity" ? (
                      <select
                        value={filters.popularity}
                        onChange={(e) => setFilters((current) => ({ ...current, popularity: e.target.value }))}
                        className="w-full min-w-0 rounded-lg border border-neutral-800 bg-black px-3 py-2.5 text-sm text-white outline-none focus:border-neutral-500"
                      >
                        {POPULARITY_OPTIONS.map((option) => (
                          <option key={option.label} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    ) : null}

                    {filterKey === "language" ? (
                      <input
                        value={filters.language}
                        onChange={(e) => setFilters((current) => ({ ...current, language: e.target.value }))}
                        placeholder="Language"
                        className="w-full min-w-0 rounded-lg border border-neutral-800 bg-black px-3 py-2.5 text-sm text-white outline-none placeholder:text-neutral-600 focus:border-neutral-500"
                      />
                    ) : null}

                    {filterKey === "resultCount" ? (
                      <select
                        value={filters.resultCount}
                        onChange={(e) => setFilters((current) => ({ ...current, resultCount: e.target.value }))}
                        className="w-full min-w-0 rounded-lg border border-neutral-800 bg-black px-3 py-2.5 text-sm text-white outline-none focus:border-neutral-500"
                      >
                        {[3, 5, 10, 15, 20].map((count) => (
                          <option key={count} value={String(count)}>
                            {count} results
                          </option>
                        ))}
                      </select>
                    ) : null}

                    {filterKey === "excludeLive" ? (
                      <label className="flex items-center gap-2 text-sm text-neutral-300">
                        <input
                          type="checkbox"
                          checked={filters.excludeLive}
                          onChange={(e) => setFilters((current) => ({ ...current, excludeLive: e.target.checked }))}
                        />
                        Exclude live versions
                      </label>
                    ) : null}

                    {filterKey === "excludeRemix" ? (
                      <label className="flex items-center gap-2 text-sm text-neutral-300">
                        <input
                          type="checkbox"
                          checked={filters.excludeRemix}
                          onChange={(e) => setFilters((current) => ({ ...current, excludeRemix: e.target.checked }))}
                        />
                        Exclude remixes
                      </label>
                    ) : null}

                    {filterKey === "excludeInstrumental" ? (
                      <label className="flex items-center gap-2 text-sm text-neutral-300">
                        <input
                          type="checkbox"
                          checked={filters.excludeInstrumental}
                          onChange={(e) => setFilters((current) => ({ ...current, excludeInstrumental: e.target.checked }))}
                        />
                        Exclude instrumental
                      </label>
                    ) : null}
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        </footer>
      </div>
    </main>
  );
}
