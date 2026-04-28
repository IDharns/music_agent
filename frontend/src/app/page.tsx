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

export default function Page() {
  const [query, setQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [data, setData] = useState<SearchResponse | null>(null);

  const hasResults = useMemo(() => {
    return !!data && Array.isArray(data.results) && data.results.length > 0;
  }, [data]);

  async function runSearch(nextQuery?: string) {
    const q = (nextQuery ?? query).trim();
    if (!q) return;

    setQuery("");
    setSubmittedQuery(q);
    setLoading(true);
    setError("");

    try {
      const result = await searchMusic(q, 10, 3, DEBUG_UI);
      setData(result);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "Unknown error");
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      void runSearch();
    }
  }

  return (
    <main className="min-h-screen bg-black text-white">
      <div className="mx-auto flex min-h-screen max-w-5xl flex-col px-4 py-5 sm:px-6">
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

        <section className="flex-1 space-y-6 overflow-y-auto py-6">
          <AssistantIntro />

          {submittedQuery ? <UserBubble query={submittedQuery} /> : null}

          {loading ? (
            <div className="flex gap-3">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-emerald-400 text-sm font-semibold text-black">
                M
              </div>
              <div className="rounded-lg border border-neutral-800 bg-neutral-950 px-4 py-3 text-sm text-neutral-300">
                Listening through the catalog...
              </div>
            </div>
          ) : null}

          {error ? (
            <div className="rounded-lg border border-red-900 bg-red-950 px-4 py-3 text-sm text-red-200">
              {error}
            </div>
          ) : null}

          {!loading && !hasResults && !error ? (
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

          {hasResults ? (
            <div className="flex gap-3">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-emerald-400 text-sm font-semibold text-black">
                M
              </div>
              <div className="min-w-0 flex-1 space-y-3">
                <div className="rounded-lg border border-neutral-800 bg-neutral-950 px-4 py-3 text-sm text-neutral-200">
                  I found {data?.result_count ?? 0} tracks that fit this direction.
                </div>
                {data!.results.map((item, index) => (
                  <ResultCard key={`${item.id}-${item.title}`} item={item} rank={index + 1} />
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
                  <p className="mt-1 text-neutral-200">{data?.query_type || "-"}</p>
                </div>
                <div>
                  <p className="text-neutral-500">semantic_query_used</p>
                  <p className="mt-1 break-words text-neutral-200">{data?.semantic_query_used || "-"}</p>
                </div>
              </div>
              <pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap break-words text-xs text-neutral-400">
                {JSON.stringify(data, null, 2)}
              </pre>
            </aside>
          ) : null}
        </section>

        <footer className="border-t border-neutral-900 pt-4">
          <div className="flex flex-col gap-3 sm:flex-row">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Try: dreamy indie pop female vocal"
              className="min-w-0 flex-1 rounded-lg border border-neutral-800 bg-neutral-950 px-4 py-3 text-white outline-none placeholder:text-neutral-600 focus:border-neutral-500"
            />
            <button
              onClick={() => void runSearch()}
              disabled={loading}
              className="rounded-lg bg-white px-5 py-3 font-medium text-black transition hover:bg-neutral-200 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? "Searching" : "Send"}
            </button>
          </div>
        </footer>
      </div>
    </main>
  );
}
