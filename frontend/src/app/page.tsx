"use client";

import { useMemo, useState } from "react";
import { searchMusic, SearchResponse, SearchResultItem } from "@/lib/api";

function formatScore(score?: number | null) {
  if (score === null || score === undefined || Number.isNaN(score)) {
    return "-";
  }
  return score.toFixed(4);
}

function ResultCard({ item }: { item: SearchResultItem }) {
  return (
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-5 shadow-sm">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold text-white">{item.title || "Untitled"}</h3>
            <p className="mt-1 text-sm text-zinc-300">{item.artist || "Unknown Artist"}</p>
          </div>
          <div className="rounded-full border border-zinc-700 px-3 py-1 text-xs text-zinc-300">
            score: {formatScore(item.score)}
          </div>
        </div>

        <div className="mt-4 grid gap-2 text-sm text-zinc-400 sm:grid-cols-2">
          <p><span className="text-zinc-500">Album:</span> {item.album || "-"}</p>
          <p><span className="text-zinc-500">Year:</span> {item.release_year ?? "-"}</p>
          <p><span className="text-zinc-500">Vocal:</span> {item.vocal_type || "-"}</p>
          <p><span className="text-zinc-500">Genre:</span> {item.genre_text || "-"}</p>
          <p><span className="text-zinc-500">Language:</span> {item.language || "-"}</p>
          <p><span className="text-zinc-500">Match:</span> {item.match_type || "-"}</p>
        </div>

        {item.reason ? (
            <div className="mt-4 rounded-xl bg-zinc-950 p-3 text-sm text-zinc-300">
              {item.reason}
            </div>
        ) : null}
      </div>
  );
}

export default function Page() {
  const [query, setQuery] = useState("类似Taylor Swift但不要太热门，更梦幻一点");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [data, setData] = useState<SearchResponse | null>(null);

  const hasResults = useMemo(() => {
    return !!data && Array.isArray(data.results) && data.results.length > 0;
  }, [data]);

  async function handleSearch() {
    const q = query.trim();
    if (!q) return;

    setLoading(true);
    setError("");

    try {
      const result = await searchMusic(q, 10, 3);
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
      void handleSearch();
    }
  }

  return (
      <main className="min-h-screen bg-black text-white">
        <div className="mx-auto max-w-6xl px-6 py-10">
          <div className="mb-8">
            <p className="text-sm uppercase tracking-[0.2em] text-zinc-500">
              Music Agent
            </p>
            <h1 className="mt-2 text-4xl font-bold">Semantic Music Search</h1>
            <p className="mt-3 max-w-3xl text-zinc-400">
              输入自然语言描述，前端调用 FastAPI 的 /search 接口，展示推荐结果与调试信息。
            </p>
          </div>

          <div className="rounded-3xl border border-zinc-800 bg-zinc-950 p-5">
            <div className="flex flex-col gap-3 sm:flex-row">
              <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="例如：sad female pop / Adele / 类似Taylor Swift但更梦幻一点"
                  className="flex-1 rounded-2xl border border-zinc-700 bg-zinc-900 px-4 py-3 text-white outline-none ring-0 placeholder:text-zinc-500 focus:border-zinc-500"
              />
              <button
                  onClick={() => void handleSearch()}
                  disabled={loading}
                  className="rounded-2xl bg-white px-5 py-3 font-medium text-black transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? "Searching..." : "Search"}
              </button>
            </div>

            {error ? (
                <div className="mt-4 rounded-xl border border-red-900 bg-red-950 px-4 py-3 text-sm text-red-300">
                  {error}
                </div>
            ) : null}
          </div>

          <div className="mt-8 grid gap-8 lg:grid-cols-[2fr_1fr]">
            <section className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold">Results</h2>
                <span className="text-sm text-zinc-500">
                {data?.result_count ?? 0} items
              </span>
              </div>

              {!loading && !hasResults ? (
                  <div className="rounded-2xl border border-dashed border-zinc-800 bg-zinc-950 p-8 text-zinc-500">
                    还没有结果。先搜一条 query。
                  </div>
              ) : null}

              {hasResults
                  ? data!.results.map((item) => (
                      <ResultCard key={`${item.id}-${item.title}`} item={item} />
                  ))
                  : null}
            </section>

            <aside className="space-y-4">
              <h2 className="text-xl font-semibold">Debug</h2>

              <div className="rounded-2xl border border-zinc-800 bg-zinc-950 p-4">
                <p className="text-xs uppercase tracking-wide text-zinc-500">query_type</p>
                <p className="mt-2 text-sm text-zinc-200">{data?.query_type || "-"}</p>
              </div>

              <div className="rounded-2xl border border-zinc-800 bg-zinc-950 p-4">
                <p className="text-xs uppercase tracking-wide text-zinc-500">semantic_query_used</p>
                <pre className="mt-2 whitespace-pre-wrap break-words text-sm text-zinc-200">
                {data?.semantic_query_used || "-"}
              </pre>
              </div>

              <div className="rounded-2xl border border-zinc-800 bg-zinc-950 p-4">
                <p className="text-xs uppercase tracking-wide text-zinc-500">parsed_query</p>
                <pre className="mt-2 max-h-[420px] overflow-auto whitespace-pre-wrap break-words text-xs text-zinc-300">
                {JSON.stringify(data?.parsed_query ?? null, null, 2)}
              </pre>
              </div>

              <div className="rounded-2xl border border-zinc-800 bg-zinc-950 p-4">
                <p className="text-xs uppercase tracking-wide text-zinc-500">raw_response</p>
                <pre className="mt-2 max-h-[320px] overflow-auto whitespace-pre-wrap break-words text-xs text-zinc-300">
                {JSON.stringify(data, null, 2)}
              </pre>
              </div>
            </aside>
          </div>
        </div>
      </main>
  );
}