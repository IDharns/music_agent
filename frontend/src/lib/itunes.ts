export type ItunesMatch = {
  artworkUrl: string;      // 500x500
  previewUrl: string | null;
  trackViewUrl: string | null;
};

// Module-level in-memory cache: "title||artist" -> result or null (null = no match found)
const cache = new Map<string, ItunesMatch | null>();

function cacheKey(title: string, artist: string): string {
  return `${title.toLowerCase()}||${artist.toLowerCase()}`;
}

export async function fetchItunesMatch(
  title: string,
  artist: string
): Promise<ItunesMatch | null> {
  const key = cacheKey(title, artist);
  if (cache.has(key)) return cache.get(key)!;

  const url = new URL("/api/itunes", window.location.origin);
  url.searchParams.set("title", title);
  url.searchParams.set("artist", artist);

  try {
    const res = await fetch(url.toString(), { cache: "no-store" });
    if (!res.ok) {
      cache.set(key, null);
      return null;
    }
    const data = (await res.json()) as Partial<ItunesMatch> | null;
    if (!data?.artworkUrl && !data?.previewUrl && !data?.trackViewUrl) {
      cache.set(key, null);
      return null;
    }

    const match: ItunesMatch = {
      artworkUrl: data.artworkUrl ?? "",
      previewUrl: data.previewUrl ?? null,
      trackViewUrl: data.trackViewUrl ?? null,
    };

    cache.set(key, match);
    return match;
  } catch {
    cache.set(key, null);
    return null;
  }
}
