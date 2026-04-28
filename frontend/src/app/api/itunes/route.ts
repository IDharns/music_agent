function norm(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9 ]/g, "").trim();
}

function similarity(a: string, b: string): number {
  const na = norm(a);
  const nb = norm(b);
  if (na === nb) return 1;
  if (na.includes(nb) || nb.includes(na)) return 0.8;
  return 0;
}

type ItunesResult = {
  trackName?: string;
  artistName?: string;
  artworkUrl100?: string;
  previewUrl?: string;
  trackViewUrl?: string;
};

export async function GET(request: Request) {
  const incoming = new URL(request.url);
  const title = incoming.searchParams.get("title")?.trim() ?? "";
  const artist = incoming.searchParams.get("artist")?.trim() ?? "";

  if (!title || !artist) {
    return Response.json(null, { status: 200 });
  }

  const term = encodeURIComponent(`${title} ${artist}`);
  const url = `https://itunes.apple.com/search?term=${term}&entity=song&limit=5`;

  try {
    const response = await fetch(url, {
      method: "GET",
      cache: "no-store",
      signal: AbortSignal.timeout(10_000),
    });

    if (!response.ok) {
      return Response.json(null, { status: 200 });
    }

    const data = (await response.json()) as { results?: ItunesResult[] };
    const results = Array.isArray(data.results) ? data.results : [];

    let best: ItunesResult | null = null;
    let bestScore = -1;

    for (const result of results) {
      const trackScore = similarity(result.trackName ?? "", title);
      const artistScore = similarity(result.artistName ?? "", artist);
      const score = trackScore + artistScore;

      if (score > bestScore) {
        bestScore = score;
        best = result;
      }
    }

    if (!best) {
      return Response.json(null, { status: 200 });
    }

    const rawArt = best.artworkUrl100 ?? "";

    return Response.json(
      {
        artworkUrl: rawArt.replace("100x100bb", "500x500bb"),
        previewUrl: best.previewUrl ?? null,
        trackViewUrl: best.trackViewUrl ?? null,
      },
      {
        status: 200,
        headers: {
          "Cache-Control": "no-store",
        },
      },
    );
  } catch {
    return Response.json(null, { status: 200 });
  }
}
