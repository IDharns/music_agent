const BACKEND_API_BASE =
  process.env.BACKEND_API_BASE ||
  process.env.NEXT_PUBLIC_API_BASE ||
  "http://127.0.0.1:8000";

export async function GET(request: Request) {
  const incoming = new URL(request.url);
  const backendUrl = new URL("/search", BACKEND_API_BASE);

  for (const [key, value] of incoming.searchParams.entries()) {
    backendUrl.searchParams.append(key, value);
  }

  try {
    const response = await fetch(backendUrl.toString(), {
      method: "GET",
      cache: "no-store",
      signal: AbortSignal.timeout(180_000),
    });
    const body = await response.text();

    return new Response(body, {
      status: response.status,
      headers: {
        "Content-Type": response.headers.get("Content-Type") || "application/json",
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown backend error";
    return Response.json(
      { error: `Backend request failed: ${message}` },
      { status: 502 },
    );
  }
}
