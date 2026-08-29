export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
export const WS_BASE = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000";

export function assetUrl(url: string): string {
  if (/^https?:\/\//i.test(url)) return url;
  if (url.startsWith("/uploads/")) return `${API_BASE}${url}`;
  return url;
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const request = () => fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
    cache: "no-store",
  });

  let response: Response;
  try {
    response = await request();
  } catch (error) {
    // Windows demo machines can briefly race the backend during first paint.
    // Retrying read-only requests once avoids a stale error banner while never
    // duplicating POST/PUT side effects.
    const method = (init?.method ?? "GET").toUpperCase();
    if (method !== "GET") throw error;
    await new Promise((resolve) => setTimeout(resolve, 250));
    response = await request();
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const message = body?.detail?.[0]?.msg ?? body?.detail ?? "服务暂时不可用";
    throw new Error(String(message));
  }
  return response.json() as Promise<T>;
}
