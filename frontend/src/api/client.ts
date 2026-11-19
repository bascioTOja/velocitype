import type { ProblemDetail } from "./types";

// Same-origin by default: Caddy serves the SPA and proxies /api. In `npm run
// dev`, Vite proxies /api to the backend, so this stays relative too.
const BASE = import.meta.env.VITE_API_BASE ?? "";

export class ApiError extends Error {
  status: number;
  problem: ProblemDetail | null;

  constructor(status: number, message: string, problem: ProblemDetail | null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.problem = problem;
  }
}

// Auth endpoints where a 401 means "bad credentials / no session" — never try
// to refresh-and-retry these (and /refresh itself would recurse).
const NO_REFRESH = new Set([
  "/api/auth/login",
  "/api/auth/register",
  "/api/auth/refresh",
  "/api/auth/logout",
]);

// The access cookie (tf_access) lives ~15 min; the refresh cookie much longer.
// On a 401 we rotate via POST /api/auth/refresh and replay the request once.
// Single-flight: the backend revokes the presented refresh token on rotation,
// and a second concurrent refresh with that (now revoked) token gets a 401 that
// *clears* the fresh cookies — so all callers share one in-flight refresh.
let refreshing: Promise<boolean> | null = null;

function refreshSession(): Promise<boolean> {
  if (!refreshing) {
    refreshing = fetch(`${BASE}/api/auth/refresh`, {
      method: "POST",
      credentials: "include",
    })
      .then((r) => r.ok)
      .catch(() => false)
      .finally(() => {
        refreshing = null;
      });
  }
  return refreshing;
}

let onSessionExpired: (() => void) | null = null;

/** Called once a refresh fails (refresh token missing/expired/revoked). */
export function setOnSessionExpired(cb: (() => void) | null): void {
  onSessionExpired = cb;
}

function send(method: string, path: string, body: string | undefined): Promise<Response> {
  return fetch(`${BASE}${path}`, {
    method,
    credentials: "include", // send/receive httpOnly auth cookies
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body,
  });
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const encoded = body !== undefined ? JSON.stringify(body) : undefined;
  let res = await send(method, path, encoded);

  if (res.status === 401 && !NO_REFRESH.has(path.split("?")[0])) {
    if (await refreshSession()) {
      res = await send(method, path, encoded);
    } else {
      onSessionExpired?.();
    }
  }

  if (res.status === 204) {
    return undefined as T;
  }

  const isJson = res.headers.get("content-type")?.includes("json");
  const payload = isJson ? await res.json().catch(() => null) : null;

  if (!res.ok) {
    const problem = (payload as ProblemDetail) ?? null;
    const message =
      problem?.detail || problem?.title || `Request failed (${res.status})`;
    throw new ApiError(res.status, message, problem);
  }

  return payload as T;
}

export const api = {
  get: <T>(path: string) => request<T>("GET", path),
  post: <T>(path: string, body?: unknown) => request<T>("POST", path, body),
  put: <T>(path: string, body?: unknown) => request<T>("PUT", path, body),
  patch: <T>(path: string, body?: unknown) => request<T>("PATCH", path, body),
  del: <T>(path: string, body?: unknown) => request<T>("DELETE", path, body),
};
