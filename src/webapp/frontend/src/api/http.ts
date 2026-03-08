const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const API_BASE_STORAGE_KEY = "polyedge.api_base";
const LOCAL_API_PORT_START = 8000;
const LOCAL_API_PORT_END = 8050;
const PROBE_TIMEOUT_MS = 450;
const ANALYSIS_ROUTE_PROBE_PATH = "/analysis/overview";

let resolvedApiBase: string | null = null;
let resolveApiBasePromise: Promise<string> | null = null;
let resolvedProbePath = "/health";

function normalizeApiBase(value: string): string {
  return value.trim().replace(/\/+$/, "");
}

function loadStoredApiBase(): string | null {
  if (typeof window === "undefined") return null;
  try {
    const stored = window.localStorage.getItem(API_BASE_STORAGE_KEY);
    return stored ? normalizeApiBase(stored) : null;
  } catch {
    return null;
  }
}

function saveStoredApiBase(value: string): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(API_BASE_STORAGE_KEY, value);
  } catch {
    // Ignore storage failures.
  }
}

function buildApiBaseCandidates(): string[] {
  const configured = normalizeApiBase(DEFAULT_API_BASE);
  const candidates: string[] = [];
  const seen = new Set<string>();

  const pushCandidate = (value: string | null | undefined) => {
    if (!value) return;
    const normalized = normalizeApiBase(value);
    if (!normalized || seen.has(normalized)) return;
    seen.add(normalized);
    candidates.push(normalized);
  };

  pushCandidate(loadStoredApiBase());
  pushCandidate(configured);

  if (typeof window !== "undefined") {
    const { protocol, hostname } = window.location;
    if (hostname) {
      for (let port = LOCAL_API_PORT_START; port <= LOCAL_API_PORT_END; port += 1) {
        pushCandidate(`${protocol}//${hostname}:${port}`);
      }
    }
  }

  for (let port = LOCAL_API_PORT_START; port <= LOCAL_API_PORT_END; port += 1) {
    pushCandidate(`http://localhost:${port}`);
    pushCandidate(`http://127.0.0.1:${port}`);
  }

  return candidates;
}

function shouldUseAnalysisProbe(path: string): boolean {
  return path.startsWith("/analysis");
}

async function probeApiBase(base: string, probePath = "/health"): Promise<boolean> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);
  try {
    const response = await fetch(`${base}${probePath}`, {
      method: "GET",
      signal: controller.signal,
    });
    if (probePath === "/health") {
      return response.ok;
    }
    // For feature-specific probing we only need a backend that knows the route.
    // A 500 still means we found the right app, whereas a 404 means the route is absent.
    return response.status !== 404;
  } catch {
    return false;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

export async function getApiBase(options?: {
  forceReprobe?: boolean;
  probePath?: string;
}): Promise<string> {
  const forceReprobe = options?.forceReprobe ?? false;
  const probePath = options?.probePath ?? "/health";
  if (!forceReprobe && resolvedApiBase && resolvedProbePath === probePath) {
    return resolvedApiBase;
  }
  if (!forceReprobe && resolveApiBasePromise && resolvedProbePath === probePath) {
    return resolveApiBasePromise;
  }

  const resolvePromise = (async () => {
    const candidates = buildApiBaseCandidates();
    for (const candidate of candidates) {
      if (await probeApiBase(candidate, probePath)) {
        resolvedApiBase = candidate;
        resolvedProbePath = probePath;
        saveStoredApiBase(candidate);
        return candidate;
      }
    }

    // Preserve the configured base for error reporting if no live backend answers.
    resolvedApiBase = normalizeApiBase(DEFAULT_API_BASE);
    resolvedProbePath = probePath;
    return resolvedApiBase;
  })();

  resolveApiBasePromise = resolvePromise;
  try {
    return await resolvePromise;
  } finally {
    if (resolveApiBasePromise === resolvePromise) {
      resolveApiBasePromise = null;
    }
  }
}

export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  const probePath = shouldUseAnalysisProbe(normalizedPath) ? ANALYSIS_ROUTE_PROBE_PATH : "/health";
  const base = await getApiBase({ probePath });

  try {
    const response = await fetch(`${base}${normalizedPath}`, init);
    if (response.status !== 404 || !shouldUseAnalysisProbe(normalizedPath)) {
      return response;
    }

    const reprobedBase = await getApiBase({ forceReprobe: true, probePath });
    if (reprobedBase === base) {
      return response;
    }
    return fetch(`${reprobedBase}${normalizedPath}`, init);
  } catch (error) {
    const reprobedBase = await getApiBase({ forceReprobe: true, probePath });
    if (reprobedBase === base) {
      throw error;
    }
    return fetch(`${reprobedBase}${normalizedPath}`, init);
  }
}
