const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const API_BASE_STORAGE_KEY = "polyedge.api_base";
const LOCAL_API_PORT_START = 8000;
const LOCAL_API_PORT_END = 8050;
const PROBE_TIMEOUT_MS = 450;

let resolvedApiBase: string | null = null;
let resolveApiBasePromise: Promise<string> | null = null;

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

async function probeApiBase(base: string): Promise<boolean> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);
  try {
    const response = await fetch(`${base}/health`, {
      method: "GET",
      signal: controller.signal,
    });
    return response.ok;
  } catch {
    return false;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

export async function getApiBase(options?: {
  forceReprobe?: boolean;
}): Promise<string> {
  const forceReprobe = options?.forceReprobe ?? false;
  if (!forceReprobe && resolvedApiBase) {
    return resolvedApiBase;
  }
  if (!forceReprobe && resolveApiBasePromise) {
    return resolveApiBasePromise;
  }

  const resolvePromise = (async () => {
    const candidates = buildApiBaseCandidates();
    for (const candidate of candidates) {
      if (await probeApiBase(candidate)) {
        resolvedApiBase = candidate;
        saveStoredApiBase(candidate);
        return candidate;
      }
    }

    // Preserve the configured base for error reporting if no live backend answers.
    resolvedApiBase = normalizeApiBase(DEFAULT_API_BASE);
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
  const base = await getApiBase();

  try {
    return await fetch(`${base}${normalizedPath}`, init);
  } catch (error) {
    const reprobedBase = await getApiBase({ forceReprobe: true });
    if (reprobedBase === base) {
      throw error;
    }
    return fetch(`${reprobedBase}${normalizedPath}`, init);
  }
}
