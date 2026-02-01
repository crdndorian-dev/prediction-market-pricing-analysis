export type PolymarketSnapshotRunRequest = {
  tickers?: string[];
  tickersCsv?: string;
  slugOverrides?: string;
  riskFreeRate?: number;
  tz?: string;
  keepNonexec?: boolean;
};

export type PolymarketSnapshotRunResponse = {
  ok: boolean;
  run_id: string | null;
  out_dir: string;
  run_dir: string | null;
  files: string[];
  stdout: string;
  stderr: string;
  duration_s: number;
  command: string[];
};

export type PolymarketSnapshotRunSummary = {
  run_id: string;
  run_time_utc: string | null;
  run_dir: string;
  files: string[];
  file_count: number;
  size_bytes: number;
  last_modified: string | null;
};

export type PolymarketSnapshotFileSummary = {
  name: string;
  path: string;
  size_bytes: number;
  last_modified: string;
  kind: string | null;
};

export type PolymarketSnapshotLatestResponse = {
  date: string | null;
  files: PolymarketSnapshotFileSummary[];
  history_file: PolymarketSnapshotFileSummary | null;
};

export type PolymarketSnapshotHistoryResponse = {
  files: PolymarketSnapshotFileSummary[];
};

export type PolymarketSnapshotPreviewResponse = {
  file: PolymarketSnapshotFileSummary;
  headers: string[];
  rows: Record<string, string | null>[];
  row_count: number | null;
  mode: string;
  limit: number;
};

export type PolymarketSnapshotListResponse = {
  out_dir: string;
  runs: PolymarketSnapshotRunSummary[];
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function runPolymarketSnapshot(
  payload: PolymarketSnapshotRunRequest,
): Promise<PolymarketSnapshotRunResponse> {
  const response = await fetch(`${API_BASE}/polymarket-snapshots/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      tickers: payload.tickers,
      tickers_csv: payload.tickersCsv,
      slug_overrides: payload.slugOverrides,
      risk_free_rate: payload.riskFreeRate,
      tz: payload.tz,
      keep_nonexec: payload.keepNonexec ?? false,
    }),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Snapshot request failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function fetchPolymarketSnapshotRuns(
  limit = 12,
): Promise<PolymarketSnapshotListResponse> {
  const response = await fetch(
    `${API_BASE}/polymarket-snapshots/runs?limit=${limit}`,
  );
  if (!response.ok) {
    throw new Error(`Runs request failed: ${response.status}`);
  }
  return response.json();
}

export async function fetchLatestPolymarketSnapshot(): Promise<PolymarketSnapshotLatestResponse> {
  const response = await fetch(`${API_BASE}/polymarket-snapshots/latest`);
  if (!response.ok) {
    throw new Error(`Latest snapshot request failed: ${response.status}`);
  }
  return response.json();
}

export async function fetchPolymarketSnapshotHistory(): Promise<PolymarketSnapshotHistoryResponse> {
  const response = await fetch(`${API_BASE}/polymarket-snapshots/history`);
  if (!response.ok) {
    throw new Error(`History request failed: ${response.status}`);
  }
  return response.json();
}

export async function fetchPolymarketSnapshotPreview(
  file: string,
  limit = 20,
  mode: "head" | "tail" = "head",
): Promise<PolymarketSnapshotPreviewResponse> {
  const params = new URLSearchParams({
    file,
    limit: String(limit),
    mode,
  });
  const response = await fetch(`${API_BASE}/polymarket-snapshots/preview?${params.toString()}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Preview request failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}
