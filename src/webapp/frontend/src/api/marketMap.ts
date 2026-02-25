export type MarketMapRunRequest = {
  runDir?: string;
  runId?: string;
  overrides?: string;
  tickers?: string;
  prnDataset?: string;
  out?: string;
  strict?: boolean;
};

export type MarketMapRunResponse = {
  output_path?: string | null;
  row_count?: number | null;
  source_run?: string | null;
  stdout?: string | null;
  stderr?: string | null;
  duration_s?: number | null;
};

export type MarketMapPreviewResponse = {
  file?: {
    name: string;
    path: string;
    last_modified?: string | null;
    size_bytes?: number | null;
  } | null;
  headers?: string[] | null;
  rows?: Record<string, string | null>[] | null;
  row_count?: number | null;
  limit?: number | null;
};

export type MarketMapDeleteResponse = {
  deleted: boolean;
  paths: string[];
};

export type MarketMapJobStatus = {
  job_id: string;
  status: "queued" | "running" | "finished" | "failed";
  result: MarketMapRunResponse | null;
  error: string | null;
  started_at: string | null;
  finished_at: string | null;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const buildPayload = (payload: MarketMapRunRequest) => ({
  ...(payload.runDir ? { run_dir: payload.runDir } : {}),
  ...(payload.runId ? { run_id: payload.runId } : {}),
  ...(payload.overrides ? { overrides: payload.overrides } : {}),
  ...(payload.tickers ? { tickers: payload.tickers } : {}),
  ...(payload.prnDataset ? { prn_dataset: payload.prnDataset } : {}),
  ...(payload.out ? { out: payload.out } : {}),
  ...(payload.strict !== undefined ? { strict: payload.strict } : {}),
});

export async function startMarketMapRun(
  payload: MarketMapRunRequest,
): Promise<MarketMapRunResponse> {
  const response = await fetch(`${API_BASE}/market-map/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildPayload(payload)),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Market map run failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function startMarketMapJob(
  payload: MarketMapRunRequest,
): Promise<MarketMapJobStatus> {
  const response = await fetch(`${API_BASE}/market-map/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildPayload(payload)),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Market map run failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function getMarketMapJob(
  jobId: string,
): Promise<MarketMapJobStatus> {
  const response = await fetch(`${API_BASE}/market-map/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error(`Market map job request failed: ${response.status}`);
  }
  return response.json();
}

export async function fetchMarketMapPreview(
  limit = 20,
  path?: string,
): Promise<MarketMapPreviewResponse> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (path) params.set("path", path);
  const response = await fetch(`${API_BASE}/market-map/preview?${params}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Market map preview failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function deleteMarketMapOutput(
  path?: string,
): Promise<MarketMapDeleteResponse> {
  const params = path ? `?path=${encodeURIComponent(path)}` : "";
  const response = await fetch(`${API_BASE}/market-map/output${params}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Market map delete failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}
