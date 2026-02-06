export type PHATEdgeRunRequest = {
  model_path: string;
  snapshot_csv: string;
  out_csv?: string;
  exclude_tickers?: string;
  require_columns_strict?: boolean;
  compute_edge?: boolean;
  skip_edge_outside_prn_range?: boolean;
};

export type PHATEdgeDistributionStats = {
  count: number;
  mean: number;
  min: number;
  max: number;
};

export type PHATEdgeRow = {
  ticker: string;
  K: number | null;
  spot: number | null;
  pHAT: number | null;
  qHAT: number | null;
  edge: number | null;
  pPM_buy: number | null;
  qPM_buy: number | null;
  edge_source: string | null;
  pRN: number | null;
  qRN: number | null;
};

export type PHATEdgeTopRow = PHATEdgeRow;

export type PHATEdgeRunResponse = {
  ok: boolean;
  command: string[];
  stdout: string;
  stderr: string;
  run_summary: {
    model_path: string;
    snapshot_csv: string;
    output_csv: string;
    duration_s: number;
    ok: boolean;
  };
  pHat_distribution: PHATEdgeDistributionStats | null;
  edge_distribution: PHATEdgeDistributionStats | null;
  top_edges: PHATEdgeRow[];
};

export type PHATEdgeFileSummary = {
  name: string;
  path: string;
  size_bytes: number;
  last_modified: string;
};

export type PHATEdgePreviewResponse = {
  file: PHATEdgeFileSummary;
  headers: string[];
  rows: Record<string, string | null>[];
  row_count: number | null;
  mode: string;
  limit: number;
};

export type PHATEdgeRunListResponse = {
  runs: PHATEdgeFileSummary[];
};

export type PHATEdgeSummaryResponse = {
  file: PHATEdgeFileSummary;
  pHat_distribution: PHATEdgeDistributionStats | null;
  edge_distribution: PHATEdgeDistributionStats | null;
  top_edges: PHATEdgeRow[];
};

export type PHATEdgeRowsResponse = {
  file: PHATEdgeFileSummary;
  rows: PHATEdgeRow[];
  row_count: number;
};

export type PHATEdgeDeleteResponse = {
  path: string;
  deleted: boolean;
};

export type PHATEdgeJobStatus = {
  job_id: string;
  status: "queued" | "running" | "finished" | "failed";
  result: PHATEdgeRunResponse | null;
  error: string | null;
  started_at: string | null;
  finished_at: string | null;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function runPhatEdge(
  payload: PHATEdgeRunRequest,
): Promise<PHATEdgeRunResponse> {
  const body = {
    model_path: payload.model_path,
    snapshot_csv: payload.snapshot_csv,
    ...(payload.out_csv ? { out_csv: payload.out_csv } : {}),
    ...(payload.exclude_tickers ? { exclude_tickers: payload.exclude_tickers } : {}),
    require_columns_strict:
      payload.require_columns_strict ?? true,
    compute_edge: payload.compute_edge ?? true,
    skip_edge_outside_prn_range:
      payload.skip_edge_outside_prn_range ?? true,
  };

  const response = await fetch(`${API_BASE}/phat-edge/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `pHAT edge request failed (${response.status}): ${
        detail || "unknown error"
      }`,
    );
  }

  return response.json();
}

export async function startPhatEdgeJob(
  payload: PHATEdgeRunRequest,
): Promise<PHATEdgeJobStatus> {
  const body = {
    model_path: payload.model_path,
    snapshot_csv: payload.snapshot_csv,
    ...(payload.out_csv ? { out_csv: payload.out_csv } : {}),
    ...(payload.exclude_tickers ? { exclude_tickers: payload.exclude_tickers } : {}),
    require_columns_strict:
      payload.require_columns_strict ?? true,
    compute_edge: payload.compute_edge ?? true,
    skip_edge_outside_prn_range:
      payload.skip_edge_outside_prn_range ?? true,
  };

  const response = await fetch(`${API_BASE}/phat-edge/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `pHAT edge job start failed (${response.status}): ${
        detail || "unknown error"
      }`,
    );
  }

  return response.json();
}

export async function getPhatEdgeJob(
  jobId: string,
): Promise<PHATEdgeJobStatus> {
  const response = await fetch(`${API_BASE}/phat-edge/jobs/${jobId}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `pHAT edge job not found (${response.status}): ${
        detail || "unknown error"
      }`,
    );
  }
  return response.json();
}

export async function fetchPhatEdgePreview(
  path: string,
  limit = 18,
  mode: "head" | "tail" = "head",
): Promise<PHATEdgePreviewResponse> {
  const params = new URLSearchParams({
    path,
    limit: String(limit),
    mode,
  });
  const response = await fetch(`${API_BASE}/phat-edge/preview?${params.toString()}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Edge preview request failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function fetchPhatEdgeRuns(
  limit = 12,
): Promise<PHATEdgeRunListResponse> {
  const response = await fetch(`${API_BASE}/phat-edge/runs?limit=${limit}`);
  if (!response.ok) {
    throw new Error(`Edge runs request failed: ${response.status}`);
  }
  return response.json();
}

export async function fetchPhatEdgeSummary(
  path: string,
): Promise<PHATEdgeSummaryResponse> {
  const params = new URLSearchParams({ path });
  const response = await fetch(`${API_BASE}/phat-edge/summary?${params.toString()}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Edge summary request failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function fetchPhatEdgeRows(
  path: string,
): Promise<PHATEdgeRowsResponse> {
  const params = new URLSearchParams({ path });
  const response = await fetch(`${API_BASE}/phat-edge/rows?${params.toString()}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Edge rows request failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function deletePhatEdgeRun(
  path: string,
): Promise<PHATEdgeDeleteResponse> {
  const params = new URLSearchParams({ path });
  const response = await fetch(`${API_BASE}/phat-edge/runs?${params.toString()}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Delete edge run failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}
