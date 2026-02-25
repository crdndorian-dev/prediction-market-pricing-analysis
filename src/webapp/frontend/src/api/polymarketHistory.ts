export type PolymarketHistoryRunRequest = {
  tickers?: string[];
  tickersCsv?: string;
  eventUrls?: string[];
  eventUrlsFile?: string;
  startDate?: string;
  endDate?: string;
  fidelityMin?: number;
  barsFreqs?: string;
  outDir?: string;
  barsDir?: string;
  dimMarketOut?: string;
  factTradeDir?: string;
  includeSubgraph?: boolean;
  maxSubgraphEntities?: number;
  dryRun?: boolean;
  buildFeatures?: boolean;
  prnDataset?: string;
  skipSubgraphLabels?: boolean;
};

export type PolymarketHistoryRunResponse = {
  ok: boolean;
  run_id: string | null;
  out_dir: string;
  run_dir: string | null;
  files: string[];
  stdout: string;
  stderr: string;
  duration_s: number;
  command: string[];
  features_built: boolean;
  features_path: string | null;
  features_manifest_path: string | null;
};

export type PolymarketHistoryJobPhase = "history" | "features" | "finalizing";

export type PipelineProgress = {
  total: number;
  completed: number;
  failed: number;
  status: "running" | "completed" | "failed";
};

export type PolymarketHistoryJobStatus = {
  job_id: string;
  status: "queued" | "running" | "finished" | "failed" | "cancelled";
  phase?: PolymarketHistoryJobPhase | null;
  progress?: PipelineProgress | null;
  features_progress?: PipelineProgress | null;
  result: PolymarketHistoryRunResponse | null;
  error: string | null;
  started_at: string | null;
  finished_at: string | null;
};

export type CsvPreview = {
  filename: string;
  headers: string[];
  rows: Record<string, string>[];
  total_rows: number;
  preview_limit: number;
  truncated: boolean;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function startPolymarketHistoryJob(
  payload: PolymarketHistoryRunRequest,
): Promise<PolymarketHistoryJobStatus> {
  const response = await fetch(`${API_BASE}/polymarket-history/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      tickers: payload.tickers,
      tickers_csv: payload.tickersCsv,
      event_urls: payload.eventUrls,
      event_urls_file: payload.eventUrlsFile,
      start_date: payload.startDate,
      end_date: payload.endDate,
      fidelity_min: payload.fidelityMin,
      bars_freqs: payload.barsFreqs,
      out_dir: payload.outDir,
      bars_dir: payload.barsDir,
      dim_market_out: payload.dimMarketOut,
      fact_trade_dir: payload.factTradeDir,
      include_subgraph: payload.includeSubgraph ?? false,
      max_subgraph_entities: payload.maxSubgraphEntities,
      dry_run: payload.dryRun ?? false,
      build_features: payload.buildFeatures ?? false,
      prn_dataset: payload.prnDataset,
      skip_subgraph_labels: payload.skipSubgraphLabels ?? false,
    }),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Weekly history job start failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function getPolymarketHistoryJob(
  jobId: string,
): Promise<PolymarketHistoryJobStatus> {
  const response = await fetch(`${API_BASE}/polymarket-history/jobs/${jobId}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Weekly history job not found (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function cancelPolymarketHistoryJob(
  jobId: string,
): Promise<PolymarketHistoryJobStatus> {
  const response = await fetch(`${API_BASE}/polymarket-history/jobs/${jobId}/cancel`, {
    method: "POST",
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Weekly history job cancel failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function getCsvPreview(
  jobId: string,
  filename: string,
  limit: number = 100,
): Promise<CsvPreview> {
  const response = await fetch(
    `${API_BASE}/polymarket-history/jobs/${jobId}/csv-preview/${filename}?limit=${limit}`,
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `CSV preview failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

// ---------------------------------------------------------------------------
// Run management types & API (Phase 2)
// ---------------------------------------------------------------------------

export type PipelineRunSummary = {
  run_id: string;
  label: string | null;
  status: string;
  created_at_utc: string | null;
  finished_at_utc: string | null;
  duration_s: number | null;
  tickers: string[] | null;
  start_date: string | null;
  end_date: string | null;
  markets: number | null;
  price_rows: number | null;
  features_built: boolean;
  pinned: boolean;
  is_active: boolean;
  artifact_count: number;
  size_bytes: number;
  error_summary: string | null;
};

export type StorageSummary = {
  total_runs: number;
  total_size_bytes: number;
  total_size_mb: number;
};

export type LatestPointer = {
  run_id: string;
  updated_at_utc: string;
  updated_by: string;
} | null;

export type PipelineRunsResponse = {
  runs: PipelineRunSummary[];
  storage: StorageSummary;
  latest: LatestPointer;
};

export async function listPipelineRuns(): Promise<PipelineRunsResponse> {
  const response = await fetch(`${API_BASE}/polymarket-history/runs`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Failed to list runs (${response.status}): ${detail}`);
  }
  return response.json();
}

export async function renamePipelineRun(
  runId: string,
  label: string | null,
): Promise<{ run_id: string; label: string | null }> {
  const response = await fetch(`${API_BASE}/polymarket-history/runs/${runId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Rename failed (${response.status}): ${detail}`);
  }
  return response.json();
}

export async function setActiveRun(
  runId: string,
): Promise<{ run_id: string; active: boolean }> {
  const response = await fetch(`${API_BASE}/polymarket-history/runs/active`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id: runId }),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Set active failed (${response.status}): ${detail}`);
  }
  return response.json();
}

export async function deletePipelineRun(
  runId: string,
): Promise<{ run_id: string; deleted: boolean }> {
  const response = await fetch(`${API_BASE}/polymarket-history/runs/${runId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Delete failed (${response.status}): ${detail}`);
  }
  return response.json();
}

export async function togglePinRun(
  runId: string,
): Promise<{ run_id: string; pinned: boolean }> {
  const response = await fetch(
    `${API_BASE}/polymarket-history/runs/${runId}/pin`,
    { method: "POST" },
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Pin toggle failed (${response.status}): ${detail}`);
  }
  return response.json();
}
