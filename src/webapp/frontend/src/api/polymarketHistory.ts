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
  runDirName?: string;
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

export type PolymarketRunFeaturesRequest = {
  prnDataset?: string;
  skipSubgraphLabels?: boolean;
};

export type PolymarketRunFeaturesResponse = {
  ok: boolean;
  run_id: string;
  run_dir: string;
  features_built: boolean;
  features_path: string | null;
  features_manifest_path: string | null;
  stdout: string;
  stderr: string;
  duration_s: number;
  command: string[];
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
  rows: Record<string, string | null>[];
  row_count?: number | null;
  mode: "head" | "tail";
  limit: number;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export const getPipelineRunFileUrl = (runId: string, filename: string): string => {
  const params = new URLSearchParams({ filename });
  return `${API_BASE}/polymarket-history/runs/${encodeURIComponent(runId)}/file?${params.toString()}`;
};

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
      run_dir_name: payload.runDirName,
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
  mode: "head" | "tail" = "head",
  limit: number = 20,
): Promise<CsvPreview> {
  const params = new URLSearchParams({
    mode,
    limit: String(limit),
  });
  const response = await fetch(
    `${API_BASE}/polymarket-history/jobs/${jobId}/csv-preview/${filename}?${params.toString()}`,
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `CSV preview failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function previewPipelineRunCsv(
  runId: string,
  filename: string,
  mode: "head" | "tail" = "head",
  limit: number = 20,
): Promise<CsvPreview> {
  const params = new URLSearchParams({
    mode,
    limit: String(limit),
  });
  const response = await fetch(
    `${API_BASE}/polymarket-history/runs/${encodeURIComponent(runId)}/csv-preview/${filename}?${params.toString()}`,
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Run CSV preview failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function buildDecisionFeaturesForRun(
  runId: string,
  payload: PolymarketRunFeaturesRequest = {},
): Promise<PolymarketRunFeaturesResponse> {
  const response = await fetch(
    `${API_BASE}/polymarket-history/runs/${encodeURIComponent(runId)}/build-features`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prn_dataset: payload.prnDataset,
        skip_subgraph_labels: payload.skipSubgraphLabels ?? false,
      }),
    },
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Decision features build failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

// ---------------------------------------------------------------------------
// Run management types & API (Phase 2)
// ---------------------------------------------------------------------------

export type PipelineRunSummary = {
  run_id: string;
  run_dir: string;
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
  csv_files: {
    name: string;
    size_bytes: number;
    row_count?: number | null;
  }[];
  size_bytes: number;
  error_summary: string | null;
  analytics?: {
    analytics_ready: boolean;
    primary_metric: string;
    requires_true_volume: boolean;
    include_subgraph_requested: boolean;
    has_trades: boolean;
    trade_entities: number | null;
    trade_days: number | null;
    trade_artifact_path?: string | null;
    warning: string | null;
  } | null;
};

export type RunDailyAnalyticsDay = {
  date: string;
  weekday: string;
  daily_notional_volume: number;
  share_volume: number;
  trade_count: number;
  active_markets: number;
  active_tickers: number;
  expected_notional_volume: number | null;
  band_lo: number | null;
  band_hi: number | null;
  realized_expected_ratio: number | null;
  day_over_day_delta: number | null;
  day_over_day_pct: number | null;
  noise_cv_28d: number | null;
  noise_mad_ratio_28d: number | null;
  unusually_active: boolean;
  unusually_inactive: boolean;
  flagged_outlier: boolean;
  baseline_source: string;
};

export type RunDailyAnalyticsSnapshot = RunDailyAnalyticsDay;

export type RunDailyAnalyticsSummary = {
  latest_day: RunDailyAnalyticsSnapshot | null;
  comparison_day: RunDailyAnalyticsSnapshot | null;
  delta_notional: number | null;
  delta_notional_pct: number | null;
  delta_share_volume: number | null;
  delta_trade_count: number | null;
  latest_noise_cv_28d: number | null;
  latest_noise_mad_ratio_28d: number | null;
  latest_unusually_active: boolean;
  latest_unusually_inactive: boolean;
  latest_flagged_outlier: boolean;
};

export type RunDailyAnalyticsCoverage = {
  artifact_path: string | null;
  total_trade_rows: number;
  valid_trade_rows: number;
  filtered_trade_rows: number;
  observed_trade_days: number;
  filled_days: number;
  effective_date_min: string | null;
  effective_date_max: string | null;
  requested_date_min: string | null;
  requested_date_max: string | null;
  requested_tickers: string[];
  requested_token_role: "all" | "yes" | "no";
};

export type RunDailyAnalyticsStructureBucket = {
  key: string;
  label: string;
  observations: number;
  mean_daily_notional_volume: number;
  median_daily_notional_volume: number;
  total_notional_volume: number;
  share_total_notional_volume: number;
};

export type RunDailyAnalyticsStructure = {
  weekday: RunDailyAnalyticsStructureBucket[];
  event_proximity: RunDailyAnalyticsStructureBucket[];
  ladder_bucket: RunDailyAnalyticsStructureBucket[];
};

export type RunDailyAnalyticsResponse = {
  run_id: string;
  analytics_ready: boolean;
  primary_metric: string;
  metric_mode: "true_volume";
  token_role: "all" | "yes" | "no";
  ci_level: number;
  exclude_flagged: boolean;
  baseline_window_days: number;
  latest_trade_date: string | null;
  comparison_date: string | null;
  available_tickers: string[];
  coverage: RunDailyAnalyticsCoverage;
  summary: RunDailyAnalyticsSummary;
  days: RunDailyAnalyticsDay[];
  structure: RunDailyAnalyticsStructure;
  warnings: string[];
};

export type RunDailyAnalyticsQuery = {
  dateMin?: string;
  dateMax?: string;
  tickers?: string[];
  tokenRole?: "all" | "yes" | "no";
  ciLevel?: number;
  excludeFlagged?: boolean;
};

export type RunDailyAnalyticsBreakdownRow = {
  key: string;
  label: string;
  ticker: string | null;
  market_id: string | null;
  threshold: number | null;
  week_friday: string | null;
  event_endDate: string | null;
  ladder_bucket: string | null;
  event_proximity_bucket: string | null;
  daily_notional_volume: number;
  share_volume: number;
  trade_count: number;
  expected_notional_volume: number | null;
  band_lo: number | null;
  band_hi: number | null;
  realized_expected_ratio: number | null;
  unusually_active: boolean;
  unusually_inactive: boolean;
  flagged_outlier: boolean;
  baseline_source: string;
  history_days: number;
  volume_share_of_day: number | null;
};

export type RunDailyAnalyticsBreakdownResponse = {
  run_id: string;
  date: string;
  group_by: "ticker" | "market";
  token_role: "all" | "yes" | "no";
  ci_level: number;
  exclude_flagged: boolean;
  available_tickers: string[];
  selected_tickers: string[];
  rows: RunDailyAnalyticsBreakdownRow[];
  warnings: string[];
};

export type RunDailyAnalyticsBreakdownQuery = RunDailyAnalyticsQuery & {
  date: string;
  groupBy?: "ticker" | "market";
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

export async function getRunDailyAnalyticsSummary(
  runId: string,
  query: RunDailyAnalyticsQuery = {},
): Promise<RunDailyAnalyticsResponse> {
  const params = new URLSearchParams();
  if (query.dateMin) params.set("date_min", query.dateMin);
  if (query.dateMax) params.set("date_max", query.dateMax);
  if (query.tickers && query.tickers.length) {
    params.set("tickers", query.tickers.join(","));
  }
  if (query.tokenRole) params.set("token_role", query.tokenRole);
  if (query.ciLevel != null) params.set("ci_level", String(query.ciLevel));
  if (query.excludeFlagged != null) {
    params.set("exclude_flagged", String(query.excludeFlagged));
  }
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const response = await fetch(
    `${API_BASE}/polymarket-history/runs/${encodeURIComponent(runId)}/analytics/daily-summary${suffix}`,
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Daily analytics failed (${response.status}): ${detail || "unknown error"}`);
  }
  return response.json();
}

export async function getRunDailyAnalyticsBreakdown(
  runId: string,
  query: RunDailyAnalyticsBreakdownQuery,
): Promise<RunDailyAnalyticsBreakdownResponse> {
  const params = new URLSearchParams({ date: query.date });
  if (query.groupBy) params.set("group_by", query.groupBy);
  if (query.dateMin) params.set("date_min", query.dateMin);
  if (query.dateMax) params.set("date_max", query.dateMax);
  if (query.tickers && query.tickers.length) {
    params.set("tickers", query.tickers.join(","));
  }
  if (query.tokenRole) params.set("token_role", query.tokenRole);
  if (query.ciLevel != null) params.set("ci_level", String(query.ciLevel));
  if (query.excludeFlagged != null) {
    params.set("exclude_flagged", String(query.excludeFlagged));
  }
  const response = await fetch(
    `${API_BASE}/polymarket-history/runs/${encodeURIComponent(runId)}/analytics/daily-breakdown?${params.toString()}`,
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Daily analytics breakdown failed (${response.status}): ${detail || "unknown error"}`);
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
