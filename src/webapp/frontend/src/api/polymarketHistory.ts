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
  master_bar_artifacts: SharedArtifactSummary[];
  artifact_groups: RunArtifactGroupSummary[];
  shared_artifacts: SharedArtifactSummary[];
  quality_summary?: PolymarketQualitySummary | null;
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

export type SharedArtifactSummary = {
  name: string;
  path: string;
  frequency?: string | null;
  size_bytes: number;
  row_count?: number | null;
  last_modified?: string | null;
};

export type RunArtifactSummary = {
  name: string;
  path: string;
  size_bytes: number;
  row_count?: number | null;
  last_modified?: string | null;
};

export type RunArtifactGroupSummary = {
  key: string;
  label: string;
  path: string;
  files: RunArtifactSummary[];
};

export type PolymarketQualityBucketCounts = {
  clean: number;
  watch: number;
  noisy: number;
};

export type PolymarketQualityFlagSummary = {
  name: string;
  count: number;
  share?: number | null;
};

export type PolymarketQualityTickerSummary = {
  ticker: string;
  market_count: number;
  flagged_market_count: number;
  flagged_share?: number | null;
  avg_issue_count?: number | null;
  clean_share?: number | null;
  watch_share?: number | null;
  noisy_share?: number | null;
};

export type PolymarketQualitySummary = {
  market_count: number;
  flagged_market_count: number;
  flagged_share: number;
  bucket_counts: PolymarketQualityBucketCounts;
  prn_coverage_counts: Record<string, number>;
  top_flags: PolymarketQualityFlagSummary[];
  top_problem_tickers: PolymarketQualityTickerSummary[];
  snapshot_anchor?: string | null;
  quality_columns?: string[];
  quality_flag_columns?: string[];
};

export type PolymarketMarketQuality = {
  market_id?: string | null;
  ticker?: string | null;
  threshold?: number | null;
  week_friday?: string | null;
  quality_issue_count?: number | null;
  quality_bucket?: string | null;
  active_flags: string[];
  snapshot_date_used?: string | null;
  snapshot_time_used?: string | null;
  snapshot_coverage_status?: string | null;
  snapshot_drop_reason?: string | null;
  snapshot_pRN?: number | null;
  snapshot_abs_log_m_fwd?: number | null;
  yes_points?: number | null;
  stale_ratio?: number | null;
  max_stale_hours?: number | null;
  midprice_cluster_ratio?: number | null;
  max_jump?: number | null;
  hours_since_last_yes_trade?: number | null;
  gamma_volume?: number | null;
  flag_not_relevant?: boolean | null;
  flag_prn_missing?: boolean | null;
  flag_pm_no_trade_history?: boolean | null;
  flag_pm_no_recent_trade?: boolean | null;
  flag_pm_stale_prices?: boolean | null;
  flag_extreme_otm?: boolean | null;
};

export type PolymarketQualityMarketSample = {
  market_id: string;
  event_id?: string | null;
  ticker: string;
  threshold?: number | null;
  week_friday: string;
  quality_issue_count?: number | null;
  quality_bucket?: string | null;
  active_flags: string[];
  snapshot_coverage_status?: string | null;
  snapshot_drop_reason?: string | null;
  hours_since_last_yes_trade?: number | null;
  stale_ratio?: number | null;
  max_stale_hours?: number | null;
  midprice_cluster_ratio?: number | null;
  max_jump?: number | null;
  gamma_volume?: number | null;
};

export type PolymarketQualityWeekSummary = {
  week_friday: string;
  market_count: number;
  flagged_market_count: number;
  flagged_share?: number | null;
  avg_issue_count?: number | null;
  quality_bucket?: string | null;
};

export type PolymarketQualityAuditResponse = {
  run_id: string;
  available: boolean;
  message?: string | null;
  summary: PolymarketQualitySummary;
  flag_distribution: PolymarketQualityFlagSummary[];
  problem_tickers: PolymarketQualityTickerSummary[];
  problem_markets: PolymarketQualityMarketSample[];
  weekly_summary: PolymarketQualityWeekSummary[];
  available_quality_flags: string[];
};

export type PolymarketQualityTelemetry = {
  phase: "quality" | "complete";
  total_markets: number;
  completed_markets: number;
  flagged_markets: number;
  flagged_share: number;
  bucket_counts: PolymarketQualityBucketCounts;
  top_flags: PolymarketQualityFlagSummary[];
  prn_coverage_counts: Record<string, number>;
  top_problem_tickers: PolymarketQualityTickerSummary[];
};

export type PolymarketHistoryJobPhase = "history" | "prn" | "quality" | "features" | "finalizing";

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
  telemetry?: PolymarketQualityTelemetry | null;
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

export const getPipelineRunArtifactFileUrl = (runId: string, path: string): string => {
  const params = new URLSearchParams({ path });
  return `${API_BASE}/polymarket-history/runs/${encodeURIComponent(runId)}/artifacts/file?${params.toString()}`;
};

export const getMasterBarFileUrl = (freq: "1h" | "1d"): string =>
  `${API_BASE}/polymarket-history/master-bars/${encodeURIComponent(freq)}/file`;

export const getRunMasterBarFileUrl = (runId: string, freq: "1h" | "1d"): string =>
  `${API_BASE}/polymarket-history/runs/${encodeURIComponent(runId)}/master-bars/${encodeURIComponent(freq)}/file`;

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

export async function previewPipelineRunArtifactCsv(
  runId: string,
  path: string,
  mode: "head" | "tail" = "head",
  limit: number = 20,
): Promise<CsvPreview> {
  const params = new URLSearchParams({
    path,
    mode,
    limit: String(limit),
  });
  const response = await fetch(
    `${API_BASE}/polymarket-history/runs/${encodeURIComponent(runId)}/artifacts/csv-preview?${params.toString()}`,
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Run artifact CSV preview failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function previewMasterBarCsv(
  freq: "1h" | "1d",
  mode: "head" | "tail" = "head",
  limit: number = 20,
): Promise<CsvPreview> {
  const params = new URLSearchParams({
    mode,
    limit: String(limit),
  });
  const response = await fetch(
    `${API_BASE}/polymarket-history/master-bars/${encodeURIComponent(freq)}/csv-preview?${params.toString()}`,
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Master bars preview failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function previewRunMasterBarCsv(
  runId: string,
  freq: "1h" | "1d",
  mode: "head" | "tail" = "head",
  limit: number = 20,
): Promise<CsvPreview> {
  const params = new URLSearchParams({
    mode,
    limit: String(limit),
  });
  const response = await fetch(
    `${API_BASE}/polymarket-history/runs/${encodeURIComponent(runId)}/master-bars/${encodeURIComponent(freq)}/csv-preview?${params.toString()}`,
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Run master bars preview failed (${response.status}): ${detail || "unknown error"}`,
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
  features_requested: boolean;
  pending_phase?: string | null;
  artifacts_accessible: boolean;
  pinned: boolean;
  is_active: boolean;
  artifact_count: number;
  csv_files: {
    name: string;
    size_bytes: number;
    row_count?: number | null;
  }[];
  master_bar_artifacts: SharedArtifactSummary[];
  artifact_groups: RunArtifactGroupSummary[];
  shared_artifacts: SharedArtifactSummary[];
  size_bytes: number;
  error_summary: string | null;
  quality_summary?: PolymarketQualitySummary | null;
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

export async function getPipelineRunQualityAudit(
  runId: string,
): Promise<PolymarketQualityAuditResponse> {
  const response = await fetch(
    `${API_BASE}/polymarket-history/runs/${encodeURIComponent(runId)}/quality-audit`,
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Failed to load quality audit (${response.status}): ${detail}`);
  }
  return response.json();
}

export async function renamePipelineRun(
  runId: string,
  label: string | null,
  newDirName?: string | null,
): Promise<{
  run_id: string;
  label: string | null;
  renamed_dir: boolean;
  run_dir: string;
}> {
  const payload: Record<string, unknown> = { label };
  if (newDirName != null) {
    payload.new_dir_name = newDirName;
  }
  const response = await fetch(`${API_BASE}/polymarket-history/runs/${runId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
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
