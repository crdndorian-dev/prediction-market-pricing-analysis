export type DatasetRunRequest = {
  outDir?: string;
  datasetName?: string;
  scheduleMode?: "weekly" | "expiry_range";
  expiryWeekdays?: string;
  asofWeekdays?: string;
  dteList?: string;
  dteMin?: number;
  dteMax?: number;
  dteStep?: number;
  writeSnapshot?: boolean;
  writePrnView?: boolean;
  writeTrainView?: boolean;
  writeLegacy?: boolean;
  prnVersion?: string;
  prnConfigHash?: string;
  tickers?: string;
  start: string;
  end: string;
  thetaBaseUrl?: string;
  stockSource?: "yfinance" | "theta" | "auto";
  timeoutS?: number;
  riskFreeRate?: number;
  maxAbsLogm?: number;
  maxAbsLogmCap?: number;
  bandWidenStep?: number;
  noAdaptiveBand?: boolean;
  maxBandStrikes?: number;
  minBandStrikes?: number;
  minBandPrnStrikes?: number;
  strikeRange?: number;
  noRetryFullChain?: boolean;
  noSatExpiryFallback?: boolean;
  threads?: number;
  preferBidask?: boolean;
  minTradeCount?: number;
  minVolume?: number;
  minChainUsedHard?: number;
  maxRelSpreadMedianHard?: number;
  hardDropCloseFallback?: boolean;
  minPrnTrain?: number;
  maxPrnTrain?: number;
  noSplitAdjust?: boolean;
  dividendSource?: "yfinance" | "none";
  dividendLookbackDays?: number;
  dividendYieldDefault?: number;
  noForwardMoneyness?: boolean;
  noGroupWeights?: boolean;
  noTickerWeights?: boolean;
  noSoftQualityWeight?: boolean;
  rvLookbackDays?: number;
  tickerReweightMode?: string;
  tickerReweightAlphaMin?: number;
  tickerReweightAlphaMax?: number;
  tradeFocusBeta?: number;
  tradeFocusTickers?: string;
  cache?: boolean;
  writeDrops?: boolean;
  sanityReport?: boolean;
  sanityDrop?: boolean;
  sanityAbsLogmMax?: number;
  sanityKOverSMin?: number;
  sanityKOverSMax?: number;
  verboseSkips?: boolean;
};

export type DatasetRunResponse = {
  ok: boolean;
  out_dir: string;
  out_name: string;
  run_dir?: string | null;
  output_file: string | null;
  drops_file: string | null;
  training_file?: string | null;
  stdout: string;
  stderr: string;
  duration_s: number;
  command: string[];
};

export type DatasetJobProgress = {
  done: number;
  total: number;
  groups: number;
  rows: number;
  lastTicker: string;
  lastWeek: string;
  lastAsof: string;
};

export type DatasetJobGroupChecks = {
  asof_close_fallback: number;
  expiry_close_fallback: number;
  expiry_saturday_fallback: number;
  quote_close_fallback: number;
  low_chain_used: number;
  wide_rel_spread: number;
};

export type DatasetJobTickerTelemetry = {
  ticker: string;
  completed_jobs: number;
  planned_jobs: number;
  kept_groups: number;
  rows: number;
  issue_count_sum: number;
  flagged_rows: number;
  fallback_rows: number;
  wide_spread_rows: number;
  clean_rows: number;
  watch_rows: number;
  noisy_rows: number;
  drop_reasons: Record<string, number>;
};

export type DatasetJobTelemetry = {
  phase:
    | "planning"
    | "preloading_stock"
    | "preloading_dividends"
    | "building"
    | "finalizing"
    | "writing_outputs"
    | "finished";
  drop_reasons: Record<string, number>;
  group_checks: DatasetJobGroupChecks;
  tickers: DatasetJobTickerTelemetry[];
};

export type DatasetJobStatus = {
  job_id: string;
  status: "queued" | "running" | "finished" | "failed" | "cancelled";
  progress: DatasetJobProgress | null;
  telemetry?: DatasetJobTelemetry | null;
  stdout: string[];
  stderr: string[];
  result: DatasetRunResponse | null;
  error: string | null;
  started_at: string | null;
  finished_at: string | null;
};

export type DatasetFileSummary = {
  name: string;
  path: string;
  size_bytes: number;
  last_modified?: string | null;
};

export type DatasetRunSummary = {
  id: string;
  run_dir: string;
  dataset_file?: DatasetFileSummary | null;
  drops_file?: DatasetFileSummary | null;
  training_file?: DatasetFileSummary | null;
  files?: DatasetFileSummary[] | null;
  last_modified?: string | null;
  status?: "ready" | "creating";
  job_id?: string | null;
};

export type DatasetBackfillRange = {
  start: string;
  end: string;
};

export type DatasetBackfillResponse = {
  ok: boolean;
  backfilled: boolean;
  run_dir: string;
  training_file?: string | null;
  used_defaults?: boolean;
  required_start?: string | null;
  required_end?: string | null;
  existing_start?: string | null;
  existing_end?: string | null;
  backfill_ranges?: DatasetBackfillRange[];
  rows_before?: number | null;
  rows_after?: number | null;
  rows_added?: number | null;
  message: string;
  duration_s: number;
};

export type DatasetBackfillRequest = {
  runDir: string;
  polymarketRunId: string;
  allowDefaults?: boolean;
};

export type DatasetListResponse = {
  base_dir: string;
  runs: DatasetRunSummary[];
};

export type DatasetPreviewResponse = {
  file: DatasetFileSummary;
  headers: string[];
  rows: Record<string, string | null>[];
  row_count?: number | null;
  mode: "head" | "tail";
  limit: number;
};

export type DatasetAuditFlagSummary = {
  name: string;
  count: number;
  share: number;
};

export type DatasetAuditDistribution = {
  name: string;
  min?: number | null;
  p05?: number | null;
  p50?: number | null;
  p95?: number | null;
  max?: number | null;
};

export type DatasetAuditTickerSummary = {
  ticker: string;
  row_count: number;
  snapshot_count?: number | null;
  avg_issue_count?: number | null;
  flagged_share?: number | null;
  fallback_share?: number | null;
  wide_spread_share?: number | null;
  clean_share?: number | null;
  watch_share?: number | null;
  noisy_share?: number | null;
};

export type DatasetAuditTimelinePoint = {
  asof_date: string;
  row_count: number;
  snapshot_count?: number | null;
  avg_issue_count?: number | null;
  flagged_share?: number | null;
};

export type DatasetAuditRow = {
  row_id?: string | null;
  ticker?: string | null;
  asof_date?: string | null;
  expiry_date?: string | null;
  K?: number | null;
  pRN?: number | null;
  quality_issue_count?: number | null;
  rel_spread_median?: number | null;
  n_chain_used?: number | null;
  flags: string[];
};

export type DatasetAuditHeatmapCell = {
  ticker: string;
  asof_date: string;
  row_count: number;
  flagged_share?: number | null;
  avg_issue_count?: number | null;
  quality_bucket?: string | null;
};

export type DatasetAuditRvBucketSummary = {
  label: "low" | "mid" | "high";
  value_min?: number | null;
  value_max?: number | null;
  row_count: number;
  row_share?: number | null;
  avg_issue_count?: number | null;
  flagged_share?: number | null;
};

export type DatasetAuditRvFeatureAudit = {
  feature: string;
  finite_row_count: number;
  finite_row_share?: number | null;
  buckets: DatasetAuditRvBucketSummary[];
};

export type DatasetRowDetailResponse = {
  file: DatasetFileSummary;
  row_id: string;
  row: Record<string, string | null>;
};

export type DatasetAuditResponse = {
  file: DatasetFileSummary;
  row_count: number;
  column_count: number;
  ticker_count?: number | null;
  snapshot_count?: number | null;
  group_count?: number | null;
  date_start?: string | null;
  date_end?: string | null;
  expiry_start?: string | null;
  expiry_end?: string | null;
  available_rv_features: string[];
  available_quality_flags: string[];
  quality_flags: DatasetAuditFlagSummary[];
  numeric_distributions: DatasetAuditDistribution[];
  rv_feature_audit: DatasetAuditRvFeatureAudit[];
  top_problem_tickers: DatasetAuditTickerSummary[];
  timeline: DatasetAuditTimelinePoint[];
  heatmap_dates: string[];
  heatmap_cells: DatasetAuditHeatmapCell[];
  noisiest_rows: DatasetAuditRow[];
};

export type DatasetCleanupCriteria = {
  qualityBuckets?: Array<"clean" | "watch" | "noisy">;
  minQualityIssueCount?: number;
  flagColumns?: string[];
  flagMatchMode?: "any" | "all";
  minRelSpreadMedian?: number;
  maxNChainUsed?: number;
};

export type DatasetCleanupRequest = {
  runDir: string;
  criteria?: DatasetCleanupCriteria;
  allowDefaults?: boolean;
};

export type DatasetCleanupPreviewResponse = {
  run_dir: string;
  training_file: string;
  rows_before: number;
  rows_to_drop: number;
  rows_after: number;
  drop_share: number;
  used_defaults: boolean;
  would_drop_all: boolean;
  dropped_bucket_counts: Record<string, number>;
  matched_flag_counts: DatasetAuditFlagSummary[];
  sample_rows: DatasetAuditRow[];
  message: string;
};

export type DatasetCleanupResponse = DatasetCleanupPreviewResponse & {
  ok: boolean;
  cleaned_run_dir: string;
  cleaned_file: string;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export const getDatasetFileUrl = (path: string): string => {
  const params = new URLSearchParams({ path });
  return `${API_BASE}/datasets/runs/file?${params.toString()}`;
};

const datasetRequestBody = (payload: DatasetRunRequest) => ({
  out_dir: payload.outDir,
  dataset_name: payload.datasetName,
  schedule_mode: payload.scheduleMode,
  expiry_weekdays: payload.expiryWeekdays,
  asof_weekdays: payload.asofWeekdays,
  dte_list: payload.dteList,
  dte_min: payload.dteMin,
  dte_max: payload.dteMax,
  dte_step: payload.dteStep,
  write_snapshot: payload.writeSnapshot,
  write_prn_view: payload.writePrnView,
  write_train_view: payload.writeTrainView,
  write_legacy: payload.writeLegacy,
  prn_version: payload.prnVersion,
  prn_config_hash: payload.prnConfigHash,
  tickers: payload.tickers,
  start: payload.start,
  end: payload.end,
  theta_base_url: payload.thetaBaseUrl,
  stock_source: payload.stockSource,
  timeout_s: payload.timeoutS,
  r: payload.riskFreeRate,
  max_abs_logm: payload.maxAbsLogm,
  max_abs_logm_cap: payload.maxAbsLogmCap,
  band_widen_step: payload.bandWidenStep,
  no_adaptive_band: payload.noAdaptiveBand,
  max_band_strikes: payload.maxBandStrikes,
  min_band_strikes: payload.minBandStrikes,
  min_band_prn_strikes: payload.minBandPrnStrikes,
  strike_range: payload.strikeRange,
  no_retry_full_chain: payload.noRetryFullChain,
  no_sat_expiry_fallback: payload.noSatExpiryFallback,
  threads: payload.threads,
  prefer_bidask: payload.preferBidask,
  min_trade_count: payload.minTradeCount,
  min_volume: payload.minVolume,
  min_chain_used_hard: payload.minChainUsedHard,
  max_rel_spread_median_hard: payload.maxRelSpreadMedianHard,
  hard_drop_close_fallback: payload.hardDropCloseFallback,
  min_prn_train: payload.minPrnTrain,
  max_prn_train: payload.maxPrnTrain,
  no_split_adjust: payload.noSplitAdjust,
  dividend_source: payload.dividendSource,
  dividend_lookback_days: payload.dividendLookbackDays,
  dividend_yield_default: payload.dividendYieldDefault,
  no_forward_moneyness: payload.noForwardMoneyness,
  no_group_weights: payload.noGroupWeights,
  no_ticker_weights: payload.noTickerWeights,
  no_soft_quality_weight: payload.noSoftQualityWeight,
  rv_lookback_days: payload.rvLookbackDays,
  ticker_reweight_mode: payload.tickerReweightMode,
  ticker_reweight_alpha_min: payload.tickerReweightAlphaMin,
  ticker_reweight_alpha_max: payload.tickerReweightAlphaMax,
  trade_focus_beta: payload.tradeFocusBeta,
  trade_focus_tickers: payload.tradeFocusTickers,
  cache: payload.cache,
  write_drops: payload.writeDrops,
  sanity_report: payload.sanityReport,
  sanity_drop: payload.sanityDrop,
  sanity_abs_logm_max: payload.sanityAbsLogmMax,
  sanity_k_over_s_min: payload.sanityKOverSMin,
  sanity_k_over_s_max: payload.sanityKOverSMax,
  verbose_skips: payload.verboseSkips,
});

export async function runDataset(
  payload: DatasetRunRequest,
): Promise<DatasetRunResponse> {
  const response = await fetch(`${API_BASE}/datasets/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(datasetRequestBody(payload)),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Dataset request failed (${response.status}): ${detail || "unknown error"}`,
    );
  }

  return response.json();
}

export async function startDatasetJob(
  payload: DatasetRunRequest,
): Promise<DatasetJobStatus> {
  const response = await fetch(`${API_BASE}/datasets/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(datasetRequestBody(payload)),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Dataset job start failed (${response.status}): ${detail || "unknown error"}`,
    );
  }

  return response.json();
}

export async function renameDatasetRun(
  runDir: string,
  newName: string,
): Promise<DatasetRunSummary> {
  const response = await fetch(`${API_BASE}/datasets/runs/rename`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_dir: runDir, new_name: newName }),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Rename run failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function listDatasetRuns(): Promise<DatasetListResponse> {
  const response = await fetch(`${API_BASE}/datasets/runs`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Dataset runs request failed (${response.status}): ${detail || "unknown error"}`,
    );
  }

  return response.json();
}

export async function backfillOptionChainDataset(
  payload: DatasetBackfillRequest,
): Promise<DatasetBackfillResponse> {
  const response = await fetch(`${API_BASE}/datasets/runs/backfill`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      run_dir: payload.runDir,
      polymarket_run_id: payload.polymarketRunId,
      allow_defaults: payload.allowDefaults ?? false,
    }),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Dataset backfill failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

const datasetCleanupRequestBody = (payload: DatasetCleanupRequest) => ({
  run_dir: payload.runDir,
  criteria: {
    quality_buckets: payload.criteria?.qualityBuckets ?? [],
    min_quality_issue_count:
      payload.criteria && "minQualityIssueCount" in payload.criteria
        ? payload.criteria.minQualityIssueCount ?? null
        : null,
    flag_columns: payload.criteria?.flagColumns ?? [],
    flag_match_mode: payload.criteria?.flagMatchMode,
    min_rel_spread_median: payload.criteria?.minRelSpreadMedian,
    max_n_chain_used: payload.criteria?.maxNChainUsed,
  },
  allow_defaults: payload.allowDefaults ?? false,
});

export async function previewDatasetCleanup(
  payload: DatasetCleanupRequest,
): Promise<DatasetCleanupPreviewResponse> {
  const response = await fetch(`${API_BASE}/datasets/runs/cleanup/preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(datasetCleanupRequestBody(payload)),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Dataset cleanup preview failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function applyDatasetCleanup(
  payload: DatasetCleanupRequest,
): Promise<DatasetCleanupResponse> {
  const response = await fetch(`${API_BASE}/datasets/runs/cleanup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(datasetCleanupRequestBody(payload)),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Dataset cleanup failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function previewDatasetFile(
  path: string,
  mode: "head" | "tail",
  limit = 20,
): Promise<DatasetPreviewResponse> {
  const params = new URLSearchParams({
    path,
    mode,
    limit: limit.toString(),
  });
  const response = await fetch(
    `${API_BASE}/datasets/runs/preview?${params.toString()}`,
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Dataset preview failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function auditDatasetFile(
  path: string,
): Promise<DatasetAuditResponse> {
  const params = new URLSearchParams({ path });
  const response = await fetch(
    `${API_BASE}/datasets/runs/audit?${params.toString()}`,
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Dataset audit failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function getDatasetRowDetail(
  path: string,
  rowId: string,
): Promise<DatasetRowDetailResponse> {
  const params = new URLSearchParams({ path, row_id: rowId });
  const response = await fetch(
    `${API_BASE}/datasets/runs/row?${params.toString()}`,
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Dataset row lookup failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function deleteDatasetRun(
  runDir: string,
): Promise<DatasetRunSummary> {
  const params = new URLSearchParams({ run_dir: runDir });
  const response = await fetch(
    `${API_BASE}/datasets/runs?${params.toString()}`,
    { method: "DELETE" },
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Dataset delete failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function getDatasetJob(jobId: string): Promise<DatasetJobStatus> {
  const response = await fetch(`${API_BASE}/datasets/jobs/${jobId}`, {
    method: "GET",
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Dataset job not found (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function killDatasetJob(jobId: string): Promise<DatasetJobStatus> {
  const response = await fetch(`${API_BASE}/datasets/jobs/${jobId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Failed to cancel dataset job (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}
