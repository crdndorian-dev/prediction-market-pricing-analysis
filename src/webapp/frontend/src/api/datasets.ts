export type DatasetRunRequest = {
  outDir?: string;
  outName?: string;
  dropsName?: string;
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
  output_file: string | null;
  drops_file: string | null;
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

export type DatasetJobStatus = {
  job_id: string;
  status: "queued" | "running" | "finished" | "failed" | "cancelled";
  progress: DatasetJobProgress | null;
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
  last_modified?: string | null;
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

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const datasetRequestBody = (payload: DatasetRunRequest) => ({
  out_dir: payload.outDir,
  out_name: payload.outName,
  drops_name: payload.dropsName,
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

export async function fetchDatasetRuns(): Promise<DatasetListResponse> {
  const response = await fetch(`${API_BASE}/datasets/runs`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Failed to list dataset runs (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function previewDatasetFile(
  path: string,
  options?: { limit?: number; mode?: "head" | "tail" },
): Promise<DatasetPreviewResponse> {
  const params = new URLSearchParams({ path });
  params.set("limit", String(options?.limit ?? 20));
  params.set("mode", options?.mode ?? "head");
  const response = await fetch(`${API_BASE}/datasets/runs/preview?${params}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Failed to preview dataset (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function deleteDatasetRun(
  runDir: string,
): Promise<DatasetRunSummary> {
  const params = new URLSearchParams({ run_dir: runDir });
  const response = await fetch(`${API_BASE}/datasets/runs?${params}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Failed to delete dataset run (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}
