import { apiFetch } from "./http";

export type AnalysisCoverageSummary = {
  latest_refresh_id?: string | null;
  latest_refresh_at?: string | null;
  volume_authoritative: boolean;
  market_day_count: number;
  stock_day_count: number;
  global_day_count: number;
  trade_coverage_ratio?: number | null;
  stock_count: number;
  active_market_count: number;
};

export type AnalysisHeadlineMetric = {
  label: string;
  value?: number | string | null;
  delta?: number | null;
  suffix?: string | null;
};

export type AnalysisAlert = {
  level: "info" | "warning" | "critical";
  title: string;
  detail: string;
};

export type AnalysisSeriesPoint = {
  date: string;
  value?: number | null;
  value_2?: number | null;
  value_3?: number | null;
  label?: string | null;
  metadata?: Record<string, unknown>;
};

export type AnalysisHistogramBin = {
  start: number;
  end: number;
  count: number;
};

export type AnalysisThresholdRow = {
  trade_date_ny: string;
  ticker?: string | null;
  variable_name: string;
  scope_used: string;
  coverage_class?: string | null;
  upper_cap?: number | null;
  q95?: number | null;
  q99?: number | null;
  q995?: number | null;
  sample_count?: number | null;
  sparse_flag: boolean;
  authoritative_flag: boolean;
};

export type AnalysisOverviewResponse = {
  coverage: AnalysisCoverageSummary;
  headline_metrics: AnalysisHeadlineMetric[];
  recent_trends: AnalysisSeriesPoint[];
  alerts: AnalysisAlert[];
  available_tickers: string[];
};

export type AnalysisVolumeResponse = {
  coverage: AnalysisCoverageSummary;
  selected_variable: string;
  selected_scope_level: string;
  total_volume_series: AnalysisSeriesPoint[];
  per_stock_series: AnalysisSeriesPoint[];
  hist_raw: AnalysisHistogramBin[];
  hist_log: AnalysisHistogramBin[];
  outlier_diagnostics: Array<Record<string, unknown>>;
  threshold_rows: AnalysisThresholdRow[];
  clipped_share?: number | null;
};

export type AnalysisStructureResponse = {
  coverage: AnalysisCoverageSummary;
  stock_concentration: AnalysisSeriesPoint[];
  active_market_counts: AnalysisSeriesPoint[];
  stock_participation: AnalysisSeriesPoint[];
  lifecycle_summary: Array<Record<string, unknown>>;
};

export type AnalysisPriceBehaviorResponse = {
  coverage: AnalysisCoverageSummary;
  close_probability_distribution: AnalysisHistogramBin[];
  volatility_series: AnalysisSeriesPoint[];
  expiry_behavior: AnalysisSeriesPoint[];
  convergence_table: Array<Record<string, unknown>>;
};

export type AnalysisDriftResponse = {
  coverage: AnalysisCoverageSummary;
  selected_variable: string;
  rolling_statistics: AnalysisSeriesPoint[];
  rolling_quantiles: AnalysisSeriesPoint[];
  drift_rows: Array<Record<string, unknown>>;
  break_rows: Array<Record<string, unknown>>;
  alerts: AnalysisAlert[];
};

export type AnalysisTableResponse = {
  table: string;
  page: number;
  page_size: number;
  total_rows: number;
  rows: Array<{ values: Record<string, unknown> }>;
};

export type ResearchNote = {
  note_id: number;
  title: string;
  body: string;
  author?: string | null;
  tags: string[];
  pinned: boolean;
  created_at_utc: string;
  updated_at_utc: string;
};

export type ResearchNotesResponse = {
  notes: ResearchNote[];
};

export type AnalysisRefreshRequest = {
  run_ids?: string[] | null;
  skip_trade_backfill?: boolean;
  force_full_rebuild?: boolean;
  notes_author?: string | null;
};

export type AnalysisRefreshResult = {
  ok: boolean;
  refresh_id?: string | null;
  stdout: string;
  stderr: string;
  duration_s: number;
  command: string[];
};

export type AnalysisProgress = {
  stage?: string | null;
  current: number;
  total: number;
  detail?: string | null;
};

export type AnalysisJobStatus = {
  job_id: string;
  status: "queued" | "running" | "finished" | "failed" | "cancelled";
  progress?: AnalysisProgress | null;
  result?: AnalysisRefreshResult | null;
  error?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
};

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    let detail = text;
    try {
      const json = JSON.parse(text);
      detail = json.detail || text;
    } catch {
      // keep raw text
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export async function fetchAnalysisOverview(): Promise<AnalysisOverviewResponse> {
  const response = await apiFetch("/analysis/overview");
  return handleResponse(response);
}

export async function fetchAnalysisVolume(params?: {
  ticker?: string;
  variable_name?: string;
  limit?: number;
}): Promise<AnalysisVolumeResponse> {
  const sp = new URLSearchParams();
  if (params?.ticker) sp.set("ticker", params.ticker);
  if (params?.variable_name) sp.set("variable_name", params.variable_name);
  if (params?.limit) sp.set("limit", String(params.limit));
  const response = await apiFetch(`/analysis/volume?${sp.toString()}`);
  return handleResponse(response);
}

export async function fetchAnalysisStructure(params?: {
  limit?: number;
}): Promise<AnalysisStructureResponse> {
  const sp = new URLSearchParams();
  if (params?.limit) sp.set("limit", String(params.limit));
  const response = await apiFetch(`/analysis/structure?${sp.toString()}`);
  return handleResponse(response);
}

export async function fetchAnalysisPriceBehavior(params?: {
  ticker?: string;
  limit?: number;
}): Promise<AnalysisPriceBehaviorResponse> {
  const sp = new URLSearchParams();
  if (params?.ticker) sp.set("ticker", params.ticker);
  if (params?.limit) sp.set("limit", String(params.limit));
  const response = await apiFetch(`/analysis/price-behavior?${sp.toString()}`);
  return handleResponse(response);
}

export async function fetchAnalysisDrift(params?: {
  ticker?: string;
  variable_name?: string;
  limit?: number;
}): Promise<AnalysisDriftResponse> {
  const sp = new URLSearchParams();
  if (params?.ticker) sp.set("ticker", params.ticker);
  if (params?.variable_name) sp.set("variable_name", params.variable_name);
  if (params?.limit) sp.set("limit", String(params.limit));
  const response = await apiFetch(`/analysis/drift?${sp.toString()}`);
  return handleResponse(response);
}

export async function fetchAnalysisTable(params: {
  table: string;
  page?: number;
  page_size?: number;
  ticker?: string;
  variable_name?: string;
}): Promise<AnalysisTableResponse> {
  const sp = new URLSearchParams();
  sp.set("table", params.table);
  if (params.page) sp.set("page", String(params.page));
  if (params.page_size) sp.set("page_size", String(params.page_size));
  if (params.ticker) sp.set("ticker", params.ticker);
  if (params.variable_name) sp.set("variable_name", params.variable_name);
  const response = await apiFetch(`/analysis/tables?${sp.toString()}`);
  return handleResponse(response);
}

export async function exportAnalysisTableCsv(params: {
  table: string;
  ticker?: string;
  variable_name?: string;
}): Promise<string> {
  const sp = new URLSearchParams();
  sp.set("table", params.table);
  if (params.ticker) sp.set("ticker", params.ticker);
  if (params.variable_name) sp.set("variable_name", params.variable_name);
  const response = await apiFetch(`/analysis/tables/export?${sp.toString()}`);
  if (!response.ok) {
    const text = await response.text();
    let detail = text;
    try {
      const json = JSON.parse(text);
      detail = json.detail || text;
    } catch {
      // keep raw text
    }
    throw new Error(detail);
  }
  return response.text();
}

export async function startAnalysisRefresh(
  payload: AnalysisRefreshRequest,
): Promise<AnalysisJobStatus> {
  const response = await apiFetch("/analysis/refresh", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse(response);
}

export async function getAnalysisJob(jobId: string): Promise<AnalysisJobStatus> {
  const response = await apiFetch(`/analysis/jobs/${jobId}`);
  return handleResponse(response);
}

export async function fetchResearchNotes(): Promise<ResearchNotesResponse> {
  const response = await apiFetch("/analysis/notes");
  return handleResponse(response);
}

export async function createResearchNote(payload: {
  title: string;
  body: string;
  author?: string | null;
  tags?: string[];
  pinned?: boolean;
}): Promise<ResearchNote> {
  const response = await apiFetch("/analysis/notes", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse(response);
}
