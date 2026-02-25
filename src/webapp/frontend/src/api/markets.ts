const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export type MarketsProgress = {
  stage?: string | null;
  current: number;
  total: number;
};

export type MarketsRefreshRequest = {
  week_friday?: string | null;
  tickers?: string[] | null;
  run_id?: string | null;
  force_refresh?: boolean;
};

export type MarketsRefreshResult = {
  ok: boolean;
  run_id?: string | null;
  week_friday?: string | null;
  run_dir?: string | null;
  stdout: string;
  stderr: string;
  duration_s: number;
  command: string[];
};

export type MarketsJobStatus = {
  job_id: string;
  status: "queued" | "running" | "finished" | "failed";
  progress?: MarketsProgress | null;
  result?: MarketsRefreshResult | null;
  error?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
};

export type MarketsSummaryItem = {
  ticker: string;
  threshold: number;
  market_id?: string | null;
  event_id?: string | null;
  event_endDate?: string | null;
  points: number;
  last_timestamp_utc?: string | null;
  has_polymarket: boolean;
  has_prn: boolean;
};

export type MarketsSummaryResponse = {
  run_id?: string | null;
  week_friday: string;
  week_monday?: string | null;
  week_sunday?: string | null;
  last_refresh_utc?: string | null;
  trading_universe_tickers?: string[];
  markets: MarketsSummaryItem[];
};

export type MarketsSeriesPoint = {
  timestamp_utc: string;
  polymarket_buy?: number | null;   // kept for backward compat
  polymarket_bid?: number | null;   // best bid (sell YES)
  polymarket_ask?: number | null;   // best ask (buy YES); service falls back to polymarket_buy
  pRN?: number | null;
  spot?: number | null;
};

export type MarketsSeriesResponse = {
  run_id?: string | null;
  ticker: string;
  threshold: number;
  week_friday: string;
  market_id?: string | null;
  event_id?: string | null;
  points: MarketsSeriesPoint[];
  metadata?: Record<string, unknown> | null;
};

export type MarketsSeriesByTickerResponse = {
  run_id?: string | null;
  ticker: string;
  week_friday: string;
  strikes: MarketsSeriesResponse[];
  metadata?: Record<string, unknown> | null;
};

async function handleResponse(response: Response) {
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
  return response.json();
}

export async function startMarketsRefresh(
  payload: MarketsRefreshRequest,
): Promise<MarketsJobStatus> {
  const response = await fetch(`${API_BASE}/markets/refresh`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse(response);
}

export async function getMarketsJob(jobId: string): Promise<MarketsJobStatus> {
  const response = await fetch(`${API_BASE}/markets/jobs/${jobId}`);
  return handleResponse(response);
}

export async function getMarketsSummary(params: {
  weekFriday?: string;
  runId?: string;
}): Promise<MarketsSummaryResponse> {
  const sp = new URLSearchParams();
  if (params.weekFriday) sp.set("week_friday", params.weekFriday);
  if (params.runId) sp.set("run_id", params.runId);
  const response = await fetch(`${API_BASE}/markets/summary?${sp.toString()}`);
  return handleResponse(response);
}

export async function getMarketsSeries(params: {
  ticker: string;
  threshold: number;
  weekFriday?: string;
  runId?: string;
}): Promise<MarketsSeriesResponse> {
  const sp = new URLSearchParams();
  sp.set("ticker", params.ticker);
  sp.set("threshold", params.threshold.toString());
  if (params.weekFriday) sp.set("week_friday", params.weekFriday);
  if (params.runId) sp.set("run_id", params.runId);
  const response = await fetch(`${API_BASE}/markets/series?${sp.toString()}`);
  return handleResponse(response);
}

export async function getMarketsSeriesByTicker(params: {
  ticker: string;
  weekFriday?: string;
  runId?: string;
}): Promise<MarketsSeriesByTickerResponse> {
  const sp = new URLSearchParams();
  sp.set("ticker", params.ticker);
  if (params.weekFriday) sp.set("week_friday", params.weekFriday);
  if (params.runId) sp.set("run_id", params.runId);
  const response = await fetch(`${API_BASE}/markets/series/by-ticker?${sp.toString()}`);
  return handleResponse(response);
}
