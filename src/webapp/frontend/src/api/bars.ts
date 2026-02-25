/**
 * API client for bar history data
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export type ViewMode = "decision_time" | "full_history";

export interface BarDataPoint {
  timestamp: string;
  timestamp_ms: number;
  price: number;
  open?: number | null;
  high?: number | null;
  low?: number | null;
  close?: number | null;
  volume?: number | null;
}

export interface BarsResponse {
  run_id?: string | null;
  market_id?: string | null;
  ticker?: string | null;
  view_mode: ViewMode;
  time_min?: string | null;
  time_max?: string | null;
  total_points: number;
  returned_points: number;
  downsampled: boolean;
  bars: BarDataPoint[];
  metadata: Record<string, any>;
}

export interface BarRun {
  run_id: string;
  has_price_history: boolean;
  price_history_size: number;
  has_manifest: boolean;
  is_active?: boolean;
  manifest?: Record<string, any>;
}

export interface BarsRunListResponse {
  runs: BarRun[];
}

export interface TradingWeek {
  start_date: string;
  end_date: string;
}

export interface TradingWeeksResponse {
  run_id?: string | null;
  ticker: string;
  weeks: TradingWeek[];
  metadata?: Record<string, any>;
}

export interface GetBarsParams {
  runId?: string;
  marketId?: string;
  ticker?: string;
  timeMin?: string;
  timeMax?: string;
  maxPoints?: number;
  viewMode?: ViewMode;
}

/**
 * Fetch bar history data with time-safety and performance optimizations
 */
export async function getBars(params: GetBarsParams): Promise<BarsResponse> {
  const searchParams = new URLSearchParams();

  if (params.runId) searchParams.set("run_id", params.runId);
  if (params.marketId) searchParams.set("market_id", params.marketId);
  if (params.ticker) searchParams.set("ticker", params.ticker);
  if (params.timeMin) searchParams.set("time_min", params.timeMin);
  if (params.timeMax) searchParams.set("time_max", params.timeMax);
  if (params.maxPoints) searchParams.set("max_points", params.maxPoints.toString());
  if (params.viewMode) searchParams.set("view_mode", params.viewMode);

  const response = await fetch(`${API_BASE}/bars?${searchParams.toString()}`);

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to fetch bars: ${error}`);
  }

  return response.json();
}

// ---------------------------------------------------------------------------
// By-strike types (BacktestsPage)
// ---------------------------------------------------------------------------

export interface StrikeSeries {
  strike: number;
  strike_label: string;
  market_id?: string | null;
  event_slug?: string | null;
  total_points: number;
  returned_points: number;
  bars: BarDataPoint[];
}

export interface ByStrikeResponse {
  run_id?: string | null;
  ticker: string;
  token_role: string;
  time_min?: string | null;
  time_max?: string | null;
  view_mode: string;
  strikes: StrikeSeries[];
  metadata: Record<string, any>;
}

export interface GetBarsByStrikeParams {
  ticker: string;
  runId?: string;
  timeMin?: string;
  timeMax?: string;
  tokenRole?: string;
  maxPointsPerStrike?: number;
  viewMode?: ViewMode;
}

/**
 * Fetch bars grouped by strike for a single ticker + date range.
 */
export async function getBarsByStrike(
  params: GetBarsByStrikeParams,
): Promise<ByStrikeResponse> {
  const sp = new URLSearchParams();
  sp.set("ticker", params.ticker);
  if (params.runId) sp.set("run_id", params.runId);
  if (params.timeMin) sp.set("time_min", params.timeMin);
  if (params.timeMax) sp.set("time_max", params.timeMax);
  if (params.tokenRole) sp.set("token_role", params.tokenRole);
  if (params.maxPointsPerStrike)
    sp.set("max_points_per_strike", params.maxPointsPerStrike.toString());
  if (params.viewMode) sp.set("view_mode", params.viewMode);

  const response = await fetch(`${API_BASE}/bars/by-strike?${sp.toString()}`);
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

/**
 * List available pipeline runs with bar history data
 */
export async function listBarRuns(): Promise<BarsRunListResponse> {
  const response = await fetch(`${API_BASE}/bars/runs`);

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Failed to list bar runs: ${error}`);
  }

  return response.json();
}

/**
 * List available trading weeks (Mon-Fri) for a ticker/run.
 */
export async function listTradingWeeks(
  params: { ticker: string; runId?: string },
): Promise<TradingWeeksResponse> {
  const sp = new URLSearchParams();
  sp.set("ticker", params.ticker);
  if (params.runId) sp.set("run_id", params.runId);

  const response = await fetch(`${API_BASE}/bars/trading-weeks?${sp.toString()}`);
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

// ---------------------------------------------------------------------------
// pRN overlay types (for BacktestsPage chart overlay)
// ---------------------------------------------------------------------------

export interface PrnPoint {
  asof_date: string;
  asof_date_ms: number;
  dte: number;
  pRN: number;
}

export interface PrnStrikeSeries {
  strike: number;
  strike_label: string;
  points: PrnPoint[];
}

export interface PrnOverlayResponse {
  ticker: string;
  dataset_path?: string | null;
  strikes: PrnStrikeSeries[];
  metadata: Record<string, any>;
}

export interface GetPrnOverlayParams {
  ticker: string;
  timeMin?: string;
  timeMax?: string;
}

export interface GetPrnOverlayThetaParams {
  ticker: string;
  timeMin?: string;
  timeMax?: string;
  dteList?: number[];
  strikes?: number[];
}

/**
 * Fetch pRN overlay data for charting alongside Polymarket price series.
 */
export async function getPrnOverlay(
  params: GetPrnOverlayParams,
): Promise<PrnOverlayResponse> {
  const sp = new URLSearchParams();
  sp.set("ticker", params.ticker);
  if (params.timeMin) sp.set("time_min", params.timeMin);
  if (params.timeMax) sp.set("time_max", params.timeMax);

  const response = await fetch(`${API_BASE}/bars/prn-overlay?${sp.toString()}`);
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

/**
 * Fetch pRN overlay data computed on-demand via Theta Terminal.
 * Uses relaxed thresholds to recover 1DTE observations that stored data often lacks.
 */
export async function getPrnOverlayTheta(
  params: GetPrnOverlayThetaParams,
): Promise<PrnOverlayResponse> {
  const sp = new URLSearchParams();
  sp.set("ticker", params.ticker);
  if (params.timeMin) sp.set("time_min", params.timeMin);
  if (params.timeMax) sp.set("time_max", params.timeMax);
  if (params.dteList && params.dteList.length > 0)
    sp.set("dte_list", params.dteList.join(","));
  if (params.strikes && params.strikes.length > 0)
    sp.set("strikes", params.strikes.join(","));

  const response = await fetch(
    `${API_BASE}/bars/prn-overlay/theta?${sp.toString()}`,
  );
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
