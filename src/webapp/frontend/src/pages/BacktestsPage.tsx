import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";

import {
  getBarsByStrike,
  listBarRuns,
  listTradingWeeks,
  getPrnOverlayTheta,
  type ByStrikeResponse,
  type BarDataPoint,
  type StrikeSeries,
  type BarRun,
  type TradingWeek,
  type PrnOverlayResponse,
  type PrnPoint,
} from "../api/bars";
import {
  getMarketsSeriesByTicker,
  type MarketsSeriesByTickerResponse,
  type MarketsSeriesResponse,
} from "../api/markets";
import PipelineStatusCard from "../components/PipelineStatusCard";
import { useAnyJobRunning } from "../contexts/jobGuard";
import "./BacktestsPage.css";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TRADING_UNIVERSE_TICKERS = [
  "AAPL",
  "GOOGL",
  "MSFT",
  "META",
  "AMZN",
  "PLTR",
  "NVDA",
  "TSLA",
  "NFLX",
  "OPEN",
];

const MAX_RANGE_DAYS = 5;
const DEFAULT_TIMEZONE = "America/New_York";
const REQUIRED_PRN_DTES = [1, 2, 3, 4];
const GAP_BREAK_MS = 4 * 3600_000; // break chart line when gap > 4 hours
const SPARSE_DATA_THRESHOLD = 0.3; // warn if points < 30% of expected hourly count

// SVG chart geometry (viewBox units) — matching MarketsPage
const W = 800;
const H = 400;
const PL = 52;  // left padding — y-axis labels
const PR = 16;  // right padding
const PT = 20;  // top padding
const PB = 44;  // bottom padding — x-axis labels

const Y_LABELS = [0, 0.25, 0.5, 0.75, 1.0];
const X_TICK_COUNT = 5;
const PLOT_W = W - PL - PR;
const PLOT_H = H - PT - PB;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Format a number as a short price label. */
const formatPrice = (value: number) => {
  if (!Number.isFinite(value)) return "--";
  if (value < 0.01) return value.toFixed(4);
  return value.toFixed(3);
};

/** Short date label for chart axes. */
const formatShortDate = (value: number) => {
  if (!Number.isFinite(value)) return "--";
  return new Date(value).toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
  });
};

const formatUtcLabel = (ms: number): string =>
  new Date(ms).toISOString().replace("T", " ").slice(0, 16) + "Z";

const formatLocalLabel = (ms: number): string =>
  new Date(ms).toLocaleString("en-US", {
    timeZone: DEFAULT_TIMEZONE,
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });

const getMarketQualityBadges = (series: StrikeSeries): string[] => {
  const quality = series.market_quality;
  if (!quality) return [];
  const badges: string[] = [];
  if (quality.flag_not_relevant) badges.push("Not relevant");
  if (quality.flag_prn_missing) badges.push("pRN missing");
  if (quality.flag_pm_no_trade_history) badges.push("No trade");
  if (quality.flag_pm_no_recent_trade) badges.push("Stale trade");
  if (quality.flag_pm_stale_prices) badges.push("Stale");
  if (quality.flag_extreme_otm) badges.push("Extreme OTM");
  return badges.slice(0, 3);
};

/** Two-line label for the SVG x-axis tick (displayed in UTC). */
function formatXTick(ms: number): { weekday: string; dateLine: string } {
  const date = new Date(ms);
  const weekday = date.toLocaleString("en-US", {
    timeZone: "UTC",
    weekday: "short",
  });
  const dateLine = date.toLocaleString("en-US", {
    timeZone: "UTC",
    month: "short",
    day: "numeric",
  });
  return { weekday, dateLine };
}

const buildStrikeKey = (series: StrikeSeries) =>
  `${strikeKey(series.strike)}:${series.market_id ?? "unknown"}`;

/** Canonical strike key — all strike matching goes through this single function. */
const strikeKey = (strike: number): string =>
  strike === Math.floor(strike) ? String(Math.floor(strike)) : strike.toFixed(2);

const MIN_DTES_WITH_HOLIDAYS = 2;

const hasRequiredPrnDtes = (points: PrnPoint[], holidaysInRange = 0): boolean => {
  if (!points || points.length === 0) return false;
  const seen = new Set<number>();
  for (const p of points) {
    if (Number.isFinite(p.dte)) seen.add(p.dte);
  }
  if (holidaysInRange > 0) {
    const matched = REQUIRED_PRN_DTES.filter((dte) => seen.has(dte)).length;
    return matched >= MIN_DTES_WITH_HOLIDAYS;
  }
  return REQUIRED_PRN_DTES.every((dte) => seen.has(dte));
};


const formatUtcDate = (date: Date) => date.toISOString().slice(0, 10);

const parseUtcDate = (value: string): Date | null => {
  if (!value) return null;
  const date = new Date(`${value}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return null;
  return date;
};

const addDaysUtc = (date: Date, delta: number): Date => {
  const next = new Date(date);
  next.setUTCDate(next.getUTCDate() + delta);
  return next;
};

const startOfWeekUtcMonday = (date: Date): Date => {
  const day = date.getUTCDay(); // 0=Sun ... 6=Sat
  const daysSinceMonday = (day + 6) % 7;
  return addDaysUtc(date, -daysSinceMonday);
};

const startOfMonthUtc = (date: Date): Date =>
  new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), 1));

const monthKey = (date: Date): number => date.getUTCFullYear() * 12 + date.getUTCMonth();

const formatMonthYear = (date: Date): string =>
  date.toLocaleDateString(undefined, {
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  });

const formatRangeLabel = (start: string, end: string): string => {
  const s = parseUtcDate(start);
  const e = parseUtcDate(end);
  if (!s || !e) return `${start} – ${end}`;
  const options: Intl.DateTimeFormatOptions = {
    month: "short",
    day: "2-digit",
    year: "numeric",
    timeZone: "UTC",
  };
  return `${s.toLocaleDateString(undefined, options)} – ${e.toLocaleDateString(undefined, options)}`;
};

type CalendarWeek = {
  weekStart: Date;
  days: Date[];
};

const buildCalendarWeeks = (month: Date): CalendarWeek[] => {
  const firstOfMonth = startOfMonthUtc(month);
  const lastOfMonth = new Date(Date.UTC(month.getUTCFullYear(), month.getUTCMonth() + 1, 0));
  const start = startOfWeekUtcMonday(firstOfMonth);
  const end = addDaysUtc(startOfWeekUtcMonday(lastOfMonth), 4);
  const weeks: CalendarWeek[] = [];

  let cursor = start;
  while (cursor.getTime() <= end.getTime()) {
    const days = [0, 1, 2, 3, 4].map((offset) => addDaysUtc(cursor, offset));
    weeks.push({ weekStart: cursor, days });
    cursor = addDaysUtc(cursor, 7);
  }
  return weeks;
};

/** Validate that the selected range is <= MAX_RANGE_DAYS days. */
function validateDateRange(
  start: string,
  end: string,
): { ok: boolean; error: string | null; days: number } {
  if (!start || !end) return { ok: false, error: null, days: 0 };
  const s = new Date(`${start}T00:00:00Z`).getTime();
  const e = new Date(`${end}T23:59:59Z`).getTime();
  if (Number.isNaN(s) || Number.isNaN(e))
    return { ok: false, error: "Invalid date.", days: 0 };
  if (e < s)
    return { ok: false, error: "End date must be after start date.", days: 0 };
  const days = Math.ceil((e - s) / (1000 * 60 * 60 * 24));
  if (days > MAX_RANGE_DAYS)
    return {
      ok: false,
      error: `Range is ${days} days — max allowed is ${MAX_RANGE_DAYS} days (1 trading week).`,
      days,
    };
  return { ok: true, error: null, days };
}

// ---------------------------------------------------------------------------
// BacktestChart (SVG, one per strike)
// ---------------------------------------------------------------------------

type MarketChartPoint = {
  timeMs: number;
  polymarketMid?: number | null;
  polymarketBid?: number | null;
  polymarketAsk?: number | null;
  prn?: number | null;
};
type PrnChartPoint = { timeMs: number; price: number; dte: number };
type ChartRange = { min: number; max: number };

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

function buildChartPointsFromMarkets(series: MarketsSeriesResponse): MarketChartPoint[] {
  return series.points
    .map((p) => ({
      timeMs: new Date(p.timestamp_utc).getTime(),
      polymarketMid: p.polymarket_mid ?? p.polymarket_buy ?? null,
      polymarketBid: p.polymarket_bid ?? null,
      polymarketAsk: p.polymarket_ask ?? null,
      prn: null,
    }))
    .filter((p) => Number.isFinite(p.timeMs));
}

function buildChartPointsFromBars(
  bars: BarDataPoint[],
): { points: MarketChartPoint[]; dropped: number } {
  const points: MarketChartPoint[] = [];
  let dropped = 0;
  for (const bar of bars) {
    if (!Number.isFinite(bar.timestamp_ms) || !Number.isFinite(bar.price)) {
      dropped += 1;
      continue;
    }
    points.push({
      timeMs: bar.timestamp_ms,
      polymarketMid: clamp01(bar.price),
      polymarketBid: null,
      polymarketAsk: null,
      prn: null,
    });
  }
  points.sort((a, b) => a.timeMs - b.timeMs);
  return { points, dropped };
}

function buildPath(
  points: MarketChartPoint[],
  accessor: (p: MarketChartPoint) => number | null | undefined,
  scaleX: (t: number) => number,
  scaleY: (v: number) => number,
): string {
  const filtered = points.filter((p) => {
    const v = accessor(p);
    return v !== null && v !== undefined && Number.isFinite(v);
  });
  if (filtered.length < 2) return "";
  return filtered
    .map((p, i) => {
      const cmd =
        i === 0
          ? "M"
          : p.timeMs - filtered[i - 1].timeMs > GAP_BREAK_MS
            ? "M"
            : "L";
      return `${cmd}${scaleX(p.timeMs).toFixed(1)},${scaleY(accessor(p) as number).toFixed(1)}`;
    })
    .join(" ");
}

function buildPrnDotsPath(
  dots: PrnChartPoint[],
  scaleX: (t: number) => number,
  scaleY: (v: number) => number,
): string {
  if (dots.length < 2) return "";
  const sorted = [...dots].sort((a, b) => a.timeMs - b.timeMs);
  return sorted
    .map(
      (p, i) =>
        `${i === 0 ? "M" : "L"}${scaleX(p.timeMs).toFixed(1)},${scaleY(p.price).toFixed(1)}`,
    )
    .join(" ");
}

function findNearestPoint(points: MarketChartPoint[], targetTime: number): MarketChartPoint | null {
  if (!points.length) return null;
  const sorted = points.slice().sort((a, b) => a.timeMs - b.timeMs);
  let left = 0;
  let right = sorted.length - 1;
  while (left <= right) {
    const mid = Math.floor((left + right) / 2);
    if (sorted[mid].timeMs === targetTime) return sorted[mid];
    if (sorted[mid].timeMs < targetTime) left = mid + 1;
    else right = mid - 1;
  }
  const candidates = [
    sorted[Math.max(0, right)],
    sorted[Math.min(sorted.length - 1, left)],
  ];
  return candidates.reduce((best, c) =>
    c && Math.abs(c.timeMs - targetTime) < Math.abs(best.timeMs - targetTime) ? c : best,
  );
}

function BacktestChart({
  points,
  prnDots,
  xRange,
}: {
  points: MarketChartPoint[];
  prnDots: PrnChartPoint[];
  xRange?: ChartRange | null;
}) {
  const filteredPoints = useMemo(() => {
    if (!xRange) return points;
    return points.filter((p) => p.timeMs >= xRange.min && p.timeMs <= xRange.max);
  }, [points, xRange]);

  const filteredDots = useMemo(() => {
    if (!prnDots) return [];
    if (!xRange) return prnDots;
    return prnDots.filter((p) => p.timeMs >= xRange.min && p.timeMs <= xRange.max);
  }, [prnDots, xRange]);

  const minTime = useMemo(() => {
    if (xRange) return xRange.min;
    const values = [
      ...filteredPoints.map((p) => p.timeMs),
      ...filteredDots.map((p) => p.timeMs),
    ].filter((v) => Number.isFinite(v));
    return values.length > 0 ? Math.min(...values) : NaN;
  }, [filteredDots, filteredPoints, xRange]);

  const maxTime = useMemo(() => {
    if (xRange) return xRange.max;
    const values = [
      ...filteredPoints.map((p) => p.timeMs),
      ...filteredDots.map((p) => p.timeMs),
    ].filter((v) => Number.isFinite(v));
    return values.length > 0 ? Math.max(...values) : NaN;
  }, [filteredDots, filteredPoints, xRange]);

  const hasTime = Number.isFinite(minTime) && Number.isFinite(maxTime) && minTime < maxTime;
  const hasMid = filteredPoints.some((p) => p.polymarketMid !== null && p.polymarketMid !== undefined);
  const hasBid = filteredPoints.some((p) => p.polymarketBid !== null && p.polymarketBid !== undefined);
  const hasAsk = filteredPoints.some((p) => p.polymarketAsk !== null && p.polymarketAsk !== undefined);
  const hasPrnLine = filteredPoints.some((p) => p.prn !== null && p.prn !== undefined);
  const hasPrnDots = filteredDots.length > 0;

  if (!hasTime || (!hasMid && !hasBid && !hasAsk && !hasPrnLine && !hasPrnDots)) {
    return (
      <div className="preview-placeholder">
        Not enough data ({filteredPoints.length} point{filteredPoints.length !== 1 ? "s" : ""}).
      </div>
    );
  }

  const xScale = useCallback(
    (t: number) => PL + ((t - minTime) / (maxTime - minTime)) * PLOT_W,
    [minTime, maxTime],
  );
  const yScale = useCallback((v: number) => PT + PLOT_H * (1 - clamp01(v)), []);

  const midPath = hasTime ? buildPath(filteredPoints, (p) => p.polymarketMid, xScale, yScale) : "";
  const bidPath = hasTime ? buildPath(filteredPoints, (p) => p.polymarketBid, xScale, yScale) : "";
  const askPath = hasTime ? buildPath(filteredPoints, (p) => p.polymarketAsk, xScale, yScale) : "";
  const prnPath = hasTime ? buildPath(filteredPoints, (p) => p.prn, xScale, yScale) : "";
  const prnDotsPath = hasTime ? buildPrnDotsPath(filteredDots, xScale, yScale) : "";

  const xTicks = useMemo(() => {
    if (!hasTime) return [];
    return Array.from({ length: X_TICK_COUNT }, (_, i) => {
      const t = minTime + (i / (X_TICK_COUNT - 1)) * (maxTime - minTime);
      return { t, ...formatXTick(t) };
    });
  }, [hasTime, minTime, maxTime]);

  const [hovered, setHovered] = useState<MarketChartPoint | null>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);

  const handleMove = (event: React.MouseEvent<SVGSVGElement>) => {
    if (!hasTime || filteredPoints.length === 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const svgX = ((event.clientX - rect.left) / rect.width) * W;
    const ratio = Math.min(1, Math.max(0, (svgX - PL) / PLOT_W));
    const time = minTime + ratio * (maxTime - minTime);
    const nearest = findNearestPoint(filteredPoints, time);
    if (nearest) {
      setHovered(nearest);
      setHoverX(xScale(nearest.timeMs));
    }
  };

  const handleLeave = () => {
    setHovered(null);
    setHoverX(null);
  };

  return (
    <div className="mdc-svg-wrap" onMouseLeave={handleLeave}>
      <svg viewBox={`0 0 ${W} ${H}`} className="mdc-svg" onMouseMove={handleMove}>
        {Y_LABELS.map((v) => (
          <g key={v}>
            <line
              x1={PL} x2={PL + PLOT_W}
              y1={yScale(v)} y2={yScale(v)}
              className={v === 0.5 ? "chart-midline" : "chart-gridline"}
            />
            <text
              x={PL - 6}
              y={yScale(v)}
              className="chart-axis-label"
              textAnchor="end"
              dominantBaseline="middle"
            >
              {v === 0 ? "0" : v === 1 ? "1" : v.toFixed(2)}
            </text>
          </g>
        ))}

        <rect x={PL} y={PT} width={PLOT_W} height={PLOT_H} className="chart-frame" />

        {xTicks.map(({ t, weekday, dateLine }) => (
          <g key={t}>
            <line
              x1={xScale(t)} x2={xScale(t)}
              y1={PT + PLOT_H} y2={PT + PLOT_H + 5}
              className="chart-tick"
            />
            <text
              x={xScale(t)}
              y={PT + PLOT_H + 16}
              className="chart-axis-label"
              textAnchor="middle"
            >
              <tspan x={xScale(t)} dy="0">{weekday}</tspan>
              <tspan x={xScale(t)} dy="14">{dateLine}</tspan>
            </text>
          </g>
        ))}

        {midPath && <path d={midPath} className="chart-line chart-line-mid" />}
        {bidPath && <path d={bidPath} className="chart-line chart-line-bid" />}
        {askPath && <path d={askPath} className="chart-line chart-line-ask" />}
        {prnPath && <path d={prnPath} className="chart-line chart-line-prn" />}
        {prnDotsPath && <path d={prnDotsPath} className="chart-line chart-line-prn-dots" />}

        {filteredDots.map((p, i) => (
          <g key={`prn-dot-${i}`}>
            <circle
              cx={xScale(p.timeMs)}
              cy={yScale(p.price)}
              r={3}
              className="chart-prn-dot"
            />
            <title>
              pRN: {p.price.toFixed(4)} (DTE{p.dte}){" "}
              {p.timeMs ? formatShortDate(p.timeMs) : ""}
            </title>
          </g>
        ))}

        {hoverX !== null && (
          <line
            x1={hoverX} x2={hoverX}
            y1={PT} y2={PT + PLOT_H}
            className="chart-hover"
          />
        )}
      </svg>

      {hovered && (
        <div className="chart-tooltip">
          <div className="chart-tooltip-time">{formatUtcLabel(hovered.timeMs)}</div>
          <div className="chart-tooltip-sub">{formatLocalLabel(hovered.timeMs)} ET</div>
          {hasMid && (
            <div className="chart-tooltip-row">
              <span className="tt-label tt-mid">PM Mid</span>
              <span>{formatPrice(hovered.polymarketMid ?? NaN)}</span>
            </div>
          )}
          {hasBid && (
            <div className="chart-tooltip-row">
              <span className="tt-label tt-bid">PM Bid</span>
              <span>{formatPrice(hovered.polymarketBid ?? NaN)}</span>
            </div>
          )}
          {hasAsk && (
            <div className="chart-tooltip-row">
              <span className="tt-label tt-ask">PM Ask</span>
              <span>{formatPrice(hovered.polymarketAsk ?? NaN)}</span>
            </div>
          )}
          {hasPrnLine && (
            <div className="chart-tooltip-row">
              <span className="tt-label tt-prn">pRN</span>
              <span>{formatPrice(hovered.prn ?? NaN)}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// StrikeCard — one card per strike
// ---------------------------------------------------------------------------

function StrikeCard({
  series,
  ticker,
  marketsSeries,
  prnPoints,
  prnSourceLabel,
  xRange,
}: {
  series: StrikeSeries;
  ticker: string;
  marketsSeries?: MarketsSeriesResponse | null;
  prnPoints?: PrnChartPoint[];
  prnSourceLabel?: string | null;
  xRange?: ChartRange | null;
}) {
  const { points: fallbackPoints, dropped } = useMemo(
    () => buildChartPointsFromBars(series.bars),
    [series.bars],
  );
  const chartPoints = useMemo(
    () => (marketsSeries ? buildChartPointsFromMarkets(marketsSeries) : fallbackPoints),
    [marketsSeries, fallbackPoints],
  );
  const hasOverlayPrn = (prnPoints?.length ?? 0) > 0;
  const displayPoints = useMemo(() => {
    if (!hasOverlayPrn) return chartPoints;
    return chartPoints.map((point) => ({ ...point, prn: null }));
  }, [chartPoints, hasOverlayPrn]);

  const hasMid = displayPoints.some((p) => p.polymarketMid !== null && p.polymarketMid !== undefined);
  const hasBid = displayPoints.some((p) => p.polymarketBid !== null && p.polymarketBid !== undefined);
  const hasAsk = displayPoints.some((p) => p.polymarketAsk !== null && p.polymarketAsk !== undefined);
  const hasPrnLine = displayPoints.some((p) => p.prn !== null && p.prn !== undefined);
  const hasPrnDots = hasOverlayPrn;
  const pointsCount = marketsSeries ? marketsSeries.points.length : fallbackPoints.length;

  const isSparseData = useMemo(() => {
    if (!xRange) return false;
    const rangeDays = (xRange.max - xRange.min) / (1000 * 60 * 60 * 24);
    const expectedHourly = Math.max(1, Math.floor(rangeDays * 24));
    return pointsCount < expectedHourly * SPARSE_DATA_THRESHOLD;
  }, [xRange, pointsCount]);

  const overlaySource = prnSourceLabel
    ? prnSourceLabel.replace(/^pRN source:\s*/i, "")
    : "Theta";
  const prnSourceChip = hasPrnDots
    ? `pRN source: ${overlaySource}`
    : null;
  const marketQualityBadges = getMarketQualityBadges(series);

  return (
    <div className="mdc-wrap">
      <div className="mdc-header">
        <div className="mdc-title">
          {ticker} <span className="mdc-strike">${series.strike_label}</span>
        </div>
        <div className="mdc-legend">
          {hasMid && (
            <span className="mdc-legend-item">
              <span className="mdc-swatch mdc-swatch-mid" />
              PM Mid
            </span>
          )}
          {hasBid && (
            <span className="mdc-legend-item">
              <span className="mdc-swatch mdc-swatch-bid" />
              PM Bid
            </span>
          )}
          {hasAsk && (
            <span className="mdc-legend-item">
              <span className="mdc-swatch mdc-swatch-ask" />
              PM Ask
            </span>
          )}
          {(hasPrnLine || hasPrnDots) && (
            <span className="mdc-legend-item">
              <span className="mdc-swatch mdc-swatch-prn" />
              pRN
            </span>
          )}
        </div>
      </div>
      <BacktestChart
        points={displayPoints}
        prnDots={prnPoints ?? []}
        xRange={xRange}
      />
      <div className="mdc-footer">
        <div className="mdc-footer-chips">
          {!hasMid && !hasBid && !hasAsk && (
            <span className="chip chip-warn">No Polymarket data</span>
          )}
          {!hasPrnLine && !hasPrnDots && (
            <span className="chip chip-warn">No pRN data</span>
          )}
          {prnSourceChip && (
            <span className="chip">{prnSourceChip}</span>
          )}
          {isSparseData && (
            <span className="chip chip-warn">Sparse data</span>
          )}
          {series.quality === "low_volume" && (
            <span className="chip chip-warn">Low volume — midprice may be unreliable</span>
          )}
          {series.quality === "suspect" && (
            <span className="chip chip-warn">
              Suspect data — {((series.midprice_cluster_ratio ?? 0) * 100).toFixed(0)}% near midprice, max jump {((series.max_jump ?? 0) * 100).toFixed(0)}%
            </span>
          )}
          {series.quality === "stale" && (
            <span className="chip chip-warn">Stale prices — {((series.stale_ratio ?? 0) * 100).toFixed(0)}% identical</span>
          )}
          {marketQualityBadges.map((badge) => (
            <span key={badge} className="chip chip-warn">
              {badge}
            </span>
          ))}
        </div>
        <div className="mdc-footer-meta">
          {pointsCount.toLocaleString()} pts · Market {series.market_id ?? "--"}
          {series.gamma_volume != null ? ` · Vol: $${series.gamma_volume.toLocaleString()}` : ""}
          {dropped > 0 ? ` · ${dropped} invalid dropped` : ""}
        </div>
      </div>
      {series.market_quality ? (
        <div className="strike-quality-card">
          <div className="strike-quality-card-header">
            <strong>Market quality</strong>
            <span>
              {series.market_quality.quality_bucket ?? "--"} · {series.market_quality.quality_issue_count ?? 0} issues
            </span>
          </div>
          <div className="strike-quality-card-grid">
            <div>
              <span className="meta-label">Snapshot</span>
              <span>{series.market_quality.snapshot_date_used ?? "--"}</span>
            </div>
            <div>
              <span className="meta-label">Coverage</span>
              <span>{series.market_quality.snapshot_coverage_status ?? "--"}</span>
            </div>
            <div>
              <span className="meta-label">Last trade gap</span>
              <span>
                {series.market_quality.hours_since_last_yes_trade != null
                  ? `${series.market_quality.hours_since_last_yes_trade.toFixed(1)}h`
                  : "--"}
              </span>
            </div>
            <div>
              <span className="meta-label">Issue flags</span>
              <span>{series.market_quality.active_flags.slice(0, 4).join(", ") || "--"}</span>
            </div>
          </div>
        </div>
      ) : null}
      {series.market_id && (
        <div className="strike-market-meta">
          <span className="strike-market-id-label">
            Market ID: <span className="strike-market-id-value">{series.market_id}</span>
          </span>
          {series.event_slug && (
            <a
              className="strike-polymarket-link"
              href={`https://polymarket.com/event/${series.event_slug}?market=${series.market_id}`}
              target="_blank"
              rel="noopener noreferrer"
            >
              View on Polymarket ↗
            </a>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Progress steps
// ---------------------------------------------------------------------------

type ProgressStep = "idle" | "fetching" | "processing" | "theta" | "done" | "error";

const STEP_LABELS: Record<ProgressStep, string> = {
  idle: "",
  fetching: "Fetching markets for selected ticker and trading week…",
  processing: "Grouping by strike and building charts…",
  theta: "Computing pRN via Theta (filling DTE gaps)…",
  done: "Charts rendered.",
  error: "An error occurred.",
};

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function BacktestsPage() {
  const { activeJobs } = useAnyJobRunning();

  // Controls
  const [selectedTicker, setSelectedTicker] = useState<string>(
    TRADING_UNIVERSE_TICKERS[0] ?? "",
  );
  const [startDate, setStartDate] = useState<string>("");
  const [endDate, setEndDate] = useState<string>("");
  const [tradingWeeks, setTradingWeeks] = useState<TradingWeek[]>([]);
  const [tradingWeeksLoading, setTradingWeeksLoading] = useState<boolean>(false);
  const [tradingWeeksError, setTradingWeeksError] = useState<string | null>(null);
  const [calendarMonth, setCalendarMonth] = useState<Date>(() =>
    startOfMonthUtc(new Date()),
  );
  const startDateRef = useRef(startDate);
  const endDateRef = useRef(endDate);

  // Run selection
  const [barRuns, setBarRuns] = useState<BarRun[]>([]);
  const [barRunsStatus, setBarRunsStatus] = useState<"loading" | "ready" | "error">("loading");
  const [barRunsError, setBarRunsError] = useState<string | null>(null);
  const [selectedBarRun, setSelectedBarRun] = useState<string>("");

  // Result
  const [result, setResult] = useState<ByStrikeResponse | null>(null);
  const [prnData, setPrnData] = useState<PrnOverlayResponse | null>(null);
  const [marketsSeriesByTicker, setMarketsSeriesByTicker] =
    useState<MarketsSeriesByTickerResponse | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProgressStep>("idle");
  const [selectedStrike, setSelectedStrike] = useState<string | null>(null);
  const [availableStrikes, setAvailableStrikes] = useState<StrikeSeries[]>([]);
  const [availableStrikesLoading, setAvailableStrikesLoading] = useState<boolean>(false);
  const [availableStrikesError, setAvailableStrikesError] = useState<string | null>(null);
  const strikeCacheRef = useRef<Map<string, StrikeSeries[]>>(new Map());
  const lastAutoRunKeyRef = useRef<string>("");

  // Quality filters
  const [hideSuspect, setHideSuspect] = useState<boolean>(false);
  const [hideNotRelevant, setHideNotRelevant] = useState<boolean>(false);
  const [hideNoTrade, setHideNoTrade] = useState<boolean>(false);
  const [hideStaleMarkets, setHideStaleMarkets] = useState<boolean>(false);
  const [hidePrnMissing, setHidePrnMissing] = useState<boolean>(false);
  const [hideExtremeOtm, setHideExtremeOtm] = useState<boolean>(false);
  const [minVolumeFilter, setMinVolumeFilter] = useState<number>(0);

  const hasAnyVolume = useMemo(
    () => availableStrikes.some((s) => s.gamma_volume != null),
    [availableStrikes],
  );

  const filteredStrikes = useMemo(() => {
    if (
      !hideSuspect &&
      !hideNotRelevant &&
      !hideNoTrade &&
      !hideStaleMarkets &&
      !hidePrnMissing &&
      !hideExtremeOtm &&
      minVolumeFilter <= 0
    ) {
      return availableStrikes;
    }
    return availableStrikes.filter((s) => {
      const marketQuality = s.market_quality;
      if (hideSuspect && (s.quality === "low_volume" || s.quality === "suspect" || s.quality === "stale")) return false;
      if (hideNotRelevant && marketQuality?.flag_not_relevant) return false;
      if (hideNoTrade && (marketQuality?.flag_pm_no_trade_history || marketQuality?.flag_pm_no_recent_trade)) return false;
      if (hideStaleMarkets && marketQuality?.flag_pm_stale_prices) return false;
      if (hidePrnMissing && marketQuality?.flag_prn_missing) return false;
      if (hideExtremeOtm && marketQuality?.flag_extreme_otm) return false;
      if (minVolumeFilter > 0 && s.gamma_volume != null && s.gamma_volume < minVolumeFilter) return false;
      return true;
    });
  }, [
    availableStrikes,
    hideSuspect,
    hideNotRelevant,
    hideNoTrade,
    hideStaleMarkets,
    hidePrnMissing,
    hideExtremeOtm,
    minVolumeFilter,
  ]);

  const hiddenStrikesCount = availableStrikes.length - filteredStrikes.length;

  const suspectStrikesCount = useMemo(() => {
    let count = 0;
    for (const s of availableStrikes) {
      if (s.quality === "low_volume" || s.quality === "suspect" || s.quality === "stale") count++;
    }
    return count;
  }, [availableStrikes]);

  const notRelevantCount = useMemo(
    () => availableStrikes.filter((s) => s.market_quality?.flag_not_relevant).length,
    [availableStrikes],
  );
  const noTradeCount = useMemo(
    () =>
      availableStrikes.filter(
        (s) => s.market_quality?.flag_pm_no_trade_history || s.market_quality?.flag_pm_no_recent_trade,
      ).length,
    [availableStrikes],
  );
  const staleMarketCount = useMemo(
    () => availableStrikes.filter((s) => s.market_quality?.flag_pm_stale_prices).length,
    [availableStrikes],
  );
  const prnMissingCount = useMemo(
    () => availableStrikes.filter((s) => s.market_quality?.flag_prn_missing).length,
    [availableStrikes],
  );
  const extremeOtmCount = useMemo(
    () => availableStrikes.filter((s) => s.market_quality?.flag_extreme_otm).length,
    [availableStrikes],
  );

  const hasHistoryRuns = barRunsStatus === "ready" && barRuns.length > 0;
  const noHistoryRuns = barRunsStatus === "ready" && barRuns.length === 0;

  useEffect(() => {
    if (
      barRunsStatus === "ready" &&
      selectedBarRun &&
      !barRuns.some((r) => r.run_id === selectedBarRun)
    ) {
      setSelectedBarRun("");
    }
  }, [barRunsStatus, barRuns, selectedBarRun]);

  const resetStrikeSelection = useCallback(() => {
    setSelectedStrike(null);
    setResult(null);
    setPrnData(null);
    setMarketsSeriesByTicker(null);
    setRunError(null);
    setProgress("idle");
    lastAutoRunKeyRef.current = "";
  }, []);

  useEffect(() => {
    startDateRef.current = startDate;
    endDateRef.current = endDate;
  }, [startDate, endDate]);

  // Load available runs on mount
  useEffect(() => {
    const setFallbackDates = () => {
      const end = new Date();
      const start = new Date(end);
      start.setUTCDate(end.getUTCDate() - (MAX_RANGE_DAYS - 1));
      const startStr = start.toISOString().slice(0, 10);
      const endStr = end.toISOString().slice(0, 10);
      setStartDate((prev) => prev || startStr);
      setEndDate((prev) => prev || endStr);
    };

    const applyManifestDates = (manifest?: Record<string, any>) => {
      const manifestStart =
        typeof manifest?.start_date === "string" ? manifest.start_date : null;
      const manifestEnd =
        typeof manifest?.end_date === "string" ? manifest.end_date : null;

      if (!manifestEnd) {
        setFallbackDates();
        return;
      }

      const end = new Date(`${manifestEnd}T00:00:00Z`);
      const start = new Date(end);
      start.setUTCDate(end.getUTCDate() - (MAX_RANGE_DAYS - 1));
      let startStr = start.toISOString().slice(0, 10);
      if (manifestStart && startStr < manifestStart) {
        startStr = manifestStart;
      }

      setStartDate((prev) => prev || startStr);
      setEndDate((prev) => prev || manifestEnd);
    };

    setBarRunsStatus("loading");
    setBarRunsError(null);
    listBarRuns()
      .then((payload) => {
        setBarRuns(payload.runs);
        setBarRunsStatus("ready");
        setBarRunsError(null);

        if (payload.runs.length === 0) {
          resetStrikeSelection();
          setTradingWeeks([]);
          setTradingWeeksError(null);
          setTradingWeeksLoading(false);
          setStartDate("");
          setEndDate("");
          setSelectedBarRun("");
          return;
        }

        applyManifestDates(payload.runs[0]?.manifest);

        const manifestTickers = payload.runs[0]?.manifest?.tickers;
        if (Array.isArray(manifestTickers)) {
          const fallbackTicker =
            manifestTickers.find((ticker) =>
              TRADING_UNIVERSE_TICKERS.includes(ticker),
            ) ??
            TRADING_UNIVERSE_TICKERS[0] ??
            "";
          if (fallbackTicker) {
            setSelectedTicker((prev) => prev || fallbackTicker);
          }
        }
      })
      .catch((err) => {
        console.error("Failed to load bar runs:", err);
        setBarRuns([]);
        setBarRunsStatus("error");
        setBarRunsError(
          err instanceof Error
            ? `Failed to load bar runs. ${err.message}`
            : "Failed to load bar runs. Check the backend connection.",
        );
        setFallbackDates();
      });
  }, [resetStrikeSelection]);

  // Load trading weeks for selected ticker/run
  useEffect(() => {
    if (!selectedTicker || !hasHistoryRuns) {
      setTradingWeeks([]);
      setTradingWeeksError(null);
      setTradingWeeksLoading(false);
      return;
    }

    let cancelled = false;
    setTradingWeeksError(null);
    setTradingWeeks([]);
    setTradingWeeksLoading(true);

    listTradingWeeks({ ticker: selectedTicker, runId: selectedBarRun || undefined })
      .then((payload) => {
        if (cancelled) return;
        setTradingWeeksLoading(false);
        setTradingWeeks(payload.weeks);

        if (payload.weeks.length === 0) {
          if (startDateRef.current || endDateRef.current) {
            resetStrikeSelection();
          }
          setStartDate("");
          setEndDate("");
          return;
        }

        const weekByStart = new Map(
          payload.weeks.map((week) => [week.start_date, week]),
        );
        const currentStart = startDateRef.current;
        const currentEnd = endDateRef.current;
        const selected = weekByStart.get(currentStart) ?? payload.weeks[payload.weeks.length - 1];

        const rangeChanged =
          selected.start_date !== currentStart || selected.end_date !== currentEnd;
        if (rangeChanged) {
          resetStrikeSelection();
        }
        if (selected.start_date !== currentStart) {
          setStartDate(selected.start_date);
        }
        if (selected.end_date !== currentEnd) {
          setEndDate(selected.end_date);
        }
      })
      .catch((err) => {
        if (cancelled) return;
        setTradingWeeksLoading(false);
        setTradingWeeks([]);
        const msg = err instanceof Error ? err.message : "Failed to load trading weeks.";
        if (selectedBarRun && /404|not found/i.test(msg)) {
          setSelectedBarRun("");
          setTradingWeeksError("Selected run no longer exists. Switched to active run.");
        } else {
          setTradingWeeksError(msg);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [resetStrikeSelection, selectedBarRun, selectedTicker, hasHistoryRuns]);

  const tradingWeeksByStart = useMemo(() => {
    return new Map(tradingWeeks.map((week) => [week.start_date, week]));
  }, [tradingWeeks]);

  const selectedTradingWeek = useMemo(() => {
    return tradingWeeksByStart.get(startDate) ?? null;
  }, [tradingWeeksByStart, startDate]);

  const calendarWeeks = useMemo(
    () => buildCalendarWeeks(calendarMonth),
    [calendarMonth],
  );

  const minCalendarMonth = useMemo(() => {
    if (tradingWeeks.length === 0) return null;
    const parsed = parseUtcDate(tradingWeeks[0].start_date);
    return parsed ? startOfMonthUtc(parsed) : null;
  }, [tradingWeeks]);

  const maxCalendarMonth = useMemo(() => {
    if (tradingWeeks.length === 0) return null;
    const parsed = parseUtcDate(tradingWeeks[tradingWeeks.length - 1].start_date);
    return parsed ? startOfMonthUtc(parsed) : null;
  }, [tradingWeeks]);

  const canPrevMonth = minCalendarMonth
    ? monthKey(calendarMonth) > monthKey(minCalendarMonth)
    : false;
  const canNextMonth = maxCalendarMonth
    ? monthKey(calendarMonth) < monthKey(maxCalendarMonth)
    : false;

  const handleMonthStep = useCallback((delta: number) => {
    setCalendarMonth((prev) =>
      new Date(Date.UTC(prev.getUTCFullYear(), prev.getUTCMonth() + delta, 1)),
    );
  }, []);

  const handleWeekSelect = useCallback((week: TradingWeek) => {
    const nextStart = week.start_date;
    const nextEnd = week.end_date;
    if (nextStart !== startDate || nextEnd !== endDate) {
      resetStrikeSelection();
    }
    setStartDate(nextStart);
    setEndDate(nextEnd);
    const parsed = parseUtcDate(week.start_date);
    if (parsed) {
      setCalendarMonth((prev) => {
        const next = startOfMonthUtc(parsed);
        return monthKey(prev) === monthKey(next) ? prev : next;
      });
    }
  }, [endDate, resetStrikeSelection, startDate]);

  useEffect(() => {
    if (!selectedTradingWeek && tradingWeeks.length === 0) return;
    const anchor = selectedTradingWeek?.start_date ?? tradingWeeks[tradingWeeks.length - 1]?.start_date;
    const parsed = anchor ? parseUtcDate(anchor) : null;
    if (!parsed) return;
    const next = startOfMonthUtc(parsed);
    setCalendarMonth((prev) => (monthKey(prev) === monthKey(next) ? prev : next));
  }, [selectedTradingWeek, tradingWeeks]);

  // Trading week validation
  const rangeValidation = useMemo(
    () => validateDateRange(startDate, endDate),
    [startDate, endDate],
  );

  const strikeRequestKey = useMemo(() => {
    if (!selectedTicker || !startDate || !endDate) return "";
    const runKey = selectedBarRun || "latest";
    return `${selectedTicker}|${runKey}|${startDate}|${endDate}`;
  }, [selectedTicker, selectedBarRun, startDate, endDate]);

  const chartRange = useMemo<ChartRange | null>(() => {
    if (!startDate || !endDate) return null;
    const min = Date.parse(`${startDate}T00:00:00Z`);
    const max = Date.parse(`${endDate}T23:59:59Z`);
    if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return null;
    return { min, max };
  }, [startDate, endDate]);

  const canRun =
    hasHistoryRuns
    && !!selectedTicker && !!startDate && !!endDate && rangeValidation.ok
    && !!selectedTradingWeek && !tradingWeeksError
    && progress !== "fetching" && progress !== "theta";

  const strikesSelectionReady =
    hasHistoryRuns
    && !!selectedTicker && !!startDate && !!endDate && rangeValidation.ok
    && !!selectedTradingWeek && !tradingWeeksError;

  useEffect(() => {
    if (!strikesSelectionReady) {
      setAvailableStrikes([]);
      setAvailableStrikesLoading(false);
      setAvailableStrikesError(null);
      return;
    }

    const cached = strikeRequestKey
      ? strikeCacheRef.current.get(strikeRequestKey)
      : null;
    if (cached) {
      setAvailableStrikes(cached);
      setAvailableStrikesLoading(false);
      setAvailableStrikesError(null);
      return;
    }

    let cancelled = false;
    setAvailableStrikes([]);
    setAvailableStrikesLoading(true);
    setAvailableStrikesError(null);

    const timeMin = `${startDate}T00:00:00Z`;
    const timeMax = `${endDate}T23:59:59Z`;

    getBarsByStrike({
      ticker: selectedTicker,
      runId: selectedBarRun || undefined,
      timeMin,
      timeMax,
      tokenRole: "yes",
      maxPointsPerStrike: 10,
      viewMode: "full_history",
    })
      .then((data) => {
        if (cancelled) return;
        setAvailableStrikes(data.strikes);
        setAvailableStrikesLoading(false);
        if (strikeRequestKey) {
          strikeCacheRef.current.set(strikeRequestKey, data.strikes);
        }
      })
      .catch((err) => {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : "Failed to load strikes.";
        setAvailableStrikes([]);
        setAvailableStrikesLoading(false);
        setAvailableStrikesError(msg);
      });

    return () => {
      cancelled = true;
    };
  }, [
    strikeRequestKey,
    strikesSelectionReady,
    selectedTicker,
    selectedBarRun,
    startDate,
    endDate,
  ]);

  useEffect(() => {
    if (!selectedStrike || availableStrikesLoading) return;
    if (filteredStrikes.length === 0) {
      setSelectedStrike(null);
      return;
    }
    const keys = new Set(filteredStrikes.map(buildStrikeKey));
    if (!keys.has(selectedStrike)) {
      setSelectedStrike(null);
    }
  }, [filteredStrikes, availableStrikesLoading, selectedStrike]);

  // Run handler
  const handleRun = useCallback(async () => {
    if (!canRun) return;

    if (strikeRequestKey) {
      lastAutoRunKeyRef.current = strikeRequestKey;
    }

    setRunError(null);
    setResult(null);
    setPrnData(null);
    setMarketsSeriesByTicker(null);
    setProgress("fetching");

    try {
      const timeMin = `${startDate}T00:00:00Z`;
      const timeMax = `${endDate}T23:59:59Z`;

      console.log(
        `[BacktestsPage] Run: ticker=${selectedTicker} range=${startDate}..${endDate} run=${selectedBarRun || "latest"}`,
      );

      // Fetch PM bars and markets series in parallel
      const [data, marketsSeries] = await Promise.all([
        getBarsByStrike({
          ticker: selectedTicker,
          runId: selectedBarRun || undefined,
          timeMin,
          timeMax,
          tokenRole: "yes",
          maxPointsPerStrike: 500,
          viewMode: "full_history",
        }),
        getMarketsSeriesByTicker({
          ticker: selectedTicker,
          weekFriday: endDate,
          runId: selectedBarRun || undefined,
        }).catch((err) => {
          console.warn("[BacktestsPage] Markets series fetch failed (non-fatal):", err);
          return null;
        }),
      ]);

      console.log(
        `[BacktestsPage] Result: ${data.strikes.length} strikes, metadata=`,
        data.metadata,
      );

      setProgress("processing");

      if (data.strikes.length === 0) {
        setRunError(
          `No markets found for ${selectedTicker} in ${startDate} – ${endDate}. ` +
            `Try a different trading week or check that the pipeline has been run for this ticker.`,
        );
        setProgress("error");
        return;
      }

      setResult(data);
      setMarketsSeriesByTicker(marketsSeries);
      setAvailableStrikes(data.strikes);
      setAvailableStrikesLoading(false);
      setAvailableStrikesError(null);
      const cacheKey = `${selectedTicker}|${selectedBarRun || "latest"}|${startDate}|${endDate}`;
      strikeCacheRef.current.set(cacheKey, data.strikes);

      // Theta is the sole pRN source — compute for all PM strikes
      const pmStrikes = data.strikes.map((s) => s.strike);
      if (pmStrikes.length > 0) {
        setProgress("theta");
        console.log(
          `[BacktestsPage] Calling Theta for ${pmStrikes.length} PM strikes: ${pmStrikes.join(",")}`,
        );

        const thetaPrn = await getPrnOverlayTheta({
          ticker: selectedTicker,
          timeMin,
          timeMax,
          strikes: pmStrikes,
        }).catch((err) => {
          console.warn("[BacktestsPage] Theta pRN fetch failed (non-fatal):", err);
          return null;
        });

        if (thetaPrn && thetaPrn.strikes.length > 0) {
          console.log(
            `[BacktestsPage] Theta pRN: ${thetaPrn.strikes.length} strikes, metadata=`,
            thetaPrn.metadata,
          );
          setPrnData(thetaPrn);
        }
      }
      setProgress("done");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setRunError(msg);
      setProgress("error");
      console.error("[BacktestsPage] Error:", err);
    }
  }, [canRun, strikeRequestKey, selectedTicker, startDate, endDate, selectedBarRun]);

  useEffect(() => {
    if (!canRun || !selectedStrike || !strikeRequestKey) return;
    if (lastAutoRunKeyRef.current === strikeRequestKey) return;
    handleRun();
  }, [canRun, selectedStrike, strikeRequestKey, handleRun]);

  return (
    <section className="page backtests-page">
      <PipelineStatusCard
        className="page-sticky-meta backtests-meta"
        activeJobsCount={activeJobs.length}
      />
      <header className="page-header backtests-page-header">
        <div className="backtests-title-row">
          <h1 className="page-title backtests-page-title">Backtests</h1>
        </div>
      </header>

      <div className="backtests-workspace">
        {/* Controls */}
        <div className="backtests-controls">
          <div className="backtests-controls-left">
            {/* Ticker selector */}
            <section className="panel ticker-panel">
              <div className="panel-header">
                <div>
                  <h2>Ticker</h2>
                  <span className="panel-hint">Pick one from the trading universe.</span>
                </div>
                {selectedTicker && (
                  <span className="meta-pill">Selected: {selectedTicker}</span>
                )}
              </div>
              <div className="panel-body">
                <div
                  className={`ticker-universe ${!TRADING_UNIVERSE_TICKERS.length ? "is-disabled" : ""}`}
                  role="group"
                >
                  {TRADING_UNIVERSE_TICKERS.map((ticker) => {
                    const isActive = ticker === selectedTicker;
                    return (
                      <button
                        key={ticker}
                        type="button"
                        className={`ticker-pill ${isActive ? "is-active" : ""}`}
                        onClick={() => setSelectedTicker(ticker)}
                        aria-pressed={isActive}
                      >
                        {ticker}
                      </button>
                    );
                  })}
                </div>
              </div>
            </section>

            {/* Available strikes */}
            <section className="panel strikes-panel">
              <div className="panel-header">
                <div>
                  <h2>Strikes</h2>
                  <span className="panel-hint">
                    {strikesSelectionReady
                      ? "Available strikes for the selected week."
                      : "Select a trading week to load strikes."}
                  </span>
                </div>
                {availableStrikes.length > 0 && (
                  <span className="meta-pill">
                    {filteredStrikes.length}
                    {hiddenStrikesCount > 0 ? ` / ${availableStrikes.length}` : ""} strike
                    {(hiddenStrikesCount > 0 ? availableStrikes.length : filteredStrikes.length) !== 1 ? "s" : ""}
                  </span>
                )}
              </div>

              {/* Quality filter toolbar */}
              {availableStrikes.length > 0 && (
                <div className="strike-filter-toolbar">
                  <div className="strike-filter-toggles">
                    <label className="strike-filter-toggle" title="Hide strikes with suspect data quality (low volume, empty book midprice, or stale prices)">
                      <input
                        type="checkbox"
                        checked={hideSuspect}
                        onChange={(e) => setHideSuspect(e.target.checked)}
                      />
                      <span>
                        Hide suspect data
                        {suspectStrikesCount > 0 && (
                          <span className="strike-filter-count">{suspectStrikesCount}</span>
                        )}
                      </span>
                    </label>
                    <label className="strike-filter-toggle" title="Hide strikes flagged as not relevant for backtests">
                      <input
                        type="checkbox"
                        checked={hideNotRelevant}
                        onChange={(e) => setHideNotRelevant(e.target.checked)}
                      />
                      <span>
                        Hide not relevant
                        {notRelevantCount > 0 && (
                          <span className="strike-filter-count">{notRelevantCount}</span>
                        )}
                      </span>
                    </label>
                    <label className="strike-filter-toggle" title="Hide markets with no trade history or no recent trades at the snapshot anchor">
                      <input
                        type="checkbox"
                        checked={hideNoTrade}
                        onChange={(e) => setHideNoTrade(e.target.checked)}
                      />
                      <span>
                        Hide no trade
                        {noTradeCount > 0 && (
                          <span className="strike-filter-count">{noTradeCount}</span>
                        )}
                      </span>
                    </label>
                    <label className="strike-filter-toggle" title="Hide markets flagged for stale price history">
                      <input
                        type="checkbox"
                        checked={hideStaleMarkets}
                        onChange={(e) => setHideStaleMarkets(e.target.checked)}
                      />
                      <span>
                        Hide stale
                        {staleMarketCount > 0 && (
                          <span className="strike-filter-count">{staleMarketCount}</span>
                        )}
                      </span>
                    </label>
                    <label className="strike-filter-toggle" title="Hide markets with missing exact pRN coverage">
                      <input
                        type="checkbox"
                        checked={hidePrnMissing}
                        onChange={(e) => setHidePrnMissing(e.target.checked)}
                      />
                      <span>
                        Hide pRN missing
                        {prnMissingCount > 0 && (
                          <span className="strike-filter-count">{prnMissingCount}</span>
                        )}
                      </span>
                    </label>
                    <label className="strike-filter-toggle" title="Hide extreme out-of-the-money markets">
                      <input
                        type="checkbox"
                        checked={hideExtremeOtm}
                        onChange={(e) => setHideExtremeOtm(e.target.checked)}
                      />
                      <span>
                        Hide extreme OTM
                        {extremeOtmCount > 0 && (
                          <span className="strike-filter-count">{extremeOtmCount}</span>
                        )}
                      </span>
                    </label>
                  </div>
                  <div className="strike-filter-vol">
                    <label htmlFor="min-vol-select" className="strike-filter-vol-label">
                      Min volume
                    </label>
                    <select
                      id="min-vol-select"
                      className="strike-filter-vol-select"
                      value={minVolumeFilter}
                      onChange={(e) => setMinVolumeFilter(Number(e.target.value))}
                      disabled={!hasAnyVolume}
                      title={hasAnyVolume ? undefined : "Volume data not available for this run — re-run pipeline to capture it"}
                    >
                      <option value={0}>{hasAnyVolume ? "Any" : "N/A"}</option>
                      <option value={1000}>$1K</option>
                      <option value={5000}>$5K</option>
                      <option value={10000}>$10K</option>
                      <option value={50000}>$50K</option>
                      <option value={100000}>$100K</option>
                    </select>
                  </div>
                  {hiddenStrikesCount > 0 && (
                    <span className="strike-filter-hidden">
                      {hiddenStrikesCount} strike{hiddenStrikesCount !== 1 ? "s" : ""} hidden
                    </span>
                  )}
                </div>
              )}

              <div className="panel-body">
                {availableStrikesLoading && (
                  <div className="empty">Loading strikes…</div>
                )}
                {!availableStrikesLoading && availableStrikesError && (
                  <div className="error-banner">{availableStrikesError}</div>
                )}
                {!availableStrikesLoading
                  && !availableStrikesError
                  && availableStrikes.length === 0 && (
                  <div className="empty">
                    {strikesSelectionReady
                      ? "No strikes found for this ticker/week."
                      : "Select a ticker and trading week to see available strikes."}
                  </div>
                )}
                {!availableStrikesLoading
                  && !availableStrikesError
                  && availableStrikes.length > 0
                  && filteredStrikes.length === 0 && (
                  <div className="empty">
                    All {availableStrikes.length} strike{availableStrikes.length !== 1 ? "s" : ""} hidden by quality filters.
                    Adjust filters above to see results.
                  </div>
                )}
                {filteredStrikes.length > 0 && (
                  <div className="strike-list" role="group">
                    {filteredStrikes.map((series) => {
                      const seriesKey = buildStrikeKey(series);
                      const isActive = selectedStrike === seriesKey;
                      const marketQualityBadges = getMarketQualityBadges(series);
                      const isSuspect =
                        (series.quality !== "good" && series.quality != null) ||
                        (series.market_quality?.quality_issue_count ?? 0) > 0;
                      const qualityLabel =
                        series.quality === "low_volume" ? "Low Vol"
                        : series.quality === "suspect" ? "Suspect"
                        : series.quality === "stale" ? "Stale"
                        : null;
                      const qualityHint =
                        series.quality === "low_volume" ? "Low trading volume"
                        : series.quality === "suspect" ? "Suspect data — possible empty orderbook"
                        : series.quality === "stale" ? "Stale / flat price data"
                        : undefined;
                      const volLabel = series.gamma_volume != null
                        ? series.gamma_volume >= 1_000_000
                          ? `$${(series.gamma_volume / 1_000_000).toFixed(1)}M vol`
                          : series.gamma_volume >= 1_000
                            ? `$${(series.gamma_volume / 1_000).toFixed(0)}K vol`
                            : `$${series.gamma_volume.toFixed(0)} vol`
                        : null;
                      return (
                        <button
                          key={seriesKey}
                          type="button"
                          className={`strike-pill ${isActive ? "is-active" : ""}${isSuspect ? " is-low-quality" : ""}`}
                          onClick={() => setSelectedStrike(seriesKey)}
                          aria-pressed={isActive}
                          title={qualityHint}
                        >
                          ${series.strike_label}
                          {series.market_id && (
                            <span className="strike-pill-market">mkt {series.market_id}</span>
                          )}
                          <span className="strike-pill-pts">
                            {series.total_points} pts
                          </span>
                          {volLabel && (
                            <span className={`strike-pill-vol${isSuspect ? " is-warn" : ""}`}>
                              {volLabel}
                            </span>
                          )}
                          {qualityLabel && (
                            <span className="strike-pill-quality-badge">
                              {qualityLabel}
                            </span>
                          )}
                          {marketQualityBadges.map((badge) => (
                            <span key={badge} className="strike-pill-quality-badge market-quality">
                              {badge}
                            </span>
                          ))}
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
            </section>
          </div>

          {/* Trading week + run selector */}
          <section className="panel date-panel">
          <div className="panel-header">
            <div>
              <h2>Date range</h2>
              <span className="panel-hint">
                Pick a trading week (Mon–Fri). Weeks without trading data are disabled.
              </span>
            </div>
            {rangeValidation.ok && (
              <span className="meta-pill">
                {rangeValidation.days} day
                {rangeValidation.days !== 1 ? "s" : ""} selected
              </span>
            )}
          </div>
          <div className="panel-body">
            <div className="trading-calendar">
              <div className="trading-calendar-header">
                <button
                  type="button"
                  className="calendar-nav-button"
                  onClick={() => handleMonthStep(-1)}
                  disabled={!canPrevMonth}
                  aria-label="Previous month"
                >
                  ‹
                </button>
                <div className="trading-calendar-title">
                  {formatMonthYear(calendarMonth)}
                </div>
                <button
                  type="button"
                  className="calendar-nav-button"
                  onClick={() => handleMonthStep(1)}
                  disabled={!canNextMonth}
                  aria-label="Next month"
                >
                  ›
                </button>
              </div>
              <div className="trading-calendar-weekdays">
                {["Mon", "Tue", "Wed", "Thu", "Fri"].map((day) => (
                  <span key={day}>{day}</span>
                ))}
              </div>
              <div className="trading-calendar-weeks">
                {calendarWeeks.map((week) => {
                  const weekKey = formatUtcDate(week.weekStart);
                  const weekInfo = tradingWeeksByStart.get(weekKey);
                  const isAvailable = Boolean(weekInfo);
                  const isSelected = startDate === weekKey;
                  return (
                    <button
                      key={weekKey}
                      type="button"
                      className={`trading-calendar-week${isSelected ? " is-selected" : ""}${!isAvailable ? " is-disabled" : ""}`}
                      onClick={() => weekInfo && handleWeekSelect(weekInfo)}
                      disabled={!isAvailable}
                      aria-label={
                        weekInfo
                          ? `Week of ${formatRangeLabel(weekInfo.start_date, weekInfo.end_date)}`
                          : "Week unavailable"
                      }
                    >
                      {week.days.map((day) => {
                        const isOutside = day.getUTCMonth() !== calendarMonth.getUTCMonth();
                        return (
                          <span
                            key={day.toISOString()}
                            className={`trading-calendar-day${isOutside ? " is-outside" : ""}`}
                          >
                            {day.getUTCDate()}
                          </span>
                        );
                      })}
                    </button>
                  );
                })}
              </div>
              <div className="trading-calendar-selection">
                {selectedTradingWeek
                  ? `Selected: ${formatRangeLabel(
                      selectedTradingWeek.start_date,
                      selectedTradingWeek.end_date,
                    )} (UTC)`
                  : "Select a trading week to continue."}
              </div>
              {tradingWeeksLoading && (
                <div className="trading-calendar-empty">Loading trading weeks…</div>
              )}
              {!tradingWeeksLoading && tradingWeeks.length === 0 && !tradingWeeksError && (
                <div className="trading-calendar-empty">
                  No trading weeks available for this ticker/run.
                </div>
              )}
            </div>

            <div className="field">
              <label htmlFor="bt-run">Pipeline run</label>
              <select
                id="bt-run"
                className="input"
                value={selectedBarRun}
                onChange={(e) => setSelectedBarRun(e.target.value)}
              >
                <option value="">Active run (latest)</option>
                {barRuns.map((run) => (
                  <option key={run.run_id} value={run.run_id}>
                    {run.run_id}
                    {run.is_active ? " (active)" : ""}
                    {run.manifest?.label ? ` — ${run.manifest.label}` : ""}
                  </option>
                ))}
              </select>
            </div>
            {barRunsError && (
              <div className="error-banner">{barRunsError}</div>
            )}
            {tradingWeeksError && (
              <div className="error-banner">{tradingWeeksError}</div>
            )}
            {rangeValidation.error && (
              <div className="error-banner">{rangeValidation.error}</div>
            )}
            <div className={`actions backtests-actions${noHistoryRuns ? " is-empty" : ""}`}>
              {noHistoryRuns ? (
                <>
                  <span className="empty">
                    No Polymarket history runs found. Build one to enable backtests.
                  </span>
                  <Link to="/polymarket-history-builder" className="button light">
                    Go to Polymarket History Builder
                  </Link>
                </>
              ) : (
                <span className="empty">
                  Runs automatically when a strike and date range are selected.
                </span>
              )}
            </div>
          </div>
        </section>
      </div>

      {/* Progress / error */}
      {progress !== "idle" && progress !== "done" && !runError && (
        <div className="progress-banner">{STEP_LABELS[progress]}</div>
      )}
      {runError && <div className="error-banner">{runError}</div>}

      {/* Results */}
      {result && (() => {
        // Build pRN lookup: canonical strike key -> PrnChartPoint[]
        const prnByStrike = new Map<string, PrnChartPoint[]>();
        const holidaysInRange = Array.isArray(prnData?.metadata?.holidays_in_range)
          ? (prnData!.metadata!.holidays_in_range as string[]).length
          : 0;
        if (prnData) {
          for (const ps of prnData.strikes) {
            if (!hasRequiredPrnDtes(ps.points, holidaysInRange)) {
              continue;
            }
            const key = strikeKey(ps.strike);
            const chartPts: PrnChartPoint[] = ps.points
              .filter((p: PrnPoint) => Number.isFinite(p.pRN) && Number.isFinite(p.asof_date_ms))
              .map((p: PrnPoint) => {
                // DEV ASSERTION: pRN must use EOD chain — its timestamp must be >= 21:00 UTC
                // of its asof_date, never placed before the data was available.
                if (import.meta.env.DEV) {
                  const asofMidnight = new Date(`${p.asof_date}T00:00:00Z`).getTime();
                  const eodMs = asofMidnight + 21 * 3600_000;
                  if (p.asof_date_ms < eodMs) {
                    console.error(
                      `[pRN AS-OF VIOLATION] strike=${ps.strike} date=${p.asof_date} dte=${p.dte}: ` +
                      `asof_date_ms=${p.asof_date_ms} < eod=${eodMs} — pRN plotted before chain was available`,
                    );
                  }
                }
                return { timeMs: p.asof_date_ms, price: p.pRN, dte: p.dte };
              });
            if (chartPts.length > 0) {
              // Merge into existing if multiple pRN sources map to same key
              const prev = prnByStrike.get(key);
              prnByStrike.set(key, prev ? [...prev, ...chartPts] : chartPts);
            }
          }
        }

        const matchPrn = (strike: number): PrnChartPoint[] | undefined => {
          return prnByStrike.get(strikeKey(strike));
        };

        const marketsByStrike = new Map<string, MarketsSeriesResponse>();
        if (marketsSeriesByTicker) {
          for (const ms of marketsSeriesByTicker.strikes) {
            const key = `${strikeKey(ms.threshold)}:${ms.market_id ?? "unknown"}`;
            marketsByStrike.set(key, ms);
            marketsByStrike.set(strikeKey(ms.threshold), ms);
          }
        }

        const matchMarketsSeries = (series: StrikeSeries): MarketsSeriesResponse | undefined => {
          return marketsByStrike.get(buildStrikeKey(series)) ?? marketsByStrike.get(strikeKey(series.strike));
        };

        const activeStrikeIndex =
          selectedStrike != null
            ? result.strikes.findIndex((s) => buildStrikeKey(s) === selectedStrike)
            : -1;
        const activeSeries =
          activeStrikeIndex >= 0 ? result.strikes[activeStrikeIndex] : null;
        const prnSourceLabel = (() => {
          if (!prnData) return null;
          const meta = prnData.metadata ?? {};
          if (meta.source === "theta_on_demand") return "pRN source: Theta";
          return "pRN source: Theta";
        })();

        return (
        <div className="backtests-results">
          <div className="history-meta">
            <span className="meta-pill">Ticker: {result.ticker}</span>
            <span className="meta-pill">Strikes: {result.strikes.length}</span>
            <span className="meta-pill">Run: {result.run_id}</span>
            <span className="meta-pill">Role: {result.token_role}</span>
            {result.metadata.rows_scanned != null && (
              <span className="meta-pill">
                Rows scanned: {result.metadata.rows_scanned.toLocaleString()}
              </span>
            )}
            {result.metadata.nan_dropped > 0 && (
              <span className="meta-pill warn-text">
                NaN dropped: {result.metadata.nan_dropped}
              </span>
            )}
            {prnData?.metadata?.theta_merged && (
              <span className="meta-pill theta-pill">
                Theta pRN: {prnData.metadata.theta_strikes} strike{prnData.metadata.theta_strikes !== 1 ? "s" : ""} merged
              </span>
            )}
          </div>

          {/* Single chart for selected strike */}
          {activeSeries ? (
            <StrikeCard
              key={buildStrikeKey(activeSeries)}
              series={activeSeries}
              ticker={result.ticker}
              marketsSeries={matchMarketsSeries(activeSeries)}
              prnPoints={matchPrn(activeSeries.strike)}
              prnSourceLabel={prnSourceLabel}
              xRange={chartRange}
            />
          ) : (
            <div className="empty">Pick a strike above to render its chart.</div>
          )}
        </div>
        );
      })()}

      {/* Empty state */}
      {progress === "idle" && !result && (
        <div className="empty">
          Select a ticker, trading week, and strike to generate per-strike charts.
        </div>
      )}
      </div>
    </section>
  );
}
