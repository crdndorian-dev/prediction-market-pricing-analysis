import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  getMarketsSeriesByTicker,
  getMarketsSummary,
  startMarketsRefresh,
  type MarketsSeriesByTickerResponse,
  type MarketsSeriesResponse,
  type MarketsSummaryResponse,
} from "../api/markets";
import { useAnyJobRunning } from "../contexts/jobGuard";
import { useMarketsJob } from "../contexts/marketsJob";
import PipelineProgressBar from "../components/PipelineProgressBar";
import PipelineStatusCard from "../components/PipelineStatusCard";
import "./MarketsPage.css";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type ChartPoint = {
  timeMs: number;
  polymarketBid?: number | null;
  polymarketAsk?: number | null;
  prn?: number | null;
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_TIMEZONE = "America/New_York";

// SVG chart geometry (viewBox units)
const W = 800;
const H = 400;
const PL = 52;  // left padding — y-axis labels
const PR = 16;  // right padding
const PT = 20;  // top padding
const PB = 44;  // bottom padding — x-axis labels

// ---------------------------------------------------------------------------
// Date / formatting helpers
// ---------------------------------------------------------------------------

function getDatePartsInTimeZone(timeZone: string, date = new Date()) {
  const formatter = new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    weekday: "short",
  });
  const parts = formatter.formatToParts(date);
  const lookup = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  const year = Number(lookup.year);
  const month = Number(lookup.month);
  const day = Number(lookup.day);
  const weekday = lookup.weekday ?? "";
  return { year, month, day, weekday };
}

function computeDefaultWeekFriday(timeZone: string): string {
  const { year, month, day } = getDatePartsInTimeZone(timeZone);
  const base = new Date(Date.UTC(year, month - 1, day));
  const weekdayUtc = base.getUTCDay(); // 0=Sun
  const weekday = (weekdayUtc + 6) % 7; // 0=Mon..6=Sun
  const delta = 4 - weekday; // Mon(0)->+4, Fri(0)->0, Sat(5)->-1, Sun(6)->-2
  const weekFriday = new Date(base.getTime());
  weekFriday.setUTCDate(base.getUTCDate() + delta);
  return weekFriday.toISOString().slice(0, 10);
}

function formatTimestamp(value?: string | null): string {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatPrice(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return value.toFixed(3);
}

function formatUtcLabel(ms: number): string {
  return new Date(ms).toISOString().replace("T", " ").slice(0, 16) + "Z";
}

function formatLocalLabel(ms: number): string {
  return new Date(ms).toLocaleString("en-US", {
    timeZone: DEFAULT_TIMEZONE,
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Compute the Monday of a week given its Friday date string (YYYY-MM-DD). */
function computeWeekMonday(friday: string): string {
  const d = new Date(friday + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() - 4);
  return d.toISOString().slice(0, 10);
}

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

function computeWeekRangeUtcMs(weekFriday?: string | null): { startMs: number; endMs: number } | null {
  if (!weekFriday) return null;
  const monday = computeWeekMonday(weekFriday);
  const start = new Date(`${monday}T00:00:00Z`).getTime();
  const end = new Date(`${weekFriday}T23:59:59Z`).getTime();
  if (!Number.isFinite(start) || !Number.isFinite(end) || start >= end) return null;
  return { startMs: start, endMs: end };
}

// ---------------------------------------------------------------------------
// Chart helpers
// ---------------------------------------------------------------------------

function buildChartPoints(series: MarketsSeriesResponse): ChartPoint[] {
  return series.points
    .map((p) => ({
      timeMs: new Date(p.timestamp_utc).getTime(),
      polymarketAsk: p.polymarket_ask ?? p.polymarket_buy ?? null,
      polymarketBid: p.polymarket_bid ?? null,
      prn: p.pRN ?? null,
    }))
    .filter((p) => Number.isFinite(p.timeMs));
}

function buildPath(
  points: ChartPoint[],
  accessor: (p: ChartPoint) => number | null | undefined,
  scaleX: (t: number) => number,
  scaleY: (v: number) => number,
): string {
  const filtered = points.filter((p) => {
    const v = accessor(p);
    return v !== null && v !== undefined && Number.isFinite(v);
  });
  if (filtered.length < 2) return "";
  return filtered
    .map(
      (p, i) =>
        `${i === 0 ? "M" : "L"}${scaleX(p.timeMs).toFixed(1)},${scaleY(accessor(p) as number).toFixed(1)}`,
    )
    .join(" ");
}

function findNearestPoint(points: ChartPoint[], targetTime: number): ChartPoint | null {
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

// ---------------------------------------------------------------------------
// MarketDetailChart — single centered chart with 3 curves
// ---------------------------------------------------------------------------

const Y_LABELS = [0, 0.25, 0.5, 0.75, 1.0];
const X_TICK_COUNT = 5;
const PLOT_W = W - PL - PR;
const PLOT_H = H - PT - PB;

function MarketDetailChart({ series }: { series: MarketsSeriesResponse }) {
  const rawPoints = useMemo(() => buildChartPoints(series), [series]);
  const weekRange = useMemo(() => computeWeekRangeUtcMs(series.week_friday), [series.week_friday]);
  const points = useMemo(() => {
    if (!weekRange) return rawPoints;
    return rawPoints.filter((p) => p.timeMs >= weekRange.startMs && p.timeMs <= weekRange.endMs);
  }, [rawPoints, weekRange]);
  const [hovered, setHovered] = useState<ChartPoint | null>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);

  const minTime = useMemo(() => {
    if (weekRange) return weekRange.startMs;
    return Math.min(...points.map((p) => p.timeMs));
  }, [points, weekRange]);
  const maxTime = useMemo(() => {
    if (weekRange) return weekRange.endMs;
    return Math.max(...points.map((p) => p.timeMs));
  }, [points, weekRange]);
  const hasTime = Number.isFinite(minTime) && Number.isFinite(maxTime) && minTime < maxTime;

  const xScale = useCallback(
    (t: number) => PL + ((t - minTime) / (maxTime - minTime)) * PLOT_W,
    [minTime, maxTime],
  );
  const yScale = useCallback((v: number) => PT + PLOT_H * (1 - v), []);

  const bidPath = hasTime ? buildPath(points, (p) => p.polymarketBid, xScale, yScale) : "";
  const askPath = hasTime ? buildPath(points, (p) => p.polymarketAsk, xScale, yScale) : "";
  const prnPath = hasTime ? buildPath(points, (p) => p.prn, xScale, yScale) : "";

  const hasBid = points.some((p) => p.polymarketBid !== null && p.polymarketBid !== undefined);
  const hasAsk = points.some((p) => p.polymarketAsk !== null && p.polymarketAsk !== undefined);
  const hasPrn = points.some((p) => p.prn !== null && p.prn !== undefined);

  const xTicks = useMemo(() => {
    if (!hasTime) return [];
    return Array.from({ length: X_TICK_COUNT }, (_, i) => {
      const t = minTime + (i / (X_TICK_COUNT - 1)) * (maxTime - minTime);
      return { t, ...formatXTick(t) };
    });
  }, [hasTime, minTime, maxTime]);

  const handleMove = (event: React.MouseEvent<SVGSVGElement>) => {
    if (!hasTime) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const svgX = ((event.clientX - rect.left) / rect.width) * W;
    const ratio = Math.min(1, Math.max(0, (svgX - PL) / PLOT_W));
    const time = minTime + ratio * (maxTime - minTime);
    const nearest = findNearestPoint(points, time);
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
    <div className="mdc-wrap">
      <div className="mdc-header">
        <div className="mdc-title">
          {series.ticker}{" "}
          <span className="mdc-strike">${series.threshold.toFixed(2)}</span>
        </div>
        <div className="mdc-legend">
          {hasBid && (
            <span className="mdc-legend-item">
              <span className="mdc-swatch mdc-swatch-bid" />
              Polymarket Bid
            </span>
          )}
          {hasAsk && (
            <span className="mdc-legend-item">
              <span className="mdc-swatch mdc-swatch-ask" />
              Polymarket Ask
            </span>
          )}
          {hasPrn && (
            <span className="mdc-legend-item">
              <span className="mdc-swatch mdc-swatch-prn" />
              Risk-Neutral Prob
            </span>
          )}
        </div>
      </div>

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

          {bidPath && <path d={bidPath} className="chart-line chart-line-bid" />}
          {askPath && <path d={askPath} className="chart-line chart-line-ask" />}
          {prnPath && <path d={prnPath} className="chart-line chart-line-prn" />}

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
            {hasBid && (
              <div className="chart-tooltip-row">
                <span className="tt-label tt-bid">PM Bid</span>
                <span>{formatPrice(hovered.polymarketBid)}</span>
              </div>
            )}
            {hasAsk && (
              <div className="chart-tooltip-row">
                <span className="tt-label tt-ask">PM Ask</span>
                <span>{formatPrice(hovered.polymarketAsk)}</span>
              </div>
            )}
            <div className="chart-tooltip-row">
              <span className="tt-label tt-prn">RN Prob</span>
              <span>{formatPrice(hovered.prn)}</span>
            </div>
          </div>
        )}
      </div>

      <div className="mdc-footer">
        <div className="mdc-footer-chips">
          {!hasBid && !hasAsk && (
            <span className="chip chip-warn">No Polymarket data</span>
          )}
          {!hasPrn && <span className="chip chip-warn">No pRN data</span>}
        </div>
        <div className="mdc-footer-meta">
          {series.points.length} pts · Market {series.market_id ?? "--"} · Event{" "}
          {series.event_id ?? "--"}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// MarketsPage
// ---------------------------------------------------------------------------

export default function MarketsPage() {
  const { anyJobRunning, activeJobs, primaryJob, maxActiveJobs } = useAnyJobRunning();
  const { jobStatus, setJobId, setJobStatus } = useMarketsJob();

  const [weekFriday, setWeekFriday] = useState(() => computeDefaultWeekFriday(DEFAULT_TIMEZONE));
  const [summary, setSummary] = useState<MarketsSummaryResponse | null>(null);
  const [seriesByTicker, setSeriesByTicker] = useState<MarketsSeriesByTickerResponse | null>(null);
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const [selectedStrike, setSelectedStrike] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadingSeries, setLoadingSeries] = useState(false);

  // Sorted ticker list derived from the summary
  const tickers = useMemo(() => {
    if (!summary) return [];
    const source =
      summary.trading_universe_tickers && summary.trading_universe_tickers.length > 0
        ? summary.trading_universe_tickers
        : summary.markets.map((m) => m.ticker);
    return Array.from(
      new Set(source.map((t) => t.trim().toUpperCase()).filter((t) => t.length > 0)),
    ).sort();
  }, [summary]);

  const availableStrikes = useMemo(
    () => seriesByTicker?.strikes.map((s) => s.threshold) ?? [],
    [seriesByTicker],
  );

  const selectedSeries = useMemo((): MarketsSeriesResponse | null => {
    if (!seriesByTicker || selectedStrike === null) return null;
    return (
      seriesByTicker.strikes.find((s) => Math.abs(s.threshold - selectedStrike) < 0.0001) ?? null
    );
  }, [seriesByTicker, selectedStrike]);

  // Keep strike selection only if still valid for the loaded ticker.
  useEffect(() => {
    if (availableStrikes.length === 0) {
      if (selectedStrike !== null) setSelectedStrike(null);
      return;
    }
    if (selectedStrike === null) return;
    const stillValid = availableStrikes.some((s) => Math.abs(s - selectedStrike) < 0.0001);
    if (!stillValid) setSelectedStrike(null);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [availableStrikes, selectedStrike]);

  // -------------------------------------------------------------------------
  // Derived job state
  // -------------------------------------------------------------------------

  const isRunning = jobStatus?.status === "queued" || jobStatus?.status === "running";

  const statusLabel =
    isRunning ? "Running"
    : jobStatus?.status === "finished" ? "Success"
    : jobStatus?.status === "failed" ? "Failed"
    : "Ready";

  const statusClass =
    isRunning ? "running"
    : jobStatus?.status === "finished" ? "success"
    : jobStatus?.status === "failed" ? "failed"
    : "idle";

  const barProgress = useMemo(() => {
    const p = jobStatus?.progress;
    if (!p) return null;
    return {
      total: p.total,
      completed: p.current,
      failed: 0,
      status: (
        jobStatus?.status === "finished" ? "completed"
        : jobStatus?.status === "failed" ? "failed"
        : "running"
      ) as "running" | "completed" | "failed",
    };
  }, [jobStatus?.progress, jobStatus?.status]);

  const progressLabel = barProgress
    ? `${barProgress.completed} / ${barProgress.total} tickers`
    : isRunning ? "Starting..." : statusLabel;

  const durationLabel = jobStatus?.result?.duration_s != null
    ? `${jobStatus.result.duration_s}s`
    : isRunning ? "Running..." : "--";

  const weekLabel = `${summary?.week_monday ?? computeWeekMonday(weekFriday)} → ${summary?.week_friday ?? weekFriday}`;

  const monitorItems = [
    { label: "Stage", value: jobStatus?.progress?.stage ?? "--" },
    { label: "Week", value: weekLabel },
    { label: "Markets", value: summary?.markets.length != null ? String(summary.markets.length) : "--" },
    { label: "Run ID", value: summary?.run_id ?? jobStatus?.result?.run_id ?? "--" },
    { label: "Last refresh", value: formatTimestamp(summary?.last_refresh_utc) },
    { label: "Duration", value: durationLabel },
    { label: "Last update", value: formatTimestamp(jobStatus?.finished_at ?? jobStatus?.started_at) },
  ];

  const stderr = jobStatus?.result?.stderr || jobStatus?.error || null;

  // -------------------------------------------------------------------------
  // Data loading
  // -------------------------------------------------------------------------

  const loadSummary = useCallback(async () => {
    try {
      const data = await getMarketsSummary({ weekFriday });
      setSummary(data);
      const available = (
        data.trading_universe_tickers && data.trading_universe_tickers.length > 0
          ? data.trading_universe_tickers
          : data.markets.map((m) => m.ticker)
      )
        .map((t) => t.trim().toUpperCase())
        .filter((t) => t.length > 0);
      setSelectedTicker((prev) => {
        const cur = prev?.trim().toUpperCase() ?? null;
        return cur && available.includes(cur) ? cur : (available[0] ?? null);
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [weekFriday]);

  const loadSeries = useCallback(
    async (ticker: string | null, runId?: string | null) => {
      if (!ticker) return;
      setLoadingSeries(true);
      try {
        const data = await getMarketsSeriesByTicker({
          ticker,
          weekFriday,
          runId: runId ?? undefined,
        });
        setSeriesByTicker(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setLoadingSeries(false);
      }
    },
    [weekFriday],
  );

  // -------------------------------------------------------------------------
  // Refresh job
  // -------------------------------------------------------------------------

  const handleRefresh = useCallback(async () => {
    if (anyJobRunning) {
      setError(
        `Another job is running (${primaryJob?.name ?? "unknown"}). Max active jobs: ${maxActiveJobs ?? "n/a"}.`,
      );
      return;
    }
    if (jobStatus?.status === "running" || jobStatus?.status === "queued") {
      setError("Markets refresh already in progress.");
      return;
    }
    setError(null);
    setJobStatus(null);
    setSeriesByTicker(null);
    setSummary(null);
    try {
      const status = await startMarketsRefresh({ week_friday: weekFriday, force_refresh: true });
      setJobId(status.job_id);
      setJobStatus(status);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [anyJobRunning, jobStatus?.status, maxActiveJobs, primaryJob?.name, weekFriday, setJobId, setJobStatus]);

  // On mount: resume in-progress job, load summary if finished, or auto-start
  useEffect(() => {
    const status = jobStatus?.status;
    if (status === "running" || status === "queued") return; // context is already polling
    if (status === "finished") { loadSummary(); return; }
    if (status === "failed") return; // user must explicitly retry
    handleRefresh();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // React to job completion/failure while the page is mounted
  const afterMountRef = useRef(false);
  useEffect(() => {
    if (!afterMountRef.current) {
      afterMountRef.current = true;
      return;
    }
    if (jobStatus?.status === "finished") {
      loadSummary();
    } else if (jobStatus?.status === "failed") {
      const message = jobStatus.result?.stderr || jobStatus.error || "Markets refresh failed.";
      if (message) setError(message.trim());
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobStatus?.status]);

  // Load series when ticker changes, passing the current run_id for cache-busting
  useEffect(() => {
    if (!selectedTicker) return;
    loadSeries(selectedTicker, summary?.run_id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTicker, summary?.run_id]);

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <div className="page markets-page">
      <PipelineStatusCard className="page-sticky-meta" activeJobsCount={activeJobs.length} />
      <header className="page-header">
        <div>
          <p className="page-kicker">Weekly Markets</p>
          <h1 className="page-title">Markets Refresh</h1>
          <p className="page-subtitle">
            Hourly Polymarket bid/ask and Black-Scholes risk-neutral probability
            curves — strictly historical, no look-ahead.
          </p>
        </div>
      </header>

      <div className="markets-pipeline-grid">
        {/* ---- Left: Controls ---- */}
        <div className="markets-pipeline-main">
          <section className="panel">
            <div className="panel-header">
              <div>
                <h2>Run Configuration</h2>
                <p className="panel-hint">Select week and tickers, then refresh</p>
              </div>
              {selectedTicker && (
                <span className="meta-pill">Selected: {selectedTicker}</span>
              )}
            </div>
            <div className="panel-body">
              <div className="markets-fields">
                <div className="markets-field">
                  <label className="markets-field-label">Week Friday</label>
                  <input
                    type="date"
                    value={weekFriday}
                    className="markets-input"
                    onChange={(e) => setWeekFriday(e.target.value)}
                  />
                </div>
              </div>

              <div className="markets-selector">
                <div className="markets-field-label">Trading Universe</div>
                <div
                  className={`ticker-universe ${!tickers.length ? "is-disabled" : ""}`}
                  role="group"
                >
                  {tickers.map((ticker) => {
                    const isActive = ticker === selectedTicker;
                    return (
                      <button
                        key={ticker}
                        type="button"
                        className={`ticker-pill ${isActive ? "is-active" : ""}`}
                        onClick={() => {
                          setSelectedTicker(ticker);
                          setSelectedStrike(null);
                          setSeriesByTicker(null);
                        }}
                        aria-pressed={isActive}
                      >
                        {ticker}
                      </button>
                    );
                  })}
                </div>
                {!tickers.length && (
                  <div className="markets-hint">No trading universe tickers available yet.</div>
                )}
              </div>

              <div className="markets-selector">
                <div className="markets-field-label">Strikes</div>
                {!selectedTicker && (
                  <div className="markets-hint">Select a ticker to load strikes.</div>
                )}
                {selectedTicker && loadingSeries && (
                  <div className="markets-hint">Loading strikes…</div>
                )}
                {selectedTicker && !loadingSeries && availableStrikes.length === 0 && (
                  <div className="markets-hint">No strikes loaded for this ticker.</div>
                )}
                {selectedTicker && !loadingSeries && availableStrikes.length > 0 && (
                  <div className="strike-list" role="group">
                    {seriesByTicker?.strikes.map((series) => {
                      const isActive = selectedStrike != null &&
                        Math.abs(series.threshold - selectedStrike) < 0.0001;
                      return (
                        <button
                          key={series.threshold}
                          type="button"
                          className={`strike-pill ${isActive ? "is-active" : ""}`}
                          onClick={() => setSelectedStrike(series.threshold)}
                          aria-pressed={isActive}
                        >
                          ${series.threshold.toFixed(2)}
                          {series.market_id && (
                            <span className="strike-pill-market">mkt {series.market_id}</span>
                          )}
                          <span className="strike-pill-pts">
                            {series.points.length} pts
                          </span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>

              {error && (
                <div className="error-banner">
                  <strong>Markets refresh error</strong>
                  <pre>{error}</pre>
                </div>
              )}

              <div className="panel-actions">
                <button
                  className="button primary large"
                  onClick={handleRefresh}
                  disabled={isRunning || anyJobRunning}
                >
                  {isRunning ? "Refreshing..." : "Refresh Now"}
                </button>
              </div>
            </div>
          </section>
        </div>

        {/* ---- Right: Monitor ---- */}
        <div className="markets-pipeline-sidebar">
          <section className="panel">
            <div className="panel-header">
              <h2>Latest Run Output</h2>
              <span className="panel-hint">Real-time job status</span>
            </div>
            <div className="panel-body">
              <div className="run-output">
                <div className="pipeline-run-monitor">
                  <div className="pipeline-run-monitor-header">
                    <div>
                      <span className="meta-label">Run monitor</span>
                      <div className="pipeline-run-monitor-title">
                        Weekly markets run
                      </div>
                    </div>
                    <span className={`status-pill ${statusClass}`}>
                      {statusLabel}
                    </span>
                  </div>
                  <div className="pipeline-run-monitor-grid">
                    {monitorItems.map((item) => (
                      <div key={item.label}>
                        <span className="meta-label">{item.label}</span>
                        <span>{item.value}</span>
                      </div>
                    ))}
                  </div>
                  <div className="pipeline-run-monitor-progress">
                    <div className="pipeline-run-monitor-progress-header">
                      <span>Progress</span>
                      <span>{progressLabel}</span>
                    </div>
                    <div className="pipeline-progress-stack">
                      <PipelineProgressBar
                        title="Markets refresh"
                        progress={barProgress}
                        running={isRunning}
                        runningLabel="Running refresh..."
                        idleLabel={statusLabel}
                        unitLabel="tickers"
                        forceError={jobStatus?.status === "failed"}
                      />
                    </div>
                    <p className="pipeline-run-monitor-note">
                      Progress updates after each ticker is fully processed.
                    </p>
                  </div>
                </div>

                {stderr && (
                  <div className="log-block">
                    <span className="meta-label">Errors / Warnings</span>
                    <pre className="log-content">{stderr}</pre>
                  </div>
                )}
              </div>
            </div>
          </section>
        </div>
      </div>

      {/* ---- Single centered chart ---- */}
      <section className="markets-chart-section">
        {loadingSeries && <div className="markets-loading">Loading series…</div>}
        {!loadingSeries && !selectedSeries && !error && (
          <div className="markets-loading">
            {!selectedTicker
              ? "Select a ticker to begin."
              : availableStrikes.length === 0
                ? "No strike data available for this ticker."
                : "Select a strike to view the chart."}
          </div>
        )}
        {!loadingSeries && selectedSeries && (
          <MarketDetailChart series={selectedSeries} />
        )}
      </section>
    </div>
  );
}
