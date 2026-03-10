import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import {
  getRunDailyAnalyticsBreakdown,
  getRunDailyAnalyticsSummary,
  listPipelineRuns,
  type PipelineRunSummary,
  type RunDailyAnalyticsBreakdownResponse,
  type RunDailyAnalyticsDay,
  type RunDailyAnalyticsResponse,
  type RunDailyAnalyticsSnapshot,
  type RunDailyAnalyticsStructureBucket,
} from "../api/polymarketHistory";
import "./AnalyticsPage.css";

type TokenRole = "all" | "yes" | "no";
type BreakdownGroupBy = "ticker" | "market";

const formatDateTime = (value?: string | null) => {
  if (!value) return "--";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
};

const formatDate = (value?: string | null) => {
  if (!value) return "--";
  const parsed = new Date(`${value}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
    timeZone: "UTC",
  });
};

const formatCount = (value?: number | null) => {
  if (value == null || !Number.isFinite(value)) return "--";
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(value);
};

const formatVolume = (value?: number | null) => {
  if (value == null || !Number.isFinite(value)) return "--";
  return new Intl.NumberFormat(undefined, {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
};

const formatDeltaVolume = (value?: number | null) => {
  if (value == null || !Number.isFinite(value)) return "--";
  const sign = value > 0 ? "+" : value < 0 ? "-" : "";
  return `${sign}${formatVolume(Math.abs(value))}`;
};

const formatPercent = (value?: number | null) => {
  if (value == null || !Number.isFinite(value)) return "Unavailable";
  return new Intl.NumberFormat(undefined, {
    style: "percent",
    maximumFractionDigits: 1,
    signDisplay: "always",
  }).format(value);
};

const formatRatio = (value?: number | null) => {
  if (value == null || !Number.isFinite(value)) return "--";
  return `${value.toFixed(2)}x`;
};

function runDisplayName(run: PipelineRunSummary): string {
  const label = (run.label ?? "").trim();
  return label ? `${label} (${run.run_id})` : run.run_id;
}

function stateLabel(snapshot: RunDailyAnalyticsSnapshot | null): string {
  if (!snapshot) return "Unavailable";
  if (snapshot.flagged_outlier) return "Flagged outlier";
  if (snapshot.unusually_active) return "Above band";
  if (snapshot.unusually_inactive) return "Below band";
  return "Within band";
}

function stateTone(snapshot: RunDailyAnalyticsSnapshot | null): string {
  if (!snapshot) return "";
  if (snapshot.flagged_outlier) return "flagged";
  if (snapshot.unusually_active) return "positive";
  if (snapshot.unusually_inactive) return "negative";
  return "";
}

type MetricCardProps = {
  label: string;
  value: string;
  detail: string;
  tone?: string;
};

function MetricCard({ label, value, detail, tone = "" }: MetricCardProps) {
  return (
    <div className={`analytics-metric-card${tone ? ` ${tone}` : ""}`}>
      <span className="meta-label">{label}</span>
      <strong>{value}</strong>
      <span>{detail}</span>
    </div>
  );
}

function StructureTable({
  title,
  rows,
}: {
  title: string;
  rows: RunDailyAnalyticsStructureBucket[];
}) {
  return (
    <section className="panel analytics-panel">
      <div className="panel-header">
        <div>
          <h3>{title}</h3>
          <span className="panel-hint">Mean, median, and total observed notional volume.</span>
        </div>
      </div>
      <div className="panel-body">
        {rows.length ? (
          <div className="table-container">
            <table className="preview-table analytics-table compact">
              <thead>
                <tr>
                  <th>Group</th>
                  <th>Obs</th>
                  <th>Mean</th>
                  <th>Median</th>
                  <th>Total</th>
                  <th>Share</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.key}>
                    <td>{row.label}</td>
                    <td>{formatCount(row.observations)}</td>
                    <td>{formatVolume(row.mean_daily_notional_volume)}</td>
                    <td>{formatVolume(row.median_daily_notional_volume)}</td>
                    <td>{formatVolume(row.total_notional_volume)}</td>
                    <td>{formatPercent(row.share_total_notional_volume)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state">No structural observations available.</div>
        )}
      </div>
    </section>
  );
}

function VolumeBandChart({
  days,
  selectedDate,
  onSelectDate,
}: {
  days: RunDailyAnalyticsDay[];
  selectedDate: string | null;
  onSelectDate: (value: string) => void;
}) {
  const chartDays = days.slice(-60);
  if (chartDays.length < 2) {
    return <div className="empty-state">At least two days are needed to render the volume chart.</div>;
  }

  const width = 960;
  const height = 320;
  const padding = { top: 18, right: 18, bottom: 36, left: 56 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const values = chartDays.flatMap((day) => [
    day.daily_notional_volume,
    day.expected_notional_volume ?? 0,
    day.band_hi ?? 0,
  ]);
  const maxValue = Math.max(...values, 1);
  const maxX = Math.max(chartDays.length - 1, 1);
  const xFor = (index: number) => padding.left + (index / maxX) * plotWidth;
  const yFor = (value: number) => padding.top + plotHeight - (value / maxValue) * plotHeight;
  const linePath = (
    accessor: (day: RunDailyAnalyticsDay) => number | null,
    indexList: RunDailyAnalyticsDay[],
  ) => {
    let seen = false;
    const points = indexList
      .map((day, index) => {
        const value = accessor(day);
        if (value == null || !Number.isFinite(value)) return null;
        const command = seen ? "L" : "M";
        seen = true;
        return `${command} ${xFor(index).toFixed(2)} ${yFor(value).toFixed(2)}`;
      })
      .filter(Boolean);
    return points.join(" ");
  };
  const areaTop = chartDays
    .map((day, index) => {
      const value = day.band_hi ?? day.expected_notional_volume ?? day.daily_notional_volume;
      return `${index === 0 ? "M" : "L"} ${xFor(index).toFixed(2)} ${yFor(value).toFixed(2)}`;
    })
    .join(" ");
  const areaBottom = [...chartDays]
    .reverse()
    .map((day, index) => {
      const value = day.band_lo ?? day.expected_notional_volume ?? day.daily_notional_volume;
      const xIndex = chartDays.length - index - 1;
      return `L ${xFor(xIndex).toFixed(2)} ${yFor(value).toFixed(2)}`;
    })
    .join(" ");
  const areaPath = `${areaTop} ${areaBottom} Z`;

  return (
    <div className="analytics-chart-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} className="analytics-chart" role="img">
        <path d={areaPath} className="analytics-chart-band" />
        <path
          d={linePath((day) => day.expected_notional_volume, chartDays)}
          className="analytics-chart-expected"
        />
        <path
          d={linePath((day) => day.daily_notional_volume, chartDays)}
          className="analytics-chart-actual"
        />
        {chartDays.map((day, index) => (
          <g key={day.date}>
            <circle
              cx={xFor(index)}
              cy={yFor(day.daily_notional_volume)}
              r={day.date === selectedDate ? 5 : 3.5}
              className={`analytics-chart-point${day.date === selectedDate ? " selected" : ""}`}
              onClick={() => onSelectDate(day.date)}
            />
          </g>
        ))}
        <text x={padding.left} y={height - 10} className="analytics-chart-label">
          {formatDate(chartDays[0]?.date)}
        </text>
        <text x={width - padding.right} y={height - 10} textAnchor="end" className="analytics-chart-label">
          {formatDate(chartDays[chartDays.length - 1]?.date)}
        </text>
        <text x={padding.left} y={padding.top + 10} className="analytics-chart-label">
          {formatVolume(maxValue)}
        </text>
      </svg>
    </div>
  );
}

export default function AnalyticsPage() {
  const [runs, setRuns] = useState<PipelineRunSummary[]>([]);
  const [loadingRuns, setLoadingRuns] = useState<boolean>(true);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [dateMin, setDateMin] = useState<string>("");
  const [dateMax, setDateMax] = useState<string>("");
  const [selectedTickers, setSelectedTickers] = useState<string[]>([]);
  const [tokenRole, setTokenRole] = useState<TokenRole>("all");
  const [ciLevel, setCiLevel] = useState<number>(90);
  const [excludeFlagged, setExcludeFlagged] = useState<boolean>(false);
  const [analyticsLoading, setAnalyticsLoading] = useState<boolean>(false);
  const [analyticsError, setAnalyticsError] = useState<string | null>(null);
  const [analyticsData, setAnalyticsData] = useState<RunDailyAnalyticsResponse | null>(null);
  const [selectedBreakdownDate, setSelectedBreakdownDate] = useState<string | null>(null);
  const [breakdownGroupBy, setBreakdownGroupBy] = useState<BreakdownGroupBy>("ticker");
  const [breakdownLoading, setBreakdownLoading] = useState<boolean>(false);
  const [breakdownError, setBreakdownError] = useState<string | null>(null);
  const [breakdownData, setBreakdownData] = useState<RunDailyAnalyticsBreakdownResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoadingRuns(true);
    setRunsError(null);

    listPipelineRuns()
      .then((payload) => {
        if (cancelled) return;
        setRuns(payload.runs ?? []);
      })
      .catch((err) => {
        if (cancelled) return;
        setRunsError(err instanceof Error ? err.message : "Failed to load Polymarket runs.");
      })
      .finally(() => {
        if (cancelled) return;
        setLoadingRuns(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!runs.length) {
      if (selectedRunId) setSelectedRunId("");
      return;
    }
    const stillValid = runs.some((run) => run.run_id === selectedRunId);
    if (stillValid) return;
    const preferred = runs.find((run) => run.is_active) ?? runs[0];
    setSelectedRunId(preferred?.run_id ?? "");
  }, [runs, selectedRunId]);

  useEffect(() => {
    setDateMin("");
    setDateMax("");
    setSelectedTickers([]);
    setTokenRole("all");
    setCiLevel(90);
    setExcludeFlagged(false);
    setBreakdownGroupBy("ticker");
    setSelectedBreakdownDate(null);
    setBreakdownData(null);
    setBreakdownError(null);
  }, [selectedRunId]);

  const selectedRun = useMemo(
    () => runs.find((run) => run.run_id === selectedRunId) ?? null,
    [runs, selectedRunId],
  );

  const readiness = selectedRun?.analytics ?? null;
  const isReady = Boolean(readiness?.analytics_ready);
  const statusClass = isReady ? "success" : "failed";
  const dateRangeLabel =
    selectedRun?.start_date && selectedRun?.end_date
      ? `${selectedRun.start_date} to ${selectedRun.end_date}`
      : "--";
  const tickersLabel =
    selectedRun?.tickers && selectedRun.tickers.length
      ? selectedRun.tickers.join(", ")
      : "--";

  const analyticsQuery = useMemo(
    () => ({
      dateMin: dateMin || undefined,
      dateMax: dateMax || undefined,
      tickers: selectedTickers.length ? selectedTickers : undefined,
      tokenRole,
      ciLevel,
      excludeFlagged,
    }),
    [ciLevel, dateMax, dateMin, excludeFlagged, selectedTickers, tokenRole],
  );

  useEffect(() => {
    let cancelled = false;

    if (!selectedRunId || !isReady) {
      setAnalyticsData(null);
      setAnalyticsError(null);
      setAnalyticsLoading(false);
      return () => {
        cancelled = true;
      };
    }

    setAnalyticsLoading(true);
    setAnalyticsError(null);

    getRunDailyAnalyticsSummary(selectedRunId, analyticsQuery)
      .then((payload) => {
        if (cancelled) return;
        setAnalyticsData(payload);
        if (!selectedBreakdownDate || !payload.days.some((day) => day.date === selectedBreakdownDate)) {
          setSelectedBreakdownDate(payload.latest_trade_date);
        }
      })
      .catch((err) => {
        if (cancelled) return;
        setAnalyticsData(null);
        setAnalyticsError(err instanceof Error ? err.message : "Failed to load daily analytics.");
      })
      .finally(() => {
        if (cancelled) return;
        setAnalyticsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [analyticsQuery, isReady, selectedBreakdownDate, selectedRunId]);

  useEffect(() => {
    let cancelled = false;
    if (!selectedRunId || !isReady || !selectedBreakdownDate) {
      setBreakdownData(null);
      setBreakdownError(null);
      setBreakdownLoading(false);
      return () => {
        cancelled = true;
      };
    }

    setBreakdownLoading(true);
    setBreakdownError(null);
    getRunDailyAnalyticsBreakdown(selectedRunId, {
      ...analyticsQuery,
      date: selectedBreakdownDate,
      groupBy: breakdownGroupBy,
    })
      .then((payload) => {
        if (cancelled) return;
        setBreakdownData(payload);
      })
      .catch((err) => {
        if (cancelled) return;
        setBreakdownData(null);
        setBreakdownError(err instanceof Error ? err.message : "Failed to load daily breakdown.");
      })
      .finally(() => {
        if (cancelled) return;
        setBreakdownLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [analyticsQuery, breakdownGroupBy, isReady, selectedBreakdownDate, selectedRunId]);

  const availableTickers = analyticsData?.available_tickers ?? selectedRun?.tickers ?? [];
  const latestDay = analyticsData?.summary.latest_day ?? null;
  const comparisonDay = analyticsData?.summary.comparison_day ?? null;
  const toggleTicker = (ticker: string) => {
    setSelectedTickers((current) =>
      current.includes(ticker)
        ? current.filter((value) => value !== ticker)
        : [...current, ticker],
    );
  };

  return (
    <section className="page analytics-page">
      {loadingRuns ? <div className="dashboard-banner">Loading analytics readiness…</div> : null}

      <header className="page-header analytics-page-header">
        <div>
          <p className="page-kicker">Analytics</p>
          <h1 className="page-title analytics-page-title">Analytics</h1>
          <p className="page-subtitle">
            True Polymarket trade volume only. This tab now includes filters,
            empirical daily baselines, {ciLevel}% bands, structural diagnostics,
            and day drilldowns.
          </p>
        </div>
      </header>

      {runsError ? (
        <div className="dashboard-banner error">
          Failed to load analytics readiness: {runsError}
        </div>
      ) : null}

      <div className="analytics-grid analytics-top-grid">
        <section className="panel analytics-panel">
          <div className="panel-header">
            <div>
              <h2>Run Selection</h2>
              <span className="panel-hint">
                Analytics only unlocks on runs that own a run-scoped <code>trades.csv</code> artifact.
              </span>
            </div>
          </div>
          <div className="panel-body">
            {runs.length === 0 && !loadingRuns ? (
              <div className="empty-state">No weekly-history runs found yet.</div>
            ) : (
              <>
                <div className="field">
                  <label htmlFor="analytics-run-select">Weekly history run</label>
                  <select
                    id="analytics-run-select"
                    className="input"
                    value={selectedRunId}
                    onChange={(event) => setSelectedRunId(event.target.value)}
                    disabled={!runs.length}
                  >
                    {runs.map((run) => (
                      <option key={run.run_id} value={run.run_id}>
                        {runDisplayName(run)}
                      </option>
                    ))}
                  </select>
                </div>

                {selectedRun ? (
                  <div className="analytics-facts-grid">
                    <div className="analytics-fact">
                      <span className="meta-label">Date range</span>
                      <span>{dateRangeLabel}</span>
                    </div>
                    <div className="analytics-fact">
                      <span className="meta-label">Markets</span>
                      <span>{formatCount(selectedRun.markets)}</span>
                    </div>
                    <div className="analytics-fact">
                      <span className="meta-label">Tickers</span>
                      <span>{tickersLabel}</span>
                    </div>
                    <div className="analytics-fact">
                      <span className="meta-label">Created</span>
                      <span>{formatDateTime(selectedRun.created_at_utc)}</span>
                    </div>
                  </div>
                ) : null}
              </>
            )}
          </div>
        </section>

        <section className="panel analytics-panel">
          <div className="panel-header">
            <div>
              <h2>Readiness</h2>
              <span className="panel-hint">
                Primary metric stays fixed to <code>daily_notional_volume</code>.
              </span>
            </div>
            {selectedRun ? (
              <span className={`status-pill ${statusClass}`}>{isReady ? "Ready" : "Blocked"}</span>
            ) : null}
          </div>
          <div className="panel-body analytics-readiness-body">
            {!selectedRun ? (
              <div className="empty-state">Select a run to inspect analytics readiness.</div>
            ) : readiness ? (
              <>
                <p className="analytics-readiness-copy">
                  {isReady
                    ? "This run owns a run-scoped trades artifact and can support the full analytics workflow."
                    : readiness.warning ?? "This run cannot support true-volume analytics yet."}
                </p>

                <div className="analytics-facts-grid">
                  <div className="analytics-fact">
                    <span className="meta-label">Include subgraph</span>
                    <span>{readiness.include_subgraph_requested ? "Yes" : "No"}</span>
                  </div>
                  <div className="analytics-fact">
                    <span className="meta-label">Trade rows</span>
                    <span>{formatCount(readiness.trade_entities)}</span>
                  </div>
                  <div className="analytics-fact">
                    <span className="meta-label">Trade days</span>
                    <span>{formatCount(readiness.trade_days)}</span>
                  </div>
                  <div className="analytics-fact">
                    <span className="meta-label">Trade artifact</span>
                    <span>{readiness.trade_artifact_path ?? "--"}</span>
                  </div>
                </div>

                {!isReady ? (
                  <div className="analytics-callout">
                    <p>
                      Rebuild the weekly history run with subgraph trade ingestion enabled to unlock the analytics workflow.
                    </p>
                    <Link className="button light" to="/polymarket-history-builder">
                      Open Polymarket History Builder
                    </Link>
                  </div>
                ) : null}
              </>
            ) : (
              <div className="empty-state">Readiness metadata is unavailable for the selected run.</div>
            )}
          </div>
        </section>
      </div>

      <section className="panel analytics-panel">
        <div className="panel-header">
          <div>
            <h2>Filters</h2>
            <span className="panel-hint">
              Ticker, date, token-role, band level, and outlier exclusion all recalculate the diagnostics.
            </span>
          </div>
        </div>
        <div className="panel-body">
          {!selectedRun ? (
            <div className="empty-state">Select a run before applying analytics filters.</div>
          ) : (
            <>
              <div className="analytics-filter-grid">
                <div className="field">
                  <label htmlFor="analytics-date-min">Date min</label>
                  <input
                    id="analytics-date-min"
                    className="input"
                    type="date"
                    value={dateMin}
                    onChange={(event) => setDateMin(event.target.value)}
                    disabled={!isReady}
                  />
                </div>
                <div className="field">
                  <label htmlFor="analytics-date-max">Date max</label>
                  <input
                    id="analytics-date-max"
                    className="input"
                    type="date"
                    value={dateMax}
                    onChange={(event) => setDateMax(event.target.value)}
                    disabled={!isReady}
                  />
                </div>
                <div className="field">
                  <label htmlFor="analytics-token-role">Token role</label>
                  <select
                    id="analytics-token-role"
                    className="input"
                    value={tokenRole}
                    onChange={(event) => setTokenRole(event.target.value as TokenRole)}
                    disabled={!isReady}
                  >
                    <option value="all">All</option>
                    <option value="yes">Yes only</option>
                    <option value="no">No only</option>
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="analytics-ci-level">Confidence band</label>
                  <select
                    id="analytics-ci-level"
                    className="input"
                    value={ciLevel}
                    onChange={(event) => setCiLevel(Number(event.target.value))}
                    disabled={!isReady}
                  >
                    <option value={90}>90%</option>
                    <option value={95}>95%</option>
                    <option value={99}>99%</option>
                  </select>
                </div>
              </div>

              <div className="analytics-filter-actions">
                <label className="analytics-checkbox">
                  <input
                    type="checkbox"
                    checked={excludeFlagged}
                    onChange={(event) => setExcludeFlagged(event.target.checked)}
                    disabled={!isReady}
                  />
                  Exclude flagged outlier days from rolling baselines
                </label>
                <button
                  type="button"
                  className="button light"
                  onClick={() => {
                    setDateMin("");
                    setDateMax("");
                    setSelectedTickers([]);
                    setTokenRole("all");
                    setCiLevel(90);
                    setExcludeFlagged(false);
                  }}
                  disabled={!isReady}
                >
                  Reset Filters
                </button>
              </div>

              <div className="analytics-ticker-picker">
                <span className="meta-label">Tickers</span>
                <div className="analytics-chip-list">
                  {availableTickers.length ? (
                    availableTickers.map((ticker) => (
                      <button
                        key={ticker}
                        type="button"
                        className={`analytics-chip${selectedTickers.includes(ticker) ? " active" : ""}`}
                        onClick={() => toggleTicker(ticker)}
                        disabled={!isReady}
                      >
                        {ticker}
                      </button>
                    ))
                  ) : (
                    <span className="analytics-chip muted">No ticker metadata</span>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      </section>

      <section className="panel analytics-panel">
        <div className="panel-header">
          <div>
            <h2>Daily Summary</h2>
            <span className="panel-hint">
              Actual volume, expected volume, rolling noise, and {ciLevel}% confidence bands.
            </span>
          </div>
        </div>
        <div className="panel-body">
          {!selectedRun ? (
            <div className="empty-state">Select a run to load analytics.</div>
          ) : !isReady ? (
            <div className="analytics-scope-list">
              <div className="analytics-scope-item">
                <span className="meta-label">Current state</span>
                <span>This run is blocked before analytics can load.</span>
              </div>
              <div className="analytics-scope-item">
                <span className="meta-label">Required action</span>
                <span>Rebuild with subgraph trade ingestion enabled to unlock the full analytics tab.</span>
              </div>
            </div>
          ) : analyticsLoading ? (
            <div className="dashboard-banner">Loading daily analytics…</div>
          ) : analyticsError ? (
            <div className="dashboard-banner error">{analyticsError}</div>
          ) : analyticsData ? (
            <>
              <div className="analytics-metric-grid">
                <MetricCard
                  label="Latest Trade Day"
                  value={latestDay ? formatVolume(latestDay.daily_notional_volume) : "--"}
                  detail={formatDate(analyticsData.latest_trade_date)}
                />
                <MetricCard
                  label="Expected Volume"
                  value={latestDay ? formatVolume(latestDay.expected_notional_volume) : "--"}
                  detail={
                    latestDay
                      ? `${formatVolume(latestDay.band_lo)} to ${formatVolume(latestDay.band_hi)}`
                      : "Unavailable"
                  }
                />
                <MetricCard
                  label="Day-over-Day Change"
                  value={comparisonDay ? formatDeltaVolume(analyticsData.summary.delta_notional) : "Unavailable"}
                  detail={
                    comparisonDay
                      ? `${formatDate(analyticsData.comparison_date)} to ${formatDate(analyticsData.latest_trade_date)} • ${formatPercent(analyticsData.summary.delta_notional_pct)}`
                      : "Needs a usable prior day."
                  }
                  tone={latestDay?.day_over_day_delta ? (latestDay.day_over_day_delta > 0 ? "positive" : "negative") : ""}
                />
                <MetricCard
                  label="Latest State"
                  value={stateLabel(latestDay)}
                  detail={latestDay ? `${formatRatio(latestDay.realized_expected_ratio)} vs expected` : "--"}
                  tone={stateTone(latestDay)}
                />
                <MetricCard
                  label="Noise CV (28d)"
                  value={latestDay ? formatPercent(latestDay.noise_cv_28d) : "--"}
                  detail="Day-to-day variability relative to the mean"
                />
                <MetricCard
                  label="MAD Ratio (28d)"
                  value={latestDay ? formatPercent(latestDay.noise_mad_ratio_28d) : "--"}
                  detail="Robust variability relative to the median"
                />
              </div>

              <div className="analytics-metric-grid analytics-secondary-grid">
                <MetricCard
                  label="Share Volume"
                  value={latestDay ? formatCount(latestDay.share_volume) : "--"}
                  detail={`Observed on ${formatDate(analyticsData.latest_trade_date)}`}
                />
                <MetricCard
                  label="Trade Count"
                  value={latestDay ? formatCount(latestDay.trade_count) : "--"}
                  detail={`Observed on ${formatDate(analyticsData.latest_trade_date)}`}
                />
                <MetricCard
                  label="Active Markets"
                  value={latestDay ? formatCount(latestDay.active_markets) : "--"}
                  detail={`Observed on ${formatDate(analyticsData.latest_trade_date)}`}
                />
                <MetricCard
                  label="Active Tickers"
                  value={latestDay ? formatCount(latestDay.active_tickers) : "--"}
                  detail={`Observed on ${formatDate(analyticsData.latest_trade_date)}`}
                />
              </div>

              <div className="analytics-facts-grid analytics-coverage-grid">
                <div className="analytics-fact">
                  <span className="meta-label">Observed trade days</span>
                  <span>{formatCount(analyticsData.coverage.observed_trade_days)}</span>
                </div>
                <div className="analytics-fact">
                  <span className="meta-label">Calendar days returned</span>
                  <span>{formatCount(analyticsData.coverage.filled_days)}</span>
                </div>
                <div className="analytics-fact">
                  <span className="meta-label">Valid trade rows</span>
                  <span>{formatCount(analyticsData.coverage.valid_trade_rows)}</span>
                </div>
                <div className="analytics-fact">
                  <span className="meta-label">Effective range</span>
                  <span>
                    {analyticsData.coverage.effective_date_min ?? "--"} to {analyticsData.coverage.effective_date_max ?? "--"}
                  </span>
                </div>
              </div>

              <VolumeBandChart
                days={analyticsData.days}
                selectedDate={selectedBreakdownDate}
                onSelectDate={setSelectedBreakdownDate}
              />

              {analyticsData.warnings.length ? (
                <div className="analytics-warning-list">
                  {analyticsData.warnings.map((warning) => (
                    <div key={warning} className="analytics-warning-item">
                      {warning}
                    </div>
                  ))}
                </div>
              ) : null}

              <div className="analytics-table-heading">
                <div>
                  <h3>Daily Observations</h3>
                  <p>Click a day to inspect ticker or market drilldowns for that date.</p>
                </div>
              </div>

              <div className="table-container">
                <table className="preview-table analytics-table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Actual</th>
                      <th>Expected</th>
                      <th>Band</th>
                      <th>Ratio</th>
                      <th>DoD</th>
                      <th>Flags</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analyticsData.days.map((day) => (
                      <tr
                        key={day.date}
                        className={day.date === selectedBreakdownDate ? "selected" : ""}
                        onClick={() => setSelectedBreakdownDate(day.date)}
                      >
                        <td>{formatDate(day.date)}</td>
                        <td>{formatVolume(day.daily_notional_volume)}</td>
                        <td>{formatVolume(day.expected_notional_volume)}</td>
                        <td>
                          {formatVolume(day.band_lo)} to {formatVolume(day.band_hi)}
                        </td>
                        <td>{formatRatio(day.realized_expected_ratio)}</td>
                        <td>{formatDeltaVolume(day.day_over_day_delta)}</td>
                        <td>
                          {day.flagged_outlier
                            ? "Outlier"
                            : day.unusually_active
                              ? "Above band"
                              : day.unusually_inactive
                                ? "Below band"
                                : "Within band"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <div className="empty-state">Observed analytics are unavailable for the selected run.</div>
          )}
        </div>
      </section>

      {analyticsData ? (
        <div className="analytics-structure-grid">
          <StructureTable title="Weekday Effects" rows={analyticsData.structure.weekday} />
          <StructureTable title="Event Proximity" rows={analyticsData.structure.event_proximity} />
          <StructureTable title="Ladder Buckets" rows={analyticsData.structure.ladder_bucket} />
        </div>
      ) : null}

      <section className="panel analytics-panel">
        <div className="panel-header">
          <div>
            <h2>Day Drilldown</h2>
            <span className="panel-hint">
              Inspect how the selected day distributes across tickers or individual markets.
            </span>
          </div>
        </div>
        <div className="panel-body">
          <div className="analytics-breakdown-toolbar">
            <div>
              <span className="meta-label">Selected day</span>
              <div className="analytics-breakdown-date">{formatDate(selectedBreakdownDate)}</div>
            </div>
            <div className="analytics-chip-list">
              <button
                type="button"
                className={`analytics-chip${breakdownGroupBy === "ticker" ? " active" : ""}`}
                onClick={() => setBreakdownGroupBy("ticker")}
                disabled={!isReady}
              >
                By ticker
              </button>
              <button
                type="button"
                className={`analytics-chip${breakdownGroupBy === "market" ? " active" : ""}`}
                onClick={() => setBreakdownGroupBy("market")}
                disabled={!isReady}
              >
                By market
              </button>
            </div>
          </div>

          {!selectedRun || !isReady ? (
            <div className="empty-state">Unlock analytics to inspect daily drilldowns.</div>
          ) : !selectedBreakdownDate ? (
            <div className="empty-state">Select a day from the chart or daily table.</div>
          ) : breakdownLoading ? (
            <div className="dashboard-banner">Loading daily drilldown…</div>
          ) : breakdownError ? (
            <div className="dashboard-banner error">{breakdownError}</div>
          ) : breakdownData ? (
            <>
              {breakdownData.warnings.length ? (
                <div className="analytics-warning-list">
                  {breakdownData.warnings.map((warning) => (
                    <div key={warning} className="analytics-warning-item">
                      {warning}
                    </div>
                  ))}
                </div>
              ) : null}
              <div className="table-container">
                <table className="preview-table analytics-table">
                  <thead>
                    <tr>
                      <th>{breakdownGroupBy === "ticker" ? "Ticker" : "Market"}</th>
                      <th>Actual</th>
                      <th>Expected</th>
                      <th>Band</th>
                      <th>Ratio</th>
                      <th>Share</th>
                      <th>History</th>
                      <th>Source</th>
                      <th>Flags</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdownData.rows.map((row) => (
                      <tr key={`${row.key}-${row.label}`}>
                        <td>
                          <div className="analytics-breakdown-label">
                            <strong>{row.label}</strong>
                            {breakdownGroupBy === "market" ? (
                              <span>
                                {row.ticker ?? "--"} · {row.ladder_bucket ?? "--"} · {row.event_proximity_bucket ?? "--"}
                              </span>
                            ) : null}
                          </div>
                        </td>
                        <td>{formatVolume(row.daily_notional_volume)}</td>
                        <td>{formatVolume(row.expected_notional_volume)}</td>
                        <td>
                          {formatVolume(row.band_lo)} to {formatVolume(row.band_hi)}
                        </td>
                        <td>{formatRatio(row.realized_expected_ratio)}</td>
                        <td>{formatPercent(row.volume_share_of_day)}</td>
                        <td>{formatCount(row.history_days)}</td>
                        <td>{row.baseline_source}</td>
                        <td>
                          {row.flagged_outlier
                            ? "Outlier"
                            : row.unusually_active
                              ? "Above band"
                              : row.unusually_inactive
                                ? "Below band"
                                : "Within band"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <div className="empty-state">Drilldown data is unavailable for the selected day.</div>
          )}
        </div>
      </section>

      <section className="panel analytics-panel">
        <div className="panel-header">
          <div>
            <h2>Methodology</h2>
            <span className="panel-hint">Short operational notes for interpreting this page.</span>
          </div>
        </div>
        <div className="panel-body analytics-methodology">
          <p>
            Baselines use a trailing {analyticsData?.baseline_window_days ?? 28}-day rolling median on observed
            daily notional volume, with weekday-aware fallback when enough same-weekday history exists.
          </p>
          <p>
            Confidence bands use empirical residual quantiles on the same rolling history. Outlier flags come from
            a rolling median/MAD diagnostic and can optionally be excluded from baseline estimation.
          </p>
          <p>
            Event-proximity and ladder-bucket structure tables are descriptive diagnostics, not causal models.
            Market drilldowns may fall back to peer baselines when the market’s own history is too short.
          </p>
        </div>
      </section>
    </section>
  );
}
