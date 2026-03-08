import { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";

import {
  createResearchNote,
  exportAnalysisTableCsv,
  fetchAnalysisDrift,
  fetchAnalysisOverview,
  fetchAnalysisPriceBehavior,
  fetchAnalysisStructure,
  fetchAnalysisTable,
  fetchAnalysisVolume,
  fetchResearchNotes,
  getAnalysisJob,
  startAnalysisRefresh,
  type AnalysisDriftResponse,
  type AnalysisJobStatus,
  type AnalysisOverviewResponse,
  type AnalysisPriceBehaviorResponse,
  type AnalysisSeriesPoint,
  type AnalysisStructureResponse,
  type AnalysisTableResponse,
  type AnalysisVolumeResponse,
  type ResearchNote,
} from "../api/analysis";
import "./DataAnalysisPage.css";

const DEFAULT_VARIABLE = "notional_volume";
const DEFAULT_TABLE = "threshold_rule";
const FALLBACK_TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NFLX", "NVDA", "OPEN", "PLTR", "TSLA"];
const VARIABLE_LABELS: Record<string, string> = {
  notional_volume: "Notional volume",
  contract_volume: "Contract volume",
  trade_count: "Trade count",
  avg_trade_size: "Average trade size",
  rv_logit_intraday: "Intraday RV (logit)",
  active_market_count: "Active market count",
  volume_hhi_within_stock: "Stock HHI",
};
const TABLES_WITH_TICKER_FILTER = new Set(["outlier_flag", "fact_stock_day", "fact_market_day"]);
const TABLES_WITH_VARIABLE_FILTER = new Set(["threshold_rule", "outlier_flag", "variable_profile", "drift_monitor", "break_event"]);

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function formatNumber(value?: number | string | null): string {
  if (value === null || value === undefined) return "--";
  if (typeof value === "string") return value;
  if (!Number.isFinite(value)) return "--";
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return value.toFixed(2);
}

function formatPct(value?: number | null): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "--";
  return `${(value * 100).toFixed(1)}%`;
}

function toLineTrace(name: string, points: AnalysisSeriesPoint[], key: keyof AnalysisSeriesPoint, color: string) {
  return {
    type: "scatter",
    mode: "lines+markers",
    name,
    x: points.map((point) => point.date),
    y: points.map((point) => (typeof point[key] === "number" ? point[key] : null)),
    line: { color, width: 2 },
    marker: { size: 5 },
  };
}

export default function DataAnalysisPage() {
  const [selectedTicker, setSelectedTicker] = useState("");
  const [selectedVariable, setSelectedVariable] = useState(DEFAULT_VARIABLE);
  const [selectedTable, setSelectedTable] = useState(DEFAULT_TABLE);
  const [tablePage, setTablePage] = useState(1);
  const [overview, setOverview] = useState<AnalysisOverviewResponse | null>(null);
  const [volume, setVolume] = useState<AnalysisVolumeResponse | null>(null);
  const [structure, setStructure] = useState<AnalysisStructureResponse | null>(null);
  const [priceBehavior, setPriceBehavior] = useState<AnalysisPriceBehaviorResponse | null>(null);
  const [drift, setDrift] = useState<AnalysisDriftResponse | null>(null);
  const [tableData, setTableData] = useState<AnalysisTableResponse | null>(null);
  const [notes, setNotes] = useState<ResearchNote[]>([]);
  const [noteTitle, setNoteTitle] = useState("");
  const [noteBody, setNoteBody] = useState("");
  const [skipTradeBackfill, setSkipTradeBackfill] = useState(false);
  const [forceFullRebuild, setForceFullRebuild] = useState(false);
  const [refreshJob, setRefreshJob] = useState<AnalysisJobStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sectionErrors, setSectionErrors] = useState<Record<string, string | null>>({});
  const tableTickerFilter = TABLES_WITH_TICKER_FILTER.has(selectedTable) ? selectedTicker || undefined : undefined;
  const tableVariableFilter = TABLES_WITH_VARIABLE_FILTER.has(selectedTable) ? selectedVariable : undefined;

  const loadAll = async (options?: { keepLoading?: boolean }) => {
    if (options?.keepLoading !== false) setIsLoading(true);
    setError(null);
    const results = await Promise.allSettled([
      fetchAnalysisOverview(),
      fetchAnalysisVolume({ ticker: selectedTicker || undefined, variable_name: selectedVariable }),
      fetchAnalysisStructure(),
      fetchAnalysisPriceBehavior({ ticker: selectedTicker || undefined }),
      fetchAnalysisDrift({ ticker: selectedTicker || undefined, variable_name: selectedVariable }),
      fetchAnalysisTable({
        table: selectedTable,
        page: tablePage,
        page_size: 25,
        ticker: tableTickerFilter,
        variable_name: tableVariableFilter,
      }),
      fetchResearchNotes(),
    ]);

    const nextSectionErrors: Record<string, string | null> = {};
    let successCount = 0;

    const [
      overviewResult,
      volumeResult,
      structureResult,
      priceResult,
      driftResult,
      tableResult,
      notesResult,
    ] = results;

    if (overviewResult.status === "fulfilled") {
      setOverview(overviewResult.value);
      successCount += 1;
    } else {
      setOverview(null);
      nextSectionErrors.overview = getErrorMessage(overviewResult.reason);
    }
    if (volumeResult.status === "fulfilled") {
      setVolume(volumeResult.value);
      successCount += 1;
    } else {
      setVolume(null);
      nextSectionErrors.volume = getErrorMessage(volumeResult.reason);
    }
    if (structureResult.status === "fulfilled") {
      setStructure(structureResult.value);
      successCount += 1;
    } else {
      setStructure(null);
      nextSectionErrors.structure = getErrorMessage(structureResult.reason);
    }
    if (priceResult.status === "fulfilled") {
      setPriceBehavior(priceResult.value);
      successCount += 1;
    } else {
      setPriceBehavior(null);
      nextSectionErrors.priceBehavior = getErrorMessage(priceResult.reason);
    }
    if (driftResult.status === "fulfilled") {
      setDrift(driftResult.value);
      successCount += 1;
    } else {
      setDrift(null);
      nextSectionErrors.drift = getErrorMessage(driftResult.reason);
    }
    if (tableResult.status === "fulfilled") {
      setTableData(tableResult.value);
      successCount += 1;
    } else {
      setTableData(null);
      nextSectionErrors.tables = getErrorMessage(tableResult.reason);
    }
    if (notesResult.status === "fulfilled") {
      setNotes(notesResult.value.notes);
      successCount += 1;
    } else {
      setNotes([]);
      nextSectionErrors.notes = getErrorMessage(notesResult.reason);
    }

    setSectionErrors(nextSectionErrors);
    if (successCount === 0) {
      setError("All analysis sections failed to load. Check the backend and analysis database configuration.");
    }
    setIsLoading(false);
  };

  useEffect(() => {
    void loadAll();
  }, [selectedTicker, selectedVariable, selectedTable, tablePage]);

  useEffect(() => {
    if (!refreshJob || !["queued", "running"].includes(refreshJob.status)) return;
    const timer = window.setInterval(() => {
      void getAnalysisJob(refreshJob.job_id)
        .then((payload) => {
          setRefreshJob(payload);
          if (payload.status === "finished") {
            void loadAll({ keepLoading: false });
          }
          if (payload.status === "failed") {
            setError(payload.error ?? "Analysis refresh failed.");
          }
        })
        .catch((err) => {
          setError(getErrorMessage(err));
        });
    }, 2000);
    return () => window.clearInterval(timer);
  }, [refreshJob]);

  const tableHeaders = useMemo(() => {
    const row = tableData?.rows[0];
    return row ? Object.keys(row.values) : [];
  }, [tableData]);

  const tickerOptions = useMemo(() => {
    const discovered = [...FALLBACK_TICKERS, ...(overview?.available_tickers ?? [])];
    if (selectedTicker && !discovered.includes(selectedTicker)) {
      discovered.unshift(selectedTicker);
    }
    return ["", ...Array.from(new Set(discovered))];
  }, [overview, selectedTicker]);

  const handleRefresh = async () => {
    try {
      setError(null);
      const payload = await startAnalysisRefresh({
        skip_trade_backfill: skipTradeBackfill,
        force_full_rebuild: forceFullRebuild,
      });
      setRefreshJob(payload);
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  const handleCreateNote = async () => {
    if (!noteTitle.trim() || !noteBody.trim()) return;
    try {
      const note = await createResearchNote({
        title: noteTitle,
        body: noteBody,
      });
      setNotes((current) => [note, ...current]);
      setNoteTitle("");
      setNoteBody("");
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  const handleExportTable = async () => {
    try {
      const csv = await exportAnalysisTableCsv({
        table: selectedTable,
        ticker: tableTickerFilter,
        variable_name: tableVariableFilter,
      });
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${selectedTable}-${selectedTicker || "all"}-${selectedVariable}.csv`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  return (
    <section className="page data-analysis-page">
      <header className="analysis-hero">
        <div>
          <span className="analysis-eyebrow">Data Analysis for Polymarket</span>
          <h1>Empirical monitoring for weekly stock ladders</h1>
          <p>
            Track daily volume, market structure, price behavior, threshold drift, and
            long-run stability from the local Postgres research store.
          </p>
        </div>
        <div className="analysis-controls">
          <div className="analysis-filter-group">
            <label htmlFor="analysisTicker">Ticker</label>
            <select
              id="analysisTicker"
              value={selectedTicker}
              onChange={(event) => {
                setSelectedTicker(event.target.value);
                setTablePage(1);
              }}
            >
              {tickerOptions.map((ticker) => (
                <option key={ticker || "all"} value={ticker}>
                  {ticker || "All tickers"}
                </option>
              ))}
            </select>
          </div>
          <div className="analysis-filter-group">
            <label htmlFor="analysisVariable">Variable</label>
            <select
              id="analysisVariable"
              value={selectedVariable}
              onChange={(event) => {
                setSelectedVariable(event.target.value);
                setTablePage(1);
              }}
            >
              <option value="notional_volume">Notional volume</option>
              <option value="contract_volume">Contract volume</option>
              <option value="trade_count">Trade count</option>
              <option value="avg_trade_size">Average trade size</option>
              <option value="rv_logit_intraday">Intraday RV (logit)</option>
              <option value="active_market_count">Active market count</option>
              <option value="volume_hhi_within_stock">Stock HHI</option>
            </select>
          </div>
          <label className="analysis-checkbox">
            <input
              type="checkbox"
              checked={skipTradeBackfill}
              onChange={(event) => setSkipTradeBackfill(event.target.checked)}
            />
            Skip trade backfill
          </label>
          <label className="analysis-checkbox">
            <input
              type="checkbox"
              checked={forceFullRebuild}
              onChange={(event) => setForceFullRebuild(event.target.checked)}
            />
            Force full rebuild
          </label>
          <button className="button primary" onClick={handleRefresh} disabled={["queued", "running"].includes(refreshJob?.status ?? "")}>
            {["queued", "running"].includes(refreshJob?.status ?? "") ? "Refreshing..." : "Refresh analysis store"}
          </button>
        </div>
      </header>

      {error ? <div className="analysis-banner error">{error}</div> : null}
      {refreshJob && ["queued", "running"].includes(refreshJob.status) ? (
        <div className="analysis-banner warning">
          Refresh {refreshJob.status}: {refreshJob.progress?.stage ?? "starting"}{" "}
          {refreshJob.progress ? `${refreshJob.progress.current}/${refreshJob.progress.total}` : ""}
        </div>
      ) : null}
      {refreshJob?.status === "failed" ? (
        <div className="analysis-banner error">
          Refresh failed: {refreshJob.error ?? "Unknown refresh error."}
        </div>
      ) : null}
      {overview && !overview.coverage.volume_authoritative ? (
        <div className="analysis-banner critical">
          Volume coverage is incomplete. Volume-based diagnostics are shown with non-authoritative flags until trade backfill coverage improves.
        </div>
      ) : null}

      <section className="analysis-grid metrics-grid">
        {overview?.headline_metrics.map((metric) => (
          <article key={metric.label} className="analysis-card metric-card">
            <span className="metric-label">{metric.label}</span>
            <strong className="metric-value">{formatNumber(metric.value)}</strong>
            <span className="metric-subvalue">
              {metric.delta !== null && metric.delta !== undefined ? `Δ ${formatPct(metric.delta)}` : "No delta"}
            </span>
          </article>
        ))}
      </section>

      <section className="analysis-grid analysis-section-grid">
        <article className="analysis-card">
          <div className="section-header">
            <div>
              <h2>Overview</h2>
              <p>Latest refresh state and daily global trends.</p>
            </div>
            <span className="coverage-pill">
              {overview?.coverage.latest_refresh_id ?? "No refresh"}
            </span>
          </div>
          {sectionErrors.overview ? <div className="analysis-banner error">{sectionErrors.overview}</div> : null}
          <Plot
            data={[
              toLineTrace("Notional volume", overview?.recent_trends ?? [], "value", "#0e7490"),
              toLineTrace("Trade count", overview?.recent_trends ?? [], "value_2", "#f97316"),
            ]}
            layout={{
              autosize: true,
              height: 320,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 48, r: 20, t: 20, b: 48 },
              legend: { orientation: "h" },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
          <div className="alert-list">
            {overview?.alerts.map((alert) => (
              <div key={`${alert.title}-${alert.detail}`} className={`alert-item ${alert.level}`}>
                <strong>{alert.title}</strong>
                <span>{alert.detail}</span>
              </div>
            ))}
          </div>
        </article>

        <article className="analysis-card">
          <div className="section-header">
            <div>
              <h2>Volume Analysis</h2>
              <p>Raw and cleaned volume behavior with threshold diagnostics.</p>
            </div>
            <span className="coverage-pill">
              {VARIABLE_LABELS[selectedVariable] ?? selectedVariable} · clipped share {formatPct(volume?.clipped_share ?? null)}
            </span>
          </div>
          {sectionErrors.volume ? <div className="analysis-banner error">{sectionErrors.volume}</div> : null}
          <Plot
            data={[
              toLineTrace(
                selectedTicker ? `${VARIABLE_LABELS[selectedVariable] ?? selectedVariable} (selected ticker)` : `${VARIABLE_LABELS[selectedVariable] ?? selectedVariable} (global)`,
                volume?.total_volume_series ?? [],
                "value",
                "#2563eb",
              ),
              ...((selectedTicker && (volume?.per_stock_series?.length ?? 0) > 0)
                ? [toLineTrace(`${selectedTicker} stock-day series`, volume?.per_stock_series ?? [], "value", "#dc2626")]
                : []),
            ]}
            layout={{
              autosize: true,
              height: 280,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 48, r: 20, t: 20, b: 48 },
              legend: { orientation: "h" },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
          <div className="section-header">
            <div>
              <p>
                Threshold scope: {volume?.selected_scope_level ?? "--"} · authoritative ratio{" "}
                {formatPct(volume?.coverage.trade_coverage_ratio ?? null)}
              </p>
            </div>
          </div>
          <div className="dual-plot-grid">
            <Plot
              data={[
                {
                  type: "bar",
                  x: (volume?.hist_raw ?? []).map((bin) => `${bin.start.toFixed(1)}-${bin.end.toFixed(1)}`),
                  y: (volume?.hist_raw ?? []).map((bin) => bin.count),
                  marker: { color: "#16a34a" },
                },
              ]}
              layout={{
                autosize: true,
                height: 240,
                title: { text: "Raw histogram", font: { size: 14 } },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 48, r: 12, t: 42, b: 48 },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
            <Plot
              data={[
                {
                  type: "bar",
                  x: (volume?.hist_log ?? []).map((bin) => `${bin.start.toFixed(2)}-${bin.end.toFixed(2)}`),
                  y: (volume?.hist_log ?? []).map((bin) => bin.count),
                  marker: { color: "#9333ea" },
                },
              ]}
              layout={{
                autosize: true,
                height: 240,
                title: { text: "Log histogram", font: { size: 14 } },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 48, r: 12, t: 42, b: 48 },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
          </div>
          <div className="table-shell compact">
            <table>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Ticker</th>
                  <th>Scope</th>
                  <th>Upper cap</th>
                  <th>Q95</th>
                  <th>Q99</th>
                  <th>Sample</th>
                </tr>
              </thead>
              <tbody>
                {volume?.threshold_rows.slice(0, 10).map((row) => (
                  <tr key={`${row.trade_date_ny}-${row.ticker}-${row.scope_used}`}>
                    <td>{row.trade_date_ny}</td>
                    <td>{row.ticker ?? "ALL"}</td>
                    <td>{row.scope_used}</td>
                    <td>{formatNumber(row.upper_cap)}</td>
                    <td>{formatNumber(row.q95)}</td>
                    <td>{formatNumber(row.q99)}</td>
                    <td>{row.sample_count ?? "--"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        <article className="analysis-card">
          <div className="section-header">
            <div>
              <h2>Market Structure</h2>
              <p>Concentration, breadth, and lifecycle evolution.</p>
            </div>
          </div>
          {sectionErrors.structure ? <div className="analysis-banner error">{sectionErrors.structure}</div> : null}
          <Plot
            data={[
              toLineTrace("Stock HHI", structure?.stock_concentration ?? [], "value", "#7c3aed"),
              toLineTrace("Top stock share", structure?.stock_concentration ?? [], "value_2", "#ea580c"),
            ]}
            layout={{
              autosize: true,
              height: 280,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 48, r: 20, t: 20, b: 48 },
              legend: { orientation: "h" },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
          <div className="dual-plot-grid">
            <Plot
              data={[toLineTrace("Active markets", structure?.active_market_counts ?? [], "value", "#0284c7")]}
              layout={{
                autosize: true,
                height: 220,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 48, r: 12, t: 20, b: 48 },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
            <Plot
              data={[toLineTrace("Active stocks", structure?.stock_participation ?? [], "value", "#65a30d")]}
              layout={{
                autosize: true,
                height: 220,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 48, r: 12, t: 20, b: 48 },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
          </div>
          <div className="table-shell compact">
            <table>
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Avg lifecycle</th>
                  <th>Avg notional</th>
                </tr>
              </thead>
              <tbody>
                {(structure?.lifecycle_summary ?? []).slice(0, 8).map((row, index) => (
                  <tr key={`lifecycle-${index}`}>
                    <td>{String(row.ticker ?? "--")}</td>
                    <td>{formatNumber((row.avg_lifecycle_progress as number | null | undefined) ?? null)}</td>
                    <td>{formatNumber((row.avg_notional_volume as number | null | undefined) ?? null)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        <article className="analysis-card">
          <div className="section-header">
            <div>
              <h2>Price / Probability Behavior</h2>
              <p>Probability distribution, volatility, and expiry proximity effects.</p>
            </div>
          </div>
          {sectionErrors.priceBehavior ? <div className="analysis-banner error">{sectionErrors.priceBehavior}</div> : null}
          <Plot
            data={[
              {
                type: "bar",
                x: (priceBehavior?.close_probability_distribution ?? []).map((bin) => `${bin.start.toFixed(2)}-${bin.end.toFixed(2)}`),
                y: (priceBehavior?.close_probability_distribution ?? []).map((bin) => bin.count),
                marker: { color: "#2563eb" },
                name: "Close probability",
              },
            ]}
            layout={{
              autosize: true,
              height: 240,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 48, r: 20, t: 20, b: 48 },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
          <Plot
            data={[
              toLineTrace("Avg intraday RV", priceBehavior?.volatility_series ?? [], "value", "#ef4444"),
              toLineTrace("Avg boundary distance", priceBehavior?.volatility_series ?? [], "value_2", "#059669"),
            ]}
            layout={{
              autosize: true,
              height: 260,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 48, r: 20, t: 20, b: 48 },
              legend: { orientation: "h" },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
          <div className="dual-plot-grid">
            <Plot
              data={[
                {
                  type: "bar",
                  x: (priceBehavior?.expiry_behavior ?? []).map((point) => point.label),
                  y: (priceBehavior?.expiry_behavior ?? []).map((point) => point.value),
                  marker: { color: "#1d4ed8" },
                  name: "Avg notional by DTR bucket",
                },
              ]}
              layout={{
                autosize: true,
                height: 220,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 48, r: 12, t: 20, b: 48 },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
            <div className="table-shell compact">
              <table>
                <thead>
                  <tr>
                    <th>Ticker</th>
                    <th>Avg boundary distance</th>
                    <th>Avg close prob</th>
                  </tr>
                </thead>
                <tbody>
                  {(priceBehavior?.convergence_table ?? []).slice(0, 8).map((row, index) => (
                    <tr key={`conv-${index}`}>
                      <td>{String(row.ticker ?? "--")}</td>
                      <td>{formatNumber((row.avg_distance_to_boundary as number | null | undefined) ?? null)}</td>
                      <td>{formatNumber((row.avg_close_prob as number | null | undefined) ?? null)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </article>

        <article className="analysis-card">
          <div className="section-header">
            <div>
              <h2>Regime and Drift Monitoring</h2>
              <p>Rolling statistics, distribution drift, and detected breakpoints.</p>
            </div>
          </div>
          {sectionErrors.drift ? <div className="analysis-banner error">{sectionErrors.drift}</div> : null}
          <Plot
            data={[
              toLineTrace(`${VARIABLE_LABELS[selectedVariable] ?? selectedVariable}`, drift?.rolling_statistics ?? [], "value", "#0f766e"),
              ...((drift?.rolling_statistics ?? []).some((point) => point.value_2 !== null && point.value_2 !== undefined)
                ? [toLineTrace("Companion series", drift?.rolling_statistics ?? [], "value_2", "#c2410c")]
                : []),
            ]}
            layout={{
              autosize: true,
              height: 260,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 48, r: 20, t: 20, b: 48 },
              legend: { orientation: "h" },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
          <Plot
            data={[
              toLineTrace("P10", drift?.rolling_quantiles ?? [], "value", "#6366f1"),
              toLineTrace("P50", drift?.rolling_quantiles ?? [], "value_2", "#0ea5e9"),
              toLineTrace("P90", drift?.rolling_quantiles ?? [], "value_3", "#f97316"),
            ]}
            layout={{
              autosize: true,
              height: 240,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 48, r: 20, t: 20, b: 48 },
              legend: { orientation: "h" },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
          <div className="alert-list">
            {drift?.alerts.map((alert) => (
              <div key={`${alert.title}-${alert.detail}`} className={`alert-item ${alert.level}`}>
                <strong>{alert.title}</strong>
                <span>{alert.detail}</span>
              </div>
            ))}
          </div>
        </article>
      </section>

      <section className="analysis-grid lower-grid">
        <article className="analysis-card">
          <div className="section-header">
            <div>
              <h2>Research Tables</h2>
              <p>Inspectable database-backed outputs with pagination.</p>
            </div>
            <div className="inline-controls">
              <select
                value={selectedTable}
                onChange={(event) => {
                  setSelectedTable(event.target.value);
                  setTablePage(1);
                }}
              >
                <option value="threshold_rule">Threshold rules</option>
                <option value="outlier_flag">Outlier flags</option>
                <option value="variable_profile">Variable profiles</option>
                <option value="drift_monitor">Drift monitor</option>
                <option value="break_event">Break events</option>
                <option value="fact_stock_day">Stock daily</option>
                <option value="fact_market_day">Market daily</option>
              </select>
              <button className="button light" onClick={handleExportTable}>
                Export CSV
              </button>
            </div>
          </div>
          {sectionErrors.tables ? <div className="analysis-banner error">{sectionErrors.tables}</div> : null}
          <div className="table-shell">
            <table>
              <thead>
                <tr>
                  {tableHeaders.map((header) => (
                    <th key={header}>{header}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tableData?.rows.map((row, index) => (
                  <tr key={`${selectedTable}-${index}`}>
                    {tableHeaders.map((header) => (
                      <td key={header}>{String(row.values[header] ?? "--")}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="pagination-row">
            <button className="button light" onClick={() => setTablePage((page) => Math.max(1, page - 1))} disabled={tablePage <= 1}>
              Previous
            </button>
            <span>
              Page {tablePage} / {tableData ? Math.max(1, Math.ceil(tableData.total_rows / tableData.page_size)) : 1}
            </span>
            <button
              className="button light"
              onClick={() => setTablePage((page) => page + 1)}
              disabled={!tableData || tablePage >= Math.ceil(tableData.total_rows / tableData.page_size)}
            >
              Next
            </button>
          </div>
        </article>

        <article className="analysis-card">
          <div className="section-header">
            <div>
              <h2>Notes / Research Log</h2>
              <p>Persist observations in the local research store.</p>
            </div>
          </div>
          {sectionErrors.notes ? <div className="analysis-banner error">{sectionErrors.notes}</div> : null}
          <div className="note-form">
            <input value={noteTitle} onChange={(event) => setNoteTitle(event.target.value)} placeholder="Note title" />
            <textarea value={noteBody} onChange={(event) => setNoteBody(event.target.value)} placeholder="Interpretation, caveat, or research next step" rows={5} />
            <button className="button primary" onClick={handleCreateNote}>
              Add note
            </button>
          </div>
          <div className="note-list">
            {notes.map((note) => (
              <article key={note.note_id} className="note-card">
                <div className="note-header">
                  <strong>{note.title}</strong>
                  <span>{note.author || "unknown"} · {new Date(note.updated_at_utc).toLocaleString()}</span>
                </div>
                <p>{note.body}</p>
              </article>
            ))}
            {!notes.length && !isLoading ? <div className="empty-state">No research notes yet.</div> : null}
          </div>
        </article>
      </section>
    </section>
  );
}
