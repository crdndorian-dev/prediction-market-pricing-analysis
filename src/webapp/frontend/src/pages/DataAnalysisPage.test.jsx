import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import DataAnalysisPage from "./DataAnalysisPage";

vi.mock("react-plotly.js", () => ({
  default: () => <div data-testid="plotly-chart" />,
}));

vi.mock("../api/analysis", () => ({
  createResearchNote: vi.fn(),
  exportAnalysisTableCsv: vi.fn(),
  fetchAnalysisDrift: vi.fn(),
  fetchAnalysisOverview: vi.fn(),
  fetchAnalysisPriceBehavior: vi.fn(),
  fetchAnalysisStructure: vi.fn(),
  fetchAnalysisTable: vi.fn(),
  fetchAnalysisVolume: vi.fn(),
  fetchResearchNotes: vi.fn(),
  getAnalysisJob: vi.fn(),
  startAnalysisRefresh: vi.fn(),
}));

import {
  fetchAnalysisDrift,
  fetchAnalysisOverview,
  fetchAnalysisPriceBehavior,
  fetchAnalysisStructure,
  fetchAnalysisTable,
  fetchAnalysisVolume,
  fetchResearchNotes,
  getAnalysisJob,
  startAnalysisRefresh,
} from "../api/analysis";

function buildOverview() {
  return {
    coverage: {
      latest_refresh_id: "refresh-1",
      latest_refresh_at: "2026-03-08T12:00:00Z",
      volume_authoritative: true,
      market_day_count: 10,
      stock_day_count: 10,
      global_day_count: 10,
      trade_coverage_ratio: 1,
      stock_count: 2,
      active_market_count: 4,
    },
    headline_metrics: [{ label: "Latest NY date", value: "2026-03-07", delta: null }],
    recent_trends: [{ date: "2026-03-07", value: 10, value_2: 3 }],
    alerts: [],
    available_tickers: ["AAPL", "NVDA"],
  };
}

function buildVolume() {
  return {
    coverage: buildOverview().coverage,
    selected_variable: "notional_volume",
    selected_scope_level: "global_day",
    total_volume_series: [{ date: "2026-03-07", value: 10, value_2: 3 }],
    per_stock_series: [{ date: "2026-03-07", value: 5, label: "AAPL" }],
    hist_raw: [{ start: 0, end: 10, count: 1 }],
    hist_log: [{ start: 0, end: 2, count: 1 }],
    outlier_diagnostics: [],
    threshold_rows: [],
    clipped_share: 0,
  };
}

function buildStructure() {
  return {
    coverage: buildOverview().coverage,
    stock_concentration: [{ date: "2026-03-07", value: 0.4, value_2: 0.6 }],
    active_market_counts: [{ date: "2026-03-07", value: 4, value_2: 3 }],
    stock_participation: [{ date: "2026-03-07", value: 2 }],
    lifecycle_summary: [{ ticker: "AAPL", avg_lifecycle_progress: 0.5, avg_notional_volume: 100 }],
  };
}

function buildPriceBehavior() {
  return {
    coverage: buildOverview().coverage,
    close_probability_distribution: [{ start: 0.1, end: 0.2, count: 2 }],
    volatility_series: [{ date: "2026-03-07", value: 0.2, value_2: 0.3 }],
    expiry_behavior: [{ date: "2026-03-07", label: "dtr_2_3", value: 5, value_2: 0.2 }],
    convergence_table: [{ ticker: "AAPL", avg_distance_to_boundary: 0.2, avg_close_prob: 0.6 }],
  };
}

function buildDrift() {
  return {
    coverage: buildOverview().coverage,
    selected_variable: "notional_volume",
    rolling_statistics: [{ date: "2026-03-07", value: 10, value_2: 3 }],
    rolling_quantiles: [{ date: "2026-03-07", value: 1, value_2: 2, value_3: 3 }],
    drift_rows: [],
    break_rows: [],
    alerts: [],
  };
}

function buildTable() {
  return {
    table: "threshold_rule",
    page: 1,
    page_size: 25,
    total_rows: 1,
    rows: [{ values: { trade_date_ny: "2026-03-07", variable_name: "notional_volume" } }],
  };
}

function mockSuccessfulLoads() {
  fetchAnalysisOverview.mockResolvedValue(buildOverview());
  fetchAnalysisVolume.mockResolvedValue(buildVolume());
  fetchAnalysisStructure.mockResolvedValue(buildStructure());
  fetchAnalysisPriceBehavior.mockResolvedValue(buildPriceBehavior());
  fetchAnalysisDrift.mockResolvedValue(buildDrift());
  fetchAnalysisTable.mockResolvedValue(buildTable());
  fetchResearchNotes.mockResolvedValue({ notes: [] });
}

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/data-analysis"]}>
      <Routes>
        <Route path="/data-analysis" element={<DataAnalysisPage />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("DataAnalysisPage", () => {
  beforeEach(() => {
    vi.useRealTimers();
    mockSuccessfulLoads();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("renders on the data-analysis route and keeps partial results visible when one section fails", async () => {
    fetchAnalysisVolume.mockRejectedValueOnce(new Error("Volume endpoint failed"));

    renderPage();

    expect(await screen.findByText("Data Analysis for Polymarket")).toBeInTheDocument();
    expect(await screen.findByText("Overview")).toBeInTheDocument();
    expect(await screen.findByText("Market Structure")).toBeInTheDocument();
    expect(await screen.findByText("Volume endpoint failed")).toBeInTheDocument();
  });

  it("keeps fallback ticker choices available when overview metadata fails", async () => {
    fetchAnalysisOverview.mockRejectedValueOnce(new Error("Overview endpoint failed"));

    renderPage();

    expect(await screen.findByText("Overview endpoint failed")).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "AAPL" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "PLTR" })).toBeInTheDocument();
  });

  it("polls queued refresh jobs until finished and reloads data", async () => {
    startAnalysisRefresh.mockResolvedValue({
      job_id: "job-1",
      status: "queued",
      progress: null,
      result: null,
      error: null,
      started_at: null,
      finished_at: null,
    });
    getAnalysisJob
      .mockResolvedValueOnce({
        job_id: "job-1",
        status: "running",
        progress: { stage: "write_marts", current: 1, total: 2, detail: null },
        result: null,
        error: null,
        started_at: "2026-03-08T12:00:00Z",
        finished_at: null,
      })
      .mockResolvedValueOnce({
        job_id: "job-1",
        status: "finished",
        progress: { stage: "done", current: 2, total: 2, detail: null },
        result: { ok: true, refresh_id: "refresh-2", stdout: "", stderr: "", duration_s: 1, command: [] },
        error: null,
        started_at: "2026-03-08T12:00:00Z",
        finished_at: "2026-03-08T12:00:02Z",
      });

    renderPage();
    await screen.findByText("Refresh analysis store");

    fireEvent.click(screen.getByText("Refresh analysis store"));

    expect(await screen.findByText(/Refresh queued:/)).toBeInTheDocument();

    await waitFor(() => expect(getAnalysisJob).toHaveBeenCalledTimes(2), { timeout: 7000 });
    await waitFor(() => expect(fetchAnalysisOverview).toHaveBeenCalledTimes(2), { timeout: 7000 });
  }, 10000);
});
