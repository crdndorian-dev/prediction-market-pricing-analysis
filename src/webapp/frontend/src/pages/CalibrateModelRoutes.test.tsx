import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  cancelCalibrationJob: vi.fn(),
  deleteCalibrationModel: vi.fn(),
  fetchCalibrationDatasets: vi.fn(),
  fetchCalibrationModelDetail: vi.fn(),
  fetchCalibrationModels: vi.fn(),
  fetchDatasetFeatures: vi.fn(),
  fetchDatasetTickers: vi.fn(),
  fetchModelFileContent: vi.fn(),
  fetchModelFileContentByPath: vi.fn(),
  fetchModelFiles: vi.fn(),
  getCalibrationJob: vi.fn(),
  previewCalibrationRegime: vi.fn(),
  previewCalibrationWeighting: vi.fn(),
  renameCalibrationModel: vi.fn(),
  startAutoCalibrationJob: vi.fn(),
  startCalibrationJob: vi.fn(),
}));

vi.mock("../api/calibrateModels", () => apiMocks);

vi.mock("../components/StataDiagnosticsPanel", () => ({
  CalibrationCurveCard: () => <div>Calibration curve</div>,
  StataDiagnosticsPanel: ({ diagnostics }: { diagnostics: { header: { model_id: string } } }) => (
    <div>Diagnostics {diagnostics.header.model_id}</div>
  ),
  isStataDiagnosticsPayload: () => true,
}));

vi.mock("../contexts/calibrationJob", () => ({
  useCalibrationJob: () => ({
    jobId: null,
    jobStatus: null,
    setJobId: vi.fn(),
    setJobStatus: vi.fn(),
  }),
}));

vi.mock("../contexts/jobGuard", () => ({
  useAnyJobRunning: () => ({
    anyJobRunning: false,
    primaryJob: null,
    activeJobs: [],
  }),
}));

import CalibrateModelsPage, { CalibrationModelDetailPage } from "./CalibrateModelsPage";

describe("calibrate model routes", () => {
  beforeEach(() => {
    vi.clearAllMocks();

    apiMocks.fetchCalibrationDatasets.mockRejectedValue(new Error("datasets unavailable"));
    apiMocks.fetchCalibrationModels.mockResolvedValue({
      base_dir: "models",
      models: [
        {
          id: "demo-model",
          path: "models/demo-model",
          last_modified: "2026-03-18T12:00:00Z",
          has_metadata: true,
          has_metrics: true,
          dataset_id: "weekly-options",
          split_strategy: "walk_forward",
          c_value: 0.3,
          calibration_method: "platt",
          run_type: "manual",
          is_two_stage: false,
        },
      ],
    });
    apiMocks.fetchCalibrationModelDetail.mockResolvedValue({
      id: "demo-model",
      path: "models/demo-model",
      last_modified: "2026-03-18T12:00:00Z",
      has_metadata: true,
      has_metrics: true,
      files: [],
      metrics_summary: {
        val: {
          split: "val",
          status: "good",
          baseline_logloss: 0.5123,
          model_logloss: 0.4876,
          delta_model_minus_baseline: -0.0247,
          baseline_brier: 0.2241,
          model_brier: 0.2198,
          delta_brier: -0.0043,
          baseline_ece_q: 0.019,
          model_ece_q: 0.014,
          delta_ece_q: -0.005,
        },
      },
      stata_diagnostics: null,
      split_row_counts: {
        train_fit: 1200,
        val: 320,
        test: 310,
      },
      split_group_counts: {
        train_fit: 410,
        val: 102,
        test: 100,
      },
      model_equation: "p = \\sigma(\\eta)",
      model_equation_spec: { compact_latex: "p = \\sigma(\\eta)" },
    });
    apiMocks.fetchModelFiles.mockResolvedValue({
      model_id: "demo-model",
      files: [],
    });
    apiMocks.fetchDatasetFeatures.mockResolvedValue({
      available_columns: [],
      selectable_features: [],
    });
    apiMocks.fetchDatasetTickers.mockResolvedValue({ tickers: [] });
    apiMocks.previewCalibrationRegime.mockResolvedValue({});
    apiMocks.previewCalibrationWeighting.mockResolvedValue({});
  });

  it("opens the models tab from the query string and navigates to a dedicated model detail route", async () => {
    render(
      <MemoryRouter initialEntries={["/calibrate?tab=models"]}>
        <Routes>
          <Route path="/calibrate" element={<CalibrateModelsPage />} />
          <Route path="/calibrate/models/:modelId" element={<CalibrationModelDetailPage />} />
        </Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByRole("heading", { name: "Models" })).toBeInTheDocument();
    expect(screen.queryByText(/add to compare/i)).not.toBeInTheDocument();

    fireEvent.click(await screen.findByRole("button", { name: /demo-model/i }));

    expect(await screen.findByRole("heading", { name: "demo-model" })).toBeInTheDocument();
    expect(screen.getByText("Back to Model Directory")).toBeInTheDocument();
    expect(screen.queryByText("Diagnostics demo-model")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("link", { name: "Back to Model Directory" }));

    expect(await screen.findByRole("heading", { name: "Models" })).toBeInTheDocument();
    expect(screen.getByText("demo-model")).toBeInTheDocument();
  });
});
