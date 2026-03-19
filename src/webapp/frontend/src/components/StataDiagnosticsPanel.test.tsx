import "@testing-library/jest-dom/vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { StataDiagnosticsPayload } from "../api/calibrateModels";
import { StataDiagnosticsPanel } from "./StataDiagnosticsPanel";

const diagnosticsFixture: StataDiagnosticsPayload = {
  schema_version: 1,
  header: {
    model_id: "logit-run-001",
    estimator: "sklearn_logit",
    inference_basis: "shadow_statsmodels_glm",
    baseline_name: "pRN",
    best_c: 1.5,
    penalty: "l2",
    solver: "lbfgs",
    threshold_default: 0.5,
    threshold_operating: null,
    split_counts: { train_fit: 120, val: 48, test: 48 },
    event_counts: { train_fit: 58, val: 24, test: 25 },
    feature_counts: { numeric: 4, categorical: 1, transformed: 8 },
    notes: [
      "Inferential rows come from a shadow statsmodels Binomial GLM on the transformed train_fit design matrix.",
      "Shadow statsmodels GLM dropped a singular column for this run.",
    ],
  },
  sections: [
    {
      id: "global_significance",
      title: "Global model significance",
      rows: [
        {
          key: "lr_chi2",
          label: "LR chi2(4)",
          train_fit: 12.42,
          val: null,
          test: null,
          status: "ok",
          format: "metric",
          source: "shadow_statsmodels_glm",
        },
        {
          key: "lr_prob",
          label: "Prob > chi2",
          train_fit: 0.0013,
          val: null,
          test: null,
          status: "ok",
          format: "pvalue",
        },
      ],
    },
    {
      id: "calibration",
      title: "Calibration",
      rows: [
        {
          key: "calibration_slope",
          label: "Calibration slope",
          train_fit: 1.04,
          val: 0.97,
          test: 0.93,
          status: "ok",
          format: "metric",
        },
      ],
    },
  ],
  production_coefficient_table: {
    basis: "production_sklearn_final_model",
    fit_scope: "train",
    subtitle: "Production final sklearn model coefficients. Matches the displayed equation exactly.",
    rows: [
      {
        feature_name: "const",
        display_name: "_cons",
        feature_group: "core",
        coefficient: 0.31,
        odds_ratio: null,
        note: "Intercept from the final production sklearn model.",
      },
      {
        feature_name: "feature_one",
        display_name: "Feature one",
        feature_group: "core",
        coefficient: 0.42,
        odds_ratio: 1.52,
        note: "Matches the displayed production equation.",
      },
      {
        feature_name: "ticker_AAPL",
        display_name: "Ticker AAPL",
        feature_group: "ticker",
        coefficient: -0.08,
        odds_ratio: 0.92,
        note: "Matches the displayed production equation.",
      },
    ],
  },
  coefficient_table: {
    basis: "shadow_statsmodels_glm",
    fit_scope: "train_fit",
    subtitle:
      "Shadow statsmodels inference on the transformed train_fit design matrix. Inferential only; coefficients are not expected to numerically match the production equation.",
    rows: [
      {
        feature_name: "feature_one",
        display_name: "Feature one",
        feature_group: "core",
        coefficient: 0.42,
        std_error: 0.11,
        z_stat: 3.82,
        p_value: 0.0001,
        ci_low: 0.20,
        ci_high: 0.64,
        odds_ratio: 1.52,
        odds_ratio_ci_low: 1.22,
        odds_ratio_ci_high: 1.90,
        note: null,
      },
      {
        feature_name: "ticker_AAPL",
        display_name: "Ticker AAPL",
        feature_group: "ticker",
        coefficient: -0.08,
        std_error: 0.20,
        z_stat: -0.40,
        p_value: 0.689,
        ci_low: -0.48,
        ci_high: 0.32,
        odds_ratio: 0.92,
        odds_ratio_ci_low: 0.62,
        odds_ratio_ci_high: 1.38,
        note: "Reference-adjusted ticker term.",
      },
    ],
  },
  marginal_effects_table: {
    basis: "shadow_statsmodels_glm",
    fit_scope: "train_fit",
    subtitle: "Shadow statsmodels marginal effects on the transformed train_fit design matrix.",
    rows: [
      {
        feature_name: "feature_one",
        display_name: "Feature one",
        feature_group: "core",
        ame: 0.08,
        std_error: 0.02,
        z_stat: 3.20,
        p_value: 0.0014,
        ci_low: 0.03,
        ci_high: 0.13,
        note: null,
      },
      {
        feature_name: "ticker_AAPL",
        display_name: "Ticker AAPL",
        feature_group: "ticker",
        ame: -0.01,
        std_error: 0.02,
        z_stat: -0.50,
        p_value: 0.6201,
        ci_low: -0.05,
        ci_high: 0.03,
        note: "Reference-adjusted ticker term.",
      },
    ],
  },
  calibration_curve: {
    split: "test",
    binning: "equal_mass",
    rows: [
      { bin: 1, n: 10, series: "model", pred_mean: 0.14, obs_rate: 0.10, abs_gap: 0.04 },
      { bin: 2, n: 10, series: "model", pred_mean: 0.36, obs_rate: 0.40, abs_gap: 0.04 },
      { bin: 3, n: 10, series: "model", pred_mean: 0.61, obs_rate: 0.58, abs_gap: 0.03 },
      { bin: 1, n: 10, series: "baseline", pred_mean: 0.18, obs_rate: 0.10, abs_gap: 0.08 },
      { bin: 2, n: 10, series: "baseline", pred_mean: 0.42, obs_rate: 0.40, abs_gap: 0.02 },
      { bin: 3, n: 10, series: "baseline", pred_mean: 0.67, obs_rate: 0.58, abs_gap: 0.09 },
    ],
  },
  warnings: ["Shadow GLM uses an unpenalized diagnostic fit."],
};

describe("StataDiagnosticsPanel", () => {
  it("renders the analytical summary and coefficient filters", () => {
    const { container } = render(<StataDiagnosticsPanel diagnostics={diagnosticsFixture} />);

    expect(screen.getByText("logit-run-001")).toBeInTheDocument();
    expect(screen.getByText("Global model significance")).toBeInTheDocument();
    expect(screen.getByText("Calibration slope")).toBeInTheDocument();
    expect(screen.getByText("Calibration curve")).toBeInTheDocument();
    expect(screen.getByText("Production coefficient detail")).toBeInTheDocument();
    expect(screen.getByText("Shadow inference detail")).toBeInTheDocument();
    expect(screen.getByText("Shadow marginal effects")).toBeInTheDocument();
    expect(screen.getAllByText("pRN").length).toBeGreaterThan(0);
    expect(
      screen.getByText("Production final sklearn model coefficients. Matches the displayed equation exactly."),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(
        "Inferential rows come from a shadow statsmodels Binomial GLM on the transformed train_fit design matrix.",
      ),
    ).not.toBeInTheDocument();
    expect(screen.getByText("Shadow statsmodels GLM dropped a singular column for this run.")).toBeInTheDocument();
    expect(screen.getAllByText("Feature one").length).toBeGreaterThan(0);
    expect(screen.queryByText("Ticker AAPL")).not.toBeInTheDocument();

    expect(screen.getByRole("columnheader", { name: "Train-fit" })).toHaveClass("stata-cell-right");
    expect(screen.getByRole("columnheader", { name: "Statistic" })).toHaveClass("stata-cell-left");
    expect(screen.getAllByRole("columnheader", { name: "Coef." })[0]).toHaveClass("stata-cell-right");
    expect(screen.getByText("12.4200").closest("td")).toHaveClass("stata-cell-right");
    expect(container.querySelector(".stata-table-wrap-static")).toBeInTheDocument();
    expect(container.querySelector(".stata-calibration-chart")).toHaveAttribute("viewBox", "0 0 560 560");
    expect(container.querySelector(".chart-line-model")).toBeInTheDocument();
    expect(container.querySelector(".chart-line-baseline")).toBeInTheDocument();

    fireEvent.click(screen.getAllByRole("button", { name: "ticker" })[0]);

    expect(screen.getAllByText("Ticker AAPL").length).toBeGreaterThan(0);
    expect(screen.getByText("Shadow GLM uses an unpenalized diagnostic fit.")).toBeInTheDocument();
  });

  it("supports an inference-only rendering mode for the selectable diagnostics artifact", () => {
    const { container } = render(
      <StataDiagnosticsPanel
        diagnostics={diagnosticsFixture}
        showHeaderStrip={false}
        showCalibrationChart={false}
        showProductionCoefficientTable={false}
      />,
    );

    expect(container.querySelector(".stata-header-strip")).not.toBeInTheDocument();
    expect(container.querySelector(".stata-production-table")).not.toBeInTheDocument();
    expect(container.querySelector(".stata-calibration-chart")).not.toBeInTheDocument();
    expect(within(container).getByText("Shadow inference detail")).toBeInTheDocument();
    expect(within(container).getByText("Shadow marginal effects")).toBeInTheDocument();
  });
});
