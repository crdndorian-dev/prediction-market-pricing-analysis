import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from "react";
import katex from "katex";

import {
  cancelCalibrationJob,
  deleteCalibrationModel,
  fetchCalibrationDatasets,
  fetchCalibrationModelDetail,
  fetchCalibrationModels,
  fetchDatasetFeatures,
  fetchDatasetTickers,
  getCalibrationJob,
  fetchModelFileContent,
  fetchModelFileContentByPath,
  fetchModelFiles,
  previewCalibrationRegime,
  previewCalibrationWeighting,
  renameCalibrationModel,
  startAutoCalibrationJob,
  startCalibrationJob,
  type AutoModelRunRequest,
  type CalibrationJobStatus,
  type CalibrateModelRunResponse,
  type DatasetFileSummary,
  type ModelDetailResponse,
  type ModelFileContentResponse,
  type ModelFileSummary,
  type ModelFilesListResponse,
  type ModelRunSummary,
  type RegimePreviewResponse,
  type WeightingPreviewResponse,
} from "../api/calibrateModels";
import PipelineStatusCard from "../components/PipelineStatusCard";
import PipelineProgressBar from "../components/PipelineProgressBar";
import { useCalibrationJob } from "../contexts/calibrationJob";
import { useAnyJobRunning } from "../contexts/jobGuard";
import { CalibrateDocContent } from "./DocumentationPage";
import "katex/dist/katex.min.css";
import "./CalibrateModelsPage.css";

type WorkspaceTab = "run_job" | "models" | "documentation";
type RunJobPanel = "configuration" | "active_run";
type RunMode = "manual" | "auto";
type SplitStrategy = "walk_forward" | "single_holdout";
type WindowMode = "rolling" | "expanding";
type CalibrationMethod = "none" | "platt";
type SelectionObjective = "logloss" | "brier" | "ece_q";
type AutoOuterSelectionMetric =
  | "median_delta_logloss"
  | "worst_delta_logloss"
  | "mean_delta_logloss";
type WeightColStrategy = "auto" | "weight_final" | "sample_weight_final" | "uniform";
type BaseWeightSource = "dataset_weight" | "uniform";
type TickerBalanceMode = "none" | "sqrt_inv_clipped";
type TickerInterceptMode = "none" | "all" | "non_foundation";
type BootstrapGroupMode = "contract_id" | "group_id" | "ticker_day" | "day" | "iid" | "auto";
type TimeRegimeKey = "mon_4" | "tue_3" | "wed_2" | "thu_1";

type CalibrateFormState = {
  runMode: RunMode;
  modelDirName: string;
  datasetPath: string;
  randomSeed: string;
  weightColStrategy: WeightColStrategy;
  timeRegime: TimeRegimeKey;
  selectedFeatures: string[];
  selectedCategoricalFeatures: string[];

  splitStrategy: SplitStrategy;
  windowMode: WindowMode;
  trainWindowWeeks: string;
  validationFolds: string;
  validationWindowWeeks: string;
  testWindowWeeks: string;
  embargoDays: string;

  cGridPreset: "coarse" | "standard" | "wide" | "custom";
  cGridCustom: string;
  calibrationMethod: CalibrationMethod;
  selectionObjective: SelectionObjective;

  tradingUniverseTickers: string[];
  trainTickers: string[];
  foundationTickers: string[];
  foundationWeight: string;
  tickerInterceptMode: TickerInterceptMode;
  perTickerInteractions: boolean;
  minSupportIntercepts: string;
  minSupportInteractions: string;

  baseWeightSource: BaseWeightSource;
  groupingKey: string;
  groupEqualization: boolean;
  renorm: "mean1";
  tradingUniverseUpweight: string;
  tickerBalanceMode: TickerBalanceMode;

  bootstrapEnabled: boolean;
  bootstrapGroup: BootstrapGroupMode;
  bootstrapDraws: string;
  bootstrapSeed: string;
  ciLevel: 90 | 95 | 99;
  perSplitReporting: boolean;
  perFoldReporting: boolean;
  splitTimeline: boolean;
  perFoldDeltaChart: boolean;
  perGroupDeltaDistribution: boolean;

  maxAbsLogm: string;
  dropPrnExtremes: boolean;
  dropPrnBelow: string;
  dropPrnAbove: string;

  autoMaxTrials: string;
  autoAllowRisky: boolean;
  autoAdvancedSearch: boolean;
  autoOuterFolds: string;
  autoOuterTestWeeks: string;
  autoOuterGapWeeks: string;
  autoOuterSelectionMetric: AutoOuterSelectionMetric;
  autoOuterMinImproveFraction: string;
  autoOuterMaxWorstDelta: string;
};

type ModelCompareSelection = {
  left: string | null;
  right: string | null;
};

type RecommendedSplitFields = {
  splitStrategy: SplitStrategy;
  windowMode: WindowMode;
  trainWindowWeeks: string;
  validationFolds: string;
  validationWindowWeeks: string;
  testWindowWeeks: string;
  embargoDays: string;
  cGridPreset: CalibrateFormState["cGridPreset"];
  cGridCustom: string;
  calibrationMethod: CalibrationMethod;
};

const STORAGE_KEY = "polyedgetool.calibrate.v2.form";
const LAST_RESULT_KEY = "polyedgetool.calibrate.v2.last_result";

const DEFAULT_WEEK_COL = "week_friday";
const DEFAULT_TICKER_COL = "ticker";
const DEFAULT_TARGET_COL = "outcome_ST_gt_K";

const DEFAULT_TRADING_UNIVERSE = [
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

const C_GRID_PRESETS: Record<string, string> = {
  coarse: "0.03,0.3,3",
  standard: "0.01,0.03,0.1,0.3,1,3,10",
  wide: "0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30",
};

const AUTO_FEATURE_SETS = [
  ["x_logit_prn"],
  ["x_logit_prn", "rv20"],
  ["x_logit_prn", "abs_log_m_fwd"],
  ["x_logit_prn", "rv20", "abs_log_m_fwd"],
  ["x_logit_prn", "rv20", "abs_log_m_fwd", "log_rel_spread"],
];
const AUTO_C_VALUES = [0.003, 0.01, 0.03, 0.1, 0.3];
const AUTO_CAL_METHODS: CalibrationMethod[] = ["none", "platt"];
const AUTO_UPWEIGHTS = [1.0, 1.25, 1.5];
const AUTO_FOUNDATION_WEIGHTS = [1.0, 1.25, 1.5];
const AUTO_TICKER_INTERCEPTS = ["off", "on"] as const;

const TIME_REGIME_OPTIONS: Array<{
  key: TimeRegimeKey;
  label: string;
  helper: string;
  asofDow: "Mon" | "Tue" | "Wed" | "Thu";
  tdays: 4 | 3 | 2 | 1;
}> = [
  { key: "mon_4", label: "Monday", helper: "4 DTE", asofDow: "Mon", tdays: 4 },
  { key: "tue_3", label: "Tuesday", helper: "3 DTE", asofDow: "Tue", tdays: 3 },
  { key: "wed_2", label: "Wednesday", helper: "2 DTE", asofDow: "Wed", tdays: 2 },
  { key: "thu_1", label: "Thursday", helper: "1 DTE", asofDow: "Thu", tdays: 1 },
];

const BASE_FEATURE = "x_logit_prn";
const FEATURE_OPTIONS = [
  "log_m_fwd",
  "abs_log_m_fwd",
  "rv20",
  "rv20_sqrtT",
  "log_m_fwd_over_volT",
  "log_rel_spread",
  "had_fallback",
  "had_intrinsic_drop",
  "had_band_clip",
  "prn_raw_gap",
  "dividend_yield",
  "x_m",
  "x_abs_m",
] as const;
const FEATURE_OPTION_SET = new Set<string>(FEATURE_OPTIONS);
const CATEGORICAL_FEATURE_OPTIONS = ["spot_scale_used"] as const;
const CATEGORICAL_FEATURE_LABELS: Record<string, string> = {
  spot_scale_used: "Spot split scale adjusted",
};
const CATEGORICAL_FEATURE_OPTION_SET = new Set<string>(CATEGORICAL_FEATURE_OPTIONS);
const FEATURE_CATEGORIES: Array<{ title: string; items: readonly string[] }> = [
  { title: "Moneyness", items: ["log_m_fwd", "abs_log_m_fwd", "log_m_fwd_over_volT"] },
  { title: "Volatility", items: ["rv20", "rv20_sqrtT"] },
  { title: "Market Quality", items: ["log_rel_spread", "prn_raw_gap", "dividend_yield"] },
  { title: "Coverage and Sanity", items: ["had_fallback", "had_intrinsic_drop", "had_band_clip"] },
  { title: "Interactions", items: ["x_m", "x_abs_m"] },
];
const FEATURE_DEPENDENCIES: Record<string, string[]> = {
  x_m: ["log_m_fwd"],
  x_abs_m: ["abs_log_m_fwd"],
};
const FEATURE_MUTUAL_EXCLUSIVE_GROUPS: string[][] = [
  ["rv20", "rv20_sqrtT"],
  ["log_m_fwd", "abs_log_m_fwd"],
  ["x_m", "x_abs_m"],
];
const DEFAULT_SELECTED_FEATURES = [
  "log_m_fwd",
  "rv20",
  "log_m_fwd_over_volT",
  "log_rel_spread",
  "had_fallback",
  "had_intrinsic_drop",
  "had_band_clip",
  "prn_raw_gap",
  "dividend_yield",
];

const defaultModelName = () => {
  const stamp = new Date().toISOString().replace(/[:.]/g, "").slice(0, 15);
  return `calibration-${stamp}`;
};

const defaultForm = (): CalibrateFormState => ({
  runMode: "manual",
  modelDirName: "",
  datasetPath: "",
  randomSeed: "7",
  weightColStrategy: "auto",
  timeRegime: "thu_1",
  selectedFeatures: [...DEFAULT_SELECTED_FEATURES],
  selectedCategoricalFeatures: [],

  splitStrategy: "walk_forward",
  windowMode: "rolling",
  trainWindowWeeks: "52",
  validationFolds: "4",
  validationWindowWeeks: "8",
  testWindowWeeks: "20",
  embargoDays: "2",

  cGridPreset: "standard",
  cGridCustom: C_GRID_PRESETS.standard,
  calibrationMethod: "none",
  selectionObjective: "logloss",

  tradingUniverseTickers: [...DEFAULT_TRADING_UNIVERSE],
  trainTickers: [...DEFAULT_TRADING_UNIVERSE],
  foundationTickers: [...DEFAULT_TRADING_UNIVERSE],
  foundationWeight: "1.25",
  tickerInterceptMode: "non_foundation",
  perTickerInteractions: false,
  minSupportIntercepts: "300",
  minSupportInteractions: "1000",

  baseWeightSource: "dataset_weight",
  groupingKey: "group_id",
  groupEqualization: true,
  renorm: "mean1",
  tradingUniverseUpweight: "1.15",
  tickerBalanceMode: "none",

  bootstrapEnabled: false,
  bootstrapGroup: "contract_id",
  bootstrapDraws: "2000",
  bootstrapSeed: "0",
  ciLevel: 95,
  perSplitReporting: true,
  perFoldReporting: true,
  splitTimeline: true,
  perFoldDeltaChart: true,
  perGroupDeltaDistribution: false,

  maxAbsLogm: "",
  dropPrnExtremes: false,
  dropPrnBelow: "0.001",
  dropPrnAbove: "0.999",

  autoMaxTrials: "",
  autoAllowRisky: false,
  autoAdvancedSearch: false,
  autoOuterFolds: "0",
  autoOuterTestWeeks: "8",
  autoOuterGapWeeks: "1",
  autoOuterSelectionMetric: "median_delta_logloss",
  autoOuterMinImproveFraction: "0.75",
  autoOuterMaxWorstDelta: "0.005",
});

const parseOptionalInt = (value: string): number | undefined => {
  const trimmed = value.trim();
  if (!trimmed) return undefined;
  const parsed = Number.parseInt(trimmed, 10);
  return Number.isFinite(parsed) ? parsed : undefined;
};

const parseOptionalFloat = (value: string): number | undefined => {
  const trimmed = value.trim();
  if (!trimmed) return undefined;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : undefined;
};

const formatMetricValue = (value?: number | null): string =>
  value == null || Number.isNaN(value) ? "--" : value.toFixed(4);

const formatCountValue = (value?: number | null): string =>
  value == null || Number.isNaN(value) ? "--" : value.toLocaleString();

type ParsedCsv = {
  headers: string[];
  rows: Record<string, string>[];
};

const parseCsvLine = (line: string): string[] => {
  const out: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let idx = 0; idx < line.length; idx += 1) {
    const char = line[idx];
    if (char === "\"") {
      const next = line[idx + 1];
      if (inQuotes && next === "\"") {
        current += "\"";
        idx += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (char === "," && !inQuotes) {
      out.push(current);
      current = "";
      continue;
    }
    current += char;
  }
  out.push(current);
  return out;
};

const parseCsvContent = (content: string, limit = 5000): ParsedCsv => {
  const lines = content.split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (!lines.length) {
    return { headers: [], rows: [] };
  }
  const headers = parseCsvLine(lines[0]).map((h) => h.trim());
  const rows: Record<string, string>[] = [];
  for (let i = 1; i < Math.min(lines.length, limit + 1); i += 1) {
    const values = parseCsvLine(lines[i]);
    const row: Record<string, string> = {};
    headers.forEach((key, idx) => {
      row[key] = values[idx] ?? "";
    });
    rows.push(row);
  }
  return { headers, rows };
};

const parseJsonContent = (content: string): Record<string, unknown> | null => {
  try {
    const parsed = JSON.parse(content);
    if (parsed && typeof parsed === "object") {
      return parsed as Record<string, unknown>;
    }
    return null;
  } catch {
    return null;
  }
};

const toNumber = (value: unknown): number | null => {
  if (value == null) return null;
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  const parsed = Number(String(value).trim());
  return Number.isFinite(parsed) ? parsed : null;
};

const formatMaybe = (value: unknown): string => {
  if (value == null) return "--";
  if (typeof value === "number") return Number.isFinite(value) ? value.toFixed(4) : "--";
  if (typeof value === "string") return value.trim().length ? value : "--";
  return JSON.stringify(value);
};

const computeAvailableWeeks = (dataset?: DatasetFileSummary | null): number | null => {
  if (!dataset) return null;
  if (typeof dataset.week_count === "number" && Number.isFinite(dataset.week_count)) {
    return Math.max(1, Math.floor(dataset.week_count));
  }
  if (!dataset.date_start || !dataset.date_end) return null;
  const start = new Date(dataset.date_start);
  const end = new Date(dataset.date_end);
  if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) return null;
  const diffDays = Math.max(0, Math.floor((end.getTime() - start.getTime()) / (24 * 3600 * 1000)));
  if (dataset.date_col_used === "week_friday") {
    return Math.max(1, Math.floor(diffDays / 7) + 1);
  }
  return Math.max(1, Math.floor(diffDays / 7));
};

const clampNumber = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(max, value));

const recommendSplitConfig = ({
  weeks,
  dteDays,
  groupsPerWeek,
}: {
  weeks: number | null;
  dteDays: number | null;
  groupsPerWeek?: number | null;
}): { fields: RecommendedSplitFields; warning: string | null } | null => {
  if (!weeks || !Number.isFinite(weeks)) return null;
  const W = Math.max(1, Math.floor(weeks));
  const embargoDays = clampNumber(Math.floor(dteDays ?? 4), 0, 14);
  const embargoWeeks = Math.ceil(embargoDays / 7);
  const minTrain = 8;
  const minValWindow = 1;
  const minTest = 4;
  let testWeeks: number;
  let valFolds: number;
  let valWindowWeeks: number;
  let trainWindowWeeks: number;
  let warning: string | null = null;
  let adjusted = false;

  if (W < 40) {
    testWeeks = 12;
    valFolds = 2;
    valWindowWeeks = clampNumber(Math.round(0.18 * W), 6, 8);
    trainWindowWeeks = Math.round(0.72 * W);
  } else {
    testWeeks = clampNumber(Math.round(0.24 * W), 16, 32);
    valFolds = W >= 70 ? 3 : 2;
    valWindowWeeks = clampNumber(Math.round(0.12 * W), 8, 16);
    trainWindowWeeks = Math.round(0.78 * W);
  }

  if (groupsPerWeek && Number.isFinite(groupsPerWeek)) {
    const estGroups = groupsPerWeek * valWindowWeeks;
    if (estGroups < 200) {
      valWindowWeeks = Math.max(valWindowWeeks, 16);
      if ((groupsPerWeek * valWindowWeeks) < 200) {
        valFolds = 2;
      }
      adjusted = true;
      warning = "Validation window increased to reach group minimums.";
    }
  }

  const computeMaxTrain = (test: number, valWindow: number, folds: number) =>
    Math.max(0, W - test - (valWindow * folds) - embargoWeeks);

  const ensureFeasible = () => {
    let maxTrain = computeMaxTrain(testWeeks, valWindowWeeks, valFolds);
    while (maxTrain < minTrain && valFolds > 1) {
      valFolds -= 1;
      adjusted = true;
      maxTrain = computeMaxTrain(testWeeks, valWindowWeeks, valFolds);
    }
    while (maxTrain < minTrain && valWindowWeeks > minValWindow) {
      valWindowWeeks -= 1;
      adjusted = true;
      maxTrain = computeMaxTrain(testWeeks, valWindowWeeks, valFolds);
    }
    while (maxTrain < minTrain && testWeeks > minTest) {
      testWeeks -= 1;
      adjusted = true;
      maxTrain = computeMaxTrain(testWeeks, valWindowWeeks, valFolds);
    }
    return maxTrain;
  };

  let maxTrain = ensureFeasible();
  if (maxTrain < minTrain) {
    return null;
  } else {
    const targetTrain =
      W < 40 ? Math.min(26, Math.round(0.72 * W)) : Math.round(0.78 * W);
    trainWindowWeeks = clampNumber(targetTrain, minTrain, maxTrain);
  }

  if (trainWindowWeeks < valWindowWeeks) {
    valWindowWeeks = Math.min(valWindowWeeks, Math.max(minValWindow, trainWindowWeeks));
    maxTrain = computeMaxTrain(testWeeks, valWindowWeeks, valFolds);
    if (maxTrain >= minTrain) {
      trainWindowWeeks = clampNumber(trainWindowWeeks, minTrain, maxTrain);
    }
    adjusted = true;
  }

  if (adjusted && !warning) {
    warning = "Adjusted to fit dataset length and guardrails.";
  }

  return {
    fields: {
      splitStrategy: "walk_forward",
      windowMode: "rolling",
      trainWindowWeeks: String(Math.max(0, Math.floor(trainWindowWeeks))),
      validationFolds: String(Math.max(1, Math.floor(valFolds))),
      validationWindowWeeks: String(Math.max(1, Math.floor(valWindowWeeks))),
      testWindowWeeks: String(Math.max(1, Math.floor(testWeeks))),
      embargoDays: String(Math.max(0, Math.floor(embargoDays))),
      cGridPreset: "standard",
      cGridCustom: "0.01,0.03,0.1,0.3,1",
      calibrationMethod: "platt",
    },
    warning,
  };
};

const ARTIFACT_DESCRIPTIONS: Record<string, string> = {
  "metrics.csv": "Split metrics for baseline vs model with deltas and confidence intervals.",
  "metrics_summary.json": "Summary metrics across validation and test splits.",
  "split_timeline.json": "Walk-forward fold timeline and embargoed spans.",
  "fold_deltas.csv": "Per-fold delta metrics for validation windows.",
  "group_delta_distribution.csv": "Distribution of per-group delta logloss in test split.",
  "audit_split_composition.csv": "Split composition, class balance, and group counts.",
  "audit_overlap.json": "Overlap checks between splits for leakage detection.",
  "audit_weight_distribution.json": "Weight distribution diagnostics before and after reweighting.",
  "config.executed.json": "Normalized config as executed by the trainer.",
  "metadata.json": "Run metadata, selection outcomes, and warnings.",
  "feature_manifest.json": "Features used and required column manifest.",
  "best_config.json": "Best configuration discovered by auto search.",
  "best_model_report.md": "Summary report from auto-search run.",
  "leaderboard.csv": "Auto-search leaderboard ranked by objective (legacy).",
  "auto_search_leaderboard.csv": "Auto-search leaderboard ranked by objective.",
  "auto_search_summary.json": "Auto-search selection summary and chosen configuration.",
  "auto_search_no_viable.json": "Reasons and baseline snapshot when no candidate is accepted.",
  "auto_search_progress.json": "Auto-search progress state captured during the run.",
  "run_manifest.json": "Run-level manifest linking selected model and auto-search artifacts.",
  "outer_folds.json": "Outer backtest fold definitions and date ranges (auto search).",
  "outer_cv_summary.json": "Outer backtest configuration and selection summary (auto search).",
  "outer_fold_results.csv": "Per-fold outer backtest deltas for a specific trial (auto search).",
  "reliability_bins.csv": "Calibration reliability bins for predicted vs observed probability.",
  "rolling_summary.csv": "Rolling window summary metrics over time.",
  "rolling_windows.csv": "Rolling window-level metrics.",
  "metrics_groups.csv": "Metrics aggregated by group.",
  "two_stage_metrics.csv": "Two-stage metrics table.",
  "two_stage_metrics_summary.json": "Two-stage metrics summary.",
  "two_stage_metadata.json": "Two-stage metadata.",
};

const fileBaseName = (path: string | null | undefined): string => {
  if (!path) return "";
  const normalized = path.replace(/\\/g, "/");
  const parts = normalized.split("/");
  return parts[parts.length - 1] || normalized;
};

const KeyValueGrid = ({ data }: { data: Record<string, unknown> }) => {
  const entries = Object.entries(data);
  if (!entries.length) return <div className="empty">No data available.</div>;
  return (
    <div className="artifact-kv-grid">
      {entries.map(([key, value]) => {
        const text =
          value && typeof value === "object"
            ? JSON.stringify(value)
            : formatMaybe(value);
        return (
          <div key={key} className="artifact-kv-item">
            <span className="meta-label">{key}</span>
            <span>{text}</span>
          </div>
        );
      })}
    </div>
  );
};

const CsvTableView = ({ parsed, limit = 50 }: { parsed: ParsedCsv; limit?: number }) => {
  if (!parsed.headers.length) {
    return <div className="empty">CSV did not include headers.</div>;
  }
  const rows = parsed.rows.slice(0, limit);
  return (
    <div className="table-container artifact-table">
      <table className="preview-table">
        <thead>
          <tr>
            {parsed.headers.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length ? (
            rows.map((row, idx) => (
              <tr key={idx}>
                {parsed.headers.map((column) => (
                  <td key={column}>{row[column] ?? ""}</td>
                ))}
              </tr>
            ))
          ) : (
            <tr>
              <td colSpan={parsed.headers.length || 1}>No rows to display.</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
};

const buildMetricsSummaryFromCsv = (parsed: ParsedCsv) => {
  const rows = parsed.rows;
  const summary: Record<string, Record<string, string | number | null>> = {};
  rows.forEach((row) => {
    const split = row.split;
    const modelTag = row.model ?? "";
    if (!split) return;
    const bucket = summary[split] ?? {
      baseline_logloss: null,
      model_logloss: null,
      baseline_brier: null,
      model_brier: null,
      baseline_ece_q: null,
      model_ece_q: null,
      delta_logloss_ci_lo: null,
      delta_logloss_ci_hi: null,
      delta_brier_ci_lo: null,
      delta_brier_ci_hi: null,
      delta_ece_q_ci_lo: null,
      delta_ece_q_ci_hi: null,
    };
    const logloss = toNumber(row.logloss);
    const brier = toNumber(row.brier);
    const eceQ = toNumber(row.ece_q);
    if (modelTag.startsWith("baseline")) {
      bucket.baseline_logloss = logloss;
      bucket.baseline_brier = brier;
      bucket.baseline_ece_q = eceQ;
    } else if (!modelTag.startsWith("rolling")) {
      bucket.model_logloss = logloss;
      bucket.model_brier = brier;
      bucket.model_ece_q = eceQ;
      bucket.delta_logloss_ci_lo = toNumber(row.delta_logloss_ci_lo);
      bucket.delta_logloss_ci_hi = toNumber(row.delta_logloss_ci_hi);
      bucket.delta_brier_ci_lo = toNumber(row.delta_brier_ci_lo);
      bucket.delta_brier_ci_hi = toNumber(row.delta_brier_ci_hi);
      bucket.delta_ece_q_ci_lo = toNumber(row.delta_ece_q_ci_lo);
      bucket.delta_ece_q_ci_hi = toNumber(row.delta_ece_q_ci_hi);
    }
    summary[split] = bucket;
  });
  return summary;
};

const MetricsCsvView = ({ parsed, ciLabel }: { parsed: ParsedCsv; ciLabel: string }) => {
  const summary = buildMetricsSummaryFromCsv(parsed);
  const splits = Object.keys(summary);
  if (!splits.length) {
    return <div className="empty">No metrics rows found.</div>;
  }
  return (
    <div className="metrics-summary">
      <div className="metrics-summary-header">
        <span className="meta-label">Metrics summary</span>
        <span className="metrics-summary-note">Delta values are model minus baseline.</span>
      </div>
      <div className="metrics-summary-grid">
        {splits.map((split) => {
          const data = summary[split];
          const deltaLogloss =
            data.model_logloss != null && data.baseline_logloss != null
              ? Number(data.model_logloss) - Number(data.baseline_logloss)
              : null;
          const deltaBrier =
            data.model_brier != null && data.baseline_brier != null
              ? Number(data.model_brier) - Number(data.baseline_brier)
              : null;
          const deltaEceQ =
            data.model_ece_q != null && data.baseline_ece_q != null
              ? Number(data.model_ece_q) - Number(data.baseline_ece_q)
              : null;
          return (
            <div key={split} className="metrics-card">
              <div className="metrics-card-heading">
                <strong>{split}</strong>
                <span className={`status-pill ${deltaLogloss != null && deltaLogloss < 0 ? "success" : "failed"}`}>
                  {deltaLogloss != null && deltaLogloss < 0 ? "good" : "unusable"}
                </span>
              </div>
              <div className="metrics-card-row">
                <span>Baseline logloss</span>
                <strong>{formatMetricValue(data.baseline_logloss as number | null)}</strong>
              </div>
              <div className="metrics-card-row">
                <span>Model logloss</span>
                <strong>{formatMetricValue(data.model_logloss as number | null)}</strong>
              </div>
              <div className="metrics-card-row">
                <span>Delta logloss</span>
                <strong className={deltaMetricClass(deltaLogloss ?? null)}>{formatMetricValue(deltaLogloss)}</strong>
              </div>
              {data.delta_logloss_ci_lo != null && data.delta_logloss_ci_hi != null ? (
                <div className="metrics-card-row metrics-card-ci">
                  <span>Logloss {ciLabel}</span>
                  <strong>[{Number(data.delta_logloss_ci_lo).toFixed(4)}, {Number(data.delta_logloss_ci_hi).toFixed(4)}]</strong>
                </div>
              ) : null}
              <div className="metrics-card-row">
                <span>Baseline brier</span>
                <strong>{formatMetricValue(data.baseline_brier as number | null)}</strong>
              </div>
              <div className="metrics-card-row">
                <span>Model brier</span>
                <strong>{formatMetricValue(data.model_brier as number | null)}</strong>
              </div>
              <div className="metrics-card-row">
                <span>Delta brier</span>
                <strong className={deltaMetricClass(deltaBrier ?? null)}>{formatMetricValue(deltaBrier)}</strong>
              </div>
              {data.delta_brier_ci_lo != null && data.delta_brier_ci_hi != null ? (
                <div className="metrics-card-row metrics-card-ci">
                  <span>Brier {ciLabel}</span>
                  <strong>[{Number(data.delta_brier_ci_lo).toFixed(4)}, {Number(data.delta_brier_ci_hi).toFixed(4)}]</strong>
                </div>
              ) : null}
              <div className="metrics-card-row">
                <span>Baseline ece_q</span>
                <strong>{formatMetricValue(data.baseline_ece_q as number | null)}</strong>
              </div>
              <div className="metrics-card-row">
                <span>Model ece_q</span>
                <strong>{formatMetricValue(data.model_ece_q as number | null)}</strong>
              </div>
              <div className="metrics-card-row">
                <span>Delta ece_q</span>
                <strong className={deltaMetricClass(deltaEceQ ?? null)}>{formatMetricValue(deltaEceQ)}</strong>
              </div>
              {data.delta_ece_q_ci_lo != null && data.delta_ece_q_ci_hi != null ? (
                <div className="metrics-card-row metrics-card-ci">
                  <span>ECE-Q {ciLabel}</span>
                  <strong>[{Number(data.delta_ece_q_ci_lo).toFixed(4)}, {Number(data.delta_ece_q_ci_hi).toFixed(4)}]</strong>
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
};

const JsonSectionView = ({
  title,
  data,
}: {
  title: string;
  data: Record<string, unknown> | null;
}) => {
  if (!data) return null;
  return (
    <section className="artifact-section">
      <span className="meta-label">{title}</span>
      <KeyValueGrid data={data} />
    </section>
  );
};

const ConfigJsonView = ({ data }: { data: Record<string, unknown> }) => (
  <div className="artifact-stack">
    <JsonSectionView title="Dataset" data={{ csv: data.csv, out_dir: data.out_dir, run_mode: data.run_mode }} />
    <JsonSectionView title="Split" data={(data.split as Record<string, unknown>) ?? null} />
    <JsonSectionView title="Regularization" data={(data.regularization as Record<string, unknown>) ?? null} />
    <JsonSectionView title="Model structure" data={(data.model_structure as Record<string, unknown>) ?? null} />
    <JsonSectionView title="Weighting" data={(data.weighting as Record<string, unknown>) ?? null} />
    <JsonSectionView title="Bootstrap" data={(data.bootstrap as Record<string, unknown>) ?? null} />
    <JsonSectionView title="Diagnostics" data={(data.diagnostics as Record<string, unknown>) ?? null} />
  </div>
);

const MetadataView = ({ data }: { data: Record<string, unknown> }) => (
  <div className="artifact-stack">
    <JsonSectionView
      title="Run summary"
      data={{
        best_C: data.best_C,
        calibration_used: data.calibration_used,
        selection_objective: data.selection_objective,
        split_strategy: data.split_strategy,
        window_mode: data.window_mode,
        train_window_weeks: data.train_window_weeks,
        validation_folds: data.validation_folds,
        validation_window_weeks: data.validation_window_weeks,
        embargo_days: data.embargo_days,
        random_state: data.random_state,
      }}
    />
    <JsonSectionView
      title="Row counts"
      data={{
        train_fit_rows: data.train_fit_rows,
        val_rows: data.val_rows,
        train_rows: data.train_rows,
        test_rows: data.test_rows,
      }}
    />
    <JsonSectionView title="Warnings" data={{ warnings: data.warnings, ignored: data.unsupported_controls_ignored }} />
  </div>
);

const FeatureManifestView = ({ data }: { data: Record<string, unknown> }) => {
  const numeric = Array.isArray(data.numeric_features) ? data.numeric_features : [];
  const categorical = Array.isArray(data.categorical_features) ? data.categorical_features : [];
  const required = Array.isArray(data.required_columns) ? data.required_columns : [];
  return (
    <div className="artifact-stack">
      <section className="artifact-section">
        <span className="meta-label">Numeric features</span>
        <div className="artifact-chip-grid">
          {numeric.length ? numeric.map((item) => <span key={item as string} className="status-pill">{item as string}</span>) : <span className="empty">None</span>}
        </div>
      </section>
      <section className="artifact-section">
        <span className="meta-label">Categorical features</span>
        <div className="artifact-chip-grid">
          {categorical.length ? categorical.map((item) => <span key={item as string} className="status-pill">{item as string}</span>) : <span className="empty">None</span>}
        </div>
      </section>
      <section className="artifact-section">
        <span className="meta-label">Required columns</span>
        <div className="artifact-chip-grid">
          {required.length ? required.map((item) => <span key={item as string} className="status-pill">{item as string}</span>) : <span className="empty">None</span>}
        </div>
      </section>
    </div>
  );
};

const ReportMarkdownView = ({ content }: { content: string }) => {
  const lines = content.split(/\r?\n/);
  return (
    <div className="artifact-markdown">
      {lines.map((line, idx) => {
        if (line.startsWith("# ")) return <h3 key={idx}>{line.replace(/^#\s*/, "")}</h3>;
        if (line.startsWith("## ")) return <h4 key={idx}>{line.replace(/^##\s*/, "")}</h4>;
        if (line.startsWith("### ")) return <h5 key={idx}>{line.replace(/^###\s*/, "")}</h5>;
        return line.trim().length ? <p key={idx}>{line}</p> : <div key={idx} className="artifact-spacer" />;
      })}
    </div>
  );
};

const SplitTimelineView = ({ data }: { data: Record<string, unknown> }) => {
  const folds = Array.isArray(data.folds) ? data.folds : [];
  const parseTime = (value: unknown) => {
    if (value == null) return null;
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (value instanceof Date) {
      const time = value.getTime();
      return Number.isFinite(time) ? time : null;
    }
    const raw = String(value).trim();
    if (!raw) return null;
    const direct = Date.parse(raw);
    if (!Number.isNaN(direct)) return direct;
    if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) {
      const fallback = Date.parse(`${raw}T00:00:00Z`);
      return Number.isNaN(fallback) ? null : fallback;
    }
    return null;
  };
  const parseRange = (range: unknown) => {
    if (!Array.isArray(range) || range.length < 2) return null;
    const start = parseTime(range[0]);
    const end = parseTime(range[1]);
    if (start == null || end == null) return null;
    return { start, end };
  };
  const entries = folds.map((fold) => ({
    fold: toNumber((fold as Record<string, unknown>).fold) ?? 0,
    trainStart: parseTime((fold as Record<string, unknown>).train_start),
    trainEnd: parseTime((fold as Record<string, unknown>).train_end),
    valStart: parseTime((fold as Record<string, unknown>).val_start),
    valEnd: parseTime((fold as Record<string, unknown>).val_end),
    nTrain: toNumber((fold as Record<string, unknown>).n_train_rows) ?? null,
    nVal: toNumber((fold as Record<string, unknown>).n_val_rows) ?? null,
    embargoDropped: toNumber((fold as Record<string, unknown>).embargo_rows_dropped_train) ?? null,
  }));
  const trainRange = parseRange(data.train_range);
  const valRange = parseRange(data.val_range);
  const testRange = parseRange(data.test_range);
  const baseGlobalRows = [
    trainRange ? { label: "Train", start: trainRange.start, end: trainRange.end, kind: "train" as const } : null,
    valRange ? { label: "Val", start: valRange.start, end: valRange.end, kind: "val" as const } : null,
    testRange ? { label: "Test", start: testRange.start, end: testRange.end, kind: "test" as const } : null,
  ].filter((row): row is { label: string; start: number; end: number; kind: "train" | "val" | "test" } => !!row);
  const globalRows = folds.length
    ? baseGlobalRows.filter((row) => row.kind === "test")
    : baseGlobalRows;
  const times = [
    ...entries.flatMap((entry) => [entry.trainStart, entry.trainEnd, entry.valStart, entry.valEnd]),
    ...globalRows.flatMap((row) => [row.start, row.end]),
  ].filter((v): v is number => v != null);
  if (!entries.length && !globalRows.length) {
    return <div className="empty">No timeline data available.</div>;
  }
  if (!times.length) return <div className="empty">Timeline dates missing.</div>;
  const minTime = Math.min(...times);
  const maxTime = Math.max(...times);
  const width = 640;
  const rowHeight = 18;
  const topPad = 24;
  const totalRows = entries.length + globalRows.length;
  const height = topPad + totalRows * rowHeight + 24;
  const scaleX = (time: number) => {
    const ratio = (time - minTime) / Math.max(1, maxTime - minTime);
    return 40 + ratio * (width - 80);
  };
  const summaryData = {
    split_strategy: data.split_strategy,
    window_mode: data.window_mode,
    fold_count: data.fold_count ?? entries.length,
    embargo_days: data.embargo_days,
    embargo_mode: data.embargo_mode,
    embargo_date_col_used: data.embargo_date_col_used,
  };
  const rangeData = {
    train_range: trainRange
      ? `${new Date(trainRange.start).toISOString().slice(0, 10)} → ${new Date(trainRange.end).toISOString().slice(0, 10)}`
      : "n/a",
    val_range: valRange
      ? `${new Date(valRange.start).toISOString().slice(0, 10)} → ${new Date(valRange.end).toISOString().slice(0, 10)}`
      : "n/a",
    test_range: testRange
      ? `${new Date(testRange.start).toISOString().slice(0, 10)} → ${new Date(testRange.end).toISOString().slice(0, 10)}`
      : "n/a",
  };
  return (
    <div className="artifact-stack">
      <div className="artifact-header-row">
        <span className="meta-label">Split timeline</span>
        <div className="artifact-legend">
          <span className="artifact-legend-item">
            <span className="artifact-swatch artifact-swatch-train" /> Train
          </span>
          <span className="artifact-legend-item">
            <span className="artifact-swatch artifact-swatch-val" /> Val
          </span>
          <span className="artifact-legend-item">
            <span className="artifact-swatch artifact-swatch-test" /> Test
          </span>
        </div>
      </div>
      <KeyValueGrid data={summaryData} />
      <KeyValueGrid data={rangeData} />
      <svg viewBox={`0 0 ${width} ${height}`} className="artifact-chart">
        <rect x={0} y={0} width={width} height={height} className="chart-frame" />
        {entries.map((entry, idx) => {
          const y = topPad + idx * rowHeight;
          const trainStart = entry.trainStart != null ? scaleX(entry.trainStart) : null;
          const trainEnd = entry.trainEnd != null ? scaleX(entry.trainEnd) : null;
          const valStart = entry.valStart != null ? scaleX(entry.valStart) : null;
          const valEnd = entry.valEnd != null ? scaleX(entry.valEnd) : null;
          return (
            <g key={`fold-${idx}`}>
              <text x={8} y={y + 12} className="artifact-axis-label">F{entry.fold || idx + 1}</text>
              {trainStart != null && trainEnd != null ? (
                <rect x={trainStart} y={y} width={Math.max(1, trainEnd - trainStart)} height={10} className="artifact-bar-train" />
              ) : null}
              {valStart != null && valEnd != null ? (
                <rect x={valStart} y={y} width={Math.max(1, valEnd - valStart)} height={10} className="artifact-bar-val" />
              ) : null}
            </g>
          );
        })}
        {globalRows.map((row, idx) => {
          const y = topPad + (entries.length + idx) * rowHeight;
          const start = scaleX(row.start);
          const end = scaleX(row.end);
          const barClass =
            row.kind === "test"
              ? "artifact-bar-test"
              : row.kind === "val"
                ? "artifact-bar-val"
                : "artifact-bar-train";
          return (
            <g key={`global-${row.label}`}>
              <text x={8} y={y + 12} className="artifact-axis-label">{row.label}</text>
              <rect x={start} y={y} width={Math.max(1, end - start)} height={10} className={barClass} />
            </g>
          );
        })}
      </svg>
      <CsvTableView
        parsed={{
          headers: [
            "fold",
            "train_start",
            "train_end",
            "val_start",
            "val_end",
            "n_train_rows",
            "n_val_rows",
            "embargo_rows_dropped_train",
          ],
          rows: entries.map((entry) => ({
            fold: String(entry.fold || ""),
            train_start: entry.trainStart ? new Date(entry.trainStart).toISOString().slice(0, 10) : "",
            train_end: entry.trainEnd ? new Date(entry.trainEnd).toISOString().slice(0, 10) : "",
            val_start: entry.valStart ? new Date(entry.valStart).toISOString().slice(0, 10) : "",
            val_end: entry.valEnd ? new Date(entry.valEnd).toISOString().slice(0, 10) : "",
            n_train_rows: entry.nTrain != null ? String(entry.nTrain) : "",
            n_val_rows: entry.nVal != null ? String(entry.nVal) : "",
            embargo_rows_dropped_train: entry.embargoDropped != null ? String(entry.embargoDropped) : "",
          })),
        }}
        limit={100}
      />
    </div>
  );
};

const AuditOverlapView = ({ data }: { data: Record<string, unknown> }) => {
  const overlap = (data.split_overlap as Record<string, unknown>) ?? {};
  const entries = Object.entries(overlap);
  return (
    <div className="artifact-stack">
      <span className="meta-label">Overlap checks</span>
      <div className="artifact-kv-grid">
        {entries.map(([key, value]) => (
          <div key={key} className="artifact-kv-item">
            <span className="meta-label">{key}</span>
            <span className={toNumber(value) && Number(value) > 0 ? "artifact-warn" : ""}>
              {formatMaybe(value)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

const AuditWeightView = ({ data }: { data: Record<string, unknown> }) => {
  const raw = (data.raw as Record<string, unknown>) ?? {};
  const finalWeights = (data.final_train_weights as Record<string, unknown>) ?? {};
  const rawSums = (data.raw_group_sums as Record<string, unknown>) ?? {};
  const finalSums = (data.final_group_sums as Record<string, unknown>) ?? {};
  return (
    <div className="artifact-stack">
      <JsonSectionView title="Weight source" data={{ weight_col_requested: data.weight_col_requested, weight_source: data.weight_source }} />
      <JsonSectionView title="Raw weights" data={raw} />
      <JsonSectionView title="Final weights" data={finalWeights} />
      <JsonSectionView title="Raw group sums" data={rawSums} />
      <JsonSectionView title="Final group sums" data={finalSums} />
    </div>
  );
};

const SplitCompositionView = ({ parsed }: { parsed: ParsedCsv }) => (
  <div className="artifact-stack">
    <span className="meta-label">Split composition</span>
    <CsvTableView parsed={parsed} limit={200} />
  </div>
);

const RollingSummaryView = ({ parsed }: { parsed: ParsedCsv }) => {
  const metricColumn = parsed.headers.find((h) => h.toLowerCase().includes("logloss")) ?? parsed.headers[0];
  const values = parsed.rows.map((row, idx) => ({ x: idx, y: toNumber(row[metricColumn]) ?? 0 }));
  if (!values.length) return <div className="empty">No rolling data.</div>;
  const min = Math.min(...values.map((p) => p.y));
  const max = Math.max(...values.map((p) => p.y));
  const width = 520;
  const height = 160;
  const pad = 20;
  const scaleX = (x: number) => pad + (x / Math.max(1, values.length - 1)) * (width - pad * 2);
  const scaleY = (y: number) => pad + (1 - (y - min) / Math.max(1e-9, max - min)) * (height - pad * 2);
  const path = values
    .map((point, idx) => `${idx === 0 ? "M" : "L"} ${scaleX(point.x)} ${scaleY(point.y)}`)
    .join(" ");
  return (
    <div className="artifact-stack">
      <span className="meta-label">Rolling summary ({metricColumn})</span>
      <svg viewBox={`0 0 ${width} ${height}`} className="artifact-chart">
        <rect x={0} y={0} width={width} height={height} className="chart-frame" />
        <path d={path} className="chart-line chart-line-prn" />
      </svg>
      <CsvTableView parsed={parsed} limit={50} />
    </div>
  );
};

const FoldDeltaView = ({ parsed }: { parsed: ParsedCsv }) => {
  const [metric, setMetric] = useState<"delta_logloss" | "delta_brier" | "delta_ece_q">("delta_logloss");
  const rows = parsed.rows.map((row) => ({
    fold: toNumber(row.fold) ?? 0,
    value: toNumber(row[metric]) ?? 0,
  }));
  const values = rows.map((row) => row.value);
  if (!rows.length) return <div className="empty">No fold deltas available.</div>;
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 0);
  const width = 520;
  const height = 180;
  const pad = 24;
  const innerWidth = width - pad * 2;
  const innerHeight = height - pad * 2;
  const scaleY = (y: number) => pad + (1 - (y - min) / Math.max(1e-9, max - min)) * innerHeight;
  const zeroY = scaleY(0);
  const step = innerWidth / Math.max(1, rows.length);
  const barWidth = Math.max(8, step * 0.6);
  return (
    <div className="artifact-stack">
      <div className="artifact-header-row">
        <span className="meta-label">Per-fold delta chart</span>
        <select className="input small" value={metric} onChange={(event) => setMetric(event.target.value as typeof metric)}>
          <option value="delta_logloss">delta_logloss</option>
          <option value="delta_brier">delta_brier</option>
          <option value="delta_ece_q">delta_ece_q</option>
        </select>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="artifact-chart">
        <rect x={0} y={0} width={width} height={height} className="chart-frame" />
        <line x1={pad} x2={width - pad} y1={zeroY} y2={zeroY} className="chart-midline" />
        {rows.map((row, idx) => {
          const x = pad + idx * step + (step - barWidth) / 2;
          const y = row.value >= 0 ? scaleY(row.value) : zeroY;
          const barHeight = Math.max(1, Math.abs(scaleY(row.value) - zeroY));
          const barClass = row.value >= 0 ? "artifact-bar-positive" : "artifact-bar-negative";
          return (
            <rect
              key={`fold-${idx}`}
              x={x}
              y={y}
              width={barWidth}
              height={barHeight}
              className={barClass}
            />
          );
        })}
      </svg>
      <CsvTableView parsed={parsed} limit={50} />
    </div>
  );
};

const GroupDeltaDistributionView = ({ parsed }: { parsed: ParsedCsv }) => {
  const values = parsed.rows.map((row) => toNumber(row.delta_logloss)).filter((v): v is number => v != null);
  if (!values.length) return <div className="empty">No group delta values.</div>;
  const bins = 20;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const step = (max - min) / bins || 1;
  const counts = Array.from({ length: bins }, () => 0);
  values.forEach((value) => {
    const idx = Math.min(bins - 1, Math.floor((value - min) / step));
    counts[idx] += 1;
  });
  const maxCount = Math.max(...counts);
  return (
    <div className="artifact-stack">
      <span className="meta-label">Group delta distribution</span>
      <div className="artifact-bar-chart">
        {counts.map((count, idx) => (
          <div key={idx} className="artifact-bar">
            <div
              className="artifact-bar-fill"
              style={{ height: `${(count / Math.max(1, maxCount)) * 100}%` }}
            />
          </div>
        ))}
      </div>
      <CsvTableView parsed={parsed} limit={50} />
    </div>
  );
};

const ReliabilityView = ({ parsed }: { parsed: ParsedCsv }) => {
  const predKey = parsed.headers.find((h) => h.toLowerCase().includes("pred")) ?? parsed.headers[0];
  const obsKey = parsed.headers.find((h) => h.toLowerCase().includes("obs")) ?? parsed.headers[1];
  const points = parsed.rows
    .map((row) => ({
      x: toNumber(row[predKey]),
      y: toNumber(row[obsKey]),
    }))
    .filter((p): p is { x: number; y: number } => p.x != null && p.y != null);
  if (!points.length) return <div className="empty">No reliability bins.</div>;
  const width = 520;
  const height = 180;
  const pad = 20;
  const scale = (v: number) => pad + v * (width - pad * 2);
  const scaleY = (v: number) => height - pad - v * (height - pad * 2);
  const path = points.map((p, idx) => `${idx === 0 ? "M" : "L"} ${scale(p.x)} ${scaleY(p.y)}`).join(" ");
  return (
    <div className="artifact-stack">
      <span className="meta-label">Reliability plot</span>
      <svg viewBox={`0 0 ${width} ${height}`} className="artifact-chart">
        <rect x={0} y={0} width={width} height={height} className="chart-frame" />
        <line x1={pad} y1={height - pad} x2={width - pad} y2={pad} className="chart-midline" />
        <path d={path} className="chart-line chart-line-prn" />
      </svg>
      <CsvTableView parsed={parsed} limit={50} />
    </div>
  );
};

const deltaMetricClass = (value?: number | null): string | undefined => {
  if (value == null || Number.isNaN(value)) return undefined;
  return value <= 0 ? "delta-negative" : "delta-positive";
};

const formatTimestamp = (value?: string | null): string => {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
};

const loadStoredForm = (): CalibrateFormState | null => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<CalibrateFormState>;
    return {
      ...defaultForm(),
      ...parsed,
      selectedFeatures: Array.isArray(parsed.selectedFeatures)
        ? parsed.selectedFeatures.filter((feature): feature is string => typeof feature === "string")
        : [...DEFAULT_SELECTED_FEATURES],
      selectedCategoricalFeatures: Array.isArray(parsed.selectedCategoricalFeatures)
        ? parsed.selectedCategoricalFeatures.filter(
            (feature): feature is string =>
              typeof feature === "string" && CATEGORICAL_FEATURE_OPTION_SET.has(feature),
          )
        : [],
    };
  } catch {
    return null;
  }
};

const loadStoredResult = (): CalibrateModelRunResponse | null => {
  try {
    const raw = localStorage.getItem(LAST_RESULT_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as CalibrateModelRunResponse;
  } catch {
    return null;
  }
};

const saveResult = (result: CalibrateModelRunResponse | null) => {
  try {
    if (!result) {
      localStorage.removeItem(LAST_RESULT_KEY);
      return;
    }
    localStorage.setItem(LAST_RESULT_KEY, JSON.stringify(result));
  } catch {
    // ignore
  }
};

const joinCsv = (values: string[]): string | undefined => {
  const cleaned = Array.from(new Set(values.map((v) => v.trim()).filter(Boolean)));
  return cleaned.length ? cleaned.join(",") : undefined;
};

const statusClass = (status?: string | null) => {
  if (status === "running" || status === "queued") return "running";
  if (status === "finished") return "success";
  if (status === "failed" || status === "cancelled") return "failed";
  return "idle";
};

const getActiveResult = (
  jobStatus: CalibrationJobStatus | null,
  lastResult: CalibrateModelRunResponse | null,
): CalibrateModelRunResponse | null => {
  if (jobStatus) {
    return jobStatus.result ?? null;
  }
  return lastResult;
};

const sanitizeModelDirName = (value: string): string =>
  value.replace(/[^A-Za-z0-9._-]/g, "").trim();

const getTickerSummaryText = (dataset?: DatasetFileSummary): string => {
  if (!dataset) return "No dataset selected.";
  const count = dataset.ticker_count ?? 0;
  const sample = dataset.ticker_sample ?? [];
  if (!count) return "Ticker summary unavailable.";
  if (!sample.length) return `${count} tickers`;
  return `${count} tickers (${sample.join(", ")}${count > sample.length ? ", ..." : ""})`;
};

const getTimeRegime = (key: TimeRegimeKey) =>
  TIME_REGIME_OPTIONS.find((option) => option.key === key) ?? TIME_REGIME_OPTIONS[0];

const featureOrderIndex = new Map<string, number>(
  FEATURE_OPTIONS.map((feature, index) => [feature, index]),
);
const categoricalFeatureOrderIndex = new Map<string, number>(
  CATEGORICAL_FEATURE_OPTIONS.map((feature, index) => [feature, index]),
);

const normalizeFeatureSelection = (
  features: string[],
  availableSet: Set<string>,
): string[] => {
  const selected = new Set(
    features.filter((feature) => FEATURE_OPTION_SET.has(feature) && availableSet.has(feature)),
  );

  let changed = true;
  while (changed) {
    changed = false;

    for (const [feature, deps] of Object.entries(FEATURE_DEPENDENCIES)) {
      if (!selected.has(feature)) continue;
      for (const dep of deps) {
        if (selected.has(dep)) continue;
        if (availableSet.has(dep)) {
          selected.add(dep);
          changed = true;
        } else {
          selected.delete(feature);
          changed = true;
          break;
        }
      }
    }

    for (const group of FEATURE_MUTUAL_EXCLUSIVE_GROUPS) {
      const chosen = group.filter((feature) => selected.has(feature));
      if (chosen.length <= 1) continue;
      const keep =
        features.find((candidate) => chosen.includes(candidate)) ??
        chosen[0];
      chosen.forEach((feature) => {
        if (feature === keep) return;
        if (selected.delete(feature)) {
          dropDependentFeatures(selected, feature);
          changed = true;
        }
      });
    }
  }

  return Array.from(selected).sort(
    (left, right) =>
      (featureOrderIndex.get(left) ?? Number.MAX_SAFE_INTEGER) -
      (featureOrderIndex.get(right) ?? Number.MAX_SAFE_INTEGER),
  );
};

const normalizeCategoricalSelection = (
  features: string[],
  availableSet: Set<string>,
): string[] => {
  const selected = Array.from(
    new Set(
      features.filter(
        (feature) =>
          CATEGORICAL_FEATURE_OPTION_SET.has(feature) && availableSet.has(feature),
      ),
    ),
  );
  return selected.sort(
    (left, right) =>
      (categoricalFeatureOrderIndex.get(left) ?? Number.MAX_SAFE_INTEGER) -
      (categoricalFeatureOrderIndex.get(right) ?? Number.MAX_SAFE_INTEGER),
  );
};

const dropDependentFeatures = (selected: Set<string>, removed: string) => {
  const queue = [removed];
  while (queue.length > 0) {
    const current = queue.pop()!;
    Object.entries(FEATURE_DEPENDENCIES).forEach(([feature, deps]) => {
      if (!selected.has(feature)) return;
      if (!deps.includes(current)) return;
      selected.delete(feature);
      queue.push(feature);
    });
  }
};

const LatexBlock = ({ latex }: { latex: string }) => {
  const rendered = useMemo(() => {
    try {
      return katex.renderToString(latex, {
        throwOnError: false,
        displayMode: true,
      });
    } catch {
      return "";
    }
  }, [latex]);

  if (!rendered) {
    return <pre className="calibrate-equation-fallback">{latex}</pre>;
  }

  return (
    <div
      className="calibrate-equation"
      dangerouslySetInnerHTML={{ __html: rendered }}
    />
  );
};

const metricsOrder = ["val", "test", "val_pool"];

export default function CalibrateModelsPage() {
  const [workspaceTab, setWorkspaceTab] = useState<WorkspaceTab>("run_job");
  const [runJobPanel, setRunJobPanel] = useState<RunJobPanel>("configuration");
  const [activeLog, setActiveLog] = useState<"stdout" | "stderr">("stdout");

  const [form, setForm] = useState<CalibrateFormState>(() => loadStoredForm() ?? defaultForm());
  const [datasets, setDatasets] = useState<DatasetFileSummary[]>([]);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [cancelError, setCancelError] = useState<string | null>(null);
  const [cancelLoading, setCancelLoading] = useState(false);
  const [guardrailWarning, setGuardrailWarning] = useState<string | null>(null);
  const [lastRunResult, setLastRunResult] = useState<CalibrateModelRunResponse | null>(() => loadStoredResult());
  const [regimePreview, setRegimePreview] = useState<RegimePreviewResponse | null>(null);
  const [regimePreviewError, setRegimePreviewError] = useState<string | null>(null);
  const [availableFeatureColumns, setAvailableFeatureColumns] = useState<string[]>([]);
  const [featureError, setFeatureError] = useState<string | null>(null);
  const [featuresLoading, setFeaturesLoading] = useState(false);

  const [availableTickers, setAvailableTickers] = useState<string[]>([]);
  const [tickersLoading, setTickersLoading] = useState(false);
  const [tickersError, setTickersError] = useState<string | null>(null);

  const [weightingPreview, setWeightingPreview] = useState<WeightingPreviewResponse | null>(null);
  const [weightingPreviewError, setWeightingPreviewError] = useState<string | null>(null);
  const [weightingPreviewLoading, setWeightingPreviewLoading] = useState(false);

  const [models, setModels] = useState<ModelRunSummary[]>([]);
  const [modelError, setModelError] = useState<string | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [isModelDetailLoading, setIsModelDetailLoading] = useState(false);
  const [modelDetail, setModelDetail] = useState<ModelDetailResponse | null>(null);
  const [modelDetailError, setModelDetailError] = useState<string | null>(null);
  const [modelFiles, setModelFiles] = useState<ModelFilesListResponse | null>(null);
  const [selectedFilePath, setSelectedFilePath] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<ModelFileContentResponse | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [fileLoading, setFileLoading] = useState(false);
  const [showRawFile, setShowRawFile] = useState(false);
  const [modelCompare, setModelCompare] = useState<ModelCompareSelection>({ left: null, right: null });
  const [deleteTarget, setDeleteTarget] = useState<ModelRunSummary | null>(null);
  const [deleteConfirmText, setDeleteConfirmText] = useState("");
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [lastRunCiLevel, setLastRunCiLevel] = useState<90 | 95 | 99 | null>(null);
  const [splitRecommended, setSplitRecommended] = useState(false);
  const [splitRecommendationWarning, setSplitRecommendationWarning] = useState<string | null>(null);
  const lastRecommendedRef = useRef<RecommendedSplitFields | null>(null);
  const lastModelRefreshRef = useRef<string | null>(null);

  const { jobId, jobStatus, setJobId, setJobStatus } = useCalibrationJob();
  const { anyJobRunning, primaryJob, activeJobs } = useAnyJobRunning();

  const selectedDataset = useMemo(
    () => datasets.find((item) => item.path === form.datasetPath),
    [datasets, form.datasetPath],
  );
  const availableWeeks = useMemo(
    () => computeAvailableWeeks(selectedDataset ?? null),
    [selectedDataset],
  );
  const selectedTimeRegime = useMemo(
    () => getTimeRegime(form.timeRegime),
    [form.timeRegime],
  );
  const isAuto = form.runMode === "auto";

  const canRecommendSplit = Boolean(form.datasetPath && availableWeeks != null && availableWeeks > 0);

  const hasLiveJob = Boolean(
    jobId && (!jobStatus || jobStatus.status === "queued" || jobStatus.status === "running"),
  );
  const isRunning = hasLiveJob;
  const isJobComplete = jobStatus?.status === "finished";

  const activeResult = useMemo(
    () => getActiveResult(jobStatus, lastRunResult),
    [jobStatus, lastRunResult],
  );
  const runProgress = useMemo(() => {
    const progress = jobStatus?.progress ?? null;
    if (!progress) return null;
    const total = progress.trials_total ?? 0;
    const completed = progress.trials_done ?? 0;
    const failed = progress.trials_failed ?? 0;
    const status =
      jobStatus?.status === "failed"
        ? "failed"
        : jobStatus?.status === "finished"
          ? "completed"
          : "running";
    return { total, completed, failed, status };
  }, [jobStatus?.progress, jobStatus?.status]);

  const metricsSummary = activeResult?.metrics_summary ?? null;
  const splitRowCounts = activeResult?.split_row_counts ?? null;
  const splitGroupCounts = activeResult?.split_group_counts ?? null;
  const trainRows = splitRowCounts?.train_fit ?? splitRowCounts?.train ?? null;
  const valRows = splitRowCounts?.val ?? null;
  const testRows = splitRowCounts?.test ?? null;
  const trainGroups = splitGroupCounts?.train_fit ?? splitGroupCounts?.train ?? null;
  const valGroups = splitGroupCounts?.val ?? null;
  const testGroups = splitGroupCounts?.test ?? null;
  const hasUsageCounts =
    trainRows != null ||
    valRows != null ||
    testRows != null ||
    trainGroups != null ||
    valGroups != null ||
    testGroups != null;
  const activeCiLevel = lastRunCiLevel ?? form.ciLevel;
  const ciLabel = activeCiLevel ? `CI (${activeCiLevel}%)` : "CI";
  const modelDetailCiLevel = toNumber(
    (modelDetail?.metadata as Record<string, unknown> | null | undefined)?.["ci_level"],
  );
  const modelCiLabel = modelDetailCiLevel ? `CI (${modelDetailCiLevel}%)` : "CI";
  const autoProgress = jobStatus?.mode === "auto" ? jobStatus.progress ?? null : null;

  const autoProgressLog = useMemo(() => {
    if (!autoProgress) return "";
    const lines: string[] = [];
    const stage = autoProgress.message || autoProgress.stage;
    if (stage) lines.push(`AUTO PROGRESS: ${stage}`);
    if (autoProgress.trials_total) {
      lines.push(`Trials: ${autoProgress.trials_done}/${autoProgress.trials_total} (failed: ${autoProgress.trials_failed})`);
    }
    if (autoProgress.candidate_index != null && autoProgress.candidate_total != null) {
      lines.push(`Candidate: ${autoProgress.candidate_index}/${autoProgress.candidate_total}`);
    }
    if (autoProgress.fold_index != null && autoProgress.fold_total != null) {
      lines.push(`Fold: ${autoProgress.fold_index}/${autoProgress.fold_total}`);
    }
    if (autoProgress.best_score_so_far != null) {
      lines.push(`Best score: ${autoProgress.best_score_so_far.toFixed(5)}`);
    }
    if (autoProgress.last_log_lines?.length) {
      lines.push("");
      lines.push(...autoProgress.last_log_lines);
    }
    return lines.join("\n");
  }, [autoProgress]);

  const groupedModelFiles = useMemo(() => {
    const groups: Record<"selected_model" | "auto_search" | "legacy_root", ModelFileSummary[]> = {
      selected_model: [],
      auto_search: [],
      legacy_root: [],
    };
    for (const file of modelFiles?.files ?? []) {
      const section = file.section ?? "legacy_root";
      if (section === "selected_model" || section === "auto_search") {
        groups[section].push(file);
      } else {
        groups.legacy_root.push(file);
      }
    }
    return groups;
  }, [modelFiles]);
  const selectedArtifactName = fileBaseName(selectedFilePath);

  const renderArtifactView = () => {
    if (!selectedFilePath || !fileContent) return null;
    if (showRawFile) {
      return <pre className="file-content">{fileContent.content}</pre>;
    }
    const parsedJson = fileContent.content_type === "json" ? parseJsonContent(fileContent.content) : null;
    const parsedCsv = fileContent.content_type === "csv" ? parseCsvContent(fileContent.content) : null;
    const selectedName = fileBaseName(fileContent.relative_path ?? selectedFilePath);

    switch (selectedName) {
      case "metrics.csv":
        return parsedCsv ? <MetricsCsvView parsed={parsedCsv} ciLabel={ciLabel} /> : <div className="empty">No metrics data.</div>;
      case "metrics_summary.json":
        return parsedJson ? <KeyValueGrid data={parsedJson} /> : <div className="empty">No summary data.</div>;
      case "config.executed.json":
      case "best_config.json":
        return parsedJson ? <ConfigJsonView data={parsedJson} /> : <div className="empty">No config data.</div>;
      case "metadata.json":
      case "two_stage_metadata.json":
        return parsedJson ? <MetadataView data={parsedJson} /> : <div className="empty">No metadata.</div>;
      case "feature_manifest.json":
        return parsedJson ? <FeatureManifestView data={parsedJson} /> : <div className="empty">No feature manifest.</div>;
      case "audit_overlap.json":
        return parsedJson ? <AuditOverlapView data={parsedJson} /> : <div className="empty">No overlap data.</div>;
      case "audit_weight_distribution.json":
        return parsedJson ? <AuditWeightView data={parsedJson} /> : <div className="empty">No weight audit.</div>;
      case "audit_split_composition.csv":
        return parsedCsv ? <SplitCompositionView parsed={parsedCsv} /> : <div className="empty">No split composition.</div>;
      case "split_timeline.json":
        return parsedJson ? <SplitTimelineView data={parsedJson} /> : <div className="empty">No timeline data.</div>;
      case "fold_deltas.csv":
        return parsedCsv ? <FoldDeltaView parsed={parsedCsv} /> : <div className="empty">No fold delta data.</div>;
      case "group_delta_distribution.csv":
        return parsedCsv ? <GroupDeltaDistributionView parsed={parsedCsv} /> : <div className="empty">No group delta data.</div>;
      case "reliability_bins.csv":
        return parsedCsv ? <ReliabilityView parsed={parsedCsv} /> : <div className="empty">No reliability bins.</div>;
      case "rolling_summary.csv":
        return parsedCsv ? <RollingSummaryView parsed={parsedCsv} /> : <div className="empty">No rolling summary.</div>;
      case "rolling_windows.csv":
      case "metrics_groups.csv":
      case "leaderboard.csv":
      case "auto_search_leaderboard.csv":
      case "two_stage_metrics.csv":
        return parsedCsv ? <CsvTableView parsed={parsedCsv} limit={50} /> : <div className="empty">No table data.</div>;
      case "two_stage_metrics_summary.json":
        return parsedJson ? <KeyValueGrid data={parsedJson} /> : <div className="empty">No summary data.</div>;
      case "auto_search_summary.json":
        return parsedJson ? <KeyValueGrid data={parsedJson} /> : <div className="empty">No summary data.</div>;
      case "auto_search_no_viable.json":
        return parsedJson ? <KeyValueGrid data={parsedJson} /> : <div className="empty">No no-viable data.</div>;
      case "auto_search_progress.json":
        return parsedJson ? <KeyValueGrid data={parsedJson} /> : <div className="empty">No progress data.</div>;
      case "best_model_report.md":
        return <ReportMarkdownView content={fileContent.content} />;
      default:
        if (parsedJson) return <KeyValueGrid data={parsedJson} />;
        if (parsedCsv) return <CsvTableView parsed={parsedCsv} limit={50} />;
        return <pre className="file-content">{fileContent.content}</pre>;
    }
  };

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(form));
    } catch {
      // ignore
    }
  }, [form]);

  useEffect(() => {
    lastRecommendedRef.current = null;
    setSplitRecommended(false);
    setSplitRecommendationWarning(null);
  }, [form.datasetPath, availableWeeks]);

  useEffect(() => {
    if (!lastRecommendedRef.current) {
      if (splitRecommended) setSplitRecommended(false);
      return;
    }
    const snapshot: RecommendedSplitFields = {
      splitStrategy: form.splitStrategy,
      windowMode: form.windowMode,
      trainWindowWeeks: form.trainWindowWeeks,
      validationFolds: form.validationFolds,
      validationWindowWeeks: form.validationWindowWeeks,
      testWindowWeeks: form.testWindowWeeks,
      embargoDays: form.embargoDays,
      cGridPreset: form.cGridPreset,
      cGridCustom: form.cGridCustom,
      calibrationMethod: form.calibrationMethod,
    };
    const matches = Object.entries(snapshot).every(
      ([key, value]) => value === (lastRecommendedRef.current as RecommendedSplitFields)[key as keyof RecommendedSplitFields],
    );
    if (matches !== splitRecommended) {
      setSplitRecommended(matches);
    }
  }, [
    form.splitStrategy,
    form.windowMode,
    form.trainWindowWeeks,
    form.validationFolds,
    form.validationWindowWeeks,
    form.testWindowWeeks,
    form.embargoDays,
    form.cGridPreset,
    form.cGridCustom,
    form.calibrationMethod,
    splitRecommended,
  ]);

  useEffect(() => {
    saveResult(lastRunResult);
  }, [lastRunResult]);

  useEffect(() => {
    if (jobStatus?.result) {
      setLastRunResult(jobStatus.result);
      setRunError(null);
    }
    if (jobStatus?.status === "failed" && jobStatus.error) {
      setRunError(jobStatus.error);
    }
    if (jobStatus) {
      setRunJobPanel("active_run");
      setWorkspaceTab((prev) => (prev === "run_job" ? prev : prev));
    }
  }, [jobStatus]);

  useEffect(() => {
    if (isRunning) {
      setRunJobPanel("active_run");
    }
  }, [isRunning]);

  useEffect(() => {
    if (runJobPanel !== "active_run") return;
    if (hasLiveJob) return;
    if (!jobStatus) {
      setRunJobPanel("configuration");
    }
  }, [hasLiveJob, jobStatus, runJobPanel]);

  useEffect(() => {
    if (lastRunResult && lastRunCiLevel == null) {
      setLastRunCiLevel(form.ciLevel);
    }
  }, [lastRunResult, lastRunCiLevel, form.ciLevel]);

  const refreshDatasets = useCallback(() => {
    fetchCalibrationDatasets()
      .then((response) => {
        setDatasets(response.datasets);
        setDatasetError(null);
        if (!form.datasetPath && response.datasets.length) {
          setForm((prev) => ({ ...prev, datasetPath: response.datasets[0].path }));
        }
      })
      .catch((error: Error) => {
        setDatasetError(error.message);
      });
  }, [form.datasetPath]);

  const refreshModels = useCallback(() => {
    fetchCalibrationModels()
      .then((response) => {
        setModels(response.models);
        setModelError(null);
      })
      .catch((error: Error) => {
        setModelError(error.message);
      });
  }, []);

  useEffect(() => {
    refreshDatasets();
    refreshModels();
  }, [refreshDatasets, refreshModels]);

  useEffect(() => {
    const outDir = jobStatus?.result?.out_dir ?? null;
    if (jobStatus?.status === "finished" && outDir && outDir !== lastModelRefreshRef.current) {
      lastModelRefreshRef.current = outDir;
      refreshModels();
    }
  }, [jobStatus?.result?.out_dir, jobStatus?.status, refreshModels]);

  useEffect(() => {
    if (workspaceTab === "models") {
      refreshModels();
    }
  }, [workspaceTab, refreshModels]);

  useEffect(() => {
    if (form.runMode === "auto" && form.selectionObjective !== "logloss") {
      setForm((prev) => ({ ...prev, selectionObjective: "logloss" }));
    }
  }, [form.runMode, form.selectionObjective]);

  useEffect(() => {
    setShowRawFile(false);
  }, [selectedFilePath]);

  useEffect(() => {
    if (!form.datasetPath) {
      setAvailableTickers([]);
      setTickersError(null);
      return;
    }
    let cancelled = false;
    setTickersLoading(true);
    fetchDatasetTickers(form.datasetPath)
      .then((response) => {
        if (cancelled) return;
        setAvailableTickers(response.tickers);
        setTickersError(null);

        setForm((prev) => {
          const fallbackUniverse = response.tickers.length ? response.tickers : DEFAULT_TRADING_UNIVERSE;
          const nextTrain = prev.trainTickers.length
            ? prev.trainTickers.filter((t) => fallbackUniverse.includes(t))
            : fallbackUniverse;
          const resolvedTrain = nextTrain.length ? nextTrain : fallbackUniverse;
          const nextFoundation = prev.foundationTickers.length
            ? prev.foundationTickers.filter((t) => resolvedTrain.includes(t))
            : resolvedTrain;
          return {
            ...prev,
            tradingUniverseTickers: resolvedTrain,
            trainTickers: resolvedTrain,
            foundationTickers: nextFoundation.length ? nextFoundation : resolvedTrain,
          };
        });
      })
      .catch((error: Error) => {
        if (cancelled) return;
        setTickersError(error.message);
        setAvailableTickers([]);
      })
      .finally(() => {
        if (!cancelled) {
          setTickersLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [form.datasetPath]);

  useEffect(() => {
    if (!form.datasetPath) {
      setAvailableFeatureColumns([]);
      setFeatureError(null);
      return;
    }
    let cancelled = false;
    setFeaturesLoading(true);
    fetchDatasetFeatures(form.datasetPath)
      .then((response) => {
        if (cancelled) return;
        const columns = response.available_columns ?? [];
        setAvailableFeatureColumns(columns);
        setFeatureError(null);
        const available = new Set(
          FEATURE_OPTIONS.filter((feature) => columns.length === 0 || columns.includes(feature)),
        );
        const availableCategorical = new Set(
          CATEGORICAL_FEATURE_OPTIONS.filter(
            (feature) => columns.length === 0 || columns.includes(feature),
          ),
        );
        setForm((prev) => ({
          ...prev,
          selectedFeatures: normalizeFeatureSelection(prev.selectedFeatures, available),
          selectedCategoricalFeatures: normalizeCategoricalSelection(
            prev.selectedCategoricalFeatures,
            availableCategorical,
          ),
        }));
      })
      .catch((error: Error) => {
        if (cancelled) return;
        setFeatureError(error.message);
        setAvailableFeatureColumns([]);
      })
      .finally(() => {
        if (!cancelled) {
          setFeaturesLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [form.datasetPath]);

  useEffect(() => {
    if (!form.datasetPath) {
      setRegimePreview(null);
      setRegimePreviewError(null);
      return;
    }

    const timeout = window.setTimeout(() => {
      previewCalibrationRegime({
        csv: form.datasetPath,
        tdaysAllowed: String(selectedTimeRegime.tdays),
        asofDowAllowed: selectedTimeRegime.asofDow,
      })
        .then((response) => {
          setRegimePreview(response);
          setRegimePreviewError(null);
        })
        .catch((error: Error) => {
          setRegimePreview(null);
          setRegimePreviewError(error.message);
        });
    }, 300);

    return () => window.clearTimeout(timeout);
  }, [form.datasetPath, selectedTimeRegime.asofDow, selectedTimeRegime.tdays]);

  const handleSelectTicker = useCallback(
    (bucket: "trainTickers" | "foundationTickers", ticker: string) => {
      setForm((prev) => {
        const next = new Set(prev[bucket]);
        if (next.has(ticker)) {
          next.delete(ticker);
        } else {
          next.add(ticker);
        }

        const nextState: CalibrateFormState = {
          ...prev,
          [bucket]: Array.from(next).sort(),
        };

        if (bucket === "trainTickers") {
          const trainingSet = new Set(nextState.trainTickers);
          nextState.foundationTickers = nextState.foundationTickers.filter((value) => trainingSet.has(value));
        }
        nextState.tradingUniverseTickers = [...nextState.trainTickers];

        return nextState;
      });
    },
    [],
  );

  const setRecommendedDefaults = useCallback(() => {
    const base = availableTickers.length ? availableTickers : DEFAULT_TRADING_UNIVERSE;
    const recommendedFeatureSet = new Set<string>(
      (availableFeatureColumns.length === 0
        ? FEATURE_OPTIONS
        : FEATURE_OPTIONS.filter((feature) => availableFeatureColumns.includes(feature))
      ).filter((feature) => {
        const deps = FEATURE_DEPENDENCIES[feature] ?? [];
        return deps.every((dep) => availableFeatureColumns.length === 0 || availableFeatureColumns.includes(dep));
      }),
    );
    setForm((prev) => ({
      ...prev,
      splitStrategy: "walk_forward",
      windowMode: "rolling",
      trainWindowWeeks: "52",
      validationFolds: "4",
      validationWindowWeeks: "8",
      testWindowWeeks: "20",
      embargoDays: "2",
      cGridPreset: "standard",
      cGridCustom: C_GRID_PRESETS.standard,
      calibrationMethod: "none",
      selectionObjective: "logloss",
      timeRegime: "thu_1",
      selectedFeatures: normalizeFeatureSelection(DEFAULT_SELECTED_FEATURES, recommendedFeatureSet),
      selectedCategoricalFeatures: [],
      tradingUniverseTickers: [...base],
      trainTickers: [...base],
      foundationTickers: [...base],
      foundationWeight: "1.25",
      tickerInterceptMode: "non_foundation",
      perTickerInteractions: false,
      minSupportIntercepts: "300",
      minSupportInteractions: "1000",
      baseWeightSource: "dataset_weight",
      groupingKey: selectedDataset?.available_grouping_keys?.[0] ?? "group_id",
      groupEqualization: true,
      renorm: "mean1",
      tradingUniverseUpweight: "1.15",
      tickerBalanceMode: "none",
      bootstrapEnabled: false,
      bootstrapGroup: "contract_id",
      bootstrapDraws: "2000",
      ciLevel: 95,
      splitTimeline: true,
      perFoldDeltaChart: true,
      perGroupDeltaDistribution: false,
      maxAbsLogm: "",
      dropPrnExtremes: false,
      dropPrnBelow: "0.001",
      dropPrnAbove: "0.999",
    }));
  }, [availableFeatureColumns, availableTickers, selectedDataset?.available_grouping_keys]);

  const validateForm = useCallback((): string | null => {
    if (!form.datasetPath) {
      return "Select a training dataset.";
    }

    const sanitizedName = sanitizeModelDirName(form.modelDirName || defaultModelName());
    if (!sanitizedName) {
      return "Model directory name can only contain letters, numbers, '.', '_' and '-'.";
    }

    const trainWindowWeeks = parseOptionalInt(form.trainWindowWeeks) ?? 0;
    const validationFolds = parseOptionalInt(form.validationFolds) ?? 0;
    const validationWindowWeeks = parseOptionalInt(form.validationWindowWeeks) ?? 0;
    const testWindowWeeks = parseOptionalInt(form.testWindowWeeks) ?? 0;
    const embargoDays = parseOptionalInt(form.embargoDays) ?? 0;
    const embargoWeeks = Math.ceil(Math.max(0, embargoDays) / 7);

    if (form.splitStrategy === "walk_forward") {
      if (trainWindowWeeks < 8) return "Train window must be at least 8 weeks.";
      if (validationFolds < 1) return "Validation folds must be at least 1.";
      if (validationWindowWeeks < 1) return "Validation window must be at least 1 week.";
      if (testWindowWeeks < 4) return "Test window must be at least 4 weeks.";
      if (trainWindowWeeks < validationWindowWeeks) {
        return "Train window must be at least as long as validation window.";
      }
      if (embargoDays < 0 || embargoDays > 14) {
        return "Embargo must be between 0 and 14 days.";
      }
    }

    if (availableWeeks != null) {
      if (form.splitStrategy === "walk_forward") {
        const baseTrainWeeks = form.windowMode === "rolling" ? trainWindowWeeks : 1;
        const requiredWeeks = testWindowWeeks + (validationWindowWeeks * validationFolds) + baseTrainWeeks + embargoWeeks;
        if (requiredWeeks > availableWeeks) {
          return `Current split requires at least ${requiredWeeks} weeks, but dataset has about ${availableWeeks} weeks.`;
        }
      } else {
        const requiredWeeks = testWindowWeeks + validationWindowWeeks + 1 + embargoWeeks;
        if (requiredWeeks > availableWeeks) {
          return `Current split requires at least ${requiredWeeks} weeks, but dataset has about ${availableWeeks} weeks.`;
        }
      }
    }

    if (form.trainTickers.length === 0) {
      return "Train tickers cannot be empty.";
    }

    const trainingSet = new Set(form.trainTickers);
    const invalidFoundation = form.foundationTickers.filter((ticker) => !trainingSet.has(ticker));
    if (invalidFoundation.length) {
      return `Foundation tickers must be a subset of training tickers (invalid: ${invalidFoundation.join(", ")}).`;
    }

    if (form.cGridPreset === "custom") {
      const parsed = form.cGridCustom
        .split(",")
        .map((token) => Number(token.trim()))
        .filter((value) => Number.isFinite(value) && value > 0);
      if (!parsed.length) {
        return "Custom C grid must contain positive numeric values.";
      }
    }

    const availableFeatureSet = new Set<string>(
      availableFeatureColumns.length === 0
        ? FEATURE_OPTIONS
        : FEATURE_OPTIONS.filter((feature) => availableFeatureColumns.includes(feature)),
    );
    const availableCategoricalSet = new Set<string>(
      availableFeatureColumns.length === 0
        ? CATEGORICAL_FEATURE_OPTIONS
        : CATEGORICAL_FEATURE_OPTIONS.filter((feature) => availableFeatureColumns.includes(feature)),
    );
    const normalizedFeatures = normalizeFeatureSelection(form.selectedFeatures, availableFeatureSet);
    for (const [feature, deps] of Object.entries(FEATURE_DEPENDENCIES)) {
      if (!normalizedFeatures.includes(feature)) continue;
      const missingDeps = deps.filter((dep) => !normalizedFeatures.includes(dep));
      if (missingDeps.length > 0) {
        return `${feature} requires ${missingDeps.join(", ")}.`;
      }
    }
    for (const group of FEATURE_MUTUAL_EXCLUSIVE_GROUPS) {
      const chosen = group.filter((feature) => normalizedFeatures.includes(feature));
      if (chosen.length > 1) {
        return `Select only one of ${group.join(" or ")}.`;
      }
    }

    const normalizedCategorical = normalizeCategoricalSelection(
      form.selectedCategoricalFeatures,
      availableCategoricalSet,
    );
    const invalidCategorical = form.selectedCategoricalFeatures.filter(
      (feature) => !normalizedCategorical.includes(feature),
    );
    if (invalidCategorical.length) {
      return `Selected categorical features not available: ${invalidCategorical.join(", ")}.`;
    }

    if (parseOptionalFloat(form.foundationWeight) === undefined) {
      return "Foundation weight must be numeric.";
    }

    if (parseOptionalFloat(form.foundationWeight)! < 1 || parseOptionalFloat(form.foundationWeight)! > 5) {
      return "Foundation weight must stay within [1.0, 5.0].";
    }

    if (form.dropPrnExtremes) {
      const lower = parseOptionalFloat(form.dropPrnBelow);
      const upper = parseOptionalFloat(form.dropPrnAbove);
      if (lower === undefined || upper === undefined) {
        return "Drop pRN below/above values must be numeric.";
      }
      if (lower < 0 || upper > 1 || lower >= upper) {
        return "Use valid pRN bounds with 0 <= below < above <= 1.";
      }
    }

    if (form.runMode === "auto" && form.autoMaxTrials.trim()) {
      const maxTrials = parseOptionalInt(form.autoMaxTrials);
      if (!maxTrials || maxTrials < 1) {
        return "Max trials must be a positive integer.";
      }
    }

    return null;
  }, [availableFeatureColumns, availableWeeks, form]);

  useEffect(() => {
    const issue = validateForm();
    setGuardrailWarning(issue);
  }, [validateForm]);

  const buildManualPayload = useCallback(() => {
    const selectedWeightCol =
      form.weightColStrategy === "weight_final" || form.weightColStrategy === "sample_weight_final"
        ? form.weightColStrategy
        : form.weightColStrategy === "uniform"
          ? "uniform"
          : "weight_final";

    const cGrid =
      form.cGridPreset === "custom"
        ? form.cGridCustom.trim()
        : C_GRID_PRESETS[form.cGridPreset];
    const availableFeatureSet = new Set<string>(
      availableFeatureColumns.length === 0
        ? FEATURE_OPTIONS
        : FEATURE_OPTIONS.filter((feature) => availableFeatureColumns.includes(feature)),
    );
    const availableCategoricalSet = new Set<string>(
      availableFeatureColumns.length === 0
        ? CATEGORICAL_FEATURE_OPTIONS
        : CATEGORICAL_FEATURE_OPTIONS.filter((feature) => availableFeatureColumns.includes(feature)),
    );
    const normalizedSelectedFeatures = normalizeFeatureSelection(form.selectedFeatures, availableFeatureSet);
    const normalizedCategorical = normalizeCategoricalSelection(
      form.selectedCategoricalFeatures,
      availableCategoricalSet,
    );
    const features = joinCsv([BASE_FEATURE, ...normalizedSelectedFeatures]);
    const categoricalFeatures = joinCsv(normalizedCategorical);
    const enableXAbsM = normalizedSelectedFeatures.includes("x_abs_m");

    return {
      csv: form.datasetPath,
      outName: sanitizeModelDirName(form.modelDirName || defaultModelName()),
      runMode: "manual" as const,
      targetCol: DEFAULT_TARGET_COL,
      weekCol: DEFAULT_WEEK_COL,
      tickerCol: DEFAULT_TICKER_COL,
      weightCol: selectedWeightCol,
      weightColStrategy: form.weightColStrategy,
      features,
      categoricalFeatures,
      enableXAbsM,
      foundationTickers: joinCsv(form.foundationTickers),
      foundationWeight: parseOptionalFloat(form.foundationWeight),
      tickerIntercepts: form.tickerInterceptMode,
      tickerXInteractions: form.perTickerInteractions,
      tickerMinSupport: parseOptionalInt(form.minSupportIntercepts),
      tickerMinSupportInteractions: parseOptionalInt(form.minSupportInteractions),
      trainTickers: joinCsv(form.trainTickers),
      tdaysAllowed: String(selectedTimeRegime.tdays),
      asofDowAllowed: selectedTimeRegime.asofDow,
      calibrate: form.calibrationMethod,
      cGrid,
      selectionObjective: form.selectionObjective,
      randomState: parseOptionalInt(form.randomSeed),
      randomSeed: parseOptionalInt(form.randomSeed),
      groupReweight: (form.groupEqualization ? "chain_snapshot" : "none") as "chain_snapshot" | "none",
      maxAbsLogm: parseOptionalFloat(form.maxAbsLogm),
      dropPrnExtremes: form.dropPrnExtremes,
      prnBelow: form.dropPrnExtremes ? parseOptionalFloat(form.dropPrnBelow) : undefined,
      prnAbove: form.dropPrnExtremes ? parseOptionalFloat(form.dropPrnAbove) : undefined,
      bootstrapCi: form.bootstrapEnabled,
      bootstrapB: form.bootstrapEnabled ? parseOptionalInt(form.bootstrapDraws) : undefined,
      bootstrapSeed: form.bootstrapEnabled ? parseOptionalInt(form.bootstrapSeed) : undefined,
      bootstrapGroup: form.bootstrapEnabled ? form.bootstrapGroup : undefined,
      split: {
        strategy: form.splitStrategy,
        windowMode: form.windowMode,
        trainWindowWeeks: parseOptionalInt(form.trainWindowWeeks),
        validationFolds: parseOptionalInt(form.validationFolds),
        validationWindowWeeks: parseOptionalInt(form.validationWindowWeeks),
        testWindowWeeks: parseOptionalInt(form.testWindowWeeks),
        embargoDays: parseOptionalInt(form.embargoDays),
      },
      regularization: {
        cGrid,
        calibrationMethod: form.calibrationMethod,
        selectionObjective: form.selectionObjective,
      },
      modelStructure: {
        tradingUniverseTickers: joinCsv(form.trainTickers),
        trainTickers: joinCsv(form.trainTickers),
        foundationTickers: joinCsv(form.foundationTickers),
        foundationWeight: parseOptionalFloat(form.foundationWeight),
        tickerIntercepts: form.tickerInterceptMode,
        tickerXInteractions: form.perTickerInteractions,
        tickerMinSupport: parseOptionalInt(form.minSupportIntercepts),
        tickerMinSupportInteractions: parseOptionalInt(form.minSupportInteractions),
      },
      weighting: {
        baseWeightSource: form.baseWeightSource,
        groupingKey: form.groupingKey || undefined,
        groupEqualization: form.groupEqualization,
        renorm: "mean1" as const,
        tradingUniverseUpweight: parseOptionalFloat(form.tradingUniverseUpweight),
        tickerBalanceMode: form.tickerBalanceMode,
      },
      bootstrap: {
        bootstrapCi: form.bootstrapEnabled,
        bootstrapGroup: form.bootstrapGroup,
        bootstrapB: parseOptionalInt(form.bootstrapDraws),
        bootstrapSeed: parseOptionalInt(form.bootstrapSeed),
        ciLevel: form.ciLevel,
        perSplitReporting: form.perSplitReporting,
        perFoldReporting: form.perFoldReporting,
      },
      diagnostics: {
        splitTimeline: form.splitTimeline,
        perFoldDeltaChart: form.perFoldDeltaChart,
        perGroupDeltaDistribution: form.perGroupDeltaDistribution,
      },
    };
  }, [availableFeatureColumns, form, selectedTimeRegime.asofDow, selectedTimeRegime.tdays]);

  const selectedGroupingKeys = selectedDataset?.available_grouping_keys ?? [];
  const resolveAutoBootstrapGroup = useCallback(
    (requested: BootstrapGroupMode): BootstrapGroupMode => {
      if (!selectedGroupingKeys.length) return requested;
      const available = new Set(selectedGroupingKeys.map((key) => key.toLowerCase()));
      if (requested === "contract_id" && !available.has("contract_id")) {
        return available.has("group_id") ? "group_id" : "auto";
      }
      if (requested === "group_id" && !available.has("group_id")) {
        return "auto";
      }
      return requested;
    },
    [selectedGroupingKeys],
  );

  const buildAutoPayload = useCallback((): AutoModelRunRequest => {
    const baseConfig = buildManualPayload();
    const autoBootstrapGroup =
      form.bootstrapEnabled && baseConfig.bootstrapGroup
        ? resolveAutoBootstrapGroup(baseConfig.bootstrapGroup)
        : baseConfig.bootstrapGroup;
    const normalizedBase = {
      ...baseConfig,
      bootstrapGroup: autoBootstrapGroup,
      bootstrap: baseConfig.bootstrap
        ? {
            ...baseConfig.bootstrap,
            bootstrapGroup: autoBootstrapGroup,
          }
        : baseConfig.bootstrap,
      selectionObjective: "logloss" as const,
      regularization: {
        ...baseConfig.regularization,
        selectionObjective: "logloss" as const,
      },
    };
    const maxTrials = parseOptionalInt(form.autoMaxTrials);
    const outerFolds = parseOptionalInt(form.autoOuterFolds);
    const outerEnabled = (outerFolds ?? 0) > 0;
    const outerTestWeeks = parseOptionalInt(form.autoOuterTestWeeks);
    const outerGapWeeks = parseOptionalInt(form.autoOuterGapWeeks);
    const outerMinImproveFraction = parseOptionalFloat(form.autoOuterMinImproveFraction);
    const outerMaxWorstDelta = parseOptionalFloat(form.autoOuterMaxWorstDelta);
    const runName = sanitizeModelDirName(form.modelDirName || defaultModelName());
    const hardwareConcurrency =
      typeof navigator !== "undefined" && navigator.hardwareConcurrency
        ? navigator.hardwareConcurrency
        : 8;
    const parallelDefault = Math.max(1, Math.min(8, hardwareConcurrency - 1));
    return {
      csv: baseConfig.csv,
      mode: "option_only",
      runName,
      baseConfig: normalizedBase,
      seed: parseOptionalInt(form.randomSeed) ?? 7,
      parallel: parallelDefault,
      search: {
        featureSets: AUTO_FEATURE_SETS,
        cValues: AUTO_C_VALUES,
        calibrationMethods: AUTO_CAL_METHODS,
        tradingUniverseUpweight: AUTO_UPWEIGHTS,
        foundationWeight: AUTO_FOUNDATION_WEIGHTS,
        tickerIntercepts: Array.from(AUTO_TICKER_INTERCEPTS),
        allowRiskyFeatures: form.autoAllowRisky,
        advancedInteractions: form.autoAdvancedSearch,
        maxTrials: maxTrials ?? undefined,
        selectionRule: "one_se",
        epsilon: 0.002,
        outerFolds: outerEnabled ? outerFolds ?? undefined : undefined,
        outerTestWeeks: outerEnabled ? outerTestWeeks ?? undefined : undefined,
        outerGapWeeks: outerEnabled ? outerGapWeeks ?? undefined : undefined,
        outerSelectionMetric: outerEnabled ? form.autoOuterSelectionMetric : undefined,
        outerMinImproveFraction: outerEnabled ? outerMinImproveFraction ?? undefined : undefined,
        outerMaxWorstDelta: outerEnabled ? outerMaxWorstDelta ?? undefined : undefined,
      },
    };
  }, [
    buildManualPayload,
    form.autoAdvancedSearch,
    form.autoAllowRisky,
    form.autoOuterFolds,
    form.autoOuterGapWeeks,
    form.autoOuterMaxWorstDelta,
    form.autoOuterMinImproveFraction,
    form.autoOuterSelectionMetric,
    form.autoOuterTestWeeks,
    form.autoMaxTrials,
    form.modelDirName,
    form.randomSeed,
    form.bootstrapEnabled,
    resolveAutoBootstrapGroup,
  ]);

  const confirmJobReachable = useCallback(
    async (candidateJobId: string, retries = 2, delayMs = 400): Promise<boolean> => {
      for (let attempt = 0; attempt <= retries; attempt += 1) {
        try {
          const status = await getCalibrationJob(candidateJobId);
          setJobStatus(status);
          return true;
        } catch {
          if (attempt >= retries) {
            break;
          }
          await new Promise<void>((resolve) => {
            window.setTimeout(() => resolve(), delayMs);
          });
        }
      }
      return false;
    },
    [setJobStatus],
  );

  const handlePreviewWeighting = useCallback(async () => {
    if (!form.datasetPath) {
      setWeightingPreview(null);
      setWeightingPreviewError("Select a dataset first.");
      return;
    }

    setWeightingPreviewLoading(true);
    setWeightingPreviewError(null);
    try {
      const response = await previewCalibrationWeighting({
        csv: form.datasetPath,
        weightColStrategy: form.weightColStrategy,
        baseWeightSource: form.baseWeightSource,
        groupingKey: form.groupingKey || undefined,
        groupEqualization: form.groupEqualization,
        tradingUniverseTickers: joinCsv(form.trainTickers),
        tradingUniverseUpweight: parseOptionalFloat(form.tradingUniverseUpweight),
        tickerBalanceMode: form.tickerBalanceMode,
        splitStrategy: form.splitStrategy,
        testWindowWeeks: parseOptionalInt(form.testWindowWeeks),
        validationWindowWeeks: parseOptionalInt(form.validationWindowWeeks),
      });
      setWeightingPreview(response);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to preview weighting.";
      setWeightingPreviewError(message);
      setWeightingPreview(null);
    } finally {
      setWeightingPreviewLoading(false);
    }
  }, [form]);

  const handleRunJob = useCallback(async () => {
    if (anyJobRunning) {
      setRunError(`Another job is running (${primaryJob?.name ?? "unknown"}). Wait for it to finish.`);
      return;
    }

    const validationError = validateForm();
    if (validationError) {
      setRunError(validationError);
      return;
    }

    setRunError(null);
    setJobStatus(null);
    setLastRunCiLevel(form.ciLevel);

    try {
      if (isAuto) {
        const payload = buildAutoPayload();
        const status = await startAutoCalibrationJob(payload);
        setJobId(status.job_id);
        setJobStatus(status);
        const confirmed = await confirmJobReachable(status.job_id);
        if (!confirmed) {
          setJobId(null);
          setJobStatus(null);
          setRunJobPanel("configuration");
          setRunError(
            "Job accepted but status not reachable; retry or check backend logs.",
          );
          return;
        }
      } else {
        const payload = buildManualPayload();
        const status = await startCalibrationJob(payload);
        setJobId(status.job_id);
        setJobStatus(status);
      }
      setRunJobPanel("active_run");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Calibration failed.";
      setRunError(message);
    }
  }, [
    anyJobRunning,
    buildAutoPayload,
    buildManualPayload,
    confirmJobReachable,
    isAuto,
    primaryJob?.name,
    setJobId,
    setJobStatus,
    validateForm,
  ]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void handleRunJob();
  };

  const handleApplyRecommendedSplit = useCallback(() => {
    const recommendation = recommendSplitConfig({
      weeks: availableWeeks,
      dteDays: selectedTimeRegime?.tdays ?? 4,
    });
    if (!recommendation) {
      setSplitRecommended(false);
      lastRecommendedRef.current = null;
      setSplitRecommendationWarning("Dataset too short for recommended settings; adjust manually.");
      return;
    }
    const { fields, warning } = recommendation;
    setForm((prev) => ({
      ...prev,
      splitStrategy: fields.splitStrategy,
      windowMode: fields.windowMode,
      trainWindowWeeks: fields.trainWindowWeeks,
      validationFolds: fields.validationFolds,
      validationWindowWeeks: fields.validationWindowWeeks,
      testWindowWeeks: fields.testWindowWeeks,
      embargoDays: fields.embargoDays,
      cGridPreset: fields.cGridPreset,
      cGridCustom: fields.cGridCustom,
      calibrationMethod: fields.calibrationMethod,
    }));
    lastRecommendedRef.current = fields;
    setSplitRecommended(true);
    setSplitRecommendationWarning(warning);
  }, [availableWeeks, selectedTimeRegime?.tdays]);

  const handleNewJob = useCallback(() => {
    if (isRunning) return;
    setRunJobPanel("configuration");
    setRunError(null);
    setCancelError(null);
    setJobStatus(null);
    setJobId(null);
  }, [isRunning, setJobId, setJobStatus]);

  const handleCancelJob = useCallback(async () => {
    const activeJobId = jobStatus?.job_id ?? jobId;
    if (!activeJobId) return;
    setCancelLoading(true);
    setCancelError(null);
    try {
      const updated = await cancelCalibrationJob(activeJobId);
      setJobStatus(updated);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to cancel job.";
      setCancelError(message);
    } finally {
      setCancelLoading(false);
    }
  }, [jobId, jobStatus?.job_id, setJobStatus]);

  const handleSelectModel = useCallback(async (modelId: string) => {
    if (selectedModelId === modelId) {
      setSelectedModelId(null);
      setModelDetail(null);
      setModelFiles(null);
      setSelectedFilePath(null);
      setFileContent(null);
      setModelDetailError(null);
      setFileError(null);
      return;
    }

    setSelectedModelId(modelId);
    setIsModelDetailLoading(true);
    setModelDetailError(null);
    setFileError(null);
    setSelectedFilePath(null);
    setFileContent(null);

    try {
      const [detail, files] = await Promise.all([
        fetchCalibrationModelDetail(modelId),
        fetchModelFiles(modelId),
      ]);
      setModelDetail(detail);
      setModelFiles(files);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to load model detail.";
      setModelDetailError(message);
      setModelDetail(null);
      setModelFiles(null);
      setSelectedFilePath(null);
    } finally {
      setIsModelDetailLoading(false);
    }
  }, [selectedModelId]);

  const handleOpenFile = useCallback(async (file: ModelFileSummary) => {
    if (!selectedModelId) return;
    const targetPath = file.relative_path ?? file.name;
    if (!targetPath) return;
    if (selectedFilePath === targetPath) {
      setSelectedFilePath(null);
      setFileContent(null);
      setFileError(null);
      return;
    }
    setSelectedFilePath(targetPath);
    setFileLoading(true);
    setFileError(null);
    try {
      let content: ModelFileContentResponse;
      if (file.relative_path) {
        content = await fetchModelFileContentByPath(selectedModelId, file.relative_path);
      } else {
        content = await fetchModelFileContent(selectedModelId, file.name);
      }
      setFileContent(content);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to read file.";
      setFileError(message);
      setFileContent(null);
    } finally {
      setFileLoading(false);
    }
  }, [selectedFilePath, selectedModelId]);

  const handleRenameModel = useCallback(async (modelId: string) => {
    const next = window.prompt("New model name", modelId);
    if (!next) return;
    const cleaned = sanitizeModelDirName(next);
    if (!cleaned) {
      setModelError("Invalid model name.");
      return;
    }
    try {
      await renameCalibrationModel(modelId, cleaned);
      refreshModels();
      if (selectedModelId === modelId) {
        setSelectedModelId(cleaned);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Rename failed.";
      setModelError(message);
    }
  }, [refreshModels, selectedModelId]);

  const handleDeleteModel = useCallback(async () => {
    if (!deleteTarget || deleteConfirmText !== "DELETE") return;
    setDeleteLoading(true);
    try {
      await deleteCalibrationModel(deleteTarget.id);
      if (selectedModelId === deleteTarget.id) {
        setSelectedModelId(null);
        setModelDetail(null);
        setModelFiles(null);
        setSelectedFilePath(null);
        setFileContent(null);
      }
      setDeleteTarget(null);
      setDeleteConfirmText("");
      refreshModels();
    } catch (error) {
      const message = error instanceof Error ? error.message : "Delete failed.";
      setModelError(message);
    } finally {
      setDeleteLoading(false);
    }
  }, [deleteConfirmText, deleteTarget, refreshModels, selectedModelId]);

  const compareLeft = useMemo(
    () => models.find((model) => model.id === modelCompare.left) ?? null,
    [models, modelCompare.left],
  );

  const compareRight = useMemo(
    () => models.find((model) => model.id === modelCompare.right) ?? null,
    [models, modelCompare.right],
  );

  const selectableFeatureOptions = useMemo(() => {
    const availableSet = new Set<string>(
      availableFeatureColumns.length === 0
        ? FEATURE_OPTIONS
        : FEATURE_OPTIONS.filter((feature) => availableFeatureColumns.includes(feature)),
    );
    return FEATURE_OPTIONS.filter((feature) => {
      if (!availableSet.has(feature)) return false;
      const deps = FEATURE_DEPENDENCIES[feature] ?? [];
      return deps.every((dep) => availableSet.has(dep));
    });
  }, [availableFeatureColumns]);

  const selectableCategoricalOptions = useMemo(
    () =>
      availableFeatureColumns.length === 0
        ? Array.from(CATEGORICAL_FEATURE_OPTIONS)
        : CATEGORICAL_FEATURE_OPTIONS.filter((feature) => availableFeatureColumns.includes(feature)),
    [availableFeatureColumns],
  );

  const selectableFeatureSet = useMemo(
    () => new Set<string>(selectableFeatureOptions),
    [selectableFeatureOptions],
  );

  const selectableCategoricalSet = useMemo(
    () => new Set<string>(selectableCategoricalOptions),
    [selectableCategoricalOptions],
  );

  const featureCategoryOptions = useMemo(
    () =>
      FEATURE_CATEGORIES.map((category) => ({
        title: category.title,
        items: category.items.filter((feature) => selectableFeatureSet.has(feature)),
      })).filter((category) => category.items.length > 0),
    [selectableFeatureSet],
  );

  const selectedFeatureList = useMemo(
    () => normalizeFeatureSelection(form.selectedFeatures, selectableFeatureSet),
    [form.selectedFeatures, selectableFeatureSet],
  );

  const selectedCategoricalList = useMemo(
    () =>
      normalizeCategoricalSelection(form.selectedCategoricalFeatures, selectableCategoricalSet),
    [form.selectedCategoricalFeatures, selectableCategoricalSet],
  );

  const featureDependencyWarnings = useMemo(() => {
    const warnings: string[] = [];
    Object.entries(FEATURE_DEPENDENCIES).forEach(([feature, deps]) => {
      if (!selectedFeatureList.includes(feature)) return;
      const missingDeps = deps.filter((dep) => !selectedFeatureList.includes(dep));
      if (missingDeps.length) {
        warnings.push(`${feature} requires ${missingDeps.join(", ")}.`);
      }
    });
    FEATURE_MUTUAL_EXCLUSIVE_GROUPS.forEach((group) => {
      const chosen = group.filter((feature) => selectedFeatureList.includes(feature));
      if (chosen.length > 1) {
        warnings.push(`Select only one of ${group.join(" or ")}.`);
      }
    });
    return warnings;
  }, [selectedFeatureList]);

  const handleToggleFeature = useCallback((feature: string) => {
    if (!FEATURE_OPTION_SET.has(feature)) return;
    setForm((prev) => {
      const available = selectableFeatureSet;
      const next = new Set(
        normalizeFeatureSelection(prev.selectedFeatures, available),
      );
      if (next.has(feature)) {
        next.delete(feature);
        dropDependentFeatures(next, feature);
      } else {
        if (!available.has(feature)) {
          return prev;
        }
        next.add(feature);
        const activated = new Set<string>([feature]);
        (FEATURE_DEPENDENCIES[feature] ?? []).forEach((dep) => {
          if (available.has(dep)) {
            next.add(dep);
            activated.add(dep);
          }
        });
        FEATURE_MUTUAL_EXCLUSIVE_GROUPS.forEach((group) => {
          const chosen = group.filter((candidate) => next.has(candidate));
          if (chosen.length <= 1) return;
          const preferred = chosen.find((candidate) => activated.has(candidate)) ?? chosen[0];
          chosen.forEach((candidate) => {
            if (candidate === preferred) return;
            next.delete(candidate);
            dropDependentFeatures(next, candidate);
          });
        });
        activated.forEach((activatedFeature) => {
          (FEATURE_DEPENDENCIES[activatedFeature] ?? []).forEach((dep) => {
            if (available.has(dep)) {
              next.add(dep);
            }
          });
        });
        FEATURE_MUTUAL_EXCLUSIVE_GROUPS.forEach((group) => {
          const chosen = group.filter((candidate) => next.has(candidate));
          if (chosen.length <= 1) return;
          const preferred = chosen.find((candidate) => activated.has(candidate)) ?? chosen[0];
          chosen.forEach((candidate) => {
            if (candidate === preferred) return;
            next.delete(candidate);
            dropDependentFeatures(next, candidate);
          });
        });
        Object.entries(FEATURE_DEPENDENCIES).forEach(([candidate, deps]) => {
          if (!next.has(candidate)) return;
          if (!deps.every((dep) => next.has(dep))) {
            next.delete(candidate);
          }
        });
      }
      return {
        ...prev,
        selectedFeatures: normalizeFeatureSelection(Array.from(next), available),
      };
    });
  }, [selectableFeatureSet]);

  const handleToggleCategoricalFeature = useCallback((feature: string) => {
    if (!CATEGORICAL_FEATURE_OPTION_SET.has(feature)) return;
    setForm((prev) => {
      const available = selectableCategoricalSet;
      const next = new Set(
        normalizeCategoricalSelection(prev.selectedCategoricalFeatures, available),
      );
      if (next.has(feature)) {
        next.delete(feature);
      } else if (available.has(feature)) {
        next.add(feature);
      }
      return {
        ...prev,
        selectedCategoricalFeatures: normalizeCategoricalSelection(Array.from(next), available),
      };
    });
  }, [selectableCategoricalSet]);

  const handleSelectRecommendedFeatures = useCallback(() => {
    setForm((prev) => ({
      ...prev,
      selectedFeatures: normalizeFeatureSelection(
        DEFAULT_SELECTED_FEATURES,
        selectableFeatureSet,
      ),
      selectedCategoricalFeatures: normalizeCategoricalSelection(
        prev.selectedCategoricalFeatures,
        selectableCategoricalSet,
      ),
    }));
  }, [selectableCategoricalSet, selectableFeatureSet]);

  const handleClearOptionalFeatures = useCallback(() => {
    setForm((prev) => ({ ...prev, selectedFeatures: [], selectedCategoricalFeatures: [] }));
  }, []);

  return (
    <section className="page calibrate-page">
      <PipelineStatusCard
        className="page-sticky-meta calibrate-meta"
        activeJobsCount={activeJobs.length}
      />

      <header className="page-header calibrate-page-header">
        <div className="calibrate-title-row">
          <h1 className="page-title calibrate-page-title">Calibrate</h1>
        </div>
      </header>

      <div className="calibrate-workspace">
        <div className="calibrate-workspace-tabs" role="tablist" aria-label="Calibration workspace tabs">
          <button
            type="button"
            role="tab"
            aria-selected={workspaceTab === "run_job"}
            className={`calibrate-workspace-tab ${workspaceTab === "run_job" ? "active" : ""}`}
            onClick={() => setWorkspaceTab("run_job")}
          >
            Run Job
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={workspaceTab === "models"}
            className={`calibrate-workspace-tab ${workspaceTab === "models" ? "active" : ""}`}
            onClick={() => setWorkspaceTab("models")}
          >
            Models
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={workspaceTab === "documentation"}
            className={`calibrate-workspace-tab ${workspaceTab === "documentation" ? "active" : ""}`}
            onClick={() => setWorkspaceTab("documentation")}
          >
            Documentation
          </button>
        </div>

        {workspaceTab === "run_job" ? (
          <div className="calibrate-tab-panel" role="tabpanel">
            {runJobPanel === "configuration" && !isRunning ? (
              <section className="panel calibrate-run-config-panel">
                <div className="panel-header calibrate-panel-header calibrate-job-config-header">
                  <div>
                    <h2 className="calibrate-job-config-title">Run Configuration</h2>
                    <span className="panel-hint">
                      Configure calibration settings, then launch a background job.
                    </span>
                  </div>
                </div>

                <div className="panel-body">
                  <form className="calibrate-form-grid" onSubmit={handleSubmit}>
                    <div className="config-summary">
                      <div>
                        <span className="meta-label">Dataset</span>
                        <span>{selectedDataset?.name ?? "None selected"}</span>
                      </div>
                      <div>
                        <span className="meta-label">Model directory</span>
                        <span>{sanitizeModelDirName(form.modelDirName || defaultModelName())}</span>
                      </div>
                      <div>
                        <span className="meta-label">Run mode</span>
                        <span>{isAuto ? "Auto search" : "Manual calibration"}</span>
                      </div>
                      <div>
                        <span className="meta-label">Split strategy</span>
                        <span>{form.splitStrategy}</span>
                      </div>
                      <div>
                        <span className="meta-label">Selection objective</span>
                        <span>{form.selectionObjective}</span>
                      </div>
                    </div>

                    <section className="section-card calibrate-section-card">
                      <h3 className="section-heading">Run Mode</h3>
                      <div className="run-mode-toggle" role="radiogroup" aria-label="Run mode selection">
                        <button
                          type="button"
                          role="radio"
                          aria-checked={form.runMode === "manual"}
                          className={`run-mode-pill ${form.runMode === "manual" ? "active" : ""}`}
                          onClick={() => setForm((prev) => ({ ...prev, runMode: "manual" }))}
                        >
                          Manual run
                        </button>
                        <button
                          type="button"
                          role="radio"
                          aria-checked={form.runMode === "auto"}
                          className={`run-mode-pill ${form.runMode === "auto" ? "active" : ""}`}
                          onClick={() => setForm((prev) => ({ ...prev, runMode: "auto" }))}
                        >
                          Auto run
                        </button>
                      </div>
                      <span className="field-hint">
                        Auto run searches a curated grid under fixed splits and never tunes on test.
                      </span>
                    </section>

                    <section className="section-card calibrate-section-card">
                      <h3 className="section-heading">Basic Settings</h3>

                      <div className="fields-grid">
                        <div className="field">
                          <label htmlFor="calibrateModelDir">Model directory name</label>
                          <input
                            id="calibrateModelDir"
                            className="input"
                            placeholder={defaultModelName()}
                            value={form.modelDirName}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, modelDirName: event.target.value }))
                            }
                          />
                        </div>
                        <div className="field">
                          <label htmlFor="calibrateDataset">Training dataset</label>
                          <select
                            id="calibrateDataset"
                            className="input"
                            value={form.datasetPath}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, datasetPath: event.target.value }))
                            }
                          >
                            {datasets.length === 0 ? <option value="">No datasets available</option> : null}
                            {datasets.map((dataset) => (
                              <option key={dataset.path} value={dataset.path}>
                                {dataset.name}
                              </option>
                            ))}
                          </select>
                          {datasetError ? <div className="error">{datasetError}</div> : null}
                        </div>
                        <div className="field">
                          <label htmlFor="calibrateRandomSeed">Random seed</label>
                          <input
                            id="calibrateRandomSeed"
                            className="input"
                            inputMode="numeric"
                            value={form.randomSeed}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, randomSeed: event.target.value }))
                            }
                          />
                        </div>
                        <div className="field">
                          <label htmlFor="calibrateWeightCol">Weight column strategy</label>
                          <select
                            id="calibrateWeightCol"
                            className="input"
                            value={form.weightColStrategy}
                            onChange={(event) =>
                              setForm((prev) => ({
                                ...prev,
                                weightColStrategy: event.target.value as WeightColStrategy,
                              }))
                            }
                          >
                            <option value="auto">auto</option>
                            <option value="weight_final">weight_final</option>
                            <option value="sample_weight_final">sample_weight_final</option>
                            <option value="uniform">uniform</option>
                          </select>
                        </div>
                        <div className="field full">
                          <label>Time regime (single day)</label>
                          <div className="run-mode-toggle" role="radiogroup" aria-label="Time regime selection">
                            {TIME_REGIME_OPTIONS.map((option) => (
                              <button
                                key={option.key}
                                type="button"
                                role="radio"
                                aria-checked={form.timeRegime === option.key}
                                className={`run-mode-pill ${form.timeRegime === option.key ? "active" : ""}`}
                                onClick={() => setForm((prev) => ({ ...prev, timeRegime: option.key }))}
                                title={`${option.label}: ${option.helper}`}
                              >
                                {option.label} ({option.helper})
                              </button>
                            ))}
                          </div>
                          <span className="field-hint">
                            Select exactly one day regime. Monday/Tuesday/Wednesday/Thursday map to 4/3/2/1 DTE.
                          </span>
                        </div>
                      </div>

                      <div className="dataset-summary-grid">
                        <div>
                          <span className="meta-label">Dataset ID</span>
                          <span>{selectedDataset?.dataset_id ?? selectedDataset?.name ?? "--"}</span>
                        </div>
                        <div>
                          <span className="meta-label">Date range</span>
                          <span>
                            {selectedDataset?.date_start && selectedDataset?.date_end
                              ? `${selectedDataset.date_start} to ${selectedDataset.date_end}`
                              : "Unavailable"}
                          </span>
                        </div>
                        <div>
                          <span className="meta-label">Tickers</span>
                          <span>{getTickerSummaryText(selectedDataset)}</span>
                        </div>
                        <div>
                          <span className="meta-label">Rows</span>
                          <span>{selectedDataset?.rows?.toLocaleString() ?? "--"}</span>
                        </div>
                        <div>
                          <span className="meta-label">Weeks</span>
                          <span>{availableWeeks != null ? availableWeeks.toLocaleString() : "--"}</span>
                        </div>
                        <div>
                          <span className="meta-label">Last modified</span>
                          <span>{formatTimestamp(selectedDataset?.last_modified)}</span>
                        </div>
                        <div>
                          <span className="meta-label">Time regime</span>
                          <span>{selectedTimeRegime.label} ({selectedTimeRegime.helper})</span>
                        </div>
                      </div>
                      {regimePreviewError ? <div className="error">{regimePreviewError}</div> : null}
                      {regimePreview ? (
                        <div className="dataset-summary-grid">
                          <div>
                            <span className="meta-label">Rows before regime filter</span>
                            <span>{regimePreview.rows_before.toLocaleString()}</span>
                          </div>
                          <div>
                            <span className="meta-label">Rows after regime filter</span>
                            <span>{regimePreview.rows_after.toLocaleString()}</span>
                          </div>
                          <div>
                            <span className="meta-label">Tickers after regime filter</span>
                            <span>{regimePreview.tickers_after.toLocaleString()}</span>
                          </div>
                        </div>
                      ) : null}
                    </section>

                    <section className="section-card calibrate-section-card">
                      <div className="calibrate-section-header-row">
                        <h3 className="section-heading">Regression and Set Settings</h3>
                        <div className="calibrate-inline-actions">
                          {splitRecommended ? <span className="status-pill success">Recommended</span> : null}
                          <button
                            className="button ghost small"
                            type="button"
                            title="Uses dataset length and the selected DTE regime."
                            onClick={handleApplyRecommendedSplit}
                            disabled={!canRecommendSplit}
                          >
                            Apply recommended settings
                          </button>
                        </div>
                      </div>
                      {splitRecommendationWarning ? <div className="warning">{splitRecommendationWarning}</div> : null}
                      <div className="fields-grid">
                        <div className="field">
                          <label htmlFor="splitStrategy">Split strategy</label>
                          <select
                            id="splitStrategy"
                            className="input"
                            value={form.splitStrategy}
                            onChange={(event) =>
                              setForm((prev) => ({
                                ...prev,
                                splitStrategy: event.target.value as SplitStrategy,
                              }))
                            }
                          >
                            <option value="walk_forward">walk_forward</option>
                            <option value="single_holdout">single_holdout</option>
                          </select>
                        </div>
                        <div className="field">
                          <label htmlFor="windowMode">Window mode</label>
                          <select
                            id="windowMode"
                            className="input"
                            value={form.windowMode}
                            onChange={(event) =>
                              setForm((prev) => ({
                                ...prev,
                                windowMode: event.target.value as WindowMode,
                              }))
                            }
                            disabled={form.splitStrategy !== "walk_forward"}
                          >
                            <option value="rolling">rolling</option>
                            <option value="expanding">expanding</option>
                          </select>
                        </div>
                        <div className="field">
                          <label htmlFor="trainWindowWeeks">Train window (weeks)</label>
                          <input
                            id="trainWindowWeeks"
                            className="input"
                            inputMode="numeric"
                            value={form.trainWindowWeeks}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, trainWindowWeeks: event.target.value }))
                            }
                            disabled={form.splitStrategy !== "walk_forward"}
                          />
                        </div>
                        <div className="field">
                          <label htmlFor="validationFolds">Validation folds</label>
                          <input
                            id="validationFolds"
                            className="input"
                            inputMode="numeric"
                            value={form.validationFolds}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, validationFolds: event.target.value }))
                            }
                            disabled={form.splitStrategy !== "walk_forward"}
                          />
                        </div>
                        <div className="field">
                          <label htmlFor="validationWindowWeeks">Validation window (weeks)</label>
                          <input
                            id="validationWindowWeeks"
                            className="input"
                            inputMode="numeric"
                            value={form.validationWindowWeeks}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, validationWindowWeeks: event.target.value }))
                            }
                            disabled={form.splitStrategy !== "walk_forward"}
                          />
                        </div>
                        <div className="field">
                          <label htmlFor="testWindowWeeks">Test window (weeks)</label>
                          <input
                            id="testWindowWeeks"
                            className="input"
                            inputMode="numeric"
                            value={form.testWindowWeeks}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, testWindowWeeks: event.target.value }))
                            }
                          />
                        </div>
                        <div className="field">
                          <label htmlFor="embargoDays">Embargo (days)</label>
                          <input
                            id="embargoDays"
                            className="input"
                            inputMode="numeric"
                            value={form.embargoDays}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, embargoDays: event.target.value }))
                            }
                          />
                          <span className="field-hint">
                            Excludes near-boundary rows between train and validation/test windows.
                          </span>
                        </div>
                        {!isAuto ? (
                          <>
                            <div className="field">
                              <label htmlFor="cGridPreset">C grid preset</label>
                              <select
                                id="cGridPreset"
                                className="input"
                                value={form.cGridPreset}
                                onChange={(event) => {
                                  const preset = event.target.value as CalibrateFormState["cGridPreset"];
                                  setForm((prev) => ({
                                    ...prev,
                                    cGridPreset: preset,
                                    cGridCustom:
                                      preset === "custom" ? prev.cGridCustom : C_GRID_PRESETS[preset],
                                  }));
                                }}
                              >
                                <option value="coarse">coarse</option>
                                <option value="standard">standard</option>
                                <option value="wide">wide</option>
                                <option value="custom">custom</option>
                              </select>
                            </div>
                            <div className="field">
                              <label htmlFor="cGridCustom">C grid</label>
                              <input
                                id="cGridCustom"
                                className="input"
                                value={form.cGridCustom}
                                onChange={(event) =>
                                  setForm((prev) => ({ ...prev, cGridCustom: event.target.value, cGridPreset: "custom" }))
                                }
                              />
                            </div>
                            <div className="field">
                              <label htmlFor="calibrationMethod">Calibration method</label>
                              <select
                                id="calibrationMethod"
                                className="input"
                                value={form.calibrationMethod}
                                onChange={(event) =>
                                  setForm((prev) => ({
                                    ...prev,
                                    calibrationMethod: event.target.value as CalibrationMethod,
                                  }))
                                }
                              >
                                <option value="none">none</option>
                                <option value="platt">platt</option>
                              </select>
                            </div>
                            <div className="field">
                              <label htmlFor="selectionObjective">Selection objective</label>
                              <select
                                id="selectionObjective"
                                className="input"
                                value={form.selectionObjective}
                                onChange={(event) =>
                                  setForm((prev) => ({
                                    ...prev,
                                    selectionObjective: event.target.value as SelectionObjective,
                                  }))
                                }
                              >
                                <option value="logloss">logloss</option>
                                <option value="brier">brier</option>
                                <option value="ece_q">ece_q</option>
                              </select>
                            </div>
                          </>
                        ) : null}
                      </div>
                      {isAuto ? (
                        <span className="field-hint">
                          Auto search varies C and calibration method. Selection objective is fixed to logloss.
                        </span>
                      ) : null}
                    </section>

                    {isAuto ? (
                      <section className="section-card calibrate-section-card">
                        <h3 className="section-heading">Auto Search Settings</h3>
                        <div className="fields-grid">
                          <div className="field">
                            <label htmlFor="autoMaxTrials">Max trials (optional)</label>
                            <input
                              id="autoMaxTrials"
                              className="input"
                              inputMode="numeric"
                              value={form.autoMaxTrials}
                              onChange={(event) =>
                                setForm((prev) => ({ ...prev, autoMaxTrials: event.target.value }))
                              }
                            />
                            <span className="field-hint">Leave blank to run the full curated grid.</span>
                          </div>
                        </div>
                        <div className="calibrate-inline-actions">
                          <label className="checkbox calibrate-checkbox-pill">
                            <input
                              type="checkbox"
                              checked={form.autoAllowRisky}
                              onChange={(event) =>
                                setForm((prev) => ({ ...prev, autoAllowRisky: event.target.checked }))
                              }
                            />
                            Allow risky features
                          </label>
                          <label className="checkbox calibrate-checkbox-pill">
                            <input
                              type="checkbox"
                              checked={form.autoAdvancedSearch}
                              onChange={(event) =>
                                setForm((prev) => ({ ...prev, autoAdvancedSearch: event.target.checked }))
                              }
                            />
                            Advanced search (ticker interactions)
                          </label>
                        </div>
                        <div className="dataset-summary-grid">
                          <div>
                            <span className="meta-label">Feature sets</span>
                            <span>{AUTO_FEATURE_SETS.length} curated sets</span>
                          </div>
                          <div>
                            <span className="meta-label">C values</span>
                            <span>{AUTO_C_VALUES.join(", ")}</span>
                          </div>
                          <div>
                            <span className="meta-label">Calibration</span>
                            <span>{AUTO_CAL_METHODS.join(", ")}</span>
                          </div>
                          <div>
                            <span className="meta-label">Upweights</span>
                            <span>{AUTO_UPWEIGHTS.join(", ")}</span>
                          </div>
                        </div>
                        <span className="field-hint">
                          Auto search uses curated feature sets and varies core hyperparameters. Time regime and split settings are
                          locked; selection is based on validation logloss unless outer backtests are enabled.
                        </span>
                      </section>
                    ) : null}

                    {isAuto ? (
                      <section className="section-card calibrate-section-card">
                        <h3 className="section-heading">Advanced Auto-Search</h3>
                        <div className="fields-grid">
                          <div className="field">
                            <label htmlFor="autoOuterFolds">Outer folds</label>
                            <input
                              id="autoOuterFolds"
                              className="input"
                              inputMode="numeric"
                              value={form.autoOuterFolds}
                              onChange={(event) =>
                                setForm((prev) => ({ ...prev, autoOuterFolds: event.target.value }))
                              }
                            />
                            <span className="field-hint">Set to 0 to disable nested backtest selection.</span>
                          </div>
                          <div className="field">
                            <label htmlFor="autoOuterTestWeeks">Outer test weeks</label>
                            <input
                              id="autoOuterTestWeeks"
                              className="input"
                              inputMode="numeric"
                              value={form.autoOuterTestWeeks}
                              onChange={(event) =>
                                setForm((prev) => ({ ...prev, autoOuterTestWeeks: event.target.value }))
                              }
                            />
                          </div>
                          <div className="field">
                            <label htmlFor="autoOuterGapWeeks">Outer gap weeks</label>
                            <input
                              id="autoOuterGapWeeks"
                              className="input"
                              inputMode="numeric"
                              value={form.autoOuterGapWeeks}
                              onChange={(event) =>
                                setForm((prev) => ({ ...prev, autoOuterGapWeeks: event.target.value }))
                              }
                            />
                          </div>
                          <div className="field">
                            <label htmlFor="autoOuterSelectionMetric">Outer selection metric</label>
                            <select
                              id="autoOuterSelectionMetric"
                              className="input"
                              value={form.autoOuterSelectionMetric}
                              onChange={(event) =>
                                setForm((prev) => ({
                                  ...prev,
                                  autoOuterSelectionMetric: event.target.value as AutoOuterSelectionMetric,
                                }))
                              }
                            >
                              <option value="median_delta_logloss">median delta logloss</option>
                              <option value="worst_delta_logloss">worst-fold delta logloss</option>
                              <option value="mean_delta_logloss">mean delta logloss</option>
                            </select>
                          </div>
                          <div className="field">
                            <label htmlFor="autoOuterMinImproveFraction">Min improve fraction</label>
                            <input
                              id="autoOuterMinImproveFraction"
                              className="input"
                              inputMode="decimal"
                              value={form.autoOuterMinImproveFraction}
                              onChange={(event) =>
                                setForm((prev) => ({
                                  ...prev,
                                  autoOuterMinImproveFraction: event.target.value,
                                }))
                              }
                            />
                          </div>
                          <div className="field">
                            <label htmlFor="autoOuterMaxWorstDelta">Max worst-fold delta</label>
                            <input
                              id="autoOuterMaxWorstDelta"
                              className="input"
                              inputMode="decimal"
                              value={form.autoOuterMaxWorstDelta}
                              onChange={(event) =>
                                setForm((prev) => ({
                                  ...prev,
                                  autoOuterMaxWorstDelta: event.target.value,
                                }))
                              }
                            />
                          </div>
                        </div>
                        <span className="field-hint">
                          Outer backtests run an extra time-series loop inside training. Expect higher runtime but
                          more stable generalization.
                        </span>
                      </section>
                    ) : null}

                    {!isAuto ? (
                      <section className="section-card calibrate-section-card">
                        <div className="calibrate-section-header-row">
                          <h3 className="section-heading">Feature Selection</h3>
                          <div className="calibrate-inline-actions">
                            <button
                              className="button ghost small"
                              type="button"
                              onClick={handleSelectRecommendedFeatures}
                            >
                              Recommended
                            </button>
                            <button
                              className="button ghost small"
                              type="button"
                              onClick={handleClearOptionalFeatures}
                            >
                              Clear optional
                            </button>
                          </div>
                        </div>
                        <p className="calibrate-mode-note">
                          Base feature <code>{BASE_FEATURE}</code> is always included. Dependencies and exclusivity
                          rules are enforced automatically.
                        </p>
                        {featuresLoading ? <div className="empty">Loading feature options…</div> : null}
                        {featureError ? <div className="error">{featureError}</div> : null}
                        <div className="feature-category-grid">
                          {featureCategoryOptions.map((category) => (
                            <div key={category.title} className="feature-category-card">
                              <h4 className="feature-category-title">{category.title}</h4>
                              <div className="feature-chip-grid">
                                {category.items.map((feature) => {
                                  const selected = selectedFeatureList.includes(feature);
                                  const deps = FEATURE_DEPENDENCIES[feature] ?? [];
                                  const label = deps.length
                                    ? `${feature} (requires ${deps.join(", ")})`
                                    : feature;
                                  return (
                                    <button
                                      key={feature}
                                      type="button"
                                      className={`feature-chip ${selected ? "selected" : ""}`}
                                      onClick={() => handleToggleFeature(feature)}
                                      title={label}
                                    >
                                      {feature}
                                    </button>
                                  );
                                })}
                              </div>
                            </div>
                          ))}
                          {selectableCategoricalOptions.length ? (
                            <div className="feature-category-card">
                              <h4 className="feature-category-title">Categorical</h4>
                              <div className="feature-chip-grid">
                                {selectableCategoricalOptions.map((feature) => {
                                  const selected = selectedCategoricalList.includes(feature);
                                  const label = CATEGORICAL_FEATURE_LABELS[feature] ?? feature;
                                  return (
                                    <button
                                      key={feature}
                                      type="button"
                                      className={`feature-chip ${selected ? "selected" : ""}`}
                                      onClick={() => handleToggleCategoricalFeature(feature)}
                                      title={feature}
                                    >
                                      {label}
                                    </button>
                                  );
                                })}
                              </div>
                            </div>
                          ) : null}
                        </div>
                        <div className="dataset-summary-grid">
                          <div>
                            <span className="meta-label">Base feature</span>
                            <span>{BASE_FEATURE}</span>
                          </div>
                          <div>
                            <span className="meta-label">Selected optional</span>
                            <span>{selectedFeatureList.length}</span>
                          </div>
                          <div>
                            <span className="meta-label">Selected categorical</span>
                            <span>{selectedCategoricalList.length}</span>
                          </div>
                          <div>
                            <span className="meta-label">Final feature count</span>
                            <span>{selectedFeatureList.length + selectedCategoricalList.length + 1}</span>
                          </div>
                          <div>
                            <span className="meta-label">Enable x_abs_m flag</span>
                            <span>{selectedFeatureList.includes("x_abs_m") ? "yes" : "no"}</span>
                          </div>
                        </div>
                        {featureDependencyWarnings.length ? (
                          <div className="warning">{featureDependencyWarnings.join(" ")}</div>
                        ) : null}
                      </section>
                    ) : null}

                    <section className="section-card calibrate-section-card">
                      <div className="calibrate-section-header-row">
                        <h3 className="section-heading">Model Structure</h3>
                        {!isAuto ? (
                          <button className="button ghost" type="button" onClick={setRecommendedDefaults}>
                            Recommended defaults
                          </button>
                        ) : null}
                      </div>
                      <div className="fields-grid">
                        {!isAuto ? (
                          <>
                            <div className="field">
                              <label htmlFor="foundationWeight">Foundation weight</label>
                              <input
                                id="foundationWeight"
                                className="input"
                                inputMode="decimal"
                                value={form.foundationWeight}
                                onChange={(event) =>
                                  setForm((prev) => ({ ...prev, foundationWeight: event.target.value }))
                                }
                              />
                              <span className="field-hint">Safe range: 1.0 to 3.0 (hard cap 5.0).</span>
                            </div>
                            <div className="field">
                              <label htmlFor="tickerInterceptMode">Ticker intercept mode</label>
                              <select
                                id="tickerInterceptMode"
                                className="input"
                                value={form.tickerInterceptMode}
                                onChange={(event) =>
                                  setForm((prev) => ({
                                    ...prev,
                                    tickerInterceptMode: event.target.value as TickerInterceptMode,
                                  }))
                                }
                              >
                                <option value="none">none</option>
                                <option value="all">all</option>
                                <option value="non_foundation">non_foundation</option>
                              </select>
                            </div>
                          </>
                        ) : null}
                        <div className="field">
                          <label htmlFor="minSupportIntercepts">Min support (intercepts)</label>
                          <input
                            id="minSupportIntercepts"
                            className="input"
                            inputMode="numeric"
                            value={form.minSupportIntercepts}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, minSupportIntercepts: event.target.value }))
                            }
                          />
                        </div>
                        <div className="field">
                          <label htmlFor="minSupportInteractions">Min support (interactions)</label>
                          <input
                            id="minSupportInteractions"
                            className="input"
                            inputMode="numeric"
                            value={form.minSupportInteractions}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, minSupportInteractions: event.target.value }))
                            }
                            disabled={!form.perTickerInteractions && !isAuto}
                          />
                        </div>
                      </div>

                      {!isAuto ? (
                        <label className="checkbox calibrate-checkbox-pill">
                          <input
                            type="checkbox"
                            checked={form.perTickerInteractions}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, perTickerInteractions: event.target.checked }))
                            }
                          />
                          Enable per-ticker interactions (advanced)
                        </label>
                      ) : null}
                      {isAuto ? (
                        <span className="field-hint">
                          Foundation weight, ticker intercepts, and interactions are selected by auto search.
                        </span>
                      ) : null}

                      <p className="calibrate-mode-note">
                        Training tickers define which symbols are fit by the model. Foundation tickers are a subset of
                        training tickers that receive extra influence via <code>foundation weight</code>.
                      </p>

                      <div className="ticker-selection-grid">
                        <div className="ticker-selection-card">
                          <div className="ticker-selection-header">
                            <span className="meta-label">Training tickers</span>
                            <span>{form.trainTickers.length} selected</span>
                          </div>
                          {tickersLoading ? <div className="empty">Loading tickers…</div> : null}
                          {tickersError ? <div className="error">{tickersError}</div> : null}
                          <div className="ticker-chip-grid">
                            {(availableTickers.length ? availableTickers : DEFAULT_TRADING_UNIVERSE).map((ticker) => {
                              const selected = form.trainTickers.includes(ticker);
                              return (
                                <button
                                  key={`train-${ticker}`}
                                  type="button"
                                  className={`ticker-chip ${selected ? "selected" : ""}`}
                                  onClick={() => handleSelectTicker("trainTickers", ticker)}
                                >
                                  {ticker}
                                </button>
                              );
                            })}
                          </div>
                        </div>

                        <div className="ticker-selection-card">
                          <div className="ticker-selection-header">
                            <span className="meta-label">Foundation tickers</span>
                            <span>{form.foundationTickers.length} selected</span>
                          </div>
                          <div className="ticker-chip-grid">
                            {form.trainTickers.map((ticker) => {
                              const selected = form.foundationTickers.includes(ticker);
                              return (
                                <button
                                  key={`foundation-${ticker}`}
                                  type="button"
                                  className={`ticker-chip ${selected ? "selected" : ""}`}
                                  onClick={() => handleSelectTicker("foundationTickers", ticker)}
                                >
                                  {ticker}
                                </button>
                              );
                            })}
                          </div>
                          <span className="field-hint">
                            Foundation tickers must be a subset of training tickers.
                          </span>
                        </div>
                      </div>
                    </section>

                    <section className="section-card calibrate-section-card">
                      <h3 className="section-heading">Weights and Groups</h3>
                      <div className="fields-grid">
                        <div className="field">
                          <label htmlFor="baseWeightSource">Base weight source</label>
                          <select
                            id="baseWeightSource"
                            className="input"
                            value={form.baseWeightSource}
                            onChange={(event) =>
                              setForm((prev) => ({
                                ...prev,
                                baseWeightSource: event.target.value as BaseWeightSource,
                              }))
                            }
                          >
                            <option value="dataset_weight">dataset_weight</option>
                            <option value="uniform">uniform</option>
                          </select>
                        </div>
                        <div className="field">
                          <label htmlFor="groupingKey">Grouping key</label>
                          <select
                            id="groupingKey"
                            className="input"
                            value={form.groupingKey}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, groupingKey: event.target.value }))
                            }
                          >
                            {selectedGroupingKeys.length === 0 ? (
                              <option value="group_id">group_id</option>
                            ) : (
                              selectedGroupingKeys.map((key) => (
                                <option key={key} value={key}>{key}</option>
                              ))
                            )}
                          </select>
                        </div>
                        <div className="field">
                          <label htmlFor="renorm">Renorm</label>
                          <select
                            id="renorm"
                            className="input"
                            value={form.renorm}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, renorm: event.target.value as "mean1" }))
                            }
                          >
                            <option value="mean1">mean1</option>
                          </select>
                        </div>
                        {!isAuto ? (
                          <div className="field">
                            <label htmlFor="tradingUniverseUpweight">Trading-universe upweight</label>
                            <input
                              id="tradingUniverseUpweight"
                              className="input"
                              inputMode="decimal"
                              value={form.tradingUniverseUpweight}
                              onChange={(event) =>
                                setForm((prev) => ({ ...prev, tradingUniverseUpweight: event.target.value }))
                              }
                            />
                          </div>
                        ) : null}
                        <div className="field">
                          <label htmlFor="tickerBalanceMode">Ticker balancing</label>
                          <select
                            id="tickerBalanceMode"
                            className="input"
                            value={form.tickerBalanceMode}
                            onChange={(event) =>
                              setForm((prev) => ({
                                ...prev,
                                tickerBalanceMode: event.target.value as TickerBalanceMode,
                              }))
                            }
                          >
                            <option value="none">none</option>
                            <option value="sqrt_inv_clipped">sqrt_inv_clipped</option>
                          </select>
                        </div>
                      </div>

                      <div className="calibrate-inline-actions calibrate-weight-actions-row">
                        <label className="checkbox calibrate-checkbox-pill">
                          <input
                            type="checkbox"
                            checked={form.groupEqualization}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, groupEqualization: event.target.checked }))
                            }
                          />
                          Per-group equalization
                        </label>
                        <button
                          className="button light"
                          type="button"
                          onClick={() => void handlePreviewWeighting()}
                          disabled={weightingPreviewLoading}
                        >
                          {weightingPreviewLoading ? "Previewing…" : "Preview weights"}
                        </button>
                      </div>

                      {weightingPreviewError ? <div className="error">{weightingPreviewError}</div> : null}
                      {weightingPreview ? (
                        <div className="weight-preview-panel">
                          <div className="weight-preview-grid">
                            <div>
                              <span className="meta-label">Selected weight</span>
                              <span>{weightingPreview.selected_weight_column ?? "uniform"}</span>
                            </div>
                            <div>
                              <span className="meta-label">Min / mean / max</span>
                              <span>
                                {weightingPreview.min_weight.toFixed(4)} / {weightingPreview.mean_weight.toFixed(4)} / {weightingPreview.max_weight.toFixed(4)}
                              </span>
                            </div>
                            <div>
                              <span className="meta-label">Group sum (min / mean / max)</span>
                              <span>
                                {weightingPreview.group_sum_min?.toFixed(4) ?? "--"} / {weightingPreview.group_sum_mean?.toFixed(4) ?? "--"} / {weightingPreview.group_sum_max?.toFixed(4) ?? "--"}
                              </span>
                            </div>
                            <div>
                              <span className="meta-label">Groups by split</span>
                              <span>
                                train={weightingPreview.split_group_counts.train ?? 0}, val={weightingPreview.split_group_counts.val ?? 0}, test={weightingPreview.split_group_counts.test ?? 0}
                              </span>
                            </div>
                            <div>
                              <span className="meta-label">Rows by split</span>
                              <span>
                                train={weightingPreview.split_row_counts.train ?? 0}, val={weightingPreview.split_row_counts.val ?? 0}, test={weightingPreview.split_row_counts.test ?? 0}
                              </span>
                            </div>
                          </div>
                          {weightingPreview.warnings.length ? (
                            <div className="warning">
                              {weightingPreview.warnings.join(" ")}
                            </div>
                          ) : null}
                        </div>
                      ) : null}
                    </section>

                    <section className="section-card calibrate-section-card">
                      <h3 className="section-heading">Bootstrap and Confidence</h3>
                      <label className="checkbox calibrate-checkbox-pill">
                        <input
                          type="checkbox"
                          checked={form.bootstrapEnabled}
                          onChange={(event) =>
                            setForm((prev) => ({ ...prev, bootstrapEnabled: event.target.checked }))
                          }
                        />
                        Enable bootstrap confidence intervals
                      </label>

                      <div className="fields-grid">
                        <div className="field">
                          <label htmlFor="bootstrapGroup">Bootstrap group key</label>
                          <select
                            id="bootstrapGroup"
                            className="input"
                            value={form.bootstrapGroup}
                            onChange={(event) =>
                              setForm((prev) => ({
                                ...prev,
                                bootstrapGroup: event.target.value as BootstrapGroupMode,
                              }))
                            }
                            disabled={!form.bootstrapEnabled}
                          >
                            <option value="contract_id">contract_id</option>
                            <option value="group_id">group_id</option>
                            <option value="ticker_day">ticker_day</option>
                            <option value="day">day</option>
                            <option value="iid">iid</option>
                            <option value="auto">auto</option>
                          </select>
                        </div>
                        <div className="field">
                          <label htmlFor="bootstrapDraws">Bootstrap draws (B)</label>
                          <input
                            id="bootstrapDraws"
                            className="input"
                            inputMode="numeric"
                            value={form.bootstrapDraws}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, bootstrapDraws: event.target.value }))
                            }
                            disabled={!form.bootstrapEnabled}
                          />
                        </div>
                        <div className="field">
                          <label htmlFor="bootstrapSeed">Bootstrap seed</label>
                          <input
                            id="bootstrapSeed"
                            className="input"
                            inputMode="numeric"
                            value={form.bootstrapSeed}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, bootstrapSeed: event.target.value }))
                            }
                            disabled={!form.bootstrapEnabled}
                          />
                        </div>
                        <div className="field">
                          <label htmlFor="ciLevel">Confidence level</label>
                          <select
                            id="ciLevel"
                            className="input"
                            value={form.ciLevel}
                            onChange={(event) =>
                              setForm((prev) => ({
                                ...prev,
                                ciLevel: Number(event.target.value) as 90 | 95 | 99,
                              }))
                            }
                          >
                            <option value={90}>90%</option>
                            <option value={95}>95%</option>
                            <option value={99}>99%</option>
                          </select>
                        </div>
                      </div>

                      <div className="toggle-grid">
                        <label className="checkbox calibrate-checkbox-pill">
                          <input
                            type="checkbox"
                            checked={form.perSplitReporting}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, perSplitReporting: event.target.checked }))
                            }
                          />
                          Per-split reporting
                        </label>
                        <label className="checkbox calibrate-checkbox-pill">
                          <input
                            type="checkbox"
                            checked={form.perFoldReporting}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, perFoldReporting: event.target.checked }))
                            }
                          />
                          Per-fold reporting
                        </label>
                        <label className="checkbox calibrate-checkbox-pill">
                          <input
                            type="checkbox"
                            checked={form.splitTimeline}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, splitTimeline: event.target.checked }))
                            }
                          />
                          Split timeline viewer
                        </label>
                        <label className="checkbox calibrate-checkbox-pill">
                          <input
                            type="checkbox"
                            checked={form.perFoldDeltaChart}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, perFoldDeltaChart: event.target.checked }))
                            }
                          />
                          Per-fold delta chart
                        </label>
                        <label className="checkbox calibrate-checkbox-pill">
                          <input
                            type="checkbox"
                            checked={form.perGroupDeltaDistribution}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, perGroupDeltaDistribution: event.target.checked }))
                            }
                          />
                          Per-group delta distribution
                        </label>
                      </div>

                      {weightingPreview && (weightingPreview.split_group_counts.val ?? 0) < 30 ? (
                        <div className="warning">
                          Estimated validation groups are low for reliable confidence intervals.
                        </div>
                      ) : null}
                    </section>

                    <section className="section-card calibrate-section-card">
                      <h3 className="section-heading">Additional filters</h3>
                      <div className="fields-grid">
                        <div className="field">
                          <label htmlFor="maxAbsLogm">Maximum absolute log moneyness</label>
                          <input
                            id="maxAbsLogm"
                            className="input"
                            value={form.maxAbsLogm}
                            onChange={(event) =>
                              setForm((prev) => ({ ...prev, maxAbsLogm: event.target.value }))
                            }
                            placeholder="0.4"
                          />
                        </div>
                        <div className="field">
                          <label>pRN bounds filter</label>
                          <label className="checkbox calibrate-checkbox-pill calibrate-inline-filter-toggle">
                            <input
                              type="checkbox"
                              checked={form.dropPrnExtremes}
                              onChange={(event) =>
                                setForm((prev) => ({ ...prev, dropPrnExtremes: event.target.checked }))
                              }
                            />
                            Drop pRN extremes
                          </label>
                        </div>
                      </div>
                      {form.dropPrnExtremes ? (
                        <div className="inline-fields calibrate-prn-bounds-row">
                          <div className="field">
                            <label htmlFor="dropPrnBelow">Drop pRN below</label>
                            <input
                              id="dropPrnBelow"
                              className="input"
                              value={form.dropPrnBelow}
                              onChange={(event) =>
                                setForm((prev) => ({ ...prev, dropPrnBelow: event.target.value }))
                              }
                            />
                          </div>
                          <div className="field">
                            <label htmlFor="dropPrnAbove">Drop pRN above</label>
                            <input
                              id="dropPrnAbove"
                              className="input"
                              value={form.dropPrnAbove}
                              onChange={(event) =>
                                setForm((prev) => ({ ...prev, dropPrnAbove: event.target.value }))
                              }
                            />
                          </div>
                        </div>
                      ) : null}
                    </section>

                    {runError ? <div className="error">{runError}</div> : null}
                    {guardrailWarning ? <div className="warning">{guardrailWarning}</div> : null}

                    <div className="panel-actions calibrate-run-actions">
                      <button
                        className="button primary large calibrate-run-job-button"
                        type="submit"
                        disabled={isRunning || anyJobRunning}
                      >
                        {isRunning ? "Running job..." : isAuto ? "Run auto search" : "Run job"}
                      </button>
                    </div>
                  </form>
                </div>
              </section>
            ) : (
              <section className="panel calibrate-active-run-panel">
                <div className="panel-header calibrate-panel-header calibrate-job-config-header">
                  <div>
                    <h2 className="calibrate-job-config-title">Active Run</h2>
                    <span className="panel-hint">Monitor current calibration jobs and inspect artifacts.</span>
                  </div>
                  <div className="calibrate-header-actions">
                    {isRunning ? (
                      <button
                        className="button ghost calibrate-stop-button"
                        type="button"
                        onClick={handleCancelJob}
                        disabled={cancelLoading}
                      >
                        {cancelLoading ? "Stopping..." : "Stop run"}
                      </button>
                    ) : null}
                    {!isRunning ? (
                      <button
                        className="button light calibrate-new-job-button"
                        type="button"
                        onClick={handleNewJob}
                      >
                        New job
                      </button>
                    ) : null}
                  </div>
                </div>

                <div className="panel-body">
                  {cancelError ? <div className="error">{cancelError}</div> : null}
                  {jobStatus ? (
                    <div className="calibrate-active-shell">
                      <aside className="calibrate-active-sidebar">
                        <div className="run-summary calibrate-run-summary-card">
                          <div className="run-summary-header">
                            <div>
                              <span className="meta-label">Run monitor</span>
                              <div className="run-title">Calibration job</div>
                            </div>
                            <span className={`status-pill ${statusClass(jobStatus.status)}`}>
                              {jobStatus.status}
                            </span>
                          </div>
                          <div className="run-meta-grid">
                            <div>
                              <span className="meta-label">Job ID</span>
                              <span>{jobStatus.job_id}</span>
                            </div>
                            <div>
                              <span className="meta-label">Mode</span>
                              <span>{jobStatus.mode}</span>
                            </div>
                            <div>
                              <span className="meta-label">Started</span>
                              <span>{formatTimestamp(jobStatus.started_at)}</span>
                            </div>
                            <div>
                              <span className="meta-label">Finished</span>
                              {isRunning ? (
                                <div className="calibrate-finished-progress">
                                  <PipelineProgressBar
                                    title="Progress"
                                    progress={runProgress}
                                    running={isRunning}
                                    runningLabel="Running..."
                                    idleLabel="--"
                                    unitLabel={jobStatus.mode === "auto" ? "trials" : "steps"}
                                    forceError={jobStatus.status === "failed"}
                                  />
                                </div>
                              ) : (
                                <span>{formatTimestamp(jobStatus.finished_at)}</span>
                              )}
                            </div>
                          </div>
                          {activeResult?.warnings?.length ? (
                            <div className="warning">{activeResult.warnings.join(" ")}</div>
                          ) : null}
                        </div>
                      </aside>

                      <div className="calibrate-active-main">
                        <div className="run-output">
                          <div className="log-tabs calibrate-log-tabs">
                            <button
                              className={`log-tab ${activeLog === "stdout" ? "active" : ""}`}
                              type="button"
                              onClick={() => setActiveLog("stdout")}
                            >
                              stdout
                            </button>
                            <button
                              className={`log-tab ${activeLog === "stderr" ? "active" : ""}`}
                              type="button"
                              onClick={() => setActiveLog("stderr")}
                            >
                              stderr
                            </button>
                          </div>
                          <div className="log-block">
                            <pre className="log-content">
                              {jobStatus?.mode === "auto" && hasLiveJob
                                ? activeLog === "stdout"
                                  ? autoProgressLog || "Auto progress not available yet."
                                  : autoProgress?.last_error || jobStatus.error || "No stderr captured yet."
                                : activeLog === "stdout"
                                  ? activeResult?.stdout || "No stdout captured yet."
                                  : activeResult?.stderr || jobStatus?.error || "No stderr captured yet."}
                            </pre>
                          </div>
                        </div>

                        {jobStatus?.mode === "auto" && autoProgress?.top_candidates?.length ? (
                          <div className="section-card calibrate-section-card">
                            <h3 className="section-heading">Top candidates</h3>
                            <div className="table-container artifact-table">
                              <table className="preview-table">
                                <thead>
                                  <tr>
                                    <th>Rank</th>
                                    <th>Score</th>
                                    <th>Features</th>
                                    <th>C</th>
                                    <th>Calibrate</th>
                                    <th>Upweight</th>
                                    <th>Foundation</th>
                                    <th>Intercepts</th>
                                    <th>Interactions</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {autoProgress.top_candidates.map((row, idx) => (
                                    <tr key={idx}>
                                      <td>{row.rank ?? idx + 1}</td>
                                      <td>{row.score != null ? Number(row.score).toFixed(5) : "--"}</td>
                                      <td>{row.features ?? "--"}</td>
                                      <td>{row.C ?? row.c ?? "--"}</td>
                                      <td>{row.calibration ?? row.calibrate ?? "--"}</td>
                                      <td>
                                        {row.trading_universe_upweight != null
                                          ? Number(row.trading_universe_upweight).toFixed(3)
                                          : "--"}
                                      </td>
                                      <td>
                                        {row.foundation_weight != null
                                          ? Number(row.foundation_weight).toFixed(3)
                                          : "--"}
                                      </td>
                                      <td>{row.ticker_intercepts ?? "--"}</td>
                                      <td>{row.ticker_interactions ? "on" : "off"}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        ) : null}

                        {isJobComplete && metricsSummary ? (
                          <div className="metrics-summary">
                            <div className="metrics-summary-header">
                              <span className="meta-label">Metrics summary</span>
                              <span className="metrics-summary-note">
                                Delta values are model minus baseline.
                              </span>
                            </div>
                            {hasUsageCounts ? (
                              <div className="run-meta-grid">
                                  <div>
                                    <span className="meta-label">Train rows</span>
                                    <span>{formatCountValue(trainRows)}</span>
                                  </div>
                                  <div>
                                    <span className="meta-label">Val rows</span>
                                    <span>{formatCountValue(valRows)}</span>
                                  </div>
                                  <div>
                                    <span className="meta-label">Test rows</span>
                                    <span>{formatCountValue(testRows)}</span>
                                  </div>
                                  <div>
                                    <span className="meta-label">Train groups</span>
                                    <span>{formatCountValue(trainGroups)}</span>
                                  </div>
                                  <div>
                                    <span className="meta-label">Val groups</span>
                                    <span>{formatCountValue(valGroups)}</span>
                                  </div>
                                  <div>
                                    <span className="meta-label">Test groups</span>
                                    <span>{formatCountValue(testGroups)}</span>
                                  </div>
                                </div>
                              ) : null}
                            <div className="metrics-summary-grid">
                              {metricsOrder
                                .map((split) => metricsSummary[split])
                                .filter(Boolean)
                                .map((metric) => (
                                  <div key={metric!.split} className="metrics-card">
                                    <div className="metrics-card-heading">
                                      <strong>{metric!.split}</strong>
                                      <span className={`status-pill ${metric!.status === "good" ? "success" : "failed"}`}>
                                        {metric!.status}
                                      </span>
                                    </div>
                                    <div className="metrics-card-row">
                                      <span>Baseline logloss</span>
                                      <strong>{formatMetricValue(metric!.baseline_logloss)}</strong>
                                    </div>
                                    <div className="metrics-card-row">
                                      <span>Model logloss</span>
                                      <strong>{formatMetricValue(metric!.model_logloss)}</strong>
                                    </div>
                                    <div className="metrics-card-row">
                                      <span>Delta logloss</span>
                                      <strong className={deltaMetricClass(metric!.delta_model_minus_baseline)}>
                                        {formatMetricValue(metric!.delta_model_minus_baseline)}
                                      </strong>
                                    </div>
                                    {metric!.delta_logloss_ci_lo != null && metric!.delta_logloss_ci_hi != null ? (
                                      <div className="metrics-card-row metrics-card-ci">
                                        <span>Logloss {ciLabel}</span>
                                        <strong>
                                          [{metric!.delta_logloss_ci_lo.toFixed(4)}, {metric!.delta_logloss_ci_hi.toFixed(4)}]
                                        </strong>
                                      </div>
                                    ) : null}
                                    <div className="metrics-card-row">
                                      <span>Baseline brier</span>
                                      <strong>{formatMetricValue(metric!.baseline_brier)}</strong>
                                    </div>
                                    <div className="metrics-card-row">
                                      <span>Model brier</span>
                                      <strong>{formatMetricValue(metric!.model_brier)}</strong>
                                    </div>
                                    <div className="metrics-card-row">
                                      <span>Delta brier</span>
                                      <strong className={deltaMetricClass(metric!.delta_brier)}>
                                        {formatMetricValue(metric!.delta_brier)}
                                      </strong>
                                    </div>
                                    {metric!.delta_brier_ci_lo != null && metric!.delta_brier_ci_hi != null ? (
                                      <div className="metrics-card-row metrics-card-ci">
                                        <span>Brier {ciLabel}</span>
                                        <strong>
                                          [{metric!.delta_brier_ci_lo.toFixed(4)}, {metric!.delta_brier_ci_hi.toFixed(4)}]
                                        </strong>
                                      </div>
                                    ) : null}
                                    <div className="metrics-card-row">
                                      <span>Baseline ece_q</span>
                                      <strong>{formatMetricValue(metric!.baseline_ece_q)}</strong>
                                    </div>
                                    <div className="metrics-card-row">
                                      <span>Model ece_q</span>
                                      <strong>{formatMetricValue(metric!.model_ece_q)}</strong>
                                    </div>
                                    <div className="metrics-card-row">
                                      <span>Delta ece_q</span>
                                      <strong className={deltaMetricClass(metric!.delta_ece_q)}>
                                        {formatMetricValue(metric!.delta_ece_q)}
                                      </strong>
                                    </div>
                                    {metric!.delta_ece_q_ci_lo != null && metric!.delta_ece_q_ci_hi != null ? (
                                      <div className="metrics-card-row metrics-card-ci">
                                        <span>ECE-Q {ciLabel}</span>
                                        <strong>
                                          [{metric!.delta_ece_q_ci_lo.toFixed(4)}, {metric!.delta_ece_q_ci_hi.toFixed(4)}]
                                        </strong>
                                      </div>
                                    ) : null}
                                  </div>
                                ))}
                            </div>
                          </div>
                        ) : null}

                        {isJobComplete && activeResult?.artifact_manifest?.length ? (
                          <div className="model-detail-section file-viewer-section">
                            <span className="meta-label">Artifacts</span>
                            <div className="file-list">
                              {activeResult.artifact_manifest.map((artifact) => (
                                <button
                                  key={artifact.name}
                                  type="button"
                                  className="file-item"
                                  onClick={() => {
                                    if (!activeResult.out_dir) return;
                                    const modelId = activeResult.out_dir.split("/").pop() || "";
                                    if (modelId) {
                                      void handleSelectModel(modelId);
                                      setWorkspaceTab("models");
                                    }
                                  }}
                                >
                                  <span className="file-name">{artifact.name}</span>
                                  <span className="file-size">{artifact.type}</span>
                                </button>
                              ))}
                            </div>
                          </div>
                        ) : null}
                      </div>
                    </div>
                  ) : jobId ? (
                    <div className="empty">Loading active job state…</div>
                  ) : (
                    <div className="empty">
                      <div>No active job. Launch a run from Run Configuration.</div>
                      <button
                        className="button light calibrate-new-job-button"
                        type="button"
                        onClick={handleNewJob}
                      >
                        Back to Run Configuration
                      </button>
                    </div>
                  )}
                </div>
              </section>
            )}
          </div>
        ) : null}

        {workspaceTab === "models" ? (
          <div className="calibrate-tab-panel" role="tabpanel">
            <section className="panel calibrate-models-panel">
              <div className="panel-header calibrate-panel-header calibrate-job-config-header">
                <div>
                  <h2 className="calibrate-job-config-title">Models</h2>
                  <span className="panel-hint">
                    Browse model directories, inspect metrics/equations, and open generated artifacts.
                  </span>
                </div>
              </div>

              <div className="panel-body calibrate-models-body">
                {modelError ? <div className="error">{modelError}</div> : null}
                {models.length === 0 ? (
                  <div className="empty">No models found in the selected model directory.</div>
                ) : (
                  <div className="calibrate-model-list">
                    {models.map((model) => {
                      const selected = selectedModelId === model.id;
                      const isAutoRun = model.run_type === "auto";
                      const autoStatus = model.auto_status ?? (model.has_selected_model ? "selected" : null);
                      return (
                        <article key={model.id} className={`model-card ${selected ? "active" : ""}`}>
                          <button
                            type="button"
                            className="model-toggle"
                            onClick={() => void handleSelectModel(model.id)}
                          >
                            <div>
                              <div className="model-title-row">
                                <span className="model-title">{model.id}</span>
                                <span className={`status-pill run-type-pill ${isAutoRun ? "running" : "idle"}`}>
                                  {isAutoRun ? "AUTO" : "MANUAL"}
                                </span>
                                <span className={`status-pill ${model.has_metrics ? "success" : "idle"}`}>
                                  {model.has_metrics ? "metrics" : "no metrics"}
                                </span>
                                {model.is_two_stage ? (
                                  <span className="status-pill running">two-stage</span>
                                ) : null}
                                {isAutoRun && autoStatus ? (
                                  <span
                                    className={`status-pill ${
                                      autoStatus === "selected" ? "success" : autoStatus === "no_viable_model" ? "failed" : "idle"
                                    }`}
                                  >
                                    {autoStatus}
                                  </span>
                                ) : null}
                              </div>
                              <div className="model-meta-grid">
                                <div>
                                  <span className="meta-label">Updated</span>
                                  <span>{formatTimestamp(model.last_modified)}</span>
                                </div>
                                <div>
                                  <span className="meta-label">Dataset</span>
                                  <span>{model.dataset_id ?? "--"}</span>
                                </div>
                                <div>
                                  <span className="meta-label">Range</span>
                                  <span>
                                    {model.train_date_start && model.train_date_end
                                      ? `${model.train_date_start} to ${model.train_date_end}`
                                      : "--"}
                                  </span>
                                </div>
                                <div>
                                  <span className="meta-label">Tickers/DOW</span>
                                  <span>{model.tickers_summary ?? model.dow_regime ?? "--"}</span>
                                </div>
                                <div>
                                  <span className="meta-label">Hyperparams</span>
                                  <span>
                                    split={model.split_strategy ?? "--"}, C={model.c_value ?? "--"}, calib={model.calibration_method ?? "--"}
                                  </span>
                                </div>
                                {isAutoRun ? (
                                  <div>
                                    <span className="meta-label">Auto selection</span>
                                    <span>
                                      {model.has_selected_model
                                        ? `selected${model.selected_trial_id != null ? ` (trial ${model.selected_trial_id})` : ""}`
                                        : "no selected model"}
                                    </span>
                                  </div>
                                ) : null}
                              </div>
                            </div>
                          </button>

                          <div className="model-actions-row">
                            <label className="checkbox inline-select">
                              <input
                                type="checkbox"
                                checked={modelCompare.left === model.id || modelCompare.right === model.id}
                                onChange={(event) => {
                                  if (!event.target.checked) {
                                    setModelCompare((prev) => ({
                                      left: prev.left === model.id ? null : prev.left,
                                      right: prev.right === model.id ? null : prev.right,
                                    }));
                                    return;
                                  }
                                  setModelCompare((prev) => {
                                    if (!prev.left) return { ...prev, left: model.id };
                                    if (!prev.right && prev.left !== model.id) return { ...prev, right: model.id };
                                    if (prev.left === model.id || prev.right === model.id) return prev;
                                    return { left: prev.left, right: model.id };
                                  });
                                }}
                              />
                              Compare
                            </label>
                            <button className="button light small" type="button" onClick={() => void handleRenameModel(model.id)}>
                              Rename model
                            </button>
                            <button className="button ghost danger small" type="button" onClick={() => setDeleteTarget(model)}>
                              Delete model
                            </button>
                          </div>

                          {selected ? (
                            <div className="model-detail-shell">
                              {isModelDetailLoading ? (
                                <div className="empty">Loading model detail…</div>
                              ) : modelDetailError ? (
                                <div className="error">{modelDetailError}</div>
                              ) : modelDetail ? (
                                <>
                                  {modelDetail.metrics_summary ? (
                                    <div className="metrics-summary">
                                      <div className="metrics-summary-header">
                                        <span className="meta-label">Performance summary</span>
                                        <span className="metrics-summary-note">Delta values are model minus baseline.</span>
                                      </div>
                                      {(() => {
                                        const modelRows = modelDetail.split_row_counts ?? {};
                                        const modelGroups = modelDetail.split_group_counts ?? {};
                                        const modelTrainRows = modelRows.train_fit ?? modelRows.train ?? null;
                                        const modelValRows = modelRows.val ?? null;
                                        const modelTestRows = modelRows.test ?? null;
                                        const modelTrainGroups = modelGroups.train_fit ?? modelGroups.train ?? null;
                                        const modelValGroups = modelGroups.val ?? null;
                                        const modelTestGroups = modelGroups.test ?? null;
                                        const hasCounts =
                                          modelTrainRows != null ||
                                          modelValRows != null ||
                                          modelTestRows != null ||
                                          modelTrainGroups != null ||
                                          modelValGroups != null ||
                                          modelTestGroups != null;
                                        if (!hasCounts) return null;
                                        return (
                                          <div className="run-meta-grid">
                                            <div>
                                              <span className="meta-label">Train rows</span>
                                              <span>{formatCountValue(modelTrainRows as number | null)}</span>
                                            </div>
                                            <div>
                                              <span className="meta-label">Val rows</span>
                                              <span>{formatCountValue(modelValRows as number | null)}</span>
                                            </div>
                                            <div>
                                              <span className="meta-label">Test rows</span>
                                              <span>{formatCountValue(modelTestRows as number | null)}</span>
                                            </div>
                                            <div>
                                              <span className="meta-label">Train groups</span>
                                              <span>{formatCountValue(modelTrainGroups as number | null)}</span>
                                            </div>
                                            <div>
                                              <span className="meta-label">Val groups</span>
                                              <span>{formatCountValue(modelValGroups as number | null)}</span>
                                            </div>
                                            <div>
                                              <span className="meta-label">Test groups</span>
                                              <span>{formatCountValue(modelTestGroups as number | null)}</span>
                                            </div>
                                          </div>
                                        );
                                      })()}
                                      <div className="metrics-summary-grid">
                                        {metricsOrder
                                          .map((split) => modelDetail.metrics_summary?.[split])
                                          .filter(Boolean)
                                          .map((metric) => (
                                          <div key={`${model.id}-${metric!.split}`} className="metrics-card">
                                            <div className="metrics-card-heading">
                                              <strong>{metric!.split}</strong>
                                              <span className={`status-pill ${metric!.status === "good" ? "success" : "failed"}`}>
                                                {metric!.status}
                                              </span>
                                            </div>
                                            <div className="metrics-card-row">
                                              <span>Baseline logloss</span>
                                              <strong>{formatMetricValue(metric!.baseline_logloss)}</strong>
                                            </div>
                                            <div className="metrics-card-row">
                                              <span>Model logloss</span>
                                              <strong>{formatMetricValue(metric!.model_logloss)}</strong>
                                            </div>
                                            <div className="metrics-card-row">
                                              <span>Delta logloss</span>
                                              <strong className={deltaMetricClass(metric!.delta_model_minus_baseline)}>
                                                {formatMetricValue(metric!.delta_model_minus_baseline)}
                                              </strong>
                                            </div>
                                            {metric!.delta_logloss_ci_lo != null && metric!.delta_logloss_ci_hi != null ? (
                                              <div className="metrics-card-row metrics-card-ci">
                                                <span>Logloss {modelCiLabel}</span>
                                                <strong>
                                                  [{metric!.delta_logloss_ci_lo.toFixed(4)}, {metric!.delta_logloss_ci_hi.toFixed(4)}]
                                                </strong>
                                              </div>
                                            ) : null}
                                            <div className="metrics-card-row">
                                              <span>Baseline brier</span>
                                              <strong>{formatMetricValue(metric!.baseline_brier)}</strong>
                                            </div>
                                            <div className="metrics-card-row">
                                              <span>Model brier</span>
                                              <strong>{formatMetricValue(metric!.model_brier)}</strong>
                                            </div>
                                            <div className="metrics-card-row">
                                              <span>Delta brier</span>
                                              <strong className={deltaMetricClass(metric!.delta_brier)}>
                                                {formatMetricValue(metric!.delta_brier)}
                                              </strong>
                                            </div>
                                            {metric!.delta_brier_ci_lo != null && metric!.delta_brier_ci_hi != null ? (
                                              <div className="metrics-card-row metrics-card-ci">
                                                <span>Brier {modelCiLabel}</span>
                                                <strong>
                                                  [{metric!.delta_brier_ci_lo.toFixed(4)}, {metric!.delta_brier_ci_hi.toFixed(4)}]
                                                </strong>
                                              </div>
                                            ) : null}
                                            <div className="metrics-card-row">
                                              <span>Baseline ece_q</span>
                                              <strong>{formatMetricValue(metric!.baseline_ece_q)}</strong>
                                            </div>
                                            <div className="metrics-card-row">
                                              <span>Model ece_q</span>
                                              <strong>{formatMetricValue(metric!.model_ece_q)}</strong>
                                            </div>
                                            <div className="metrics-card-row">
                                              <span>Delta ece_q</span>
                                              <strong className={deltaMetricClass(metric!.delta_ece_q)}>
                                                {formatMetricValue(metric!.delta_ece_q)}
                                              </strong>
                                            </div>
                                            {metric!.delta_ece_q_ci_lo != null && metric!.delta_ece_q_ci_hi != null ? (
                                              <div className="metrics-card-row metrics-card-ci">
                                                <span>ECE-Q {modelCiLabel}</span>
                                                <strong>
                                                  [{metric!.delta_ece_q_ci_lo.toFixed(4)}, {metric!.delta_ece_q_ci_hi.toFixed(4)}]
                                                </strong>
                                              </div>
                                            ) : null}
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  ) : null}

                                  {modelDetail.model_equation ? (
                                    <div className="equation-summary">
                                      <span className="meta-label">Model equation</span>
                                      <LatexBlock latex={modelDetail.model_equation} />
                                    </div>
                                  ) : null}

                                  {modelDetail.stage1_equation ? (
                                    <div className="equation-summary">
                                      <span className="meta-label">Stage A equation</span>
                                      <LatexBlock latex={modelDetail.stage1_equation} />
                                    </div>
                                  ) : null}

                                  {modelDetail.two_stage_equation ? (
                                    <div className="equation-summary">
                                      <span className="meta-label">Stage B equation</span>
                                      <LatexBlock latex={modelDetail.two_stage_equation} />
                                    </div>
                                  ) : null}

                                  {isAutoRun && !model.has_selected_model ? (
                                    <div className="warning auto-no-viable-callout">
                                      No viable model was selected from this auto run. Search diagnostics remain available below.
                                    </div>
                                  ) : null}

                                  {modelFiles?.files?.length ? (
                                    <div className="model-detail-section file-viewer-section">
                                      <span className="meta-label">Artifacts</span>
                                      {isAutoRun ? (
                                        <div className="artifact-section-stack">
                                          <div className="artifact-section-block">
                                            <span className="meta-label">Selected Model</span>
                                            {groupedModelFiles.selected_model.length ? (
                                              <div className="file-list">
                                                {groupedModelFiles.selected_model.map((file) => {
                                                  const filePath = file.relative_path ?? file.name;
                                                  return (
                                                    <button
                                                      key={`${model.id}-${filePath}`}
                                                      type="button"
                                                      className={`file-item ${selectedFilePath === filePath ? "active" : ""}`}
                                                      onClick={() => file.is_viewable && void handleOpenFile(file)}
                                                      disabled={!file.is_viewable}
                                                    >
                                                      <span className="file-name">{file.name}</span>
                                                      {file.relative_path && file.relative_path !== file.name ? (
                                                        <span className="file-path">{file.relative_path}</span>
                                                      ) : null}
                                                      <span className="file-size">
                                                        {file.size_bytes < 1024
                                                          ? `${file.size_bytes} B`
                                                          : `${(file.size_bytes / 1024).toFixed(1)} KB`}
                                                      </span>
                                                    </button>
                                                  );
                                                })}
                                              </div>
                                            ) : (
                                              <div className="empty">No selected-model artifacts.</div>
                                            )}
                                          </div>
                                          <div className="artifact-section-block">
                                            <span className="meta-label">Auto Search</span>
                                            {groupedModelFiles.auto_search.length ? (
                                              <div className="file-list">
                                                {groupedModelFiles.auto_search.map((file) => {
                                                  const filePath = file.relative_path ?? file.name;
                                                  return (
                                                    <button
                                                      key={`${model.id}-${filePath}`}
                                                      type="button"
                                                      className={`file-item ${selectedFilePath === filePath ? "active" : ""}`}
                                                      onClick={() => file.is_viewable && void handleOpenFile(file)}
                                                      disabled={!file.is_viewable}
                                                    >
                                                      <span className="file-name">{file.name}</span>
                                                      {file.relative_path && file.relative_path !== file.name ? (
                                                        <span className="file-path">{file.relative_path}</span>
                                                      ) : null}
                                                      <span className="file-size">
                                                        {file.size_bytes < 1024
                                                          ? `${file.size_bytes} B`
                                                          : `${(file.size_bytes / 1024).toFixed(1)} KB`}
                                                      </span>
                                                    </button>
                                                  );
                                                })}
                                              </div>
                                            ) : (
                                              <div className="empty">No auto-search artifacts.</div>
                                            )}
                                          </div>
                                          {groupedModelFiles.legacy_root.length ? (
                                            <div className="artifact-section-block">
                                              <span className="meta-label">Legacy Root</span>
                                              <div className="file-list">
                                                {groupedModelFiles.legacy_root.map((file) => {
                                                  const filePath = file.relative_path ?? file.name;
                                                  return (
                                                    <button
                                                      key={`${model.id}-${filePath}`}
                                                      type="button"
                                                      className={`file-item ${selectedFilePath === filePath ? "active" : ""}`}
                                                      onClick={() => file.is_viewable && void handleOpenFile(file)}
                                                      disabled={!file.is_viewable}
                                                    >
                                                      <span className="file-name">{file.name}</span>
                                                      {file.relative_path && file.relative_path !== file.name ? (
                                                        <span className="file-path">{file.relative_path}</span>
                                                      ) : null}
                                                      <span className="file-size">
                                                        {file.size_bytes < 1024
                                                          ? `${file.size_bytes} B`
                                                          : `${(file.size_bytes / 1024).toFixed(1)} KB`}
                                                      </span>
                                                    </button>
                                                  );
                                                })}
                                              </div>
                                            </div>
                                          ) : null}
                                        </div>
                                      ) : (
                                        <div className="file-list">
                                          {groupedModelFiles.legacy_root.map((file) => {
                                            const filePath = file.relative_path ?? file.name;
                                            return (
                                              <button
                                                key={`${model.id}-${filePath}`}
                                                type="button"
                                                className={`file-item ${selectedFilePath === filePath ? "active" : ""}`}
                                                onClick={() => file.is_viewable && void handleOpenFile(file)}
                                                disabled={!file.is_viewable}
                                              >
                                                <span className="file-name">{file.name}</span>
                                                {file.relative_path && file.relative_path !== file.name ? (
                                                  <span className="file-path">{file.relative_path}</span>
                                                ) : null}
                                                <span className="file-size">
                                                  {file.size_bytes < 1024
                                                    ? `${file.size_bytes} B`
                                                    : `${(file.size_bytes / 1024).toFixed(1)} KB`}
                                                </span>
                                              </button>
                                            );
                                          })}
                                        </div>
                                      )}
                                      {selectedFilePath ? (
                                        <div className="file-content-panel">
                                          <div className="file-content-header">
                                            <div className="file-content-title">
                                              {selectedFilePath}
                                              {ARTIFACT_DESCRIPTIONS[selectedArtifactName] ? (
                                                <span className="file-content-subtitle">
                                                  {ARTIFACT_DESCRIPTIONS[selectedArtifactName]}
                                                </span>
                                              ) : null}
                                            </div>
                                            <div className="file-content-actions">
                                              <button
                                                className="button light small"
                                                type="button"
                                                onClick={() => setShowRawFile((prev) => !prev)}
                                              >
                                                {showRawFile ? "View visual" : "View raw"}
                                              </button>
                                              <button
                                                className="button small"
                                                type="button"
                                                onClick={() => {
                                                  setSelectedFilePath(null);
                                                  setFileContent(null);
                                                  setFileError(null);
                                                }}
                                              >
                                                Close
                                              </button>
                                            </div>
                                          </div>
                                          {fileLoading ? <div className="empty">Loading file…</div> : null}
                                          {fileError ? <div className="error">{fileError}</div> : null}
                                          {fileContent?.truncated ? (
                                            <div className="warning">File preview truncated to 512 KB.</div>
                                          ) : null}
                                          {fileContent ? renderArtifactView() : null}
                                        </div>
                                      ) : null}
                                    </div>
                                  ) : null}
                                </>
                              ) : null}
                            </div>
                          ) : null}
                        </article>
                      );
                    })}
                  </div>
                )}

                <section className="section-card calibrate-section-card">
                  <h3 className="section-heading">Compare Models</h3>
                  {!compareLeft || !compareRight ? (
                    <div className="empty">Select exactly two models to compare.</div>
                  ) : (
                    <div className="compare-grid">
                      <div>
                        <span className="meta-label">Model A</span>
                        <span>{compareLeft.id}</span>
                      </div>
                      <div>
                        <span className="meta-label">Model B</span>
                        <span>{compareRight.id}</span>
                      </div>
                      <div>
                        <span className="meta-label">Delta (B-A) C</span>
                        <span>
                          {compareRight.c_value != null && compareLeft.c_value != null
                            ? (compareRight.c_value - compareLeft.c_value).toFixed(4)
                            : "--"}
                        </span>
                      </div>
                      <div>
                        <span className="meta-label">Split strategy</span>
                        <span>{compareLeft.split_strategy ?? "--"} vs {compareRight.split_strategy ?? "--"}</span>
                      </div>
                    </div>
                  )}
                </section>
              </div>
            </section>
          </div>
        ) : null}

        {workspaceTab === "documentation" ? (
          <div className="calibrate-tab-panel" role="tabpanel">
            <section className="panel calibrate-documentation-panel">
              <div className="panel-header calibrate-panel-header calibrate-job-config-header">
                <div>
                  <h2 className="calibrate-job-config-title">Documentation</h2>
                  <span className="panel-hint">
                    Calibration methodology and control-by-control guidance.
                  </span>
                </div>
              </div>
              <div className="panel-body calibrate-documentation-body">
                <CalibrateDocContent className="calibrate-doc-embedded" />
              </div>
            </section>
          </div>
        ) : null}
      </div>

      {deleteTarget ? (
        <div
          className="calibrate-delete-modal-overlay"
          onClick={() => {
            if (deleteLoading) return;
            setDeleteTarget(null);
            setDeleteConfirmText("");
          }}
        >
          <div
            className="calibrate-delete-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="calibrate-delete-modal-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="calibrate-delete-modal-header">
              <h3 id="calibrate-delete-modal-title">Delete calibration model</h3>
              <p>
                This permanently deletes <span className="calibrate-delete-modal-code">{deleteTarget.id}</span> and its artifacts.
              </p>
            </div>
            <div className="calibrate-delete-modal-body">
              <label htmlFor="calibrateDeleteConfirmInput">Type <strong>DELETE</strong> to confirm</label>
              <input
                id="calibrateDeleteConfirmInput"
                className="input"
                type="text"
                value={deleteConfirmText}
                onChange={(event) => setDeleteConfirmText(event.target.value)}
                placeholder="DELETE"
                autoFocus
                disabled={deleteLoading}
              />
            </div>
            <div className="calibrate-delete-modal-actions">
              <button className="button ghost" type="button" disabled={deleteLoading} onClick={() => setDeleteTarget(null)}>
                Cancel
              </button>
              <button
                className="button danger calibrate-delete-modal-confirm"
                type="button"
                onClick={() => void handleDeleteModel()}
                disabled={deleteConfirmText !== "DELETE" || deleteLoading}
              >
                {deleteLoading ? "Deleting…" : "Delete permanently"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
