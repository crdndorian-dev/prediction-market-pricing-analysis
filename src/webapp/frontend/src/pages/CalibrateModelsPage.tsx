import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from "react";
import katex from "katex";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";

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
  type SelectableFeatureDescriptor,
  type StataDiagnosticsPayload,
  type WeightingPreviewResponse,
} from "../api/calibrateModels";
import PipelineStatusCard from "../components/PipelineStatusCard";
import PipelineProgressBar from "../components/PipelineProgressBar";
import {
  CalibrationCurveCard,
  StataDiagnosticsPanel,
  isStataDiagnosticsPayload,
} from "../components/StataDiagnosticsPanel";
import { useCalibrationJob } from "../contexts/calibrationJob";
import { useAnyJobRunning } from "../contexts/jobGuard";
import {
  buildRetiredFeatureNotice,
  coerceWarningList,
  findDiagnosticsSkipWarning,
  stripRetiredFeatureSelections,
} from "./calibrateFeaturePolicy";
import { buildModelInspectorArtifactState } from "./calibrateModelInspector";
import "katex/dist/katex.min.css";
import "./CalibrateModelsPage.css";

type WorkspaceTab = "run_job" | "models";
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
  autoAdvancedSearch: boolean;
  autoOuterFolds: string;
  autoOuterTestWeeks: string;
  autoOuterGapWeeks: string;
  autoOuterSelectionMetric: AutoOuterSelectionMetric;
  autoOuterMinImproveFraction: string;
  autoOuterMaxWorstDelta: string;
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
const PENDING_RUN_AGAIN_CONFIG_KEY = "polyedgetool.calibrate.v2.pendingRunAgainConfig";

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
  ["x_logit_prn", "rv20", "abs_log_m_fwd", "rel_spread_median"],
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
const DEFAULT_SELECTED_FEATURES = [
  "log_m_fwd",
  "rv20",
  "rel_spread_median",
  "dividend_yield",
];

const defaultModelName = () => {
  const stamp = new Date().toISOString().replace(/[:.]/g, "").slice(0, 15);
  return `calibration-${stamp}`;
};

const parseWorkspaceTab = (search: string): WorkspaceTab => {
  const params = new URLSearchParams(search);
  return params.get("tab") === "models" ? "models" : "run_job";
};

const buildCalibrateTabHref = (tab: WorkspaceTab): string =>
  tab === "models" ? "/calibrate?tab=models" : "/calibrate";

const buildCalibrationModelDetailHref = (modelId: string): string =>
  `/calibrate/models/${encodeURIComponent(modelId)}`;

const loadPendingRunAgainConfig = (): Record<string, unknown> | null => {
  if (typeof window === "undefined") return null;
  try {
    const raw = sessionStorage.getItem(PENDING_RUN_AGAIN_CONFIG_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    // ignore storage failures
  }
  return null;
};

const storePendingRunAgainConfig = (config: Record<string, unknown>) => {
  try {
    sessionStorage.setItem(PENDING_RUN_AGAIN_CONFIG_KEY, JSON.stringify(config));
  } catch {
    // ignore storage failures
  }
};

const clearPendingRunAgainConfig = () => {
  try {
    sessionStorage.removeItem(PENDING_RUN_AGAIN_CONFIG_KEY);
  } catch {
    // ignore storage failures
  }
};

const resolveBootstrapGroupValue = (
  requested: BootstrapGroupMode,
  availableGroupingKeys?: string[] | null,
): BootstrapGroupMode => {
  if (!availableGroupingKeys || availableGroupingKeys.length === 0) return requested;
  const available = new Set(availableGroupingKeys.map((key) => key.toLowerCase()));
  if (requested === "contract_id" && !available.has("contract_id")) {
    return available.has("group_id") ? "group_id" : "auto";
  }
  if (requested === "group_id" && !available.has("group_id")) {
    return "auto";
  }
  if (requested === "ticker_day" && !available.has("ticker_day")) {
    return "auto";
  }
  if (requested === "day" && !available.has("day")) {
    return "auto";
  }
  return requested;
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

const DEFAULT_REGULARIZATION_PENALTY = "l2";
const DEFAULT_REGULARIZATION_SOLVER = "lbfgs";

const inferRegularizationSummary = (data: Record<string, unknown>): Record<string, unknown> => {
  const regularization =
    data.regularization && typeof data.regularization === "object"
      ? (data.regularization as Record<string, unknown>)
      : null;
  return {
    penalty:
      regularization?.penalty ??
      data.regularization_penalty ??
      DEFAULT_REGULARIZATION_PENALTY,
    solver:
      regularization?.solver ??
      data.regularization_solver ??
      DEFAULT_REGULARIZATION_SOLVER,
    best_c:
      regularization?.best_c ??
      data.best_C ??
      null,
    c_grid:
      regularization?.c_grid ??
      null,
    selection_rule:
      regularization?.selection_rule ??
      data.c_selection_rule ??
      null,
  };
};

const splitCsvValue = (value: unknown): string[] => {
  if (typeof value !== "string") return [];
  return Array.from(
    new Set(
      value
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean),
    ),
  );
};

const toProjectRelativePath = (value: unknown): string => {
  if (typeof value !== "string") return "";
  const normalized = value.replace(/\\/g, "/").trim();
  if (!normalized) return "";
  if (!normalized.startsWith("/")) return normalized;
  const srcIndex = normalized.lastIndexOf("/src/");
  if (srcIndex >= 0) return normalized.slice(srcIndex + 1);
  const dataIndex = normalized.lastIndexOf("/data/");
  if (dataIndex >= 0) return normalized.slice(dataIndex + 1);
  return normalized;
};

const resolveTimeRegimeKey = (config: Record<string, unknown>): TimeRegimeKey => {
  const filters =
    config.filters && typeof config.filters === "object"
      ? (config.filters as Record<string, unknown>)
      : null;
  const topLevelDow = typeof config.asof_dow_allowed === "string" ? config.asof_dow_allowed : null;
  const filterDowRaw = Array.isArray(filters?.asof_dow_allowed) ? filters?.asof_dow_allowed[0] : filters?.asof_dow_allowed;
  const filterTdaysRaw = Array.isArray(filters?.tdays_allowed) ? filters?.tdays_allowed[0] : filters?.tdays_allowed;
  const topLevelTdays = toNumber(config.tdays_allowed);
  const filterTdays = toNumber(filterTdaysRaw);
  const dowMap: Record<string, TimeRegimeKey> = {
    mon: "mon_4",
    monday: "mon_4",
    tue: "tue_3",
    tues: "tue_3",
    tuesday: "tue_3",
    wed: "wed_2",
    wednesday: "wed_2",
    thu: "thu_1",
    thur: "thu_1",
    thurs: "thu_1",
    thursday: "thu_1",
  };
  if (topLevelDow) {
    const match = dowMap[topLevelDow.trim().toLowerCase()];
    if (match) return match;
  }
  if (typeof filterDowRaw === "string") {
    const match = dowMap[filterDowRaw.trim().toLowerCase()];
    if (match) return match;
  }
  const dowIndex = toNumber(filterDowRaw);
  if (dowIndex != null) {
    if (dowIndex === 0) return "mon_4";
    if (dowIndex === 1) return "tue_3";
    if (dowIndex === 2) return "wed_2";
    if (dowIndex === 3) return "thu_1";
  }
  if (topLevelTdays != null) {
    if (topLevelTdays === 4) return "mon_4";
    if (topLevelTdays === 3) return "tue_3";
    if (topLevelTdays === 2) return "wed_2";
    if (topLevelTdays === 1) return "thu_1";
  }
  if (filterTdays != null) {
    if (filterTdays === 4) return "mon_4";
    if (filterTdays === 3) return "tue_3";
    if (filterTdays === 2) return "wed_2";
    if (filterTdays === 1) return "thu_1";
  }
  return "thu_1";
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
  "diagnostics_table.json": "Informational shadow-model diagnostics and inferential tables.",
  "metrics.csv": "Split metrics for baseline vs model with deltas and confidence intervals.",
  "metrics_summary.json": "Summary metrics across validation and test splits.",
  "coefficient_diagnostics.csv": "Coefficient diagnostics with inferential statistics from the shadow GLM.",
  "marginal_effects.csv": "Average marginal effects table from the shadow GLM.",
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

const ARTIFACT_TITLES: Record<string, string> = {
  "diagnostics_table.json": "Inferred model diagnostics",
  "metrics.csv": "Metrics",
  "metrics_summary.json": "Metrics Summary",
  "coefficient_diagnostics.csv": "Coefficient Diagnostics",
  "marginal_effects.csv": "Marginal Effects",
  "split_timeline.json": "Split Timeline",
  "fold_deltas.csv": "Fold Delta",
  "group_delta_distribution.csv": "Group Delta Distribution",
  "audit_split_composition.csv": "Split Composition Audit",
  "audit_overlap.json": "Overlap Audit",
  "audit_weight_distribution.json": "Weight Distribution Audit",
  "config.executed.json": "Executed Config",
  "metadata.json": "Run Metadata",
  "feature_manifest.json": "Feature Manifest",
  "best_config.json": "Best Config",
  "best_model_report.md": "Best Model Report",
  "leaderboard.csv": "Leaderboard",
  "auto_search_leaderboard.csv": "Auto Search Leaderboard",
  "auto_search_summary.json": "Auto Search Summary",
  "auto_search_progress.json": "Auto Search Progress",
  "run_manifest.json": "Run Manifest",
  "outer_folds.json": "Outer Folds",
  "outer_cv_summary.json": "Outer CV Summary",
  "outer_fold_results.csv": "Outer Fold Result",
  "reliability_bins.csv": "Reliability Plot",
  "rolling_summary.csv": "Rolling Summary",
  "rolling_windows.csv": "Rolling Window",
  "metrics_groups.csv": "Metrics by Group",
  "two_stage_metrics.csv": "Two-Stage Metrics",
  "two_stage_metrics_summary.json": "Two-Stage Metrics Summary",
  "two_stage_metadata.json": "Two-Stage Metadata",
  "progress.json": "Progress",
  "trial_result.json": "Trial Result",
};

const INFERRED_DIAGNOSTICS_DISCLAIMER =
  "Informational only: these inferred statistics come from a shadow statistical model, not from the production ML model optimized for logloss.";

const HIDDEN_ARTIFACT_NAMES = new Set<string>(["auto_search_no_viable.json"]);
const DEFAULT_CHART_WIDTH = 960;
const GENERAL_EQUATION_NOTE_PREFIXES = [
  "displayed coefficients are the base logistic layer;",
  "base logistic layer is fit with sklearn logisticregression",
  "compact equation uses ticker-dependent placeholder terms;",
  "equation is shown in transformed model basis",
  "categorical one-hot encoding uses drop-first reference levels",
  "stage b final probabilities may apply an additional platt calibration transform",
  "stage b logistic layer uses sklearn logisticregression",
] as const;

const fileBaseName = (path: string | null | undefined): string => {
  if (!path) return "";
  const normalized = path.replace(/\\/g, "/");
  const parts = normalized.split("/");
  return parts[parts.length - 1] || normalized;
};

const artifactFilePath = (file: ModelFileSummary): string => file.relative_path ?? file.name;

const humanizeLabel = (value: string): string =>
  value
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());

const isHiddenArtifactPath = (path: string | null | undefined): boolean =>
  HIDDEN_ARTIFACT_NAMES.has(fileBaseName(path));

const normalizeNote = (value: string): string =>
  value.trim().replace(/\s+/g, " ").toLowerCase();

const isGeneralEquationNote = (note: string): boolean => {
  const normalized = normalizeNote(note);
  return GENERAL_EQUATION_NOTE_PREFIXES.some((prefix) => normalized.startsWith(prefix));
};

const artifactDisplayTitle = (path: string | null | undefined): string => {
  const base = fileBaseName(path);
  if (!base) return "Artifact";
  if (ARTIFACT_TITLES[base]) return ARTIFACT_TITLES[base];
  return humanizeLabel(base.replace(/\.[^.]+$/, "").replace(/[.]+/g, " "));
};

const formatFileSizeLabel = (sizeBytes: number): string =>
  sizeBytes < 1024 ? `${sizeBytes} B` : `${(sizeBytes / 1024).toFixed(1)} KB`;

const compactMetaLine = (...values: Array<string | null | undefined>): string => {
  const visible = values
    .map((value) => value?.trim())
    .filter((value): value is string => Boolean(value));
  return visible.join(" • ");
};

const formatModelRange = (
  start?: string | null,
  end?: string | null,
): string =>
  start && end ? `${start} → ${end}` : "--";

const formatModelListMeta = (model: ModelRunSummary): string =>
  compactMetaLine(
    model.dataset_id ?? "--",
    `split=${model.split_strategy ?? "--"}`,
    `C=${model.c_value != null ? formatMaybe(model.c_value) : "--"}`,
    `calib=${model.calibration_method ?? "--"}`,
  );

const isProbablyNumeric = (value: string): boolean => {
  const trimmed = value.trim();
  if (!trimmed) return false;
  const normalized = trimmed.replace(/,/g, "");
  if (/^(true|false|null|none|nan)$/i.test(normalized)) return false;
  const parsed = Number(normalized);
  return Number.isFinite(parsed);
};

const inferNumericColumns = (parsed: ParsedCsv): Set<string> => {
  const numericColumns = new Set<string>();
  parsed.headers.forEach((column) => {
    const values = parsed.rows
      .map((row) => String(row[column] ?? "").trim())
      .filter(Boolean);
    if (!values.length) return;
    const numericCount = values.filter(isProbablyNumeric).length;
    if (numericCount / values.length >= 0.85) {
      numericColumns.add(column);
    }
  });
  return numericColumns;
};

const buildLinearTicks = (min: number, max: number, count = 5): number[] => {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (Math.abs(max - min) < 1e-9) return [min];
  return Array.from({ length: Math.max(2, count) }, (_, idx) => (
    min + ((max - min) * idx) / (Math.max(2, count) - 1)
  ));
};

const buildIndexTicks = (size: number, count = 6): number[] => {
  if (size <= 0) return [];
  if (size === 1) return [0];
  const steps = Math.min(size, Math.max(2, count));
  const ticks = new Set<number>();
  for (let idx = 0; idx < steps; idx += 1) {
    ticks.add(Math.round(((size - 1) * idx) / (steps - 1)));
  }
  return Array.from(ticks).sort((left, right) => left - right);
};

const formatChartNumber = (value: number): string => {
  if (!Number.isFinite(value)) return "--";
  const abs = Math.abs(value);
  if (abs >= 1000) return value.toFixed(0);
  if (abs >= 100) return value.toFixed(1);
  if (abs >= 1) return value.toFixed(3);
  if (abs >= 0.01) return value.toFixed(4);
  return value.toExponential(1);
};

const formatChartDate = (timestamp: number, rangeMs = 0): string => {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    ...(rangeMs > 330 * 24 * 3600 * 1000 ? { year: "numeric" as const } : {}),
  });
};

const parseChartDateLabel = (value: string): number | null => {
  const trimmed = value.trim();
  if (!trimmed || !/[-/]|[A-Za-z]/.test(trimmed)) return null;
  const parsed = Date.parse(trimmed);
  return Number.isNaN(parsed) ? null : parsed;
};

const truncateAxisLabel = (value: string, max = 16): string =>
  value.length > max ? `${value.slice(0, max - 1)}…` : value;

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
            <span className="artifact-kv-value">{text}</span>
          </div>
        );
      })}
    </div>
  );
};

const CsvTableView = ({ parsed, limit = 50 }: { parsed: ParsedCsv; limit?: number }) => {
  const numericColumns = inferNumericColumns(parsed);
  const tableMinWidth = Math.max(640, parsed.headers.length * 140);
  if (!parsed.headers.length) {
    return <div className="empty">CSV did not include headers.</div>;
  }
  const rows = parsed.rows.slice(0, limit);
  return (
    <div className="table-container artifact-table">
      <table className="preview-table artifact-preview-table" style={{ minWidth: `${tableMinWidth}px` }}>
        <thead>
          <tr>
            {parsed.headers.map((column) => (
              <th
                key={column}
                className={numericColumns.has(column) ? "artifact-table-cell-number" : "artifact-table-cell-text"}
              >
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length ? (
            rows.map((row, idx) => (
              <tr key={idx}>
                {parsed.headers.map((column) => (
                  <td
                    key={column}
                    className={numericColumns.has(column) ? "artifact-table-cell-number" : "artifact-table-cell-text"}
                  >
                    {row[column] ?? ""}
                  </td>
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

const ArtifactFileButton = ({
  titlePath,
  displayPath,
  meta,
  isActive = false,
  disabled = false,
  className,
  onClick,
}: {
  titlePath: string;
  displayPath: string;
  meta: string;
  isActive?: boolean;
  disabled?: boolean;
  className?: string;
  onClick: () => void;
}) => (
  <button
    type="button"
    className={`file-item${className ? ` ${className}` : ""}${isActive ? " active" : ""}`}
    onClick={onClick}
    disabled={disabled}
  >
    <span className="file-name">{artifactDisplayTitle(titlePath)}</span>
    <span className="file-path">{displayPath}</span>
    <span className="file-size">{meta}</span>
  </button>
);

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

const EquationNotes = ({ spec }: { spec?: ModelDetailResponse["model_equation_spec"] | null }) => {
  const notes = Array.isArray(spec?.notes)
    ? spec.notes.filter(
        (note): note is string =>
          typeof note === "string" && note.trim().length > 0 && !isGeneralEquationNote(note),
      )
    : [];
  if (!notes.length) return null;
  return (
    <div className="equation-notes">
      {notes.map((note) => (
        <div key={note} className="equation-note">
          {note}
        </div>
      ))}
    </div>
  );
};

const ConfigJsonView = ({
  data,
  title,
  onRunAgain,
}: {
  data: Record<string, unknown>;
  title?: string;
  onRunAgain?: (() => void) | null;
}) => (
  <div className="artifact-stack">
    <section className="artifact-section">
      <div className="artifact-section-header">
        <span className="meta-label">{title ?? "Config"}</span>
        {onRunAgain ? (
          <button type="button" className="button light small" onClick={onRunAgain}>
            Run again
          </button>
        ) : null}
      </div>
      <span className="artifact-section-copy">
        Load this configuration into the Run Job tab so you can rerun it directly.
      </span>
    </section>
    <JsonSectionView title="Dataset" data={{ csv: data.csv, out_dir: data.out_dir, run_mode: data.run_mode }} />
    <JsonSectionView title="Split" data={(data.split as Record<string, unknown>) ?? null} />
    <JsonSectionView title="Regularization" data={inferRegularizationSummary(data)} />
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
        regularization_penalty: data.regularization_penalty ?? DEFAULT_REGULARIZATION_PENALTY,
        regularization_solver: data.regularization_solver ?? DEFAULT_REGULARIZATION_SOLVER,
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
  const width = DEFAULT_CHART_WIDTH;
  const leftPad = 110;
  const rightPad = 28;
  const topPad = 28;
  const bottomPad = 72;
  const rowHeight = 26;
  const barHeight = 14;
  const totalRows = entries.length + globalRows.length;
  const plotHeight = Math.max(120, totalRows * rowHeight);
  const height = topPad + plotHeight + bottomPad;
  const xAxisY = topPad + plotHeight + 8;
  const scaleX = (time: number) => {
    const ratio = (time - minTime) / Math.max(1, maxTime - minTime);
    return leftPad + ratio * (width - leftPad - rightPad);
  };
  const xTicks = buildLinearTicks(minTime, maxTime, 5);
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
      <div className="artifact-chart-panel">
        <svg viewBox={`0 0 ${width} ${height}`} className="artifact-chart artifact-chart-tall">
          <rect x={0} y={0} width={width} height={height} className="chart-frame" />
          {xTicks.map((tick) => {
            const x = scaleX(tick);
            return (
              <g key={`timeline-tick-${tick}`}>
                <line x1={x} x2={x} y1={topPad} y2={topPad + plotHeight} className="artifact-grid-line" />
                <text x={x} y={xAxisY + 18} textAnchor="middle" className="artifact-tick-label">
                  {formatChartDate(tick, maxTime - minTime)}
                </text>
              </g>
            );
          })}
          <line x1={leftPad} x2={width - rightPad} y1={xAxisY} y2={xAxisY} className="artifact-axis-line" />
          <text x={(leftPad + width - rightPad) / 2} y={height - 16} textAnchor="middle" className="artifact-axis-title">
            Calendar date
          </text>
          <text
            x={26}
            y={topPad + plotHeight / 2}
            transform={`rotate(-90 26 ${topPad + plotHeight / 2})`}
            textAnchor="middle"
            className="artifact-axis-title"
          >
            Fold / split
          </text>
        {entries.map((entry, idx) => {
          const y = topPad + idx * rowHeight + (rowHeight - barHeight) / 2;
          const trainStart = entry.trainStart != null ? scaleX(entry.trainStart) : null;
          const trainEnd = entry.trainEnd != null ? scaleX(entry.trainEnd) : null;
          const valStart = entry.valStart != null ? scaleX(entry.valStart) : null;
          const valEnd = entry.valEnd != null ? scaleX(entry.valEnd) : null;
          return (
            <g key={`fold-${idx}`}>
              <text x={leftPad - 10} y={y + 11} textAnchor="end" className="artifact-axis-label">
                F{entry.fold || idx + 1}
              </text>
              {trainStart != null && trainEnd != null ? (
                <rect x={trainStart} y={y} width={Math.max(1, trainEnd - trainStart)} height={barHeight} className="artifact-bar-train" />
              ) : null}
              {valStart != null && valEnd != null ? (
                <rect x={valStart} y={y} width={Math.max(1, valEnd - valStart)} height={barHeight} className="artifact-bar-val" />
              ) : null}
            </g>
          );
        })}
        {globalRows.map((row, idx) => {
          const y = topPad + (entries.length + idx) * rowHeight + (rowHeight - barHeight) / 2;
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
              <text x={leftPad - 10} y={y + 11} textAnchor="end" className="artifact-axis-label">{row.label}</text>
              <rect x={start} y={y} width={Math.max(1, end - start)} height={barHeight} className={barClass} />
            </g>
          );
        })}
        </svg>
      </div>
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
            <span className={`artifact-kv-value ${toNumber(value) && Number(value) > 0 ? "artifact-warn" : ""}`.trim()}>
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
  const xColumn = parsed.headers.find(
    (header) => header !== metricColumn && /(date|week|window|end|start|split|fold|time)/i.test(header),
  );
  const points = parsed.rows
    .map((row, idx) => ({
      x: idx,
      y: toNumber(row[metricColumn]),
      label: xColumn ? String(row[xColumn] ?? "") : String(idx + 1),
    }))
    .filter((point): point is { x: number; y: number; label: string } => point.y != null);
  if (!points.length) return <div className="empty">No rolling data.</div>;
  const min = Math.min(...points.map((point) => point.y));
  const max = Math.max(...points.map((point) => point.y));
  const yMin = min === max ? min - 0.01 : min;
  const yMax = min === max ? max + 0.01 : max;
  const width = DEFAULT_CHART_WIDTH;
  const height = 320;
  const leftPad = 76;
  const rightPad = 28;
  const topPad = 24;
  const bottomPad = 68;
  const plotWidth = width - leftPad - rightPad;
  const plotHeight = height - topPad - bottomPad;
  const scaleX = (x: number) => leftPad + (x / Math.max(1, points.length - 1)) * plotWidth;
  const scaleY = (y: number) => topPad + (1 - (y - yMin) / Math.max(1e-9, yMax - yMin)) * plotHeight;
  const yTicks = buildLinearTicks(yMin, yMax, 5);
  const xTickIndexes = buildIndexTicks(points.length, 6);
  const path = points
    .map((point, idx) => `${idx === 0 ? "M" : "L"} ${scaleX(point.x)} ${scaleY(point.y)}`)
    .join(" ");
  return (
    <div className="artifact-stack">
      <span className="meta-label">Rolling summary ({metricColumn})</span>
      <div className="artifact-chart-panel">
        <svg viewBox={`0 0 ${width} ${height}`} className="artifact-chart">
          <rect x={0} y={0} width={width} height={height} className="chart-frame" />
          {yTicks.map((tick) => {
            const y = scaleY(tick);
            return (
              <g key={`rolling-y-${tick}`}>
                <line x1={leftPad} x2={width - rightPad} y1={y} y2={y} className="artifact-grid-line" />
                <text x={leftPad - 10} y={y + 4} textAnchor="end" className="artifact-tick-label">
                  {formatChartNumber(tick)}
                </text>
              </g>
            );
          })}
          {xTickIndexes.map((tickIndex) => {
            const point = points[tickIndex];
            const x = scaleX(point.x);
            const maybeTime = parseChartDateLabel(point.label);
            return (
              <g key={`rolling-x-${tickIndex}`}>
                <line x1={x} x2={x} y1={topPad} y2={height - bottomPad} className="artifact-grid-line" />
                <text x={x} y={height - bottomPad + 22} textAnchor="middle" className="artifact-tick-label">
                  {maybeTime == null ? truncateAxisLabel(point.label) : formatChartDate(maybeTime)}
                </text>
              </g>
            );
          })}
          <line x1={leftPad} x2={width - rightPad} y1={height - bottomPad} y2={height - bottomPad} className="artifact-axis-line" />
          <line x1={leftPad} x2={leftPad} y1={topPad} y2={height - bottomPad} className="artifact-axis-line" />
          <path d={path} className="chart-line chart-line-prn" />
          <text x={(leftPad + width - rightPad) / 2} y={height - 14} textAnchor="middle" className="artifact-axis-title">
            {xColumn ? humanizeLabel(xColumn) : "Window index"}
          </text>
          <text
            x={22}
            y={topPad + plotHeight / 2}
            transform={`rotate(-90 22 ${topPad + plotHeight / 2})`}
            textAnchor="middle"
            className="artifact-axis-title"
          >
            {humanizeLabel(metricColumn)}
          </text>
        </svg>
      </div>
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
  const width = DEFAULT_CHART_WIDTH;
  const height = 340;
  const leftPad = 76;
  const rightPad = 28;
  const topPad = 24;
  const bottomPad = 74;
  const innerWidth = width - leftPad - rightPad;
  const innerHeight = height - topPad - bottomPad;
  const scaleY = (y: number) => topPad + (1 - (y - min) / Math.max(1e-9, max - min)) * innerHeight;
  const zeroY = scaleY(0);
  const step = innerWidth / Math.max(1, rows.length);
  const barWidth = Math.max(8, step * 0.6);
  const yTicks = buildLinearTicks(min, max, 5);
  const xTickIndexes = buildIndexTicks(rows.length, 8);
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
      <div className="artifact-chart-panel">
        <svg viewBox={`0 0 ${width} ${height}`} className="artifact-chart">
          <rect x={0} y={0} width={width} height={height} className="chart-frame" />
          {yTicks.map((tick) => {
            const y = scaleY(tick);
            return (
              <g key={`fold-y-${tick}`}>
                <line x1={leftPad} x2={width - rightPad} y1={y} y2={y} className="artifact-grid-line" />
                <text x={leftPad - 10} y={y + 4} textAnchor="end" className="artifact-tick-label">
                  {formatChartNumber(tick)}
                </text>
              </g>
            );
          })}
          {xTickIndexes.map((tickIndex) => {
            const row = rows[tickIndex];
            const x = leftPad + tickIndex * step + step / 2;
            return (
              <text key={`fold-x-${tickIndex}`} x={x} y={height - bottomPad + 22} textAnchor="middle" className="artifact-tick-label">
                F{row.fold || tickIndex + 1}
              </text>
            );
          })}
          <line x1={leftPad} x2={width - rightPad} y1={height - bottomPad} y2={height - bottomPad} className="artifact-axis-line" />
          <line x1={leftPad} x2={leftPad} y1={topPad} y2={height - bottomPad} className="artifact-axis-line" />
          <line x1={leftPad} x2={width - rightPad} y1={zeroY} y2={zeroY} className="chart-midline" />
          {rows.map((row, idx) => {
            const x = leftPad + idx * step + (step - barWidth) / 2;
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
          <text x={(leftPad + width - rightPad) / 2} y={height - 14} textAnchor="middle" className="artifact-axis-title">
            Fold
          </text>
          <text
            x={22}
            y={topPad + innerHeight / 2}
            transform={`rotate(-90 22 ${topPad + innerHeight / 2})`}
            textAnchor="middle"
            className="artifact-axis-title"
          >
            {humanizeLabel(metric)}
          </text>
        </svg>
      </div>
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
  const width = DEFAULT_CHART_WIDTH;
  const height = 340;
  const leftPad = 76;
  const rightPad = 28;
  const topPad = 24;
  const bottomPad = 72;
  const plotWidth = width - leftPad - rightPad;
  const plotHeight = height - topPad - bottomPad;
  const barGap = Math.max(2, plotWidth / bins * 0.08);
  const barWidth = (plotWidth - barGap * (bins - 1)) / bins;
  const yScale = (count: number) => topPad + (1 - count / Math.max(1, maxCount)) * plotHeight;
  const yTicks = buildLinearTicks(0, maxCount, 5);
  const xTicks = buildLinearTicks(min, max, 5);
  const xRange = Math.max(max - min, 1e-9);
  return (
    <div className="artifact-stack">
      <span className="meta-label">Group delta distribution</span>
      <div className="artifact-chart-panel">
        <svg viewBox={`0 0 ${width} ${height}`} className="artifact-chart">
          <rect x={0} y={0} width={width} height={height} className="chart-frame" />
          {yTicks.map((tick) => {
            const y = yScale(tick);
            return (
              <g key={`group-y-${tick}`}>
                <line x1={leftPad} x2={width - rightPad} y1={y} y2={y} className="artifact-grid-line" />
                <text x={leftPad - 10} y={y + 4} textAnchor="end" className="artifact-tick-label">
                  {Math.round(tick)}
                </text>
              </g>
            );
          })}
          {xTicks.map((tick) => {
            const x = leftPad + ((tick - min) / xRange) * plotWidth;
            return (
              <text key={`group-x-${tick}`} x={x} y={height - bottomPad + 22} textAnchor="middle" className="artifact-tick-label">
                {formatChartNumber(tick)}
              </text>
            );
          })}
          <line x1={leftPad} x2={width - rightPad} y1={height - bottomPad} y2={height - bottomPad} className="artifact-axis-line" />
          <line x1={leftPad} x2={leftPad} y1={topPad} y2={height - bottomPad} className="artifact-axis-line" />
          {counts.map((count, idx) => {
            const x = leftPad + idx * (barWidth + barGap);
            const y = yScale(count);
            return (
              <rect
                key={`group-bar-${idx}`}
                x={x}
                y={y}
                width={Math.max(1, barWidth)}
                height={Math.max(1, height - bottomPad - y)}
                className="artifact-bar-neutral"
              />
            );
          })}
          <text x={(leftPad + width - rightPad) / 2} y={height - 14} textAnchor="middle" className="artifact-axis-title">
            Delta logloss
          </text>
          <text
            x={22}
            y={topPad + plotHeight / 2}
            transform={`rotate(-90 22 ${topPad + plotHeight / 2})`}
            textAnchor="middle"
            className="artifact-axis-title"
          >
            Group count
          </text>
        </svg>
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
  const width = DEFAULT_CHART_WIDTH;
  const height = 340;
  const leftPad = 76;
  const rightPad = 28;
  const topPad = 24;
  const bottomPad = 68;
  const plotWidth = width - leftPad - rightPad;
  const plotHeight = height - topPad - bottomPad;
  const xTicks = buildLinearTicks(0, 1, 5);
  const scaleX = (v: number) => leftPad + v * plotWidth;
  const scaleY = (v: number) => topPad + (1 - v) * plotHeight;
  const path = points.map((p, idx) => `${idx === 0 ? "M" : "L"} ${scaleX(p.x)} ${scaleY(p.y)}`).join(" ");
  return (
    <div className="artifact-stack">
      <span className="meta-label">Reliability plot</span>
      <div className="artifact-chart-panel">
        <svg viewBox={`0 0 ${width} ${height}`} className="artifact-chart">
          <rect x={0} y={0} width={width} height={height} className="chart-frame" />
          {xTicks.map((tick) => {
            const x = scaleX(tick);
            const y = scaleY(tick);
            return (
              <g key={`reliability-tick-${tick}`}>
                <line x1={x} x2={x} y1={topPad} y2={height - bottomPad} className="artifact-grid-line" />
                <line x1={leftPad} x2={width - rightPad} y1={y} y2={y} className="artifact-grid-line" />
                <text x={x} y={height - bottomPad + 22} textAnchor="middle" className="artifact-tick-label">
                  {tick.toFixed(2)}
                </text>
                <text x={leftPad - 10} y={y + 4} textAnchor="end" className="artifact-tick-label">
                  {tick.toFixed(2)}
                </text>
              </g>
            );
          })}
          <line x1={leftPad} x2={width - rightPad} y1={height - bottomPad} y2={height - bottomPad} className="artifact-axis-line" />
          <line x1={leftPad} x2={leftPad} y1={topPad} y2={height - bottomPad} className="artifact-axis-line" />
          <line x1={leftPad} y1={height - bottomPad} x2={width - rightPad} y2={topPad} className="chart-midline" />
          <path d={path} className="chart-line chart-line-prn" />
          {points.map((point, idx) => (
            <circle key={`reliability-point-${idx}`} cx={scaleX(point.x)} cy={scaleY(point.y)} r={4} className="artifact-chart-point" />
          ))}
          <text x={(leftPad + width - rightPad) / 2} y={height - 14} textAnchor="middle" className="artifact-axis-title">
            {humanizeLabel(predKey)}
          </text>
          <text
            x={22}
            y={topPad + plotHeight / 2}
            transform={`rotate(-90 22 ${topPad + plotHeight / 2})`}
            textAnchor="middle"
            className="artifact-axis-title"
          >
            {humanizeLabel(obsKey)}
          </text>
        </svg>
      </div>
      <CsvTableView parsed={parsed} limit={50} />
    </div>
  );
};

const deltaMetricClass = (value?: number | null): string | undefined => {
  if (value == null || Number.isNaN(value)) return undefined;
  return value <= 0 ? "delta-negative" : "delta-positive";
};

const autoStatusPillLabel = (status?: string | null): string => {
  if (status === "selected") return "accepted";
  if (status === "no_viable_model") return "rejected";
  return status ? humanizeLabel(status) : "--";
};

const describeAutoSelection = ({
  status,
  selectedTrialId,
  hasSelectedModel,
}: {
  status?: string | null;
  selectedTrialId?: number | null;
  hasSelectedModel?: boolean | null;
}): string => {
  if (status === "selected") {
    return selectedTrialId != null ? `Selected (trial ${selectedTrialId})` : "Selected";
  }
  if (status === "no_viable_model") {
    if (hasSelectedModel) {
      return selectedTrialId != null
        ? `Best candidate materialized (trial ${selectedTrialId}), not accepted`
        : "Best candidate materialized, not accepted";
    }
    return "No candidate passed acceptance gates";
  }
  if (hasSelectedModel) {
    return selectedTrialId != null ? `Materialized candidate (trial ${selectedTrialId})` : "Materialized candidate";
  }
  return "No selected model";
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

const loadStoredForm = (): { form: CalibrateFormState | null; notice: string | null } => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { form: null, notice: null };
    const parsed = JSON.parse(raw) as Partial<CalibrateFormState>;
    const sanitizedSelections = stripRetiredFeatureSelections({
      selectedFeatures: Array.isArray(parsed.selectedFeatures)
        ? parsed.selectedFeatures.filter((feature): feature is string => typeof feature === "string")
        : [...DEFAULT_SELECTED_FEATURES],
      selectedCategoricalFeatures: Array.isArray(parsed.selectedCategoricalFeatures)
        ? parsed.selectedCategoricalFeatures.filter(
            (feature): feature is string => typeof feature === "string",
          )
        : [],
    });
    return {
      form: {
        ...defaultForm(),
        ...parsed,
        selectedFeatures: sanitizedSelections.selectedFeatures,
        selectedCategoricalFeatures: sanitizedSelections.selectedCategoricalFeatures,
      },
      notice: buildRetiredFeatureNotice(
        sanitizedSelections.removedFeatures,
        "Stored selection",
      ),
    };
  } catch {
    return { form: null, notice: null };
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

const normalizeFeatureSelection = (
  features: string[],
  availableSet: Set<string>,
  orderIndex: Map<string, number>,
  mutexGroupByFeature: Map<string, string> = new Map(),
): string[] => {
  const deduped = Array.from(new Set(features.filter((feature) => availableSet.has(feature))));
  const activeMutexSelections = new Map<string, string>();
  deduped.forEach((feature) => {
    const mutexGroup = mutexGroupByFeature.get(feature);
    if (mutexGroup) {
      activeMutexSelections.set(mutexGroup, feature);
    }
  });
  return deduped
    .filter((feature) => {
      const mutexGroup = mutexGroupByFeature.get(feature);
      return !mutexGroup || activeMutexSelections.get(mutexGroup) === feature;
    })
    .sort(
    (left, right) =>
      (orderIndex.get(left) ?? Number.MAX_SAFE_INTEGER) -
      (orderIndex.get(right) ?? Number.MAX_SAFE_INTEGER),
  );
};

const normalizeCategoricalSelection = (
  features: string[],
  availableSet: Set<string>,
  orderIndex: Map<string, number>,
): string[] => {
  return Array.from(new Set(features.filter((feature) => availableSet.has(feature)))).sort(
    (left, right) =>
      (orderIndex.get(left) ?? Number.MAX_SAFE_INTEGER) -
      (orderIndex.get(right) ?? Number.MAX_SAFE_INTEGER),
  );
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
  const location = useLocation();
  const navigate = useNavigate();
  const requestedWorkspaceTab = useMemo(
    () => parseWorkspaceTab(location.search),
    [location.search],
  );
  const [workspaceTab, setWorkspaceTab] = useState<WorkspaceTab>(requestedWorkspaceTab);
  const [runJobPanel, setRunJobPanel] = useState<RunJobPanel>("configuration");
  const [activeLog, setActiveLog] = useState<"stdout" | "stderr">("stdout");

  const [storedFormState] = useState(() => loadStoredForm());
  const [pendingRunAgainConfig, setPendingRunAgainConfig] = useState<Record<string, unknown> | null>(
    () => loadPendingRunAgainConfig(),
  );
  const [form, setForm] = useState<CalibrateFormState>(() => storedFormState.form ?? defaultForm());
  const [datasets, setDatasets] = useState<DatasetFileSummary[]>([]);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [cancelError, setCancelError] = useState<string | null>(null);
  const [cancelLoading, setCancelLoading] = useState(false);
  const [guardrailWarning, setGuardrailWarning] = useState<string | null>(null);
  const [lastRunResult, setLastRunResult] = useState<CalibrateModelRunResponse | null>(() => loadStoredResult());
  const [regimePreview, setRegimePreview] = useState<RegimePreviewResponse | null>(null);
  const [regimePreviewError, setRegimePreviewError] = useState<string | null>(null);
  const [, setAvailableFeatureColumns] = useState<string[]>([]);
  const [selectableFeatures, setSelectableFeatures] = useState<SelectableFeatureDescriptor[]>([]);
  const [featureError, setFeatureError] = useState<string | null>(null);
  const [featureRetirementNotice, setFeatureRetirementNotice] = useState<string | null>(
    () => storedFormState.notice,
  );
  const [featuresLoading, setFeaturesLoading] = useState(false);

  const [availableTickers, setAvailableTickers] = useState<string[]>([]);
  const [tickersLoading, setTickersLoading] = useState(false);
  const [tickersError, setTickersError] = useState<string | null>(null);

  const [weightingPreview, setWeightingPreview] = useState<WeightingPreviewResponse | null>(null);
  const [weightingPreviewError, setWeightingPreviewError] = useState<string | null>(null);
  const [weightingPreviewLoading, setWeightingPreviewLoading] = useState(false);

  const [models, setModels] = useState<ModelRunSummary[]>([]);
  const [modelError, setModelError] = useState<string | null>(null);
  const [lastRunCiLevel, setLastRunCiLevel] = useState<90 | 95 | 99 | null>(null);
  const [splitRecommended, setSplitRecommended] = useState(false);
  const [splitRecommendationWarning, setSplitRecommendationWarning] = useState<string | null>(null);
  const lastRecommendedRef = useRef<RecommendedSplitFields | null>(null);
  const lastModelRefreshRef = useRef<string | null>(null);

  const { jobId, jobStatus, setJobId, setJobStatus } = useCalibrationJob();
  const { anyJobRunning, primaryJob, activeJobs } = useAnyJobRunning();

  const handleWorkspaceTabChange = useCallback((nextTab: WorkspaceTab) => {
    setWorkspaceTab(nextTab);
    const nextSearch = nextTab === "models" ? "?tab=models" : "";
    const nextHref = `/calibrate${nextSearch}`;
    const currentHref = `${location.pathname}${location.search}`;
    if (currentHref !== nextHref) {
      navigate(nextHref, { replace: true });
    }
  }, [location.pathname, location.search, navigate]);

  const selectedDataset = useMemo(
    () => datasets.find((item) => item.path === form.datasetPath),
    [datasets, form.datasetPath],
  );
  const selectedDatasetPath = selectedDataset?.path ?? "";
  const availableWeeks = useMemo(
    () => computeAvailableWeeks(selectedDataset ?? null),
    [selectedDataset],
  );
  const selectedTimeRegime = useMemo(
    () => getTimeRegime(form.timeRegime),
    [form.timeRegime],
  );
  const isAuto = form.runMode === "auto";
  const selectableNumericFeatures = useMemo(
    () =>
      selectableFeatures
        .filter((feature) => feature.kind === "numeric")
        .slice()
        .sort((left, right) => left.order - right.order),
    [selectableFeatures],
  );
  const selectableCategoricalFeatures = useMemo(
    () =>
      selectableFeatures
        .filter((feature) => feature.kind === "categorical")
        .slice()
        .sort((left, right) => left.order - right.order),
    [selectableFeatures],
  );
  const featureOrderIndex = useMemo(
    () => new Map<string, number>(selectableNumericFeatures.map((feature, index) => [feature.name, index])),
    [selectableNumericFeatures],
  );
  const categoricalFeatureOrderIndex = useMemo(
    () =>
      new Map<string, number>(
        selectableCategoricalFeatures.map((feature, index) => [feature.name, index]),
      ),
    [selectableCategoricalFeatures],
  );
  const selectableFeatureSet = useMemo(
    () => new Set<string>(selectableNumericFeatures.map((feature) => feature.name)),
    [selectableNumericFeatures],
  );
  const selectableCategoricalSet = useMemo(
    () => new Set<string>(selectableCategoricalFeatures.map((feature) => feature.name)),
    [selectableCategoricalFeatures],
  );
  const featureCategoryOptions = useMemo(() => {
    const grouped = new Map<string, SelectableFeatureDescriptor[]>();
    selectableNumericFeatures.forEach((feature) => {
      const bucket = grouped.get(feature.group);
      if (bucket) {
        bucket.push(feature);
      } else {
        grouped.set(feature.group, [feature]);
      }
    });
    return Array.from(grouped.entries()).map(([title, items]) => ({ title, items }));
  }, [selectableNumericFeatures]);
  const mutexGroupByFeature = useMemo(() => {
    const groups = new Map<string, string>();
    selectableNumericFeatures.forEach((feature) => {
      if (feature.mutex_group) {
        groups.set(feature.name, feature.mutex_group);
      }
    });
    return groups;
  }, [selectableNumericFeatures]);
  const defaultSelectableFeatures = useMemo(() => {
    const metadataDefaults = selectableNumericFeatures
      .filter((feature) => feature.default_selected)
      .map((feature) => feature.name);
    const defaults = metadataDefaults.length ? metadataDefaults : DEFAULT_SELECTED_FEATURES;
    return normalizeFeatureSelection(
      defaults,
      selectableFeatureSet,
      featureOrderIndex,
      mutexGroupByFeature,
    );
  }, [featureOrderIndex, mutexGroupByFeature, selectableFeatureSet, selectableNumericFeatures]);

  const canRecommendSplit = Boolean(selectedDataset && availableWeeks != null && availableWeeks > 0);
  const hasValidModelDirName =
    form.modelDirName.trim().length === 0 || Boolean(sanitizeModelDirName(form.modelDirName));
  const basicSettingsReady = hasValidModelDirName && Boolean(selectedDataset);

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
    const status: "running" | "failed" | "completed" =
      jobStatus?.status === "failed"
        ? "failed"
        : jobStatus?.status === "finished"
          ? "completed"
          : "running";
    return { total, completed, failed, status };
  }, [jobStatus?.progress, jobStatus?.status]);

  const metricsSummary = activeResult?.metrics_summary ?? null;
  const activeStataDiagnostics = activeResult?.stata_diagnostics ?? null;
  const activeDiagnosticsSkipWarning = findDiagnosticsSkipWarning(activeResult?.warnings ?? null);
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

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(form));
    } catch {
      // ignore
    }
  }, [form]);

  useEffect(() => {
    if (workspaceTab !== requestedWorkspaceTab) {
      setWorkspaceTab(requestedWorkspaceTab);
    }
  }, [requestedWorkspaceTab, workspaceTab]);

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
        const paths = new Set(response.datasets.map((d) => d.path));
        const hasCurrentSelection = form.datasetPath && paths.has(form.datasetPath);
        if (!hasCurrentSelection) {
          setForm((prev) => {
            const nextDatasetPath = response.datasets[0]?.path ?? "";
            if (prev.datasetPath === nextDatasetPath) {
              return prev;
            }
            return { ...prev, datasetPath: nextDatasetPath };
          });
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
    if (!selectedDatasetPath) {
      setAvailableTickers([]);
      setTickersError(null);
      return;
    }
    let cancelled = false;
    setTickersLoading(true);
    fetchDatasetTickers(selectedDatasetPath)
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
  }, [selectedDatasetPath]);

  useEffect(() => {
    if (!selectedDatasetPath) {
      setAvailableFeatureColumns([]);
      setSelectableFeatures([]);
      setFeatureError(null);
      return;
    }
    let cancelled = false;
    setFeaturesLoading(true);
    fetchDatasetFeatures(selectedDatasetPath)
      .then((response) => {
        if (cancelled) return;
        const columns = response.available_columns ?? [];
        const nextSelectableFeatures = response.selectable_features ?? [];
        setAvailableFeatureColumns(columns);
        setSelectableFeatures(nextSelectableFeatures);
        setFeatureError(null);
        const available = new Set(
          nextSelectableFeatures
            .filter((feature) => feature.kind === "numeric")
            .map((feature) => feature.name),
        );
        const availableCategorical = new Set(
          nextSelectableFeatures
            .filter((feature) => feature.kind === "categorical")
            .map((feature) => feature.name),
        );
        const numericOrder = new Map(
          nextSelectableFeatures
            .filter((feature) => feature.kind === "numeric")
            .sort((left, right) => left.order - right.order)
            .map((feature, index) => [feature.name, index] as const),
        );
        const categoricalOrder = new Map(
          nextSelectableFeatures
            .filter((feature) => feature.kind === "categorical")
            .sort((left, right) => left.order - right.order)
            .map((feature, index) => [feature.name, index] as const),
        );
        setForm((prev) => ({
          ...prev,
          selectedFeatures: normalizeFeatureSelection(
            prev.selectedFeatures,
            available,
            numericOrder,
            new Map(
              nextSelectableFeatures
                .filter((feature) => feature.kind === "numeric" && feature.mutex_group)
                .map((feature) => [feature.name, feature.mutex_group!] as const),
            ),
          ),
          selectedCategoricalFeatures: normalizeCategoricalSelection(
            prev.selectedCategoricalFeatures,
            availableCategorical,
            categoricalOrder,
          ),
        }));
      })
      .catch((error: Error) => {
        if (cancelled) return;
        setFeatureError(error.message);
        setAvailableFeatureColumns([]);
        setSelectableFeatures([]);
      })
      .finally(() => {
        if (!cancelled) {
          setFeaturesLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [selectedDatasetPath]);

  useEffect(() => {
    if (selectableFeatures.length === 0) {
      return;
    }
    setForm((prev) => {
      const nextSelectedFeatures = normalizeFeatureSelection(
        prev.selectedFeatures,
        selectableFeatureSet,
        featureOrderIndex,
        mutexGroupByFeature,
      );
      const nextSelectedCategorical = normalizeCategoricalSelection(
        prev.selectedCategoricalFeatures,
        selectableCategoricalSet,
        categoricalFeatureOrderIndex,
      );
      const featuresUnchanged =
        nextSelectedFeatures.length === prev.selectedFeatures.length &&
        nextSelectedFeatures.every((feature, index) => feature === prev.selectedFeatures[index]);
      const categoricalUnchanged =
        nextSelectedCategorical.length === prev.selectedCategoricalFeatures.length &&
        nextSelectedCategorical.every(
          (feature, index) => feature === prev.selectedCategoricalFeatures[index],
        );
      if (featuresUnchanged && categoricalUnchanged) {
        return prev;
      }
      return {
        ...prev,
        selectedFeatures: nextSelectedFeatures,
        selectedCategoricalFeatures: nextSelectedCategorical,
      };
    });
  }, [
    categoricalFeatureOrderIndex,
    featureOrderIndex,
    mutexGroupByFeature,
    selectableFeatures.length,
    selectableCategoricalSet,
    selectableFeatureSet,
  ]);

  useEffect(() => {
    if (!selectedDatasetPath) {
      setRegimePreview(null);
      setRegimePreviewError(null);
      return;
    }

    const timeout = window.setTimeout(() => {
      previewCalibrationRegime({
        csv: selectedDatasetPath,
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
  }, [selectedDatasetPath, selectedTimeRegime.asofDow, selectedTimeRegime.tdays]);

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
      selectedFeatures: defaultSelectableFeatures,
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
      bootstrapGroup: resolveBootstrapGroupValue(
        "contract_id",
        selectedDataset?.available_grouping_keys,
      ),
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
  }, [availableTickers, defaultSelectableFeatures, selectedDataset?.available_grouping_keys]);

  const validateForm = useCallback((): string | null => {
    if (!selectedDataset) {
      return "Select a training dataset.";
    }
    if (featuresLoading) {
      return "Feature metadata is still loading.";
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
      return `Remove invalid foundation tickers: ${invalidFoundation.join(", ")}.`;
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

    const normalizedFeatures = normalizeFeatureSelection(
      form.selectedFeatures,
      selectableFeatureSet,
      featureOrderIndex,
      mutexGroupByFeature,
    );
    const invalidFeatures = form.selectedFeatures.filter(
      (feature) => !normalizedFeatures.includes(feature),
    );
    if (invalidFeatures.length) {
      return `Selected optional features not available: ${invalidFeatures.join(", ")}.`;
    }

    const normalizedCategorical = normalizeCategoricalSelection(
      form.selectedCategoricalFeatures,
      selectableCategoricalSet,
      categoricalFeatureOrderIndex,
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
  }, [
    availableWeeks,
    categoricalFeatureOrderIndex,
    featureOrderIndex,
    featuresLoading,
    form,
    mutexGroupByFeature,
    selectableCategoricalSet,
    selectableFeatureSet,
    selectedDataset,
  ]);

  useEffect(() => {
    const issue = selectedDataset ? validateForm() : null;
    setGuardrailWarning(issue);
  }, [selectedDataset, validateForm]);

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
    const normalizedSelectedFeatures = normalizeFeatureSelection(
      form.selectedFeatures,
      selectableFeatureSet,
      featureOrderIndex,
      mutexGroupByFeature,
    );
    const normalizedCategorical = normalizeCategoricalSelection(
      form.selectedCategoricalFeatures,
      selectableCategoricalSet,
      categoricalFeatureOrderIndex,
    );
    const features = joinCsv([BASE_FEATURE, ...normalizedSelectedFeatures]);
    const categoricalFeatures = joinCsv(normalizedCategorical);
    const resolvedBootstrapGroup = form.bootstrapEnabled
      ? resolveBootstrapGroupValue(form.bootstrapGroup, selectedDataset?.available_grouping_keys)
      : form.bootstrapGroup;

    return {
      csv: selectedDatasetPath,
      outName: sanitizeModelDirName(form.modelDirName || defaultModelName()),
      runMode: "manual" as const,
      targetCol: DEFAULT_TARGET_COL,
      weekCol: DEFAULT_WEEK_COL,
      tickerCol: DEFAULT_TICKER_COL,
      weightCol: selectedWeightCol,
      weightColStrategy: form.weightColStrategy,
      features,
      categoricalFeatures,
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
      bootstrapGroup: form.bootstrapEnabled ? resolvedBootstrapGroup : undefined,
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
        bootstrapGroup: resolvedBootstrapGroup,
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
  }, [
    categoricalFeatureOrderIndex,
    featureOrderIndex,
    form,
    mutexGroupByFeature,
    selectableCategoricalSet,
    selectableFeatureSet,
    selectedDataset?.available_grouping_keys,
    selectedDatasetPath,
    selectedTimeRegime.asofDow,
    selectedTimeRegime.tdays,
  ]);

  const selectedGroupingKeys = selectedDataset?.available_grouping_keys ?? [];
  const resolveBootstrapGroup = useCallback(
    (requested: BootstrapGroupMode): BootstrapGroupMode =>
      resolveBootstrapGroupValue(requested, selectedGroupingKeys),
    [selectedGroupingKeys],
  );

  useEffect(() => {
    const resolved = resolveBootstrapGroup(form.bootstrapGroup);
    if (resolved !== form.bootstrapGroup) {
      setForm((prev) => ({ ...prev, bootstrapGroup: resolved }));
    }
  }, [form.bootstrapGroup, resolveBootstrapGroup]);

  const buildAutoPayload = useCallback((): AutoModelRunRequest => {
    const baseConfig = buildManualPayload();
    const autoBootstrapGroup =
      form.bootstrapEnabled && baseConfig.bootstrapGroup
        ? resolveBootstrapGroup(baseConfig.bootstrapGroup)
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
    resolveBootstrapGroup,
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
    if (!selectedDatasetPath) {
      setWeightingPreview(null);
      setWeightingPreviewError(null);
      return;
    }

    setWeightingPreviewLoading(true);
    setWeightingPreviewError(null);
    try {
      const response = await previewCalibrationWeighting({
        csv: selectedDatasetPath,
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
  }, [form, selectedDatasetPath]);

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

  const handleRunAgainFromConfig = useCallback((config: Record<string, unknown>) => {
    const split =
      config.split && typeof config.split === "object"
        ? (config.split as Record<string, unknown>)
        : {};
    const regularization =
      config.regularization && typeof config.regularization === "object"
        ? (config.regularization as Record<string, unknown>)
        : {};
    const modelStructure =
      config.model_structure && typeof config.model_structure === "object"
        ? (config.model_structure as Record<string, unknown>)
        : {};
    const weighting =
      config.weighting && typeof config.weighting === "object"
        ? (config.weighting as Record<string, unknown>)
        : {};
    const bootstrap =
      config.bootstrap && typeof config.bootstrap === "object"
        ? (config.bootstrap as Record<string, unknown>)
        : {};
    const diagnostics =
      config.diagnostics && typeof config.diagnostics === "object"
        ? (config.diagnostics as Record<string, unknown>)
        : {};
    const csv = toProjectRelativePath(config.csv);
    const matchedDataset =
      datasets.find((dataset) => {
        const relative = dataset.path.replace(/\\/g, "/");
        return csv === relative || csv.endsWith(relative);
      })?.path ?? csv;
    const retiredFeatureSelections = stripRetiredFeatureSelections({
      selectedFeatures: splitCsvValue(config.features).filter((feature) => feature !== BASE_FEATURE),
      selectedCategoricalFeatures: splitCsvValue(config.categorical_features),
    });
    const features = retiredFeatureSelections.selectedFeatures;
    const categorical = retiredFeatureSelections.selectedCategoricalFeatures;
    const cGrid =
      typeof regularization.c_grid === "string"
        ? regularization.c_grid
        : typeof config.c_grid === "string"
          ? config.c_grid
          : "";
    const cGridPreset =
      (Object.entries(C_GRID_PRESETS).find(([, preset]) => preset === cGrid)?.[0] as CalibrateFormState["cGridPreset"] | undefined) ??
      "custom";
    const ciLevelValue = toNumber(bootstrap.ci_level);
    const nextModelName =
      sanitizeModelDirName(defaultModelName()) || defaultModelName();
    const shouldSanitizeLoadedFeatures =
      matchedDataset === selectedDatasetPath && selectableFeatures.length > 0;
    setFeatureRetirementNotice(
      buildRetiredFeatureNotice(retiredFeatureSelections.removedFeatures, "Loaded config"),
    );

    setForm((prev) => ({
      ...prev,
      runMode: "manual",
      modelDirName: nextModelName,
      datasetPath: matchedDataset,
      randomSeed: String(toNumber(config.random_state) ?? toNumber(config.randomSeed) ?? 7),
      weightColStrategy:
        config.weight_col_strategy === "auto" ||
        config.weight_col_strategy === "weight_final" ||
        config.weight_col_strategy === "sample_weight_final" ||
        config.weight_col_strategy === "uniform"
          ? config.weight_col_strategy
          : prev.weightColStrategy,
      timeRegime: resolveTimeRegimeKey(config),
      selectedFeatures: shouldSanitizeLoadedFeatures
        ? normalizeFeatureSelection(
            features,
            selectableFeatureSet,
            featureOrderIndex,
            mutexGroupByFeature,
          )
        : features,
      selectedCategoricalFeatures: shouldSanitizeLoadedFeatures
        ? normalizeCategoricalSelection(
            categorical,
            selectableCategoricalSet,
            categoricalFeatureOrderIndex,
          )
        : categorical,
      splitStrategy: split.strategy === "single_holdout" ? "single_holdout" : "walk_forward",
      windowMode: split.window_mode === "expanding" ? "expanding" : "rolling",
      trainWindowWeeks: String(toNumber(split.train_window_weeks) ?? prev.trainWindowWeeks),
      validationFolds: String(toNumber(split.validation_folds) ?? prev.validationFolds),
      validationWindowWeeks: String(toNumber(split.validation_window_weeks) ?? prev.validationWindowWeeks),
      testWindowWeeks: String(toNumber(split.test_window_weeks) ?? toNumber(config.test_weeks) ?? prev.testWindowWeeks),
      embargoDays: String(toNumber(split.embargo_days) ?? prev.embargoDays),
      cGridPreset,
      cGridCustom: cGrid || prev.cGridCustom,
      calibrationMethod:
        regularization.calibration_method === "platt" || config.calibrate === "platt" ? "platt" : "none",
      selectionObjective:
        regularization.selection_objective === "brier" || regularization.selection_objective === "ece_q"
          ? regularization.selection_objective
          : regularization.selection_objective === "logloss"
            ? "logloss"
            : prev.selectionObjective,
      tradingUniverseTickers: splitCsvValue(
        modelStructure.trading_universe_tickers ??
        weighting.trading_universe_tickers ??
        modelStructure.train_tickers ??
        config.train_tickers,
      ),
      trainTickers: splitCsvValue(modelStructure.train_tickers ?? config.train_tickers),
      foundationTickers: splitCsvValue(modelStructure.foundation_tickers ?? config.foundation_tickers),
      foundationWeight: String(toNumber(modelStructure.foundation_weight ?? config.foundation_weight) ?? prev.foundationWeight),
      tickerInterceptMode:
        modelStructure.ticker_intercepts === "none" ||
        modelStructure.ticker_intercepts === "all" ||
        modelStructure.ticker_intercepts === "non_foundation"
          ? modelStructure.ticker_intercepts
          : prev.tickerInterceptMode,
      perTickerInteractions: Boolean(modelStructure.ticker_x_interactions ?? config.ticker_x_interactions),
      minSupportIntercepts: String(toNumber(modelStructure.ticker_min_support ?? config.ticker_min_support) ?? prev.minSupportIntercepts),
      minSupportInteractions: String(
        toNumber(modelStructure.ticker_min_support_interactions ?? config.ticker_min_support_interactions) ??
        prev.minSupportInteractions,
      ),
      baseWeightSource: weighting.base_weight_source === "uniform" ? "uniform" : "dataset_weight",
      groupingKey: typeof weighting.grouping_key === "string" ? weighting.grouping_key : prev.groupingKey,
      groupEqualization:
        typeof weighting.group_equalization === "boolean"
          ? weighting.group_equalization
          : (weighting.group_reweight ?? config.group_reweight) === "chain_snapshot",
      renorm: "mean1",
      tradingUniverseUpweight: String(toNumber(weighting.trading_universe_upweight) ?? prev.tradingUniverseUpweight),
      tickerBalanceMode: weighting.ticker_balance_mode === "sqrt_inv_clipped" ? "sqrt_inv_clipped" : "none",
      bootstrapEnabled: Boolean(bootstrap.bootstrap_ci ?? config.bootstrap_ci),
      bootstrapGroup:
        bootstrap.bootstrap_group === "contract_id" ||
        bootstrap.bootstrap_group === "group_id" ||
        bootstrap.bootstrap_group === "ticker_day" ||
        bootstrap.bootstrap_group === "day" ||
        bootstrap.bootstrap_group === "iid" ||
        bootstrap.bootstrap_group === "auto"
          ? bootstrap.bootstrap_group
          : prev.bootstrapGroup,
      bootstrapDraws: String(toNumber(bootstrap.bootstrap_b ?? bootstrap.bootstrap_B ?? config.bootstrap_B) ?? prev.bootstrapDraws),
      bootstrapSeed: String(toNumber(bootstrap.bootstrap_seed ?? config.bootstrap_seed) ?? prev.bootstrapSeed),
      ciLevel: ciLevelValue === 90 || ciLevelValue === 99 ? (ciLevelValue as 90 | 99) : 95,
      perSplitReporting: Boolean(bootstrap.per_split_reporting ?? prev.perSplitReporting),
      perFoldReporting: Boolean(bootstrap.per_fold_reporting ?? prev.perFoldReporting),
      splitTimeline: Boolean(diagnostics.split_timeline ?? prev.splitTimeline),
      perFoldDeltaChart: Boolean(diagnostics.per_fold_delta_chart ?? prev.perFoldDeltaChart),
      perGroupDeltaDistribution: Boolean(diagnostics.per_group_delta_distribution ?? prev.perGroupDeltaDistribution),
      maxAbsLogm: String(toNumber(config.max_abs_logm) ?? prev.maxAbsLogm),
      dropPrnExtremes: Boolean(config.drop_prn_extremes),
      dropPrnBelow: String(toNumber(config.prn_below) ?? prev.dropPrnBelow),
      dropPrnAbove: String(toNumber(config.prn_above) ?? prev.dropPrnAbove),
    }));
    setRunError(null);
    setCancelError(null);
    handleWorkspaceTabChange("run_job");
    setRunJobPanel("configuration");
  }, [
    categoricalFeatureOrderIndex,
    datasets,
    featureOrderIndex,
    handleWorkspaceTabChange,
    mutexGroupByFeature,
    selectableCategoricalSet,
    selectableFeatureSet,
    selectableFeatures.length,
    selectedDatasetPath,
  ]);

  useEffect(() => {
    if (!pendingRunAgainConfig || datasets.length === 0) return;
    handleRunAgainFromConfig(pendingRunAgainConfig);
    clearPendingRunAgainConfig();
    setPendingRunAgainConfig(null);
  }, [datasets.length, handleRunAgainFromConfig, pendingRunAgainConfig]);

  const selectedFeatureList = useMemo(
    () =>
      normalizeFeatureSelection(
        form.selectedFeatures,
        selectableFeatureSet,
        featureOrderIndex,
        mutexGroupByFeature,
      ),
    [featureOrderIndex, form.selectedFeatures, mutexGroupByFeature, selectableFeatureSet],
  );

  const selectedCategoricalList = useMemo(
    () =>
      normalizeCategoricalSelection(
        form.selectedCategoricalFeatures,
        selectableCategoricalSet,
        categoricalFeatureOrderIndex,
      ),
    [categoricalFeatureOrderIndex, form.selectedCategoricalFeatures, selectableCategoricalSet],
  );

  const handleToggleFeature = useCallback((feature: string) => {
    if (!selectableFeatureSet.has(feature)) return;
    setFeatureRetirementNotice(null);
    setForm((prev) => {
      const next = normalizeFeatureSelection(
        prev.selectedFeatures,
        selectableFeatureSet,
        featureOrderIndex,
        mutexGroupByFeature,
      );
      if (next.includes(feature)) {
        return {
          ...prev,
          selectedFeatures: next.filter((candidate) => candidate !== feature),
        };
      }
      const mutexGroup = mutexGroupByFeature.get(feature);
      const withoutConflicts = mutexGroup
        ? next.filter((candidate) => mutexGroupByFeature.get(candidate) !== mutexGroup)
        : next;
      return {
        ...prev,
        selectedFeatures: normalizeFeatureSelection(
          [...withoutConflicts, feature],
          selectableFeatureSet,
          featureOrderIndex,
          mutexGroupByFeature,
        ),
      };
    });
  }, [featureOrderIndex, mutexGroupByFeature, selectableFeatureSet]);

  const handleToggleCategoricalFeature = useCallback((feature: string) => {
    if (!selectableCategoricalSet.has(feature)) return;
    setFeatureRetirementNotice(null);
    setForm((prev) => {
      const next = new Set(
        normalizeCategoricalSelection(
          prev.selectedCategoricalFeatures,
          selectableCategoricalSet,
          categoricalFeatureOrderIndex,
        ),
      );
      if (next.has(feature)) {
        next.delete(feature);
      } else {
        next.add(feature);
      }
      return {
        ...prev,
        selectedCategoricalFeatures: normalizeCategoricalSelection(
          Array.from(next),
          selectableCategoricalSet,
          categoricalFeatureOrderIndex,
        ),
      };
    });
  }, [categoricalFeatureOrderIndex, selectableCategoricalSet]);

  const handleSelectRecommendedFeatures = useCallback(() => {
    setFeatureRetirementNotice(null);
    setForm((prev) => ({
      ...prev,
      selectedFeatures: defaultSelectableFeatures,
      selectedCategoricalFeatures: normalizeCategoricalSelection(
        prev.selectedCategoricalFeatures,
        selectableCategoricalSet,
        categoricalFeatureOrderIndex,
      ),
    }));
  }, [categoricalFeatureOrderIndex, defaultSelectableFeatures, selectableCategoricalSet]);

  const handleClearOptionalFeatures = useCallback(() => {
    setFeatureRetirementNotice(null);
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
            onClick={() => handleWorkspaceTabChange("run_job")}
          >
            Run Job
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={workspaceTab === "models"}
            className={`calibrate-workspace-tab ${workspaceTab === "models" ? "active" : ""}`}
            onClick={() => handleWorkspaceTabChange("models")}
          >
            Models
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
                        </div>
                      </div>

                      <div className="dataset-summary-grid calibrate-info-card-grid calibrate-info-card-grid-wide">
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
                        <div className="dataset-summary-grid calibrate-info-card-grid">
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

                    {basicSettingsReady ? (
                      <>
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
                                  checked={form.autoAdvancedSearch}
                                  onChange={(event) =>
                                    setForm((prev) => ({ ...prev, autoAdvancedSearch: event.target.checked }))
                                  }
                                />
                                Advanced search (ticker interactions)
                              </label>
                            </div>
                            <div className="dataset-summary-grid calibrate-info-card-grid">
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
                            {featuresLoading ? <div className="empty">Loading feature options…</div> : null}
                            {featureError ? <div className="error">{featureError}</div> : null}
                            {featureRetirementNotice ? <div className="warning">{featureRetirementNotice}</div> : null}
                            <div className="feature-category-grid">
                              {featureCategoryOptions.map((category) => (
                                <div key={category.title} className="feature-category-card">
                                  <h4 className="feature-category-title">{category.title}</h4>
                                  <div className="feature-chip-grid">
                                    {category.items.map((feature) => {
                                      const selected = selectedFeatureList.includes(feature.name);
                                      const title = feature.label || feature.name;
                                      return (
                                        <button
                                          key={feature.name}
                                          type="button"
                                          className={`feature-chip ${selected ? "selected" : ""}`}
                                          onClick={() => handleToggleFeature(feature.name)}
                                          title={title}
                                        >
                                          {feature.name}
                                        </button>
                                      );
                                    })}
                                  </div>
                                </div>
                              ))}
                              {selectableCategoricalFeatures.length ? (
                                <div className="feature-category-card">
                                  <h4 className="feature-category-title">Categorical</h4>
                                  <div className="feature-chip-grid">
                                    {selectableCategoricalFeatures.map((feature) => {
                                      const selected = selectedCategoricalList.includes(feature.name);
                                      return (
                                        <button
                                          key={feature.name}
                                          type="button"
                                          className={`feature-chip ${selected ? "selected" : ""}`}
                                          onClick={() => handleToggleCategoricalFeature(feature.name)}
                                          title={feature.label || feature.name}
                                        >
                                          {feature.name}
                                        </button>
                                      );
                                    })}
                                  </div>
                                </div>
                              ) : null}
                            </div>
                            <div className="dataset-summary-grid calibrate-info-card-grid">
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
                                <span className="meta-label">Available optional</span>
                                <span>{selectableNumericFeatures.length}</span>
                              </div>
                            </div>
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
                      </>
                    ) : null}

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
                      <div className="calibrate-active-summary">
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
                      </div>

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

                        {isJobComplete && !activeStataDiagnostics && activeDiagnosticsSkipWarning ? (
                          <div className="warning">{activeDiagnosticsSkipWarning}</div>
                        ) : null}

                        {isJobComplete && metricsSummary ? (
                          <div className="metrics-summary">
                            <div className="metrics-summary-header">
                              <span className="meta-label">Model performance</span>
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

                        {isJobComplete && activeResult?.artifact_manifest?.some(
                          (artifact) => !isHiddenArtifactPath(artifact.relative_path ?? artifact.path ?? artifact.name),
                        ) ? (
                          <div className="model-detail-section file-viewer-section">
                            <span className="meta-label">Artifacts</span>
                            <div className="file-list">
                              {activeResult.artifact_manifest
                                .filter((artifact) => !isHiddenArtifactPath(artifact.relative_path ?? artifact.path ?? artifact.name))
                                .map((artifact) => {
                                  const artifactPath = artifact.relative_path ?? artifact.path ?? artifact.name;
                                  return (
                                    <ArtifactFileButton
                                      key={artifactPath}
                                      titlePath={artifactPath}
                                      displayPath={artifactPath}
                                      meta={artifact.type}
                                      onClick={() => {
                                        if (!activeResult.out_dir) return;
                                        const modelId = activeResult.out_dir.split("/").pop() || "";
                                        if (modelId) {
                                          navigate(buildCalibrationModelDetailHref(modelId));
                                        }
                                      }}
                                    />
                                  );
                                })}
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
                  <div className="model-directory-list model-directory-list-standalone">
                    {models.map((model) => {
                      const isAutoRun = model.run_type === "auto";
                      const autoStatus = model.auto_status ?? (model.has_selected_model ? "selected" : null);
                      return (
                        <article key={model.id} className="model-list-item">
                          <button
                            type="button"
                            className="model-list-button"
                            onClick={() => navigate(buildCalibrationModelDetailHref(model.id))}
                          >
                            <div className="model-list-heading">
                              <span className="model-list-title">{model.id}</span>
                              <span className="model-list-updated">{formatTimestamp(model.last_modified)}</span>
                            </div>
                            <div className="model-list-badges">
                              <span className={`status-pill run-type-pill ${isAutoRun ? "running" : "idle"}`}>
                                {isAutoRun ? "AUTO" : "MANUAL"}
                              </span>
                              <span className={`status-pill ${model.has_metrics ? "success" : "idle"}`}>
                                {model.has_metrics ? "metrics" : "no metrics"}
                              </span>
                              {model.is_two_stage ? <span className="status-pill running">two-stage</span> : null}
                              {isAutoRun && autoStatus ? (
                                <span
                                  className={`status-pill ${
                                    autoStatus === "selected" ? "success" : autoStatus === "no_viable_model" ? "failed" : "idle"
                                  }`}
                                >
                                  {autoStatusPillLabel(autoStatus)}
                                </span>
                              ) : null}
                            </div>
                            <div className="model-list-meta">{formatModelListMeta(model)}</div>
                          </button>
                        </article>
                      );
                    })}
                  </div>
                )}
              </div>
            </section>
          </div>
        ) : null}

      </div>

    </section>
  );
}

export function CalibrationModelDetailPage() {
  const { modelId: routeModelId } = useParams();
  const navigate = useNavigate();
  const resolvedModelId = useMemo(() => {
    if (!routeModelId) return "";
    try {
      return decodeURIComponent(routeModelId);
    } catch {
      return routeModelId;
    }
  }, [routeModelId]);

  const [models, setModels] = useState<ModelRunSummary[]>([]);
  const [modelListError, setModelListError] = useState<string | null>(null);
  const [isModelDetailLoading, setIsModelDetailLoading] = useState(false);
  const [modelDetail, setModelDetail] = useState<ModelDetailResponse | null>(null);
  const [modelDetailError, setModelDetailError] = useState<string | null>(null);
  const [modelFiles, setModelFiles] = useState<ModelFilesListResponse | null>(null);
  const [selectedFilePath, setSelectedFilePath] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<ModelFileContentResponse | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [fileLoading, setFileLoading] = useState(false);
  const [showRawFile, setShowRawFile] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [deleteConfirmText, setDeleteConfirmText] = useState("");
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);

  const refreshModels = useCallback(async () => {
    try {
      const response = await fetchCalibrationModels();
      setModels(response.models);
      setModelListError(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to load models.";
      setModelListError(message);
    }
  }, []);

  useEffect(() => {
    void refreshModels();
  }, [refreshModels]);

  const selectedModel = useMemo(
    () => models.find((model) => model.id === resolvedModelId) ?? null,
    [models, resolvedModelId],
  );

  const loadModelFileContent = useCallback(async (file: ModelFileSummary) => {
    if (!resolvedModelId) {
      throw new Error("Missing model id.");
    }
    return file.relative_path
      ? fetchModelFileContentByPath(resolvedModelId, file.relative_path)
      : fetchModelFileContent(resolvedModelId, file.name);
  }, [resolvedModelId]);

  const openModelFile = useCallback(async (
    file: ModelFileSummary,
    options?: { allowToggle?: boolean },
  ) => {
    const targetPath = artifactFilePath(file);
    if (!targetPath) return;
    const allowToggle = options?.allowToggle ?? true;
    if (allowToggle && selectedFilePath === targetPath) {
      setSelectedFilePath(null);
      setFileContent(null);
      setFileError(null);
      return;
    }
    setSelectedFilePath(targetPath);
    setFileLoading(true);
    setFileError(null);
    setFileContent(null);
    try {
      const content = await loadModelFileContent(file);
      setFileContent(content);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to read file.";
      setFileError(message);
      setFileContent(null);
    } finally {
      setFileLoading(false);
    }
  }, [loadModelFileContent, selectedFilePath]);

  useEffect(() => {
    setShowRawFile(false);
  }, [selectedFilePath]);

  useEffect(() => {
    if (!resolvedModelId) {
      setModelDetailError("Missing model id.");
      setModelDetail(null);
      setModelFiles(null);
      return;
    }

    let cancelled = false;
    const load = async () => {
      setIsModelDetailLoading(true);
      setModelDetailError(null);
      setActionError(null);
      setSelectedFilePath(null);
      setFileContent(null);
      setFileError(null);
      try {
        const [detail, files] = await Promise.all([
          fetchCalibrationModelDetail(resolvedModelId),
          fetchModelFiles(resolvedModelId),
        ]);
        if (cancelled) return;
        setModelDetail(detail);
        setModelFiles(files);
      } catch (error) {
        if (cancelled) return;
        const message = error instanceof Error ? error.message : "Failed to load model detail.";
        setModelDetailError(message);
        setModelDetail(null);
        setModelFiles(null);
        setSelectedFilePath(null);
        setFileContent(null);
      } finally {
        if (!cancelled) {
          setIsModelDetailLoading(false);
        }
      }
    };

    void load();
    return () => {
      cancelled = true;
    };
  }, [resolvedModelId]);

  const modelInspectorArtifacts = useMemo(
    () => buildModelInspectorArtifactState(modelFiles?.files),
    [modelFiles],
  );
  const artifactSections = modelInspectorArtifacts.sections;
  const selectedArtifactName = fileBaseName(selectedFilePath);

  const handleQueueRunAgainFromConfig = useCallback((config: Record<string, unknown>) => {
    storePendingRunAgainConfig(config);
    navigate(buildCalibrateTabHref("run_job"));
  }, [navigate]);

  const renderArtifactView = useCallback(() => {
    if (!selectedFilePath || !fileContent) return null;
    if (showRawFile) {
      return <pre className="file-content">{fileContent.content}</pre>;
    }
    const parsedJson = fileContent.content_type === "json" ? parseJsonContent(fileContent.content) : null;
    const parsedCsv = fileContent.content_type === "csv" ? parseCsvContent(fileContent.content) : null;
    const selectedName = fileBaseName(fileContent.relative_path ?? selectedFilePath);

    switch (selectedName) {
      case "diagnostics_table.json":
        return modelDetail?.stata_diagnostics && isStataDiagnosticsPayload(modelDetail.stata_diagnostics)
          ? (
            <StataDiagnosticsPanel
              diagnostics={modelDetail.stata_diagnostics}
              showHeaderStrip={false}
              showCalibrationChart={false}
              showProductionCoefficientTable={false}
            />
          )
          : parsedJson && isStataDiagnosticsPayload(parsedJson)
            ? (
              <StataDiagnosticsPanel
                diagnostics={parsedJson as StataDiagnosticsPayload}
                showHeaderStrip={false}
                showCalibrationChart={false}
                showProductionCoefficientTable={false}
              />
            )
            : <div className="empty">No diagnostics table data.</div>;
      case "metrics.csv":
        return parsedCsv ? <MetricsCsvView parsed={parsedCsv} ciLabel={modelCiLabel} /> : <div className="empty">No metrics data.</div>;
      case "metrics_summary.json":
        return parsedJson ? <KeyValueGrid data={parsedJson} /> : <div className="empty">No summary data.</div>;
      case "coefficient_diagnostics.csv":
      case "marginal_effects.csv":
        return parsedCsv ? <CsvTableView parsed={parsedCsv} limit={100} /> : <div className="empty">No table data.</div>;
      case "config.executed.json":
      case "best_config.json":
        return parsedJson ? (
          <ConfigJsonView
            data={parsedJson}
            title={artifactDisplayTitle(selectedName)}
            onRunAgain={() => handleQueueRunAgainFromConfig(parsedJson)}
          />
        ) : <div className="empty">No config data.</div>;
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
      case "auto_search_progress.json":
        return parsedJson ? <KeyValueGrid data={parsedJson} /> : <div className="empty">No progress data.</div>;
      case "best_model_report.md":
        return <ReportMarkdownView content={fileContent.content} />;
      default:
        if (parsedJson) return <KeyValueGrid data={parsedJson} />;
        if (parsedCsv) return <CsvTableView parsed={parsedCsv} limit={50} />;
        return <pre className="file-content">{fileContent.content}</pre>;
    }
  }, [fileContent, handleQueueRunAgainFromConfig, modelDetail?.stata_diagnostics, selectedFilePath, showRawFile]);

  const handleOpenFile = useCallback(async (file: ModelFileSummary) => {
    await openModelFile(file, { allowToggle: false });
  }, [openModelFile]);

  const handleRenameModel = useCallback(async () => {
    const next = window.prompt("New model name", resolvedModelId);
    if (!next) return;
    const cleaned = sanitizeModelDirName(next);
    if (!cleaned) {
      setActionError("Invalid model name.");
      return;
    }
    try {
      await renameCalibrationModel(resolvedModelId, cleaned);
      await refreshModels();
      navigate(buildCalibrationModelDetailHref(cleaned), { replace: true });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Rename failed.";
      setActionError(message);
    }
  }, [navigate, refreshModels, resolvedModelId]);

  const closeDeleteModal = useCallback(() => {
    if (deleteLoading) return;
    setShowDeleteModal(false);
    setDeleteConfirmText("");
  }, [deleteLoading]);

  const handleDeleteModel = useCallback(async () => {
    if (deleteConfirmText !== "DELETE") return;
    setDeleteLoading(true);
    try {
      await deleteCalibrationModel(resolvedModelId);
      navigate(buildCalibrateTabHref("models"), { replace: true });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Delete failed.";
      setActionError(message);
    } finally {
      setDeleteLoading(false);
    }
  }, [deleteConfirmText, navigate, resolvedModelId]);

  const modelDetailCiLevel = toNumber(
    (modelDetail?.metadata as Record<string, unknown> | null | undefined)?.["ci_level"],
  );
  const modelCiLabel = modelDetailCiLevel ? `CI (${modelDetailCiLevel}%)` : "CI";
  const modelDetailWarnings = coerceWarningList(
    (modelDetail?.metadata as Record<string, unknown> | null | undefined)?.["warnings"],
  );
  const modelDetailDiagnosticsSkipWarning = findDiagnosticsSkipWarning(modelDetailWarnings);
  const autoSelectionSummary = modelDetail?.auto_selection_summary ?? null;

  const modelRows = modelDetail?.split_row_counts ?? {};
  const modelGroups = modelDetail?.split_group_counts ?? {};
  const modelTrainRows = modelRows.train_fit ?? modelRows.train ?? null;
  const modelValRows = modelRows.val ?? null;
  const modelTestRows = modelRows.test ?? null;
  const modelTrainGroups = modelGroups.train_fit ?? modelGroups.train ?? null;
  const modelValGroups = modelGroups.val ?? null;
  const modelTestGroups = modelGroups.test ?? null;
  const hasCoverage =
    modelTrainRows != null ||
    modelValRows != null ||
    modelTestRows != null ||
    modelTrainGroups != null ||
    modelValGroups != null ||
    modelTestGroups != null;

  const isAutoRun = selectedModel?.run_type === "auto";
  const autoStatus = selectedModel?.auto_status ?? (selectedModel?.has_selected_model ? "selected" : null);
  const autoSelectionText = describeAutoSelection({
    status: autoStatus,
    selectedTrialId: selectedModel?.selected_trial_id,
    hasSelectedModel: selectedModel?.has_selected_model,
  });
  const overviewItems = [
    {
      label: "Dataset",
      value: selectedModel?.dataset_id ?? "--",
    },
    {
      label: "Updated",
      value: formatTimestamp(selectedModel?.last_modified ?? modelDetail?.last_modified),
    },
    {
      label: "Range",
      value: selectedModel ? formatModelRange(selectedModel.train_date_start, selectedModel.train_date_end) : "--",
    },
    {
      label: "Hyperparams",
      value: compactMetaLine(
        `split=${selectedModel?.split_strategy ?? "--"}`,
        `C=${selectedModel?.c_value != null ? formatMaybe(selectedModel.c_value) : "--"}`,
        `calib=${selectedModel?.calibration_method ?? "--"}`,
      ) || "--",
    },
    {
      label: "Coverage",
      value: hasCoverage
        ? compactMetaLine(
          `train ${formatCountValue(modelTrainRows as number | null)}`,
          `val ${formatCountValue(modelValRows as number | null)}`,
          `test ${formatCountValue(modelTestRows as number | null)}`,
        )
        : "--",
      note: hasCoverage
        ? compactMetaLine(
          `groups ${formatCountValue(modelTrainGroups as number | null)}/${formatCountValue(modelValGroups as number | null)}/${formatCountValue(modelTestGroups as number | null)}`,
        )
        : null,
    },
    {
      label: "Universe",
      value: selectedModel?.tickers_summary ?? selectedModel?.dow_regime ?? "--",
    },
    isAutoRun ? {
      label: "Auto selection",
      value: autoSelectionText,
      note: compactMetaLine(
        autoSelectionSummary?.selection_rule ? `rule=${autoSelectionSummary.selection_rule}` : null,
        autoSelectionSummary?.best_score != null ? `score=${formatMetricValue(autoSelectionSummary.best_score)}` : null,
      ) || null,
    } : null,
  ].filter(Boolean);

  const hasDiagnostics = Boolean(
    modelDetail?.stata_diagnostics || modelDetail?.metrics_summary || modelDetailDiagnosticsSkipWarning,
  );
  const hasEquations = Boolean(
    modelDetail?.model_equation
    || modelDetail?.stage1_equation
    || modelDetail?.two_stage_equation
    || modelDetail?.combined_p_hat_equation,
  );
  const productionCalibrationCurve = modelDetail?.stata_diagnostics?.calibration_curve ?? null;
  const isDiagnosticsArtifactSelected = selectedArtifactName === "diagnostics_table.json";

  return (
    <section className="page calibration-model-detail-page">
      <header className="page-header model-detail-page-header">
        <div className="model-detail-page-title-block">
          <Link className="button light small model-detail-back-button" to={buildCalibrateTabHref("models")}>
            Back to Model Directory
          </Link>
          <p className="page-kicker">Calibrate Model</p>
          <h1 className="page-title model-detail-page-title">{(modelDetail?.id ?? resolvedModelId) || "Model"}</h1>
          <p className="page-subtitle model-detail-page-subtitle">
            {compactMetaLine(
              selectedModel?.dataset_id ?? null,
              selectedModel?.split_strategy ? `split=${selectedModel.split_strategy}` : null,
              selectedModel?.c_value != null ? `C=${formatMaybe(selectedModel.c_value)}` : null,
              selectedModel?.calibration_method ? `calib=${selectedModel.calibration_method}` : null,
            ) || "Production performance, equations, and artifacts for the selected model."}
          </p>
        </div>
        <div className="model-detail-page-actions">
          {selectedModel ? (
            <>
              <span className={`status-pill run-type-pill ${isAutoRun ? "running" : "idle"}`}>
                {isAutoRun ? "AUTO" : "MANUAL"}
              </span>
              <span className={`status-pill ${(selectedModel.has_metrics || modelDetail?.has_metrics) ? "success" : "idle"}`}>
                {(selectedModel.has_metrics || modelDetail?.has_metrics) ? "metrics" : "no metrics"}
              </span>
              {selectedModel.is_two_stage ? <span className="status-pill running">two-stage</span> : null}
              {isAutoRun && autoStatus ? (
                <span
                  className={`status-pill ${
                    autoStatus === "selected" ? "success" : autoStatus === "no_viable_model" ? "failed" : "idle"
                  }`}
                >
                  {autoStatusPillLabel(autoStatus)}
                </span>
              ) : null}
            </>
          ) : null}
          <button
            className="button light small"
            type="button"
            onClick={() => void handleRenameModel()}
            disabled={!resolvedModelId}
          >
            Rename
          </button>
          <button
            className="button ghost danger small"
            type="button"
            onClick={() => setShowDeleteModal(true)}
            disabled={!resolvedModelId}
          >
            Delete
          </button>
        </div>
      </header>

      {actionError ? <div className="error">{actionError}</div> : null}
      {modelListError ? <div className="warning">{modelListError}</div> : null}

      {!resolvedModelId ? (
        <div className="empty model-detail-empty-state">
          <div>Missing model id.</div>
          <Link className="button light small" to={buildCalibrateTabHref("models")}>
            Back to Model Directory
          </Link>
        </div>
      ) : isModelDetailLoading ? (
        <div className="empty model-detail-empty-state">Loading model detail…</div>
      ) : modelDetailError ? (
        <div className="model-detail-error-shell">
          <div className="error">{modelDetailError}</div>
          <Link className="button light small" to={buildCalibrateTabHref("models")}>
            Back to Model Directory
          </Link>
        </div>
      ) : modelDetail ? (
        <div className="model-detail-stack">
          <section className="model-detail-section model-detail-performance-panel">
            <div className="model-detail-section-header">
              <span className="meta-label">Overview</span>
            </div>

            <div className="model-overview-grid model-detail-overview-grid">
              {overviewItems.map((item) => (
                <div key={item!.label} className="model-overview-item">
                  <span className="meta-label">{item!.label}</span>
                  <strong className="model-overview-value">{item!.value}</strong>
                  {item!.note ? <span className="model-overview-note">{item!.note}</span> : null}
                </div>
              ))}
            </div>

            {!modelDetail.stata_diagnostics && modelDetailDiagnosticsSkipWarning ? (
              <div className="warning auto-no-viable-callout">{modelDetailDiagnosticsSkipWarning}</div>
            ) : null}

            {isAutoRun && autoStatus === "no_viable_model" && !selectedModel?.has_selected_model ? (
              <div className="warning auto-no-viable-callout">
                No candidate passed the acceptance gates for this auto run. Search diagnostics remain available below.
              </div>
            ) : null}
          </section>

          <section className="model-detail-section model-detail-performance-panel">
            <div className="model-detail-section-header">
              <span className="meta-label">Production model performance</span>
            </div>

            {modelDetail.metrics_summary ? (
              <div className="metrics-summary-grid model-detail-metrics-grid">
                {metricsOrder
                  .map((split) => modelDetail.metrics_summary?.[split])
                  .filter(Boolean)
                  .map((metric) => (
                    <div key={`${resolvedModelId}-${metric!.split}`} className="metrics-card">
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
            ) : (
              <div className="empty">No production metrics available.</div>
            )}

            {productionCalibrationCurve ? (
              <CalibrationCurveCard
                rows={productionCalibrationCurve.rows}
                split={productionCalibrationCurve.split}
                baselineLabel={modelDetail.stata_diagnostics?.header.baseline_name ?? "pRN"}
                subtitle={`${productionCalibrationCurve.split.toUpperCase()} equal-mass bins • production model vs ${modelDetail.stata_diagnostics?.header.baseline_name ?? "pRN"}`}
              />
            ) : hasDiagnostics ? (
              <div className="empty">Calibration curve unavailable for this model.</div>
            ) : null}
          </section>

          {hasEquations ? (
            <section className="model-detail-section model-detail-section-secondary">
              <div className="model-detail-section-header">
                <span className="meta-label">Model equations</span>
              </div>
              <div className="model-equation-stack">
                {modelDetail.model_equation ? (
                  <div className="equation-summary">
                    <span className="meta-label">Model equation</span>
                    <LatexBlock latex={modelDetail.model_equation} />
                    <EquationNotes spec={modelDetail.model_equation_spec} />
                  </div>
                ) : null}

                {modelDetail.stage1_equation ? (
                  <div className="equation-summary">
                    <span className="meta-label">Stage A equation</span>
                    <LatexBlock latex={modelDetail.stage1_equation} />
                    <EquationNotes spec={modelDetail.stage1_equation_spec} />
                  </div>
                ) : null}

                {modelDetail.two_stage_equation ? (
                  <div className="equation-summary">
                    <span className="meta-label">Stage B equation</span>
                    <LatexBlock latex={modelDetail.two_stage_equation} />
                    <EquationNotes spec={modelDetail.two_stage_equation_spec} />
                  </div>
                ) : null}

                {modelDetail.combined_p_hat_equation ? (
                  <div className="equation-summary">
                    <span className="meta-label">Combined p-hat</span>
                    <LatexBlock latex={modelDetail.combined_p_hat_equation} />
                    <EquationNotes spec={modelDetail.combined_p_hat_equation_spec} />
                  </div>
                ) : null}
              </div>
            </section>
          ) : null}

          <section className="model-detail-section model-detail-section-secondary">
            <div className="model-detail-section-header">
              <span className="meta-label">Artifacts</span>
            </div>
            {artifactSections.length ? (
              <div className="model-artifact-selection-stack">
                {artifactSections.map((section) => (
                  <div key={`${resolvedModelId}-${section.id}`} className="model-artifact-row-section">
                    <div className="model-artifact-nav-header">
                      <span className="meta-label">{section.title}</span>
                      <span className="model-artifact-nav-count">{section.files.length}</span>
                    </div>
                    <div className="model-artifact-row-list">
                      {section.files.map((file) => {
                        const filePath = artifactFilePath(file);
                        return (
                          <ArtifactFileButton
                            key={`${resolvedModelId}-${section.id}-${filePath}`}
                            titlePath={filePath}
                            displayPath={filePath}
                            meta={file.is_viewable ? formatFileSizeLabel(file.size_bytes) : "Unavailable"}
                            className="file-item-artifact-card"
                            isActive={selectedFilePath === filePath}
                            onClick={() => file.is_viewable && void handleOpenFile(file)}
                            disabled={!file.is_viewable}
                          />
                        );
                      })}
                    </div>
                  </div>
                ))}

                {isDiagnosticsArtifactSelected ? (
                  <div className="artifact-selection-note model-artifact-disclaimer">
                    {INFERRED_DIAGNOSTICS_DISCLAIMER}
                  </div>
                ) : null}

                <div className="model-artifact-preview">
                  {selectedFilePath ? (
                    <div className="file-content-panel model-artifact-preview-panel">
                      <div className="file-content-header">
                        <div className="file-content-title">
                          <span className="file-name">{artifactDisplayTitle(selectedFilePath)}</span>
                          <span className="file-content-path">{selectedFilePath}</span>
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
                  ) : (
                    <div className="empty model-artifact-empty">Select an artifact to preview.</div>
                  )}
                </div>
              </div>
            ) : (
              <div className="empty">No viewable artifacts found for this model.</div>
            )}
          </section>
        </div>
      ) : null}

      {showDeleteModal ? (
        <div className="calibrate-delete-modal-overlay" onClick={closeDeleteModal}>
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
                This permanently deletes <span className="calibrate-delete-modal-code">{resolvedModelId}</span> and its artifacts.
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
              <button className="button ghost" type="button" disabled={deleteLoading} onClick={closeDeleteModal}>
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
