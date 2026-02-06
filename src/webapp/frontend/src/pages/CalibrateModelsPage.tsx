import { Fragment, useEffect, useMemo, useState, type FormEvent } from "react";
import katex from "katex";

import {
  fetchCalibrationDatasets,
  fetchCalibrationModels,
  deleteCalibrationModel,
  fetchCalibrationModelDetail,
  fetchModelFiles,
  fetchModelFileContent,
  renameCalibrationModel,
  startCalibrationJob,
  startAutoCalibrationJob,
  type CalibrateModelRunResponse,
  type DatasetFileSummary,
  type ModelDetailResponse,
  type ModelFileContentResponse,
  type ModelFilesListResponse,
  type ModelRunSummary,
  previewCalibrationRegime,
  type RegimePreviewResponse,
} from "../api/calibrateModels";
import { useCalibrationJob } from "../contexts/calibrationJob";
import { useAnyJobRunning } from "../contexts/jobGuard";
import "./CalibrateModelsPage.css";
import "katex/dist/katex.min.css";

type CalibrateFormState = {
  datasetPath: string;
  outName: string;
  selectedFeatures: string[];
  customFeatures: string;
  categoricalFeatures: string;
  addInteractions: boolean;
  calibrate: "none" | "platt";
  cGrid: string;
  trainDecayHalfLifeWeeks: string;
  calibFracOfTrain: string;
  fitWeightRenorm: "none" | "mean1";
  testWeeks: string;
  valWindows: string;
  valWindowWeeks: string;
  nBins: string;
  randomState: string;
  foundationTickers: string;
  foundationWeight: string;
  mode: "baseline" | "pooled" | "two_stage";
  tickerIntercepts: "none" | "all" | "non_foundation";
  tickerXInteractions: boolean;
  tickerMinSupport: string;
  tickerMinSupportInteractions: string;
  trainTickers: string;
  metricsTopTickers: string;
  tdaysAllowed: string;
  asofDowAllowed: string;
  eceqBins: string;
  selectionObjective: "delta_vs_baseline";
  fallbackToBaselineIfWorse: boolean;
  autoDropNearConstant: boolean;
  enableXAbsM: boolean;
  groupReweight: "none" | "chain";
  maxAbsLogm: string;
  dropPrnExtremes: boolean;
  prnEps: string;
  bootstrapCi: boolean;
  bootstrapB: string;
  bootstrapSeed: string;
  bootstrapGroup: "auto" | "ticker_day" | "day" | "iid";
};

type FeatureDefinition = {
  key: string;
  label: string;
  description: string;
};

type FeatureGroup = {
  id: string;
  title: string;
  hint: string;
  keys: string[];
  exclusive?: boolean;
};

const featureCatalog: FeatureDefinition[] = [
  {
    key: "x_logit_prn",
    label: "pRN logit",
    description: "Logit transform of pRN (baseline signal strength).",
  },
  {
    key: "log_m_fwd",
    label: "Forward log-moneyness",
    description: "Forward log-moneyness at as-of date.",
  },
  {
    key: "abs_log_m_fwd",
    label: "Absolute forward log-moneyness",
    description: "Absolute forward log-moneyness (distance from ATM).",
  },
  {
    key: "T_days",
    label: "Horizon (days)",
    description: "Time to expiration in days.",
  },
  {
    key: "sqrt_T_years",
    label: "√Time (years)",
    description: "Square-root time to expiration in years.",
  },
  {
    key: "rv20",
    label: "20d realized vol",
    description: "20-day realized volatility proxy.",
  },
  {
    key: "rv20_sqrtT",
    label: "Vol × √T",
    description: "Volatility scaled by sqrt(T).",
  },
  {
    key: "log_m_fwd_over_volT",
    label: "Log m / (vol×T)",
    description: "Log-moneyness normalized by volatility and time.",
  },
  {
    key: "x_m",
    label: "pRN logit × moneyness",
    description: "Interaction: x_logit_prn × moneyness (log_m_fwd or log_m).",
  },
  {
    key: "x_abs_m",
    label: "pRN logit × |moneyness|",
    description: "Interaction: x_logit_prn × |moneyness| (absolute value).",
  },
  {
    key: "log_rel_spread",
    label: "Log relative spread",
    description: "Log of relative bid/ask spread.",
  },
  {
    key: "had_fallback",
    label: "Fallback indicator",
    description: "Binary flag for any as-of/expiry fallback.",
  },
  {
    key: "had_band_clip",
    label: "Band clip flag",
    description: "Binary flag when band coverage < 1.",
  },
  {
    key: "had_intrinsic_drop",
    label: "Intrinsic drop flag",
    description: "Binary flag for intrinsic value guardrails.",
  },
  {
    key: "prn_raw_gap",
    label: "pRN gap (raw)",
    description: "Gap between raw and adjusted pRN.",
  },
  {
    key: "dividend_yield",
    label: "Dividend yield",
    description: "Dividend yield proxy.",
  },
];

const FEATURE_GROUPS: FeatureGroup[] = [
  {
    id: "core",
    title: "Core signals",
    hint: "Baseline + microstructure signals that pair with most runs.",
    keys: [
      "x_logit_prn",
      "log_rel_spread",
      "had_fallback",
      "had_band_clip",
      "had_intrinsic_drop",
      "prn_raw_gap",
      "dividend_yield",
    ],
  },
  {
    id: "moneyness",
    title: "Moneyness transforms",
    hint: "Choose one to avoid redundant forward moneyness signals.",
    keys: ["log_m_fwd", "abs_log_m_fwd", "log_m_fwd_over_volT"],
    exclusive: true,
  },
  {
    id: "time",
    title: "Time horizon",
    hint: "Disabled for regime-specific training.",
    keys: ["T_days", "sqrt_T_years"],
  },
  {
    id: "volatility",
    title: "Volatility scaling",
    hint: "Choose one to avoid overlapping vol signals.",
    keys: ["rv20", "rv20_sqrtT"],
    exclusive: true,
  },
];

const FEATURE_INDEX = featureCatalog.reduce(
  (acc, feature) => {
    acc[feature.key] = feature;
    return acc;
  },
  {} as Record<string, FeatureDefinition>,
);

const EXCLUSIVE_GROUPS = FEATURE_GROUPS.filter((group) => group.exclusive);
const EXCLUSIVE_GROUPS_BY_ID = EXCLUSIVE_GROUPS.reduce(
  (acc, group) => {
    acc[group.id] = group.keys;
    return acc;
  },
  {} as Record<string, string[]>,
);
const EXCLUSIVE_GROUP_BY_KEY = EXCLUSIVE_GROUPS.reduce(
  (acc, group) => {
    group.keys.forEach((key) => {
      acc[key] = group.id;
    });
    return acc;
  },
  {} as Record<string, string>,
);

const REGIME_LOCKED_FEATURES = new Set(["T_days", "sqrt_T_years", "rv20_sqrtT"]);

const DEFAULT_FEATURE_SELECTION = [
  "x_logit_prn",
  "log_m_fwd",
  "T_days",
  "sqrt_T_years",
  "rv20",
  "log_rel_spread",
  "had_fallback",
  "had_band_clip",
  "had_intrinsic_drop",
  "prn_raw_gap",
  "dividend_yield",
];

const defaultFeatures = DEFAULT_FEATURE_SELECTION.join(",");

const DEFAULT_DATASET_COLUMNS = {
  target: "outcome_ST_gt_K",
  week: "week_friday",
  ticker: "ticker",
  weight: "sample_weight_final",
} as const;

const defaultModelName = () => {
  const stamp = new Date().toISOString().replace(/[:.]/g, "").slice(0, 15);
  return `calibration-${stamp}`;
};

const defaultAutoRunName = () => {
  const stamp = new Date().toISOString().replace(/[:.]/g, "").slice(0, 15);
  return `auto-run-${stamp}`;
};

const MODELS_BASE = "src/data/models";

type AutoFormState = {
  runName: string;
  maxTrials: string;
  seed: string;
  parallel: string;
  baselineArgs: string;
  objective: "logloss";
  foundationTickers: string;
  foundationWeight: string;
  bootstrapCi: boolean;
  bootstrapB: string;
  bootstrapSeed: string;
  bootstrapGroup: "auto" | "ticker_day" | "day" | "iid";
};

const buildDefaultAutoForm = (): AutoFormState => ({
  runName: defaultAutoRunName(),
  maxTrials: "",
  seed: "42",
  parallel: "1",
  baselineArgs: "",
  objective: "logloss",
  foundationTickers: "SPY,QQQ,IWM",
  foundationWeight: "1.0",
  bootstrapCi: false,
  bootstrapB: "2000",
  bootstrapSeed: "0",
  bootstrapGroup: "auto",
});

const defaultForm: CalibrateFormState = {
  datasetPath: "",
  outName: "",
  selectedFeatures: defaultFeatures.split(","),
  customFeatures: "",
  categoricalFeatures: "spot_scale_used",
  addInteractions: false,
  calibrate: "none",
  cGrid: "0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10",
  trainDecayHalfLifeWeeks: "0.0",
  calibFracOfTrain: "0.20",
  fitWeightRenorm: "mean1",
  testWeeks: "20",
  valWindows: "4",
  valWindowWeeks: "10",
  nBins: "15",
  randomState: "7",
  foundationTickers: "",
  foundationWeight: "1.0",
  mode: "pooled",
  tickerIntercepts: "non_foundation",
  tickerXInteractions: false,
  tickerMinSupport: "300",
  tickerMinSupportInteractions: "1000",
  trainTickers: "",
  metricsTopTickers: "10",
  tdaysAllowed: "",
  asofDowAllowed: "",
  eceqBins: "10",
  selectionObjective: "delta_vs_baseline",
  fallbackToBaselineIfWorse: true,
  autoDropNearConstant: true,
  enableXAbsM: false,
  groupReweight: "none",
  maxAbsLogm: "",
  dropPrnExtremes: false,
  prnEps: "0.0001",
  bootstrapCi: false,
  bootstrapB: "2000",
  bootstrapSeed: "0",
  bootstrapGroup: "auto",
};

const STORAGE_KEY = "polyedgetool.calibrate.form";
const RUN_RESULT_STORAGE_KEY = "polyedgetool.calibrate.lastRun";

const parseOptionalNumber = (value: string): number | undefined => {
  const trimmed = value.trim();
  if (!trimmed) return undefined;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : undefined;
};

const parseOptionalInt = (value: string): number | undefined => {
  const trimmed = value.trim();
  if (!trimmed) return undefined;
  const parsed = Number.parseInt(trimmed, 10);
  return Number.isFinite(parsed) ? parsed : undefined;
};

const loadStoredForm = (): Partial<CalibrateFormState> | null => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    return parsed as Partial<CalibrateFormState>;
  } catch {
    return null;
  }
};

const loadStoredRunResult = (): CalibrateModelRunResponse | null => {
  try {
    const raw = localStorage.getItem(RUN_RESULT_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    return parsed as CalibrateModelRunResponse;
  } catch {
    return null;
  }
};

const saveRunResult = (result: CalibrateModelRunResponse | null): void => {
  try {
    if (result === null) {
      localStorage.removeItem(RUN_RESULT_STORAGE_KEY);
    } else {
      localStorage.setItem(RUN_RESULT_STORAGE_KEY, JSON.stringify(result));
    }
  } catch {
    // ignore storage failures
  }
};

const formatTimestamp = (value?: string | null): string => {
  if (!value) return "Unknown time";
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

const formatBytes = (value: number): string => {
  if (value < 1024) return `${value} B`;
  const kb = value / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(1)} MB`;
};

const TDAYS_OPTIONS = [1, 2, 3, 4, 5, 7, 10, 14, 21, 30];
const REGIME_WARNING_MIN_ROWS = 200;

const formatParamValue = (value: unknown): string => {
  if (value == null) return "—";
  if (Array.isArray(value)) {
    if (value.length === 0) return "—";
    return value.map((item) => String(item)).join(", ");
  }
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "—";
  if (typeof value === "string") return value || "—";
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
};

const parseTdaysList = (value: string): number[] => {
  if (!value) return [];
  const tokens = value.split(",").map((token) => token.trim()).filter(Boolean);
  const parsed: number[] = [];
  for (const token of tokens) {
    const num = Number.parseInt(token, 10);
    if (!Number.isFinite(num) || num < 0) continue;
    if (!parsed.includes(num)) {
      parsed.push(num);
    }
  }
  return parsed;
};

const formatTdaysList = (values: number[]): string => {
  if (!values.length) return "";
  const unique = Array.from(new Set(values));
  unique.sort((a, b) => a - b);
  return unique.join(",");
};

const areArraysEqual = (left: string[], right: string[]): boolean => {
  if (left.length !== right.length) return false;
  return left.every((value, index) => value === right[index]);
};

const isRegimeSpecific = (state: Pick<CalibrateFormState, "tdaysAllowed" | "asofDowAllowed">): boolean =>
  Boolean(state.tdaysAllowed.trim() || state.asofDowAllowed.trim());

const applyFeatureConstraints = (
  selected: string[],
  regimeSpecific: boolean,
): string[] => {
  let next = Array.from(new Set(selected));
  if (regimeSpecific) {
    next = next.filter((feature) => !REGIME_LOCKED_FEATURES.has(feature));
  }
  EXCLUSIVE_GROUPS.forEach((group) => {
    let found = false;
    next = next.filter((feature) => {
      if (!group.keys.includes(feature)) return true;
      if (!found) {
        found = true;
        return true;
      }
      return false;
    });
  });
  return next;
};

const LatexEquation = ({ latex }: { latex: string }) => {
  const rendered = useMemo(() => {
    try {
      return katex.renderToString(latex, {
        throwOnError: false,
        displayMode: true,
      });
    } catch {
      return null;
    }
  }, [latex]);

  if (!rendered) {
    return (
      <div className="latex-equation latex-equation-fallback">
        {latex}
      </div>
    );
  }

  return (
    <div
      className="latex-equation"
      dangerouslySetInnerHTML={{ __html: rendered }}
    />
  );
};

const buildManualCommandPreview = (
  state: CalibrateFormState,
  defaultName: string,
): string => {
  const args: string[] = [
    "python",
    "src/scripts/2-calibrate-logit-model-v1.5.py",
  ];

  const addValue = (flag: string, value: string) => {
    const trimmed = value.trim();
    if (!trimmed) return;
    args.push(flag, trimmed);
  };
  const addFlag = (flag: string, enabled: boolean) => {
    if (enabled) args.push(flag);
  };

  const effectiveName = state.outName.trim() || defaultName;
  addValue("--csv", state.datasetPath);
  addValue("--out-dir", `src/data/models/${effectiveName}`);
  addValue("--target-col", DEFAULT_DATASET_COLUMNS.target);
  addValue("--week-col", DEFAULT_DATASET_COLUMNS.week);
  addValue("--ticker-col", DEFAULT_DATASET_COLUMNS.ticker);
  addValue("--weight-col", DEFAULT_DATASET_COLUMNS.weight);
  addValue("--foundation-tickers", state.foundationTickers);
  addValue("--foundation-weight", state.foundationWeight);
  addValue("--mode", state.mode);
  addValue("--ticker-intercepts", state.tickerIntercepts);
  addFlag("--ticker-x-interactions", state.tickerXInteractions);
  addValue("--ticker-min-support", state.tickerMinSupport);
  addValue("--ticker-min-support-interactions", state.tickerMinSupportInteractions);
  addValue("--train-tickers", state.trainTickers);
  addValue("--tdays-allowed", state.tdaysAllowed);
  addValue("--asof-dow-allowed", state.asofDowAllowed);
  const selected = applyFeatureConstraints(
    state.selectedFeatures,
    isRegimeSpecific(state),
  );
  const custom = state.customFeatures
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  const featureList = [...selected, ...custom].join(",");
  addValue("--features", featureList);
  addValue("--categorical-features", state.categoricalFeatures);
  addFlag("--add-interactions", state.addInteractions);
  addValue("--calibrate", state.calibrate);
  addValue("--C-grid", state.cGrid);
  addValue("--train-decay-half-life-weeks", state.trainDecayHalfLifeWeeks);
  addValue("--calib-frac-of-train", state.calibFracOfTrain);
  addValue("--fit-weight-renorm", state.fitWeightRenorm);
  addValue("--test-weeks", state.testWeeks);
  addValue("--val-windows", state.valWindows);
  addValue("--val-window-weeks", state.valWindowWeeks);
  addValue("--n-bins", state.nBins);
  addValue("--eceq-bins", state.eceqBins);
  addValue("--selection-objective", state.selectionObjective);
  addValue("--metrics-top-tickers", state.metricsTopTickers);
  addValue("--random-state", state.randomState);
  if (!state.fallbackToBaselineIfWorse) {
    args.push("--no-fallback-to-baseline-if-worse");
  }
  if (!state.autoDropNearConstant) {
    args.push("--no-auto-drop-near-constant");
  }
  addFlag("--enable-x-abs-m", state.enableXAbsM);
  if (state.groupReweight !== "none") {
    addValue("--group-reweight", state.groupReweight);
  }
  addValue("--max-abs-logm", state.maxAbsLogm);
  if (state.dropPrnExtremes) {
    args.push("--drop-prn-extremes");
    addValue("--prn-eps", state.prnEps);
  }
  if (state.bootstrapCi) {
    args.push("--bootstrap-ci");
    addValue("--bootstrap-B", state.bootstrapB);
    addValue("--bootstrap-seed", state.bootstrapSeed);
    addValue("--bootstrap-group", state.bootstrapGroup);
  }

  return args.join(" ");
};

const buildAutoCommandPreview = (
  datasetPath: string,
  autoForm: AutoFormState,
  tdaysAllowed: string,
  asofDowAllowed: string,
): string => {
  const args: string[] = [
    "python",
    "src/scripts/2-auto-calibrate-logit-model.py",
  ];

  const addValue = (flag: string, value: string) => {
    const trimmed = value.trim();
    if (!trimmed) return;
    args.push(flag, trimmed);
  };

  addValue("--csv", datasetPath);
  addValue("--out-dir", MODELS_BASE);
  addValue("--objective", autoForm.objective);
  addValue("--max-trials", autoForm.maxTrials);
  addValue("--seed", autoForm.seed);
  addValue("--parallel", autoForm.parallel);
  addValue("--tdays-allowed", tdaysAllowed);
  addValue("--asof-dow-allowed", asofDowAllowed);
  addValue("--foundation-tickers", autoForm.foundationTickers);
  addValue("--foundation-weight", autoForm.foundationWeight);

  // Build baseline-args including bootstrap flags
  const baselineArgParts: string[] = [];
  if (autoForm.baselineArgs.trim()) {
    baselineArgParts.push(autoForm.baselineArgs.trim());
  }
  if (autoForm.bootstrapCi) {
    baselineArgParts.push("--bootstrap-ci");
    if (autoForm.bootstrapB.trim()) {
      baselineArgParts.push(`--bootstrap-B ${autoForm.bootstrapB.trim()}`);
    }
    if (autoForm.bootstrapSeed.trim()) {
      baselineArgParts.push(`--bootstrap-seed ${autoForm.bootstrapSeed.trim()}`);
    }
    baselineArgParts.push(`--bootstrap-group ${autoForm.bootstrapGroup}`);
  }
  if (baselineArgParts.length > 0) {
    addValue("--baseline-args", `"${baselineArgParts.join(" ")}"`);
  }

  return args.join(" ");
};

export default function CalibrateModelsPage() {
  const [defaultName] = useState(() => defaultModelName());
  const [formState, setFormState] = useState<CalibrateFormState>(defaultForm);
  const [datasets, setDatasets] = useState<DatasetFileSummary[]>([]);
  const [models, setModels] = useState<ModelRunSummary[]>([]);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [runResult, setRunResult] = useState<CalibrateModelRunResponse | null>(
    null,
  );
  const [activeLog, setActiveLog] = useState<"stdout" | "stderr">("stdout");
  const [storageReady, setStorageReady] = useState(false);
  const [deletingModelId, setDeletingModelId] = useState<string | null>(null);
  const [renamingModelId, setRenamingModelId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState<string>("");
  const [runMode, setRunMode] = useState<"manual" | "auto">("manual");
  const [autoForm, setAutoForm] = useState<AutoFormState>(
    () => buildDefaultAutoForm(),
  );
  const [regimePreview, setRegimePreview] = useState<RegimePreviewResponse | null>(null);
  const [regimePreviewError, setRegimePreviewError] = useState<string | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [modelDetail, setModelDetail] = useState<ModelDetailResponse | null>(null);
  const [modelDetailError, setModelDetailError] = useState<string | null>(null);
  const [isModelDetailLoading, setIsModelDetailLoading] = useState(false);
  const [modelFiles, setModelFiles] = useState<ModelFilesListResponse | null>(null);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<ModelFileContentResponse | null>(null);
  const [isFileLoading, setIsFileLoading] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);
  const { jobStatus, jobId, setJobId, setJobStatus } = useCalibrationJob();
  const { anyJobRunning, primaryJob } = useAnyJobRunning();

  const isRunning =
    jobStatus?.status === "queued" || jobStatus?.status === "running";
  const activeRunMode = jobStatus?.mode ?? runMode;

  const SPLIT_LABELS: Record<string, string> = {
    test: "Test set",
    val_pool: "Validation pool",
  };

  const selectedDataset = useMemo(
    () => datasets.find((item) => item.path === formState.datasetPath),
    [datasets, formState.datasetPath],
  );

  const renderDatasetSection = (showMeta: boolean) => (
    <div className="section-card calibrate-section">
      <h3 className="section-heading">Dataset selection</h3>
      <div className="field">
        <label htmlFor="datasetSelect">Dataset</label>
        <select
          id="datasetSelect"
          className="input"
          value={formState.datasetPath}
          onChange={(event) =>
            setFormState((prev) => ({
              ...prev,
              datasetPath: event.target.value,
            }))
          }
        >
          {datasets.length === 0 ? (
            <option value="">No datasets available</option>
          ) : null}
          {datasets.map((dataset) => (
            <option key={dataset.path} value={dataset.path}>
              {dataset.name}
            </option>
          ))}
        </select>
        <span className="field-hint">
          Only non-drop CSVs in <code>src/data/raw/option-chains</code>.
        </span>
        {datasetError ? <div className="error">{datasetError}</div> : null}
      </div>
      {showMeta && selectedDataset ? (
        <div className="dataset-meta">
          <div>
            <span className="meta-label">Last modified</span>
            <span>{formatTimestamp(selectedDataset.last_modified)}</span>
          </div>
          <div>
            <span className="meta-label">Size</span>
            <span>{formatBytes(selectedDataset.size_bytes)}</span>
          </div>
          <div>
            <span className="meta-label">Path</span>
            <span>{selectedDataset.path}</span>
          </div>
        </div>
      ) : null}
    </div>
  );

  const commandPreview = useMemo(
    () =>
      runMode === "manual"
        ? buildManualCommandPreview(formState, defaultName)
        : buildAutoCommandPreview(
            formState.datasetPath,
            autoForm,
            formState.tdaysAllowed,
            formState.asofDowAllowed,
          ),
    [runMode, formState, defaultName, autoForm],
  );

  const regimeSpecific = isRegimeSpecific(formState);

  const constrainedSelectedFeatures = useMemo(
    () => applyFeatureConstraints(formState.selectedFeatures, regimeSpecific),
    [formState.selectedFeatures, regimeSpecific],
  );

  const selectedFeatureLabels = useMemo(
    () => constrainedSelectedFeatures,
    [constrainedSelectedFeatures],
  );

  const selectedFeatureSet = useMemo(
    () => new Set(constrainedSelectedFeatures),
    [constrainedSelectedFeatures],
  );

  const exclusiveSelections = useMemo(() => {
    const selections: Record<string, string | null> = {};
    EXCLUSIVE_GROUPS.forEach((group) => {
      selections[group.id] =
        group.keys.find((key) => selectedFeatureSet.has(key)) ?? null;
    });
    return selections;
  }, [selectedFeatureSet]);

  const redundantFeatures = useMemo(() => {
    const redundant = new Set<string>();
    const tdaysList = parseTdaysList(formState.tdaysAllowed);
    if (tdaysList.length === 1) {
      [
        "T_days",
        "T_years",
        "log_T_days",
        "sqrt_T_years",
        "x_prn_x_tdays",
      ].forEach((feature) => redundant.add(feature));
    }
    return redundant;
  }, [formState.tdaysAllowed]);

  const effectiveOutName = formState.outName.trim() || defaultName;
  const modelMetadata = modelDetail?.metadata as Record<string, unknown> | null;
  const modelFitWeights =
    modelMetadata && typeof modelMetadata["fit_weights"] === "object" && modelMetadata["fit_weights"]
      ? (modelMetadata["fit_weights"] as Record<string, unknown>)
      : null;
  const optionalFilters =
    modelMetadata &&
    typeof modelMetadata["optional_filters"] === "object" &&
    modelMetadata["optional_filters"]
      ? (modelMetadata["optional_filters"] as Record<string, unknown>)
      : null;

  const refreshDatasets = () => {
    fetchCalibrationDatasets()
      .then((data) => {
        setDatasets(data.datasets);
        if (!formState.datasetPath && data.datasets.length > 0) {
          setFormState((prev) => ({
            ...prev,
            datasetPath: data.datasets[0].path,
          }));
        }
        setDatasetError(null);
      })
      .catch((err: Error) => {
        setDatasetError(err.message);
      });
  };

  const refreshModels = () => {
    fetchCalibrationModels()
      .then((data) => {
        setModels(data.models);
        setModelError(null);
      })
      .catch((err: Error) => {
        setModelError(err.message);
      });
  };

  useEffect(() => {
    refreshDatasets();
    refreshModels();
    const storedResult = loadStoredRunResult();
    if (storedResult) {
      setRunResult(storedResult);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const stored = loadStoredForm();
    if (stored) {
      setFormState((prev) => ({ ...prev, ...stored }));
    }
    setStorageReady(true);
  }, []);

  useEffect(() => {
    if (formState.asofDowAllowed !== "Mon") return;
    if (formState.tdaysAllowed.trim()) return;
    setFormState((prev) => ({
      ...prev,
      tdaysAllowed: "4",
    }));
  }, [formState.asofDowAllowed, formState.tdaysAllowed]);

  useEffect(() => {
    if (areArraysEqual(formState.selectedFeatures, constrainedSelectedFeatures)) {
      return;
    }
    setFormState((prev) => {
      if (areArraysEqual(prev.selectedFeatures, constrainedSelectedFeatures)) {
        return prev;
      }
      return { ...prev, selectedFeatures: constrainedSelectedFeatures };
    });
  }, [constrainedSelectedFeatures, formState.selectedFeatures]);

  useEffect(() => {
    const hasFilters = Boolean(
      formState.tdaysAllowed.trim() || formState.asofDowAllowed.trim(),
    );
    if (!formState.datasetPath || !hasFilters) {
      setRegimePreview(null);
      setRegimePreviewError(null);
      return;
    }
    const timer = window.setTimeout(() => {
      previewCalibrationRegime({
        csv: formState.datasetPath,
        tdaysAllowed: formState.tdaysAllowed.trim() || undefined,
        asofDowAllowed: formState.asofDowAllowed.trim() || undefined,
      })
        .then((preview) => {
          setRegimePreview(preview);
          setRegimePreviewError(null);
        })
        .catch((err: Error) => {
          setRegimePreview(null);
          setRegimePreviewError(err.message);
        });
    }, 300);
    return () => window.clearTimeout(timer);
  }, [formState.datasetPath, formState.tdaysAllowed, formState.asofDowAllowed]);

  useEffect(() => {
    if (!storageReady) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(formState));
    } catch {
      // ignore storage failures
    }
  }, [formState, storageReady]);

  useEffect(() => {
    if (!runResult) return;
    if (!runResult.ok && runResult.stderr) {
      setActiveLog("stderr");
    } else {
      setActiveLog("stdout");
    }
    saveRunResult(runResult);
  }, [runResult]);

  useEffect(() => {
    if (!jobStatus) return;
    if (jobStatus.result) {
      setRunResult(jobStatus.result);
    }
    if (jobStatus.status === "failed" && jobStatus.error) {
      setRunError(jobStatus.error);
    }
    if (jobStatus.status === "finished") {
      refreshModels();
    }
  }, [jobStatus]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (anyJobRunning) {
      setRunError(
        `Another job is running (${primaryJob?.name ?? "unknown"}). Wait for it to finish.`,
      );
      return;
    }
    setRunError(null);
    setRunResult(null);
    setJobStatus(null);

    try {
      if (!formState.datasetPath) {
        throw new Error("Select a dataset to calibrate.");
      }
      if (runMode === "manual") {
        const selectedFeatures = applyFeatureConstraints(
          formState.selectedFeatures,
          regimeSpecific,
        );
        const customFeatures = formState.customFeatures
          .split(",")
          .map((value) => value.trim())
          .filter(Boolean)
          ;
        const featuresList = [...selectedFeatures, ...customFeatures].join(",");
        const payload = {
          csv: formState.datasetPath,
          outName: formState.outName.trim() || undefined,
          targetCol: DEFAULT_DATASET_COLUMNS.target,
          weekCol: DEFAULT_DATASET_COLUMNS.week,
          tickerCol: DEFAULT_DATASET_COLUMNS.ticker,
          weightCol: DEFAULT_DATASET_COLUMNS.weight,
          foundationTickers: formState.foundationTickers.trim() || undefined,
          foundationWeight: parseOptionalNumber(formState.foundationWeight),
          mode: formState.mode,
          tickerIntercepts: formState.tickerIntercepts,
          tickerXInteractions: formState.tickerXInteractions,
          trainTickers: formState.trainTickers.trim() || undefined,
          tdaysAllowed: formState.tdaysAllowed.trim() || undefined,
          asofDowAllowed: formState.asofDowAllowed.trim() || undefined,
          features: featuresList || undefined,
          categoricalFeatures: formState.categoricalFeatures.trim() || undefined,
          addInteractions: formState.addInteractions,
          calibrate: formState.calibrate,
          cGrid: formState.cGrid.trim() || undefined,
          trainDecayHalfLifeWeeks: parseOptionalNumber(
            formState.trainDecayHalfLifeWeeks,
          ),
          calibFracOfTrain: parseOptionalNumber(formState.calibFracOfTrain),
          fitWeightRenorm: formState.fitWeightRenorm,
          testWeeks: parseOptionalInt(formState.testWeeks),
          valWindows: parseOptionalInt(formState.valWindows),
          valWindowWeeks: parseOptionalInt(formState.valWindowWeeks),
          nBins: parseOptionalInt(formState.nBins),
          eceqBins: parseOptionalInt(formState.eceqBins),
          selectionObjective: formState.selectionObjective,
          fallbackToBaselineIfWorse: formState.fallbackToBaselineIfWorse,
          autoDropNearConstant: formState.autoDropNearConstant,
          metricsTopTickers: parseOptionalInt(formState.metricsTopTickers),
          tickerMinSupport: parseOptionalInt(formState.tickerMinSupport),
          tickerMinSupportInteractions: parseOptionalInt(formState.tickerMinSupportInteractions),
          randomState: parseOptionalInt(formState.randomState),
          enableXAbsM: formState.enableXAbsM,
          groupReweight:
            formState.groupReweight === "none"
              ? undefined
              : formState.groupReweight,
          maxAbsLogm: parseOptionalNumber(formState.maxAbsLogm),
          dropPrnExtremes: formState.dropPrnExtremes,
          prnEps: formState.dropPrnExtremes
            ? parseOptionalNumber(formState.prnEps)
            : undefined,
          bootstrapCi: formState.bootstrapCi || undefined,
          bootstrapB: formState.bootstrapCi
            ? parseOptionalInt(formState.bootstrapB)
            : undefined,
          bootstrapSeed: formState.bootstrapCi
            ? parseOptionalInt(formState.bootstrapSeed)
            : undefined,
          bootstrapGroup: formState.bootstrapCi
            ? formState.bootstrapGroup
            : undefined,
        };

        const status = await startCalibrationJob(payload);
        setJobId(status.job_id);
        setJobStatus(status);
      } else {
        const payload = {
          csv: formState.datasetPath,
          runName: autoForm.runName.trim() || undefined,
          objective: autoForm.objective,
          maxTrials: parseOptionalInt(autoForm.maxTrials),
          seed: parseOptionalInt(autoForm.seed),
          parallel: parseOptionalInt(autoForm.parallel),
          baselineArgs: autoForm.baselineArgs.trim() || undefined,
          tdaysAllowed: formState.tdaysAllowed.trim() || undefined,
          asofDowAllowed: formState.asofDowAllowed.trim() || undefined,
          foundationTickers: autoForm.foundationTickers.trim() || undefined,
          foundationWeight: parseOptionalNumber(autoForm.foundationWeight),
          bootstrapCi: autoForm.bootstrapCi || undefined,
          bootstrapB: autoForm.bootstrapCi
            ? parseOptionalInt(autoForm.bootstrapB)
            : undefined,
          bootstrapSeed: autoForm.bootstrapCi
            ? parseOptionalInt(autoForm.bootstrapSeed)
            : undefined,
          bootstrapGroup: autoForm.bootstrapCi
            ? autoForm.bootstrapGroup
            : undefined,
        };
        const status = await startAutoCalibrationJob(payload);
        setJobId(status.job_id);
        setJobStatus(status);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setRunError(message);
    }
  };

  const handleDeleteModel = async (modelId: string) => {
    if (!window.confirm(`Delete calibration model "${modelId}"? This removes the folder.`)) {
      return;
    }
    setDeletingModelId(modelId);
    try {
      await deleteCalibrationModel(modelId);
      refreshModels();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setModelError(message);
    } finally {
      setDeletingModelId(null);
    }
  };

  const handleStartRename = (modelId: string) => {
    setRenamingModelId(modelId);
    setRenameValue(modelId);
  };

  const handleCancelRename = () => {
    setRenamingModelId(null);
    setRenameValue("");
  };

  const handleConfirmRename = async (modelId: string) => {
    const trimmed = renameValue.trim();
    if (!trimmed) {
      setModelError("Model name cannot be empty.");
      return;
    }
    if (trimmed === modelId) {
      handleCancelRename();
      return;
    }
    if (models.some((m) => m.id === trimmed)) {
      setModelError(`Model "${trimmed}" already exists.`);
      return;
    }
    try {
      await renameCalibrationModel(modelId, trimmed);
      refreshModels();
      handleCancelRename();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setModelError(message);
    }
  };

  const handleSelectModel = async (modelId: string) => {
    if (selectedModelId === modelId) {
      setSelectedModelId(null);
      setModelDetail(null);
      setModelDetailError(null);
      return;
    }
    setSelectedModelId(modelId);
    setIsModelDetailLoading(true);
    setModelDetail(null);
    setModelDetailError(null);
    setModelFiles(null);
    setSelectedFile(null);
    setFileContent(null);
    setFileError(null);
    try {
      const [detail, files] = await Promise.all([
        fetchCalibrationModelDetail(modelId),
        fetchModelFiles(modelId),
      ]);
      setModelDetail(detail);
      setModelFiles(files);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setModelDetailError(message);
    } finally {
      setIsModelDetailLoading(false);
    }
  };

  const handleFileSelect = async (filename: string) => {
    if (!selectedModelId) return;
    if (selectedFile === filename) {
      setSelectedFile(null);
      setFileContent(null);
      setFileError(null);
      return;
    }
    setSelectedFile(filename);
    setIsFileLoading(true);
    setFileContent(null);
    setFileError(null);
    try {
      const content = await fetchModelFileContent(selectedModelId, filename);
      setFileContent(content);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setFileError(message);
    } finally {
      setIsFileLoading(false);
    }
  };

  const handleFeatureToggle = (featureKey: string, checked: boolean) => {
    if (regimeSpecific && REGIME_LOCKED_FEATURES.has(featureKey)) {
      return;
    }
    setFormState((prev) => {
      let next = prev.selectedFeatures.filter((feature) => feature !== featureKey);
      if (checked) {
        const groupId = EXCLUSIVE_GROUP_BY_KEY[featureKey];
        if (groupId) {
          const groupKeys = EXCLUSIVE_GROUPS_BY_ID[groupId] ?? [];
          next = next.filter((feature) => !groupKeys.includes(feature));
        }
        next = [...next, featureKey];
      }
      next = applyFeatureConstraints(next, regimeSpecific);
      return { ...prev, selectedFeatures: next };
    });
  };

  return (
    <section className="page calibrate-page">
      <header className="page-header">
        <div>
          <p className="page-kicker">Calibrate models</p>
          <h1 className="page-title">Configure and train a new calibration run</h1>
          <p className="page-subtitle">
            Select a dataset from <code>src/data/raw/option-chains</code>, tune
            the CLI arguments, and generate new model artifacts.
          </p>
        </div>
        <div className="meta-card calibrate-meta page-goal-card">
          <span className="meta-label">Goal</span>
          <span>Train and tune calibration models that drive pHAT scoring.</span>
          <div className="meta-pill">Models live in src/data/models</div>
        </div>
      </header>

      <form className="calibrate-grid" onSubmit={handleSubmit}>
        <section className="panel run-mode-panel">
          <div className="panel-header">
            <div>
              <h2>Calibration configuration</h2>
              <span className="panel-hint">
                Dataset selection is restricted to option-chain outputs.
              </span>
            </div>
            <div className="run-mode-tabs">
              <button
                type="button"
                className={`run-mode-tab ${
                  runMode === "manual" ? "active" : ""
                }`}
                onClick={() => setRunMode("manual")}
              >
                Manual run
              </button>
              <button
                type="button"
                className={`run-mode-tab ${
                  runMode === "auto" ? "active" : ""
                }`}
                onClick={() => setRunMode("auto")}
              >
                Auto run
              </button>
            </div>
          </div>
          <div className="panel-body">
            {runMode === "manual" ? (
              <>
                <div className="config-summary">
                  <div>
                    <span className="meta-label">Dataset</span>
                    <span>{selectedDataset?.name ?? "None selected"}</span>
                  </div>
                  <div>
                    <span className="meta-label">Output dir</span>
                    <span>src/data/models/{effectiveOutName}</span>
                  </div>
                  <div>
                    <span className="meta-label">Calibration</span>
                    <span>{formState.calibrate}</span>
                  </div>
                </div>
                {renderDatasetSection(true)}
                <div className="section-card calibrate-section">
                  <h3 className="section-heading">Horizon regime</h3>
                  <div className="tdays-picker regime-toggle">
                    <span className="meta-label">Regime mode</span>
                    <div className="tdays-chips">
                      <button
                        type="button"
                        className={`chip ${!regimeSpecific ? "active" : ""}`}
                        onClick={() =>
                          setFormState((prev) => ({
                            ...prev,
                            tdaysAllowed: "",
                            asofDowAllowed: "",
                          }))
                        }
                      >
                        Train pooled across all snapshots
                      </button>
                      <button
                        type="button"
                        className={`chip ${regimeSpecific ? "active" : ""}`}
                        onClick={() => {
                          if (!regimeSpecific) {
                            setFormState((prev) => ({
                              ...prev,
                              asofDowAllowed: prev.asofDowAllowed || "Mon",
                            }));
                          }
                        }}
                      >
                        Train a regime-specific model
                      </button>
                    </div>
                  </div>
                  <div className="fields-grid">
                    <div className="field">
                      <label htmlFor="asofDowAllowed">Training day-of-week</label>
                      <select
                        id="asofDowAllowed"
                        className="input"
                        value={formState.asofDowAllowed}
                        onChange={(event) =>
                          setFormState((prev) => ({
                            ...prev,
                            asofDowAllowed: event.target.value,
                          }))
                        }
                      >
                        <option value="">Any day</option>
                        <option value="Mon">Mon</option>
                        <option value="Tue">Tue</option>
                        <option value="Wed">Wed</option>
                        <option value="Thu">Thu</option>
                        <option value="Fri">Fri</option>
                      </select>
                      <span className="field-hint">
                        Filters snapshots by as-of day. Monday defaults to <code>T_days=4</code>.
                      </span>
                    </div>
                    <div className="field">
                      <label htmlFor="tdaysAllowed">Allowed T_days</label>
                      <input
                        id="tdaysAllowed"
                        className="input"
                        placeholder="Auto"
                        value={formState.tdaysAllowed}
                        onChange={(event) =>
                          setFormState((prev) => ({
                            ...prev,
                            tdaysAllowed: event.target.value,
                          }))
                        }
                      />
                      <span className="field-hint">
                        Comma-separated list. Leave blank for no filter.
                      </span>
                    </div>
                  </div>
                  <div className="tdays-picker">
                    <span className="meta-label">Quick T_days</span>
                    <div className="tdays-chips">
                      <button
                        type="button"
                        className={`chip ${formState.tdaysAllowed.trim() ? "" : "active"}`}
                        onClick={() =>
                          setFormState((prev) => ({ ...prev, tdaysAllowed: "" }))
                        }
                      >
                        Auto
                      </button>
                      {TDAYS_OPTIONS.map((value) => {
                        const selected = parseTdaysList(formState.tdaysAllowed).includes(value);
                        return (
                          <button
                            key={value}
                            type="button"
                            className={`chip ${selected ? "active" : ""}`}
                            onClick={() => {
                              const current = parseTdaysList(formState.tdaysAllowed);
                              const next = selected
                                ? current.filter((item) => item !== value)
                                : [...current, value];
                              setFormState((prev) => ({
                                ...prev,
                                tdaysAllowed: formatTdaysList(next),
                              }));
                            }}
                          >
                            {value}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                  {regimePreviewError ? (
                    <div className="error">{regimePreviewError}</div>
                  ) : null}
                  {regimePreview ? (
                    <div className="regime-preview">
                      <div>
                        <span className="meta-label">Rows after filter</span>
                        <span>{regimePreview.rows_after}</span>
                      </div>
                      <div>
                        <span className="meta-label">Tickers retained</span>
                        <span>{regimePreview.tickers_after}</span>
                      </div>
                    </div>
                  ) : null}
                  {regimePreview &&
                  (regimePreview.rows_after < REGIME_WARNING_MIN_ROWS ||
                    regimePreview.tickers_after < 2) ? (
                    <div className="warning">
                      Regime filters are very tight. Loosen day/T_days or add more
                      history before training.
                    </div>
                  ) : null}
                </div>
            <div className="section-card calibrate-section">
              <h3 className="section-heading">Output</h3>
              <div className="field">
                <label>Output directory</label>
                <div className="input readonly">src/data/models</div>
              </div>
              <div className="field">
                <label htmlFor="outName">Model folder name</label>
                <input
                  id="outName"
                  className="input"
                  placeholder={defaultName}
                  value={formState.outName}
                  onChange={(event) =>
                    setFormState((prev) => ({
                      ...prev,
                      outName: event.target.value,
                    }))
                  }
                />
                <span className="field-hint">
                  Leave blank to use <code>{effectiveOutName}</code>.
                </span>
              </div>
            </div>

            <div className="features-controls-stack">
              <details className="advanced">
                <summary>Features control</summary>
                {regimeSpecific ? (
                  <div className="warning feature-warning">
                    Regime-specific training locks: {Array.from(REGIME_LOCKED_FEATURES).join(", ")}.
                  </div>
                ) : null}
                <div className="feature-groups">
                  {FEATURE_GROUPS.map((group) => {
                    const groupSelection = group.exclusive
                      ? exclusiveSelections[group.id]
                      : null;
                    const groupLockedAll =
                      regimeSpecific &&
                      group.keys.every((key) => REGIME_LOCKED_FEATURES.has(key));
                    return (
                      <div key={group.id} className="feature-group">
                        <div className="feature-group-header">
                          <div>
                            <div className="feature-group-title">{group.title}</div>
                            <div className="feature-group-hint">{group.hint}</div>
                          </div>
                          <div className="feature-group-meta">
                            {group.exclusive ? (
                              <span className="feature-tag">
                                {groupSelection ? `Using ${groupSelection}` : "Choose one"}
                              </span>
                            ) : null}
                            {groupLockedAll ? (
                              <span className="feature-tag locked">Locked by regime</span>
                            ) : null}
                          </div>
                        </div>
                        <div className="feature-grid feature-group-grid">
                          {group.keys.map((featureKey) => {
                            const feature = FEATURE_INDEX[featureKey];
                            if (!feature) return null;
                            const isRedundant = redundantFeatures.has(feature.key);
                            const isLocked =
                              regimeSpecific && REGIME_LOCKED_FEATURES.has(feature.key);
                            const isSelected = selectedFeatureSet.has(feature.key);
                            return (
                              <label
                                key={feature.key}
                                className={`feature-item${isRedundant ? " is-redundant" : ""}${
                                  isLocked ? " is-disabled" : ""
                                }`}
                              >
                                <input
                                  type="checkbox"
                                  checked={isSelected}
                                  disabled={isLocked}
                                  onChange={(event) =>
                                    handleFeatureToggle(feature.key, event.target.checked)
                                  }
                                />
                                <div>
                                  <div className="feature-title">
                                    {feature.key}
                                    {isRedundant ? (
                                      <span className="feature-tag redundant">
                                        Constant under regime
                                      </span>
                                    ) : null}
                                    {isLocked ? (
                                      <span className="feature-tag locked">
                                        Locked by regime
                                      </span>
                                    ) : null}
                                  </div>
                                  <div className="feature-desc">{feature.description}</div>
                                </div>
                              </label>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="field field-spaced">
                  <label htmlFor="customFeatures">Custom features</label>
                  <input
                    id="customFeatures"
                    className="input"
                    placeholder="extra_feature_a,extra_feature_b"
                    value={formState.customFeatures}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        customFeatures: event.target.value,
                      }))
                    }
                  />
                  <span className="field-hint">
                    Comma-separated list appended to the selected features.
                  </span>
                </div>
                <div className="field field-spaced">
                  <label htmlFor="categoricalFeatures">Categorical features</label>
                  <input
                    id="categoricalFeatures"
                    className="input"
                    value={formState.categoricalFeatures}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        categoricalFeatures: event.target.value,
                      }))
                    }
                  />
                </div>
                <label className="checkbox checkbox-spaced">
                  <input
                    type="checkbox"
                    checked={formState.addInteractions}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        addInteractions: event.target.checked,
                      }))
                    }
                  />
                  Add numeric interactions (x_logit_prn × T_days/rv20/log_m)
                </label>
                <label className="checkbox checkbox-spaced">
                  <input
                    type="checkbox"
                    checked={formState.enableXAbsM}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        enableXAbsM: event.target.checked,
                      }))
                    }
                  />
                  Add x_abs_m feature (|m| × logit(pRN))
                </label>
                <span className="field-hint">
                  x_m is auto-included when log-moneyness exists; x_abs_m needs this toggle.
                </span>
                <div className="feature-summary">
                  <span className="meta-label">Selected features</span>
                  <span>
                    {selectedFeatureLabels.length > 0
                      ? selectedFeatureLabels.join(", ")
                      : "None selected"}
                  </span>
                </div>
              </details>

              <details className="advanced">
              <summary>Model structure</summary>
              <div className="fields-grid">
                <div className="field">
                  <label htmlFor="modeSelect">Mode</label>
                  <select
                    id="modeSelect"
                    className="input"
                    value={formState.mode}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        mode: event.target.value as CalibrateFormState["mode"],
                      }))
                    }
                  >
                    <option value="baseline">baseline</option>
                    <option value="pooled">pooled</option>
                    <option value="two_stage">two_stage</option>
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="tickerIntercepts">Ticker intercepts</label>
                  <select
                    id="tickerIntercepts"
                    className="input"
                    value={formState.tickerIntercepts}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        tickerIntercepts: event.target.value as CalibrateFormState["tickerIntercepts"],
                      }))
                    }
                  >
                    <option value="none">none</option>
                    <option value="all">all</option>
                    <option value="non_foundation">non_foundation</option>
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="foundationWeight">Foundation weight</label>
                  <input
                    id="foundationWeight"
                    className="input"
                    inputMode="decimal"
                    value={formState.foundationWeight}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        foundationWeight: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="metricsTopTickers">Metrics top tickers</label>
                  <input
                    id="metricsTopTickers"
                    className="input"
                    inputMode="numeric"
                    value={formState.metricsTopTickers}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        metricsTopTickers: event.target.value,
                      }))
                    }
                  />
                  <span className="field-hint">
                    How many top tickers to include in diagnostics.
                  </span>
                </div>
                <div className="field">
                  <label htmlFor="tickerMinSupport">Ticker min support</label>
                  <input
                    id="tickerMinSupport"
                    className="input"
                    inputMode="numeric"
                    value={formState.tickerMinSupport}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        tickerMinSupport: event.target.value,
                      }))
                    }
                  />
                  <span className="field-hint">
                    Min TRAIN_FIT rows to keep ticker intercepts (others → OTHER).
                  </span>
                </div>
                <div className="field">
                  <label htmlFor="tickerMinSupportInteractions">Ticker min support (interactions)</label>
                  <input
                    id="tickerMinSupportInteractions"
                    className="input"
                    inputMode="numeric"
                    value={formState.tickerMinSupportInteractions}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        tickerMinSupportInteractions: event.target.value,
                      }))
                    }
                  />
                  <span className="field-hint">
                    Stricter support threshold for ticker × logit interactions.
                  </span>
                </div>
              </div>
              <div className="field">
                <label htmlFor="foundationTickers">Foundation tickers</label>
                <input
                  id="foundationTickers"
                  className="input"
                  placeholder="ticker1,ticker2,..."
                  value={formState.foundationTickers}
                  onChange={(event) =>
                    setFormState((prev) => ({
                      ...prev,
                      foundationTickers: event.target.value,
                    }))
                  }
                />
              </div>
              <div className="field">
                <label htmlFor="trainTickers">Train tickers</label>
                <input
                  id="trainTickers"
                  className="input"
                  placeholder="ticker1,ticker2,..."
                  value={formState.trainTickers}
                  onChange={(event) =>
                    setFormState((prev) => ({
                      ...prev,
                      trainTickers: event.target.value,
                    }))
                  }
                />
                <span className="field-hint">
                  Restrict training to a subset of tickers.
                </span>
              </div>
              <label className="checkbox checkbox-spaced">
                <input
                  type="checkbox"
                  checked={formState.tickerXInteractions}
                  onChange={(event) =>
                    setFormState((prev) => ({
                      ...prev,
                      tickerXInteractions: event.target.checked,
                    }))
                  }
                />
                Enable per-ticker interactions (ticker × x_logit_prn)
              </label>
              </details>

              <details className="advanced">
              <summary>Calibration & validation</summary>
                <div className="fields-grid">
                  <div className="field">
                    <label htmlFor="calibrate">Calibration</label>
                    <select
                      id="calibrate"
                      className="input"
                      value={formState.calibrate}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          calibrate: event.target.value as CalibrateFormState["calibrate"],
                        }))
                      }
                    >
                      <option value="none">none</option>
                      <option value="platt">platt</option>
                    </select>
                  </div>
                  <div className="field">
                    <label htmlFor="cGrid">C grid</label>
                    <input
                      id="cGrid"
                      className="input"
                      value={formState.cGrid}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          cGrid: event.target.value,
                        }))
                      }
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="trainDecay">Train decay half-life</label>
                    <input
                      id="trainDecay"
                      className="input"
                      inputMode="decimal"
                      value={formState.trainDecayHalfLifeWeeks}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          trainDecayHalfLifeWeeks: event.target.value,
                        }))
                      }
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="calibFrac">Calib frac of train</label>
                    <input
                      id="calibFrac"
                      className="input"
                      inputMode="decimal"
                      value={formState.calibFracOfTrain}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          calibFracOfTrain: event.target.value,
                        }))
                      }
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="fitWeightRenorm">Fit weight renorm</label>
                    <select
                      id="fitWeightRenorm"
                      className="input"
                      value={formState.fitWeightRenorm}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          fitWeightRenorm:
                            event.target.value as CalibrateFormState["fitWeightRenorm"],
                        }))
                      }
                    >
                      <option value="mean1">mean1</option>
                      <option value="none">none</option>
                    </select>
                  </div>
                  <div className="field">
                    <label htmlFor="testWeeks">Test weeks</label>
                    <input
                      id="testWeeks"
                      className="input"
                      inputMode="numeric"
                      value={formState.testWeeks}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          testWeeks: event.target.value,
                        }))
                      }
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="valWindows">Val windows</label>
                    <input
                      id="valWindows"
                      className="input"
                      inputMode="numeric"
                      value={formState.valWindows}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          valWindows: event.target.value,
                        }))
                      }
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="valWindowWeeks">Val window weeks</label>
                    <input
                      id="valWindowWeeks"
                      className="input"
                      inputMode="numeric"
                      value={formState.valWindowWeeks}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          valWindowWeeks: event.target.value,
                        }))
                      }
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="nBins">Bins</label>
                    <input
                      id="nBins"
                      className="input"
                      inputMode="numeric"
                      value={formState.nBins}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          nBins: event.target.value,
                        }))
                      }
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="eceqBins">ECE-Q bins</label>
                    <input
                      id="eceqBins"
                      className="input"
                      inputMode="numeric"
                      value={formState.eceqBins}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          eceqBins: event.target.value,
                        }))
                      }
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="selectionObjective">Selection objective</label>
                    <select
                      id="selectionObjective"
                      className="input"
                      value={formState.selectionObjective}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          selectionObjective:
                            event.target.value as CalibrateFormState["selectionObjective"],
                        }))
                      }
                    >
                      <option value="delta_vs_baseline">delta vs baseline</option>
                    </select>
                  </div>
                  <div className="field">
                    <label htmlFor="randomState">Random state</label>
                    <input
                      id="randomState"
                      className="input"
                      inputMode="numeric"
                      value={formState.randomState}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          randomState: event.target.value,
                        }))
                      }
                    />
                  </div>
                </div>
                <label className="checkbox checkbox-spaced">
                  <input
                    type="checkbox"
                    checked={formState.fallbackToBaselineIfWorse}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        fallbackToBaselineIfWorse: event.target.checked,
                      }))
                    }
                  />
                  Fallback to baseline if worse than pRN
                </label>
                <label className="checkbox checkbox-spaced">
                  <input
                    type="checkbox"
                    checked={formState.autoDropNearConstant}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        autoDropNearConstant: event.target.checked,
                      }))
                    }
                  />
                  Auto-drop near-constant features after regime filters
                </label>
              </details>
              <details className="advanced">
              <summary>Weights & filters</summary>
              <div className="fields-grid">
                <div className="field">
                  <label htmlFor="groupReweight">Group reweight</label>
                  <select
                    id="groupReweight"
                    className="input"
                    value={formState.groupReweight}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        groupReweight: event.target.value as CalibrateFormState["groupReweight"],
                      }))
                    }
                  >
                    <option value="none">none</option>
                    <option value="chain">chain snapshots</option>
                  </select>
                  <span className="field-hint">
                    Equalize total TRAIN_FIT weight per chain snapshot (ticker + asof + expiry).
                  </span>
                </div>
                <div className="field">
                  <label htmlFor="maxAbsLogm">Max |log moneyness|</label>
                  <input
                    id="maxAbsLogm"
                    className="input"
                    inputMode="decimal"
                    placeholder="0.4"
                    value={formState.maxAbsLogm}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        maxAbsLogm: event.target.value,
                      }))
                    }
                  />
                  <span className="field-hint">
                    Filters rows using log_m_fwd (fallback log_m) threshold.
                  </span>
                </div>
              </div>
              <label className="checkbox checkbox-spaced">
                <input
                  type="checkbox"
                  checked={formState.dropPrnExtremes}
                  onChange={(event) =>
                    setFormState((prev) => ({
                      ...prev,
                      dropPrnExtremes: event.target.checked,
                    }))
                  }
                />
                Drop pRN extremes near 0 or 1
              </label>
              <div className="fields-grid">
                <div className="field">
                  <label htmlFor="prnEps">pRN epsilon</label>
                  <input
                    id="prnEps"
                    className="input"
                    inputMode="decimal"
                    value={formState.prnEps}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        prnEps: event.target.value,
                      }))
                    }
                    disabled={!formState.dropPrnExtremes}
                  />
                  <span className="field-hint">
                    Drops rows when pRN &lt;= eps or &gt;= 1 - eps.
                  </span>
                </div>
              </div>
              <span className="field-hint">
                Filters apply before split creation and are logged.
              </span>
              </details>

              <details className="advanced">
              <summary>Bootstrap confidence intervals</summary>
              <label className="checkbox checkbox-spaced">
                <input
                  type="checkbox"
                  checked={formState.bootstrapCi}
                  onChange={(event) =>
                    setFormState((prev) => ({
                      ...prev,
                      bootstrapCi: event.target.checked,
                    }))
                  }
                />
                Compute bootstrap CIs for delta metrics
              </label>
              <div className="fields-grid">
                <div className="field">
                  <label htmlFor="bootstrapB">Resamples (B)</label>
                  <input
                    id="bootstrapB"
                    className="input"
                    inputMode="numeric"
                    value={formState.bootstrapB}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        bootstrapB: event.target.value,
                      }))
                    }
                    disabled={!formState.bootstrapCi}
                  />
                  <span className="field-hint">
                    Number of bootstrap resamples (default 2000).
                  </span>
                </div>
                <div className="field">
                  <label htmlFor="bootstrapSeed">Seed</label>
                  <input
                    id="bootstrapSeed"
                    className="input"
                    inputMode="numeric"
                    value={formState.bootstrapSeed}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        bootstrapSeed: event.target.value,
                      }))
                    }
                    disabled={!formState.bootstrapCi}
                  />
                </div>
                <div className="field">
                  <label htmlFor="bootstrapGroup">Group strategy</label>
                  <select
                    id="bootstrapGroup"
                    className="input"
                    value={formState.bootstrapGroup}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        bootstrapGroup: event.target
                          .value as CalibrateFormState["bootstrapGroup"],
                      }))
                    }
                    disabled={!formState.bootstrapCi}
                  >
                    <option value="auto">auto (ticker+date+expiry)</option>
                    <option value="ticker_day">ticker + date</option>
                    <option value="day">date only</option>
                    <option value="iid">iid (no blocking)</option>
                  </select>
                  <span className="field-hint">
                    Block bootstrap groups for correlated data.
                  </span>
                </div>
              </div>
              </details>
            </div>

            <div className="actions">
              <button
                className="button primary"
                type="submit"
                disabled={isRunning || anyJobRunning || !formState.datasetPath}
              >
                {isRunning ? "Calibrating..." : "Run calibration"}
              </button>
              <button
                className="button ghost"
                type="button"
                disabled={isRunning || anyJobRunning}
                onClick={() => setFormState(defaultForm)}
              >
                Reset
              </button>
            </div>
          </>
        ) : (
          <>
            {renderDatasetSection(false)}
            <div className="section-card calibrate-section">
              <h3 className="section-heading">Horizon regime</h3>
              <div className="tdays-picker regime-toggle">
                <span className="meta-label">Regime mode</span>
                <div className="tdays-chips">
                  <button
                    type="button"
                    className={`chip ${!regimeSpecific ? "active" : ""}`}
                    onClick={() =>
                      setFormState((prev) => ({
                        ...prev,
                        tdaysAllowed: "",
                        asofDowAllowed: "",
                      }))
                    }
                  >
                    Train pooled across all snapshots
                  </button>
                  <button
                    type="button"
                    className={`chip ${regimeSpecific ? "active" : ""}`}
                    onClick={() => {
                      if (!regimeSpecific) {
                        setFormState((prev) => ({
                          ...prev,
                          asofDowAllowed: prev.asofDowAllowed || "Mon",
                        }));
                      }
                    }}
                  >
                    Train a regime-specific model
                  </button>
                </div>
              </div>
              <div className="fields-grid">
                <div className="field">
                  <label htmlFor="asofDowAllowedAuto">Training day-of-week</label>
                  <select
                    id="asofDowAllowedAuto"
                    className="input"
                    value={formState.asofDowAllowed}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        asofDowAllowed: event.target.value,
                      }))
                    }
                  >
                    <option value="">Any day</option>
                    <option value="Mon">Mon</option>
                    <option value="Tue">Tue</option>
                    <option value="Wed">Wed</option>
                    <option value="Thu">Thu</option>
                    <option value="Fri">Fri</option>
                  </select>
                  <span className="field-hint">
                    Filters snapshots by as-of day. Monday defaults to <code>T_days=4</code>.
                  </span>
                </div>
                <div className="field">
                  <label htmlFor="tdaysAllowedAuto">Allowed T_days</label>
                  <input
                    id="tdaysAllowedAuto"
                    className="input"
                    placeholder="Auto"
                    value={formState.tdaysAllowed}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        tdaysAllowed: event.target.value,
                      }))
                    }
                  />
                  <span className="field-hint">
                    Comma-separated list. Leave blank for no filter.
                  </span>
                </div>
              </div>
              <div className="tdays-picker">
                <span className="meta-label">Quick T_days</span>
                <div className="tdays-chips">
                  <button
                    type="button"
                    className={`chip ${formState.tdaysAllowed.trim() ? "" : "active"}`}
                    onClick={() =>
                      setFormState((prev) => ({ ...prev, tdaysAllowed: "" }))
                    }
                  >
                    Auto
                  </button>
                  {TDAYS_OPTIONS.map((value) => {
                    const selected = parseTdaysList(formState.tdaysAllowed).includes(value);
                    return (
                      <button
                        key={value}
                        type="button"
                        className={`chip ${selected ? "active" : ""}`}
                        onClick={() => {
                          const current = parseTdaysList(formState.tdaysAllowed);
                          const next = selected
                            ? current.filter((item) => item !== value)
                            : [...current, value];
                          setFormState((prev) => ({
                            ...prev,
                            tdaysAllowed: formatTdaysList(next),
                          }));
                        }}
                      >
                        {value}
                      </button>
                    );
                  })}
                </div>
              </div>
              {regimePreviewError ? (
                <div className="error">{regimePreviewError}</div>
              ) : null}
              {regimePreview ? (
                <div className="regime-preview">
                  <div>
                    <span className="meta-label">Rows after filter</span>
                    <span>{regimePreview.rows_after}</span>
                  </div>
                  <div>
                    <span className="meta-label">Tickers retained</span>
                    <span>{regimePreview.tickers_after}</span>
                  </div>
                </div>
              ) : null}
              {regimePreview &&
              (regimePreview.rows_after < REGIME_WARNING_MIN_ROWS ||
                regimePreview.tickers_after < 2) ? (
                <div className="warning">
                  Regime filters are very tight. Loosen day/T_days or add more
                  history before training.
                </div>
              ) : null}
            </div>
            <div className="section-card auto-section">
              <h3 className="section-heading">Auto tuner</h3>
              <div className="field">
                <label htmlFor="autoRunName">Output folder name</label>
                <input
                  id="autoRunName"
                  className="input"
                  value={autoForm.runName}
                  onChange={(event) =>
                    setAutoForm((prev) => ({ ...prev, runName: event.target.value }))
                  }
                />
                <span className="field-hint">
                  Best model will be exported to <code>{MODELS_BASE}/{autoForm.runName || 'auto-run'}</code>
                </span>
              </div>
              <div className="fields-grid">
                <div className="field">
                  <label htmlFor="autoMaxTrials">Max trials</label>
                  <input
                    id="autoMaxTrials"
                    className="input"
                    inputMode="numeric"
                    value={autoForm.maxTrials}
                    onChange={(event) =>
                      setAutoForm((prev) => ({ ...prev, maxTrials: event.target.value }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="autoSeed">Seed</label>
                  <input
                    id="autoSeed"
                    className="input"
                    inputMode="numeric"
                    value={autoForm.seed}
                    onChange={(event) =>
                      setAutoForm((prev) => ({ ...prev, seed: event.target.value }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="autoParallel">Parallel trials</label>
                  <input
                    id="autoParallel"
                    className="input"
                    inputMode="numeric"
                    value={autoForm.parallel}
                    onChange={(event) =>
                      setAutoForm((prev) => ({ ...prev, parallel: event.target.value }))
                    }
                  />
                </div>
              </div>
              <div className="field">
                <label htmlFor="baselineArgs">Baseline args</label>
                <textarea
                  id="baselineArgs"
                  className="input"
                  value={autoForm.baselineArgs}
                  onChange={(event) =>
                    setAutoForm((prev) => ({ ...prev, baselineArgs: event.target.value }))
                  }
                />
                <span className="field-hint">
                  Extra calibrator args (e.g. <code>--mode pooled</code>).
                </span>
              </div>
              <div className="fields-grid">
                <div className="field">
                  <label htmlFor="autoFoundationTickers">Foundation tickers</label>
                  <input
                    id="autoFoundationTickers"
                    className="input"
                    value={autoForm.foundationTickers}
                    onChange={(event) =>
                      setAutoForm((prev) => ({
                        ...prev,
                        foundationTickers: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="autoFoundationWeight">Foundation weight</label>
                  <input
                    id="autoFoundationWeight"
                    className="input"
                    inputMode="decimal"
                    value={autoForm.foundationWeight}
                    onChange={(event) =>
                      setAutoForm((prev) => ({
                        ...prev,
                        foundationWeight: event.target.value,
                      }))
                    }
                  />
                </div>
              </div>

              <details className="advanced">
                <summary>Bootstrap confidence intervals</summary>
                <label className="checkbox checkbox-spaced">
                  <input
                    type="checkbox"
                    checked={autoForm.bootstrapCi}
                    onChange={(event) =>
                      setAutoForm((prev) => ({
                        ...prev,
                        bootstrapCi: event.target.checked,
                      }))
                    }
                  />
                  Compute bootstrap CIs for delta metrics
                </label>
                <div className="fields-grid">
                  <div className="field">
                    <label htmlFor="autoBootstrapB">Resamples (B)</label>
                    <input
                      id="autoBootstrapB"
                      className="input"
                      inputMode="numeric"
                      value={autoForm.bootstrapB}
                      onChange={(event) =>
                        setAutoForm((prev) => ({
                          ...prev,
                          bootstrapB: event.target.value,
                        }))
                      }
                      disabled={!autoForm.bootstrapCi}
                    />
                    <span className="field-hint">
                      Number of bootstrap resamples (default 2000).
                    </span>
                  </div>
                  <div className="field">
                    <label htmlFor="autoBootstrapSeed">Seed</label>
                    <input
                      id="autoBootstrapSeed"
                      className="input"
                      inputMode="numeric"
                      value={autoForm.bootstrapSeed}
                      onChange={(event) =>
                        setAutoForm((prev) => ({
                          ...prev,
                          bootstrapSeed: event.target.value,
                        }))
                      }
                      disabled={!autoForm.bootstrapCi}
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="autoBootstrapGroup">Group strategy</label>
                    <select
                      id="autoBootstrapGroup"
                      className="input"
                      value={autoForm.bootstrapGroup}
                      onChange={(event) =>
                        setAutoForm((prev) => ({
                          ...prev,
                          bootstrapGroup: event.target
                            .value as AutoFormState["bootstrapGroup"],
                        }))
                      }
                      disabled={!autoForm.bootstrapCi}
                    >
                      <option value="auto">auto (ticker+date+expiry)</option>
                      <option value="ticker_day">ticker + date</option>
                      <option value="day">date only</option>
                      <option value="iid">iid (no blocking)</option>
                    </select>
                    <span className="field-hint">
                      Block bootstrap groups for correlated data.
                    </span>
                  </div>
                </div>
              </details>

            <div className="auto-summary">
                <p>
                  Auto runs log trials under <code>{MODELS_BASE}</code>.
                </p>
                <p>
                  Final artifacts are linked into <code>src/data/models</code>.
                </p>
              </div>
            </div>
            <div className="actions">
              <button
                className="button primary"
                type="submit"
                disabled={isRunning || anyJobRunning || !formState.datasetPath}
              >
                {isRunning ? "Running auto tuner..." : "Start auto tuner"}
              </button>
              <button
                className="button ghost"
                type="button"
                disabled={isRunning || anyJobRunning}
                onClick={() => setAutoForm(buildDefaultAutoForm())}
              >
                Reset
              </button>
            </div>
          </>
        )}
          {runError ? <div className="error">{runError}</div> : null}
          </div>
        </section>

        <section className="panel latest-run-panel">
          <div className="panel-header">
            <h2>Latest run output</h2>
            <span className="panel-hint">
              Captures stdout/stderr from the calibration script.
            </span>
          </div>
          <div className="panel-body">
            {isRunning && activeRunMode === "auto" ? (
              <div className="auto-progress">
                <div className="auto-progress-header">
                  <span className="meta-label">Auto-select in progress</span>
                  <div className="spinner"></div>
                </div>
                <p className="auto-progress-message">
                  Running automated model selection. This may take several minutes depending on the number of trials and parallelization settings.
                </p>
                <div className="auto-progress-details">
                  <div>
                    <span className="meta-label">Max trials</span>
                    <span>{autoForm.maxTrials || "Not set"}</span>
                  </div>
                  <div>
                    <span className="meta-label">Parallel</span>
                    <span>{autoForm.parallel || "1"}</span>
                  </div>
                  <div>
                    <span className="meta-label">Objective</span>
                    <span>{autoForm.objective}</span>
                  </div>
                </div>
              </div>
            ) : !runResult ? (
              <div className="empty">No calibration run yet.</div>
            ) : (
              <div className="run-output">
                <div className="run-summary">
                  <div className="run-summary-header">
                    <div>
                      <span className="meta-label">Output dir</span>
                      <div className="run-id">{runResult.out_dir}</div>
                    </div>
                    <span
                      className={`status-pill ${
                        runResult.ok ? "success" : "failed"
                      }`}
                    >
                      {runResult.ok ? "Success" : "Failed"}
                    </span>
                  </div>
                  <div className="run-meta-grid">
                    <div>
                      <span className="meta-label">Duration</span>
                      <span>{runResult.duration_s.toFixed(2)}s</span>
                    </div>
                    <div>
                      <span className="meta-label">Files</span>
                      <span>{runResult.files.length}</span>
                    </div>
                    <div>
                      <span className="meta-label">Dataset</span>
                      <span>{selectedDataset?.name ?? "Unknown"}</span>
                    </div>
                  </div>
                   {runResult.auto_out_dir ? (
                     <div className="run-extra">
                       <span className="meta-label">Auto log dir</span>
                       <span>{runResult.auto_out_dir}</span>
                     </div>
                   ) : null}
                </div>
                <div className="log-tabs">
                  <button
                    className={`log-tab ${
                      activeLog === "stdout" ? "active" : ""
                    }`}
                    type="button"
                    onClick={() => setActiveLog("stdout")}
                  >
                    stdout
                  </button>
                  <button
                    className={`log-tab ${
                      activeLog === "stderr" ? "active" : ""
                    }`}
                    type="button"
                    onClick={() => setActiveLog("stderr")}
                  >
                    stderr
                  </button>
                </div>
                <div className="log-block">
                  <span className="meta-label">
                    {activeLog === "stdout" ? "stdout" : "stderr"}
                  </span>
                  <pre>
                    {activeLog === "stdout"
                      ? runResult.stdout || "No stdout captured."
                      : runResult.stderr || "No stderr captured."}
                  </pre>
                </div>
                {runResult.metrics_summary ? (
                  <div className="metrics-summary">
                    <div className="metrics-summary-header">
                      <span className="meta-label">Model diagnostics</span>
                      <span className="metrics-summary-note">
                        Highlights vs baseline on each split.
                      </span>
                    </div>
                    <div className="metrics-summary-grid">
                      {Object.entries(runResult.metrics_summary).map(
                        ([split, summary]) => {
                          const delta = summary.delta_model_minus_baseline;
                          const deltaSign = delta > 0 ? "+" : "";
                          const splitLabel = SPLIT_LABELS[split] ?? split;
                          return (
                            <div
                              key={split}
                              className={`metrics-card status-${summary.status}`}
                            >
                              <div className="metrics-card-heading">
                                <span>{splitLabel}</span>
                                <span
                                  className={`status-pill ${
                                    summary.status === "good"
                                      ? "success"
                                      : "failed"
                                  }`}
                                >
                                  {summary.verdict.split(".")[0]}
                                </span>
                              </div>
                              <p className="metrics-card-verdict">
                                {summary.verdict}
                              </p>
                              <div className="metrics-card-row">
                                <span>Baseline logloss</span>
                                <strong>{summary.baseline_logloss.toFixed(4)}</strong>
                              </div>
                              <div className="metrics-card-row">
                                <span>Model logloss</span>
                                <strong>{summary.model_logloss.toFixed(4)}</strong>
                              </div>
                              <div className="metrics-card-row">
                                <span>Delta logloss</span>
                                <div className="metrics-delta-cell">
                                  <strong
                                    className={
                                      delta < 0 ? "delta-negative" : "delta-positive"
                                    }
                                  >
                                    {deltaSign}
                                    {delta.toFixed(4)}
                                  </strong>
                                  {summary.delta_logloss_ci_lo != null && summary.delta_logloss_ci_hi != null ? (
                                    <span className="ci-annotation">
                                      95% CI: [{summary.delta_logloss_ci_lo.toFixed(4)}, {summary.delta_logloss_ci_hi.toFixed(4)}]
                                    </span>
                                  ) : null}
                                </div>
                              </div>
                              {summary.baseline_brier != null && summary.model_brier != null ? (
                                <>
                                  <div className="metrics-card-row">
                                    <span>Baseline Brier</span>
                                    <strong>{summary.baseline_brier.toFixed(4)}</strong>
                                  </div>
                                  <div className="metrics-card-row">
                                    <span>Model Brier</span>
                                    <strong>{summary.model_brier.toFixed(4)}</strong>
                                  </div>
                                  {summary.delta_brier != null ? (
                                    <div className="metrics-card-row">
                                      <span>Delta Brier</span>
                                      <div className="metrics-delta-cell">
                                        <strong
                                          className={
                                            summary.delta_brier < 0 ? "delta-negative" : "delta-positive"
                                          }
                                        >
                                          {summary.delta_brier >= 0 ? "+" : ""}
                                          {summary.delta_brier.toFixed(4)}
                                        </strong>
                                        {summary.delta_brier_ci_lo != null && summary.delta_brier_ci_hi != null ? (
                                          <span className="ci-annotation">
                                            95% CI: [{summary.delta_brier_ci_lo.toFixed(4)}, {summary.delta_brier_ci_hi.toFixed(4)}]
                                          </span>
                                        ) : null}
                                      </div>
                                    </div>
                                  ) : null}
                                </>
                              ) : null}
                              {summary.baseline_ece != null && summary.model_ece != null ? (
                                <>
                                  <div className="metrics-card-row">
                                    <span>Baseline ECE</span>
                                    <strong>{summary.baseline_ece.toFixed(4)}</strong>
                                  </div>
                                  <div className="metrics-card-row">
                                    <span>Model ECE</span>
                                    <strong>{summary.model_ece.toFixed(4)}</strong>
                                  </div>
                                  {summary.delta_ece != null ? (
                                    <div className="metrics-card-row">
                                      <span>Delta ECE</span>
                                      <div className="metrics-delta-cell">
                                        <strong
                                          className={
                                            summary.delta_ece < 0 ? "delta-negative" : "delta-positive"
                                          }
                                        >
                                          {summary.delta_ece >= 0 ? "+" : ""}
                                          {summary.delta_ece.toFixed(4)}
                                        </strong>
                                        {summary.delta_ece_ci_lo != null && summary.delta_ece_ci_hi != null ? (
                                          <span className="ci-annotation">
                                            95% CI: [{summary.delta_ece_ci_lo.toFixed(4)}, {summary.delta_ece_ci_hi.toFixed(4)}]
                                          </span>
                                        ) : null}
                                      </div>
                                    </div>
                                  ) : null}
                                </>
                              ) : null}
                              {summary.baseline_ece_q != null && summary.model_ece_q != null ? (
                                <>
                                  <div className="metrics-card-row">
                                    <span>Baseline ECE-Q</span>
                                    <strong>{summary.baseline_ece_q.toFixed(4)}</strong>
                                  </div>
                                  <div className="metrics-card-row">
                                    <span>Model ECE-Q</span>
                                    <strong>{summary.model_ece_q.toFixed(4)}</strong>
                                  </div>
                                  {summary.delta_ece_q != null ? (
                                    <div className="metrics-card-row">
                                      <span>Delta ECE-Q</span>
                                      <div className="metrics-delta-cell">
                                        <strong
                                          className={
                                            summary.delta_ece_q < 0 ? "delta-negative" : "delta-positive"
                                          }
                                        >
                                          {summary.delta_ece_q >= 0 ? "+" : ""}
                                          {summary.delta_ece_q.toFixed(4)}
                                        </strong>
                                        {summary.delta_ece_q_ci_lo != null && summary.delta_ece_q_ci_hi != null ? (
                                          <span className="ci-annotation">
                                            95% CI: [{summary.delta_ece_q_ci_lo.toFixed(4)}, {summary.delta_ece_q_ci_hi.toFixed(4)}]
                                          </span>
                                        ) : null}
                                      </div>
                                    </div>
                                  ) : null}
                                </>
                              ) : null}
                              {summary.bootstrap_n_groups != null && summary.bootstrap_B != null ? (
                                <div className="ci-meta">
                                  {summary.bootstrap_n_groups} groups, B={summary.bootstrap_B}
                                </div>
                              ) : null}
                            </div>
                          );
                        },
                      )}
                    </div>
                  </div>
                ) : null}
                {runResult.model_equation ? (
                  <div className="equation-summary">
                    <span className="meta-label">pHAT equation</span>
                    <div className="equation-display">
                      <LatexEquation latex={runResult.model_equation} />
                    </div>
                  </div>
                ) : null}
                {runMode === "auto" && runResult.features && runResult.features.length > 0 ? (
                  <div className="features-summary">
                    <span className="meta-label">Features selected</span>
                    <div className="features-list">
                      {runResult.features.map((feature, idx) => (
                        <span key={idx} className="feature-chip">
                          {feature}
                        </span>
                      ))}
                    </div>
                  </div>
                ) : null}
                <details className="command-details">
                  <summary>Command used</summary>
                  <code>{runResult.command.join(" ")}</code>
                </details>
              </div>
            )}
          </div>
        </section>
      </form>

      <section className="panel">
        <div className="panel-header">
          <h2>Model registry</h2>
          <span className="panel-hint">Artifacts stored in src/data/models.</span>
        </div>
        <div className="panel-body">
          {modelError ? <div className="error">{modelError}</div> : null}
          {models.length === 0 ? (
            <div className="empty">No models found.</div>
          ) : (
            <div className="models-list">
              {models.map((model) => {
                const isSelected = selectedModelId === model.id;
                const showDetail = isSelected && modelDetail?.id === model.id;
                return (
                  <Fragment key={model.id}>
                    <div
                      className={`model-card ${isSelected ? "active" : ""}`}
                    >
                      <div>
                        {renamingModelId === model.id ? (
                          <div className="rename-input-wrapper">
                            <input
                              className="input rename-input"
                              type="text"
                              value={renameValue}
                              onChange={(e) => setRenameValue(e.target.value)}
                              onKeyDown={(e) => {
                                if (e.key === "Enter") {
                                  handleConfirmRename(model.id);
                                } else if (e.key === "Escape") {
                                  handleCancelRename();
                                }
                              }}
                              autoFocus
                            />
                            <div className="rename-actions">
                              <button
                                className="button ghost small"
                                type="button"
                                onClick={() => handleConfirmRename(model.id)}
                              >
                                Save
                              </button>
                              <button
                                className="button ghost small"
                                type="button"
                                onClick={handleCancelRename}
                              >
                                Cancel
                              </button>
                            </div>
                          </div>
                        ) : (
                          <>
                            <div className="model-title">{model.id}</div>
                            <div className="model-subtitle">
                              {formatTimestamp(model.last_modified)}
                            </div>
                          </>
                        )}
                      </div>
                      <div className="model-meta">
                        <span>{model.path}</span>
                        <span>
                          {model.has_metadata ? "metadata.json" : "no metadata"}
                        </span>
                        <span>{model.has_metrics ? "metrics.csv" : "no metrics"}</span>
                      </div>
                      <div className="model-card-actions">
                        <button
                          className="button ghost"
                          type="button"
                          disabled={
                            renamingModelId === model.id ||
                            deletingModelId === model.id
                          }
                          onClick={() => handleSelectModel(model.id)}
                        >
                          {isSelected ? "Hide" : "View"}
                        </button>
                        <button
                          className="button ghost"
                          type="button"
                          disabled={
                            renamingModelId === model.id ||
                            deletingModelId === model.id
                          }
                          onClick={() => handleStartRename(model.id)}
                        >
                          Rename
                        </button>
                        <button
                          className="button ghost danger"
                          type="button"
                          disabled={
                            renamingModelId === model.id ||
                            deletingModelId === model.id
                          }
                          onClick={() => handleDeleteModel(model.id)}
                        >
                          {deletingModelId === model.id ? "Deleting…" : "Delete"}
                        </button>
                      </div>
                    </div>
                    {isSelected ? (
                      <>
                        {modelDetailError ? (
                          <div className="error">{modelDetailError}</div>
                        ) : null}
                        {isModelDetailLoading ? (
                          <div className="empty">Loading model details…</div>
                        ) : null}
                        {showDetail ? (
                          <div className="model-detail">
                            <div className="model-detail-header">
                              <div>
                                <h3>Model overview</h3>
                                <span className="panel-hint">{modelDetail.path}</span>
                              </div>
                              <span className="meta-label">
                                {modelDetail.last_modified
                                  ? formatTimestamp(modelDetail.last_modified)
                                  : "Unknown time"}
                              </span>
                            </div>

                            <div className="model-detail-grid">
                              <div className="model-detail-section">
                                <span className="meta-label">Features used</span>
                                <div className="features-list">
                                  {modelDetail.features_used &&
                                  modelDetail.features_used.length > 0 ? (
                                    modelDetail.features_used.map((feature, idx) => (
                                      <span key={`feat-${idx}`} className="feature-chip">
                                        {feature}
                                      </span>
                                    ))
                                  ) : (
                                    <span className="empty">
                                      No numeric features recorded.
                                    </span>
                                  )}
                                </div>
                                <span className="meta-label">Categorical features</span>
                                <div className="features-list">
                                  {modelDetail.categorical_features_used &&
                                  modelDetail.categorical_features_used.length > 0 ? (
                                    modelDetail.categorical_features_used.map(
                                      (feature, idx) => (
                                        <span
                                          key={`cat-${idx}`}
                                          className="feature-chip"
                                        >
                                          {feature}
                                        </span>
                                      ),
                                    )
                                  ) : (
                                    <span className="empty">
                                      No categorical features recorded.
                                    </span>
                                  )}
                                </div>
                                {modelDetail.model_equation ? (
                                  <div className="equation-summary">
                                    <span className="meta-label">pHAT equation</span>
                                    <div className="equation-display">
                                      <LatexEquation latex={modelDetail.model_equation} />
                                    </div>
                                  </div>
                                ) : null}
                              </div>

                              <div className="model-detail-section">
                                <span className="meta-label">Model diagnostics</span>
                                {modelDetail.metrics_summary ? (
                                  <div className="metrics-summary-grid">
                                    {Object.entries(modelDetail.metrics_summary).map(
                                      ([split, summary]) => {
                                        const delta =
                                          summary.delta_model_minus_baseline;
                                        const deltaSign = delta > 0 ? "+" : "";
                                        const splitLabel = SPLIT_LABELS[split] ?? split;
                                        return (
                                          <div
                                            key={split}
                                            className={`metrics-card status-${summary.status}`}
                                          >
                                            <div className="metrics-card-heading">
                                              <span>{splitLabel}</span>
                                              <span
                                                className={`status-pill ${
                                                  summary.status === "good"
                                                    ? "success"
                                                    : "failed"
                                                }`}
                                              >
                                                {summary.verdict.split(".")[0]}
                                              </span>
                                            </div>
                                            <p className="metrics-card-verdict">
                                              {summary.verdict}
                                            </p>
                                            <div className="metrics-card-row">
                                              <span>Baseline logloss</span>
                                              <strong>
                                                {summary.baseline_logloss.toFixed(4)}
                                              </strong>
                                            </div>
                                            <div className="metrics-card-row">
                                              <span>Model logloss</span>
                                              <strong>
                                                {summary.model_logloss.toFixed(4)}
                                              </strong>
                                            </div>
                                            <div className="metrics-card-row">
                                              <span>Delta logloss</span>
                                              <div className="metrics-delta-cell">
                                                <strong
                                                  className={
                                                    delta < 0
                                                      ? "delta-negative"
                                                      : "delta-positive"
                                                  }
                                                >
                                                  {deltaSign}
                                                  {delta.toFixed(4)}
                                                </strong>
                                                {summary.delta_logloss_ci_lo != null && summary.delta_logloss_ci_hi != null ? (
                                                  <span className="ci-annotation">
                                                    95% CI: [{summary.delta_logloss_ci_lo.toFixed(4)}, {summary.delta_logloss_ci_hi.toFixed(4)}]
                                                  </span>
                                                ) : null}
                                              </div>
                                            </div>
                                            {summary.baseline_brier != null &&
                                            summary.model_brier != null ? (
                                              <>
                                                <div className="metrics-card-row">
                                                  <span>Baseline Brier</span>
                                                  <strong>
                                                    {summary.baseline_brier.toFixed(4)}
                                                  </strong>
                                                </div>
                                                <div className="metrics-card-row">
                                                  <span>Model Brier</span>
                                                  <strong>
                                                    {summary.model_brier.toFixed(4)}
                                                  </strong>
                                                </div>
                                                {summary.delta_brier != null ? (
                                                  <div className="metrics-card-row">
                                                    <span>Delta Brier</span>
                                                    <div className="metrics-delta-cell">
                                                      <strong
                                                        className={
                                                          summary.delta_brier < 0
                                                            ? "delta-negative"
                                                            : "delta-positive"
                                                        }
                                                      >
                                                        {summary.delta_brier >= 0 ? "+" : ""}
                                                        {summary.delta_brier.toFixed(4)}
                                                      </strong>
                                                      {summary.delta_brier_ci_lo != null && summary.delta_brier_ci_hi != null ? (
                                                        <span className="ci-annotation">
                                                          95% CI: [{summary.delta_brier_ci_lo.toFixed(4)}, {summary.delta_brier_ci_hi.toFixed(4)}]
                                                        </span>
                                                      ) : null}
                                                    </div>
                                                  </div>
                                                ) : null}
                                              </>
                                            ) : null}
                                            {summary.baseline_ece != null &&
                                            summary.model_ece != null ? (
                                              <>
                                                <div className="metrics-card-row">
                                                  <span>Baseline ECE</span>
                                                  <strong>
                                                    {summary.baseline_ece.toFixed(4)}
                                                  </strong>
                                                </div>
                                                <div className="metrics-card-row">
                                                  <span>Model ECE</span>
                                                  <strong>
                                                    {summary.model_ece.toFixed(4)}
                                                  </strong>
                                                </div>
                                                {summary.delta_ece != null ? (
                                                  <div className="metrics-card-row">
                                                    <span>Delta ECE</span>
                                                    <div className="metrics-delta-cell">
                                                      <strong
                                                        className={
                                                          summary.delta_ece < 0
                                                            ? "delta-negative"
                                                            : "delta-positive"
                                                        }
                                                      >
                                                        {summary.delta_ece >= 0 ? "+" : ""}
                                                        {summary.delta_ece.toFixed(4)}
                                                      </strong>
                                                      {summary.delta_ece_ci_lo != null && summary.delta_ece_ci_hi != null ? (
                                                        <span className="ci-annotation">
                                                          95% CI: [{summary.delta_ece_ci_lo.toFixed(4)}, {summary.delta_ece_ci_hi.toFixed(4)}]
                                                        </span>
                                                      ) : null}
                                                    </div>
                                                  </div>
                                                ) : null}
                                              </>
                                            ) : null}
                                            {summary.baseline_ece_q != null &&
                                            summary.model_ece_q != null ? (
                                              <>
                                                <div className="metrics-card-row">
                                                  <span>Baseline ECE-Q</span>
                                                  <strong>
                                                    {summary.baseline_ece_q.toFixed(4)}
                                                  </strong>
                                                </div>
                                                <div className="metrics-card-row">
                                                  <span>Model ECE-Q</span>
                                                  <strong>
                                                    {summary.model_ece_q.toFixed(4)}
                                                  </strong>
                                                </div>
                                                {summary.delta_ece_q != null ? (
                                                  <div className="metrics-card-row">
                                                    <span>Delta ECE-Q</span>
                                                    <div className="metrics-delta-cell">
                                                      <strong
                                                        className={
                                                          summary.delta_ece_q < 0
                                                            ? "delta-negative"
                                                            : "delta-positive"
                                                        }
                                                      >
                                                        {summary.delta_ece_q >= 0 ? "+" : ""}
                                                        {summary.delta_ece_q.toFixed(4)}
                                                      </strong>
                                                      {summary.delta_ece_q_ci_lo != null && summary.delta_ece_q_ci_hi != null ? (
                                                        <span className="ci-annotation">
                                                          95% CI: [{summary.delta_ece_q_ci_lo.toFixed(4)}, {summary.delta_ece_q_ci_hi.toFixed(4)}]
                                                        </span>
                                                      ) : null}
                                                    </div>
                                                  </div>
                                                ) : null}
                                              </>
                                            ) : null}
                                            {summary.bootstrap_n_groups != null && summary.bootstrap_B != null ? (
                                              <div className="ci-meta">
                                                {summary.bootstrap_n_groups} groups, B={summary.bootstrap_B}
                                              </div>
                                            ) : null}
                                          </div>
                                        );
                                      },
                                    )}
                                  </div>
                                ) : (
                                  <div className="empty">
                                    No metrics summary available.
                                  </div>
                                )}
                              </div>
                            </div>

                            <div className="model-detail-section">
                              <span className="meta-label">Training parameters</span>
                              <div className="param-grid">
                                {modelMetadata ? (
                                  [
                                    { label: "Mode", value: modelMetadata["mode"] },
                                    {
                                      label: "Calibration",
                                      value:
                                        modelMetadata["calibration_used"] ??
                                        modelMetadata["calibration_requested"],
                                    },
                                    {
                                      label: "Selection objective",
                                      value: modelMetadata["selection_objective"],
                                    },
                                    {
                                      label: "Fallback to baseline",
                                      value: modelMetadata["fallback_to_baseline_triggered"],
                                    },
                                    {
                                      label: "Deployed model",
                                      value: modelMetadata["model_kind_deployed"],
                                    },
                                    { label: "Best C", value: modelMetadata["best_C"] },
                                    {
                                      label: "Ticker intercepts",
                                      value: modelMetadata["ticker_intercepts"],
                                    },
                                    {
                                      label: "Ticker interactions",
                                      value: modelMetadata["ticker_x_interactions"],
                                    },
                                    {
                                      label: "Foundation tickers",
                                      value: modelMetadata["foundation_tickers"],
                                    },
                                    {
                                      label: "Foundation weight",
                                      value: modelMetadata["foundation_weight"],
                                    },
                                    {
                                      label: "Train decay half-life (weeks)",
                                      value: modelFitWeights?.[
                                        "train_decay_half_life_weeks"
                                      ],
                                    },
                                    {
                                      label: "Fit weight renorm",
                                      value: modelFitWeights?.["fit_weight_renorm"],
                                    },
                                    {
                                      label: "Group reweight",
                                      value: modelMetadata["group_reweight"],
                                    },
                                    {
                                      label: "Moneyness column",
                                      value: modelMetadata["moneyness_column_used"],
                                    },
                                    {
                                      label: "Enable x_abs_m",
                                      value: modelMetadata["enable_x_abs_m"],
                                    },
                                    {
                                      label: "Max |log m|",
                                      value: optionalFilters?.["max_abs_logm"],
                                    },
                                    {
                                      label: "Drop pRN extremes",
                                      value: optionalFilters?.["drop_prn_extremes"],
                                    },
                                    {
                                      label: "pRN epsilon",
                                      value: optionalFilters?.["prn_eps"],
                                    },
                                    {
                                      label: "Rows filtered",
                                      value: optionalFilters?.["rows_filtered"],
                                    },
                                    {
                                      label: "ECE-Q bins",
                                      value: modelMetadata["eceq_bins"],
                                    },
                                    {
                                      label: "T_days allowed",
                                      value: modelMetadata["tdays_allowed"],
                                    },
                                    {
                                      label: "As-of DOW allowed",
                                      value: modelMetadata["asof_dow_allowed"],
                                    },
                                    {
                                      label: "T_days constant",
                                      value: modelMetadata["tdays_constant"],
                                    },
                                    {
                                      label: "As-of DOW constant",
                                      value: modelMetadata["asof_dow_constant"],
                                    },
                                    {
                                      label: "Train end date",
                                      value: modelMetadata["train_end_date"],
                                    },
                                    {
                                      label: "Training timestamp",
                                      value: modelMetadata["training_timestamp"],
                                    },
                                    {
                                      label: "Forbidden features removed",
                                      value: modelMetadata["forbidden_features_removed"],
                                    },
                                    {
                                      label: "Near-constant features removed",
                                      value: modelMetadata["near_constant_features_removed"],
                                    },
                                    {
                                      label: "Forward fallback rows dropped",
                                      value: modelMetadata["forward_fallback_rows_dropped"],
                                    },
                                  ].map((item) => (
                                    <div key={item.label} className="param-row">
                                      <span className="param-label">{item.label}</span>
                                      <span className="param-value">
                                        {formatParamValue(item.value)}
                                      </span>
                                    </div>
                                  ))
                                ) : (
                                  <div className="empty">No metadata.json available.</div>
                                )}
                              </div>
                              {modelMetadata ? (
                                <details className="command-details">
                                  <summary>Raw metadata</summary>
                                  <pre>{JSON.stringify(modelMetadata, null, 2)}</pre>
                                </details>
                              ) : null}
                            </div>

                            {modelFiles && modelFiles.files.length > 0 ? (
                              <div className="model-detail-section file-viewer-section">
                                <span className="meta-label">Model files</span>
                                <div className="file-list">
                                  {modelFiles.files.map((file) => (
                                    <button
                                      key={file.name}
                                      type="button"
                                      className={`file-item ${selectedFile === file.name ? "active" : ""} ${!file.is_viewable ? "too-large" : ""}`}
                                      onClick={() => file.is_viewable && handleFileSelect(file.name)}
                                      disabled={!file.is_viewable}
                                      title={file.is_viewable ? `View ${file.name}` : `File too large (${(file.size_bytes / 1024).toFixed(1)} KB)`}
                                    >
                                      <span className="file-name">{file.name}</span>
                                      <span className="file-size">
                                        {file.size_bytes < 1024
                                          ? `${file.size_bytes} B`
                                          : `${(file.size_bytes / 1024).toFixed(1)} KB`}
                                      </span>
                                    </button>
                                  ))}
                                </div>
                                {selectedFile ? (
                                  <div className="file-content-panel">
                                    <div className="file-content-header">
                                      <span className="file-content-title">{selectedFile}</span>
                                      {fileContent?.truncated ? (
                                        <span className="file-truncated-badge">Truncated</span>
                                      ) : null}
                                      <button
                                        type="button"
                                        className="button small"
                                        onClick={() => {
                                          setSelectedFile(null);
                                          setFileContent(null);
                                          setFileError(null);
                                        }}
                                      >
                                        Close
                                      </button>
                                    </div>
                                    {isFileLoading ? (
                                      <div className="empty">Loading file…</div>
                                    ) : fileError ? (
                                      <div className="error">{fileError}</div>
                                    ) : fileContent ? (
                                      <pre className={`file-content file-type-${fileContent.content_type}`}>
                                        {fileContent.content}
                                      </pre>
                                    ) : null}
                                  </div>
                                ) : null}
                              </div>
                            ) : null}
                          </div>
                        ) : null}
                      </>
                    ) : null}
                  </Fragment>
                );
              })}
            </div>
          )}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>CLI preview</h2>
          <span className="panel-hint">Mirrors the command that will run.</span>
        </div>
        <div className="panel-body">
          <div className="command-preview">
            <span className="meta-label">Command</span>
            <pre>{commandPreview}</pre>
          </div>
        </div>
      </section>
    </section>
  );
}
