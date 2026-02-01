import { useEffect, useMemo, useState, type FormEvent } from "react";

import {
  fetchCalibrationDatasets,
  fetchCalibrationModels,
  deleteCalibrationModel,
  runCalibration,
  type CalibrateModelRunResponse,
  type DatasetFileSummary,
  type ModelRunSummary,
} from "../api/calibrateModels";
import "./CalibrateModelsPage.css";

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
};

const featureCatalog = [
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
    key: "log_rel_spread",
    label: "Log relative spread",
    description: "Log of relative bid/ask spread.",
  },
  {
    key: "chain_used_frac",
    label: "Chain retention",
    description: "Fraction of option chain retained after filtering.",
  },
  {
    key: "band_inside_frac",
    label: "Band coverage",
    description: "Share of strikes inside the pRN band.",
  },
  {
    key: "drop_intrinsic_frac",
    label: "Intrinsic drop",
    description: "Fraction dropped for intrinsic value checks.",
  },
  {
    key: "asof_fallback_days",
    label: "As-of fallback",
    description: "Days used for as-of price fallback.",
  },
  {
    key: "split_events_in_preload_range",
    label: "Split count",
    description: "Count of split events in preload window.",
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

const featureLabelByKey: Record<string, string> = featureCatalog.reduce(
  (acc, feature) => {
    acc[feature.key] = feature.label;
    return acc;
  },
  {} as Record<string, string>,
);

const defaultFeatures = featureCatalog.map((feature) => feature.key).join(",");

const defaultModelName = () => {
  const stamp = new Date().toISOString().replace(/[:.]/g, "").slice(0, 15);
  return `calibration-${stamp}`;
};

const defaultForm: CalibrateFormState = {
  datasetPath: "",
  outName: "",
  targetCol: "",
  weekCol: "week_friday",
  tickerCol: "ticker",
  weightCol: "",
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
};

const STORAGE_KEY = "polyedgetool.calibrate.form";

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

const buildCommandPreview = (
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
  addValue("--target-col", state.targetCol);
  addValue("--week-col", state.weekCol);
  addValue("--ticker-col", state.tickerCol);
  addValue("--weight-col", state.weightCol);
  const selected = state.selectedFeatures;
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
  addValue("--random-state", state.randomState);

  return args.join(" ");
};

export default function CalibrateModelsPage() {
  const [defaultName] = useState(() => defaultModelName());
  const [formState, setFormState] = useState<CalibrateFormState>(defaultForm);
  const [datasets, setDatasets] = useState<DatasetFileSummary[]>([]);
  const [models, setModels] = useState<ModelRunSummary[]>([]);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [runResult, setRunResult] = useState<CalibrateModelRunResponse | null>(
    null,
  );
  const [activeLog, setActiveLog] = useState<"stdout" | "stderr">("stdout");
  const [storageReady, setStorageReady] = useState(false);
  const [deletingModelId, setDeletingModelId] = useState<string | null>(null);

  const selectedDataset = useMemo(
    () => datasets.find((item) => item.path === formState.datasetPath),
    [datasets, formState.datasetPath],
  );

  const commandPreview = useMemo(
    () => buildCommandPreview(formState, defaultName),
    [formState, defaultName],
  );

  const selectedFeatureLabels = useMemo(
    () => formState.selectedFeatures,
    [formState.selectedFeatures],
  );

  const effectiveOutName = formState.outName.trim() || defaultName;

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
  }, [runResult]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setRunError(null);
    setRunResult(null);
    setIsRunning(true);

    try {
      if (!formState.datasetPath) {
        throw new Error("Select a dataset to calibrate.");
      }
      const payload = {
        csv: formState.datasetPath,
        outName: formState.outName.trim() || undefined,
        targetCol: formState.targetCol.trim() || undefined,
        weekCol: formState.weekCol.trim() || undefined,
        tickerCol: formState.tickerCol.trim() || undefined,
        weightCol: formState.weightCol.trim() || undefined,
        features:
          [
            ...formState.selectedFeatures,
            ...formState.customFeatures
              .split(",")
              .map((value) => value.trim())
              .filter(Boolean),
          ].join(",") || undefined,
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
        randomState: parseOptionalInt(formState.randomState),
      };

      const result = await runCalibration(payload);
      setRunResult(result);
      refreshModels();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setRunError(message);
    } finally {
      setIsRunning(false);
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
        <div className="meta-card calibrate-meta">
          <span className="meta-label">Pipeline stage</span>
          <span>2-calibrate-logit-model-v1.5.py</span>
          <div className="meta-pill">Outputs to src/data/models</div>
        </div>
      </header>

      <div className="calibrate-grid">
        <section className="panel">
          <div className="panel-header">
            <h2>Calibration configuration</h2>
            <span className="panel-hint">
              Dataset selection is restricted to option-chain outputs.
            </span>
          </div>
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
          <form className="panel-body" onSubmit={handleSubmit}>
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
                {datasetError ? (
                  <div className="error">{datasetError}</div>
                ) : null}
              </div>
              {selectedDataset ? (
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

            <details className="advanced">
              <summary>Schema & feature controls</summary>
              <div className="fields-grid">
                <div className="field">
                  <label htmlFor="targetCol">Target column</label>
                  <input
                    id="targetCol"
                    className="input"
                    value={formState.targetCol}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        targetCol: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="weekCol">Week column</label>
                  <input
                    id="weekCol"
                    className="input"
                    value={formState.weekCol}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        weekCol: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="tickerCol">Ticker column</label>
                  <input
                    id="tickerCol"
                    className="input"
                    value={formState.tickerCol}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        tickerCol: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="weightCol">Weight column</label>
                  <input
                    id="weightCol"
                    className="input"
                    value={formState.weightCol}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        weightCol: event.target.value,
                      }))
                    }
                  />
                </div>
              </div>
              <div className="feature-grid">
                {featureCatalog.map((feature) => (
                  <label key={feature.key} className="feature-item">
                    <input
                      type="checkbox"
                      checked={formState.selectedFeatures.includes(feature.key)}
                      onChange={(event) => {
                        const checked = event.target.checked;
                        setFormState((prev) => {
                          const next = new Set(prev.selectedFeatures);
                          if (checked) {
                            next.add(feature.key);
                          } else {
                            next.delete(feature.key);
                          }
                          return { ...prev, selectedFeatures: Array.from(next) };
                        });
                      }}
                    />
                    <div>
                      <div className="feature-title">{feature.key}</div>
                      <div className="feature-desc">{feature.description}</div>
                    </div>
                  </label>
                ))}
              </div>
              <div className="field">
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
              <div className="field">
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
              <label className="checkbox">
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
                Add interactions
              </label>
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
            </details>

            {runError ? <div className="error">{runError}</div> : null}

            <div className="actions">
              <button
                className="button primary"
                type="submit"
                disabled={isRunning || !formState.datasetPath}
              >
                {isRunning ? "Calibrating..." : "Run calibration"}
              </button>
              <button
                className="button ghost"
                type="button"
                disabled={isRunning}
                onClick={() => setFormState(defaultForm)}
              >
                Reset
              </button>
            </div>
          </form>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>Latest run output</h2>
            <span className="panel-hint">
              Captures stdout/stderr from the calibration script.
            </span>
          </div>
          <div className="panel-body">
            {!runResult ? (
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
                      <span className="meta-label">Dataset</span>
                      <span>{selectedDataset?.name ?? "Unknown"}</span>
                    </div>
                  </div>
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
                <details className="command-details">
                  <summary>Command used</summary>
                  <code>{runResult.command.join(" ")}</code>
                </details>
              </div>
            )}
          </div>
        </section>
      </div>

      <section className="panel">
        <div className="panel-header">
          <h2>Model registry</h2>
          <span className="panel-hint">Artifacts stored in src/data/models.</span>
        </div>
        <div className="panel-body">
          <div className="panel-actions">
            <button className="button ghost" type="button" onClick={refreshModels}>
              Refresh list
            </button>
            <button
              className="button ghost"
              type="button"
              onClick={refreshDatasets}
            >
              Refresh datasets
            </button>
          </div>
          {modelError ? <div className="error">{modelError}</div> : null}
          {models.length === 0 ? (
            <div className="empty">No models found.</div>
          ) : (
            <div className="models-list">
              {models.map((model) => (
                <div key={model.id} className="model-card">
                  <div>
                    <div className="model-title">{model.id}</div>
                    <div className="model-subtitle">
                      {formatTimestamp(model.last_modified)}
                    </div>
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
                      className="button ghost danger"
                      type="button"
                      disabled={deletingModelId === model.id}
                      onClick={() => handleDeleteModel(model.id)}
                    >
                      {deletingModelId === model.id ? "Deleting…" : "Delete"}
                    </button>
                  </div>
                </div>
              ))}
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
