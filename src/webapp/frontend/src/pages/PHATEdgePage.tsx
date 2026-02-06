import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";

import {
  fetchCalibrationModels,
  type ModelRunSummary,
} from "../api/calibrateModels";
import {
  fetchPolymarketSnapshotRuns,
  type PolymarketSnapshotRunSummary,
} from "../api/polymarketSnapshots";
import {
  deletePhatEdgeRun,
  fetchPhatEdgePreview,
  fetchPhatEdgeRows,
  fetchPhatEdgeRuns,
  fetchPhatEdgeSummary,
  startPhatEdgeJob,
  type PHATEdgeFileSummary,
  type PHATEdgePreviewResponse,
  type PHATEdgeRowsResponse,
  type PHATEdgeRunResponse,
  type PHATEdgeSummaryResponse,
} from "../api/phatEdge";
import { usePhatEdgeJob } from "../contexts/phatEdgeJob";
import { useAnyJobRunning } from "../contexts/jobGuard";
import "./PHATEdgePage.css";

type SnapshotOption = {
  label: string;
  value: string;
  runId: string;
  timestamp: string | null;
  contractType: "weekly" | "1dte" | "legacy";
};

type FormState = {
  modelId: string;
  snapshotCsv: string;
  snapshotContractType: "weekly" | "1dte";
  outCsv: string;
  requireColumnsStrict: boolean;
  computeEdge: boolean;
  skipEdgeOutsidePrnRange: boolean;
};

const STORAGE_KEY = "polyedgetool.phat-edge.form";
const RUN_STORAGE_KEY = "polyedgetool.phat-edge.latestRun";
const DEFAULT_KELLY_PRICE = "0.50";
const DEFAULT_KELLY_PROB = "0.55";

const defaultForm: FormState = {
  modelId: "",
  snapshotCsv: "",
  snapshotContractType: "weekly",
  outCsv: "",
  requireColumnsStrict: true,
  computeEdge: true,
  skipEdgeOutsidePrnRange: true,
};

const pickSnapshotFileName = (files: string[]): string | null => {
  if (files.length === 0) {
    return null;
  }
  const dataset = files.find((value) =>
    value.toLowerCase().includes("dataset"),
  );
  return dataset ?? files[0];
};

const formatBytes = (value: number): string => {
  if (value < 1024) return `${value} B`;
  const kb = value / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(1)} MB`;
};

const formatTimestamp = (value?: string | null) => {
  if (!value) {
    return "Unknown";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
};

const loadStoredForm = (): FormState | null => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return null;
    const parsed = JSON.parse(stored);
    if (typeof parsed !== "object" || parsed === null) return null;
    if ("excludeTickers" in parsed) {
      delete parsed.excludeTickers;
    }
    return {
      ...defaultForm,
      ...parsed,
    } as FormState;
  } catch {
    return null;
  }
};

const loadStoredRun = (): PHATEdgeRunResponse | null => {
  try {
    const raw = localStorage.getItem(RUN_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    if (typeof (parsed as { ok?: unknown }).ok !== "boolean") return null;
    if (!Array.isArray((parsed as { command?: unknown }).command)) return null;
    if (
      typeof (parsed as { run_summary?: unknown }).run_summary !== "object" ||
      (parsed as { run_summary?: unknown }).run_summary === null
    ) {
      return null;
    }
    return parsed as PHATEdgeRunResponse;
  } catch {
    return null;
  }
};

const buildSnapshotOptions = (
  runs: PolymarketSnapshotRunSummary[],
): SnapshotOption[] =>
  runs
    .map((run) => {
      const datasetFile = pickSnapshotFileName(run.files);
      if (!datasetFile || !run.run_dir) return null;
      const contractType: SnapshotOption["contractType"] =
        run.run_dir.includes("/runs/1dte/") ? "1dte"
        : run.run_dir.includes("/runs/weekly/") ? "weekly"
        : "legacy";
      return {
        label: `${contractType === "legacy" ? "weekly" : contractType} · ${run.run_id ?? "run"} · ${datasetFile}`,
        value: `${run.run_dir}/${datasetFile}`,
        runId: run.run_id,
        timestamp: run.run_time_utc ?? run.last_modified,
        contractType,
      };
    })
    .filter(
      (entry): entry is SnapshotOption => entry !== null,
    );

const parseDecimal = (value: string): number | null => {
  if (!value) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const isValidProbability = (value: number | null): value is number =>
  value !== null && value > 0 && value < 1;

const formatPercent = (value: number | null, digits = 2): string => {
  if (value === null) return "--";
  return `${(value * 100).toFixed(digits)}%`;
};

const formatNumber = (value: number | null, digits = 2): string => {
  if (value === null) return "--";
  return value.toFixed(digits);
};

export default function PHATEdgePage() {
  const [formState, setFormState] = useState<FormState>(() => {
    return loadStoredForm() ?? defaultForm;
  });
  const [models, setModels] = useState<ModelRunSummary[]>([]);
  const [modelError, setModelError] = useState<string | null>(null);
  const [snapshots, setSnapshots] = useState<SnapshotOption[]>([]);
  const [snapshotRuns, setSnapshotRuns] = useState<PolymarketSnapshotRunSummary[]>([]);
  const [snapshotError, setSnapshotError] = useState<string | null>(null);
  const [runResult, setRunResult] =
    useState<PHATEdgeRunResponse | null>(() => loadStoredRun());
  const [runError, setRunError] = useState<string | null>(null);
  const [edgePreview, setEdgePreview] = useState<PHATEdgePreviewResponse | null>(null);
  const [edgePreviewError, setEdgePreviewError] = useState<string | null>(null);
  const [edgePreviewLoading, setEdgePreviewLoading] = useState(false);
  const lastEdgePreviewPathRef = useRef<string | null>(null);
  const [edgeRuns, setEdgeRuns] = useState<PHATEdgeFileSummary[]>([]);
  const [edgeRunsError, setEdgeRunsError] = useState<string | null>(null);
  const [selectedEdgeRunPath, setSelectedEdgeRunPath] = useState<string | null>(null);
  const [edgeSummary, setEdgeSummary] = useState<PHATEdgeSummaryResponse | null>(null);
  const [edgeSummaryError, setEdgeSummaryError] = useState<string | null>(null);
  const [edgeSummaryLoading, setEdgeSummaryLoading] = useState(false);
  const [edgeRows, setEdgeRows] = useState<PHATEdgeRowsResponse | null>(null);
  const [edgeRowsError, setEdgeRowsError] = useState<string | null>(null);
  const [edgeRowsLoading, setEdgeRowsLoading] = useState(false);
  const lastEdgeRowsPathRef = useRef<string | null>(null);
  const [activeLog, setActiveLog] = useState<"stdout" | "stderr">("stdout");
  const lastRunResultKeyRef = useRef<string | null>(null);
  const [kellySide, setKellySide] = useState<"yes" | "no">("yes");
  const [kellyPriceInput, setKellyPriceInput] =
    useState(DEFAULT_KELLY_PRICE);
  const [kellyProbInput, setKellyProbInput] = useState(DEFAULT_KELLY_PROB);
  const [kellyBankrollInput, setKellyBankrollInput] = useState("");
  const [deletingRunId, setDeletingRunId] = useState<string | null>(null);
  const [runDeleteError, setRunDeleteError] = useState<string | null>(null);
  const [edgesPerTickerLimit, setEdgesPerTickerLimit] =
    useState<"all" | 1 | 3>("all");
  const [excludePrnOutOfRange, setExcludePrnOutOfRange] = useState(false);
  const { jobStatus, setJobId, setJobStatus } = usePhatEdgeJob();
  const { anyJobRunning, primaryJob } = useAnyJobRunning();

  const isRunning =
    jobStatus?.status === "queued" || jobStatus?.status === "running";

  useEffect(() => {
    fetchCalibrationModels()
      .then((payload) => {
        setModels(payload.models);
        setModelError(null);
      })
      .catch((err: Error) => {
        setModelError(err.message);
      });
  }, []);

  useEffect(() => {
    if (models.length === 0) return;
    setFormState((prev) => {
      const already = models.some((model) => model.id === prev.modelId);
      if (already) {
        return prev;
      }
      return { ...prev, modelId: models[0].id };
    });
  }, [models]);

  const refreshSnapshotRuns = useCallback(() => {
    fetchPolymarketSnapshotRuns()
      .then((payload) => {
        setSnapshotRuns(payload.runs);
        setSnapshots(buildSnapshotOptions(payload.runs));
        setSnapshotError(null);
      })
      .catch((err: Error) => {
        setSnapshotError(err.message);
      });
  }, []);

  useEffect(() => {
    refreshSnapshotRuns();
  }, [refreshSnapshotRuns]);

  const refreshEdgeRuns = useCallback(() => {
    fetchPhatEdgeRuns()
      .then((payload) => {
        setEdgeRuns(payload.runs);
        setEdgeRunsError(null);
      })
      .catch((err: Error) => {
        setEdgeRunsError(err.message);
      });
  }, []);

  useEffect(() => {
    refreshEdgeRuns();
  }, [refreshEdgeRuns]);

  const filteredSnapshots = useMemo(() => {
    return snapshots.filter((snapshot) =>
      formState.snapshotContractType === "weekly"
        ? snapshot.contractType === "weekly" || snapshot.contractType === "legacy"
        : snapshot.contractType === "1dte",
    );
  }, [snapshots, formState.snapshotContractType]);

  useEffect(() => {
    setFormState((prev) => {
      if (filteredSnapshots.length === 0) {
        return prev.snapshotCsv ? { ...prev, snapshotCsv: "" } : prev;
      }
      if (prev.snapshotCsv && filteredSnapshots.some((snapshot) => snapshot.value === prev.snapshotCsv)) {
        return prev;
      }
      return { ...prev, snapshotCsv: filteredSnapshots[0].value };
    });
  }, [filteredSnapshots]);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(formState));
    } catch {
      // ignore storage failures
    }
  }, [formState]);

  useEffect(() => {
    if (!runResult) return;
    try {
      localStorage.setItem(RUN_STORAGE_KEY, JSON.stringify(runResult));
    } catch {
      // ignore storage failures
    }
  }, [runResult]);

  useEffect(() => {
    if (!runResult) return;
    const runSummary = runResult.run_summary;
    const runKey = runSummary
      ? `${runSummary.output_csv}|${runSummary.snapshot_csv}|${runResult.ok}`
      : null;
    if (runKey && runKey === lastRunResultKeyRef.current) return;
    if (runKey) {
      lastRunResultKeyRef.current = runKey;
    }
    if (!runResult.ok && runResult.stderr) {
      setActiveLog("stderr");
    } else {
      setActiveLog("stdout");
    }
  }, [runResult]);

  useEffect(() => {
    if (runResult?.ok && runResult.run_summary?.output_csv) {
      setSelectedEdgeRunPath(runResult.run_summary.output_csv);
      refreshEdgeRuns();
      return;
    }
    if (runResult && !runResult.ok) {
      refreshEdgeRuns();
    }
  }, [runResult, refreshEdgeRuns]);

  useEffect(() => {
    if (selectedEdgeRunPath && edgeRuns.some((run) => run.path === selectedEdgeRunPath)) {
      return;
    }
    if (edgeRuns.length > 0) {
      setSelectedEdgeRunPath(edgeRuns[0].path);
    }
  }, [edgeRuns, selectedEdgeRunPath]);

  const outputCsvPath = selectedEdgeRunPath;

  useEffect(() => {
    if (!outputCsvPath) {
      setEdgePreview(null);
      setEdgePreviewError(null);
      setEdgePreviewLoading(false);
      lastEdgePreviewPathRef.current = null;
      return;
    }
    if (lastEdgePreviewPathRef.current === outputCsvPath && edgePreview) {
      return;
    }
    lastEdgePreviewPathRef.current = outputCsvPath;
    let isMounted = true;
    setEdgePreviewLoading(true);
    setEdgePreviewError(null);
    fetchPhatEdgePreview(outputCsvPath, 18, "head")
      .then((data) => {
        if (!isMounted) return;
        setEdgePreview(data);
        setEdgePreviewError(null);
      })
      .catch((err: Error) => {
        if (!isMounted) return;
        setEdgePreview(null);
        setEdgePreviewError(err.message);
        if (err.message.toLowerCase().includes("path not found")) {
          setSelectedEdgeRunPath(null);
          refreshEdgeRuns();
        }
      })
      .finally(() => {
        if (!isMounted) return;
        setEdgePreviewLoading(false);
      });
    return () => {
      isMounted = false;
    };
  }, [outputCsvPath, edgePreview, refreshEdgeRuns]);

  useEffect(() => {
    if (!outputCsvPath) {
      setEdgeSummary(null);
      setEdgeSummaryError(null);
      setEdgeSummaryLoading(false);
      return;
    }
    let isMounted = true;
    setEdgeSummaryLoading(true);
    setEdgeSummaryError(null);
    fetchPhatEdgeSummary(outputCsvPath)
      .then((data) => {
        if (!isMounted) return;
        setEdgeSummary(data);
        setEdgeSummaryError(null);
      })
      .catch((err: Error) => {
        if (!isMounted) return;
        setEdgeSummary(null);
        setEdgeSummaryError(err.message);
        if (err.message.toLowerCase().includes("path not found")) {
          setSelectedEdgeRunPath(null);
          refreshEdgeRuns();
        }
      })
      .finally(() => {
        if (!isMounted) return;
        setEdgeSummaryLoading(false);
      });
    return () => {
      isMounted = false;
    };
  }, [outputCsvPath, refreshEdgeRuns]);

  useEffect(() => {
    if (!outputCsvPath) {
      setEdgeRows(null);
      setEdgeRowsError(null);
      setEdgeRowsLoading(false);
      lastEdgeRowsPathRef.current = null;
      return;
    }
    if (lastEdgeRowsPathRef.current === outputCsvPath && edgeRows) {
      return;
    }
    lastEdgeRowsPathRef.current = outputCsvPath;
    let isMounted = true;
    setEdgeRowsLoading(true);
    setEdgeRowsError(null);
    fetchPhatEdgeRows(outputCsvPath)
      .then((data) => {
        if (!isMounted) return;
        setEdgeRows(data);
        setEdgeRowsError(null);
      })
      .catch((err: Error) => {
        if (!isMounted) return;
        setEdgeRows(null);
        setEdgeRowsError(err.message);
        if (err.message.toLowerCase().includes("path not found")) {
          setSelectedEdgeRunPath(null);
          refreshEdgeRuns();
        }
      })
      .finally(() => {
        if (!isMounted) return;
        setEdgeRowsLoading(false);
      });
    return () => {
      isMounted = false;
    };
  }, [outputCsvPath, edgeRows, refreshEdgeRuns]);

  useEffect(() => {
    if (!jobStatus) return;
    if (jobStatus.result) {
      setRunResult(jobStatus.result);
    }
    if (jobStatus.status === "failed" && jobStatus.error) {
      setRunError(jobStatus.error);
    }
  }, [jobStatus]);

  const selectedModel = useMemo(
    () => models.find((model) => model.id === formState.modelId) ?? null,
    [models, formState.modelId],
  );

  const selectedSnapshot = useMemo(
    () => filteredSnapshots.find((snapshot) => snapshot.value === formState.snapshotCsv) ?? null,
    [filteredSnapshots, formState.snapshotCsv],
  );

  const kellyMetrics = useMemo(() => {
    const price = parseDecimal(kellyPriceInput);
    const probability = parseDecimal(kellyProbInput);
    const bankroll = parseDecimal(kellyBankrollInput);
    if (!isValidProbability(price) || !isValidProbability(probability)) {
      return {
        price: null,
        probability: null,
        bankroll: bankroll && bankroll > 0 ? bankroll : null,
        edge: null,
        kellyFraction: null,
        suggestedFraction: null,
        stake: null,
        decimalOdds: null,
      };
    }
    const b = (1 - price) / price;
    const q = 1 - probability;
    const kellyFraction = (b * probability - q) / b;
    const suggestedFraction = Math.max(0, Math.min(1, kellyFraction));
    const edge = probability - price;
    const decimalOdds = 1 / price;
    const stake =
      bankroll && bankroll > 0 ? bankroll * suggestedFraction : null;
    return {
      price,
      probability,
      bankroll: bankroll && bankroll > 0 ? bankroll : null,
      edge,
      kellyFraction,
      suggestedFraction,
      stake,
      decimalOdds,
    };
  }, [kellyPriceInput, kellyProbInput, kellyBankrollInput]);

  const displayedEdgeRows = useMemo(() => {
    const rows = edgeRows?.rows ?? edgeSummary?.top_edges ?? runResult?.top_edges ?? [];
    if (rows.length === 0) return [];
    const filtered = excludePrnOutOfRange
      ? rows.filter((row) => row.edge_source !== "pRN_out_of_range")
      : rows;
    const sorted = [...filtered].sort((a, b) => {
      const aEdge = a.edge ?? Number.NEGATIVE_INFINITY;
      const bEdge = b.edge ?? Number.NEGATIVE_INFINITY;
      return bEdge - aEdge;
    });
    if (edgesPerTickerLimit === "all") {
      return sorted;
    }
    const limit = edgesPerTickerLimit;
    const counts = new Map<string, number>();
    return sorted.filter((row) => {
      const key = row.ticker?.toUpperCase?.() ?? "UNKNOWN";
      const count = counts.get(key) ?? 0;
      if (count >= limit) return false;
      counts.set(key, count + 1);
      return true;
    });
  }, [
    edgeRows,
    edgeSummary,
    runResult,
    excludePrnOutOfRange,
    edgesPerTickerLimit,
  ]);

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
    if (!selectedModel) {
      setRunError("Select a trained model before running inference.");
      return;
    }
    if (!formState.snapshotCsv) {
      setRunError("Select a snapshot CSV to score.");
      return;
    }
    try {
      const payload = {
        model_path: `${selectedModel.path}/model.joblib`,
        snapshot_csv: formState.snapshotCsv,
        out_csv: formState.outCsv.trim() || undefined,
        require_columns_strict: formState.requireColumnsStrict,
        compute_edge: formState.computeEdge,
        skip_edge_outside_prn_range: formState.skipEdgeOutsidePrnRange,
      };
      const status = await startPhatEdgeJob(payload);
      setJobId(status.job_id);
      setJobStatus(status);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setRunError(message);
    }
  };

  const handleDeleteEdgeRun = useCallback(
    async (path: string) => {
      if (!window.confirm(`Delete edge run ${path}?`)) {
        return;
      }
      setRunDeleteError(null);
      setDeletingRunId(path);
      try {
        await deletePhatEdgeRun(path);
        refreshEdgeRuns();
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unknown error";
        setRunDeleteError(message);
      } finally {
        setDeletingRunId((prev) => (prev === path ? null : prev));
      }
    },
    [refreshEdgeRuns],
  );

  const pHatStats = edgeSummary?.pHat_distribution ?? runResult?.pHat_distribution;
  const edgeStats = edgeSummary?.edge_distribution ?? runResult?.edge_distribution;
  const edgeRowCount = edgeRows?.row_count ?? edgeRows?.rows.length ?? 0;
  const edgesPerTickerLabel =
    edgesPerTickerLimit === "all"
      ? "All edges"
      : `${edgesPerTickerLimit} edge${edgesPerTickerLimit === 1 ? "" : "s"} per ticker`;

  return (
    <section className="page phat-edge-page">
      <header className="page-header">
        <div>
          <p className="page-kicker">Edge</p>
          <h1 className="page-title">Score a snapshot with a saved calibration</h1>
          <p className="page-subtitle">
            Load an artifact from <code>src/data/models</code>, pick a Polymarket snapshot, and emit a queue of pHAT + edge signals for the trading desk.
          </p>
        </div>
        <div className="meta-card phat-meta page-goal-card">
          <span className="meta-label">Goal</span>
          <span>Score a Polymarket snapshot with the latest calibration to produce pHAT & edge signals.</span>
          <div className="meta-pill">Outputs stored under src/data/analysis/phat-edge</div>
        </div>
      </header>

      <div className="phat-edge-grid">
        <section className="panel form-panel">
          <div className="panel-header">
            <div>
              <h2>Run configuration</h2>
              <span className="panel-hint">
                Inputs are validated before the inference script runs.
              </span>
            </div>
          </div>
          <form className="panel-body phat-form" onSubmit={handleSubmit}>
            <div className="field">
              <label htmlFor="modelSelect">Trained model</label>
              <select
                id="modelSelect"
                className="input"
                value={formState.modelId}
                onChange={(event) =>
                  setFormState((prev) => ({
                    ...prev,
                    modelId: event.target.value,
                  }))
                }
              >
                {models.length === 0 ? (
                  <option value="">No models found</option>
                ) : null}
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.id}
                  </option>
                ))}
              </select>
              {modelError ? <div className="error">{modelError}</div> : null}
              <span className="field-hint">
                Resolved from <code>src/data/models/&lt;model-id&gt;</code>.
              </span>
            </div>

            <div className="field">
              <label htmlFor="snapshotType">Snapshot contract type</label>
              <select
                id="snapshotType"
                className="input"
                value={formState.snapshotContractType}
                onChange={(event) =>
                  setFormState((prev) => ({
                    ...prev,
                    snapshotContractType: event.target.value as "weekly" | "1dte",
                  }))
                }
              >
                <option value="weekly">weekly</option>
                <option value="1dte">1dte</option>
              </select>
            </div>

            <div className="field">
              <label htmlFor="snapshotSelect">Snapshot CSV</label>
              <select
                id="snapshotSelect"
                className="input"
                value={formState.snapshotCsv}
                onChange={(event) =>
                  setFormState((prev) => ({
                    ...prev,
                    snapshotCsv: event.target.value,
                  }))
                }
              >
                {filteredSnapshots.length === 0 ? (
                  <option value="">No snapshot runs available</option>
                ) : null}
                {filteredSnapshots.map((snapshot) => (
                  <option key={snapshot.value} value={snapshot.value}>
                    {snapshot.label}
                  </option>
                ))}
              </select>
              {snapshotError ? <div className="error">{snapshotError}</div> : null}
              <span className="field-hint">
                Runs stored under <code>src/data/raw/polymarket/runs/weekly</code> or{" "}
                <code>src/data/raw/polymarket/runs/1dte</code>.
                {selectedSnapshot?.timestamp ? (
                  <>
                    <br />
                    Selected run: {formatTimestamp(selectedSnapshot.timestamp)}
                  </>
                ) : null}
              </span>
            </div>

            <div className="field">
              <label htmlFor="outCsv">Output CSV (optional)</label>
              <input
                id="outCsv"
                className="input"
                placeholder="src/data/analysis/phat-edge/my-edge.csv"
                value={formState.outCsv}
                onChange={(event) =>
                  setFormState((prev) => ({
                    ...prev,
                    outCsv: event.target.value,
                  }))
                }
              />
              <span className="field-hint">
                Leave empty to auto-generate a timestamped file under <code>src/data/analysis/phat-edge</code>.
              </span>
            </div>

            <div className="toggle-grid">
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={formState.requireColumnsStrict}
                  onChange={(event) =>
                    setFormState((prev) => ({
                      ...prev,
                      requireColumnsStrict: event.target.checked,
                    }))
                  }
                />
                Require all manifest columns
              </label>
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={formState.computeEdge}
                  onChange={(event) =>
                    setFormState((prev) => ({
                      ...prev,
                      computeEdge: event.target.checked,
                    }))
                  }
                />
                Compute edge vs buy price
              </label>
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={formState.skipEdgeOutsidePrnRange}
                  onChange={(event) =>
                    setFormState((prev) => ({
                      ...prev,
                      skipEdgeOutsidePrnRange: event.target.checked,
                    }))
                  }
                />
                Skip edges outside model pRN range
              </label>
            </div>
            <span className="field-hint">
              Uses the training pRN band stored in the model artifact (if available). When missing, all rows are scored.
            </span>

            {runError ? <div className="error">{runError}</div> : null}

            <div className="actions">
              <button
                className="button primary"
                type="submit"
                disabled={isRunning || anyJobRunning}
              >
                {isRunning ? "Computing pHAT..." : "Run inference"}
              </button>
            </div>
          </form>
        </section>

        <section className="panel summary-panel">
          <div className="panel-header">
            <div>
              <h2>Latest run</h2>
              <span className="panel-hint">
                Captures logs and metadata returned by the backend run.
              </span>
            </div>
          </div>
          <div className="panel-body panel-summary">
            {!runResult ? (
              <div className="empty">Run the script to view a summary.</div>
            ) : (
              <>
                <div className="run-summary-card">
                  <div className="run-summary-header">
                    <div>
                      <span className="meta-label">Status</span>
                      <div className="run-status">
                        <span
                          className={`status-pill ${
                            runResult.ok ? "success" : "failed"
                          }`}
                        >
                          {runResult.ok ? "Success" : "Failed"}
                        </span>
                      </div>
                    </div>
                    <span className="meta-label">
                      Duration: {runResult.run_summary.duration_s.toFixed(2)}s
                    </span>
                  </div>
                  <div className="run-summary-details">
                    <div>
                      <span className="meta-label">Model artifact</span>
                      <span>{runResult.run_summary.model_path}</span>
                    </div>
                    <div>
                      <span className="meta-label">Snapshot</span>
                      <span>{runResult.run_summary.snapshot_csv}</span>
                    </div>
                    <div>
                      <span className="meta-label">Output CSV</span>
                      <span>{runResult.run_summary.output_csv}</span>
                    </div>
                  </div>
                </div>

                <div className="edge-preview">
                  <div className="edge-preview-header">
                    <div>
                      <span className="meta-label">Edge output preview</span>
                      <span className="edge-preview-file">
                        {edgePreview?.file.name ?? outputCsvPath ?? "Output CSV"}
                      </span>
                    </div>
                    {edgePreviewLoading ? (
                      <span className="meta-pill">Refreshing</span>
                    ) : null}
                  </div>
                  {edgePreview ? (
                    <div className="table-container">
                      <table className="preview-table">
                        <thead>
                          <tr>
                            {edgePreview.headers.map((header) => (
                              <th key={header}>{header}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {edgePreview.rows.map((row, idx) => (
                            <tr key={`${edgePreview.file.path}-${idx}`}>
                              {edgePreview.headers.map((header) => (
                                <td key={`${header}-${idx}`}>
                                  {row[header] ?? ""}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : edgePreviewLoading ? (
                    <div className="preview-placeholder">Loading preview…</div>
                  ) : edgePreviewError ? (
                    <div className="error">{edgePreviewError}</div>
                  ) : (
                    <div className="empty">No output preview available yet.</div>
                  )}
                </div>

                <div className="log-tabs">
                  <button
                    type="button"
                    className={`log-tab ${activeLog === "stdout" ? "active" : ""}`}
                    onClick={() => setActiveLog("stdout")}
                  >
                    stdout
                  </button>
                  <button
                    type="button"
                    className={`log-tab ${activeLog === "stderr" ? "active" : ""}`}
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
              </>
            )}
          </div>
        </section>
      </div>

      <section className="panel stats-panel">
        <div className="panel-header">
          <div>
            <h2>Distribution insights</h2>
            <span className="panel-hint">
              Summary of pHAT and filtered edges.
            </span>
          </div>
          {edgeSummaryLoading ? (
            <span className="meta-pill">Refreshing</span>
          ) : null}
        </div>
        <div className="panel-body">
          {edgeSummaryError ? <div className="error">{edgeSummaryError}</div> : null}
          <div className="stat-grid">
            <div className="stat-card">
              <div className="stat-label">pHAT count</div>
              <div className="stat-value">
                {pHatStats ? pHatStats.count : "--"}
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Mean pHAT</div>
              <div className="stat-value">
                {pHatStats ? pHatStats.mean.toFixed(4) : "--"}
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Min pHAT</div>
              <div className="stat-value">
                {pHatStats ? pHatStats.min.toFixed(4) : "--"}
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Max pHAT</div>
              <div className="stat-value">
                {pHatStats ? pHatStats.max.toFixed(4) : "--"}
              </div>
            </div>
          </div>
          <div className="stat-grid">
            <div className="stat-card">
              <div className="stat-label">Edge count</div>
              <div className="stat-value">
                {edgeStats ? edgeStats.count : "--"}
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Mean edge</div>
              <div className="stat-value">
                {edgeStats ? edgeStats.mean.toFixed(4) : "--"}
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Min edge</div>
              <div className="stat-value">
                {edgeStats ? edgeStats.min.toFixed(4) : "--"}
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Max edge</div>
              <div className="stat-value">
                {edgeStats ? edgeStats.max.toFixed(4) : "--"}
              </div>
            </div>
          </div>
          <div className="table-wrapper">
            <div className="table-heading">
              <div>
                <span className="meta-label">Edge section</span>
                <p className="table-sub">
                  Full capture of edge rows (sorted by edge, descending).
                  {edgeRowCount > 0
                    ? ` Showing ${displayedEdgeRows.length} of ${edgeRowCount} rows.`
                    : ""}
                </p>
              </div>
              <div className="edge-filter-group">
                {edgeRowsLoading ? (
                  <span className="meta-pill">Refreshing</span>
                ) : null}
                <button
                  type="button"
                  className="button light edge-filter-button"
                  onClick={() =>
                    setEdgesPerTickerLimit((prev) =>
                      prev === "all" ? 1 : prev === 1 ? 3 : "all",
                    )
                  }
                >
                  {edgesPerTickerLabel}
                </button>
                <button
                  type="button"
                  className="button light edge-filter-button"
                  onClick={() =>
                    setExcludePrnOutOfRange((prev) => !prev)
                  }
                >
                  {excludePrnOutOfRange
                    ? "Include pRN out-of-range"
                    : "Exclude pRN out-of-range"}
                </button>
              </div>
            </div>
            {edgeRowsError ? <div className="error">{edgeRowsError}</div> : null}
            {displayedEdgeRows.length > 0 ? (
              <div className="table-container">
                <table className="edge-table">
                  <thead>
                    <tr>
                      <th>Ticker</th>
                      <th>K</th>
                      <th>Spot</th>
                      <th>pHAT</th>
                      <th>pRN</th>
                      <th>buy yes price</th>
                      <th>qHAT</th>
                      <th>qRN</th>
                      <th>buy no price</th>
                      <th>edge type</th>
                      <th>edge</th>
                    </tr>
                  </thead>
                  <tbody>
                    {displayedEdgeRows.map((row, idx) => (
                      <tr key={`${row.ticker}-${row.K ?? "na"}-${idx}`}>
                        <td>{row.ticker}</td>
                        <td>{row.K?.toFixed(4) ?? "–"}</td>
                        <td>{row.spot?.toFixed(4) ?? "–"}</td>
                        <td>{row.pHAT?.toFixed(4) ?? "–"}</td>
                        <td>{row.pRN?.toFixed(4) ?? "–"}</td>
                        <td>{row.pPM_buy?.toFixed(4) ?? "–"}</td>
                        <td>
                          {(row.qHAT ?? (row.pHAT !== null && row.pHAT !== undefined ? 1 - row.pHAT : null))?.toFixed(4) ?? "–"}
                        </td>
                        <td>{row.qRN?.toFixed(4) ?? "–"}</td>
                        <td>{row.qPM_buy?.toFixed(4) ?? "–"}</td>
                        <td>{row.edge_source ?? "–"}</td>
                        <td>{row.edge?.toFixed(4) ?? "–"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty">
                {edgeRowsLoading
                  ? "Loading edge rows..."
                  : runResult
                    ? "No edge rows available."
                    : "Edge rows appear after a successful run."}
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="panel kelly-panel">
        <div className="panel-header">
          <div>
            <h2>Kelly fraction calculator</h2>
            <span className="panel-hint">
              Size a YES/NO contract using your probability estimate and market price.
            </span>
          </div>
        </div>
        <div className="panel-body kelly-grid">
          <div className="kelly-inputs">
            <div className="field">
              <label>Contract side</label>
              <div className="kelly-toggle" role="group" aria-label="Contract side">
                <button
                  type="button"
                  className={`kelly-toggle-button ${kellySide === "yes" ? "active" : ""}`}
                  onClick={() => setKellySide("yes")}
                >
                  YES
                </button>
                <button
                  type="button"
                  className={`kelly-toggle-button ${kellySide === "no" ? "active" : ""}`}
                  onClick={() => setKellySide("no")}
                >
                  NO
                </button>
              </div>
              <span className="field-hint">
                Enter inputs for the selected side only.
              </span>
            </div>

            <div className="field">
              <label htmlFor="kelly-price">
                Market price ({kellySide.toUpperCase()})
              </label>
              <input
                id="kelly-price"
                className="input"
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={kellyPriceInput}
                onChange={(event) => setKellyPriceInput(event.target.value)}
              />
              <span className="field-hint">Price between 0 and 1.</span>
            </div>

            <div className="field">
              <label htmlFor="kelly-prob">
                Estimated probability ({kellySide.toUpperCase()})
              </label>
              <input
                id="kelly-prob"
                className="input"
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={kellyProbInput}
                onChange={(event) => setKellyProbInput(event.target.value)}
              />
              <span className="field-hint">Your forecast for this side.</span>
            </div>

            <div className="field">
              <label htmlFor="kelly-bankroll">Bankroll (optional)</label>
              <input
                id="kelly-bankroll"
                className="input"
                type="number"
                step="1"
                min="0"
                placeholder="1000"
                value={kellyBankrollInput}
                onChange={(event) => setKellyBankrollInput(event.target.value)}
              />
              <span className="field-hint">Used to compute a stake size.</span>
            </div>
          </div>

          <div className="kelly-output">
            <div className="kelly-metrics">
              <div className="stat-card">
                <div className="stat-label">Kelly fraction</div>
                <div className="stat-value">
                  {formatPercent(kellyMetrics.kellyFraction)}
                </div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Suggested allocation</div>
                <div className="stat-value">
                  {formatPercent(kellyMetrics.suggestedFraction)}
                </div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Edge vs price</div>
                <div className="stat-value">
                  {formatPercent(kellyMetrics.edge)}
                </div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Decimal odds</div>
                <div className="stat-value">
                  {formatNumber(kellyMetrics.decimalOdds, 2)}
                </div>
              </div>
              <div className="stat-card">
                <div className="stat-label">Stake size</div>
                <div className="stat-value">
                  {kellyMetrics.stake !== null
                    ? formatNumber(kellyMetrics.stake, 2)
                    : "--"}
                </div>
              </div>
            </div>
            <div className="kelly-note">
              Suggested allocation is the Kelly fraction capped between 0% and 100%.
            </div>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <h2>Recent edge runs</h2>
            <span className="panel-hint">
              Outputs stored under <code>src/data/analysis/phat-edge</code>. Click a run to preview its CSV.
            </span>
          </div>
        </div>
        <div className="panel-body">
          {edgeRunsError ? <div className="error">{edgeRunsError}</div> : null}
          {runDeleteError ? <div className="error">{runDeleteError}</div> : null}
          {edgeRuns.length === 0 ? (
            <div className="empty">No edge runs found.</div>
          ) : (
            <div className="runs-list">
              {edgeRuns.map((run, index) => {
                const selected = selectedEdgeRunPath === run.path;
                return (
                  <div
                    key={run.path}
                    role="button"
                    tabIndex={0}
                    className={`run-card ${selected ? "selected" : ""} ${index === 0 ? "latest" : ""}`}
                    onClick={() => setSelectedEdgeRunPath(run.path)}
                    onKeyDown={(event: KeyboardEvent<HTMLDivElement>) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        setSelectedEdgeRunPath(run.path);
                      }
                    }}
                  >
                    <div>
                      <div className="run-title">{run.name}</div>
                      <div className="run-subtitle">
                        {formatTimestamp(run.last_modified)}
                      </div>
                    </div>
                    <div className="run-stats">
                      <span>{formatBytes(run.size_bytes)}</span>
                    </div>
                    <div className="run-card-actions">
                      <button
                        type="button"
                        className="button ghost danger run-card-delete"
                        disabled={deletingRunId === run.path}
                        onClick={(event) => {
                          event.stopPropagation();
                          handleDeleteEdgeRun(run.path);
                        }}
                      >
                        {deletingRunId === run.path ? "Deleting…" : "Delete"}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </section>
    </section>
  );
}
