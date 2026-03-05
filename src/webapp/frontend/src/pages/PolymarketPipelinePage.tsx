import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";

import {
  startPolymarketHistoryJob,
  cancelPolymarketHistoryJob,
  listPipelineRuns,
  renamePipelineRun,
  setActiveRun,
  deletePipelineRun,
  getPipelineRunFileUrl,
  previewPipelineRunCsv,
  buildDecisionFeaturesForRun,
  type CsvPreview,
  type PipelineProgress,
  type PipelineRunSummary,
  type StorageSummary,
} from "../api/polymarketHistory";
import { backfillOptionChainDataset, listDatasetRuns, type DatasetRunSummary } from "../api/datasets";
import {
  startMarketMapJob,
} from "../api/marketMap";
import PipelineStatusCard from "../components/PipelineStatusCard";
import PipelineProgressBar from "../components/PipelineProgressBar";
import { usePolymarketHistoryJob } from "../contexts/polymarketHistoryJob";
import { useMarketMapJob } from "../contexts/marketMapJob";
import { useAnyJobRunning } from "../contexts/jobGuard";
import "./PolymarketPipelinePage.css";

const FORM_STORAGE_KEY = "polyedgetool.polymarket.pipeline.form";

const TRADING_UNIVERSE_TICKERS = [
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

const defaultForm = {
  overrides: "config/polymarket_market_overrides.csv",
  tickers: TRADING_UNIVERSE_TICKERS.join(", "),
  useWeeklyHistory: true,
  historyRunDirName: "",
  historyStartDate: "",
  historyEndDate: "",
  historyFidelityMin: "60",
  historyBarsFreqs: "1h,1d",
  historyIncludeSubgraph: true,
  historyBuildFeatures: false,
  historyPrnDataset: "",
};

type FormState = typeof defaultForm;

const loadStoredForm = (): FormState | null => {
  try {
    const raw = localStorage.getItem(FORM_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    const merged = { ...defaultForm, ...parsed } as FormState;
    return {
      ...merged,
      useWeeklyHistory: true,
      tickers: sanitizeTickers(merged.tickers),
    };
  } catch {
    return null;
  }
};

const formatDateTime = (value?: string | null) => {
  if (!value) return "--";
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

const formatCount = (value?: number | null) => {
  if (value === null || value === undefined) return "--";
  return value.toLocaleString();
};

const formatByteCount = (bytes?: number | null) => {
  if (bytes === null || bytes === undefined) return "--";
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(1)} MB`;
};

const toKebabCase = (value: string) =>
  value
    .trim()
    .toLowerCase()
    .replace(/[\s_]+/g, "-")
    .replace(/[^a-z0-9-]/g, "")
    .replace(/-{2,}/g, "-")
    .replace(/^-|-$/g, "");

const MONTH_NAMES = [
  "january",
  "february",
  "march",
  "april",
  "may",
  "june",
  "july",
  "august",
  "september",
  "october",
  "november",
  "december",
];

const parseTickers = (value: string) =>
  value
    .split(",")
    .map((item) => item.trim().toUpperCase())
    .filter(Boolean);

const normalizeTickers = (values: string[]) => {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values) {
    const normalized = value.trim().toUpperCase();
    if (!normalized || seen.has(normalized)) continue;
    if (!TRADING_UNIVERSE_TICKERS.includes(normalized)) continue;
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
};

const orderTickers = (values: string[]) =>
  TRADING_UNIVERSE_TICKERS.filter((ticker) => values.includes(ticker));

const formatTickerList = (values: string[]) => values.join(", ");

const sanitizeTickers = (value: string) => {
  const parsed = normalizeTickers(parseTickers(value));
  const ordered = orderTickers(parsed);
  const safeList = ordered.length > 0 ? ordered : TRADING_UNIVERSE_TICKERS;
  return formatTickerList(safeList);
};

const parseIsoDateUtc = (value: string) => {
  const parts = value.split("-").map((item) => Number(item));
  if (parts.length !== 3) return null;
  const [year, month, day] = parts;
  if (!year || !month || !day) return null;
  return new Date(Date.UTC(year, month - 1, day));
};

const getFridayDates = (start: Date, end: Date) => {
  const fridayIndex = 5; // Sunday=0 ... Friday=5
  const startDow = start.getUTCDay();
  const delta = (fridayIndex - startDow + 7) % 7;
  const current = new Date(start.getTime());
  current.setUTCDate(current.getUTCDate() + delta);
  const dates: Date[] = [];
  while (current <= end) {
    dates.push(new Date(current.getTime()));
    current.setUTCDate(current.getUTCDate() + 7);
  }
  return dates;
};

const buildAutoEventUrls = (
  tickers: string[],
  startDate: string,
  endDate: string,
) => {
  if (!startDate || !endDate) {
    return { urls: [], fridays: [], error: "Start date and end date are required." };
  }
  const start = parseIsoDateUtc(startDate);
  const end = parseIsoDateUtc(endDate);
  if (!start || !end) {
    return { urls: [], fridays: [], error: "Invalid start or end date." };
  }
  if (start > end) {
    return { urls: [], fridays: [], error: "Start date must be before end date." };
  }

  const fridays = getFridayDates(start, end);
  if (!fridays.length) {
    return { urls: [], fridays, error: "No Fridays found in the selected range." };
  }

  const urlSet = new Set<string>();
  tickers.forEach((ticker) => {
    const slugPrefix = ticker.toLowerCase();
    fridays.forEach((friday) => {
      const month = MONTH_NAMES[friday.getUTCMonth()];
      const day = friday.getUTCDate();
      const year = friday.getUTCFullYear();
      const slug = `${slugPrefix}-above-on-${month}-${day}-${year}`;
      urlSet.add(`https://polymarket.com/event/${slug}`);
    });
  });

  return { urls: Array.from(urlSet), fridays, error: null };
};

const mergeProgress = (
  prev: PipelineProgress | null,
  next: PipelineProgress | null,
): PipelineProgress | null => {
  if (!next) return prev;
  if (!prev) return next;
  const total = Math.max(prev.total, next.total);
  const completed = Math.min(total, Math.max(prev.completed, next.completed));
  const failed = Math.min(completed, Math.max(prev.failed, next.failed));
  const status: PipelineProgress["status"] =
    completed >= total ? (failed > 0 ? "failed" : "completed") : "running";
  if (
    total === prev.total &&
    completed === prev.completed &&
    failed === prev.failed &&
    status === prev.status
  ) {
    return prev;
  }
  return { total, completed, failed, status };
};

type PreviewMode = "head" | "tail";

type RunCsvPreviewTarget = {
  runId: string;
  filename: string;
  label: string;
};

type RunCsvFileSummary = NonNullable<PipelineRunSummary["csv_files"]>[number];

const PREVIEW_LIMIT_DEFAULT = 20;
const PREVIEW_LIMIT_OPTIONS = [20, 50, 100] as const;
const PREVIEW_MODE_OPTIONS: { value: PreviewMode; label: string }[] = [
  { value: "head", label: "First" },
  { value: "tail", label: "Last" },
];

const sortRunCsvFiles = (files: RunCsvFileSummary[]): RunCsvFileSummary[] =>
  [...files].sort((a, b) => a.name.localeCompare(b.name));

const isDecisionFeaturesCsv = (name: string): boolean => {
  const lower = name.toLowerCase();
  return lower === "decision_features.csv" || lower.endsWith("decision-features.csv");
};

export default function PolymarketPipelinePage() {
  const [form, setForm] = useState<FormState>(() => loadStoredForm() ?? defaultForm);
  const [formError, setFormError] = useState<string | null>(null);
  const [stopLoading, setStopLoading] = useState(false);
  const [datasetRuns, setDatasetRuns] = useState<DatasetRunSummary[]>([]);
  const [datasetRunsError, setDatasetRunsError] = useState<string | null>(null);
  const [isDatasetRunsLoading, setIsDatasetRunsLoading] = useState(false);
  const [workspaceTab, setWorkspaceTab] = useState<
    "run_job" | "history" | "documentation"
  >("run_job");
  const [runJobPanel, setRunJobPanel] = useState<"configuration" | "active_run">(
    "configuration",
  );
  const [activeLogView, setActiveLogView] = useState<"stdout" | "stderr" | null>(null);

  // --- Runs browser state ---
  const [pipelineRuns, setPipelineRuns] = useState<PipelineRunSummary[]>([]);
  const [runsStorage, setRunsStorage] = useState<StorageSummary | null>(null);
  const [runsLoading, setRunsLoading] = useState(false);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [renamingRunId, setRenamingRunId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [renameError, setRenameError] = useState<string | null>(null);
  const [renameLoading, setRenameLoading] = useState(false);
  const [openRunId, setOpenRunId] = useState<string | null>(null);
  const [runCsvPreviewTarget, setRunCsvPreviewTarget] =
    useState<RunCsvPreviewTarget | null>(null);
  const [runCsvPreviewResponse, setRunCsvPreviewResponse] =
    useState<CsvPreview | null>(null);
  const [runCsvPreviewError, setRunCsvPreviewError] = useState<string | null>(null);
  const [runCsvPreviewLoading, setRunCsvPreviewLoading] = useState(false);
  const [runCsvPreviewMode, setRunCsvPreviewMode] = useState<PreviewMode>("head");
  const [runCsvPreviewLimit, setRunCsvPreviewLimit] = useState<number>(
    PREVIEW_LIMIT_DEFAULT,
  );
  const [featuresBuildRunId, setFeaturesBuildRunId] = useState<string | null>(null);
  const [featuresBuildError, setFeaturesBuildError] = useState<{
    runId: string;
    message: string;
  } | null>(null);
  const [featuresModalRunId, setFeaturesModalRunId] = useState<string | null>(null);
  const [featuresModalDatasetId, setFeaturesModalDatasetId] = useState<string | null>(
    null,
  );
  const [featuresModalError, setFeaturesModalError] = useState<string | null>(null);
  const [backfillLoading, setBackfillLoading] = useState(false);
  const [backfillMessage, setBackfillMessage] = useState<string | null>(null);
  const [backfillError, setBackfillError] = useState<string | null>(null);
  const [backfillAllowDefaults, setBackfillAllowDefaults] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [deleteConfirmText, setDeleteConfirmText] = useState("");
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [historyProgressState, setHistoryProgressState] = useState<PipelineProgress | null>(null);
  const [featuresProgressState, setFeaturesProgressState] = useState<PipelineProgress | null>(null);
  const lastHistoryJobId = useRef<string | null>(null);

  const { jobStatus: mapJobStatus, setJobId: setMapJobId } = useMarketMapJob();
  const {
    jobStatus: historyJobStatus,
    setJobId: setHistoryJobId,
    refreshJob: refreshHistoryJob,
  } = usePolymarketHistoryJob();
  const { anyJobRunning, activeJobs } = useAnyJobRunning();

  const isRunning =
    mapJobStatus?.status === "queued" ||
    mapJobStatus?.status === "running" ||
    historyJobStatus?.status === "queued" ||
    historyJobStatus?.status === "running";
  const isFeaturesBuildRunning = featuresBuildRunId !== null;

  const historyJobId = historyJobStatus?.job_id ?? null;

  useEffect(() => {
    if (!historyJobId) {
      lastHistoryJobId.current = null;
      setHistoryProgressState(null);
      setFeaturesProgressState(null);
      return;
    }

    if (historyJobId !== lastHistoryJobId.current) {
      lastHistoryJobId.current = historyJobId;
      setHistoryProgressState(historyJobStatus?.progress ?? null);
      setFeaturesProgressState(historyJobStatus?.features_progress ?? null);
      return;
    }

    setHistoryProgressState((prev) =>
      mergeProgress(prev, historyJobStatus?.progress ?? null),
    );
    setFeaturesProgressState((prev) =>
      mergeProgress(prev, historyJobStatus?.features_progress ?? null),
    );
  }, [
    historyJobId,
    historyJobStatus?.progress,
    historyJobStatus?.features_progress,
  ]);

  useEffect(() => {
    localStorage.setItem(FORM_STORAGE_KEY, JSON.stringify(form));
  }, [form]);

  const optionChainRuns = useMemo(
    () =>
      datasetRuns.filter(
        (run) =>
          run.run_dir.startsWith("src/data/raw/option-chain/") ||
          run.run_dir.startsWith("data/raw/option-chain/"),
      ),
    [datasetRuns],
  );
  const optionChainTrainingRuns = useMemo(() => {
    const filtered = optionChainRuns.filter((run) => run.training_file?.path);
    return [...filtered].sort((a, b) => {
      const aStamp = a.last_modified ?? "";
      const bStamp = b.last_modified ?? "";
      return bStamp.localeCompare(aStamp);
    });
  }, [optionChainRuns]);
  const selectedDatasetRun = useMemo(
    () => optionChainRuns.find((run) => run.run_dir === form.historyPrnDataset) ?? null,
    [optionChainRuns, form.historyPrnDataset],
  );
  const resolvedHistoryPrnDataset = useMemo(() => {
    if (!selectedDatasetRun) return null;
    return selectedDatasetRun.training_file?.path ?? null;
  }, [selectedDatasetRun]);

  const loadDatasetRuns = useCallback(async () => {
    setIsDatasetRunsLoading(true);
    setDatasetRunsError(null);
    try {
      const payload = await listDatasetRuns();
      setDatasetRuns(payload.runs ?? []);
    } catch (err) {
      setDatasetRuns([]);
      setDatasetRunsError(
        err instanceof Error ? err.message : "Failed to load dataset directories",
      );
    } finally {
      setIsDatasetRunsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDatasetRuns();
  }, [loadDatasetRuns]);

  // --- Runs browser: load on mount + after pipeline finishes ---
  const loadRuns = useCallback(() => {
    setRunsLoading(true);
    setRunsError(null);
    listPipelineRuns()
      .then((data) => {
        setPipelineRuns(data.runs);
        setRunsStorage(data.storage);
      })
      .catch((err) => {
        setRunsError(err instanceof Error ? err.message : "Failed to load runs");
      })
      .finally(() => setRunsLoading(false));
  }, []);

  useEffect(() => { loadRuns(); }, [loadRuns]);

  // Refresh runs list when a job finishes
  const prevHistoryStatus = useMemo(() => historyJobStatus?.status, [historyJobStatus?.status]);
  useEffect(() => {
    if (prevHistoryStatus === "finished" || prevHistoryStatus === "failed" || prevHistoryStatus === "cancelled") {
      loadRuns();
    }
  }, [prevHistoryStatus, loadRuns]);

  useEffect(() => {
    if (!runCsvPreviewTarget) {
      setRunCsvPreviewResponse(null);
      setRunCsvPreviewError(null);
      setRunCsvPreviewLoading(false);
      return;
    }

    let cancelled = false;
    setRunCsvPreviewLoading(true);
    setRunCsvPreviewError(null);
    previewPipelineRunCsv(
      runCsvPreviewTarget.runId,
      runCsvPreviewTarget.filename,
      runCsvPreviewMode,
      runCsvPreviewLimit,
    )
      .then((preview) => {
        if (cancelled) return;
        setRunCsvPreviewResponse(preview);
      })
      .catch((err) => {
        if (cancelled) return;
        setRunCsvPreviewResponse(null);
        setRunCsvPreviewError(err instanceof Error ? err.message : "Unknown error");
      })
      .finally(() => {
        if (!cancelled) setRunCsvPreviewLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [runCsvPreviewTarget, runCsvPreviewMode, runCsvPreviewLimit]);

  useEffect(() => {
    if (!runCsvPreviewTarget) return;
    const previewRunStillExists = pipelineRuns.some(
      (run) => run.run_id === runCsvPreviewTarget.runId,
    );
    if (!previewRunStillExists) {
      setRunCsvPreviewTarget(null);
    }
  }, [pipelineRuns, runCsvPreviewTarget]);

  useEffect(() => {
    if (!openRunId) return;
    const stillExists = pipelineRuns.some((run) => run.run_id === openRunId);
    if (!stillExists) {
      setOpenRunId(null);
    }
  }, [openRunId, pipelineRuns]);

  const handleSetActive = useCallback(async (runId: string) => {
    try {
      await setActiveRun(runId);
      loadRuns();
    } catch (err) {
      console.error("Set active failed:", err);
    }
  }, [loadRuns]);

  const handleStartRename = useCallback((runId: string, currentLabel: string) => {
    setRenamingRunId(runId);
    setRenameValue(currentLabel);
    setRenameError(null);
  }, []);

  const handleCancelRename = useCallback(() => {
    setRenamingRunId(null);
    setRenameValue("");
    setRenameError(null);
  }, []);

  const handleConfirmRename = useCallback(async (runId: string) => {
    setRenameLoading(true);
    setRenameError(null);
    try {
      const cleaned = renameValue.trim();
      await renamePipelineRun(runId, cleaned);
      await loadRuns();
      handleCancelRename();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setRenameError(message);
    } finally {
      setRenameLoading(false);
    }
  }, [handleCancelRename, loadRuns, renameValue]);

  const handleOpenFeaturesModal = useCallback((runId: string) => {
    setFeaturesModalRunId(runId);
    setFeaturesModalError(null);
    setBackfillMessage(null);
    setBackfillError(null);
    setBackfillAllowDefaults(false);
    const preferred = optionChainTrainingRuns.find(
      (run) => run.run_dir === form.historyPrnDataset,
    );
    const fallback = optionChainTrainingRuns[0] ?? null;
    setFeaturesModalDatasetId(preferred?.id ?? fallback?.id ?? null);
  }, [form.historyPrnDataset, optionChainTrainingRuns]);

  const handleCloseFeaturesModal = useCallback(() => {
    if (featuresBuildRunId || backfillLoading) return;
    setFeaturesModalRunId(null);
    setFeaturesModalDatasetId(null);
    setFeaturesModalError(null);
    setBackfillMessage(null);
    setBackfillError(null);
    setBackfillAllowDefaults(false);
  }, [backfillLoading, featuresBuildRunId]);

  const handleConfirmBuildDecisionFeatures = useCallback(async () => {
    if (!featuresModalRunId || featuresBuildRunId || backfillLoading) return;
    const selected = optionChainTrainingRuns.find(
      (run) => run.id === featuresModalDatasetId,
    );
    if (!selected || !selected.training_file?.path) {
      setFeaturesModalError("Select a training dataset with a training CSV.");
      return;
    }

    setFeaturesBuildRunId(featuresModalRunId);
    setFeaturesBuildError(null);
    setFeaturesModalError(null);
    try {
      const response = await buildDecisionFeaturesForRun(featuresModalRunId, {
        prnDataset: selected.training_file.path,
      });
      if (!response.ok) {
        throw new Error(response.stderr || response.stdout || "Decision features build failed.");
      }
      await loadRuns();
      setFeaturesModalRunId(null);
      setFeaturesModalDatasetId(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Decision features build failed.";
      setFeaturesBuildError({ runId: featuresModalRunId, message });
      setFeaturesModalError(message);
    } finally {
      setFeaturesBuildRunId(null);
    }
  }, [
    backfillLoading,
    featuresBuildRunId,
    featuresModalDatasetId,
    featuresModalRunId,
    loadRuns,
    optionChainTrainingRuns,
  ]);

  const handleBackfillDataset = useCallback(async () => {
    if (!featuresModalRunId || backfillLoading) return;
    const selected = optionChainTrainingRuns.find(
      (run) => run.id === featuresModalDatasetId,
    );
    if (!selected) {
      setBackfillError("Select a training dataset to backfill.");
      return;
    }

    setBackfillLoading(true);
    setBackfillError(null);
    setBackfillMessage(null);
    try {
      const response = await backfillOptionChainDataset({
        runDir: selected.run_dir,
        polymarketRunId: featuresModalRunId,
        allowDefaults: backfillAllowDefaults,
      });
      if (!response.ok) {
        throw new Error(response.message || "Dataset backfill failed.");
      }
      setBackfillMessage(response.message);
      await loadDatasetRuns();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Dataset backfill failed.";
      setBackfillError(message);
    } finally {
      setBackfillLoading(false);
    }
  }, [
    backfillAllowDefaults,
    backfillLoading,
    featuresModalDatasetId,
    featuresModalRunId,
    loadDatasetRuns,
    optionChainTrainingRuns,
  ]);

  useEffect(() => {
    if (!renamingRunId) return;
    const stillExists = pipelineRuns.some((run) => run.run_id === renamingRunId);
    if (!stillExists) {
      handleCancelRename();
    }
  }, [handleCancelRename, pipelineRuns, renamingRunId]);

  useEffect(() => {
    if (!featuresBuildError) return;
    const stillExists = pipelineRuns.some((run) => run.run_id === featuresBuildError.runId);
    if (!stillExists) {
      setFeaturesBuildError(null);
    }
  }, [featuresBuildError, pipelineRuns]);

  useEffect(() => {
    if (!featuresModalRunId) return;
    const stillExists = pipelineRuns.some((run) => run.run_id === featuresModalRunId);
    if (!stillExists) {
      handleCloseFeaturesModal();
    }
  }, [featuresModalRunId, handleCloseFeaturesModal, pipelineRuns]);

  useEffect(() => {
    if (!featuresModalRunId) return;
    const hasSelection = featuresModalDatasetId
      ? optionChainTrainingRuns.some((run) => run.id === featuresModalDatasetId)
      : false;
    if (hasSelection) return;
    const preferred = optionChainTrainingRuns.find(
      (run) => run.run_dir === form.historyPrnDataset,
    );
    const fallback = optionChainTrainingRuns[0] ?? null;
    setFeaturesModalDatasetId(preferred?.id ?? fallback?.id ?? null);
  }, [
    featuresModalDatasetId,
    featuresModalRunId,
    form.historyPrnDataset,
    optionChainTrainingRuns,
  ]);

  const handleToggleRunCsvPreview = useCallback((target: RunCsvPreviewTarget) => {
    setRunCsvPreviewTarget((prev) => {
      if (
        prev &&
        prev.runId === target.runId &&
        prev.filename === target.filename
      ) {
        return null;
      }
      return target;
    });
  }, []);

  const handleDeleteConfirm = useCallback(async () => {
    if (!deleteTarget || deleteConfirmText !== "DELETE") return;
    setDeleteLoading(true);
    try {
      await deletePipelineRun(deleteTarget);
      if (runCsvPreviewTarget?.runId === deleteTarget) {
        setRunCsvPreviewTarget(null);
      }
      setDeleteTarget(null);
      setDeleteConfirmText("");
      loadRuns();
    } catch (err) {
      console.error("Delete failed:", err);
    } finally {
      setDeleteLoading(false);
    }
  }, [deleteTarget, deleteConfirmText, loadRuns, runCsvPreviewTarget]);

  const formatSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const usingWeeklyHistory = form.useWeeklyHistory;
  const historyRunDirNameKebab = toKebabCase(form.historyRunDirName);
  const pipelineStatus = usingWeeklyHistory
    ? historyJobStatus?.status || "idle"
    : mapJobStatus?.status || "idle";
  const pipelineStatusLabel =
    pipelineStatus === "queued" || pipelineStatus === "running"
      ? "Running"
      : pipelineStatus === "finished"
        ? "Success"
        : pipelineStatus === "cancelled"
          ? "Cancelled"
        : pipelineStatus === "failed"
          ? "Failed"
          : "Ready";

  const pipelineStatusClass =
    pipelineStatus === "queued" || pipelineStatus === "running"
      ? "running"
      : pipelineStatus === "finished"
        ? "success"
        : pipelineStatus === "cancelled"
          ? "failed"
        : pipelineStatus === "failed"
        ? "failed"
        : "idle";

  const historyPhase = historyJobStatus?.phase ?? null;
  const historyStageLabel = useMemo(() => {
    if (!historyJobStatus) return "--";
    if (historyJobStatus.status === "finished") {
      return historyJobStatus.result?.features_built
        ? "Decision features built"
        : "History complete";
    }
    if (historyJobStatus.status === "failed") return "Failed";
    if (historyJobStatus.status === "cancelled") return "Cancelled";
    if (historyJobStatus.status === "queued") return "Queued";
    if (historyPhase === "features") return "Building decision features";
    if (historyPhase === "finalizing") return "Finalizing outputs";
    return "Fetching weekly history";
  }, [historyJobStatus, historyPhase]);

  const handleRunPipeline = async (event: FormEvent) => {
    event.preventDefault();
    if (anyJobRunning) {
      return;
    }

    try {
      if (usingWeeklyHistory) {
        setFormError(null);

        if (form.historyRunDirName.trim() && !historyRunDirNameKebab) {
          setFormError("Run directory name must contain at least one alphanumeric character.");
          return;
        }

        const effectiveTickers =
          selectedTickers.length > 0 ? selectedTickers : TRADING_UNIVERSE_TICKERS;
        const autoUrls = buildAutoEventUrls(
          effectiveTickers,
          form.historyStartDate,
          form.historyEndDate,
        );
        if (autoUrls.error) {
          setFormError(autoUrls.error);
          return;
        }
        if (!autoUrls.urls.length) {
          setFormError("Auto-generation produced zero event URLs.");
          return;
        }

        if (form.historyBuildFeatures && !resolvedHistoryPrnDataset) {
          setFormError("Select a pRN dataset directory with a training CSV.");
          return;
        }

        const payload = {
          tickers: effectiveTickers,
          eventUrls: autoUrls.urls,
          startDate: form.historyStartDate || undefined,
          endDate: form.historyEndDate || undefined,
          fidelityMin: Number(form.historyFidelityMin) || undefined,
          barsFreqs: form.historyBarsFreqs || undefined,
          runDirName: form.historyRunDirName.trim() || undefined,
          includeSubgraph: form.historyIncludeSubgraph,
          buildFeatures: form.historyBuildFeatures,
          prnDataset: resolvedHistoryPrnDataset || undefined,
          skipSubgraphLabels: false,
        };
        const status = await startPolymarketHistoryJob(payload);
        setHistoryJobId(status.job_id);
        setRunJobPanel("active_run");
        setWorkspaceTab("run_job");
        setActiveLogView(null);
      } else {
        const status = await startMarketMapJob({
          runDir: undefined,
          overrides: form.overrides || undefined,
          tickers: formatTickerList(selectedTickers),
          prnDataset: undefined,
          out: undefined,
          strict: false,
        });
        setMapJobId(status.job_id);
        setRunJobPanel("active_run");
        setWorkspaceTab("run_job");
        setActiveLogView(null);
      }
    } catch (err) {
      console.error("Pipeline failed:", err);
    }
  };

  const handleStopRun = async () => {
    if (!historyJobStatus?.job_id) return;
    setStopLoading(true);
    try {
      await cancelPolymarketHistoryJob(historyJobStatus.job_id);
      await refreshHistoryJob();
    } catch (err) {
      console.error("Stop run failed:", err);
    } finally {
      setStopLoading(false);
    }
  };

  const handleToggleLog = useCallback((target: "stdout" | "stderr") => {
    setActiveLogView((prev) => (prev === target ? null : target));
  }, []);

  const handleNewJob = useCallback(() => {
    setRunJobPanel("configuration");
    setWorkspaceTab("run_job");
    setActiveLogView(null);
  }, []);

  const mapStdout = mapJobStatus?.result?.stdout ?? "";
  const mapStderr = mapJobStatus?.result?.stderr ?? "";
  const mapError = mapJobStatus?.error ?? "";
  const historyStdout = historyJobStatus?.result?.stdout ?? "";
  const historyStderr = historyJobStatus?.result?.stderr ?? "";
  const historyError = historyJobStatus?.error ?? "";

  const selectedTickers = useMemo(() => {
    const parsed = normalizeTickers(parseTickers(form.tickers));
    return orderTickers(parsed);
  }, [form.tickers]);

  const selectedTickerSet = useMemo(
    () => new Set(selectedTickers),
    [selectedTickers],
  );

  const updateTickers = useCallback((next: string[]) => {
    const normalized = normalizeTickers(next);
    const ordered = orderTickers(normalized);
    const safeList = ordered.length > 0 ? ordered : TRADING_UNIVERSE_TICKERS;
    setForm((prev) => ({ ...prev, tickers: formatTickerList(safeList) }));
  }, []);

  useEffect(() => {
    if (selectedTickers.length === 0) {
      updateTickers(TRADING_UNIVERSE_TICKERS);
    }
  }, [selectedTickers, updateTickers]);

  const toggleUniverseTicker = useCallback(
    (ticker: string) => {
      const next = selectedTickerSet.has(ticker)
        ? selectedTickers.filter((value) => value !== ticker)
        : [...selectedTickers, ticker];
      updateTickers(next);
    },
    [selectedTickers, selectedTickerSet, updateTickers],
  );

  const autoEventState = useMemo(() => {
    const tickers =
      selectedTickers.length > 0 ? selectedTickers : TRADING_UNIVERSE_TICKERS;
    return buildAutoEventUrls(tickers, form.historyStartDate, form.historyEndDate);
  }, [selectedTickers, form.historyStartDate, form.historyEndDate]);

  const hasEventSources = (autoEventState?.urls.length ?? 0) > 0;
  const canStopHistory =
    historyJobStatus?.status === "queued" || historyJobStatus?.status === "running";

  const defaultActiveLog = usingWeeklyHistory
    ? historyStderr || historyError
      ? "stderr"
      : "stdout"
    : mapStderr || mapError
      ? "stderr"
      : "stdout";
  const activeLog = activeLogView;
  const hasAnyLogOutput = usingWeeklyHistory
    ? Boolean(historyStdout || historyStderr || historyError)
    : Boolean(mapStdout || mapStderr || mapError);
  const terminalStatus =
    pipelineStatus === "finished" ||
    pipelineStatus === "failed" ||
    pipelineStatus === "cancelled";
  const hasRunRecord = usingWeeklyHistory ? Boolean(historyJobStatus) : Boolean(mapJobStatus);
  const showNewJobButton = terminalStatus && hasRunRecord;

  const historyRunning =
    historyJobStatus?.status === "queued" || historyJobStatus?.status === "running";
  const historyProgress = historyProgressState ?? historyJobStatus?.progress ?? null;
  const featuresProgress =
    featuresProgressState ?? historyJobStatus?.features_progress ?? null;
  const historyProgressPercent =
    historyProgress && historyProgress.total > 0
      ? Math.round((historyProgress.completed / historyProgress.total) * 100)
      : 0;
  const showFeatureProgress = Boolean(
    featuresProgress ||
      historyPhase === "features" ||
      historyJobStatus?.result?.features_built ||
      (historyRunning && form.historyBuildFeatures),
  );
  const featuresRunning = historyRunning && historyPhase === "features";

  const historyRunId =
    historyJobStatus?.result?.run_id ?? historyJobStatus?.job_id ?? "--";
  const historyRunDir = historyJobStatus?.result?.run_dir ?? "--";
  const historyDurationLabel = historyJobStatus?.result?.duration_s
    ? `${historyJobStatus.result.duration_s}s`
    : isRunning
      ? "Running..."
      : "--";
  const historyFilesLabel = formatCount(historyJobStatus?.result?.files?.length);
  const historyLastUpdatedLabel = formatDateTime(
    historyJobStatus?.finished_at ?? historyJobStatus?.started_at,
  );
  const historyDateRangeLabel =
    form.historyStartDate && form.historyEndDate
      ? `${form.historyStartDate} to ${form.historyEndDate}`
      : "--";
  const historyEventCountLabel = formatCount(autoEventState?.urls.length);
  const historyProgressLabel = historyProgress
    ? `${historyProgress.completed} / ${historyProgress.total} jobs completed (${historyProgressPercent}%)`
    : historyRunning
      ? "Running pipeline..."
      : pipelineStatusLabel;
  const runJobDisabled =
    isRunning || anyJobRunning || isFeaturesBuildRunning || backfillLoading;
  const runJobLabel = isRunning
    ? "Running job..."
    : isFeaturesBuildRunning
      ? "Building decision features..."
      : backfillLoading
        ? "Backfilling dataset..."
        : "Run Job";
  const featuresModalRun = featuresModalRunId
    ? pipelineRuns.find((run) => run.run_id === featuresModalRunId) ?? null
    : null;
  const featuresModalRunLabel =
    (featuresModalRun?.label ?? "").trim() ||
    featuresModalRun?.run_id ||
    featuresModalRunId ||
    "";
  const historyMonitorItems = [
    { label: "Stage", value: historyStageLabel },
    { label: "Tickers", value: `${selectedTickers.length} selected` },
    { label: "Date range", value: historyDateRangeLabel },
    { label: "Event URLs", value: historyEventCountLabel },
    { label: "Run ID", value: historyRunId },
    { label: "Files", value: historyFilesLabel },
    { label: "Duration", value: historyDurationLabel },
    { label: "Run dir", value: historyRunDir },
    { label: "Last update", value: historyLastUpdatedLabel },
  ];

  const mapRunId = mapJobStatus?.job_id ?? "--";
  const mapOutputLabel = mapJobStatus?.result?.output_path ?? "--";
  const mapRowsLabel = formatCount(mapJobStatus?.result?.row_count);
  const mapDurationLabel = mapJobStatus?.result?.duration_s
    ? `${mapJobStatus.result.duration_s}s`
    : isRunning
      ? "Running..."
      : "--";
  const mapLastUpdatedLabel = formatDateTime(
    mapJobStatus?.finished_at ?? mapJobStatus?.started_at,
  );
  const mapRunning =
    mapJobStatus?.status === "queued" || mapJobStatus?.status === "running";
  const mapProgressLabel = mapRunning ? "Running pipeline..." : pipelineStatusLabel;
  const mapMonitorItems = [
    { label: "Tickers", value: `${selectedTickers.length} selected` },
    { label: "Overrides", value: form.overrides || "--" },
    { label: "Run ID", value: mapRunId },
    { label: "Output file", value: mapOutputLabel },
    { label: "Rows", value: mapRowsLabel },
    { label: "Duration", value: mapDurationLabel },
    { label: "Last update", value: mapLastUpdatedLabel },
  ];

  return (
    <section className="page polymarket-pipeline-page">
      <PipelineStatusCard
        className="page-sticky-meta polymarket-meta"
        activeJobsCount={activeJobs.length}
      />
      <header className="page-header polymarket-page-header">
        <div className="polymarket-title-row">
          <h1 className="page-title polymarket-page-title">
            Polymarket History Buider
          </h1>
        </div>
      </header>

      <div className="polymarket-workspace">
        <div
          className="polymarket-workspace-tabs"
          role="tablist"
          aria-label="Polymarket history builder workspace"
        >
          <button
            id="polymarket-tab-run-job"
            type="button"
            role="tab"
            aria-selected={workspaceTab === "run_job"}
            aria-controls="polymarket-panel-run-job"
            className={`polymarket-workspace-tab ${
              workspaceTab === "run_job" ? "active" : ""
            }`}
            onClick={() => setWorkspaceTab("run_job")}
          >
            Run job
          </button>
          <button
            id="polymarket-tab-history"
            type="button"
            role="tab"
            aria-selected={workspaceTab === "history"}
            aria-controls="polymarket-panel-history"
            className={`polymarket-workspace-tab ${
              workspaceTab === "history" ? "active" : ""
            }`}
            onClick={() => setWorkspaceTab("history")}
          >
            Run directory
          </button>
          <button
            id="polymarket-tab-documentation"
            type="button"
            role="tab"
            aria-selected={workspaceTab === "documentation"}
            aria-controls="polymarket-panel-documentation"
            className={`polymarket-workspace-tab ${
              workspaceTab === "documentation" ? "active" : ""
            }`}
            onClick={() => setWorkspaceTab("documentation")}
          >
            Documentation
          </button>
        </div>

        {workspaceTab === "run_job" ? (
          <div
            id="polymarket-panel-run-job"
            role="tabpanel"
            aria-labelledby="polymarket-tab-run-job"
            className="polymarket-tab-panel"
          >
            {runJobPanel === "configuration" ? (
              <section className="panel polymarket-config-panel">
                <div className="panel-header polymarket-panel-header polymarket-job-config-header">
                  <div>
                    <h2 className="polymarket-job-config-title">Job Configuration</h2>
                    <span className="panel-hint">
                      Configure weekly history backfill or a market-map refresh.
                    </span>
                  </div>
                  {hasRunRecord ? (
                    <button
                      className="button light polymarket-fixed-action-button"
                      type="button"
                      onClick={() => {
                        setRunJobPanel("active_run");
                        setActiveLogView(null);
                      }}
                    >
                      View latest run
                    </button>
                  ) : null}
                </div>
                <form className="panel-body polymarket-config-form" onSubmit={handleRunPipeline}>
                  <div className="section-card polymarket-config-card">
                    <div className="fields-grid">
                      <div className="field full">
                        <label>Trading universe</label>
                        <div className="ticker-grid">
                          {TRADING_UNIVERSE_TICKERS.map((ticker) => (
                            <button
                              key={ticker}
                              type="button"
                              className={`ticker-chip ${
                                selectedTickerSet.has(ticker) ? "selected" : ""
                              }`}
                              aria-pressed={selectedTickerSet.has(ticker)}
                              onClick={() => toggleUniverseTicker(ticker)}
                            >
                              {ticker}
                            </button>
                          ))}
                        </div>
                        <span className="field-hint">
                          Pick from the core trading universe. Only these tickers
                          are allowed.
                        </span>
                        <span className="field-hint polymarket-selected-tickers">
                          Selected: {selectedTickers.join(", ")}
                        </span>
                      </div>
                    </div>
                  </div>

                  {usingWeeklyHistory ? (
                    <div className="section-card polymarket-config-card">
                      <div className="fields-grid">
                        <div className="field">
                          <label>History start date (UTC)</label>
                          <input
                            className="input"
                            type="date"
                            value={form.historyStartDate}
                            onChange={(event) =>
                              setForm((prev) => ({
                                ...prev,
                                historyStartDate: event.target.value,
                              }))
                            }
                          />
                          <span className="field-hint">Required for auto-generation.</span>
                        </div>
                        <div className="field">
                          <label>History end date (UTC)</label>
                          <input
                            className="input"
                            type="date"
                            value={form.historyEndDate}
                            onChange={(event) =>
                              setForm((prev) => ({
                                ...prev,
                                historyEndDate: event.target.value,
                              }))
                            }
                          />
                          <span className="field-hint">Required for auto-generation.</span>
                        </div>
                        <div className="field">
                          <label>CLOB fidelity (minutes)</label>
                          <input
                            className="input"
                            type="number"
                            min="1"
                            value={form.historyFidelityMin}
                            onChange={(event) =>
                              setForm((prev) => ({
                                ...prev,
                                historyFidelityMin: event.target.value,
                              }))
                            }
                          />
                        </div>
                        <div className="field">
                          <label>Bar frequencies</label>
                          <input
                            className="input"
                            value={form.historyBarsFreqs}
                            onChange={(event) =>
                              setForm((prev) => ({
                                ...prev,
                                historyBarsFreqs: event.target.value,
                              }))
                            }
                          />
                          <span className="field-hint">
                            Comma-separated (e.g. 1h,1d).
                          </span>
                        </div>
                        <div className="field">
                          <label>Run directory name</label>
                          <input
                            className="input"
                            value={form.historyRunDirName}
                            placeholder="Optional (e.g. fed-weeklies-jan-2026)"
                            onChange={(event) =>
                              setForm((prev) => ({
                                ...prev,
                                historyRunDirName: event.target.value,
                              }))
                            }
                          />
                          <span className="field-hint">
                            Optional. Saved as kebab-case on disk. CSVs will use{" "}
                            <code>
                              {(historyRunDirNameKebab || "run-directory")}
                              {"-{csv-type}.csv"}
                            </code>
                            .
                          </span>
                        </div>
                        <div className="field full">
                          <label>History options</label>
                          <div className="polymarket-toggle-card-grid" role="group" aria-label="History options">
                            <button
                              type="button"
                              className={`polymarket-toggle-card ${
                                form.historyIncludeSubgraph ? "selected" : ""
                              }`}
                              aria-pressed={form.historyIncludeSubgraph}
                              onClick={() =>
                                setForm((prev) => ({
                                  ...prev,
                                  historyIncludeSubgraph: !prev.historyIncludeSubgraph,
                                }))
                              }
                            >
                              <span className="polymarket-toggle-card-title">
                                Attempt subgraph trade ingest
                              </span>
                              <span className="polymarket-toggle-card-copy">
                                Use configured subgraph ingest alongside history fetches.
                              </span>
                            </button>
                            <button
                              type="button"
                              className={`polymarket-toggle-card ${
                                form.historyBuildFeatures ? "selected" : ""
                              }`}
                              aria-pressed={form.historyBuildFeatures}
                              onClick={() =>
                                setForm((prev) => ({
                                  ...prev,
                                  historyBuildFeatures: !prev.historyBuildFeatures,
                                }))
                              }
                            >
                              <span className="polymarket-toggle-card-title">
                                Build decision features
                              </span>
                              <span className="polymarket-toggle-card-copy">
                                Generate feature outputs for model calibration.
                              </span>
                            </button>
                          </div>
                        </div>
                        {form.historyBuildFeatures ? (
                          <div className="field full">
                            <label>Option Chain History Directory</label>
                            {isDatasetRunsLoading ? (
                              <div className="polymarket-dataset-selector-state">
                                Loading dataset directories...
                              </div>
                            ) : optionChainRuns.length ? (
                              <div
                                className="polymarket-dataset-grid"
                                role="group"
                                aria-label="Option Chain history dataset selection"
                              >
                                {optionChainRuns.map((run) => {
                                  const isSelected = form.historyPrnDataset === run.run_dir;
                                  const runName = run.run_dir.split("/").pop() ?? run.run_dir;
                                  const hasTrainingCsv = Boolean(run.training_file?.path);

                                  return (
                                    <button
                                      key={run.run_dir}
                                      type="button"
                                      aria-pressed={isSelected}
                                      className={`polymarket-dataset-card${
                                        isSelected ? " selected" : ""
                                      }${hasTrainingCsv ? "" : " is-missing-training"}`}
                                      onClick={() =>
                                        setForm((prev) => ({
                                          ...prev,
                                          historyPrnDataset:
                                            prev.historyPrnDataset === run.run_dir
                                              ? ""
                                              : run.run_dir,
                                        }))
                                      }
                                      title={run.run_dir}
                                    >
                                      <span className="polymarket-dataset-card-name">
                                        {runName}
                                      </span>
                                      <span className="polymarket-dataset-card-path">
                                        {run.run_dir}
                                      </span>
                                      <span className="polymarket-dataset-card-meta">
                                        {hasTrainingCsv
                                          ? "Has training CSV"
                                          : "Training CSV missing"}
                                      </span>
                                    </button>
                                  );
                                })}
                              </div>
                            ) : null}
                            {datasetRunsError ? (
                              <span className="field-hint">{datasetRunsError}</span>
                            ) : !optionChainRuns.length ? (
                              <span className="field-hint">
                                No option-chain dataset directories found.
                              </span>
                            ) : null}
                          </div>
                        ) : null}
                      </div>
                    </div>
                  ) : (
                    <div className="section-card polymarket-config-card polymarket-map-mode-note">
                      <div className="polymarket-doc-list">
                        <p>
                          Market-map mode refreshes live market mappings and writes
                          <code>dim_market</code> outputs for the selected trading universe.
                        </p>
                        <p>
                          It uses <code>{form.overrides}</code> and does not require a
                          date range.
                        </p>
                      </div>
                    </div>
                  )}

                  {usingWeeklyHistory && !hasEventSources ? (
                    <div className="help-note">
                      <strong>Heads up:</strong> weekly finance markets are not
                      discoverable by pagination. Provide a date range to
                      auto-generate event URLs.
                    </div>
                  ) : null}

                  {formError ? (
                    <div className="error-banner">
                      <strong>Input Error:</strong>
                      <pre>{formError}</pre>
                    </div>
                  ) : null}

                  <div className="panel-actions polymarket-config-actions">
                    <button
                      className="button primary large polymarket-fixed-action-button polymarket-run-job-button"
                      type="submit"
                      disabled={runJobDisabled}
                    >
                      {runJobLabel}
                    </button>
                  </div>
                </form>
              </section>
            ) : (
              <section className="panel polymarket-active-run-panel">
                <div className="panel-header polymarket-panel-header">
                  <div>
                    <h2 className="polymarket-job-config-title">Active Run</h2>
                    <span className="panel-hint">
                      Monitor progress and inspect stdout/stderr for the current job.
                    </span>
                  </div>
                  <div className="polymarket-active-run-header-actions">
                    {canStopHistory ? (
                      <button
                        className="button ghost danger"
                        type="button"
                        onClick={handleStopRun}
                        disabled={stopLoading}
                      >
                        {stopLoading ? "Stopping…" : "Stop run"}
                      </button>
                    ) : null}
                    {showNewJobButton ? (
                      <button
                        className="button light"
                        type="button"
                        onClick={handleNewJob}
                      >
                        New job
                      </button>
                    ) : null}
                  </div>
                </div>
                <div className="panel-body">
                  {usingWeeklyHistory ? (
                    historyJobStatus ? (
                      <div className="polymarket-active-run-shell">
                        <aside className="polymarket-active-run-sidebar">
                          <div className="pipeline-run-monitor">
                            <div className="pipeline-run-monitor-header">
                              <div>
                                <span className="meta-label">Run monitor</span>
                                <div className="pipeline-run-monitor-title">
                                  Weekly history run
                                </div>
                              </div>
                              <span className={`status-pill ${pipelineStatusClass}`}>
                                {pipelineStatusLabel}
                              </span>
                            </div>
                            <div className="pipeline-run-monitor-grid">
                              {historyMonitorItems.map((item) => (
                                <div key={item.label}>
                                  <span className="meta-label">{item.label}</span>
                                  <span>{item.value}</span>
                                </div>
                              ))}
                            </div>
                            <div className="pipeline-run-monitor-progress">
                              <div className="pipeline-run-monitor-progress-header">
                                <span>Progress</span>
                                <span>{historyProgressLabel}</span>
                              </div>
                              <div className="pipeline-progress-stack">
                                <PipelineProgressBar
                                  title="Stage 1: Fetching / processing markets"
                                  progress={historyProgress}
                                  running={historyRunning}
                                  runningLabel="Running pipeline..."
                                  idleLabel={pipelineStatusLabel}
                                  unitLabel="jobs"
                                  forceError={historyJobStatus?.status === "failed"}
                                />
                                {showFeatureProgress ? (
                                  <PipelineProgressBar
                                    title="Stage 2: Creating features"
                                    progress={featuresProgress}
                                    running={featuresRunning}
                                    runningLabel="Creating features..."
                                    idleLabel={
                                      historyJobStatus?.result?.features_built
                                        ? "Complete"
                                        : historyRunning
                                          ? "Queued"
                                          : "Not started"
                                    }
                                    unitLabel="steps"
                                    forceError={
                                      historyJobStatus?.status === "failed" &&
                                      historyPhase === "features"
                                    }
                                  />
                                ) : null}
                              </div>
                              <p className="pipeline-run-monitor-note">
                                Progress updates only when each market finishes, so
                                counts never move backward.
                              </p>
                            </div>
                          </div>
                        </aside>

                        <div className="polymarket-active-run-main">
                          <div className="run-output pipeline-run-output">
                            <div className="polymarket-log-tabs">
                              <button
                                className={`log-tab ${activeLog === "stdout" ? "active" : ""}`}
                                type="button"
                                aria-pressed={activeLog === "stdout"}
                                onClick={() => handleToggleLog("stdout")}
                              >
                                stdout
                              </button>
                              <button
                                className={`log-tab ${activeLog === "stderr" ? "active" : ""}`}
                                type="button"
                                aria-pressed={activeLog === "stderr"}
                                onClick={() => handleToggleLog("stderr")}
                              >
                                stderr
                              </button>
                            </div>
                            <div className="log-block">
                              {!hasAnyLogOutput ? (
                                <div className="log-empty-state">
                                  No stdout or stderr captured yet.
                                </div>
                              ) : activeLog ? (
                                <>
                                  <span className="meta-label">{activeLog}</span>
                                  <pre className="log-content">
                                    {activeLog === "stdout"
                                      ? historyStdout || "No output captured."
                                      : historyStderr || historyError || "No errors."}
                                  </pre>
                                </>
                              ) : (
                                <div className="log-empty-state">
                                  Select <strong>stdout</strong> or <strong>stderr</strong>{" "}
                                  to view logs. Default stream is{" "}
                                  <strong>{defaultActiveLog}</strong>.
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="empty">
                        Start a weekly history run to see progress and logs.
                      </div>
                    )
                  ) : mapJobStatus ? (
                    <div className="polymarket-active-run-shell">
                      <aside className="polymarket-active-run-sidebar">
                        <div className="pipeline-run-monitor">
                          <div className="pipeline-run-monitor-header">
                            <div>
                              <span className="meta-label">Run monitor</span>
                              <div className="pipeline-run-monitor-title">
                                Market map run
                              </div>
                            </div>
                            <span className={`status-pill ${pipelineStatusClass}`}>
                              {pipelineStatusLabel}
                            </span>
                          </div>
                          <div className="pipeline-run-monitor-grid">
                            {mapMonitorItems.map((item) => (
                              <div key={item.label}>
                                <span className="meta-label">{item.label}</span>
                                <span>{item.value}</span>
                              </div>
                            ))}
                          </div>
                          <div className="pipeline-run-monitor-progress">
                            <div className="pipeline-run-monitor-progress-header">
                              <span>Progress</span>
                              <span>{mapProgressLabel}</span>
                            </div>
                            <div className="pipeline-progress-stack">
                              <PipelineProgressBar
                                title="Pipeline run"
                                progress={null}
                                running={mapRunning}
                                runningLabel="Running pipeline..."
                                idleLabel={pipelineStatusLabel}
                                unitLabel="jobs"
                                forceError={mapJobStatus?.status === "failed"}
                              />
                            </div>
                            <p className="pipeline-run-monitor-note">
                              Live progress telemetry is not available for market
                              map runs yet.
                            </p>
                          </div>
                        </div>
                      </aside>

                      <div className="polymarket-active-run-main">
                        <div className="run-output pipeline-run-output">
                          <div className="polymarket-log-tabs">
                            <button
                              className={`log-tab ${activeLog === "stdout" ? "active" : ""}`}
                              type="button"
                              aria-pressed={activeLog === "stdout"}
                              onClick={() => handleToggleLog("stdout")}
                            >
                              stdout
                            </button>
                            <button
                              className={`log-tab ${activeLog === "stderr" ? "active" : ""}`}
                              type="button"
                              aria-pressed={activeLog === "stderr"}
                              onClick={() => handleToggleLog("stderr")}
                            >
                              stderr
                            </button>
                          </div>
                          <div className="log-block">
                            {!hasAnyLogOutput ? (
                              <div className="log-empty-state">
                                No stdout or stderr captured yet.
                              </div>
                            ) : activeLog ? (
                              <>
                                <span className="meta-label">{activeLog}</span>
                                <pre className="log-content">
                                  {activeLog === "stdout"
                                    ? mapStdout || "No output captured."
                                    : mapStderr || mapError || "No errors."}
                                </pre>
                              </>
                            ) : (
                              <div className="log-empty-state">
                                Select <strong>stdout</strong> or <strong>stderr</strong>{" "}
                                to view logs. Default stream is{" "}
                                <strong>{defaultActiveLog}</strong>.
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="empty">Run the pipeline to see output.</div>
                  )}
                </div>
              </section>
            )}
          </div>
        ) : workspaceTab === "history" ? (
          <div
            id="polymarket-panel-history"
            role="tabpanel"
            aria-labelledby="polymarket-tab-history"
            className="polymarket-tab-panel"
          >
            <section className="panel polymarket-history-panel">
              <div className="panel-header polymarket-panel-header polymarket-job-config-header">
                <div>
                  <h2 className="polymarket-job-config-title">Run directory</h2>
                  <span className="panel-hint">
                    Browse, preview, rename, activate, and delete prior weekly backfill
                    run directories and CSV outputs.
                  </span>
                </div>
                {runsStorage ? (
                  <div className="runs-storage-badge" aria-label="Run storage summary">
                    <span>{runsStorage.total_runs} runs</span>
                    <span>{runsStorage.total_size_mb.toFixed(1)} MB</span>
                  </div>
                ) : null}
              </div>

              <div className="panel-body polymarket-history-body">
                {runsLoading ? (
                  <div className="empty">Loading build history…</div>
                ) : runsError ? (
                  <div className="error-banner">
                    <strong>Failed to load runs:</strong>
                    <pre>{runsError}</pre>
                  </div>
                ) : pipelineRuns.length === 0 ? (
                  <div className="runs-empty">
                    No pipeline runs found. Run the weekly backfill to create your
                    first run.
                  </div>
                ) : (
                  <div className="polymarket-runs-list">
                    {pipelineRuns.map((run) => {
                      const statusClass =
                        run.status === "success"
                          ? "success"
                          : run.status === "failed"
                            ? "failed"
                            : run.status === "cancelled"
                              ? "cancelled"
                              : "unknown";
                      const statusLabel =
                        run.status.charAt(0).toUpperCase() + run.status.slice(1);
                      const dateRangeLabel =
                        run.start_date && run.end_date
                          ? `${run.start_date} to ${run.end_date}`
                          : "--";
                      const displayLabel = (run.label ?? "").trim();
                      const csvFiles = sortRunCsvFiles(run.csv_files ?? []);
                      const hasDecisionFeaturesCsv = csvFiles.some((file) =>
                        isDecisionFeaturesCsv(file.name),
                      );
                      const canBuildDecisionFeatures = !hasDecisionFeaturesCsv;
                      const fileCount = csvFiles.length;
                      const csvFilesLabel = fileCount
                        ? `${fileCount} CSV${fileCount === 1 ? "" : "s"}`
                        : "No CSV files";
                      const isRenaming = renamingRunId === run.run_id;
                      const isOpen = openRunId === run.run_id;
                      const isPreviewingRun =
                        isOpen && runCsvPreviewTarget?.runId === run.run_id;
                      const isBuildingFeatures = featuresBuildRunId === run.run_id;
                      const featuresBuildErrorMessage =
                        featuresBuildError?.runId === run.run_id
                          ? featuresBuildError.message
                          : null;
                      const runMainContent = (
                        <>
                          <div className="polymarket-run-title-row">
                            <span
                              className={`run-status-dot ${statusClass}`}
                              title={run.status}
                            />
                            <div className="polymarket-run-title-group">
                              {displayLabel ? (
                                <div className="polymarket-run-title">{displayLabel}</div>
                              ) : null}
                              <div className="polymarket-run-badges">
                                {run.status !== "success" ? (
                                  <span className={`status-pill ${statusClass}`}>
                                    {statusLabel}
                                  </span>
                                ) : null}
                                {run.is_active ? (
                                  <span className="run-active-badge">Active</span>
                                ) : null}
                                {run.features_built ? (
                                  <span className="polymarket-run-tag">Features</span>
                                ) : null}
                              </div>
                              <div className="run-id-mono">{run.run_id}</div>
                            </div>
                          </div>
                          <div className="polymarket-run-metrics">
                            <div>
                              <span className="meta-label">Date range</span>
                              <span>{dateRangeLabel}</span>
                            </div>
                            <div>
                              <span className="meta-label">Markets</span>
                              <span>{formatCount(run.markets)}</span>
                            </div>
                            <div>
                              <span className="meta-label">CSVs</span>
                              <span>{csvFilesLabel}</span>
                            </div>
                            <div>
                              <span className="meta-label">Size</span>
                              <span>{formatSize(run.size_bytes)}</span>
                            </div>
                            <div>
                              <span className="meta-label">Created</span>
                              <span>{formatDateTime(run.created_at_utc)}</span>
                            </div>
                            <div>
                              <span className="meta-label">Duration</span>
                              <span>
                                {run.duration_s != null ? `${run.duration_s}s` : "--"}
                              </span>
                            </div>
                          </div>
                          <div className="polymarket-run-path">{run.run_dir}</div>
                        </>
                      );

                      return (
                        <article
                          key={run.run_id}
                          className={`polymarket-run-item ${
                            run.is_active ? "is-active" : ""
                          }${isOpen ? " is-open" : ""}`}
                        >
                          <div className="polymarket-run-top">
                            {isRenaming ? (
                              <div className="polymarket-run-main polymarket-run-main--renaming">
                                <div className="polymarket-rename-input-wrapper">
                                  <input
                                    className="input polymarket-rename-input"
                                    type="text"
                                    placeholder={run.run_id}
                                    value={renameValue}
                                    onChange={(event) =>
                                      setRenameValue(event.target.value)
                                    }
                                    onKeyDown={(event) => {
                                      if (event.key === "Enter") {
                                        handleConfirmRename(run.run_id);
                                      } else if (event.key === "Escape") {
                                        handleCancelRename();
                                      }
                                    }}
                                    autoFocus
                                  />
                                  {renameError ? (
                                    <div className="error">{renameError}</div>
                                  ) : null}
                                  <div className="polymarket-rename-actions">
                                    <button
                                      className="button ghost small"
                                      type="button"
                                      onClick={() => handleConfirmRename(run.run_id)}
                                      disabled={renameLoading}
                                    >
                                      {renameLoading ? "Saving…" : "Save"}
                                    </button>
                                    <button
                                      className="button ghost small"
                                      type="button"
                                      onClick={handleCancelRename}
                                      disabled={renameLoading}
                                    >
                                      Cancel
                                    </button>
                                  </div>
                                </div>
                                {runMainContent}
                              </div>
                            ) : (
                              <button
                                type="button"
                                className="polymarket-run-toggle"
                                aria-expanded={fileCount > 0 ? isOpen : undefined}
                                aria-controls={
                                  fileCount > 0 ? `polymarket-run-files-${run.run_id}` : undefined
                                }
                                onClick={() => {
                                  if (fileCount <= 0) return;
                                  setOpenRunId((prev) =>
                                    prev === run.run_id ? null : run.run_id,
                                  );
                                }}
                              >
                                <div className="polymarket-run-main">
                                  {runMainContent}
                                </div>
                              </button>
                            )}

                            <div className="polymarket-run-actions">
                              {canBuildDecisionFeatures ? (
                                <button
                                  className="button light small"
                                  type="button"
                                  onClick={() => handleOpenFeaturesModal(run.run_id)}
                                  disabled={
                                    isBuildingFeatures ||
                                    isRunning ||
                                    anyJobRunning ||
                                    isFeaturesBuildRunning ||
                                    backfillLoading
                                  }
                                  title={
                                    "Select an option-chain training dataset and build decision features."
                                  }
                                >
                                  {isBuildingFeatures
                                    ? "Building features..."
                                    : "Build decision features"}
                                </button>
                              ) : null}
                              {!run.is_active ? (
                                <button
                                  className="button light small"
                                  type="button"
                                  onClick={() => handleSetActive(run.run_id)}
                                  title="Set as active run"
                                >
                                  Activate
                                </button>
                              ) : null}
                              <button
                                className="button light small"
                                type="button"
                                onClick={() => handleStartRename(run.run_id, displayLabel)}
                                disabled={isRenaming || renameLoading}
                                title="Rename run"
                              >
                                Rename run
                              </button>
                              {!run.is_active ? (
                                <button
                                  className="button ghost danger small"
                                  type="button"
                                  onClick={() => {
                                    setDeleteTarget(run.run_id);
                                    setDeleteConfirmText("");
                                  }}
                                  title="Delete run"
                                >
                                  Delete
                                </button>
                              ) : null}
                            </div>
                          </div>

                          {run.error_summary || featuresBuildErrorMessage ? (
                            <div className="polymarket-run-footer">
                              {run.error_summary ? (
                                <div
                                  className="polymarket-run-error-summary"
                                  title={run.error_summary}
                                >
                                  {run.error_summary}
                                </div>
                              ) : null}
                              {featuresBuildErrorMessage ? (
                                <div
                                  className="polymarket-run-error-summary"
                                  title={featuresBuildErrorMessage}
                                >
                                  Decision features build failed: {featuresBuildErrorMessage}
                                </div>
                              ) : null}
                            </div>
                          ) : null}
                          {fileCount > 0 ? (
                            <div
                              id={`polymarket-run-files-${run.run_id}`}
                              className={`polymarket-run-files-drawer${isOpen ? " is-open" : ""}`}
                              hidden={!isOpen}
                            >
                              <div className="polymarket-run-files">
                                {csvFiles.map((file) => {
                                  const isPreviewingFile =
                                    runCsvPreviewTarget?.runId === run.run_id &&
                                    runCsvPreviewTarget.filename === file.name;
                                  return (
                                    <div key={`${run.run_id}-${file.name}`} className="polymarket-run-file">
                                      <div className="polymarket-run-file-info">
                                        <div className="polymarket-run-file-name">
                                          {file.name}
                                        </div>
                                        <div className="polymarket-run-file-meta">
                                          <span>{formatByteCount(file.size_bytes)}</span>
                                          <span>
                                            {file.row_count != null
                                              ? `${file.row_count.toLocaleString()} rows`
                                              : "Row count unknown"}
                                          </span>
                                        </div>
                                      </div>
                                      <div className="polymarket-run-file-actions">
                                        <button
                                          className="button light small"
                                          type="button"
                                          onClick={() =>
                                            handleToggleRunCsvPreview({
                                              runId: run.run_id,
                                              filename: file.name,
                                              label: file.name,
                                            })
                                          }
                                          title={isPreviewingFile ? "Hide preview" : `Preview ${file.name}`}
                                        >
                                          {isPreviewingFile ? "Hide preview" : "Preview"}
                                        </button>
                                        <a
                                          className="button light small"
                                          href={getPipelineRunFileUrl(run.run_id, file.name)}
                                          target="_blank"
                                          rel="noopener noreferrer"
                                          title={`Open ${file.name}`}
                                        >
                                          Open
                                        </a>
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                              {isPreviewingRun ? (
                                <div className="polymarket-csv-preview-panel">
                                  <div className="polymarket-csv-preview-header">
                                    <div>
                                      <span className="meta-label">CSV preview</span>
                                      <p className="polymarket-csv-preview-title">
                                        {runCsvPreviewTarget?.label ?? "Select a CSV"}
                                      </p>
                                    </div>
                                    <div className="polymarket-csv-preview-controls">
                                      <label className="polymarket-csv-preview-control">
                                        <span className="meta-label">Range</span>
                                        <select
                                          className="input"
                                          value={runCsvPreviewMode}
                                          onChange={(event) =>
                                            setRunCsvPreviewMode(
                                              event.target.value as PreviewMode,
                                            )
                                          }
                                        >
                                          {PREVIEW_MODE_OPTIONS.map((option) => (
                                            <option key={option.value} value={option.value}>
                                              {option.label}
                                            </option>
                                          ))}
                                        </select>
                                      </label>
                                      <label className="polymarket-csv-preview-control">
                                        <span className="meta-label">Rows</span>
                                        <select
                                          className="input"
                                          value={runCsvPreviewLimit}
                                          onChange={(event) => {
                                            const next = Number.parseInt(
                                              event.target.value,
                                              10,
                                            );
                                            setRunCsvPreviewLimit(
                                              Number.isFinite(next)
                                                ? next
                                                : PREVIEW_LIMIT_DEFAULT,
                                            );
                                          }}
                                        >
                                          {PREVIEW_LIMIT_OPTIONS.map((limit) => (
                                            <option key={limit} value={limit}>
                                              {limit} rows
                                            </option>
                                          ))}
                                        </select>
                                      </label>
                                    </div>
                                  </div>
                                  {runCsvPreviewLoading ? (
                                    <div className="polymarket-csv-preview-empty">
                                      Loading preview…
                                    </div>
                                  ) : runCsvPreviewError ? (
                                    <div className="error">{runCsvPreviewError}</div>
                                  ) : runCsvPreviewResponse ? (
                                    <>
                                      {runCsvPreviewResponse.headers.length > 0 ? (
                                        <div className="table-container polymarket-csv-preview-table">
                                          <table className="preview-table">
                                            <thead>
                                              <tr>
                                                {runCsvPreviewResponse.headers.map((column) => (
                                                  <th key={column}>{column}</th>
                                                ))}
                                              </tr>
                                            </thead>
                                            <tbody>
                                              {runCsvPreviewResponse.rows.length > 0 ? (
                                                runCsvPreviewResponse.rows.map((row, index) => (
                                                  <tr key={index}>
                                                    {runCsvPreviewResponse.headers.map((column) => (
                                                      <td key={column}>{row[column] ?? ""}</td>
                                                    ))}
                                                  </tr>
                                                ))
                                              ) : (
                                                <tr>
                                                  <td
                                                    colSpan={
                                                      runCsvPreviewResponse.headers.length || 1
                                                    }
                                                  >
                                                    No rows to display.
                                                  </td>
                                                </tr>
                                              )}
                                            </tbody>
                                          </table>
                                        </div>
                                      ) : (
                                        <div className="polymarket-csv-preview-empty">
                                          CSV preview did not include column headers.
                                        </div>
                                      )}
                                      <div className="polymarket-csv-preview-meta">
                                        <span className="meta-label">
                                          Showing{" "}
                                          {runCsvPreviewResponse.mode === "tail"
                                            ? "last"
                                            : "first"}{" "}
                                          ({runCsvPreviewResponse.limit} rows)
                                        </span>
                                        <span>
                                          {runCsvPreviewResponse.row_count != null
                                            ? `${runCsvPreviewResponse.row_count.toLocaleString()} total rows`
                                            : "Row count unknown"}
                                        </span>
                                      </div>
                                    </>
                                  ) : null}
                                </div>
                              ) : null}
                            </div>
                          ) : null}
                        </article>
                      );
                    })}
                  </div>
                )}
              </div>
            </section>
          </div>
        ) : (
          <div
            id="polymarket-panel-documentation"
            role="tabpanel"
            aria-labelledby="polymarket-tab-documentation"
            className="polymarket-tab-panel"
          >
            <section className="panel polymarket-documentation-panel">
              <div className="panel-header polymarket-panel-header polymarket-job-config-header">
                <div>
                  <h2 className="polymarket-job-config-title">Documentation</h2>
                  <span className="panel-hint">
                    Page-local guidance for running backfills and managing build history.
                  </span>
                </div>
              </div>
              <div className="panel-body polymarket-documentation-body">
                <section className="section-card polymarket-doc-section">
                  <h3 className="polymarket-doc-title">Workspace Tabs</h3>
                  <div className="polymarket-doc-list">
                    <p>
                      <strong>Run job</strong> contains the configuration form and the
                      state-driven <em>Active Run</em> panel used for progress and logs.
                    </p>
                    <p>
                      <strong>Run directory</strong> lists prior weekly backfill runs and
                      preserves per-run CSV browsing/preview, rename, activate, and
                      delete actions.
                    </p>
                    <p>
                      <strong>Documentation</strong> is page-local reference content and does
                      not modify runtime state.
                    </p>
                  </div>
                </section>

                <section className="section-card polymarket-doc-section">
                  <h3 className="polymarket-doc-title">Weekly History Workflow</h3>
                  <div className="polymarket-doc-list">
                    <p>
                      Weekly history mode requires <strong>start</strong> and
                      <strong> end</strong> dates to auto-generate Friday event URLs for the
                      selected trading-universe tickers.
                    </p>
                    <p>
                      The optional <strong>Run directory name</strong> field is sanitized to
                      kebab-case and used as the on-disk run folder name when provided.
                    </p>
                    <p>
                      Optional feature building requires selecting an Option Chain dataset
                      directory with a resolved <code>training-*.csv</code> pRN file.
                    </p>
                    <p>
                      Event URLs are auto-generated for each Friday in the selected date
                      range using:
                      {" "}
                      <code>
                        https://polymarket.com/event/{"{ticker}"}-above-on-{"{month}"}-{"{friday_day}"}-{"{year}"}
                      </code>
                    </p>
                    <p>
                      After launching a run, the UI transitions to <em>Active Run</em> so
                      progress and logs remain visible without changing API behavior.
                    </p>
                    <p>
                      Run-directory CSV files are normalized to the
                      {" "}
                      <code>{"{run-directory}-{csv-type}.csv"}</code>
                      convention for consistent browsing and export naming.
                    </p>
                  </div>
                </section>

                <section className="section-card polymarket-doc-section">
                  <h3 className="polymarket-doc-title">Build History Actions</h3>
                  <div className="polymarket-doc-list">
                    <p>
                      Use <strong>Rename run</strong> to update the optional run label
                      (saved immediately).
                    </p>
                    <p>
                      If a run is missing <code>decision_features.csv</code>, use{" "}
                      <strong>Build decision features</strong> to generate the file
                      inside the run directory and pick a training dataset from the
                      Option Chain History list. Use the backfill control in the
                      modal if the training dataset needs additional overlap; it
                      extends the dataset and recomputes weights automatically.
                    </p>
                    <p>
                      Each run can be expanded to view every CSV in the directory and preview
                      rows inline with First/Last controls.
                    </p>
                    <p>
                      <strong>Activate</strong> updates the active run pointer, and
                      <strong> Delete</strong> opens a confirmation modal.
                    </p>
                    <p>
                      Deletion still requires typing <code>DELETE</code> before the destructive
                      button is enabled.
                    </p>
                  </div>
                </section>
              </div>
            </section>
          </div>
        )}
      </div>

      {featuresModalRunId ? (
        <div
          className="polymarket-features-modal-overlay"
          onClick={handleCloseFeaturesModal}
        >
          <div
            className="polymarket-features-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="polymarket-features-modal-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="polymarket-features-modal-header">
              <h3 id="polymarket-features-modal-title">Build decision features</h3>
              <p>
                Choose a training dataset from Option Chain History to populate
                decision features for{" "}
                <span className="polymarket-features-modal-code">
                  {featuresModalRunLabel}
                </span>
                .
              </p>
            </div>
            <div className="polymarket-features-modal-body">
              {isDatasetRunsLoading ? (
                <div className="polymarket-features-modal-empty">
                  Loading option-chain datasets…
                </div>
              ) : datasetRunsError ? (
                <div className="error">{datasetRunsError}</div>
              ) : optionChainTrainingRuns.length > 0 ? (
                <div className="polymarket-features-dataset-list">
                  {optionChainTrainingRuns.map((run) => {
                    const trainingFile = run.training_file;
                    const runName = run.run_dir.split("/").pop() ?? run.id;
                    const isSelected = featuresModalDatasetId === run.id;
                    return (
                      <button
                        key={run.id}
                        type="button"
                        className={`polymarket-features-dataset-card${isSelected ? " selected" : ""}`}
                        onClick={() => {
                          setFeaturesModalDatasetId(run.id);
                          setFeaturesModalError(null);
                          setBackfillError(null);
                          setBackfillMessage(null);
                        }}
                      >
                        <div className="polymarket-features-dataset-title">{runName}</div>
                        <div className="polymarket-features-dataset-meta">
                          <span>{trainingFile?.name ?? "training-*.csv"}</span>
                          <span>
                            {formatDateTime(trainingFile?.last_modified ?? run.last_modified)}
                          </span>
                        </div>
                        <div className="polymarket-features-dataset-path">
                          {trainingFile?.path ?? run.run_dir}
                        </div>
                      </button>
                    );
                  })}
                </div>
              ) : (
                <div className="polymarket-features-modal-empty">
                  No option-chain training datasets found.
                </div>
              )}
              {featuresModalError ? (
                <div className="error">{featuresModalError}</div>
              ) : null}
              <div className="polymarket-features-backfill">
                <div className="polymarket-features-backfill-header">
                  <div>
                    <div className="polymarket-features-backfill-title">
                      Missing overlap?
                    </div>
                    <p className="polymarket-features-backfill-copy">
                      Backfill the selected option-chain dataset with missing expiry
                      ranges for this run and recompute weights to keep training
                      logic consistent.
                    </p>
                  </div>
                  <button
                    className="button light small"
                    type="button"
                    onClick={handleBackfillDataset}
                    disabled={
                      backfillLoading ||
                      isFeaturesBuildRunning ||
                      !featuresModalDatasetId ||
                      anyJobRunning ||
                      isRunning
                    }
                  >
                    {backfillLoading ? "Backfilling…" : "Backfill dataset"}
                  </button>
                </div>
                <label className="polymarket-features-backfill-toggle">
                  <input
                    type="checkbox"
                    checked={backfillAllowDefaults}
                    onChange={(event) => setBackfillAllowDefaults(event.target.checked)}
                    disabled={backfillLoading}
                  />
                  Use defaults if build metadata is missing.
                </label>
                {backfillMessage ? (
                  <div className="polymarket-features-backfill-message">
                    {backfillMessage}
                  </div>
                ) : null}
                {backfillError ? (
                  <div className="error">{backfillError}</div>
                ) : null}
              </div>
            </div>
            <div className="polymarket-features-modal-actions">
              <button
                className="button ghost"
                type="button"
                onClick={handleCloseFeaturesModal}
                disabled={isFeaturesBuildRunning || backfillLoading}
              >
                Cancel
              </button>
              <button
                className="button primary"
                type="button"
                onClick={handleConfirmBuildDecisionFeatures}
                disabled={
                  isFeaturesBuildRunning ||
                  backfillLoading ||
                  optionChainTrainingRuns.length === 0
                }
              >
                {isFeaturesBuildRunning ? "Building…" : "Build decision features"}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {deleteTarget ? (
        <div
          className="polymarket-delete-modal-overlay"
          onClick={() => {
            if (deleteLoading) return;
            setDeleteTarget(null);
            setDeleteConfirmText("");
          }}
        >
          <div
            className="polymarket-delete-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="polymarket-delete-modal-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="polymarket-delete-modal-header">
              <h3 id="polymarket-delete-modal-title">Delete pipeline run</h3>
              <p>
                This will permanently delete{" "}
                <span className="polymarket-delete-modal-code">{deleteTarget}</span>{" "}
                and all associated artifacts.
              </p>
            </div>
            <div className="polymarket-delete-modal-body">
              <label htmlFor="polymarketDeleteConfirmInput">
                Type <strong>DELETE</strong> to confirm
              </label>
              <input
                id="polymarketDeleteConfirmInput"
                className="input"
                type="text"
                value={deleteConfirmText}
                onChange={(event) => setDeleteConfirmText(event.target.value)}
                placeholder="DELETE"
                autoFocus
                disabled={deleteLoading}
              />
            </div>
            <div className="polymarket-delete-modal-actions">
              <button
                className="button ghost"
                type="button"
                onClick={() => {
                  setDeleteTarget(null);
                  setDeleteConfirmText("");
                }}
                disabled={deleteLoading}
              >
                Cancel
              </button>
              <button
                className="button danger polymarket-delete-modal-confirm"
                type="button"
                disabled={deleteConfirmText !== "DELETE" || deleteLoading}
                onClick={handleDeleteConfirm}
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
