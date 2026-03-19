import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";

import {
  startPolymarketHistoryJob,
  cancelPolymarketHistoryJob,
  listPipelineRuns,
  getPipelineRunQualityAudit,
  renamePipelineRun,
  setActiveRun,
  deletePipelineRun,
  getPipelineRunArtifactFileUrl,
  getRunMasterBarFileUrl,
  previewPipelineRunArtifactCsv,
  buildDecisionFeaturesForRun,
  type CsvPreview,
  type PolymarketQualityAuditResponse,
  type PolymarketQualitySummary,
  type PolymarketQualityTelemetry,
  type PipelineProgress,
  type PipelineRunSummary,
  type RunArtifactSummary,
  type SharedArtifactSummary,
} from "../api/polymarketHistory";
import PipelineStatusCard from "../components/PipelineStatusCard";
import PipelineProgressBar from "../components/PipelineProgressBar";
import { usePolymarketHistoryJob } from "../contexts/polymarketHistoryJob";
import { useAnyJobRunning } from "../contexts/jobGuard";
import "./DatasetsPage.css";
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

const DEFAULT_HISTORY_FIDELITY_MIN = 60;
const DEFAULT_HISTORY_BAR_FREQS = "1d,1h";

const defaultForm = {
  tickers: TRADING_UNIVERSE_TICKERS.join(", "),
  historyRunDirName: "",
  historyStartDate: "",
  historyEndDate: "",
  historyIncludeSubgraph: true,
  historyBuildFeatures: false,
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

const formatPercent = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return `${(value * 100).toFixed(1)}%`;
};

const formatQualityFlags = (summary?: PolymarketQualitySummary | null) =>
  summary?.top_flags?.slice(0, 5) ?? [];

const formatProblemTickers = (summary?: PolymarketQualitySummary | null) =>
  summary?.top_problem_tickers?.slice(0, 5) ?? [];

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

const DATE_RANGE_ERROR = "End date must be on or after start date.";

const hasInvalidDateRange = (start: string, end: string): boolean => {
  if (!start || !end) return false;
  const startDate = new Date(`${start}T00:00:00Z`);
  const endDate = new Date(`${end}T00:00:00Z`);
  if (Number.isNaN(startDate.getTime()) || Number.isNaN(endDate.getTime())) {
    return false;
  }
  return endDate < startDate;
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
    return { urls: [], fridays: [], error: DATE_RANGE_ERROR };
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

type RunArtifactPreviewTarget = {
  runId: string;
  path: string;
  label: string;
};

type RunCsvFileSummary = NonNullable<PipelineRunSummary["csv_files"]>[number];
type MasterBarPreviewTarget = "1h" | "1d";

const PREVIEW_LIMIT_DEFAULT = 20;
const PREVIEW_LIMIT_OPTIONS = [20, 50, 100] as const;
const PREVIEW_MODE_OPTIONS: { value: PreviewMode; label: string }[] = [
  { value: "head", label: "First" },
  { value: "tail", label: "Last" },
];

const sortRunCsvFiles = (files: RunCsvFileSummary[]): RunCsvFileSummary[] =>
  [...files].sort((a, b) => a.name.localeCompare(b.name));

const sortSharedArtifacts = (files: SharedArtifactSummary[]): SharedArtifactSummary[] =>
  [...files].sort((a, b) => (a.frequency ?? a.name).localeCompare(b.frequency ?? b.name));

const sortRunArtifacts = (files: RunArtifactSummary[]): RunArtifactSummary[] =>
  [...files].sort((a, b) => a.path.localeCompare(b.path));

const isDecisionFeaturesCsv = (name: string): boolean => {
  const lower = name.toLowerCase();
  return lower === "decision_features.csv" || lower.endsWith("decision-features.csv");
};

export default function PolymarketPipelinePage() {
  const [form, setForm] = useState<FormState>(() => loadStoredForm() ?? defaultForm);
  const [formError, setFormError] = useState<string | null>(null);
  const [stopLoading, setStopLoading] = useState(false);
  const [workspaceTab, setWorkspaceTab] = useState<"run_job" | "run_directory">(
    "run_job",
  );
  const [runJobPanel, setRunJobPanel] = useState<"configuration" | "active_run">(
    "configuration",
  );
  const [activeLogView, setActiveLogView] = useState<"stdout" | "stderr" | null>(null);

  // --- Runs browser state ---
  const [pipelineRuns, setPipelineRuns] = useState<PipelineRunSummary[]>([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [renamingRunId, setRenamingRunId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [renameError, setRenameError] = useState<string | null>(null);
  const [renameLoading, setRenameLoading] = useState(false);
  const [openRunId, setOpenRunId] = useState<string | null>(null);
  const [runArtifactPreviewTarget, setRunArtifactPreviewTarget] =
    useState<RunArtifactPreviewTarget | null>(null);
  const [runArtifactPreviewResponse, setRunArtifactPreviewResponse] =
    useState<CsvPreview | null>(null);
  const [runArtifactPreviewError, setRunArtifactPreviewError] = useState<string | null>(null);
  const [runArtifactPreviewLoading, setRunArtifactPreviewLoading] = useState(false);
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
  const [featuresModalError, setFeaturesModalError] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [deleteConfirmText, setDeleteConfirmText] = useState("");
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [historyProgressState, setHistoryProgressState] = useState<PipelineProgress | null>(null);
  const [featuresProgressState, setFeaturesProgressState] = useState<PipelineProgress | null>(null);
  const [qualityAuditByRun, setQualityAuditByRun] = useState<Record<string, PolymarketQualityAuditResponse>>({});
  const [qualityAuditLoadingRunId, setQualityAuditLoadingRunId] = useState<string | null>(null);
  const [qualityAuditError, setQualityAuditError] = useState<Record<string, string>>({});
  const lastHistoryJobId = useRef<string | null>(null);
  const didAutoOpenActiveRunRef = useRef(false);

  const {
    jobStatus: historyJobStatus,
    setJobId: setHistoryJobId,
    refreshJob: refreshHistoryJob,
  } = usePolymarketHistoryJob();
  const { anyJobRunning, activeJobs } = useAnyJobRunning();

  const isRunning =
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

  useEffect(() => {
    const status = historyJobStatus?.status;
    const isActive = status === "queued" || status === "running";
    const isTerminal =
      status === "finished" || status === "failed" || status === "cancelled";

    if (isActive && !didAutoOpenActiveRunRef.current) {
      setWorkspaceTab("run_job");
      setRunJobPanel("active_run");
      didAutoOpenActiveRunRef.current = true;
      return;
    }

    if (!status || isTerminal) {
      didAutoOpenActiveRunRef.current = false;
    }
  }, [historyJobStatus?.status]);

  // --- Runs browser: load on mount + after pipeline finishes ---
  const loadRuns = useCallback(() => {
    setRunsLoading(true);
    setRunsError(null);
    listPipelineRuns()
      .then((data) => {
        setPipelineRuns(data.runs);
      })
      .catch((err) => {
        setRunsError(err instanceof Error ? err.message : "Failed to load runs");
      })
      .finally(() => setRunsLoading(false));
  }, []);

  useEffect(() => { loadRuns(); }, [loadRuns]);

  useEffect(() => {
    if (!openRunId || qualityAuditByRun[openRunId] || qualityAuditLoadingRunId === openRunId) {
      return;
    }
    let cancelled = false;
    setQualityAuditLoadingRunId(openRunId);
    setQualityAuditError((prev) => {
      if (!(openRunId in prev)) return prev;
      const next = { ...prev };
      delete next[openRunId];
      return next;
    });
    getPipelineRunQualityAudit(openRunId)
      .then((audit) => {
        if (cancelled) return;
        setQualityAuditByRun((prev) => ({ ...prev, [openRunId]: audit }));
      })
      .catch((err) => {
        if (cancelled) return;
        setQualityAuditError((prev) => ({
          ...prev,
          [openRunId]: err instanceof Error ? err.message : "Failed to load quality audit",
        }));
      })
      .finally(() => {
        if (!cancelled) {
          setQualityAuditLoadingRunId((prev) => (prev === openRunId ? null : prev));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [openRunId, qualityAuditByRun, qualityAuditLoadingRunId]);

  // Refresh runs list when a job finishes
  const prevHistoryStatus = useMemo(() => historyJobStatus?.status, [historyJobStatus?.status]);
  useEffect(() => {
    if (prevHistoryStatus === "finished" || prevHistoryStatus === "failed" || prevHistoryStatus === "cancelled") {
      loadRuns();
    }
  }, [prevHistoryStatus, loadRuns]);

  useEffect(() => {
    if (!runArtifactPreviewTarget) {
      setRunArtifactPreviewResponse(null);
      setRunArtifactPreviewError(null);
      setRunArtifactPreviewLoading(false);
      return;
    }

    let cancelled = false;
    setRunArtifactPreviewLoading(true);
    setRunArtifactPreviewError(null);
    previewPipelineRunArtifactCsv(
      runArtifactPreviewTarget.runId,
      runArtifactPreviewTarget.path,
      runCsvPreviewMode,
      runCsvPreviewLimit,
    )
      .then((preview) => {
        if (cancelled) return;
        setRunArtifactPreviewResponse(preview);
      })
      .catch((err) => {
        if (cancelled) return;
        setRunArtifactPreviewResponse(null);
        setRunArtifactPreviewError(err instanceof Error ? err.message : "Unknown error");
      })
      .finally(() => {
        if (!cancelled) setRunArtifactPreviewLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [runArtifactPreviewTarget, runCsvPreviewMode, runCsvPreviewLimit]);

  useEffect(() => {
    if (!runArtifactPreviewTarget) return;
    const previewRunStillExists = pipelineRuns.some(
      (run) => run.run_id === runArtifactPreviewTarget.runId,
    );
    if (!previewRunStillExists) {
      setRunArtifactPreviewTarget(null);
    }
  }, [pipelineRuns, runArtifactPreviewTarget]);

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
      await renamePipelineRun(runId, cleaned, cleaned || null);
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
  }, []);

  const handleCloseFeaturesModal = useCallback(() => {
    if (featuresBuildRunId) return;
    setFeaturesModalRunId(null);
    setFeaturesModalError(null);
  }, [featuresBuildRunId]);

  const handleConfirmBuildDecisionFeatures = useCallback(async () => {
    if (!featuresModalRunId || featuresBuildRunId) return;

    setFeaturesBuildRunId(featuresModalRunId);
    setFeaturesBuildError(null);
    setFeaturesModalError(null);
    try {
      const response = await buildDecisionFeaturesForRun(featuresModalRunId);
      if (!response.ok) {
        throw new Error(response.stderr || response.stdout || "Decision features build failed.");
      }
      await loadRuns();
      setFeaturesModalRunId(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Decision features build failed.";
      setFeaturesBuildError({ runId: featuresModalRunId, message });
      setFeaturesModalError(message);
    } finally {
      setFeaturesBuildRunId(null);
    }
  }, [featuresBuildRunId, featuresModalRunId, loadRuns]);

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

  const handleToggleRunArtifactPreview = useCallback((target: RunArtifactPreviewTarget) => {
    setRunArtifactPreviewTarget((prev) => {
      if (prev && prev.runId === target.runId && prev.path === target.path) {
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
      if (runArtifactPreviewTarget?.runId === deleteTarget) {
        setRunArtifactPreviewTarget(null);
      }
      setDeleteTarget(null);
      setDeleteConfirmText("");
      loadRuns();
    } catch (err) {
      console.error("Delete failed:", err);
    } finally {
      setDeleteLoading(false);
    }
  }, [deleteTarget, deleteConfirmText, loadRuns, runArtifactPreviewTarget]);

  const formatSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const historyRunDirNameKebab = toKebabCase(form.historyRunDirName);
  const pipelineStatus = historyJobStatus?.status || "idle";
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
    if (historyPhase === "prn") return "Refreshing run-local pRN";
    if (historyPhase === "quality") return "Computing market quality flags";
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
      setFormError(null);

      if (form.historyRunDirName.trim() && !historyRunDirNameKebab) {
        setFormError("Run directory name must contain at least one alphanumeric character.");
        return;
      }
      if (historyDateRangeInvalid) {
        setFormError(DATE_RANGE_ERROR);
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

      const payload = {
        tickers: effectiveTickers,
        eventUrls: autoUrls.urls,
        startDate: form.historyStartDate || undefined,
        endDate: form.historyEndDate || undefined,
        fidelityMin: DEFAULT_HISTORY_FIDELITY_MIN,
        barsFreqs: DEFAULT_HISTORY_BAR_FREQS,
        runDirName: form.historyRunDirName.trim() || undefined,
        includeSubgraph: form.historyIncludeSubgraph,
        buildFeatures: form.historyBuildFeatures,
        skipSubgraphLabels: false,
      };
      const status = await startPolymarketHistoryJob(payload);
      setHistoryJobId(status.job_id);
      setRunJobPanel("active_run");
      setWorkspaceTab("run_job");
      setActiveLogView(null);
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

  const handleResetConfig = useCallback(() => {
    setForm({ ...defaultForm });
    setFormError(null);
    setWorkspaceTab("run_job");
    setRunJobPanel("configuration");
    setActiveLogView(null);
  }, []);

  const handleViewLatestRun = useCallback(() => {
    setWorkspaceTab("run_job");
    setRunJobPanel("active_run");
    setActiveLogView(null);
  }, []);

  const historyStdout = historyJobStatus?.result?.stdout ?? "";
  const historyStderr = historyJobStatus?.result?.stderr ?? "";
  const historyError = historyJobStatus?.error ?? "";

  const selectedTickers = useMemo(() => {
    const parsed = normalizeTickers(parseTickers(form.tickers));
    return orderTickers(parsed);
  }, [form.tickers]);
  const todayDateString = useMemo(
    () => new Date().toISOString().slice(0, 10),
    [],
  );
  const historyDateRangeInvalid = useMemo(
    () => hasInvalidDateRange(form.historyStartDate, form.historyEndDate),
    [form.historyStartDate, form.historyEndDate],
  );

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

  const handleHistoryStartDateChange = useCallback((value: string) => {
    setForm((prev) => ({ ...prev, historyStartDate: value }));
    setFormError((current) => (current === DATE_RANGE_ERROR ? null : current));
  }, []);

  const handleHistoryEndDateChange = useCallback((value: string) => {
    setForm((prev) => ({ ...prev, historyEndDate: value }));
    setFormError((current) => (current === DATE_RANGE_ERROR ? null : current));
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
    if (historyDateRangeInvalid) {
      return { urls: [], fridays: [], error: DATE_RANGE_ERROR };
    }
    return buildAutoEventUrls(tickers, form.historyStartDate, form.historyEndDate);
  }, [
    selectedTickers,
    form.historyStartDate,
    form.historyEndDate,
    historyDateRangeInvalid,
  ]);

  const hasEventSources = (autoEventState?.urls.length ?? 0) > 0;
  const canStopHistory =
    historyJobStatus?.status === "queued" || historyJobStatus?.status === "running";

  const defaultActiveLog = historyStderr || historyError ? "stderr" : "stdout";
  const activeLog = activeLogView;
  const hasAnyLogOutput = Boolean(historyStdout || historyStderr || historyError);
  const terminalStatus =
    pipelineStatus === "finished" ||
    pipelineStatus === "failed" ||
    pipelineStatus === "cancelled";
  const hasRunRecord = Boolean(historyJobStatus);
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
  const resolvedConfigDateRange =
    form.historyStartDate && form.historyEndDate
      ? `${form.historyStartDate} -> ${form.historyEndDate}`
      : "Select a date range";
  const resolvedConfigTickersLabel = `${selectedTickers.length} tickers`;
  const resolvedConfigDatasetPath = `src/data/raw/polymarket/weekly_history/runs/${
    historyRunDirNameKebab || "(auto-named)"
  }`;
  const hasResolvedAutoEventSummary =
    Boolean(form.historyStartDate && form.historyEndDate) &&
    !historyDateRangeInvalid &&
    !autoEventState.error;
  const resolvedConfigEventUrlsLabel = hasResolvedAutoEventSummary
    ? `${autoEventState.urls.length} URLs`
    : "Event URLs pending";
  const resolvedConfigWeeksLabel = hasResolvedAutoEventSummary
    ? `${autoEventState.fridays.length} weeks`
    : "Weeks pending";
  const historyEventCountLabel = formatCount(autoEventState?.urls.length);
  const historyProgressLabel = historyProgress
    ? `${historyProgress.completed} / ${historyProgress.total} jobs completed (${historyProgressPercent}%)`
    : historyRunning
      ? "Running pipeline..."
      : pipelineStatusLabel;
  const runJobDisabled = isRunning || anyJobRunning || isFeaturesBuildRunning;
  const runJobLabel = isRunning
    ? "Running job..."
    : isFeaturesBuildRunning
      ? "Building decision features..."
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
  const activeQualityTelemetry: PolymarketQualityTelemetry | null =
    historyJobStatus?.telemetry ?? null;
  const activeQualitySummary: PolymarketQualitySummary | null =
    historyJobStatus?.result?.quality_summary ?? null;
  const activeQualityFlags = activeQualityTelemetry?.top_flags ?? formatQualityFlags(activeQualitySummary);
  const activeQualityTickers =
    activeQualityTelemetry?.top_problem_tickers ?? formatProblemTickers(activeQualitySummary);
  const activeQualityBucketCounts =
    activeQualityTelemetry?.bucket_counts ?? activeQualitySummary?.bucket_counts ?? null;
  const activeQualityCoverage =
    activeQualityTelemetry?.prn_coverage_counts ?? activeQualitySummary?.prn_coverage_counts ?? {};
  const activeQualityPrnMissingCount = activeQualityCoverage["missing"] ?? 0;

  return (
    <section className="page polymarket-pipeline-page">
      <PipelineStatusCard
        className="page-sticky-meta polymarket-meta"
        activeJobsCount={activeJobs.length}
      />
      <header className="page-header polymarket-page-header">
        <div className="polymarket-title-row">
          <h1 className="page-title polymarket-page-title">
            Polymarket History Builder
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
            id="polymarket-tab-run-directory"
            type="button"
            role="tab"
            aria-selected={workspaceTab === "run_directory"}
            aria-controls="polymarket-panel-run-directory"
            className={`polymarket-workspace-tab ${
              workspaceTab === "run_directory" ? "active" : ""
            }`}
            onClick={() => setWorkspaceTab("run_directory")}
          >
            Datasets
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
                  </div>
                  <div className="polymarket-job-config-actions">
                    <button
                      className="button ghost polymarket-config-action-button"
                      type="button"
                      disabled={isRunning}
                      onClick={handleResetConfig}
                    >
                      Reset config
                    </button>
                    <button
                      className="button ghost polymarket-config-action-button"
                      type="button"
                      disabled={!hasRunRecord}
                      onClick={handleViewLatestRun}
                    >
                      View latest run
                    </button>
                  </div>
                </div>
                <div className="config-summary">
                  <div>
                    <span className="meta-label">Date range</span>
                    <span>{resolvedConfigDateRange}</span>
                  </div>
                  <div>
                    <span className="meta-label">Tickers</span>
                    <span>{resolvedConfigTickersLabel}</span>
                  </div>
                  <div>
                    <span className="meta-label">Dataset</span>
                    <span>{resolvedConfigDatasetPath}</span>
                  </div>
                  <div>
                    <span className="meta-label">Event URLs</span>
                    <span>{resolvedConfigEventUrlsLabel}</span>
                  </div>
                  <div>
                    <span className="meta-label">Weeks</span>
                    <span>{resolvedConfigWeeksLabel}</span>
                  </div>
                </div>
                <form className="panel-body polymarket-config-form" onSubmit={handleRunPipeline}>
                  <div className="section-card polymarket-config-card polymarket-basic-settings-card">
                    <h3>Basic settings</h3>
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
                      </div>
                    </div>
                    <div className="fields-grid">
                      <div className="field full">
                        <div className="polymarket-date-range-fields">
                          <div className="field">
                            <label htmlFor="polymarketHistoryStartDate">
                              History start date (UTC)
                            </label>
                            <input
                              id="polymarketHistoryStartDate"
                              className={`input ${
                                historyDateRangeInvalid ? "input-invalid" : ""
                              }`}
                              type="date"
                              min="2023-06-01"
                              max={todayDateString}
                              required
                              value={form.historyStartDate}
                              aria-invalid={historyDateRangeInvalid}
                              onChange={(event) =>
                                handleHistoryStartDateChange(event.target.value)
                              }
                            />
                          </div>
                          <div className="field">
                            <label htmlFor="polymarketHistoryEndDate">
                              History end date (UTC)
                            </label>
                            <input
                              id="polymarketHistoryEndDate"
                              className={`input ${
                                historyDateRangeInvalid ? "input-invalid" : ""
                              }`}
                              type="date"
                              min="2023-06-01"
                              max={todayDateString}
                              required
                              value={form.historyEndDate}
                              aria-invalid={historyDateRangeInvalid}
                              onChange={(event) =>
                                handleHistoryEndDateChange(event.target.value)
                              }
                            />
                          </div>
                        </div>
                        <p
                          className={`field-hint polymarket-date-range-hint${
                            historyDateRangeInvalid ? " is-invalid" : ""
                          }`}
                        >
                          {historyDateRangeInvalid ? DATE_RANGE_ERROR : ""}
                        </p>
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
                              Use subgraph-first bars
                            </span>
                            <span className="polymarket-toggle-card-copy">
                              YES-side trade bars come from the subgraph first, with CLOB fallback for missing timestamps.
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
                    </div>
                  </div>

                  {!hasEventSources && !historyDateRangeInvalid ? (
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
                <div className="panel-body">
                  {historyJobStatus ? (
                    <div className="polymarket-active-run-shell">
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
                        </div>
                        {canStopHistory ? (
                          <button
                            className="button ghost danger polymarket-stop-run-button"
                            type="button"
                            onClick={handleStopRun}
                            disabled={stopLoading}
                          >
                            {stopLoading ? "Stopping…" : "Stop run"}
                          </button>
                        ) : null}
                      </div>

                      <div className="polymarket-run-summary polymarket-quality-summary">
                        <div className="polymarket-run-summary-header">
                          <div>
                            <span className="meta-label">Quality snapshot</span>
                            <div className="polymarket-run-summary-title">
                              Market quality audit
                            </div>
                          </div>
                        </div>
                        {activeQualityTelemetry || activeQualitySummary ? (
                          <>
                            <div className="polymarket-run-meta-grid polymarket-quality-kpis">
                              <div>
                                <span className="meta-label">Flagged</span>
                                <span>
                                  {formatCount(
                                    activeQualityTelemetry?.flagged_markets ??
                                      activeQualitySummary?.flagged_market_count,
                                  )}{" "}
                                  ({formatPercent(
                                    activeQualityTelemetry?.flagged_share ??
                                      activeQualitySummary?.flagged_share,
                                  )})
                                </span>
                              </div>
                              <div>
                                <span className="meta-label">Clean</span>
                                <span>{formatCount(activeQualityBucketCounts?.clean)}</span>
                              </div>
                              <div>
                                <span className="meta-label">Watch</span>
                                <span>{formatCount(activeQualityBucketCounts?.watch)}</span>
                              </div>
                              <div>
                                <span className="meta-label">Noisy</span>
                                <span>{formatCount(activeQualityBucketCounts?.noisy)}</span>
                              </div>
                              <div>
                                <span className="meta-label">PRN missing</span>
                                <span>{formatCount(activeQualityPrnMissingCount)}</span>
                              </div>
                              <div>
                                <span className="meta-label">Progress</span>
                                <span>
                                  {activeQualityTelemetry
                                    ? `${activeQualityTelemetry.completed_markets} / ${activeQualityTelemetry.total_markets}`
                                    : formatCount(activeQualitySummary?.market_count)}
                                </span>
                              </div>
                            </div>
                            <div className="polymarket-quality-detail-grid">
                              <section className="dataset-audit-card">
                                <div className="dataset-audit-card-header">
                                  <h3>Top flags</h3>
                                  <span>Most common issues</span>
                                </div>
                                {activeQualityFlags.length > 0 ? (
                                  <div className="dataset-audit-list">
                                    {activeQualityFlags.map((flag) => (
                                      <div key={flag.name} className="dataset-audit-list-row">
                                        <div className="dataset-audit-list-label">
                                          <strong>{flag.name.replace(/^flag_/, "")}</strong>
                                          <span>{formatPercent(flag.share ?? null)}</span>
                                        </div>
                                        <div className="dataset-audit-bar-track">
                                          <span
                                            className="dataset-audit-bar-fill"
                                            style={{ width: `${Math.max(4, (flag.share ?? 0) * 100)}%` }}
                                          />
                                        </div>
                                        <span className="dataset-audit-list-value">
                                          {flag.count.toLocaleString()}
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                ) : (
                                  <div className="polymarket-quality-empty">No flags recorded yet.</div>
                                )}
                              </section>
                              <section className="dataset-audit-card">
                                <div className="dataset-audit-card-header">
                                  <h3>Problem tickers</h3>
                                  <span>Highest flagged share</span>
                                </div>
                                {activeQualityTickers.length > 0 ? (
                                  <div className="dataset-audit-table">
                                    <div className="dataset-audit-table-head">
                                      <div>Ticker</div>
                                      <div>Flagged</div>
                                      <div>Avg issues</div>
                                    </div>
                                    {activeQualityTickers.map((tickerSummary) => (
                                      <div
                                        key={tickerSummary.ticker}
                                        className="dataset-audit-table-row"
                                      >
                                        <div>
                                          <strong>{tickerSummary.ticker}</strong>
                                        </div>
                                        <div>
                                          {formatPercent(tickerSummary.flagged_share ?? null)}
                                        </div>
                                        <div>{tickerSummary.avg_issue_count?.toFixed(2) ?? "--"}</div>
                                      </div>
                                    ))}
                                  </div>
                                ) : (
                                  <div className="polymarket-quality-empty">
                                    Ticker rankings will appear once the quality phase runs.
                                  </div>
                                )}
                              </section>
                            </div>
                          </>
                        ) : (
                          <div className="polymarket-quality-empty">
                            Quality audit starts after history and exact pRN are ready.
                          </div>
                        )}
                      </div>

                      <div className="polymarket-run-summary">
                        <div className="polymarket-run-summary-header">
                          <div>
                            <span className="meta-label">Output</span>
                            <div className="polymarket-run-summary-title">
                              {historyRunDir}
                            </div>
                          </div>
                          <div className="polymarket-run-summary-actions">
                            <span className={`status-pill ${pipelineStatusClass}`}>
                              {pipelineStatusLabel}
                            </span>
                          </div>
                        </div>
                        <div className="polymarket-run-meta-grid">
                          <div>
                            <span className="meta-label">Duration</span>
                            <span>{historyDurationLabel}</span>
                          </div>
                          <div>
                            <span className="meta-label">Output dir</span>
                            <span>{historyRunDir}</span>
                          </div>
                          <div>
                            <span className="meta-label">Decision features</span>
                            <span>
                              {historyJobStatus?.result?.features_built
                                ? "Written"
                                : form.historyBuildFeatures
                                  ? historyRunning
                                    ? "Pending"
                                    : "Not written"
                                  : "Disabled"}
                            </span>
                          </div>
                          <div>
                            <span className="meta-label">Files</span>
                            <span>{historyFilesLabel}</span>
                          </div>
                        </div>
                      </div>

                      <div className="polymarket-log-panel">
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
                  ) : (
                    <div className="empty">
                      Start a weekly history run to see progress and logs.
                    </div>
                  )}
                </div>
              </section>
            )}
          </div>
        ) : (
          <div
            id="polymarket-panel-run-directory"
            role="tabpanel"
            aria-labelledby="polymarket-tab-run-directory"
            className="polymarket-tab-panel"
          >
            <section className="panel polymarket-history-panel">
              <div className="panel-header polymarket-panel-header polymarket-job-config-header">
                <div>
                  <h2 className="polymarket-job-config-title">Datasets</h2>
                </div>
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
                    No pipeline runs found. Run the weekly history job to create your
                    first run.
                  </div>
                ) : (
                  <div className="polymarket-runs-list">
                    {pipelineRuns.map((run) => {
                      const statusClass =
                        run.status === "success"
                          ? "success"
                          : run.status === "pending"
                            ? "pending"
                            : run.status === "queued" || run.status === "running"
                              ? "running"
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
                      const runTitle = displayLabel || run.run_id;
                      const csvFiles = sortRunCsvFiles(run.csv_files ?? []);
                      const artifactGroups = (run.artifact_groups ?? []).filter(
                        (group) => group.key !== "bars_history",
                      );
                      const masterBarArtifacts = sortSharedArtifacts(run.master_bar_artifacts ?? []);
                      const hasDecisionFeaturesCsv = csvFiles.some((file) =>
                        isDecisionFeaturesCsv(file.name),
                      );
                      const canBuildDecisionFeatures = !hasDecisionFeaturesCsv;
                      const showPendingRunNotice =
                        run.pending_phase === "features" &&
                        !run.artifacts_accessible &&
                        run.features_requested &&
                        !run.features_built;
                      const fileCount = (run.artifact_groups ?? []).reduce(
                        (sum, group) => sum + (group.files?.length ?? 0),
                        0,
                      );
                      const artifactCountLabel = fileCount
                        ? `${fileCount} file${fileCount === 1 ? "" : "s"}`
                        : "No files";
                      const runMetaItems = [
                        formatDateTime(run.created_at_utc),
                        dateRangeLabel !== "--" ? dateRangeLabel : null,
                        `${formatCount(run.markets)} market${run.markets === 1 ? "" : "s"}`,
                        artifactCountLabel,
                        formatSize(run.size_bytes),
                        run.duration_s != null ? `${run.duration_s}s` : null,
                      ].filter((value): value is string => Boolean(value));
                      const isRenaming = renamingRunId === run.run_id;
                      const isOpen = openRunId === run.run_id;
                      const isPreviewingRun =
                        isOpen && runArtifactPreviewTarget?.runId === run.run_id;
                      const isBuildingFeatures = featuresBuildRunId === run.run_id;
                      const runQualityAudit = qualityAuditByRun[run.run_id];
                      const runQualityAuditError = qualityAuditError[run.run_id];
                      const runQualityLoading = qualityAuditLoadingRunId === run.run_id;
                      const featuresBuildErrorMessage =
                        featuresBuildError?.runId === run.run_id
                          ? featuresBuildError.message
                          : null;
                      const hasDrawerContent =
                        fileCount > 0 ||
                        masterBarArtifacts.length > 0 ||
                        Boolean(
                          showPendingRunNotice ||
                            run.error_summary ||
                            featuresBuildErrorMessage ||
                            run.quality_summary,
                        );
                      const runMainContent = (
                        <>
                          <div className="polymarket-run-heading">
                            <div className="polymarket-run-title">{runTitle}</div>
                            {run.status !== "success" ? (
                              <span className={`status-pill ${statusClass}`}>
                                {statusLabel}
                              </span>
                            ) : null}
                          </div>
                          <div className="polymarket-run-meta">
                            {runMetaItems.map((item, index) => (
                              <span key={`${run.run_id}-meta-${index}`}>{item}</span>
                            ))}
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
                              aria-expanded={hasDrawerContent ? isOpen : undefined}
                              aria-controls={
                                hasDrawerContent ? `polymarket-run-files-${run.run_id}` : undefined
                              }
                              onClick={() => {
                                if (!hasDrawerContent) return;
                                setOpenRunId((prev) =>
                                  prev === run.run_id ? null : run.run_id,
                                );
                              }}
                            >
                              <div className="polymarket-run-main">
                                {runMainContent}
                              </div>
                              {hasDrawerContent ? (
                                <span className="polymarket-run-chevron" aria-hidden="true">
                                  ▸
                                </span>
                              ) : null}
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
                                  isFeaturesBuildRunning
                                }
                                title="Build decision features for this run."
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
                              title="Rename dataset"
                            >
                              Rename dataset
                            </button>
                            <button
                              className="button ghost danger small"
                              type="button"
                              onClick={() => {
                                setDeleteTarget(run.run_id);
                                setDeleteConfirmText("");
                              }}
                              disabled={run.is_active}
                              title={
                                run.is_active
                                  ? "Set another run active before deleting this dataset."
                                  : "Delete dataset"
                              }
                            >
                              Delete dataset
                            </button>
                          </div>

                          {hasDrawerContent ? (
                            <div
                              id={`polymarket-run-files-${run.run_id}`}
                              className={`polymarket-run-files-drawer${isOpen ? " is-open" : ""}`}
                              hidden={!isOpen}
                            >
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
                              {showPendingRunNotice ? (
                                <section className="polymarket-run-pending-notice">
                                  <div>
                                    <span className="meta-label">Pending dataset</span>
                                    <h3>Decision features still need to run</h3>
                                    <p>
                                      {run.status === "running" || run.status === "queued"
                                        ? "This dataset is resuming the decision-features stage. Artifact access stays locked until features are written."
                                        : "This dataset finished history collection, but the decision-features stage did not complete. Artifact access stays locked until features are written."}
                                    </p>
                                  </div>
                                  {canBuildDecisionFeatures ? (
                                    <button
                                      className="button light small"
                                      type="button"
                                      onClick={() => handleOpenFeaturesModal(run.run_id)}
                                      disabled={
                                        isBuildingFeatures ||
                                        isRunning ||
                                        anyJobRunning ||
                                        isFeaturesBuildRunning
                                      }
                                    >
                                      {isBuildingFeatures
                                        ? "Building features..."
                                        : "Build decision features"}
                                    </button>
                                  ) : null}
                                </section>
                              ) : null}
                              <div className="polymarket-quality-audit-wrap">
                                {runQualityLoading || (!runQualityAudit && !runQualityAuditError) ? (
                                  <div className="polymarket-csv-preview-empty">
                                    Loading quality audit…
                                  </div>
                                ) : runQualityAuditError ? (
                                  <div className="error">{runQualityAuditError}</div>
                                ) : runQualityAudit?.available ? (
                                  <div className="dataset-audit-panel polymarket-quality-audit-panel">
                                    <div className="dataset-audit-header">
                                      <div>
                                        <span className="meta-label">Quality audit</span>
                                        <p className="dataset-audit-title">Market quality overview</p>
                                      </div>
                                      <div className="dataset-audit-meta">
                                        <span>
                                          {formatCount(runQualityAudit.summary.market_count)} markets
                                        </span>
                                        <span>
                                          {formatPercent(runQualityAudit.summary.flagged_share)} flagged
                                        </span>
                                      </div>
                                    </div>
                                    <div className="dataset-audit-kpis">
                                      <div>
                                        <span className="meta-label">Flagged</span>
                                        <strong>
                                          {formatCount(runQualityAudit.summary.flagged_market_count)}
                                        </strong>
                                      </div>
                                      <div>
                                        <span className="meta-label">Clean</span>
                                        <strong>
                                          {formatCount(runQualityAudit.summary.bucket_counts.clean)}
                                        </strong>
                                      </div>
                                      <div>
                                        <span className="meta-label">Watch</span>
                                        <strong>
                                          {formatCount(runQualityAudit.summary.bucket_counts.watch)}
                                        </strong>
                                      </div>
                                      <div>
                                        <span className="meta-label">Noisy</span>
                                        <strong>
                                          {formatCount(runQualityAudit.summary.bucket_counts.noisy)}
                                        </strong>
                                      </div>
                                    </div>
                                    <div className="dataset-audit-grid">
                                      <section className="dataset-audit-card">
                                        <div className="dataset-audit-card-header">
                                          <h3>Top flags</h3>
                                          <span>Most common issues</span>
                                        </div>
                                        {runQualityAudit.flag_distribution.length > 0 ? (
                                          <div className="dataset-audit-list">
                                            {runQualityAudit.flag_distribution.slice(0, 6).map((flag) => (
                                              <div key={flag.name} className="dataset-audit-list-row">
                                                <div className="dataset-audit-list-label">
                                                  <strong>{flag.name.replace(/^flag_/, "")}</strong>
                                                  <span>{formatPercent(flag.share ?? null)}</span>
                                                </div>
                                                <div className="dataset-audit-bar-track">
                                                  <span
                                                    className="dataset-audit-bar-fill"
                                                    style={{ width: `${Math.max(4, (flag.share ?? 0) * 100)}%` }}
                                                  />
                                                </div>
                                                <span className="dataset-audit-list-value">
                                                  {flag.count.toLocaleString()}
                                                </span>
                                              </div>
                                            ))}
                                          </div>
                                        ) : (
                                          <div className="polymarket-quality-empty">
                                            No flags recorded for this run.
                                          </div>
                                        )}
                                      </section>
                                      <section className="dataset-audit-card">
                                        <div className="dataset-audit-card-header">
                                          <h3>Problem tickers</h3>
                                          <span>Highest flagged share</span>
                                        </div>
                                        {runQualityAudit.problem_tickers.length > 0 ? (
                                          <div className="dataset-audit-table">
                                            <div className="dataset-audit-table-head">
                                              <div>Ticker</div>
                                              <div>Flagged</div>
                                              <div>Avg issues</div>
                                            </div>
                                            {runQualityAudit.problem_tickers.slice(0, 5).map((tickerSummary) => (
                                              <div key={tickerSummary.ticker} className="dataset-audit-table-row">
                                                <div>
                                                  <strong>{tickerSummary.ticker}</strong>
                                                </div>
                                                <div>{formatPercent(tickerSummary.flagged_share ?? null)}</div>
                                                <div>{tickerSummary.avg_issue_count?.toFixed(2) ?? "--"}</div>
                                              </div>
                                            ))}
                                          </div>
                                        ) : (
                                          <div className="polymarket-quality-empty">
                                            No problem tickers recorded.
                                          </div>
                                        )}
                                      </section>
                                      <section className="dataset-audit-card dataset-audit-card-wide">
                                        <div className="dataset-audit-card-header">
                                          <h3>Problem markets</h3>
                                          <span>Sample flagged contracts</span>
                                        </div>
                                        {runQualityAudit.problem_markets.length > 0 ? (
                                          <div className="dataset-audit-table">
                                            <div className="dataset-audit-table-head">
                                              <div>Market</div>
                                              <div>Bucket</div>
                                              <div>Flags</div>
                                            </div>
                                            {runQualityAudit.problem_markets.slice(0, 5).map((market) => (
                                              <div key={market.market_id} className="dataset-audit-table-row">
                                                <div>
                                                  <strong>
                                                    {market.ticker} ${market.threshold ?? "--"}
                                                  </strong>
                                                  <span>{market.market_id}</span>
                                                </div>
                                                <div>{market.quality_bucket ?? "--"}</div>
                                                <div className="dataset-audit-flag-stack">
                                                  {market.active_flags.slice(0, 4).map((flag) => (
                                                    <span key={flag} className="dataset-audit-chip subtle">
                                                      {flag.replace(/^flag_/, "")}
                                                    </span>
                                                  ))}
                                                </div>
                                              </div>
                                            ))}
                                          </div>
                                        ) : (
                                          <div className="polymarket-quality-empty">
                                            No flagged markets sampled.
                                          </div>
                                        )}
                                      </section>
                                    </div>
                                  </div>
                                ) : (
                                  <div className="polymarket-quality-empty">
                                    Quality audit unavailable for this run.
                                  </div>
                                )}
                              </div>
                              {masterBarArtifacts.length > 0 ? (
                                <section className="polymarket-artifact-group polymarket-master-bars-card">
                                  <div className="polymarket-artifact-group-header">
                                    <div>
                                      <span className="meta-label">bars_history</span>
                                      <h3>Master bars</h3>
                                    </div>
                                    <span className="polymarket-artifact-group-path">run-local</span>
                                  </div>
                                  <div className="polymarket-run-files">
                                    {masterBarArtifacts.map((artifact) => {
                                      const freq = artifact.frequency as MasterBarPreviewTarget | undefined;
                                      const isPreviewingFile =
                                        runArtifactPreviewTarget?.runId === run.run_id &&
                                        runArtifactPreviewTarget.path === artifact.path;
                                      return (
                                        <div
                                          key={`${run.run_id}-${artifact.path}`}
                                          className="polymarket-run-file"
                                        >
                                          <div className="polymarket-run-file-info">
                                            <div className="polymarket-run-file-name">
                                              {artifact.path}
                                            </div>
                                            <div className="polymarket-run-file-meta">
                                              <span>{formatByteCount(artifact.size_bytes)}</span>
                                              <span>
                                                {artifact.row_count != null
                                                  ? `${artifact.row_count.toLocaleString()} rows`
                                                  : "Row count unknown"}
                                              </span>
                                              <span>
                                                {artifact.last_modified
                                                  ? formatDateTime(artifact.last_modified)
                                                  : "--"}
                                              </span>
                                            </div>
                                          </div>
                                          <div className="polymarket-run-file-actions">
                                            <button
                                              className="button light small"
                                              type="button"
                                              onClick={() =>
                                                handleToggleRunArtifactPreview({
                                                  runId: run.run_id,
                                                  path: artifact.path,
                                                  label: artifact.path,
                                                })
                                              }
                                            >
                                              {isPreviewingFile ? "Hide preview" : "Preview"}
                                            </button>
                                            {freq ? (
                                              <a
                                                className="button light small"
                                                href={getRunMasterBarFileUrl(run.run_id, freq)}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                              >
                                                Open
                                              </a>
                                            ) : null}
                                          </div>
                                        </div>
                                      );
                                    })}
                                  </div>
                                </section>
                              ) : null}
                              {artifactGroups.map((group) => (
                                <section key={`${run.run_id}-${group.key}`} className="polymarket-artifact-group">
                                  <div className="polymarket-artifact-group-header">
                                    <div>
                                      <span className="meta-label">{group.path}</span>
                                      <h3>{group.label}</h3>
                                    </div>
                                    <span className="polymarket-artifact-group-path">
                                      {group.files.length} file{group.files.length === 1 ? "" : "s"}
                                    </span>
                                  </div>
                                  <div className="polymarket-run-files">
                                    {sortRunArtifacts(group.files ?? []).map((file) => {
                                      const isPreviewingFile =
                                        runArtifactPreviewTarget?.runId === run.run_id &&
                                        runArtifactPreviewTarget.path === file.path;
                                      return (
                                        <div key={`${run.run_id}-${file.path}`} className="polymarket-run-file">
                                          <div className="polymarket-run-file-info">
                                            <div className="polymarket-run-file-name">
                                              {file.path}
                                            </div>
                                            <div className="polymarket-run-file-meta">
                                              <span>{formatByteCount(file.size_bytes)}</span>
                                              <span>
                                                {file.row_count != null
                                                  ? `${file.row_count.toLocaleString()} rows`
                                                  : "Row count unknown"}
                                              </span>
                                              <span>
                                                {file.last_modified
                                                  ? formatDateTime(file.last_modified)
                                                  : "--"}
                                              </span>
                                            </div>
                                          </div>
                                          <div className="polymarket-run-file-actions">
                                            {file.path.toLowerCase().endsWith(".csv") ? (
                                              <button
                                                className="button light small"
                                                type="button"
                                                onClick={() =>
                                                  handleToggleRunArtifactPreview({
                                                    runId: run.run_id,
                                                    path: file.path,
                                                    label: file.path,
                                                  })
                                                }
                                                title={
                                                  isPreviewingFile
                                                    ? "Hide preview"
                                                    : `Preview ${file.path}`
                                                }
                                              >
                                                {isPreviewingFile ? "Hide preview" : "Preview"}
                                              </button>
                                            ) : null}
                                            <a
                                              className="button light small"
                                              href={getPipelineRunArtifactFileUrl(run.run_id, file.path)}
                                              target="_blank"
                                              rel="noopener noreferrer"
                                              title={`Open ${file.path}`}
                                            >
                                              Open
                                            </a>
                                          </div>
                                        </div>
                                      );
                                    })}
                                  </div>
                                </section>
                              ))}
                              {isPreviewingRun ? (
                                <div className="polymarket-csv-preview-panel">
                                  <div className="polymarket-csv-preview-header">
                                    <div>
                                      <span className="meta-label">CSV preview</span>
                                      <p className="polymarket-csv-preview-title">
                                        {runArtifactPreviewTarget?.label ?? "Select a CSV"}
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
                                  {runArtifactPreviewLoading ? (
                                    <div className="polymarket-csv-preview-empty">
                                      Loading preview…
                                    </div>
                                  ) : runArtifactPreviewError ? (
                                    <div className="error">{runArtifactPreviewError}</div>
                                  ) : runArtifactPreviewResponse ? (
                                    <>
                                      {runArtifactPreviewResponse.headers.length > 0 ? (
                                        <div className="table-container polymarket-csv-preview-table">
                                          <table className="preview-table">
                                            <thead>
                                              <tr>
                                                {runArtifactPreviewResponse.headers.map((column) => (
                                                  <th key={column}>{column}</th>
                                                ))}
                                              </tr>
                                            </thead>
                                            <tbody>
                                              {runArtifactPreviewResponse.rows.length > 0 ? (
                                                runArtifactPreviewResponse.rows.map((row, index) => (
                                                  <tr key={index}>
                                                    {runArtifactPreviewResponse.headers.map((column) => (
                                                      <td key={column}>{row[column] ?? ""}</td>
                                                    ))}
                                                  </tr>
                                                ))
                                              ) : (
                                                <tr>
                                                  <td
                                                    colSpan={
                                                      runArtifactPreviewResponse.headers.length || 1
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
                                          {runArtifactPreviewResponse.mode === "tail"
                                            ? "last"
                                            : "first"}{" "}
                                          ({runArtifactPreviewResponse.limit} rows)
                                        </span>
                                        <span>
                                          {runArtifactPreviewResponse.row_count != null
                                            ? `${runArtifactPreviewResponse.row_count.toLocaleString()} total rows`
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
                Build decision features for{" "}
                <span className="polymarket-features-modal-code">
                  {featuresModalRunLabel}
                </span>
                .
              </p>
            </div>
            <div className="polymarket-features-modal-body">
              <div className="polymarket-features-modal-empty">
                The backend will load the required artifacts for this run
                automatically.
              </div>
              {featuresModalError ? (
                <div className="error">{featuresModalError}</div>
              ) : null}
            </div>
            <div className="polymarket-features-modal-actions">
              <button
                className="button ghost"
                type="button"
                onClick={handleCloseFeaturesModal}
                disabled={isFeaturesBuildRunning}
              >
                Cancel
              </button>
              <button
                className="button primary"
                type="button"
                onClick={handleConfirmBuildDecisionFeatures}
                disabled={isFeaturesBuildRunning}
              >
                {isFeaturesBuildRunning ? "Building…" : "Build decision features"}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {deleteTarget ? (
        <div
          className="dataset-delete-modal-overlay"
          onClick={() => {
            if (deleteLoading) return;
            setDeleteTarget(null);
            setDeleteConfirmText("");
          }}
        >
          <div
            className="dataset-delete-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="dataset-delete-modal-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="dataset-delete-modal-header">
              <h3 id="dataset-delete-modal-title">Delete dataset directory</h3>
              <p>
                This permanently deletes{" "}
                <span className="dataset-delete-modal-code">{deleteTarget}</span>{" "}
                and all exported files.
              </p>
            </div>
            <div className="dataset-delete-modal-body">
              <label htmlFor="datasetDeleteConfirmInput">
                Type <strong>DELETE</strong> to confirm
              </label>
              <input
                id="datasetDeleteConfirmInput"
                className="input"
                type="text"
                value={deleteConfirmText}
                onChange={(event) => setDeleteConfirmText(event.target.value)}
                placeholder="DELETE"
                autoFocus
                disabled={deleteLoading}
              />
            </div>
            <div className="dataset-delete-modal-actions">
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
                className="button ghost danger"
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
