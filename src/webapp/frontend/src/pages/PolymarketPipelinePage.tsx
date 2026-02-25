import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";

import {
  startPolymarketHistoryJob,
  cancelPolymarketHistoryJob,
  listPipelineRuns,
  renamePipelineRun,
  setActiveRun,
  deletePipelineRun,
  togglePinRun,
  type PipelineProgress,
  type PipelineRunSummary,
  type StorageSummary,
} from "../api/polymarketHistory";
import { listDatasetRuns, type DatasetRunSummary } from "../api/datasets";
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
  strict: false,
  useWeeklyHistory: true,
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
    return { ...merged, tickers: sanitizeTickers(merged.tickers) };
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

export default function PolymarketPipelinePage() {
  const [form, setForm] = useState<FormState>(() => loadStoredForm() ?? defaultForm);
  const [formError, setFormError] = useState<string | null>(null);
  const [stopLoading, setStopLoading] = useState(false);
  const [datasetRuns, setDatasetRuns] = useState<DatasetRunSummary[]>([]);
  const [datasetRunsError, setDatasetRunsError] = useState<string | null>(null);
  const [isDatasetRunsLoading, setIsDatasetRunsLoading] = useState(false);

  // --- Runs browser state ---
  const [pipelineRuns, setPipelineRuns] = useState<PipelineRunSummary[]>([]);
  const [runsStorage, setRunsStorage] = useState<StorageSummary | null>(null);
  const [runsLoading, setRunsLoading] = useState(false);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [editingLabel, setEditingLabel] = useState<Record<string, string>>({});
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [deleteConfirmText, setDeleteConfirmText] = useState("");
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [runsOpen, setRunsOpen] = useState(true);
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
    () => datasetRuns.filter((run) => run.run_dir.startsWith("src/data/raw/option-chain/")),
    [datasetRuns],
  );
  const selectedDatasetRun = useMemo(
    () => optionChainRuns.find((run) => run.run_dir === form.historyPrnDataset) ?? null,
    [optionChainRuns, form.historyPrnDataset],
  );
  const resolvedHistoryPrnDataset = useMemo(() => {
    if (!selectedDatasetRun) return null;
    return selectedDatasetRun.training_file?.path ?? null;
  }, [selectedDatasetRun]);

  useEffect(() => {
    let cancelled = false;
    setIsDatasetRunsLoading(true);
    setDatasetRunsError(null);
    listDatasetRuns()
      .then((payload) => {
        if (cancelled) return;
        setDatasetRuns(payload.runs ?? []);
      })
      .catch((err) => {
        if (cancelled) return;
        setDatasetRuns([]);
        setDatasetRunsError(
          err instanceof Error ? err.message : "Failed to load dataset directories",
        );
      })
      .finally(() => {
        if (!cancelled) setIsDatasetRunsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

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

  const handleSetActive = useCallback(async (runId: string) => {
    try {
      await setActiveRun(runId);
      loadRuns();
    } catch (err) {
      console.error("Set active failed:", err);
    }
  }, [loadRuns]);

  const handleRenameBlur = useCallback(async (runId: string) => {
    const label = editingLabel[runId];
    if (label === undefined) return;
    try {
      await renamePipelineRun(runId, label);
      loadRuns();
    } catch (err) {
      console.error("Rename failed:", err);
    }
    setEditingLabel((prev) => {
      const next = { ...prev };
      delete next[runId];
      return next;
    });
  }, [editingLabel, loadRuns]);

  const handleTogglePin = useCallback(async (runId: string) => {
    try {
      await togglePinRun(runId);
      loadRuns();
    } catch (err) {
      console.error("Pin toggle failed:", err);
    }
  }, [loadRuns]);

  const handleDeleteConfirm = useCallback(async () => {
    if (!deleteTarget || deleteConfirmText !== "DELETE") return;
    setDeleteLoading(true);
    try {
      await deletePipelineRun(deleteTarget);
      setDeleteTarget(null);
      setDeleteConfirmText("");
      loadRuns();
    } catch (err) {
      console.error("Delete failed:", err);
    } finally {
      setDeleteLoading(false);
    }
  }, [deleteTarget, deleteConfirmText, loadRuns]);

  const formatSize = (bytes: number) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const usingWeeklyHistory = form.useWeeklyHistory;
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
          includeSubgraph: form.historyIncludeSubgraph,
          buildFeatures: form.historyBuildFeatures,
          prnDataset: resolvedHistoryPrnDataset || undefined,
          skipSubgraphLabels: false,
        };
        const status = await startPolymarketHistoryJob(payload);
        setHistoryJobId(status.job_id);
      } else {
        const status = await startMarketMapJob({
          runDir: undefined,
          overrides: form.overrides || undefined,
          tickers: formatTickerList(selectedTickers),
          prnDataset: undefined,
          out: undefined,
          strict: form.strict,
        });
        setMapJobId(status.job_id);
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

  const activeLog = usingWeeklyHistory
    ? historyStderr || historyError
      ? "stderr"
      : "stdout"
    : mapStderr || mapError
      ? "stderr"
      : "stdout";

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
    { label: "Strict", value: form.strict ? "Yes" : "No" },
    { label: "Overrides", value: form.overrides || "--" },
    { label: "Run ID", value: mapRunId },
    { label: "Output file", value: mapOutputLabel },
    { label: "Rows", value: mapRowsLabel },
    { label: "Duration", value: mapDurationLabel },
    { label: "Last update", value: mapLastUpdatedLabel },
  ];

  return (
    <section className="page polymarket-pipeline-page">
      <header className="page-header">
        <div>
          <p className="page-kicker">Polymarket</p>
          <h1 className="page-title">Fetch markets and build dim_market</h1>
          <p className="page-subtitle">
            Run the pipeline against live markets or switch to weekly history backfill for closed events.
          </p>
        </div>
        <PipelineStatusCard activeJobsCount={activeJobs.length} />
      </header>

      <div className="pipeline-grid">
        <div className="pipeline-main">
          <section className="panel">
            <div className="panel-header">
              <h2>Run Configuration</h2>
              <span className="panel-hint">Configure and run the backfill</span>
            </div>
            <form className="panel-body" onSubmit={handleRunPipeline}>
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={form.useWeeklyHistory}
                  onChange={(event) =>
                    setForm((prev) => ({ ...prev, useWeeklyHistory: event.target.checked }))
                  }
                />
                Use weekly history backfill (closed markets + CLOB history)
              </label>

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
                    Pick from the core trading universe. Only these tickers are allowed.
                  </span>
                  <div className="field-hint">
                    Selected: {selectedTickers.join(", ")}
                  </div>
                </div>
              </div>
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={form.strict}
                  disabled={usingWeeklyHistory}
                  onChange={(event) =>
                    setForm((prev) => ({ ...prev, strict: event.target.checked }))
                  }
                />
                Strict mapping enforcement (fail if any market lacks ticker/threshold)
              </label>

              {usingWeeklyHistory ? (
                <div className="fields-grid">
                  <div className="field">
                    <label>History start date (UTC)</label>
                    <input
                      className="input"
                      type="date"
                      value={form.historyStartDate}
                      onChange={(event) =>
                        setForm((prev) => ({ ...prev, historyStartDate: event.target.value }))
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
                        setForm((prev) => ({ ...prev, historyEndDate: event.target.value }))
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
                        setForm((prev) => ({ ...prev, historyFidelityMin: event.target.value }))
                      }
                    />
                  </div>
                  <div className="field">
                    <label>Bar frequencies</label>
                    <input
                      className="input"
                      value={form.historyBarsFreqs}
                      onChange={(event) =>
                        setForm((prev) => ({ ...prev, historyBarsFreqs: event.target.value }))
                      }
                    />
                    <span className="field-hint">Comma-separated (e.g. 1h,1d).</span>
                  </div>
                  <div className="field full">
                    <div className="help-note">
                      <strong>Event URLs:</strong>{" "}
                      Auto-generated using pattern{" "}
                      <span className="mono-inline">
                        https://polymarket.com/event/&#123;ticker&#125;-above-on-&#123;month&#125;-&#123;friday_day&#125;-&#123;year&#125;
                      </span>
                      <br />
                      {autoEventState?.error
                        ? autoEventState.error
                        : `Generating ${formatCount(autoEventState.urls.length)} URLs across ${formatCount(
                            autoEventState.fridays.length,
                          )} Fridays for ${
                            selectedTickers.length || TRADING_UNIVERSE_TICKERS.length
                          } tickers.`}
                    </div>
                  </div>
                  <div className="field">
                    <label className="checkbox">
                      <input
                        type="checkbox"
                        checked={form.historyIncludeSubgraph}
                        onChange={(event) =>
                          setForm((prev) => ({ ...prev, historyIncludeSubgraph: event.target.checked }))
                        }
                      />
                      Attempt subgraph trade ingest if configured
                    </label>
                  </div>
                  <div className="field">
                    <label className="checkbox">
                      <input
                        type="checkbox"
                        checked={form.historyBuildFeatures}
                        onChange={(event) =>
                          setForm((prev) => ({ ...prev, historyBuildFeatures: event.target.checked }))
                        }
                      />
                      Build decision features for model calibration
                    </label>
                  </div>
                  {form.historyBuildFeatures ? (
                    <div className="field full">
                      <label>pRN dataset directory</label>
                      <select
                        className="input"
                        value={form.historyPrnDataset}
                        onChange={(event) =>
                          setForm((prev) => ({ ...prev, historyPrnDataset: event.target.value }))
                        }
                        disabled={isDatasetRunsLoading}
                      >
                        <option value="">
                          {isDatasetRunsLoading
                            ? "Loading dataset directories..."
                            : "Select a dataset directory"}
                        </option>
                        {optionChainRuns.map((run) => (
                          <option key={run.run_dir} value={run.run_dir}>
                            {run.run_dir}
                          </option>
                        ))}
                      </select>
                      <span className="field-hint">
                        {datasetRunsError
                          ? datasetRunsError
                          : optionChainRuns.length
                            ? "Choose a directory under src/data/raw/option-chain."
                            : "No option-chain dataset directories found."}
                      </span>
                      <span className="field-hint">
                        Resolved pRN CSV (training-*.csv):{" "}
                        <span className="mono-inline">{resolvedHistoryPrnDataset ?? "--"}</span>
                      </span>
                    </div>
                  ) : null}
                </div>
              ) : null}

              {usingWeeklyHistory && !hasEventSources ? (
                <div className="help-note">
                  <strong>Heads up:</strong> weekly finance markets are not discoverable by
                  pagination. Provide a date range to auto-generate event URLs.
                </div>
              ) : null}

              {formError ? (
                <div className="error-banner">
                  <strong>Input Error:</strong>
                  <pre>{formError}</pre>
                </div>
              ) : null}

              <div className="panel-actions">
                <button
                  className="button primary large"
                  type="submit"
                  disabled={isRunning || anyJobRunning}
                >
                  {isRunning
                    ? "Running pipeline..."
                    : usingWeeklyHistory
                      ? "Run Weekly Backfill"
                      : "Run Full Pipeline"}
                </button>
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
              </div>

              {(usingWeeklyHistory ? historyError || historyStderr : mapError || mapStderr) ? (
                <div className="error-banner">
                  <strong>Pipeline Error:</strong>
                  <pre>{usingWeeklyHistory ? historyError || historyStderr : mapError || mapStderr}</pre>
                </div>
              ) : null}
            </form>
          </section>
        </div>

        <div className="pipeline-sidebar">
          <section className="panel">
            <div className="panel-header">
              <h2>Latest Run Output</h2>
              <span className="panel-hint">Real-time job status</span>
            </div>
            <div className="panel-body">
              {usingWeeklyHistory ? (
                historyJobStatus ? (
                  <div className="run-output pipeline-run-output">
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
                          Progress updates only when each market finishes, so counts never move backward.
                        </p>
                      </div>
                    </div>
                    {historyStdout || historyStderr || historyError ? (
                      <div className="log-block">
                        <span className="meta-label">
                          {activeLog === "stdout" ? "Output" : "Errors"}
                        </span>
                        <pre className="log-content">
                          {activeLog === "stdout"
                            ? historyStdout || "No output captured."
                            : historyStderr || historyError || "No errors."}
                        </pre>
                      </div>
                    ) : null}
                  </div>
                ) : (
                  <div className="empty">Run the backfill to see output</div>
                )
              ) : mapJobStatus ? (
                <div className="run-output pipeline-run-output">
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
                        Live progress telemetry is not available for market map runs yet.
                      </p>
                    </div>
                  </div>

                  {mapStdout || mapStderr || mapError ? (
                    <div className="log-block">
                      <span className="meta-label">
                        {activeLog === "stdout" ? "Output" : "Errors"}
                      </span>
                      <pre className="log-content">
                        {activeLog === "stdout"
                          ? mapStdout || "No output captured."
                          : mapStderr || mapError || "No errors."}
                      </pre>
                    </div>
                  ) : null}
                </div>
              ) : (
                <div className="empty">Run the pipeline to see output</div>
              )}
            </div>
          </section>
        </div>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Past Runs Browser (Phase 2)                                        */}
      {/* ------------------------------------------------------------------ */}
      <section className="runs-browser-section">
        <div className="panel">
          <div className="panel-header runs-browser-header">
            <div>
              <h2
                style={{ cursor: "pointer", userSelect: "none" }}
                onClick={() => setRunsOpen((prev) => !prev)}
              >
                {runsOpen ? "Past Runs" : "Past Runs (collapsed)"}
              </h2>
              <span className="panel-hint">
                {runsLoading
                  ? "Loading..."
                  : `${pipelineRuns.length} run${pipelineRuns.length !== 1 ? "s" : ""}`}
              </span>
            </div>
            {runsStorage && runsOpen && (
              <div className="runs-storage-badge">
                <span>{runsStorage.total_runs} runs</span>
                <span>{runsStorage.total_size_mb.toFixed(1)} MB</span>
              </div>
            )}
          </div>

          {runsOpen && (
            <div className="panel-body">
              {runsError && (
                <div className="error-banner">
                  <strong>Failed to load runs:</strong>
                  <pre>{runsError}</pre>
                </div>
              )}

              {pipelineRuns.length === 0 && !runsLoading && !runsError ? (
                <div className="runs-empty">
                  No pipeline runs found. Run the weekly backfill to create your first run.
                </div>
              ) : pipelineRuns.length > 0 ? (
                <div className="runs-table-wrapper">
                  <table className="runs-table">
                    <thead>
                      <tr>
                        <th>Status</th>
                        <th>Label / Run ID</th>
                        <th>Date Range</th>
                        <th>Markets</th>
                        <th>Artifacts</th>
                        <th>Size</th>
                        <th>Created</th>
                        <th>Duration</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {pipelineRuns.map((run) => {
                        const statusClass =
                          run.status === "success" ? "success"
                          : run.status === "failed" ? "failed"
                          : run.status === "cancelled" ? "cancelled"
                          : "unknown";
                        return (
                          <tr key={run.run_id} className={run.is_active ? "is-active" : ""}>
                            <td>
                              <span className={`run-status-dot ${statusClass}`} title={run.status} />
                            </td>
                            <td>
                              <div className="run-label-cell">
                                {run.is_active && <span className="run-active-badge">Active</span>}
                                {run.pinned && <span className="run-pinned-badge" title="Pinned">*</span>}
                                <input
                                  className="run-label-input"
                                  type="text"
                                  placeholder={run.run_id}
                                  value={
                                    editingLabel[run.run_id] !== undefined
                                      ? editingLabel[run.run_id]
                                      : run.label ?? ""
                                  }
                                  onChange={(e) =>
                                    setEditingLabel((prev) => ({ ...prev, [run.run_id]: e.target.value }))
                                  }
                                  onBlur={() => handleRenameBlur(run.run_id)}
                                  onKeyDown={(e) => {
                                    if (e.key === "Enter") (e.target as HTMLInputElement).blur();
                                  }}
                                />
                              </div>
                              <div className="run-id-mono">{run.run_id}</div>
                            </td>
                            <td>
                              {run.start_date && run.end_date
                                ? `${run.start_date} - ${run.end_date}`
                                : "--"}
                            </td>
                            <td>{run.markets != null ? run.markets.toLocaleString() : "--"}</td>
                            <td>
                              {run.artifact_count}
                              {run.features_built && (
                                <span style={{ fontSize: "0.6875rem", color: "var(--ink-500)", marginLeft: 4 }}>
                                  +feat
                                </span>
                              )}
                            </td>
                            <td className="run-size-text">{formatSize(run.size_bytes)}</td>
                            <td style={{ fontSize: "0.8125rem" }}>
                              {run.created_at_utc
                                ? formatDateTime(run.created_at_utc)
                                : "--"}
                            </td>
                            <td style={{ fontSize: "0.8125rem" }}>
                              {run.duration_s != null ? `${run.duration_s}s` : "--"}
                            </td>
                            <td>
                              <div className="run-actions">
                                {!run.is_active && (
                                  <button
                                    className="run-action-btn"
                                    onClick={() => handleSetActive(run.run_id)}
                                    title="Set as active run"
                                  >
                                    Activate
                                  </button>
                                )}
                                <button
                                  className="run-action-btn"
                                  onClick={() => handleTogglePin(run.run_id)}
                                  title={run.pinned ? "Unpin" : "Pin"}
                                >
                                  {run.pinned ? "Unpin" : "Pin"}
                                </button>
                                {!run.is_active && (
                                  <button
                                    className="run-action-btn danger"
                                    onClick={() => {
                                      setDeleteTarget(run.run_id);
                                      setDeleteConfirmText("");
                                    }}
                                    title="Delete run"
                                  >
                                    Delete
                                  </button>
                                )}
                              </div>
                              {run.error_summary && (
                                <div
                                  style={{
                                    fontSize: "0.6875rem",
                                    color: "var(--color-error, #c62828)",
                                    marginTop: 4,
                                    maxWidth: 200,
                                    overflow: "hidden",
                                    textOverflow: "ellipsis",
                                    whiteSpace: "nowrap",
                                  }}
                                  title={run.error_summary}
                                >
                                  {run.error_summary}
                                </div>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : null}
            </div>
          )}
        </div>
      </section>

      {/* Delete confirmation modal */}
      {deleteTarget && (
        <div className="delete-modal-overlay" onClick={() => setDeleteTarget(null)}>
          <div className="delete-modal" onClick={(e) => e.stopPropagation()}>
            <h3>Delete pipeline run</h3>
            <p>
              This will permanently delete{" "}
              <span className="mono-inline">{deleteTarget}</span>{" "}
              and all its artifacts. This cannot be undone.
            </p>
            <label style={{ fontSize: "0.875rem", display: "block", marginBottom: "0.5rem" }}>
              Type <strong>DELETE</strong> to confirm:
            </label>
            <input
              className="input"
              type="text"
              value={deleteConfirmText}
              onChange={(e) => setDeleteConfirmText(e.target.value)}
              placeholder="DELETE"
              autoFocus
              style={{ width: "100%" }}
            />
            <div className="delete-modal-actions">
              <button
                className="button ghost"
                onClick={() => {
                  setDeleteTarget(null);
                  setDeleteConfirmText("");
                }}
              >
                Cancel
              </button>
              <button
                className="button primary"
                disabled={deleteConfirmText !== "DELETE" || deleteLoading}
                onClick={handleDeleteConfirm}
                style={{ background: "var(--color-error, #c62828)" }}
              >
                {deleteLoading ? "Deleting..." : "Delete permanently"}
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
