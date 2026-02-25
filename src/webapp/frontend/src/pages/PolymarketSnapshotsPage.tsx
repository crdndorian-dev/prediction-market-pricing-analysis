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
  fetchPolymarketSnapshotRuns,
  fetchPolymarketSnapshotHistory,
  fetchPolymarketSnapshotPreview,
  startPolymarketSnapshotJob,
  deletePolymarketSnapshotRun,
  buildPolymarketSnapshotFileUrl,
  type PolymarketSnapshotFileSummary,
  type PolymarketSnapshotHistoryResponse,
  type PolymarketSnapshotPreviewResponse,
  type PolymarketSnapshotRunResponse,
  type PolymarketSnapshotRunSummary,
} from "../api/polymarketSnapshots";
import PipelineStatusCard from "../components/PipelineStatusCard";
import {
  startPolymarketHistoryJob,
  getCsvPreview,
  type PolymarketHistoryRunResponse,
  type CsvPreview,
} from "../api/polymarketHistory";
import { usePolymarketJob } from "../contexts/polymarketJob";
import { usePolymarketHistoryJob } from "../contexts/polymarketHistoryJob";
import { useAnyJobRunning } from "../contexts/jobGuard";
import "./PolymarketSnapshotsPage.css";

type RunFormState = {
  tickers: string;
  tickersCsv: string;
  slugOverrides: string;
  riskFreeRate: string;
  tz: string;
  contractType: "weekly" | "1dte";
  contract1dte: "close_today" | "close_tomorrow";
  targetDate: string;
  exchangeCalendar: string;
  allowNonlive: boolean;
  dryRun: boolean;
  keepNonexec: boolean;
};

const HARD_CODED_RISK_FREE_RATE = "0.03";
const RUN_STORAGE_KEY = "polyedgetool.polymarket.latestRun";
const HISTORY_STORAGE_KEY = "polyedgetool.polymarket.weeklyHistory.form";

const WEEKLY_TICKERS = [
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
const ONE_DTE_TICKERS = ["TSLA", "GOOGL", "NVDA", "MSFT", "AAPL", "AMZN"];

const defaultForm: RunFormState = {
  tickers: WEEKLY_TICKERS.join(", "),
  tickersCsv: "",
  slugOverrides: "",
  riskFreeRate: HARD_CODED_RISK_FREE_RATE,
  tz: "America/New_York",
  contractType: "weekly",
  contract1dte: "close_tomorrow",
  targetDate: "",
  exchangeCalendar: "XNYS",
  allowNonlive: false,
  dryRun: false,
  keepNonexec: false,
};

const STORAGE_KEY = "polyedgetool.polymarket.form";

type HistoryFormState = {
  tickers: string;
  startDate: string;
  endDate: string;
  fidelityMin: string;
  barsFreqs: string;
  outDir: string;
  barsDir: string;
  dimMarketOut: string;
  factTradeDir: string;
  includeSubgraph: boolean;
  buildFeatures: boolean;
  dryRun: boolean;
};

const defaultHistoryForm: HistoryFormState = {
  tickers: WEEKLY_TICKERS.join(", "),
  startDate: "",
  endDate: "",
  fidelityMin: "60",
  barsFreqs: "1h,1d",
  outDir: "",
  barsDir: "",
  dimMarketOut: "",
  factTradeDir: "",
  includeSubgraph: true,
  buildFeatures: false,
  dryRun: false,
};

function parseTickers(raw: string): string[] | undefined {
  const cleaned = raw
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  return cleaned.length > 0 ? cleaned : undefined;
}

function normalizeTickers(values: string[]): string[] {
  const unique = new Set<string>();
  values.forEach((value) => {
    const normalized = value.trim().toUpperCase();
    if (normalized) {
      unique.add(normalized);
    }
  });
  return Array.from(unique);
}

function formatTickerList(values: string[]): string {
  return values.join(", ");
}

function getAllowedTickers(
  contractType: "weekly" | "1dte",
): string[] {
  return contractType === "1dte" ? ONE_DTE_TICKERS : WEEKLY_TICKERS;
}

const loadStoredForm = (): Partial<RunFormState> | null => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    return parsed as Partial<RunFormState>;
  } catch {
    return null;
  }
};

const loadStoredHistoryForm = (): Partial<HistoryFormState> | null => {
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    return parsed as Partial<HistoryFormState>;
  } catch {
    return null;
  }
};

const loadStoredRun = (): PolymarketSnapshotRunResponse | null => {
  try {
    const raw = localStorage.getItem(RUN_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    if (typeof (parsed as { ok?: unknown }).ok !== "boolean") return null;
    if (!Array.isArray((parsed as { command?: unknown }).command)) return null;
    return parsed as PolymarketSnapshotRunResponse;
  } catch {
    return null;
  }
};

function formatTimestamp(value?: string | null): string {
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
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  const kb = value / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(1)} MB`;
}

function getDatePartsInTimeZone(timeZone: string, date = new Date()) {
  const formatter = new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    weekday: "short",
  });
  const parts = formatter.formatToParts(date);
  const lookup = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  const year = Number(lookup.year);
  const month = Number(lookup.month);
  const day = Number(lookup.day);
  const weekday = lookup.weekday ?? "";
  return { year, month, day, weekday };
}

function isWeekendInTimeZone(timeZone: string): boolean {
  const { weekday } = getDatePartsInTimeZone(timeZone);
  return weekday === "Sat" || weekday === "Sun";
}

function addDaysUtc(date: Date, delta: number): Date {
  const next = new Date(date);
  next.setUTCDate(next.getUTCDate() + delta);
  return next;
}

function parseIsoDate(value?: string): Date | null {
  if (!value) return null;
  const parsed = new Date(`${value}T00:00:00Z`);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function formatLongDate(value: Date, timeZone: string): string {
  return value.toLocaleDateString(undefined, {
    timeZone,
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

function getWeeklyBoundsText(timeZone: string, overrideDate?: string): string {
  const override = parseIsoDate(overrideDate);
  const localDateUtc = override
    ? override
    : (() => {
        const { year, month, day } = getDatePartsInTimeZone(timeZone);
        return new Date(Date.UTC(year, month - 1, day));
      })();
  const weekday = localDateUtc.getUTCDay(); // 0=Sun..6=Sat
  const monday = addDaysUtc(localDateUtc, weekday === 0 ? -6 : 1 - weekday);
  const friday = addDaysUtc(monday, 4);
  return `This contract covers the week starting Monday ${formatLongDate(
    monday,
    timeZone,
  )} and ending Friday ${formatLongDate(friday, timeZone)}.`;
}

function get1dteTargetText(
  timeZone: string,
  mode: "close_today" | "close_tomorrow",
  overrideDate?: string,
): string {
  const override = parseIsoDate(overrideDate);
  const localDateUtc = override
    ? override
    : (() => {
        const { year, month, day } = getDatePartsInTimeZone(timeZone);
        return new Date(Date.UTC(year, month - 1, day));
      })();
  const target = override ? localDateUtc : addDaysUtc(localDateUtc, mode === "close_today" ? 0 : 1);
  return `Target expiration date: ${formatLongDate(target, timeZone)} (market close).`;
}

const SNAPSHOT_PREVIEW_LIMIT = 18;

type SnapshotPreviewKind = "final" | "polymarket" | "rn";

const PREVIEW_KIND_LABELS: Record<SnapshotPreviewKind, string> = {
  final: "Final",
  polymarket: "Polymarket",
  rn: "RN",
};

const PREVIEW_KIND_TARGETS: Record<SnapshotPreviewKind, string[]> = {
  final: ["final.csv"],
  polymarket: ["polymarket.csv"],
  rn: ["rn.csv"],
};

function pickSnapshotFileName(files: string[]): string | null {
  if (files.length === 0) return null;
  const dataset = files.find((value) =>
    value.toLowerCase().includes("dataset"),
  );
  return dataset ?? files[0];
}

function pickSnapshotFileByKind(
  files: string[],
  kind: SnapshotPreviewKind,
  allowFallback = true,
): string | null {
  if (files.length === 0) return null;
  const targets = PREVIEW_KIND_TARGETS[kind];
  const match = files.find((value) => {
    const lowered = value.toLowerCase();
    return targets.some((target) => lowered.endsWith(target));
  });
  if (match) return match;
  if (!allowFallback) return null;
  return kind === "final" ? pickSnapshotFileName(files) : null;
}

function previewKindAvailable(files: string[], kind: SnapshotPreviewKind): boolean {
  return Boolean(pickSnapshotFileByKind(files, kind, false));
}

function buildRunFilePath(runDir: string | null, fileName: string): string {
  if (!fileName) return runDir ?? "";
  return runDir ? `${runDir}/${fileName}` : fileName;
}

function historyContractTypeFromName(
  name: string,
): "weekly" | "1dte" | null {
  const lowered = name.toLowerCase();
  if (lowered.includes("history-weekly")) return "weekly";
  if (lowered.includes("history-1dte")) return "1dte";
  return null;
}

export default function PolymarketSnapshotsPage() {
  const [formState, setFormState] = useState<RunFormState>(defaultForm);
  const [historyFormState, setHistoryFormState] = useState<HistoryFormState>(
    defaultHistoryForm,
  );
  const [runs, setRuns] = useState<PolymarketSnapshotRunSummary[]>([]);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [runResult, setRunResult] =
    useState<PolymarketSnapshotRunResponse | null>(() => loadStoredRun());
  const [activeLog, setActiveLog] = useState<"stdout" | "stderr">("stdout");
  const [historyRunResult, setHistoryRunResult] =
    useState<PolymarketHistoryRunResponse | null>(null);
  const [historyRunError, setHistoryRunError] = useState<string | null>(null);
  const [historyActiveLog, setHistoryActiveLog] = useState<"stdout" | "stderr">(
    "stdout",
  );
  const [featuresCsvPreview, setFeaturesCsvPreview] = useState<CsvPreview | null>(null);
  const [featuresCsvPreviewLoading, setFeaturesCsvPreviewLoading] = useState(false);
  const [featuresCsvPreviewError, setFeaturesCsvPreviewError] = useState<string | null>(null);
  const [history, setHistory] =
    useState<PolymarketSnapshotHistoryResponse | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [historyPreview, setHistoryPreview] =
    useState<PolymarketSnapshotPreviewResponse | null>(null);
  const [activeHistoryFile, setActiveHistoryFile] =
    useState<PolymarketSnapshotFileSummary | null>(null);
  const [storageReady, setStorageReady] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [selectedRunPreview, setSelectedRunPreview] =
    useState<PolymarketSnapshotPreviewResponse | null>(null);
  const [selectedRunPreviewError, setSelectedRunPreviewError] =
    useState<string | null>(null);
  const [selectedRunPreviewLoading, setSelectedRunPreviewLoading] =
    useState(false);
  const [runSnapshotPreview, setRunSnapshotPreview] =
    useState<PolymarketSnapshotPreviewResponse | null>(null);
  const [runSnapshotPreviewError, setRunSnapshotPreviewError] =
    useState<string | null>(null);
  const [runSnapshotPreviewLoading, setRunSnapshotPreviewLoading] =
    useState(false);
  const lastRunPreviewPathRef = useRef<string | null>(null);
  const lastSelectedPreviewPathRef = useRef<string | null>(null);
  const [deletingRunId, setDeletingRunId] = useState<string | null>(null);
  const [runDeleteError, setRunDeleteError] = useState<string | null>(null);
  const [runPreviewKind, setRunPreviewKind] =
    useState<SnapshotPreviewKind>("final");
  const [selectedRunPreviewKind, setSelectedRunPreviewKind] =
    useState<SnapshotPreviewKind>("final");
  const { jobStatus, setJobId, setJobStatus } = usePolymarketJob();
  const {
    jobStatus: historyJobStatus,
    setJobId: setHistoryJobId,
    setJobStatus: setHistoryJobStatus,
  } = usePolymarketHistoryJob();
  const { anyJobRunning, primaryJob, activeJobs } = useAnyJobRunning();

  const isRunning =
    jobStatus?.status === "queued" || jobStatus?.status === "running";

  const allowedTickers = useMemo(
    () => getAllowedTickers(formState.contractType),
    [formState.contractType],
  );

  const selectedTickers = useMemo(() => {
    const parsed = parseTickers(formState.tickers);
    if (!parsed) {
      return allowedTickers;
    }
    const normalized = normalizeTickers(parsed);
    const filtered = normalized.filter((ticker) =>
      allowedTickers.includes(ticker),
    );
    return filtered.length > 0 ? filtered : allowedTickers;
  }, [formState.tickers, allowedTickers]);

  const tickersList = useMemo(
    () => (selectedTickers.length > 0 ? selectedTickers : undefined),
    [selectedTickers],
  );

  const selectedRun = useMemo(
    () => runs.find((run) => run.run_id === selectedRunId) ?? null,
    [runs, selectedRunId],
  );

  const runResultDatasetFileName = useMemo(() => {
    if (!runResult) return null;
    return pickSnapshotFileByKind(runResult.files, runPreviewKind);
  }, [runResult, runPreviewKind]);

  const runResultDatasetPath = useMemo(() => {
    if (!runResult) return null;
    const datasetFile = pickSnapshotFileByKind(runResult.files, runPreviewKind);
    if (!datasetFile) return null;
    return buildRunFilePath(runResult.run_dir, datasetFile);
  }, [runResult, runPreviewKind]);

  const selectedRunDatasetFileName = useMemo(() => {
    if (!selectedRun) return null;
    return pickSnapshotFileByKind(selectedRun.files, selectedRunPreviewKind);
  }, [selectedRun, selectedRunPreviewKind]);

  const selectedRunDatasetPath = useMemo(() => {
    if (!selectedRun) return null;
    const datasetFile = pickSnapshotFileByKind(selectedRun.files, selectedRunPreviewKind);
    if (!datasetFile) return null;
    return buildRunFilePath(selectedRun.run_dir, datasetFile);
  }, [selectedRun, selectedRunPreviewKind]);

  const runResultPreviewAvailability = useMemo(() => {
    if (!runResult) {
      return { final: false, polymarket: false, rn: false };
    }
    return {
      final: previewKindAvailable(runResult.files, "final"),
      polymarket: previewKindAvailable(runResult.files, "polymarket"),
      rn: previewKindAvailable(runResult.files, "rn"),
    };
  }, [runResult]);

  const selectedRunPreviewAvailability = useMemo(() => {
    if (!selectedRun) {
      return { final: false, polymarket: false, rn: false };
    }
    return {
      final: previewKindAvailable(selectedRun.files, "final"),
      polymarket: previewKindAvailable(selectedRun.files, "polymarket"),
      rn: previewKindAvailable(selectedRun.files, "rn"),
    };
  }, [selectedRun]);

  const refreshRuns = useCallback(() => {
    let isMounted = true;
    fetchPolymarketSnapshotRuns()
      .then((data) => {
        if (!isMounted) return;
        setRuns(data.runs);
        setRunsError(null);
      })
      .catch((err: Error) => {
        if (!isMounted) return;
        setRunsError(err.message);
      });
    return () => {
      isMounted = false;
    };
  }, []);

  const refreshHistory = useCallback(() => {
    let isMounted = true;
    fetchPolymarketSnapshotHistory()
      .then((data) => {
        if (!isMounted) return;
        setHistory(data);
        setHistoryError(null);
      })
      .catch((err: Error) => {
        if (!isMounted) return;
        setHistoryError(err.message);
      });
    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    const stopRuns = refreshRuns();
    const stopHistory = refreshHistory();
    return () => {
      stopRuns();
      stopHistory();
    };
  }, [refreshRuns, refreshHistory]);

  useEffect(() => {
    if (runs.length === 0) {
      setSelectedRunId(null);
      return;
    }
    setSelectedRunId((prev) => {
      if (prev && runs.some((run) => run.run_id === prev)) {
        return prev;
      }
      return runs[0].run_id;
    });
  }, [runs]);

  useEffect(() => {
    const stored = loadStoredForm();
    if (stored) {
      setFormState((prev) => ({ ...prev, ...stored }));
    }
    const storedHistory = loadStoredHistoryForm();
    if (storedHistory) {
      setHistoryFormState((prev) => ({ ...prev, ...storedHistory }));
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
    if (!storageReady) return;
    try {
      localStorage.setItem(
        HISTORY_STORAGE_KEY,
        JSON.stringify(historyFormState),
      );
    } catch {
      // ignore storage failures
    }
  }, [historyFormState, storageReady]);

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
    if (!runResult.ok && runResult.stderr) {
      setActiveLog("stderr");
    } else {
      setActiveLog("stdout");
    }
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
      refreshRuns();
      refreshHistory();
    }
  }, [jobStatus, refreshRuns, refreshHistory]);

  useEffect(() => {
    if (!historyJobStatus) return;
    if (historyJobStatus.result) {
      setHistoryRunResult(historyJobStatus.result);
    }
    if (historyJobStatus.status === "failed" && historyJobStatus.error) {
      setHistoryRunError(historyJobStatus.error);
    }
    if (historyJobStatus.result?.stderr) {
      setHistoryActiveLog("stderr");
    } else {
      setHistoryActiveLog("stdout");
    }
  }, [historyJobStatus]);

  useEffect(() => {
    if (!historyRunResult || !historyRunResult.features_built || !historyRunResult.features_path) {
      setFeaturesCsvPreview(null);
      setFeaturesCsvPreviewError(null);
      setFeaturesCsvPreviewLoading(false);
      return;
    }
    if (!historyJobStatus?.job_id) return;

    const filename = historyRunResult.features_path.split('/').pop();
    if (!filename) return;

    let isMounted = true;
    setFeaturesCsvPreviewLoading(true);
    setFeaturesCsvPreviewError(null);
    getCsvPreview(historyJobStatus.job_id, filename, 20)
      .then((data) => {
        if (!isMounted) return;
        setFeaturesCsvPreview(data);
        setFeaturesCsvPreviewError(null);
      })
      .catch((err: Error) => {
        if (!isMounted) return;
        setFeaturesCsvPreview(null);
        setFeaturesCsvPreviewError(err.message);
      })
      .finally(() => {
        if (!isMounted) return;
        setFeaturesCsvPreviewLoading(false);
      });
    return () => {
      isMounted = false;
    };
  }, [historyRunResult, historyJobStatus?.job_id]);

  useEffect(() => {
    const path = runResultDatasetPath;
    if (!path) {
      setRunSnapshotPreview(null);
      setRunSnapshotPreviewError(null);
      setRunSnapshotPreviewLoading(false);
      lastRunPreviewPathRef.current = null;
      return;
    }
    if (lastRunPreviewPathRef.current === path && runSnapshotPreview) {
      return;
    }
    lastRunPreviewPathRef.current = path;
    let isMounted = true;
    setRunSnapshotPreviewLoading(true);
    setRunSnapshotPreviewError(null);
    fetchPolymarketSnapshotPreview(path, SNAPSHOT_PREVIEW_LIMIT, "head")
      .then((data) => {
        if (!isMounted) return;
        setRunSnapshotPreview(data);
        setRunSnapshotPreviewError(null);
      })
      .catch((err: Error) => {
        if (!isMounted) return;
        setRunSnapshotPreview(null);
        setRunSnapshotPreviewError(err.message);
      })
      .finally(() => {
        if (!isMounted) return;
        setRunSnapshotPreviewLoading(false);
      });
    return () => {
      isMounted = false;
    };
  }, [runResultDatasetPath, runSnapshotPreview]);

  useEffect(() => {
    if (!history || history.files.length === 0) {
      setActiveHistoryFile(null);
      return;
    }
    const preferred = history.files.find(
      (file) => historyContractTypeFromName(file.name) === formState.contractType,
    );
    setActiveHistoryFile(preferred ?? history.files[0]);
  }, [history, formState.contractType]);

  useEffect(() => {
    if (!activeHistoryFile) {
      setHistoryPreview(null);
      return;
    }
    let isMounted = true;
    fetchPolymarketSnapshotPreview(activeHistoryFile.path, 18, "tail")
      .then((data) => {
        if (!isMounted) return;
        setHistoryPreview(data);
        setHistoryError(null);
      })
      .catch((err: Error) => {
        if (!isMounted) return;
        setHistoryError(err.message);
      });
    return () => {
      isMounted = false;
    };
  }, [activeHistoryFile]);

  useEffect(() => {
    if (!selectedRun) {
      setSelectedRunPreview(null);
      setSelectedRunPreviewError(null);
      setSelectedRunPreviewLoading(false);
      lastSelectedPreviewPathRef.current = null;
      return;
    }
    if (!selectedRun.files.length || !selectedRun.run_dir) {
      setSelectedRunPreview(null);
      setSelectedRunPreviewError(null);
      setSelectedRunPreviewLoading(false);
      lastSelectedPreviewPathRef.current = null;
      return;
    }
    const datasetFile = pickSnapshotFileByKind(selectedRun.files, selectedRunPreviewKind);
    if (!datasetFile) {
      setSelectedRunPreview(null);
      setSelectedRunPreviewError(null);
      setSelectedRunPreviewLoading(false);
      lastSelectedPreviewPathRef.current = null;
      return;
    }
    const path = buildRunFilePath(selectedRun.run_dir, datasetFile);
    if (lastSelectedPreviewPathRef.current === path && selectedRunPreview) {
      return;
    }
    lastSelectedPreviewPathRef.current = path;
    let isMounted = true;
    setSelectedRunPreviewLoading(true);
    setSelectedRunPreviewError(null);
    fetchPolymarketSnapshotPreview(path, SNAPSHOT_PREVIEW_LIMIT, "head")
      .then((data) => {
        if (!isMounted) return;
        setSelectedRunPreview(data);
        setSelectedRunPreviewError(null);
      })
      .catch((err: Error) => {
        if (!isMounted) return;
        setSelectedRunPreview(null);
        setSelectedRunPreviewError(err.message);
      })
      .finally(() => {
        if (!isMounted) return;
        setSelectedRunPreviewLoading(false);
      });
    return () => {
      isMounted = false;
    };
  }, [selectedRun, selectedRunPreviewKind, selectedRunPreview]);

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
      const tickersCsv = formState.tickersCsv.trim();
      const payload = {
        tickers: tickersCsv ? undefined : tickersList,
        tickersCsv: tickersCsv || undefined,
        slugOverrides: formState.slugOverrides.trim() || undefined,
        riskFreeRate: Number(HARD_CODED_RISK_FREE_RATE),
        tz: formState.tz.trim() || undefined,
        contractType: formState.contractType,
        contract1dte: formState.contract1dte,
        targetDate: formState.targetDate.trim() || undefined,
        exchangeCalendar: formState.exchangeCalendar.trim() || undefined,
        allowNonlive: formState.allowNonlive,
        dryRun: formState.dryRun,
        keepNonexec: formState.keepNonexec,
      };
      const status = await startPolymarketSnapshotJob(payload);
      setJobId(status.job_id);
      setJobStatus(status);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setRunError(message);
    }
  };

  const handleHistorySubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (anyJobRunning) {
      setHistoryRunError(
        `Another job is running (${primaryJob?.name ?? "unknown"}). Wait for it to finish.`,
      );
      return;
    }
    setHistoryRunError(null);
    setHistoryRunResult(null);
    setHistoryJobStatus(null);
    try {
      const parsedTickers = parseTickers(historyFormState.tickers);
      const tickers = parsedTickers ? normalizeTickers(parsedTickers) : undefined;
      const fidelityRaw = historyFormState.fidelityMin.trim();
      const fidelityMin = fidelityRaw ? Number(fidelityRaw) : undefined;
      const payload = {
        tickers,
        startDate: historyFormState.startDate.trim() || undefined,
        endDate: historyFormState.endDate.trim() || undefined,
        fidelityMin:
          fidelityMin !== undefined && Number.isFinite(fidelityMin)
            ? fidelityMin
            : undefined,
        barsFreqs: historyFormState.barsFreqs.trim() || undefined,
        outDir: historyFormState.outDir.trim() || undefined,
        barsDir: historyFormState.barsDir.trim() || undefined,
        dimMarketOut: historyFormState.dimMarketOut.trim() || undefined,
        factTradeDir: historyFormState.factTradeDir.trim() || undefined,
        includeSubgraph: historyFormState.includeSubgraph,
        buildFeatures: historyFormState.buildFeatures,
        dryRun: historyFormState.dryRun,
      };
      const status = await startPolymarketHistoryJob(payload);
      setHistoryJobId(status.job_id);
      setHistoryJobStatus(status);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setHistoryRunError(message);
    }
  };

  const toggleTicker = useCallback((ticker: string) => {
    setFormState((prev) => {
      const allowed = getAllowedTickers(prev.contractType);
      const normalized = normalizeTickers(
        parseTickers(prev.tickers) ?? allowed,
      );
      const next = normalized.includes(ticker)
        ? normalized.filter((value) => value !== ticker)
        : [...normalized, ticker];
      const ordered = allowed.filter((value) => next.includes(value));
      const safeList = ordered.length > 0 ? ordered : allowed;
      return {
        ...prev,
        tickers: formatTickerList(safeList),
      };
    });
  }, []);

  const handleDeleteRun = async (runId: string) => {
    if (
      !window.confirm(
        `Delete run ${runId}? This will also drop history rows tied to the run.`,
      )
    ) {
      return;
    }
    setRunDeleteError(null);
    setDeletingRunId(runId);
    setHistoryPreview(null);
    setActiveHistoryFile(null);
    setHistory(null);
    try {
      await deletePolymarketSnapshotRun(runId);
      const [runsResponse, historyData] = await Promise.all([
        fetchPolymarketSnapshotRuns(),
        fetchPolymarketSnapshotHistory(),
      ]);
      setRuns(runsResponse.runs);
      setRunsError(null);
      setHistory(historyData);
      setHistoryError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setRunDeleteError(message);
    } finally {
      setDeletingRunId((prev) => (prev === runId ? null : prev));
    }
  };

  const resolvedTickersLabel = formState.tickersCsv.trim()
    ? "Using tickers CSV"
    : `${selectedTickers.length} selected`;

  const historyFiles = history?.files ?? [];

  const runSnapshotDownloadUrl = runResultDatasetPath
    ? buildPolymarketSnapshotFileUrl(runResultDatasetPath)
    : null;

  const selectedRunDownloadUrl = selectedRunDatasetPath
    ? buildPolymarketSnapshotFileUrl(selectedRunDatasetPath)
    : null;

  const historyDownloadUrl = historyPreview?.file.path
    ? buildPolymarketSnapshotFileUrl(historyPreview.file.path)
    : null;

  const historyStatusLabel = historyJobStatus
    ? historyJobStatus.status === "queued" || historyJobStatus.status === "running"
      ? "Running"
      : historyJobStatus.status === "finished"
        ? "Success"
        : "Failed"
    : historyRunResult
      ? "Success"
      : "Idle";

  const historyStdout = historyRunResult?.stdout ?? "";
  const historyStderr = historyRunResult?.stderr ?? historyJobStatus?.error ?? "";

  const runStatusState =
    jobStatus?.status ??
    (runResult ? (runResult.ok ? "finished" : "failed") : "idle");
  const runStatusLabel =
    runStatusState === "queued"
      ? "Queued"
      : runStatusState === "running"
        ? "Running"
        : runStatusState === "finished"
          ? "Success"
          : runStatusState === "failed"
            ? "Failed"
            : "Idle";
  const runStatusClass =
    runStatusState === "queued" || runStatusState === "running"
      ? "running"
      : runStatusState === "failed"
        ? "failed"
        : runStatusState === "finished"
          ? "success"
          : "idle";
  const runProgressLabel =
    runStatusState === "queued" || runStatusState === "running"
      ? "In progress"
      : runStatusState === "finished"
        ? "Complete"
        : runStatusState === "failed"
          ? "Failed"
          : "Idle";
  const runIdLabel =
    runResult?.run_id ?? jobStatus?.job_id ?? "Pending";
  const runOutputLabel =
    runResult?.run_dir ?? runResult?.out_dir ?? "Pending";
  const runFilesLabel = runResult
    ? `${runResult.files.length.toLocaleString()} files`
    : "N/A";
  const runDurationLabel = runResult
    ? `${runResult.duration_s.toFixed(2)}s`
    : "N/A";
  const runRowCount =
    runSnapshotPreview?.row_count ??
    (runSnapshotPreview ? runSnapshotPreview.rows.length : null);
  const runRowsLabel =
    runRowCount !== null && runRowCount !== undefined
      ? `${runRowCount.toLocaleString()} rows`
      : "N/A";
  const runLastUpdatedRaw =
    jobStatus?.finished_at ?? jobStatus?.started_at ?? null;
  const runLastUpdatedLabel = runLastUpdatedRaw
    ? formatTimestamp(runLastUpdatedRaw)
    : "N/A";
  const tickersLabel = formState.tickersCsv.trim()
    ? "From CSV"
    : `${selectedTickers.length} selected`;
  const contractLabel =
    formState.contractType === "1dte"
      ? `1DTE - ${formState.contract1dte.replace("_", " ")}`
      : "Weekly";
  const timezoneLabel = formState.tz.trim() || "Default";
  const targetDateLabel = formState.targetDate.trim() || "Auto";
  const runMonitorItems = [
    { label: "Tickers", value: tickersLabel },
    { label: "Contract", value: contractLabel },
    { label: "Target date", value: targetDateLabel },
    { label: "Timezone", value: timezoneLabel },
    { label: "Run ID", value: runIdLabel },
    { label: "Output", value: runOutputLabel },
    { label: "Files", value: runFilesLabel },
    { label: "Rows", value: runRowsLabel },
    { label: "Duration", value: runDurationLabel },
    { label: "Last update", value: runLastUpdatedLabel },
  ];
  const hasRunOutput = Boolean(runResult || jobStatus);
  const runOutputPlaceholder =
    runStatusState === "failed"
      ? jobStatus?.error ?? "Run failed. No output captured."
      : runStatusState === "finished"
        ? "Run finished but no output payload was returned."
        : "Run in progress. Snapshot preview and logs will appear once the job completes.";

  return (
    <section className="page polymarket-page">
      <header className="page-header">
        <div>
          <p className="page-kicker">Snapshot</p>
          <h1 className="page-title">Capture a fresh snapshot</h1>
          <p className="page-subtitle">
            Execute the snapshot ingestion script and store outputs under
            <code>src/data/raw/polymarket</code>.
          </p>
        </div>
        <PipelineStatusCard
          className="polymarket-meta"
          activeJobsCount={activeJobs.length}
        />
      </header>

      <div className="polymarket-grid">
        <section className="panel">
          <div className="panel-header">
            <h2>Run configuration</h2>
            <span className="panel-hint">
              Leave blank to use the script defaults.
            </span>
          </div>
          <div className="config-summary">
            <div>
              <span className="meta-label">Resolved tickers</span>
              <span>{resolvedTickersLabel}</span>
            </div>
            <div>
              <span className="meta-label">Timezone</span>
              <span>{formState.tz || "Default"}</span>
            </div>
            <div>
              <span className="meta-label">Contract type</span>
              <span>{formState.contractType}</span>
            </div>
          </div>
          <form className="panel-body" onSubmit={handleSubmit}>
            <div className="field">
              <label>Trading universe</label>
              <div className="ticker-grid">
                {allowedTickers.map((ticker) => {
                  const isSelected = selectedTickers.includes(ticker);
                  return (
                    <button
                      key={ticker}
                      type="button"
                      className={`ticker-chip ${isSelected ? "selected" : ""}`}
                      aria-pressed={isSelected}
                      onClick={() => toggleTicker(ticker)}
                    >
                      {ticker}
                    </button>
                  );
                })}
              </div>
              <span className="field-hint">
                Select tickers to include. Weekly and 1DTE snapshots have fixed
                universes. A tickers CSV (if provided) overrides this selection.
              </span>
              <div className="ticker-selection-summary">
                Selected: {selectedTickers.join(", ")}
              </div>
            </div>

            <div className="field">
              <label htmlFor="contractType">Contract type</label>
              <select
                id="contractType"
                className="input"
                value={formState.contractType}
                onChange={(event) =>
                  setFormState((prev) => {
                    const nextType = event.target.value as "weekly" | "1dte";
                    const allowed = getAllowedTickers(nextType);
                    const normalized = normalizeTickers(
                      parseTickers(prev.tickers) ?? allowed,
                    );
                    const filtered = allowed.filter((value) =>
                      normalized.includes(value),
                    );
                    return {
                      ...prev,
                      contractType: nextType,
                      tickers: formatTickerList(
                        filtered.length > 0 ? filtered : allowed,
                      ),
                    };
                  })
                }
              >
                <option value="weekly">weekly</option>
                <option value="1dte">1dte</option>
              </select>
            </div>

            {formState.contractType === "1dte" ? (
              <div className="field">
                <label htmlFor="contract1dte">1DTE expiration</label>
                <select
                  id="contract1dte"
                  className="input"
                  value={formState.contract1dte}
                  onChange={(event) =>
                    setFormState((prev) => ({
                      ...prev,
                      contract1dte: event.target.value as "close_today" | "close_tomorrow",
                    }))
                  }
                >
                  <option value="close_today">Contract closing today</option>
                  <option value="close_tomorrow">Contract closing tomorrow</option>
                </select>
                <span className="field-hint">
                  {get1dteTargetText(
                    formState.tz || "UTC",
                    formState.contract1dte,
                    formState.targetDate.trim() || undefined,
                  )}
                </span>
                {isWeekendInTimeZone(formState.tz || "UTC") ? (
                  <span className="field-hint">
                    This snapshot was taken during the weekend. No Polymarket contract will
                    be fetched unless it explicitly matches the requested expiration.
                  </span>
                ) : null}
              </div>
            ) : (
              <div className="field">
                <label>Weekly contract window</label>
                <div className="field-hint">
                  {getWeeklyBoundsText(
                    formState.tz || "UTC",
                    formState.targetDate.trim() || undefined,
                  )}
                </div>
              </div>
            )}

            <div className="field">
              <label htmlFor="tz">Timezone</label>
              <input
                id="tz"
                className="input"
                value={formState.tz}
                onChange={(event) =>
                  setFormState((prev) => ({
                    ...prev,
                    tz: event.target.value,
                  }))
                }
              />
            </div>

            <details className="advanced">
              <summary>Advanced options</summary>
              <div className="field">
                <label htmlFor="targetDate">Target date (optional)</label>
                <input
                  id="targetDate"
                  className="input"
                  type="date"
                  value={formState.targetDate}
                  onChange={(event) =>
                    setFormState((prev) => ({
                      ...prev,
                      targetDate: event.target.value,
                    }))
                  }
                />
              </div>
              <div className="field">
                <label htmlFor="exchangeCalendar">Exchange calendar</label>
                <input
                  id="exchangeCalendar"
                  className="input"
                  value={formState.exchangeCalendar}
                  onChange={(event) =>
                    setFormState((prev) => ({
                      ...prev,
                      exchangeCalendar: event.target.value,
                    }))
                  }
                />
              </div>
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={formState.allowNonlive}
                  onChange={(event) =>
                    setFormState((prev) => ({
                      ...prev,
                      allowNonlive: event.target.checked,
                    }))
                  }
                />
                Allow non-live (historical) snapshots
              </label>
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={formState.dryRun}
                  onChange={(event) =>
                    setFormState((prev) => ({
                      ...prev,
                      dryRun: event.target.checked,
                    }))
                  }
                />
                Dry run (validate only, no files written)
              </label>
            </details>

            {runError ? <div className="error">{runError}</div> : null}

            <div className="actions">
              <button
                className="button primary"
                type="submit"
                disabled={isRunning}
              >
                {isRunning ? "Running snapshot..." : "Run snapshot"}
              </button>
            </div>
          </form>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>Latest run output</h2>
            <span className="panel-hint">
              Captures stdout/stderr from the snapshot script.
            </span>
          </div>
          <div className="panel-body">
            {!hasRunOutput ? (
              <div className="empty">No run output yet.</div>
            ) : (
              <div className="run-output snapshot-run-output">
                <div className="snapshot-run-monitor">
                  <div className="snapshot-run-monitor-header">
                    <div>
                      <span className="meta-label">Run monitor</span>
                      <div className="snapshot-run-monitor-title">
                        Latest snapshot run
                      </div>
                    </div>
                    <span className={`status-pill ${runStatusClass}`}>
                      {runStatusLabel}
                    </span>
                  </div>
                  <div className="snapshot-run-monitor-grid">
                    {runMonitorItems.map((item) => (
                      <div key={item.label}>
                        <span className="meta-label">{item.label}</span>
                        <span>{item.value}</span>
                      </div>
                    ))}
                  </div>
                  <div className="snapshot-run-monitor-progress">
                    <div className="snapshot-run-monitor-progress-header">
                      <span>Progress</span>
                      <span>{runProgressLabel}</span>
                    </div>
                    <p className="snapshot-run-monitor-note">
                      Progress telemetry is not available yet. Watch stdout/stderr
                      below for live updates.
                    </p>
                  </div>
                </div>

                {runResult ? (
                  <>
                    <div className="run-generated-preview">
                      <div className="run-generated-preview-header">
                        <div>
                          <span className="meta-label">Snapshot preview</span>
                          <span className="run-preview-file-name">
                            {runSnapshotPreview?.file.name ??
                              runResultDatasetFileName ??
                              "Snapshot dataset"}
                          </span>
                          <div className="preview-file-tabs">
                            {(Object.keys(PREVIEW_KIND_LABELS) as SnapshotPreviewKind[]).map(
                              (kind) => (
                                <button
                                  key={kind}
                                  type="button"
                                  className={`preview-file-tab ${
                                    runPreviewKind === kind ? "active" : ""
                                  }`}
                                  onClick={() => setRunPreviewKind(kind)}
                                  disabled={!runResultPreviewAvailability[kind]}
                                  title={
                                    runResultPreviewAvailability[kind]
                                      ? `${PREVIEW_KIND_LABELS[kind]} CSV`
                                      : "File not available for this run"
                                  }
                                >
                                  {PREVIEW_KIND_LABELS[kind]}
                                </button>
                              ),
                            )}
                          </div>
                        </div>
                        <div className="preview-actions">
                          {runSnapshotPreviewLoading ? (
                            <span className="meta-pill">Refreshing</span>
                          ) : null}
                          {runSnapshotDownloadUrl ? (
                            <a
                              className="preview-action-link"
                              href={runSnapshotDownloadUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                            >
                              Open full dataset ↗
                            </a>
                          ) : null}
                        </div>
                      </div>
                      {runSnapshotPreview ? (
                        <>
                          <div className="table-container">
                            <table className="preview-table">
                              <thead>
                                <tr>
                                  {runSnapshotPreview.headers.map((header) => (
                                    <th key={header}>{header}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {runSnapshotPreview.rows.map((row, idx) => (
                                  <tr key={`${runSnapshotPreview.file.path}-${idx}`}>
                                    {runSnapshotPreview.headers.map((header) => (
                                      <td key={`${header}-${idx}`}>
                                        {row[header] ?? ""}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                          <div className="preview-callout">
                            <span className="preview-callout-note">
                              Showing{" "}
                              {runSnapshotPreview.mode === "tail" ? "last" : "first"}{" "}
                              {runSnapshotPreview.limit} rows of the dataset preview.
                              Use "Open full dataset" above to inspect the entire CSV.
                            </span>
                          </div>
                        </>
                      ) : runSnapshotPreviewLoading ? (
                        <div className="preview-placeholder">Loading preview…</div>
                      ) : runSnapshotPreviewError ? (
                        <div className="error">{runSnapshotPreviewError}</div>
                      ) : (
                        <div className="empty">Snapshot preview unavailable.</div>
                      )}
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
                  </>
                ) : (
                  <div className="snapshot-run-output-placeholder">
                    <div className="empty">{runOutputPlaceholder}</div>
                  </div>
                )}
              </div>
            )}
          </div>
        </section>
      </div>

      <section className="panel">
        <div className="panel-header">
          <h2>Recent snapshot runs</h2>
          <span className="panel-hint">
            Stored under <code>src/data/polymarket/runs/weekly</code> and{" "}
            <code>src/data/polymarket/runs/1dte</code>.
          </span>
        </div>
        <div className="panel-body">
          {runsError ? <div className="error">{runsError}</div> : null}
          {runs.length === 0 ? (
            <div className="empty">No runs found.</div>
          ) : (
            <>
              <div className="runs-list">
                {runs.map((run, index) => (
                  <div
                    key={run.run_id}
                    role="button"
                    tabIndex={0}
                    className={`run-card ${selectedRunId === run.run_id ? "selected" : ""} ${index === 0 ? "latest" : ""}`}
                    onClick={() => {
                      setSelectedRunId(run.run_id);
                      setRunDeleteError(null);
                    }}
                    onKeyDown={(event: KeyboardEvent<HTMLDivElement>) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        setSelectedRunId(run.run_id);
                        setRunDeleteError(null);
                      }
                    }}
                  >
                    <div>
                      <div className="run-title">{run.run_id}</div>
                      <div className="run-subtitle">
                        {formatTimestamp(run.run_time_utc ?? run.last_modified)}
                      </div>
                    </div>
                    <div className="run-stats">
                      <span>{run.file_count} files</span>
                      <span>{formatBytes(run.size_bytes)}</span>
                    </div>
                    {run.files.length > 0 ? (
                      <div className="run-files">
                        {run.files.slice(0, 4).map((file) => (
                          <span key={file}>{file}</span>
                        ))}
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
              <div className="selected-run-preview">
                <div className="selected-run-preview-header">
                  <div>
                    <span className="meta-label">Previewing run</span>
                    <div className="run-id">
                      {selectedRun?.run_id ?? "Select a run"}
                    </div>
                    <span className="selected-run-file">
                      {selectedRunPreview?.file.name ??
                        selectedRunDatasetFileName ??
                        "Snapshot dataset CSV"}
                    </span>
                    <div className="preview-file-tabs">
                      {(Object.keys(PREVIEW_KIND_LABELS) as SnapshotPreviewKind[]).map(
                        (kind) => (
                          <button
                            key={kind}
                            type="button"
                            className={`preview-file-tab ${
                              selectedRunPreviewKind === kind ? "active" : ""
                            }`}
                            onClick={() => setSelectedRunPreviewKind(kind)}
                            disabled={!selectedRunPreviewAvailability[kind]}
                            title={
                              selectedRunPreviewAvailability[kind]
                                ? `${PREVIEW_KIND_LABELS[kind]} CSV`
                                : "File not available for this run"
                            }
                          >
                            {PREVIEW_KIND_LABELS[kind]}
                          </button>
                        ),
                      )}
                    </div>
                  </div>
                  <div className="selected-run-preview-actions">
                    {selectedRunPreviewLoading ? (
                      <span className="meta-pill">Refreshing</span>
                    ) : null}
                    {selectedRunDownloadUrl ? (
                      <a
                        className="preview-action-link"
                        href={selectedRunDownloadUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        Open full dataset ↗
                      </a>
                    ) : null}
                    {selectedRun ? (
                      <button
                        type="button"
                        className="button ghost danger"
                        onClick={() => handleDeleteRun(selectedRun.run_id)}
                        disabled={deletingRunId === selectedRun.run_id}
                      >
                        {deletingRunId === selectedRun.run_id
                          ? "Deleting…"
                          : "Delete run"}
                      </button>
                    ) : null}
                  </div>
                </div>
                {runDeleteError ? (
                  <div className="error">{runDeleteError}</div>
                ) : null}
                {selectedRunPreview ? (
                  <>
                    <div className="table-container">
                      <table className="preview-table">
                        <thead>
                          <tr>
                            {selectedRunPreview.headers.map((header) => (
                              <th key={header}>{header}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {selectedRunPreview.rows.map((row, idx) => (
                            <tr
                              key={`${selectedRunPreview.file.path}-${idx}`}
                            >
                              {selectedRunPreview.headers.map((header) => (
                                <td key={`${header}-${idx}`}>
                                  {row[header] ?? ""}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div className="preview-callout">
                      <span className="preview-callout-note">
                        Showing{" "}
                        {selectedRunPreview.mode === "tail" ? "last" : "first"}{" "}
                        {selectedRunPreview.limit} rows of the selected run
                        dataset preview. Use "Open full dataset" above to open
                        the complete CSV.
                      </span>
                    </div>
                  </>
                ) : selectedRunPreviewLoading ? (
                  <div className="preview-placeholder">Loading preview…</div>
                ) : selectedRunPreviewError ? (
                  <div className="error">{selectedRunPreviewError}</div>
                ) : (
                  <div className="empty">
                    Select a run to preview its snapshot CSV.
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Snapshot history</h2>
          <span className="panel-hint">
            Weekly and 1DTE histories are stored separately under{" "}
            <code>src/data/raw/polymarket/snapshot_history</code>.
          </span>
        </div>
        <div className="panel-body">
          {historyError ? <div className="error">{historyError}</div> : null}
          {historyFiles.length === 0 ? (
            <div className="empty">No history files found yet.</div>
          ) : (
            <>
              <div className="file-tabs">
                {historyFiles.map((file) => {
                  const contractType = historyContractTypeFromName(file.name);
                  return (
                    <button
                      key={file.path}
                      type="button"
                      className={`file-tab ${
                        activeHistoryFile?.path === file.path ? "active" : ""
                      }`}
                      onClick={() => setActiveHistoryFile(file)}
                    >
                      <span className="file-label">history</span>
                      {contractType ? (
                        <span className="file-label">{contractType}</span>
                      ) : null}
                      <span className="file-name">{file.name}</span>
                    </button>
                  );
                })}
              </div>
              {historyPreview ? (
                <>
                  <div className="history-preview-heading">
                    <div>
                      <span className="meta-label">History preview</span>
                      <span className="history-preview-name">
                        {historyPreview.file.name}
                      </span>
                      {historyContractTypeFromName(historyPreview.file.name) ? (
                        <span className="meta-pill">
                          {historyContractTypeFromName(historyPreview.file.name)}
                        </span>
                      ) : null}
                    </div>
                    {historyDownloadUrl ? (
                      <a
                        className="preview-action-link"
                        href={historyDownloadUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        Open full history dataset ↗
                      </a>
                    ) : null}
                  </div>
                  <div className="table-container">
                    <table className="preview-table">
                      <thead>
                        <tr>
                          {historyPreview.headers.map((header) => (
                            <th key={header}>{header}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {historyPreview.rows.map((row, idx) => (
                          <tr key={`${historyPreview.file.path}-${idx}`}>
                            {historyPreview.headers.map((header) => (
                              <td key={`${header}-${idx}`}>
                                {row[header] ?? ""}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div className="preview-callout">
                    <span className="preview-callout-note">
                      Showing{" "}
                      {historyPreview.mode === "tail" ? "last" : "first"}{" "}
                      {historyPreview.limit} rows of the rolling history
                      preview. Use "Open full history dataset" above to view the
                      complete CSV.
                    </span>
                  </div>
                </>
              ) : (
                <div className="empty">Loading history…</div>
              )}
            </>
          )}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Weekly history backfill</h2>
          <span className="panel-hint">
            Fetch historical weekly markets, price history, and build 1h/1d bars.
          </span>
        </div>
        <form className="panel-body" onSubmit={handleHistorySubmit}>
          <div className="field">
            <label htmlFor="historyTickers">Tickers</label>
            <input
              id="historyTickers"
              className="input"
              value={historyFormState.tickers}
              onChange={(event) =>
                setHistoryFormState((prev) => ({
                  ...prev,
                  tickers: event.target.value,
                }))
              }
              placeholder={WEEKLY_TICKERS.join(", ")}
            />
            <span className="field-hint">
              Comma-separated. Leave blank to use the default weekly universe.
            </span>
          </div>

          <div className="field">
            <label htmlFor="historyStart">Start date (UTC)</label>
            <input
              id="historyStart"
              type="date"
              className="input"
              value={historyFormState.startDate}
              onChange={(event) =>
                setHistoryFormState((prev) => ({
                  ...prev,
                  startDate: event.target.value,
                }))
              }
            />
          </div>

          <div className="field">
            <label htmlFor="historyEnd">End date (UTC)</label>
            <input
              id="historyEnd"
              type="date"
              className="input"
              value={historyFormState.endDate}
              onChange={(event) =>
                setHistoryFormState((prev) => ({
                  ...prev,
                  endDate: event.target.value,
                }))
              }
            />
            <span className="field-hint">
              Leave blank to backfill as far as the API allows.
            </span>
          </div>

          <div className="field">
            <label htmlFor="historyFidelity">CLOB fidelity (minutes)</label>
            <input
              id="historyFidelity"
              type="number"
              min={1}
              className="input"
              value={historyFormState.fidelityMin}
              onChange={(event) =>
                setHistoryFormState((prev) => ({
                  ...prev,
                  fidelityMin: event.target.value,
                }))
              }
            />
          </div>

          <div className="field">
            <label htmlFor="historyBarsFreqs">Bar frequencies</label>
            <input
              id="historyBarsFreqs"
              className="input"
              value={historyFormState.barsFreqs}
              onChange={(event) =>
                setHistoryFormState((prev) => ({
                  ...prev,
                  barsFreqs: event.target.value,
                }))
              }
            />
            <span className="field-hint">
              Comma-separated (e.g. 1h,1d). Written under
              <code>src/data/analysis/polymarket/bars_history</code> by default.
            </span>
          </div>

          <div className="field">
            <label className="checkbox">
              <input
                type="checkbox"
                checked={historyFormState.includeSubgraph}
                onChange={(event) =>
                  setHistoryFormState((prev) => ({
                    ...prev,
                    includeSubgraph: event.target.checked,
                  }))
                }
              />
              Attempt subgraph trade ingest if configured
            </label>
          </div>

          <div className="field">
            <label className="checkbox">
              <input
                type="checkbox"
                checked={historyFormState.buildFeatures}
                onChange={(event) =>
                  setHistoryFormState((prev) => ({
                    ...prev,
                    buildFeatures: event.target.checked,
                  }))
                }
              />
              Build decision features after history completes
            </label>
          </div>

          <div className="field">
            <label className="checkbox">
              <input
                type="checkbox"
                checked={historyFormState.dryRun}
                onChange={(event) =>
                  setHistoryFormState((prev) => ({
                    ...prev,
                    dryRun: event.target.checked,
                  }))
                }
              />
              Dry run (no files written)
            </label>
          </div>

          <div className="form-actions">
            <button
              type="submit"
              className="button primary"
              disabled={anyJobRunning}
            >
              {historyStatusLabel === "Running"
                ? "Backfill running…"
                : "Run weekly backfill"}
            </button>
            {historyStatusLabel !== "Idle" ? (
              <span className="meta-pill">{historyStatusLabel}</span>
            ) : null}
          </div>
        </form>

        {(historyRunResult || historyJobStatus || historyRunError) && (
          <div className="panel-body">
            <div className="run-generated-preview">
              <div className="run-generated-preview-header">
                <div>
                  <span className="meta-label">Latest weekly history run</span>
                  <div className="run-id">
                    {historyRunResult?.run_id ?? historyJobStatus?.job_id ?? "Pending"}
                  </div>
                  <span className="run-preview-file-name">
                    {historyRunResult?.run_dir ?? "Output pending"}
                  </span>
                </div>
                {historyRunResult?.files?.length ? (
                  <div className="run-files">
                    {historyRunResult.files.slice(0, 5).map((file) => (
                      <span key={file}>{file}</span>
                    ))}
                  </div>
                ) : null}
              </div>

              {historyRunError ? (
                <div className="error">{historyRunError}</div>
              ) : null}

              {historyRunResult ? (
                <>
                  <div className="log-tabs">
                    <button
                      className={`log-tab ${
                        historyActiveLog === "stdout" ? "active" : ""
                      }`}
                      type="button"
                      onClick={() => setHistoryActiveLog("stdout")}
                    >
                      stdout
                    </button>
                    <button
                      className={`log-tab ${
                        historyActiveLog === "stderr" ? "active" : ""
                      }`}
                      type="button"
                      onClick={() => setHistoryActiveLog("stderr")}
                    >
                      stderr
                    </button>
                  </div>
                  <div className="log-block">
                    <span className="meta-label">
                      {historyActiveLog === "stdout" ? "stdout" : "stderr"}
                    </span>
                    <pre>
                      {historyActiveLog === "stdout"
                        ? historyStdout || "No stdout captured."
                        : historyStderr || "No stderr captured."}
                    </pre>
                  </div>
                  <details className="command-details">
                    <summary>Command used</summary>
                    <code>{historyRunResult.command.join(" ")}</code>
                  </details>

                  {historyRunResult.features_built && (
                    <div className="run-generated-preview">
                      <div className="run-generated-preview-header">
                        <div>
                          <span className="meta-label">Output CSV files</span>
                          <span className="run-preview-file-name">
                            Build features output
                          </span>
                        </div>
                      </div>

                      <div className="run-files">
                        {historyRunResult.features_path && historyRunResult.run_dir && (
                          <a
                            href={buildPolymarketSnapshotFileUrl(
                              `${historyRunResult.run_dir}/${historyRunResult.features_path.split('/').pop()}`
                            )}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="file-download-link"
                          >
                            {historyRunResult.features_path.split('/').pop() || 'decision_features.csv'} ↗
                          </a>
                        )}
                        {historyRunResult.features_manifest_path && historyRunResult.run_dir && (
                          <a
                            href={buildPolymarketSnapshotFileUrl(
                              `${historyRunResult.run_dir}/${historyRunResult.features_manifest_path.split('/').pop()}`
                            )}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="file-download-link"
                          >
                            {historyRunResult.features_manifest_path.split('/').pop() || 'feature_manifest.json'} ↗
                          </a>
                        )}
                      </div>

                      {featuresCsvPreview && (
                        <>
                          <div className="run-generated-preview-header">
                            <div>
                              <span className="meta-label">Features preview</span>
                              <span className="run-preview-file-name">
                                {featuresCsvPreview.filename}
                              </span>
                            </div>
                          </div>
                          <div className="table-container">
                            <table className="preview-table">
                              <thead>
                                <tr>
                                  {featuresCsvPreview.headers.map((header) => (
                                    <th key={header}>{header}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {featuresCsvPreview.rows.map((row, idx) => (
                                  <tr key={`${featuresCsvPreview.filename}-${idx}`}>
                                    {featuresCsvPreview.headers.map((header) => (
                                      <td key={`${header}-${idx}`}>
                                        {row[header] ?? ""}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                          <div className="preview-callout">
                            <span className="preview-callout-note">
                              Showing first {featuresCsvPreview.rows.length} rows
                              {featuresCsvPreview.truncated && ` of ${featuresCsvPreview.total_rows} total rows`}.
                            </span>
                          </div>
                        </>
                      )}

                      {featuresCsvPreviewLoading && (
                        <div className="preview-placeholder">Loading features preview…</div>
                      )}

                      {featuresCsvPreviewError && (
                        <div className="error">{featuresCsvPreviewError}</div>
                      )}
                    </div>
                  )}
                </>
              ) : (
                <div className="empty">Job is running…</div>
              )}
            </div>
          </div>
        )}
      </section>
    </section>
  );
}
