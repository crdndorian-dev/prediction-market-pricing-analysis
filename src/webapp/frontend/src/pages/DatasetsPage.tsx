import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type FormEvent,
} from "react";

import {
  deleteDatasetRun,
  getDatasetFileUrl,
  killDatasetJob,
  listDatasetRuns,
  previewDatasetFile,
  renameDatasetRun,
  startDatasetJob,
  type DatasetJobStatus,
  type DatasetFileSummary,
  type DatasetPreviewResponse,
  type DatasetRunResponse,
  type DatasetRunSummary,
} from "../api/datasets";
import { useDatasetJob } from "../contexts/datasetJob";
import { useAnyJobRunning } from "../contexts/jobGuard";
import "./DatasetsPage.css";

type DatasetFormState = {
  outDir: string;
  datasetName: string;
  scheduleMode: "weekly" | "expiry_range";
  expiryWeekdays: string;
  asofWeekdays: string;
  dteList: string;
  dteMin: string;
  dteMax: string;
  dteStep: string;
  writeSnapshot: boolean;
  writePrnView: boolean;
  writeTrainView: boolean;
  writeLegacy: boolean;
  prnVersion: string;
  prnConfigHash: string;
  tickers: string;
  start: string;
  end: string;
  thetaBaseUrl: string;
  stockSource: "yfinance" | "theta" | "auto";
  timeoutS: string;
  riskFreeRate: string;
  maxAbsLogm: string;
  maxAbsLogmCap: string;
  bandWidenStep: string;
  adaptiveBand: boolean;
  maxBandStrikes: string;
  minBandStrikes: string;
  minBandPrnStrikes: string;
  strikeRange: string;
  retryFullChain: boolean;
  saturdayExpiryFallback: boolean;
  threads: string;
  preferBidask: boolean;
  minTradeCount: string;
  minVolume: string;
  minChainUsedHard: string;
  maxRelSpreadMedianHard: string;
  hardDropCloseFallback: boolean;
  minPrnTrain: string;
  maxPrnTrain: string;
  splitAdjust: boolean;
  dividendSource: "yfinance" | "none";
  dividendLookbackDays: string;
  dividendYieldDefault: string;
  forwardMoneyness: boolean;
  groupWeights: boolean;
  tickerWeights: boolean;
  softQualityWeight: boolean;
  rvLookbackDays: string;
  cache: boolean;
  writeDrops: boolean;
  sanityReport: boolean;
  sanityDrop: boolean;
  sanityAbsLogmMax: string;
  sanityKOverSMin: string;
  sanityKOverSMax: string;
  verboseSkips: boolean;
};

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

const defaultForm: DatasetFormState = {
  outDir: "src/data/raw/option-chain",
  datasetName: "",
  scheduleMode: "weekly",
  expiryWeekdays: "fri",
  asofWeekdays: "mon,tue,wed,thu",
  dteList: "",
  dteMin: "",
  dteMax: "",
  dteStep: "1",
  writeSnapshot: true,
  writePrnView: true,
  writeTrainView: true,
  writeLegacy: true,
  prnVersion: "v1",
  prnConfigHash: "",
  tickers: TRADING_UNIVERSE_TICKERS.join(", "),
  start: "",
  end: "",
  thetaBaseUrl: "http://127.0.0.1:25503/v3",
  stockSource: "yfinance",
  timeoutS: "30",
  riskFreeRate: "0.03",
  maxAbsLogm: "0.06",
  maxAbsLogmCap: "0.10",
  bandWidenStep: "0.01",
  adaptiveBand: true,
  maxBandStrikes: "0",
  minBandStrikes: "10",
  minBandPrnStrikes: "7",
  strikeRange: "60",
  retryFullChain: true,
  saturdayExpiryFallback: true,
  threads: "6",
  preferBidask: true,
  minTradeCount: "0",
  minVolume: "0",
  minChainUsedHard: "0",
  maxRelSpreadMedianHard: "1000000000",
  hardDropCloseFallback: false,
  minPrnTrain: "0.10",
  maxPrnTrain: "0.90",
  splitAdjust: true,
  dividendSource: "yfinance",
  dividendLookbackDays: "365",
  dividendYieldDefault: "0.0",
  forwardMoneyness: true,
  groupWeights: true,
  tickerWeights: true,
  softQualityWeight: true,
  rvLookbackDays: "20",
  cache: true,
  writeDrops: false,
  sanityReport: false,
  sanityDrop: false,
  sanityAbsLogmMax: "0.40",
  sanityKOverSMin: "0.25",
  sanityKOverSMax: "4.0",
  verboseSkips: false,
};

const toKebabCase = (value: string) =>
  value
    .trim()
    .toLowerCase()
    .replace(/[\s_]+/g, "-")
    .replace(/[^a-z0-9-]/g, "")
    .replace(/-{2,}/g, "-")
    .replace(/^-|-$/g, "");

const STORAGE_KEY = "polyedgetool.datasets.form";
const CALIBRATE_STORAGE_KEY = "polyedgetool.calibrate.form";

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

const loadStoredForm = (): Partial<DatasetFormState> | null => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    // Migration: convert old fields to datasetName
    if (!parsed.datasetName && parsed.runDirName) {
      parsed.datasetName = parsed.runDirName;
    }
    parsed.writeTrainView = true;
    delete parsed.outName;
    delete parsed.runDirName;
    delete parsed.trainViewName;
    delete parsed.trainingDataset;
    return parsed as Partial<DatasetFormState>;
  } catch {
    return null;
  }
};

const parseTickers = (raw: string): string[] | undefined => {
  const cleaned = raw
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  return cleaned.length > 0 ? cleaned : undefined;
};

const normalizeTickers = (values: string[]): string[] => {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values) {
    const normalized = value.trim().toUpperCase();
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
};

const orderTickers = (values: string[]): string[] => {
  const universeSet = new Set(TRADING_UNIVERSE_TICKERS);
  const universeOrdered = TRADING_UNIVERSE_TICKERS.filter((ticker) =>
    values.includes(ticker),
  );
  const extras = values.filter((ticker) => !universeSet.has(ticker));
  return [...universeOrdered, ...extras];
};

const formatTickerList = (values: string[]): string => values.join(", ");

const splitTickerInput = (raw: string): string[] => {
  return raw
    .split(/[\s,]+/)
    .map((value) => value.trim())
    .filter(Boolean);
};

const WEEKDAY_MAP: Record<string, number> = {
  mon: 1,
  monday: 1,
  tue: 2,
  tues: 2,
  tuesday: 2,
  wed: 3,
  weds: 3,
  wednesday: 3,
  thu: 4,
  thur: 4,
  thurs: 4,
  thursday: 4,
  fri: 5,
  friday: 5,
  sat: 6,
  saturday: 6,
  sun: 0,
  sunday: 0,
};

const parseWeekdays = (raw: string): number[] | null => {
  const cleaned = raw
    .split(",")
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean);
  if (cleaned.length === 0) return null;
  const out: number[] = [];
  const seen = new Set<number>();
  for (const item of cleaned) {
    let day: number | undefined;
    if (/^\d+$/.test(item)) {
      const scriptDay = Number.parseInt(item, 10);
      day = Number.isFinite(scriptDay) ? (scriptDay + 1) % 7 : undefined;
    } else {
      day = WEEKDAY_MAP[item];
    }
    if (day === undefined || Number.isNaN(day)) return null;
    if (!seen.has(day)) {
      seen.add(day);
      out.push(day);
    }
  }
  return out;
};

const countWeekdaysInRange = (
  start: string,
  end: string,
  weekdays: number[] | null,
): number | null => {
  if (!start || !end || !weekdays || weekdays.length === 0) return null;
  const startDate = new Date(`${start}T00:00:00Z`);
  const endDate = new Date(`${end}T00:00:00Z`);
  if (Number.isNaN(startDate.getTime()) || Number.isNaN(endDate.getTime())) {
    return null;
  }
  if (endDate < startDate) return null;
  const daySet = new Set(weekdays);
  let count = 0;
  const cursor = new Date(startDate);
  while (cursor <= endDate) {
    if (daySet.has(cursor.getUTCDay())) count += 1;
    cursor.setUTCDate(cursor.getUTCDate() + 1);
  }
  return count;
};

const resolveDteCount = (
  listRaw: string,
  minRaw: string,
  maxRaw: string,
  stepRaw: string,
): number | null => {
  const parts = listRaw
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  if (parts.length > 0) {
    const values = new Set<number>();
    for (const part of parts) {
      if (part.includes("-")) {
        const [startStr, endStr] = part.split("-", 2);
        const start = Number.parseInt(startStr, 10);
        const end = Number.parseInt(endStr, 10);
        if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
        const lo = Math.min(start, end);
        const hi = Math.max(start, end);
        for (let v = lo; v <= hi; v += 1) values.add(v);
      } else {
        const v = Number.parseInt(part, 10);
        if (!Number.isFinite(v)) return null;
        values.add(v);
      }
    }
    return values.size;
  }

  const minVal = minRaw.trim() ? Number.parseInt(minRaw, 10) : null;
  const maxVal = maxRaw.trim() ? Number.parseInt(maxRaw, 10) : null;
  if (minVal === null && maxVal === null) return null;
  const step = stepRaw.trim() ? Number.parseInt(stepRaw, 10) : 1;
  if (!Number.isFinite(step) || step <= 0) return null;
  const lo = minVal ?? 0;
  const hi = maxVal ?? lo;
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi < lo) return null;
  return Math.floor((hi - lo) / step) + 1;
};

const formatByteCount = (bytes?: number | null): string => {
  if (!bytes) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(1)} MB`;
};

const formatTimestamp = (value?: string | null): string =>
  value ? new Date(value).toLocaleString() : "Unknown";

const PREVIEW_LIMIT = 20;

type PreviewTarget = {
  label: string;
  path: string;
};

const dedupeFiles = (files: DatasetFileSummary[]): DatasetFileSummary[] => {
  const seen = new Set<string>();
  return files.filter((file) => {
    if (seen.has(file.path)) return false;
    seen.add(file.path);
    return true;
  });
};

const buildRunFiles = (run: DatasetRunSummary): DatasetFileSummary[] => {
  const listed = run.files?.filter(Boolean) ?? [];
  if (listed.length > 0) {
    return dedupeFiles(listed);
  }
  const fallback = [
    run.training_file,
    run.dataset_file,
    run.drops_file,
  ].filter(Boolean) as DatasetFileSummary[];
  return dedupeFiles(fallback);
};

const sortRunFiles = (
  files: DatasetFileSummary[],
  trainingPath?: string | null,
): DatasetFileSummary[] => {
  return [...files].sort((a, b) => {
    const aIsTraining = trainingPath && a.path === trainingPath;
    const bIsTraining = trainingPath && b.path === trainingPath;
    if (aIsTraining && !bIsTraining) return -1;
    if (!aIsTraining && bIsTraining) return 1;
    return a.name.localeCompare(b.name);
  });
};

const countMondaysInRange = (start: string, end: string): number | null => {
  if (!start || !end) return null;
  const startDate = new Date(`${start}T00:00:00Z`);
  const endDate = new Date(`${end}T00:00:00Z`);
  if (Number.isNaN(startDate.getTime()) || Number.isNaN(endDate.getTime())) {
    return null;
  }
  if (endDate < startDate) return null;
  let count = 0;
  const cursor = new Date(startDate);
  while (cursor <= endDate) {
    if (cursor.getUTCDay() === 1) count += 1;
    cursor.setUTCDate(cursor.getUTCDate() + 1);
  }
  return count;
};

const buildCommandPreview = (state: DatasetFormState): string => {
  const args: string[] = [
    "python",
    "src/scripts/1-option-chain-build-historic-dataset-v1.0.py",
  ];

  const addValue = (flag: string, value: string) => {
    const trimmed = value.trim();
    if (!trimmed) return;
    args.push(flag, trimmed);
  };
  const addFlag = (flag: string, enabled: boolean) => {
    if (enabled) args.push(flag);
  };
  const addOptionalBool = (flag: string, negFlag: string, value: boolean) => {
    args.push(value ? flag : negFlag);
  };

  addValue("--out-dir", state.outDir);
  addValue("--dataset-name", state.datasetName);
  addValue("--schedule-mode", state.scheduleMode);
  addValue("--expiry-weekdays", state.expiryWeekdays);
  addValue("--asof-weekdays", state.asofWeekdays);
  addValue("--dte-list", state.dteList);
  addValue("--dte-min", state.dteMin);
  addValue("--dte-max", state.dteMax);
  addValue("--dte-step", state.dteStep);
  addOptionalBool("--write-snapshot", "--no-write-snapshot", state.writeSnapshot);
  addOptionalBool("--write-prn-view", "--no-write-prn-view", state.writePrnView);
  addOptionalBool("--write-train-view", "--no-write-train-view", state.writeTrainView);
  addOptionalBool("--write-legacy", "--no-write-legacy", state.writeLegacy);
  addValue("--prn-version", state.prnVersion);
  addValue("--prn-config-hash", state.prnConfigHash);
  addValue("--tickers", state.tickers);
  addValue("--start", state.start);
  addValue("--end", state.end);
  addValue("--theta-base-url", state.thetaBaseUrl);
  addValue("--stock-source", state.stockSource);
  addValue("--timeout-s", state.timeoutS);
  addValue("--r", state.riskFreeRate);

  addValue("--max-abs-logm", state.maxAbsLogm);
  addValue("--max-abs-logm-cap", state.maxAbsLogmCap);
  addValue("--band-widen-step", state.bandWidenStep);
  addFlag("--no-adaptive-band", !state.adaptiveBand);
  addValue("--max-band-strikes", state.maxBandStrikes);
  addValue("--min-band-strikes", state.minBandStrikes);
  addValue("--min-band-prn-strikes", state.minBandPrnStrikes);

  addValue("--strike-range", state.strikeRange);
  addFlag("--no-retry-full-chain", !state.retryFullChain);
  addFlag("--no-sat-expiry-fallback", !state.saturdayExpiryFallback);
  addValue("--threads", state.threads);

  addOptionalBool("--prefer-bidask", "--no-prefer-bidask", state.preferBidask);
  addValue("--min-trade-count", state.minTradeCount);
  addValue("--min-volume", state.minVolume);

  addValue("--min-chain-used-hard", state.minChainUsedHard);
  addValue("--max-rel-spread-median-hard", state.maxRelSpreadMedianHard);
  addFlag("--hard-drop-close-fallback", state.hardDropCloseFallback);

  addValue("--min-prn-train", state.minPrnTrain);
  addValue("--max-prn-train", state.maxPrnTrain);

  addFlag("--no-split-adjust", !state.splitAdjust);

  addValue("--dividend-source", state.dividendSource);
  addValue("--dividend-lookback-days", state.dividendLookbackDays);
  addValue("--dividend-yield-default", state.dividendYieldDefault);
  addFlag("--no-forward-moneyness", !state.forwardMoneyness);

  addFlag("--no-group-weights", !state.groupWeights);
  addFlag("--no-ticker-weights", !state.tickerWeights);
  addFlag("--no-soft-quality-weight", !state.softQualityWeight);

  addValue("--rv-lookback-days", state.rvLookbackDays);

  addOptionalBool("--cache", "--no-cache", state.cache);

  addFlag("--write-drops", state.writeDrops);

  addFlag("--sanity-report", state.sanityReport);
  addFlag("--sanity-drop", state.sanityDrop);
  addValue("--sanity-abs-logm-max", state.sanityAbsLogmMax);
  addValue("--sanity-k-over-s-min", state.sanityKOverSMin);
  addValue("--sanity-k-over-s-max", state.sanityKOverSMax);

  addFlag("--verbose-skips", state.verboseSkips);

  return args.join(" ");
};

export default function DatasetsPage() {
  const [formState, setFormState] = useState<DatasetFormState>(defaultForm);
  const [isRunning, setIsRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [runResult, setRunResult] = useState<DatasetRunResponse | null>(null);
  const [activeLog, setActiveLog] = useState<"stdout" | "stderr">("stdout");
  const [storageReady, setStorageReady] = useState(false);
  const { jobStatus, jobId, setJobId, setJobStatus: setGlobalJobStatus } =
    useDatasetJob();
  const { anyJobRunning, primaryJob } = useAnyJobRunning();
  const [killLoading, setKillLoading] = useState(false);
  const [datasetRuns, setDatasetRuns] = useState<DatasetRunSummary[]>([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [previewTarget, setPreviewTarget] = useState<PreviewTarget | null>(null);
  const [previewResponse, setPreviewResponse] =
    useState<DatasetPreviewResponse | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [deleteConfirmRun, setDeleteConfirmRun] = useState<string | null>(null);
  const [deleteLoadingRun, setDeleteLoadingRun] = useState<string | null>(null);
  const [renamingRunId, setRenamingRunId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState<string>("");
  const [renameError, setRenameError] = useState<string | null>(null);
  const [renameLoading, setRenameLoading] = useState(false);
  const [customTickerInput, setCustomTickerInput] = useState("");
  const refreshDatasetRuns = useCallback(async () => {
    setRunsLoading(true);
    setRunsError(null);
    try {
      const response = await listDatasetRuns();
      setDatasetRuns(response.runs);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setRunsError(message);
    } finally {
      setRunsLoading(false);
    }
  }, []);

  const selectedTickers = useMemo(() => {
    const parsed = normalizeTickers(parseTickers(formState.tickers) ?? []);
    return orderTickers(parsed);
  }, [formState.tickers]);
  const selectedTickerSet = useMemo(
    () => new Set(selectedTickers),
    [selectedTickers],
  );
  const customTickers = useMemo(
    () =>
      selectedTickers.filter(
        (ticker) => !TRADING_UNIVERSE_TICKERS.includes(ticker),
      ),
    [selectedTickers],
  );
  const customTickerCount = customTickers.length;
  const resolvedTickersCount = selectedTickers.length;
  const weeksCount = useMemo(
    () => countMondaysInRange(formState.start, formState.end),
    [formState.start, formState.end],
  );
  const expiryWeekdays = useMemo(
    () => parseWeekdays(formState.expiryWeekdays),
    [formState.expiryWeekdays],
  );
  const asofWeekdays = useMemo(
    () => parseWeekdays(formState.asofWeekdays),
    [formState.asofWeekdays],
  );
  const expiriesCount = useMemo(() => {
    if (formState.scheduleMode === "weekly") return weeksCount;
    return countWeekdaysInRange(formState.start, formState.end, expiryWeekdays);
  }, [
    formState.scheduleMode,
    formState.start,
    formState.end,
    expiryWeekdays,
    weeksCount,
  ]);
  const dteCount = useMemo(
    () =>
      resolveDteCount(
        formState.dteList,
        formState.dteMin,
        formState.dteMax,
        formState.dteStep,
      ),
    [formState.dteList, formState.dteMin, formState.dteMax, formState.dteStep],
  );
  const asofCount = dteCount ?? (asofWeekdays ? asofWeekdays.length : null);
  const snapshotCountLabel = dteCount
    ? `${dteCount} DTEs`
    : asofWeekdays
      ? `${asofWeekdays.length} weekdays`
      : "—";
  const plannedJobs =
    expiriesCount && asofCount && resolvedTickersCount
      ? expiriesCount * asofCount * resolvedTickersCount
      : null;
  const commandPreview = useMemo(
    () => buildCommandPreview(formState),
    [formState],
  );

  const updateTickers = useCallback((next: string[]) => {
    const normalized = normalizeTickers(next);
    if (normalized.length === 0) return;
    const ordered = orderTickers(normalized);
    setFormState((prev) => ({
      ...prev,
      tickers: formatTickerList(ordered),
    }));
  }, []);

  const updateJobState = (status: DatasetJobStatus) => {
    setGlobalJobStatus(status);
    setJobId(status.job_id);
    const running = status.status === "queued" || status.status === "running";
    setIsRunning(running);
    if (status.result) {
      setRunResult(status.result);
    }
    if (status.status === "failed" && status.error) {
      setRunError(status.error);
    } else if (status.status === "cancelled") {
      setRunError("Dataset creation was cancelled.");
    } else if (running) {
      setRunError(null);
    }
  };

  useEffect(() => {
    if (!jobStatus) return;
    if (jobStatus.result && !jobStatus.result.ok && jobStatus.result.stderr) {
      setActiveLog("stderr");
    } else {
      setActiveLog("stdout");
    }
  }, [jobStatus]);

  useEffect(() => {
    refreshDatasetRuns();
  }, [refreshDatasetRuns]);

  useEffect(() => {
    if (selectedTickers.length === 0) {
      updateTickers(TRADING_UNIVERSE_TICKERS);
    }
  }, [selectedTickers, updateTickers]);

  useEffect(() => {
    if (!jobStatus) return;
    if (["finished", "failed", "cancelled"].includes(jobStatus.status)) {
      refreshDatasetRuns();
    }
  }, [jobStatus?.status, refreshDatasetRuns]);

  const resolvedRange =
    formState.start && formState.end
      ? `${formState.start} → ${formState.end}`
      : "Select a date range";
  const resolvedTickersLabel = customTickers.length
    ? `${resolvedTickersCount} tickers (${customTickers.length} custom)`
    : `${resolvedTickersCount} tickers`;
  const plannedJobsLabel = plannedJobs
    ? `${plannedJobs.toLocaleString()} jobs`
    : "Set a date range";
  const plannedWeeksLabel =
    expiriesCount !== null ? `${expiriesCount} expiries` : "Expiries pending";
  const jobProgress = jobStatus?.progress ?? null;
  const progressPercent =
    jobProgress && jobProgress.total > 0
      ? Math.round((jobProgress.done / jobProgress.total) * 100)
      : jobStatus && (jobStatus.status === "running" || jobStatus.status === "queued")
      ? 0
      : null;
  const stdoutText =
    jobStatus?.stdout.join("") ||
    jobStatus?.result?.stdout ||
    runResult?.stdout ||
    "";
  const stderrText =
    jobStatus?.stderr.join("") ||
    jobStatus?.result?.stderr ||
    runResult?.stderr ||
    "";
  const currentResult = jobStatus?.result ?? runResult;
  const statusClass = jobStatus
    ? jobStatus.status === "running" || jobStatus.status === "queued"
      ? "running"
      : jobStatus.status === "finished"
        ? currentResult?.ok
          ? "success"
          : "failed"
        : "failed"
    : "running";
  const statusLabel = jobStatus
    ? jobStatus.status === "running" || jobStatus.status === "queued"
      ? "Running"
      : jobStatus.status === "finished"
        ? currentResult?.ok
          ? "Success"
          : "Failed"
        : "Cancelled"
    : "Idle";
  const datasetNameKebab = toKebabCase(formState.datasetName);
  const trainingDatasetPath =
    currentResult?.out_dir && datasetNameKebab
      ? `${currentResult.out_dir}/${datasetNameKebab}/training-${datasetNameKebab}.csv`
      : null;
  const trainingDatasetEnabled = formState.writeTrainView;

  useEffect(() => {
    if (!jobStatus || jobStatus.status !== "finished") return;
    if (!currentResult?.ok) return;
    if (!trainingDatasetPath || !trainingDatasetEnabled) return;
    try {
      const raw = localStorage.getItem(CALIBRATE_STORAGE_KEY);
      const parsed = raw ? JSON.parse(raw) : {};
      const next = { ...parsed, datasetPath: trainingDatasetPath };
      localStorage.setItem(CALIBRATE_STORAGE_KEY, JSON.stringify(next));
    } catch {
      // ignore storage failures
    }
  }, [
    jobStatus?.status,
    currentResult?.ok,
    trainingDatasetPath,
    trainingDatasetEnabled,
  ]);

  useEffect(() => {
    if (!previewTarget) {
      setPreviewResponse(null);
      setPreviewError(null);
      setPreviewLoading(false);
      return;
    }
    let cancelled = false;
    setPreviewLoading(true);
    setPreviewError(null);
    previewDatasetFile(previewTarget.path, "head", PREVIEW_LIMIT)
      .then((result) => {
        if (cancelled) return;
        setPreviewResponse(result);
      })
      .catch((err) => {
        if (cancelled) return;
        setPreviewResponse(null);
        const message = err instanceof Error ? err.message : "Unknown error";
        setPreviewError(message);
      })
      .finally(() => {
        if (!cancelled) setPreviewLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [previewTarget]);

  useEffect(() => {
    const stored = loadStoredForm();
    if (stored) {
      setFormState((prev) => ({ ...prev, ...stored }));
    }
    setStorageReady(true);
  }, []);

  useEffect(() => {
    if (!formState.writeTrainView) {
      setFormState((prev) => ({ ...prev, writeTrainView: true }));
    }
  }, [formState.writeTrainView]);

  useEffect(() => {
    if (!storageReady) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(formState));
    } catch {
      // ignore storage failures
    }
  }, [formState, storageReady]);

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
    setGlobalJobStatus(null);
    setJobId(null);
    setActiveLog("stdout");
    setIsRunning(true);

    try {
      if (!formState.datasetName.trim()) {
        setRunError("Dataset name is required.");
        setIsRunning(false);
        return;
      }
      const payload = {
        outDir: formState.outDir.trim() || undefined,
        datasetName: formState.datasetName.trim(),
        scheduleMode: formState.scheduleMode,
        expiryWeekdays: formState.expiryWeekdays.trim() || undefined,
        asofWeekdays: formState.asofWeekdays.trim() || undefined,
        dteList: formState.dteList.trim() || undefined,
        dteMin: parseOptionalInt(formState.dteMin),
        dteMax: parseOptionalInt(formState.dteMax),
        dteStep: parseOptionalInt(formState.dteStep),
        writeSnapshot: formState.writeSnapshot,
        writePrnView: formState.writePrnView,
        writeTrainView: formState.writeTrainView,
        writeLegacy: formState.writeLegacy,
        prnVersion: formState.prnVersion.trim() || undefined,
        prnConfigHash: formState.prnConfigHash.trim() || undefined,
        tickers: formState.tickers.trim() || undefined,
        start: formState.start,
        end: formState.end,
        thetaBaseUrl: formState.thetaBaseUrl.trim() || undefined,
        stockSource: formState.stockSource,
        timeoutS: parseOptionalInt(formState.timeoutS),
        riskFreeRate: parseOptionalNumber(formState.riskFreeRate),
        maxAbsLogm: parseOptionalNumber(formState.maxAbsLogm),
        maxAbsLogmCap: parseOptionalNumber(formState.maxAbsLogmCap),
        bandWidenStep: parseOptionalNumber(formState.bandWidenStep),
        noAdaptiveBand: !formState.adaptiveBand,
        maxBandStrikes: parseOptionalInt(formState.maxBandStrikes),
        minBandStrikes: parseOptionalInt(formState.minBandStrikes),
        minBandPrnStrikes: parseOptionalInt(formState.minBandPrnStrikes),
        strikeRange: parseOptionalInt(formState.strikeRange),
        noRetryFullChain: !formState.retryFullChain,
        noSatExpiryFallback: !formState.saturdayExpiryFallback,
        threads: parseOptionalInt(formState.threads),
        preferBidask: formState.preferBidask,
        minTradeCount: parseOptionalInt(formState.minTradeCount),
        minVolume: parseOptionalInt(formState.minVolume),
        minChainUsedHard: parseOptionalInt(formState.minChainUsedHard),
        maxRelSpreadMedianHard: parseOptionalNumber(
          formState.maxRelSpreadMedianHard,
        ),
        hardDropCloseFallback: formState.hardDropCloseFallback,
        minPrnTrain: parseOptionalNumber(formState.minPrnTrain),
        maxPrnTrain: parseOptionalNumber(formState.maxPrnTrain),
        noSplitAdjust: !formState.splitAdjust,
        dividendSource: formState.dividendSource,
        dividendLookbackDays: parseOptionalInt(formState.dividendLookbackDays),
        dividendYieldDefault: parseOptionalNumber(
          formState.dividendYieldDefault,
        ),
        noForwardMoneyness: !formState.forwardMoneyness,
        noGroupWeights: !formState.groupWeights,
        noTickerWeights: !formState.tickerWeights,
        noSoftQualityWeight: !formState.softQualityWeight,
        rvLookbackDays: parseOptionalInt(formState.rvLookbackDays),
        cache: formState.cache,
        writeDrops: formState.writeDrops,
        sanityReport: formState.sanityReport,
        sanityDrop: formState.sanityDrop,
        sanityAbsLogmMax: parseOptionalNumber(formState.sanityAbsLogmMax),
        sanityKOverSMin: parseOptionalNumber(formState.sanityKOverSMin),
        sanityKOverSMax: parseOptionalNumber(formState.sanityKOverSMax),
        verboseSkips: formState.verboseSkips,
      };

      const status = await startDatasetJob(payload);
      updateJobState(status);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setRunError(message);
      setIsRunning(false);
    }
  };

  const handleKill = async () => {
    if (!jobId) return;
    setKillLoading(true);
    try {
      const status = await killDatasetJob(jobId);
      updateJobState(status);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setRunError(message);
    } finally {
      setKillLoading(false);
    }
  };

  const handlePreviewSelection = useCallback((target: PreviewTarget) => {
    setPreviewTarget(target);
  }, []);

  const handleDeleteRun = async (runDir: string) => {
    setDeleteLoadingRun(runDir);
    setDeleteConfirmRun(null);
    try {
      await deleteDatasetRun(runDir);
      if (previewTarget?.path.startsWith(runDir)) {
        setPreviewTarget(null);
      }
      await refreshDatasetRuns();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setRunsError(message);
    } finally {
      setDeleteLoadingRun(null);
    }
  };

  const handleStartRename = (runId: string, currentName: string) => {
    setRenamingRunId(runId);
    setRenameValue(currentName);
    setRenameError(null);
  };

  const handleCancelRename = () => {
    setRenamingRunId(null);
    setRenameValue("");
    setRenameError(null);
  };

  const handleConfirmRename = async (runId: string, runDir: string) => {
    const trimmed = renameValue.trim();
    if (!trimmed) {
      setRenameError("Directory name cannot be empty.");
      return;
    }
    setRenameLoading(true);
    setRenameError(null);
    try {
      const updated = await renameDatasetRun(runDir, trimmed);
      setDatasetRuns((prev) =>
        prev.map((run) => (run.id === runId ? updated : run)),
      );
      if (previewTarget?.path.startsWith(runDir)) {
        setPreviewTarget(null);
      }
      handleCancelRename();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setRenameError(message);
    } finally {
      setRenameLoading(false);
    }
  };

  const toggleUniverseTicker = useCallback(
    (ticker: string) => {
      const next = selectedTickerSet.has(ticker)
        ? selectedTickers.filter((value) => value !== ticker)
        : [...selectedTickers, ticker];
      updateTickers(next);
    },
    [selectedTickers, selectedTickerSet, updateTickers],
  );

  const removeSelectedTicker = useCallback(
    (ticker: string) => {
      if (!selectedTickerSet.has(ticker)) return;
      const next = selectedTickers.filter((value) => value !== ticker);
      updateTickers(next);
    },
    [selectedTickers, selectedTickerSet, updateTickers],
  );

  const handleAddCustomTicker = useCallback(() => {
    const tokens = splitTickerInput(customTickerInput);
    if (!tokens.length) return;
    const next = [...selectedTickers, ...tokens];
    updateTickers(next);
    setCustomTickerInput("");
  }, [customTickerInput, selectedTickers, updateTickers]);

  return (
    <section className="page datasets-page">
      <header className="page-header">
        <div>
          <p className="page-kicker">Option Chain</p>
          <h1 className="page-title">Build the option chain dataset</h1>
          <p className="page-subtitle">
            Generate the option-chain dataset that feeds calibration and keep every CLI input tracked in one place.
          </p>
        </div>
        <div className="meta-card datasets-meta page-goal-card">
          <span className="meta-label">Goal</span>
          <span>Produce the reproducible option-chain dataset that powers calibration.</span>
          <div className="meta-pill">Outputs stored under src/data/raw/option-chain</div>
        </div>
      </header>

      <div className="datasets-grid">
        <section className="panel">
          <div className="panel-header">
            <h2>Run configuration</h2>
            <span className="panel-hint">
              Start/end dates are required; everything else can be tuned.
            </span>
          </div>
          <div className="config-summary">
            <div>
              <span className="meta-label">Date range</span>
              <span>{resolvedRange}</span>
            </div>
            <div>
              <span className="meta-label">Tickers</span>
              <span>{resolvedTickersLabel}</span>
            </div>
            <div>
              <span className="meta-label">Dataset</span>
              <span>
                {formState.outDir}/{datasetNameKebab || "(unnamed)"}
              </span>
            </div>
            <div>
              <span className="meta-label">Schedule</span>
              <span>
                {formState.scheduleMode}
                {dteCount
                  ? ` · DTE ${formState.dteList || `${formState.dteMin}-${formState.dteMax}`}`
                  : ` · asof ${formState.asofWeekdays}`}
              </span>
            </div>
            <div>
              <span className="meta-label">Planned workload</span>
              <span>{plannedJobsLabel}</span>
            </div>
          </div>
          <form className="panel-body" onSubmit={handleSubmit}>
            <div className="section-card dataset-section">
              <h3>Core range</h3>
              <div className="inline-fields">
                <div className="field">
                  <label htmlFor="datasetStart">Start date</label>
                  <input
                    id="datasetStart"
                    className="input"
                    type="date"
                    min="2023-06-01"
                    max={new Date().toISOString().slice(0, 10)}
                    required
                    value={formState.start}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        start: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="datasetEnd">End date</label>
                  <input
                    id="datasetEnd"
                    className="input"
                    type="date"
                    min={formState.start || "2023-06-01"}
                    max={new Date().toISOString().slice(0, 10)}
                    required
                    value={formState.end}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        end: event.target.value,
                      }))
                    }
                  />
                </div>
              </div>
              <div className="field">
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
                  Pick from the core trading universe, then add any extra tickers below.
                </span>
              </div>
              <div className="field">
                <label htmlFor="datasetCustomTicker">Add custom tickers</label>
                <div className="inline-fields compact">
                  <input
                    id="datasetCustomTicker"
                    className="input"
                    placeholder="e.g. AMD, INTC"
                    value={customTickerInput}
                    onChange={(event) => setCustomTickerInput(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        event.preventDefault();
                        handleAddCustomTicker();
                      }
                    }}
                  />
                  <button
                    type="button"
                    className="button light"
                    onClick={handleAddCustomTicker}
                  >
                    Add
                  </button>
                </div>
                <span className="field-hint">
                  Custom tickers are appended to the selection and can be removed below.
                </span>
              </div>
              <div className="field">
                <label>Selected tickers</label>
                <div className="ticker-selection">
                  {selectedTickers.map((ticker) => {
                    const isCustom = !TRADING_UNIVERSE_TICKERS.includes(ticker);
                    return (
                      <button
                        key={ticker}
                        type="button"
                        className={`ticker-chip selected ${
                          isCustom ? "custom" : ""
                        }`}
                        onClick={() => removeSelectedTicker(ticker)}
                        aria-label={`Remove ${ticker}`}
                      >
                        <span>{ticker}</span>
                        <span className="ticker-remove">×</span>
                      </button>
                    );
                  })}
                </div>
                <span className="field-hint">
                  Click a ticker to remove it from the selection.
                </span>
              </div>
            </div>

            <div className="section-card dataset-section">
              <h3>Output targets</h3>
              <div className="inline-fields">
                <div className="field">
                  <label>Output directory</label>
                  <div className="field-value">{formState.outDir}</div>
                </div>
                <div className="field">
                  <label htmlFor="datasetName">Dataset name</label>
                  <input
                    id="datasetName"
                    className="input"
                    placeholder="e.g. pm10-mon-thu-v2"
                    required
                    value={formState.datasetName}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        datasetName: event.target.value,
                      }))
                    }
                  />
                </div>
              </div>
              <div className="inline-fields">
                <div className="field">
                  <label htmlFor="prnVersion">pRN version</label>
                  <input
                    id="prnVersion"
                    className="input"
                    value={formState.prnVersion}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        prnVersion: event.target.value,
                      }))
                    }
                  />
                </div>
              </div>
              <div className="inline-fields">
                <div className="field">
                  <label>Outputs to generate</label>
                  <label className="checkbox">
                    <input
                      type="checkbox"
                      checked={formState.writeTrainView}
                      disabled
                      readOnly
                    />
                    training-{datasetNameKebab || "{name}"}.csv
                  </label>
                  <label className="checkbox">
                    <input
                      type="checkbox"
                      checked={formState.writeSnapshot}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          writeSnapshot: event.target.checked,
                        }))
                      }
                    />
                    snapshot-{datasetNameKebab || "{name}"}.csv
                  </label>
                  <label className="checkbox">
                    <input
                      type="checkbox"
                      checked={formState.writePrnView}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          writePrnView: event.target.checked,
                        }))
                      }
                    />
                    prn-view-{datasetNameKebab || "{name}"}.csv
                  </label>
                  <label className="checkbox">
                    <input
                      type="checkbox"
                      checked={formState.writeLegacy}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          writeLegacy: event.target.checked,
                        }))
                      }
                    />
                    legacy-{datasetNameKebab || "{name}"}.csv
                  </label>
                  <label
                    className="checkbox"
                    title="Written as drops-{name}.csv"
                  >
                    <input
                      type="checkbox"
                      checked={formState.writeDrops}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          writeDrops: event.target.checked,
                        }))
                      }
                    />
                    drops-{datasetNameKebab || "{name}"}.csv
                  </label>
                </div>
              </div>
            </div>

            <div className="section-card dataset-section">
              <h3>Snapshot schedule</h3>
              <div className="inline-fields">
                <div className="field">
                  <label htmlFor="scheduleMode">How are start/end dates used?</label>
                  <select
                    id="scheduleMode"
                    className="input"
                    value={formState.scheduleMode}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        scheduleMode: event.target
                          .value as DatasetFormState["scheduleMode"],
                      }))
                    }
                  >
                    <option value="weekly">Weekly (generate Monday anchors)</option>
                    <option value="expiry_range">Expiry range (start/end are expiry dates)</option>
                  </select>
                </div>
                {formState.scheduleMode === "expiry_range" ? (
                  <div className="field">
                    <label htmlFor="expiryWeekdays">Expiry weekdays</label>
                    <input
                      id="expiryWeekdays"
                      className="input"
                      value={formState.expiryWeekdays}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          expiryWeekdays: event.target.value,
                        }))
                      }
                    />
                    <span className="field-hint">
                      e.g. fri or mon,fri (comma-separated).
                    </span>
                  </div>
                ) : null}
              </div>
              <div className="inline-fields">
                <div className="field">
                  <label htmlFor="asofWeekdays">Observation weekdays</label>
                  <input
                    id="asofWeekdays"
                    className="input"
                    value={formState.asofWeekdays}
                    disabled={!!formState.dteList.trim()}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        asofWeekdays: event.target.value,
                      }))
                    }
                  />
                  <span className="field-hint">
                    Days to observe the chain.{" "}
                    {formState.dteList.trim()
                      ? "Disabled because DTE list is set."
                      : "e.g. mon,tue,wed,thu"}
                  </span>
                </div>
                <div className="field">
                  <label htmlFor="dteList">DTE list (overrides weekdays)</label>
                  <input
                    id="dteList"
                    className="input"
                    placeholder="e.g. 1,2,3 or 1-5"
                    value={formState.dteList}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        dteList: event.target.value,
                      }))
                    }
                  />
                  <span className="field-hint">
                    Days-to-expiry to observe. Leave blank to use weekdays above.
                  </span>
                </div>
              </div>
              {formState.dteList.trim() ? (
                <div className="inline-fields">
                  <div className="field">
                    <label htmlFor="dteMin">DTE min</label>
                    <input
                      id="dteMin"
                      className="input"
                      inputMode="numeric"
                      value={formState.dteMin}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          dteMin: event.target.value,
                        }))
                      }
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="dteMax">DTE max</label>
                    <input
                      id="dteMax"
                      className="input"
                      inputMode="numeric"
                      value={formState.dteMax}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          dteMax: event.target.value,
                        }))
                      }
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="dteStep">DTE step</label>
                    <input
                      id="dteStep"
                      className="input"
                      inputMode="numeric"
                      value={formState.dteStep}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          dteStep: event.target.value,
                        }))
                      }
                    />
                  </div>
                </div>
              ) : null}
            </div>

            <details className="advanced">
              <summary>Market data & runtime</summary>
              <div className="fields-grid">
                <div className="field">
                  <label htmlFor="thetaBaseUrl">Theta base URL</label>
                  <input
                    id="thetaBaseUrl"
                    className="input"
                    value={formState.thetaBaseUrl}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        thetaBaseUrl: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="stockSource">Stock source</label>
                  <select
                    id="stockSource"
                    className="input"
                    value={formState.stockSource}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        stockSource: event.target.value as DatasetFormState["stockSource"],
                      }))
                    }
                  >
                    <option value="yfinance">yfinance</option>
                    <option value="theta">theta</option>
                    <option value="auto">auto</option>
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="timeoutS">Timeout (s)</label>
                  <input
                    id="timeoutS"
                    className="input"
                    inputMode="numeric"
                    value={formState.timeoutS}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        timeoutS: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="riskFreeRate">Risk-free rate (r)</label>
                  <input
                    id="riskFreeRate"
                    className="input"
                    inputMode="decimal"
                    value={formState.riskFreeRate}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        riskFreeRate: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="threads">Threads</label>
                  <input
                    id="threads"
                    className="input"
                    inputMode="numeric"
                    value={formState.threads}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        threads: event.target.value,
                      }))
                    }
                  />
                </div>
              </div>
            </details>

            <details className="advanced">
              <summary>Band selection & training</summary>
              <div className="fields-grid">
                <div className="field">
                  <label htmlFor="maxAbsLogm">Max abs log-m</label>
                  <input
                    id="maxAbsLogm"
                    className="input"
                    inputMode="decimal"
                    value={formState.maxAbsLogm}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        maxAbsLogm: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="maxAbsLogmCap">Max abs log-m cap</label>
                  <input
                    id="maxAbsLogmCap"
                    className="input"
                    inputMode="decimal"
                    value={formState.maxAbsLogmCap}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        maxAbsLogmCap: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="bandWidenStep">Band widen step</label>
                  <input
                    id="bandWidenStep"
                    className="input"
                    inputMode="decimal"
                    value={formState.bandWidenStep}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        bandWidenStep: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="maxBandStrikes">Max band strikes</label>
                  <input
                    id="maxBandStrikes"
                    className="input"
                    inputMode="numeric"
                    value={formState.maxBandStrikes}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        maxBandStrikes: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="minBandStrikes">Min band strikes</label>
                  <input
                    id="minBandStrikes"
                    className="input"
                    inputMode="numeric"
                    value={formState.minBandStrikes}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        minBandStrikes: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="minBandPrnStrikes">
                    Min band pRN strikes
                  </label>
                  <input
                    id="minBandPrnStrikes"
                    className="input"
                    inputMode="numeric"
                    value={formState.minBandPrnStrikes}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        minBandPrnStrikes: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="minPrnTrain">Min pRN train</label>
                  <input
                    id="minPrnTrain"
                    className="input"
                    inputMode="decimal"
                    value={formState.minPrnTrain}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        minPrnTrain: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="maxPrnTrain">Max pRN train</label>
                  <input
                    id="maxPrnTrain"
                    className="input"
                    inputMode="decimal"
                    value={formState.maxPrnTrain}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        maxPrnTrain: event.target.value,
                      }))
                    }
                  />
                </div>
              </div>
              <div className="toggle-grid">
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.adaptiveBand}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        adaptiveBand: event.target.checked,
                      }))
                    }
                  />
                  Adaptive band (disable with <code>--no-adaptive-band</code>)
                </label>
              </div>
            </details>

            <details className="advanced">
              <summary>Option chain & expiry</summary>
              <div className="fields-grid">
                <div className="field">
                  <label htmlFor="strikeRange">Strike range</label>
                  <input
                    id="strikeRange"
                    className="input"
                    inputMode="numeric"
                    value={formState.strikeRange}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        strikeRange: event.target.value,
                      }))
                    }
                  />
                </div>
              </div>
              <div className="toggle-grid">
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.retryFullChain}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        retryFullChain: event.target.checked,
                      }))
                    }
                  />
                  Retry full chain if band thin
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.saturdayExpiryFallback}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        saturdayExpiryFallback: event.target.checked,
                      }))
                    }
                  />
                  Saturday expiry fallback
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.splitAdjust}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        splitAdjust: event.target.checked,
                      }))
                    }
                  />
                  Apply split adjustment
                </label>
              </div>
            </details>

            <details className="advanced">
              <summary>Liquidity & filters</summary>
              <div className="fields-grid">
                <div className="field">
                  <label htmlFor="minTradeCount">Min trade count</label>
                  <input
                    id="minTradeCount"
                    className="input"
                    inputMode="numeric"
                    value={formState.minTradeCount}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        minTradeCount: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="minVolume">Min volume</label>
                  <input
                    id="minVolume"
                    className="input"
                    inputMode="numeric"
                    value={formState.minVolume}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        minVolume: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="minChainUsedHard">Min chain used hard</label>
                  <input
                    id="minChainUsedHard"
                    className="input"
                    inputMode="numeric"
                    value={formState.minChainUsedHard}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        minChainUsedHard: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="maxRelSpreadMedianHard">
                    Max rel spread median hard
                  </label>
                  <input
                    id="maxRelSpreadMedianHard"
                    className="input"
                    inputMode="decimal"
                    value={formState.maxRelSpreadMedianHard}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        maxRelSpreadMedianHard: event.target.value,
                      }))
                    }
                  />
                </div>
              </div>
              <div className="toggle-grid">
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.preferBidask}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        preferBidask: event.target.checked,
                      }))
                    }
                  />
                  Prefer bid/ask quotes
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.hardDropCloseFallback}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        hardDropCloseFallback: event.target.checked,
                      }))
                    }
                  />
                  Hard drop close fallback
                </label>
              </div>
            </details>

            <details className="advanced">
              <summary>Dividends, weights & volatility</summary>
              <div className="fields-grid">
                <div className="field">
                  <label htmlFor="dividendSource">Dividend source</label>
                  <select
                    id="dividendSource"
                    className="input"
                    value={formState.dividendSource}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        dividendSource: event.target.value as DatasetFormState["dividendSource"],
                      }))
                    }
                  >
                    <option value="yfinance">yfinance</option>
                    <option value="none">none</option>
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="dividendLookbackDays">
                    Dividend lookback days
                  </label>
                  <input
                    id="dividendLookbackDays"
                    className="input"
                    inputMode="numeric"
                    value={formState.dividendLookbackDays}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        dividendLookbackDays: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="dividendYieldDefault">
                    Dividend yield default
                  </label>
                  <input
                    id="dividendYieldDefault"
                    className="input"
                    inputMode="decimal"
                    value={formState.dividendYieldDefault}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        dividendYieldDefault: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="rvLookbackDays">RV lookback days</label>
                  <input
                    id="rvLookbackDays"
                    className="input"
                    inputMode="numeric"
                    value={formState.rvLookbackDays}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        rvLookbackDays: event.target.value,
                      }))
                    }
                  />
                </div>
              </div>
              <div className="toggle-grid">
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.forwardMoneyness}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        forwardMoneyness: event.target.checked,
                      }))
                    }
                  />
                  Use forward moneyness
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.groupWeights}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        groupWeights: event.target.checked,
                      }))
                    }
                  />
                  Add group weights
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.tickerWeights}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        tickerWeights: event.target.checked,
                      }))
                    }
                  />
                  Add ticker weights
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.softQualityWeight}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        softQualityWeight: event.target.checked,
                      }))
                    }
                  />
                  Soft quality weighting
                </label>
              </div>
            </details>

            <details className="advanced">
              <summary>Cache & sanity checks</summary>
              <div className="fields-grid">
                <div className="field">
                  <label htmlFor="sanityAbsLogmMax">Sanity abs log-m max</label>
                  <input
                    id="sanityAbsLogmMax"
                    className="input"
                    inputMode="decimal"
                    value={formState.sanityAbsLogmMax}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        sanityAbsLogmMax: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="sanityKOverSMin">Sanity K/S min</label>
                  <input
                    id="sanityKOverSMin"
                    className="input"
                    inputMode="decimal"
                    value={formState.sanityKOverSMin}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        sanityKOverSMin: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="sanityKOverSMax">Sanity K/S max</label>
                  <input
                    id="sanityKOverSMax"
                    className="input"
                    inputMode="decimal"
                    value={formState.sanityKOverSMax}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        sanityKOverSMax: event.target.value,
                      }))
                    }
                  />
                </div>
              </div>
              <div className="toggle-grid">
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.cache}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        cache: event.target.checked,
                      }))
                    }
                  />
                  Enable cache
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.sanityReport}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        sanityReport: event.target.checked,
                      }))
                    }
                  />
                  Sanity report
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.sanityDrop}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        sanityDrop: event.target.checked,
                      }))
                    }
                  />
                  Drop rows failing sanity
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.verboseSkips}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        verboseSkips: event.target.checked,
                      }))
                    }
                  />
                  Verbose skips
                </label>
              </div>
            </details>

            {runError ? <div className="error">{runError}</div> : null}

            <div className="actions">
              <button
                className="button primary"
                type="submit"
                disabled={isRunning || anyJobRunning}
              >
                {isRunning ? "Building dataset..." : "Run dataset build"}
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
              Captures stdout/stderr from the dataset script.
            </span>
          </div>
          <div className="panel-body">
            <div className="run-shell">
              <aside className="run-shell-sidebar">
                <div className="run-progress-panel">
                  <div className="run-progress-heading">
                    <span className="meta-label">Run monitor</span>
                    {jobStatus ? (
                      <span className={`status-pill ${statusClass}`}>
                        {statusLabel}
                      </span>
                    ) : (
                      <span className="status-pill idle">Idle</span>
                    )}
                  </div>
                  {jobStatus ? (
                    <>
                      <div className="run-progress-grid">
                        <div>
                          <span className="meta-label">Weeks</span>
                          <span>{plannedWeeksLabel}</span>
                        </div>
                        <div>
                          <span className="meta-label">Tickers</span>
                          <span>
                            {resolvedTickersCount.toLocaleString()}
                            {customTickerCount ? ` (${customTickerCount} custom)` : ""}
                          </span>
                        </div>
                        <div>
                          <span className="meta-label">Snapshot days</span>
                          <span>{snapshotCountLabel} per expiry</span>
                        </div>
                        <div>
                          <span className="meta-label">Planned jobs</span>
                          <span>{plannedJobsLabel}</span>
                        </div>
                        <div>
                          <span className="meta-label">Progress</span>
                          <span>
                            {jobProgress
                              ? `${jobProgress.done.toLocaleString()}/${jobProgress.total.toLocaleString()} (${progressPercent ?? 0}%)`
                              : "Waiting for progress…"}
                          </span>
                        </div>
                        <div>
                          <span className="meta-label">Groups kept</span>
                          <span>
                            {jobProgress ? jobProgress.groups.toLocaleString() : "—"}
                          </span>
                        </div>
                        <div>
                          <span className="meta-label">Rows</span>
                          <span>
                            {jobProgress ? jobProgress.rows.toLocaleString() : "—"}
                          </span>
                        </div>
                        <div>
                          <span className="meta-label">Last job</span>
                          <span>
                            {jobProgress
                              ? `${jobProgress.lastTicker} · ${jobProgress.lastWeek} · ${jobProgress.lastAsof}`
                              : "—"}
                          </span>
                        </div>
                      </div>
                      <div className="progress-tracker">
                        <div className="progress-label">
                          <span>Progress</span>
                          <span>
                            {progressPercent !== null ? `${progressPercent}%` : "—"}
                          </span>
                        </div>
                        <div className="progress-bar">
                          <div
                            className="progress-fill"
                            style={{ width: `${progressPercent ?? 0}%` }}
                          />
                        </div>
                      </div>
                      <p className="run-feedback-note">
                        Progress updates print every 100 jobs in stdout.
                      </p>
                      {(jobStatus.status === "running" ||
                        jobStatus.status === "queued") ? (
                        <button
                          className="button ghost kill-button"
                          type="button"
                          disabled={killLoading}
                          onClick={handleKill}
                        >
                          {killLoading ? "Stopping…" : "Stop run"}
                        </button>
                      ) : null}
                    </>
                  ) : (
                    <div className="run-progress-empty">
                      <p className="meta-label">No dataset build running</p>
                      <p>Start a run to see progress here.</p>
                    </div>
                  )}
                </div>
              </aside>
              <div className="run-shell-main">
                {!jobStatus ? (
                  <div className="empty">No dataset run yet.</div>
                ) : (
                  <div className="run-output">
                    <div className="run-summary">
                      <div className="run-summary-header">
                        <div>
                          <span className="meta-label">Output</span>
                          <div className="run-id">
                            {currentResult?.output_file ??
                              `${currentResult?.out_dir ?? formState.outDir}/${datasetNameKebab || "(pending)"}`}
                          </div>
                        </div>
                        <div className="run-summary-actions">
                          <span className={`status-pill ${statusClass}`}>
                            {statusLabel}
                          </span>
                        </div>
                      </div>
                      <div className="run-meta-grid">
                        <div>
                          <span className="meta-label">Duration</span>
                          <span>
                            {currentResult
                              ? `${currentResult.duration_s.toFixed(2)}s`
                              : jobStatus.status === "running" ||
                                jobStatus.status === "queued"
                              ? "Running"
                              : "Pending"}
                          </span>
                        </div>
                        <div>
                          <span className="meta-label">Output dir</span>
                          <span>{currentResult?.out_dir ?? formState.outDir}</span>
                        </div>
                        <div>
                          <span className="meta-label">Training dataset</span>
                          <span>
                            {trainingDatasetEnabled
                              ? trainingDatasetPath ?? "Pending"
                              : "Not written"}
                          </span>
                        </div>
                        <div>
                          <span className="meta-label">Drops file</span>
                          <span>
                            {currentResult?.drops_file ??
                              (formState.writeDrops ? "Pending" : "Not written")}
                          </span>
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
                          ? stdoutText || "No stdout captured."
                          : stderrText || "No stderr captured."}
                      </pre>
                    </div>
                    {currentResult?.command ? (
                      <details className="command-details">
                        <summary>Command used</summary>
                        <code>{currentResult.command.join(" ")}</code>
                      </details>
                    ) : null}
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>
      </div>

      <section className="panel dataset-registry-panel">
        <div className="panel-header">
          <div>
            <h2>Dataset registry</h2>
            <span className="panel-hint">
              Preview any CSV, rename the run directory, and clean out stale exports.
            </span>
          </div>
        </div>
        <div className="panel-body dataset-registry-body">
          <div className="dataset-runs-list">
            {runsLoading ? (
              <div className="empty">Loading dataset exports…</div>
            ) : runsError ? (
              <div className="error">{runsError}</div>
            ) : datasetRuns.length === 0 ? (
              <div className="empty">
                No dataset exports yet. Run the builder to create a CSV snapshot.
              </div>
            ) : (
              datasetRuns.map((run) => {
                const runName = run.run_dir.split("/").pop() ?? run.id;
                const trainingFile = run.training_file ?? null;
                const trainingPath = trainingFile?.path ?? null;
                const files = sortRunFiles(buildRunFiles(run), trainingPath);
                const fileCount = files.length;
                const filesLabel = fileCount
                  ? `${fileCount} CSV${fileCount === 1 ? "" : "s"}`
                  : "No CSV files";
                const trainingLabel = trainingFile
                  ? `Training: ${trainingFile.name}`
                  : "Training file missing";
                const isRenaming = renamingRunId === run.id;
                return (
                  <article key={run.id} className="dataset-run-item">
                    <div className="dataset-run-main">
                      {isRenaming ? (
                        <div className="rename-input-wrapper">
                          <input
                            className="input rename-input"
                            type="text"
                            value={renameValue}
                            onChange={(event) =>
                              setRenameValue(event.target.value)
                            }
                            onKeyDown={(event) => {
                              if (event.key === "Enter") {
                                handleConfirmRename(run.id, run.run_dir);
                              } else if (event.key === "Escape") {
                                handleCancelRename();
                              }
                            }}
                            autoFocus
                          />
                          {renameError ? (
                            <div className="error">{renameError}</div>
                          ) : null}
                          <div className="rename-actions">
                            <button
                              className="button ghost small"
                              type="button"
                              onClick={() =>
                                handleConfirmRename(run.id, run.run_dir)
                              }
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
                      ) : (
                        <div className="dataset-run-title">{runName}</div>
                      )}
                      <div className="dataset-run-meta">
                        <span>{formatTimestamp(run.last_modified)}</span>
                        <span>{filesLabel}</span>
                        <span>{trainingLabel}</span>
                      </div>
                      <div className="dataset-run-path">{run.run_dir}</div>
                    </div>
                    <div className="dataset-run-actions">
                      <button
                        type="button"
                        className="button light small"
                        onClick={() => handleStartRename(run.id, runName)}
                        disabled={isRenaming || renameLoading}
                      >
                        Rename dataset
                      </button>
                      <button
                        type="button"
                        className="button ghost danger small"
                        onClick={() => setDeleteConfirmRun(run.id)}
                        disabled={deleteLoadingRun === run.id}
                      >
                        {deleteLoadingRun === run.id
                          ? "Deleting…"
                          : "Delete dataset"}
                      </button>
                      {deleteConfirmRun === run.id ? (
                        <div className="dataset-delete-confirm inline">
                          <p>
                            Permanently delete <strong>{runName}</strong> and all its files?
                          </p>
                          <div className="dataset-delete-actions">
                            <button
                              type="button"
                              className="button danger small"
                              onClick={() => handleDeleteRun(run.run_dir)}
                              disabled={deleteLoadingRun === run.id}
                            >
                              {deleteLoadingRun === run.id
                                ? "Deleting…"
                                : "Delete"}
                            </button>
                            <button
                              type="button"
                              className="button ghost small"
                              onClick={() => setDeleteConfirmRun(null)}
                              disabled={deleteLoadingRun === run.id}
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                      ) : null}
                    </div>
                    <div className="dataset-run-files">
                      {files.length === 0 ? (
                        <div className="dataset-run-files-empty">
                          No CSV files found in this run.
                        </div>
                      ) : (
                        files.map((file) => {
                          const isTraining = trainingPath === file.path;
                          return (
                            <div key={file.path} className="dataset-run-file">
                              <div className="dataset-run-file-info">
                                <div className="dataset-run-file-name">
                                  {file.name}
                                </div>
                                <div className="dataset-run-file-meta">
                                  {isTraining ? (
                                    <span className="dataset-run-file-tag">
                                      Training
                                    </span>
                                  ) : null}
                                  <span>{formatByteCount(file.size_bytes)}</span>
                                </div>
                              </div>
                              <div className="dataset-run-file-actions">
                                <button
                                  type="button"
                                  className="button light small"
                                  onClick={() =>
                                    handlePreviewSelection({
                                      label: `${file.name}${isTraining ? " (training)" : ""}`,
                                      path: file.path,
                                    })
                                  }
                                >
                                  Preview
                                </button>
                                <a
                                  href={getDatasetFileUrl(file.path)}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="button light small"
                                >
                                  Open
                                </a>
                              </div>
                            </div>
                          );
                        })
                      )}
                    </div>
                  </article>
                );
              })
            )}
          </div>
          <div className="dataset-preview-panel">
            <div className="dataset-preview-header">
              <div>
                <span className="meta-label">CSV preview</span>
                <p className="dataset-preview-title">
                  {previewTarget?.label ??
                    "Select a CSV to peek at its rows."}
                </p>
              </div>
            </div>
            {previewLoading ? (
              <div className="dataset-preview-empty">Loading preview…</div>
            ) : previewError ? (
              <div className="error">{previewError}</div>
            ) : previewResponse ? (
              <>
                {previewResponse.headers.length > 0 ? (
                  <div className="table-container">
                    <table className="preview-table">
                      <thead>
                        <tr>
                          {previewResponse.headers.map((column) => (
                            <th key={column}>{column}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {previewResponse.rows.length > 0 ? (
                          previewResponse.rows.map((row, index) => (
                            <tr key={index}>
                              {previewResponse.headers.map((column) => (
                                <td key={column}>{row[column] ?? ""}</td>
                              ))}
                            </tr>
                          ))
                        ) : (
                          <tr>
                            <td
                              colSpan={
                                previewResponse.headers.length || 1
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
                  <div className="dataset-preview-empty">
                    CSV preview did not include column headers.
                  </div>
                )}
                <div className="dataset-preview-meta">
                  <span className="meta-label">
                    Showing {previewResponse.mode} ({previewResponse.limit} rows)
                  </span>
                  <span>
                    {previewResponse.row_count
                      ? `${previewResponse.row_count.toLocaleString()} total rows`
                      : "Row count unknown"}
                  </span>
                </div>
              </>
            ) : (
              <div className="dataset-preview-empty">
                Select a file and click preview to inspect the CSV contents.
              </div>
            )}
          </div>
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
