import { useEffect, useMemo, useState, type FormEvent } from "react";

import {
  killDatasetJob,
  startDatasetJob,
  type DatasetJobStatus,
  type DatasetRunResponse,
} from "../api/datasets";
import { useDatasetJob } from "../contexts/datasetJob";
import "./DatasetsPage.css";

type DatasetFormState = {
  outDir: string;
  outName: string;
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

const defaultForm: DatasetFormState = {
  outDir: "src/data/raw/option-chain",
  outName: "options-chain-dataset.csv",
  tickers: "AAPL, GOOGL, MSFT, META, AMZN, PLTR, NVDA, NFLX, OPEN, TSLA",
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

const DEFAULT_OUT_NAME = defaultForm.outName;

const trimCsvStem = (value: string) =>
  value.replace(/\.csv$/i, "").trim() || value.trim();

const deriveDropsName = (outName: string) =>
  `${trimCsvStem(outName)}-drops.csv`;

const STORAGE_KEY = "polyedgetool.datasets.form";

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
  addValue("--out-name", state.outName);
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
  if (state.writeDrops) {
    const effectiveOutName = state.outName.trim() || DEFAULT_OUT_NAME;
    addValue("--drops-name", deriveDropsName(effectiveOutName));
  }

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
  const [killLoading, setKillLoading] = useState(false);
  const defaultTickers = useMemo(
    () => parseTickers(defaultForm.tickers) ?? [],
    [],
  );

  const tickersList = useMemo(
    () => parseTickers(formState.tickers),
    [formState.tickers],
  );
  const resolvedTickersCount =
    tickersList?.length ?? defaultTickers.length;
  const isUsingDefaultTickers = !tickersList;
  const weeksCount = useMemo(
    () => countMondaysInRange(formState.start, formState.end),
    [formState.start, formState.end],
  );
  const snapshotsPerWeek = 4;
  const plannedJobs =
    weeksCount && resolvedTickersCount
      ? weeksCount * snapshotsPerWeek * resolvedTickersCount
      : null;
  const commandPreview = useMemo(
    () => buildCommandPreview(formState),
    [formState],
  );

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

  const resolvedRange =
    formState.start && formState.end
      ? `${formState.start} → ${formState.end}`
      : "Select a date range";
  const resolvedTickersLabel = tickersList
    ? `${tickersList.length} tickers`
    : `Using script defaults (${defaultTickers.length})`;
  const plannedJobsLabel = plannedJobs
    ? `${plannedJobs.toLocaleString()} jobs`
    : "Set a date range";
  const plannedWeeksLabel =
    weeksCount !== null ? `${weeksCount} weeks` : "Weeks pending";
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

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setRunError(null);
    setRunResult(null);
    setGlobalJobStatus(null);
    setJobId(null);
    setActiveLog("stdout");
    setIsRunning(true);

    try {
      const payload = {
        outDir: formState.outDir.trim() || undefined,
        outName: formState.outName.trim() || undefined,
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

  return (
    <section className="page datasets-page">
      <header className="page-header">
        <div>
          <p className="page-kicker">Database</p>
          <h1 className="page-title">Build the options dataset</h1>
          <p className="page-subtitle">
            Generate the historic options chain dataset and track every CLI
            input in one place.
          </p>
        </div>
        <div className="meta-card datasets-meta">
          <span className="meta-label">Pipeline stage</span>
          <span>1-option-chain-build-historic-dataset-v1.0.py</span>
          <div className="meta-pill">CSV + optional drops report</div>
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
              <span className="meta-label">Output target</span>
              <span>
                {formState.outDir}/{formState.outName}
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
                <label htmlFor="datasetTickers">
                  Tickers (comma-separated)
                </label>
                <input
                  id="datasetTickers"
                  className="input"
                  value={formState.tickers}
                  onChange={(event) =>
                    setFormState((prev) => ({
                      ...prev,
                      tickers: event.target.value,
                    }))
                  }
                />
                <span className="field-hint">
                  Leave blank to use the PM10 defaults from the script.
                </span>
              </div>
            </div>

            <div className="section-card dataset-section">
              <h3>Output targets</h3>
              <div className="inline-fields">
                <div className="field">
                  <label>Output directory</label>
                  <div className="input readonly">{formState.outDir}</div>
                </div>
            <div className="field">
              <label htmlFor="outName">Dataset filename</label>
              <div
                className={`input-with-default${
                  formState.outName ? " filled" : ""
                }`}
                data-default={DEFAULT_OUT_NAME}
              >
                <input
                  id="outName"
                  className="input"
                  value={formState.outName}
                  onChange={(event) =>
                    setFormState((prev) => ({
                      ...prev,
                      outName: event.target.value,
                    }))
                  }
                />
              </div>
            </div>
              </div>
              <div className="inline-fields">
            <div className="field">
              <label>Write drops report</label>
                  <label className="checkbox">
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
                    Enable <code>--write-drops</code>
                  </label>
                </div>
              </div>
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
                disabled={isRunning}
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
                              `${currentResult?.out_dir ?? formState.outDir}/${currentResult?.out_name ?? formState.outName}`}
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
                            {isUsingDefaultTickers ? " (default)" : ""}
                          </span>
                        </div>
                        <div>
                          <span className="meta-label">Snapshot days</span>
                          <span>{snapshotsPerWeek} per week</span>
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
            </div>
          </div>
        </section>
      </div>

      <section className="panel">
        <div className="panel-header">
          <h2>CLI preview</h2>
          <span className="panel-hint">
            Mirrors the exact arguments that will be sent to the script.
          </span>
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
