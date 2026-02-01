import { useEffect, useMemo, useState, type FormEvent } from "react";

import {
  fetchPolymarketSnapshotRuns,
  fetchLatestPolymarketSnapshot,
  fetchPolymarketSnapshotHistory,
  fetchPolymarketSnapshotPreview,
  runPolymarketSnapshot,
  type PolymarketSnapshotFileSummary,
  type PolymarketSnapshotHistoryResponse,
  type PolymarketSnapshotLatestResponse,
  type PolymarketSnapshotPreviewResponse,
  type PolymarketSnapshotRunResponse,
  type PolymarketSnapshotRunSummary,
} from "../api/polymarketSnapshots";
import "./PolymarketSnapshotsPage.css";

type RunFormState = {
  tickers: string;
  tickersCsv: string;
  slugOverrides: string;
  riskFreeRate: string;
  tz: string;
  keepNonexec: boolean;
};

const HARD_CODED_RISK_FREE_RATE = "0.03";

const defaultForm: RunFormState = {
  tickers: "",
  tickersCsv: "",
  slugOverrides: "",
  riskFreeRate: HARD_CODED_RISK_FREE_RATE,
  tz: "Europe/Paris",
  keepNonexec: false,
};

const STORAGE_KEY = "polyedgetool.polymarket.form";

function parseTickers(raw: string): string[] | undefined {
  const cleaned = raw
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  return cleaned.length > 0 ? cleaned : undefined;
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

export default function PolymarketSnapshotsPage() {
  const [formState, setFormState] = useState<RunFormState>(defaultForm);
  const [isRunning, setIsRunning] = useState(false);
  const [runs, setRuns] = useState<PolymarketSnapshotRunSummary[]>([]);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [runResult, setRunResult] =
    useState<PolymarketSnapshotRunResponse | null>(null);
  const [activeLog, setActiveLog] = useState<"stdout" | "stderr">("stdout");
  const [latestSnapshot, setLatestSnapshot] =
    useState<PolymarketSnapshotLatestResponse | null>(null);
  const [latestError, setLatestError] = useState<string | null>(null);
  const [latestPreview, setLatestPreview] =
    useState<PolymarketSnapshotPreviewResponse | null>(null);
  const [activeSnapshotFile, setActiveSnapshotFile] =
    useState<PolymarketSnapshotFileSummary | null>(null);
  const [history, setHistory] =
    useState<PolymarketSnapshotHistoryResponse | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [historyPreview, setHistoryPreview] =
    useState<PolymarketSnapshotPreviewResponse | null>(null);
  const [activeHistoryFile, setActiveHistoryFile] =
    useState<PolymarketSnapshotFileSummary | null>(null);
  const [storageReady, setStorageReady] = useState(false);

  const tickersList = useMemo(
    () => parseTickers(formState.tickers),
    [formState.tickers],
  );

  useEffect(() => {
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

    fetchLatestPolymarketSnapshot()
      .then((data) => {
        if (!isMounted) return;
        setLatestSnapshot(data);
        setLatestError(null);
      })
      .catch((err: Error) => {
        if (!isMounted) return;
        setLatestError(err.message);
      });

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

  useEffect(() => {
    if (!latestSnapshot || latestSnapshot.files.length === 0) {
      setActiveSnapshotFile(null);
      return;
    }
    const preferred = ["dataset", "pPM", "pRN"];
    const next =
      latestSnapshot.files.find((file) => file.kind === "dataset") ??
      latestSnapshot.files.find((file) => file.kind === "pPM") ??
      latestSnapshot.files.find((file) => file.kind === "pRN") ??
      latestSnapshot.files[0];
    setActiveSnapshotFile(next);
  }, [latestSnapshot]);

  useEffect(() => {
    if (!activeSnapshotFile) {
      setLatestPreview(null);
      return;
    }
    let isMounted = true;
    fetchPolymarketSnapshotPreview(activeSnapshotFile.path, 18, "head")
      .then((data) => {
        if (!isMounted) return;
        setLatestPreview(data);
        setLatestError(null);
      })
      .catch((err: Error) => {
        if (!isMounted) return;
        setLatestError(err.message);
      });
    return () => {
      isMounted = false;
    };
  }, [activeSnapshotFile]);

  useEffect(() => {
    if (!history || history.files.length === 0) {
      setActiveHistoryFile(null);
      return;
    }
    setActiveHistoryFile(history.files[0]);
  }, [history]);

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

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setRunError(null);
    setRunResult(null);
    setIsRunning(true);
    try {
      const payload = {
        tickers: tickersList,
        tickersCsv: formState.tickersCsv.trim() || undefined,
        slugOverrides: formState.slugOverrides.trim() || undefined,
        riskFreeRate: Number(HARD_CODED_RISK_FREE_RATE),
        tz: formState.tz.trim() || undefined,
        keepNonexec: formState.keepNonexec,
      };
      const result = await runPolymarketSnapshot(payload);
      setRunResult(result);
      const list = await fetchPolymarketSnapshotRuns();
      setRuns(list.runs);
      const latest = await fetchLatestPolymarketSnapshot();
      setLatestSnapshot(latest);
      if (latest.history_file) {
        setHistory((prev) =>
          prev
            ? { files: [latest.history_file, ...prev.files] }
            : { files: [latest.history_file] },
        );
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setRunError(message);
    } finally {
      setIsRunning(false);
    }
  };

  const resolvedTickersLabel = tickersList
    ? `${tickersList.length} selected`
    : "Using script defaults";

  const latestFiles = latestSnapshot?.files ?? [];
  const historyFiles = history?.files ?? [];

  return (
    <section className="page polymarket-page">
      <header className="page-header">
        <div>
          <p className="page-kicker">Polymarket snapshots</p>
          <h1 className="page-title">Fetch a new weekly snapshot</h1>
          <p className="page-subtitle">
            Execute the snapshot ingestion script and store outputs under
            <code>src/data/raw/polymarket</code>.
          </p>
        </div>
        <div className="meta-card polymarket-meta">
          <span className="meta-label">Defaults</span>
          <span>Finish-week contracts, pRN merge, rolling history append.</span>
        </div>
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
              <span className="meta-label">Risk-free rate</span>
              <span>{HARD_CODED_RISK_FREE_RATE} (locked)</span>
            </div>
            <div>
              <span className="meta-label">Timezone</span>
              <span>{formState.tz || "Default"}</span>
            </div>
          </div>
          <form className="panel-body" onSubmit={handleSubmit}>
            <div className="field">
              <label htmlFor="tickers">Tickers (comma-separated)</label>
              <input
                id="tickers"
                className="input"
                placeholder="NVDA, AAPL, MSFT"
                value={formState.tickers}
                onChange={(event) =>
                  setFormState((prev) => ({
                    ...prev,
                    tickers: event.target.value,
                  }))
                }
              />
              <span className="field-hint">
                If provided, overrides any tickers CSV.
              </span>
            </div>

            <div className="inline-fields">
              <div className="field">
                <label htmlFor="riskFreeRate">Risk-free rate</label>
                <input
                  id="riskFreeRate"
                  className="input"
                  inputMode="decimal"
                  value={HARD_CODED_RISK_FREE_RATE}
                  disabled
                />
                <span className="field-hint locked-hint">
                  Locked for consistency. Change requires a code update.
                </span>
              </div>
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
            </div>

            <details className="advanced">
              <summary>Advanced inputs</summary>
              <div className="field">
                <label htmlFor="tickersCsv">Tickers CSV (relative path)</label>
                <input
                  id="tickersCsv"
                  className="input"
                  placeholder="src/data/tickers.csv"
                  value={formState.tickersCsv}
                  onChange={(event) =>
                    setFormState((prev) => ({
                      ...prev,
                      tickersCsv: event.target.value,
                    }))
                  }
                />
              </div>
              <div className="field">
                <label htmlFor="slugOverrides">
                  Slug overrides (relative path)
                </label>
                <input
                  id="slugOverrides"
                  className="input"
                  placeholder="src/data/polymarket/slugs.json"
                  value={formState.slugOverrides}
                  onChange={(event) =>
                    setFormState((prev) => ({
                      ...prev,
                      slugOverrides: event.target.value,
                    }))
                  }
                />
              </div>
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={formState.keepNonexec}
                  onChange={(event) =>
                    setFormState((prev) => ({
                      ...prev,
                      keepNonexec: event.target.checked,
                    }))
                  }
                />
                Keep pm_ok=false rows
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
              Captures stdout/stderr from the snapshot script.
            </span>
          </div>
          <div className="panel-body">
            {!runResult ? (
              <div className="empty">No run output yet.</div>
            ) : (
              <div className="run-output">
                <div className="run-summary">
                  <div className="run-summary-header">
                    <div>
                      <span className="meta-label">Run ID</span>
                      <div className="run-id">
                        {runResult.run_id ?? "unknown"}
                      </div>
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
                      <span className="meta-label">Output</span>
                      <span>
                        {runResult.run_dir ?? runResult.out_dir ?? "Unknown"}
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
          <h2>Recent snapshot runs</h2>
          <span className="panel-hint">
            Stored under <code>src/data/polymarket/runs</code>.
          </span>
        </div>
        <div className="panel-body">
          {runsError ? <div className="error">{runsError}</div> : null}
          {runs.length === 0 ? (
            <div className="empty">No runs found.</div>
          ) : (
            <div className="runs-list">
              {runs.map((run, index) => (
                <div
                  key={run.run_id}
                  className={`run-card ${index === 0 ? "latest" : ""}`}
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
          )}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Latest snapshot datasets</h2>
          <span className="panel-hint">
            Quick preview of the latest snapshot files.
          </span>
        </div>
        <div className="panel-body">
          {latestError ? <div className="error">{latestError}</div> : null}
          {latestFiles.length === 0 ? (
            <div className="empty">No snapshot datasets found yet.</div>
          ) : (
            <>
              <div className="file-tabs">
                {latestFiles.map((file) => (
                  <button
                    key={file.path}
                    type="button"
                    className={`file-tab ${
                      activeSnapshotFile?.path === file.path ? "active" : ""
                    }`}
                    onClick={() => setActiveSnapshotFile(file)}
                  >
                    <span className="file-label">
                      {file.kind ?? "snapshot"}
                    </span>
                    <span className="file-name">{file.name}</span>
                  </button>
                ))}
              </div>
              {latestPreview ? (
                <div className="table-container">
                  <table className="preview-table">
                    <thead>
                      <tr>
                        {latestPreview.headers.map((header) => (
                          <th key={header}>{header}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {latestPreview.rows.map((row, idx) => (
                        <tr key={`${latestPreview.file.path}-${idx}`}>
                          {latestPreview.headers.map((header) => (
                            <td key={`${header}-${idx}`}>
                              {row[header] ?? ""}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="empty">Loading preview…</div>
              )}
            </>
          )}
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Snapshot history</h2>
          <span className="panel-hint">
            Rolling history preview from <code>src/data/polymarket/history</code>.
          </span>
        </div>
        <div className="panel-body">
          {historyError ? <div className="error">{historyError}</div> : null}
          {historyFiles.length === 0 ? (
            <div className="empty">No history files found yet.</div>
          ) : (
            <>
              <div className="file-tabs">
                {historyFiles.map((file) => (
                  <button
                    key={file.path}
                    type="button"
                    className={`file-tab ${
                      activeHistoryFile?.path === file.path ? "active" : ""
                    }`}
                    onClick={() => setActiveHistoryFile(file)}
                  >
                    <span className="file-label">history</span>
                    <span className="file-name">{file.name}</span>
                  </button>
                ))}
              </div>
              {historyPreview ? (
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
              ) : (
                <div className="empty">Loading history…</div>
              )}
            </>
          )}
        </div>
      </section>
    </section>
  );
}
