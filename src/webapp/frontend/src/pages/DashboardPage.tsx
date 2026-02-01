import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { DashboardData, fetchDashboard } from "../api/dashboard";
import "./DashboardPage.css";

const formatDate = (value?: string | null) => {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
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

const formatMetric = (value?: number | null, digits = 3) => {
  if (value === null || value === undefined) return "--";
  return value.toFixed(digits);
};

export default function DashboardPage() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;

    fetchDashboard()
      .then((payload) => {
        if (!isMounted) return;
        setData(payload);
        setError(null);
      })
      .catch((err) => {
        if (!isMounted) return;
        setError(err instanceof Error ? err.message : "Failed to load");
      })
      .finally(() => {
        if (!isMounted) return;
        setIsLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const hero = data?.hero;
  const readiness = data?.readiness ?? [];
  const runQueue = data?.runQueue ?? [];
  const recentRuns = data?.recentRuns ?? [];
  const snapshot = data?.calibrationSnapshot;
  const signalBars = data?.signalBars ?? [];
  const readinessState = readiness.length
    ? readiness.every((item) => item.status === "Ready")
      ? "Ready"
      : "Needs review"
    : "Awaiting data";
  const readinessStateClass =
    readinessState === "Ready" ? "status-ready" : "status-warn";

  const freshnessDays =
    hero?.dataFreshnessDays !== null && hero?.dataFreshnessDays !== undefined
      ? hero.dataFreshnessDays
      : null;
  const dayUnit = freshnessDays === 1 ? "day" : "days";
  const dataFreshnessLabel =
    freshnessDays !== null
      ? `${freshnessDays} ${dayUnit} since last snapshot`
      : "No dataset timestamp yet";
  const dataFreshnessDate = hero?.dataFreshnessDate
    ? formatDate(hero.dataFreshnessDate)
    : "Awaiting data";
  const dataSourceLabel = hero?.dataSourceLabel ?? "Dataset not detected";

  const calibrationLabel =
    snapshot?.ece !== null && snapshot?.ece !== undefined
      ? `ECE ${formatMetric(snapshot.ece, 3)}`
      : "No metrics yet";
  const calibrationSub = snapshot?.model
    ? `Model: ${snapshot.model}`
    : "Run calibration to populate";

  const lastRunLabel = hero?.lastRunTime
    ? formatDateTime(hero.lastRunTime)
    : "No runs recorded";
  const lastRunSub = hero?.lastRunSummary ?? "Start a run to capture outputs";

  const quickActions = [
    {
      title: "Run full pipeline",
      description: "Ingestion -> Calibration -> Analysis using the last saved configuration.",
      meta: hero?.lastRunId
        ? `Last run: ${hero.lastRunId}`
        : "No previous run detected",
      to: "/pipeline",
      tone: "primary",
    },
    {
      title: "Compare pRN, pHAT, pPM",
      description: "Line up model outputs and calibration deltas side by side.",
      meta: recentRuns.length
        ? `Compare ${recentRuns[0].id} vs ${recentRuns[1]?.id ?? "another"}`
        : "Choose any two runs",
      to: "/results",
      tone: "default",
    },
    {
      title: "Inspect datasets",
      description: "Review snapshots, coverage windows, and data health checks.",
      meta: `${dataSourceLabel} · ${dataFreshnessDate}`,
      to: "/datasets",
      tone: "default",
    },
    {
      title: "Resume last run",
      description: "Continue a paused calibration sweep without resetting inputs.",
      meta: hero?.lastRunId ? `Run ID: ${hero.lastRunId}` : "No active run",
      to: "/results",
      tone: "ghost",
    },
  ];

  return (
    <div className="dashboard">
      {error ? (
        <div className="dashboard-banner error">
          Unable to load dashboard data. Start the backend on port 8000.
        </div>
      ) : null}
      {isLoading ? (
        <div className="dashboard-banner">Loading latest run data...</div>
      ) : null}

      <section className="dashboard-hero reveal delay-1">
        <div className="hero-content">
          <span className="hero-eyebrow">Local pipeline control</span>
          <h1>Run, track, and compare every model pass in one place.</h1>
          <p className="hero-lede">
            Launch ingestion, calibration, and analysis from a single workspace.
            Keep pRN, pHAT, and pPM outputs reproducible and easy to audit.
          </p>
          <div className="hero-actions">
            <Link className="button primary" to="/pipeline">
              Run full pipeline
            </Link>
            <Link className="button light" to="/results">
              Compare latest outputs
            </Link>
          </div>
          <div className="hero-badges">
            <div className="badge-card">
              <div className="badge-label">Data freshness</div>
              <div className="badge-value">{dataFreshnessLabel}</div>
              <div className="badge-sub">
                {dataSourceLabel} · {dataFreshnessDate}
              </div>
            </div>
            <div className="badge-card">
              <div className="badge-label">Calibration health</div>
              <div className="badge-value">{calibrationLabel}</div>
              <div className="badge-sub">{calibrationSub}</div>
            </div>
            <div className="badge-card">
              <div className="badge-label">Last completed run</div>
              <div className="badge-value">{lastRunLabel}</div>
              <div className="badge-sub">{lastRunSub}</div>
            </div>
          </div>
        </div>
        <div className="hero-panel">
          <div className="panel-card">
            <div className="card-header">
              <div>
                <h2>Pipeline readiness</h2>
                <p>Confirm inputs before you run.</p>
              </div>
              <span className={`status-pill ${readinessStateClass}`}>
                {readinessState}
              </span>
            </div>
            <ul className="checklist">
              {readiness.length ? (
                readiness.map((item) => (
                  <li key={item.title}>
                    <div className="checklist-row">
                      <div>
                        <div className="checklist-title">{item.title}</div>
                        <div className="checklist-sub">{item.detail}</div>
                      </div>
                      <span
                        className={`status-pill ${
                          item.status === "Ready"
                            ? "status-ready"
                            : "status-warn"
                        }`}
                      >
                        {item.status}
                      </span>
                    </div>
                    <div className="progress">
                      <span style={{ width: `${item.progress}%` }} />
                    </div>
                  </li>
                ))
              ) : (
                <li className="empty-state">
                  No readiness data available yet.
                </li>
              )}
            </ul>
            <div className="panel-actions">
              <Link className="button ghost" to="/pipeline">
                Review pipeline settings
              </Link>
            </div>
          </div>
          <div className="panel-card">
            <div className="card-header">
              <div>
                <h2>Run queue</h2>
                <p>Watch active and scheduled jobs.</p>
              </div>
              <span className="status-pill status-info">
                {runQueue.length ? `${runQueue.length} jobs` : "Idle"}
              </span>
            </div>
            <ul className="queue">
              {runQueue.length ? (
                runQueue.map((item) => (
                  <li key={item.name}>
                    <div>
                      <div className="queue-title">{item.name}</div>
                      <div className="queue-sub">{item.detail}</div>
                    </div>
                    <span className="queue-state">{item.state}</span>
                  </li>
                ))
              ) : (
                <li className="empty-state">No active jobs.</li>
              )}
            </ul>
            <div className="panel-actions">
              <Link className="button ghost" to="/results">
                Open run monitor
              </Link>
            </div>
          </div>
        </div>
      </section>

      <section className="dashboard-actions reveal delay-2">
        <div className="section-heading">
          <div>
            <h2>Quick actions</h2>
            <p>Jump into the workflows you use most.</p>
          </div>
          <Link className="button ghost" to="/pipeline">
            Open pipeline runner
          </Link>
        </div>
        <div className="action-grid">
          {quickActions.map((action) => (
            <Link
              key={action.title}
              to={action.to}
              className={`action-card ${action.tone}`}
            >
              <div className="action-title">{action.title}</div>
              <div className="action-body">{action.description}</div>
              <div className="action-meta">{action.meta}</div>
            </Link>
          ))}
        </div>
      </section>

      <section className="dashboard-insights reveal delay-3">
        <div className="insight-card">
          <div className="card-header">
            <div>
              <h2>Recent runs</h2>
              <p>Review outputs and log summaries.</p>
            </div>
            <Link className="button ghost" to="/results">
              View all runs
            </Link>
          </div>
          <ul className="run-list">
            {recentRuns.length ? (
              recentRuns.map((run) => (
                <li key={run.id}>
                  <div className="run-main">
                    <div className="run-title">{run.focus}</div>
                    <div className="run-meta">
                      {run.dataset} | {run.id}
                    </div>
                  </div>
                  <div className="run-aside">
                    <span
                      className={`status-pill ${
                        run.status === "Success"
                          ? "status-ready"
                          : "status-warn"
                      }`}
                    >
                      {run.status}
                    </span>
                    <div className="run-time">{formatDateTime(run.time)}</div>
                  </div>
                </li>
              ))
            ) : (
              <li className="empty-state">No runs recorded yet.</li>
            )}
          </ul>
        </div>
        <div className="insight-card">
          <div className="card-header">
            <div>
              <h2>Calibration snapshot</h2>
              <p>Latest metrics pulled from models.</p>
            </div>
            <span className="status-pill status-info">
              {snapshot?.split ? snapshot.split : "Awaiting metrics"}
            </span>
          </div>
          <div className="metric-grid">
            <div className="metric">
              <div className="metric-label">Logloss</div>
              <div className="metric-value">
                {formatMetric(snapshot?.logloss, 3)}
              </div>
              <div className="metric-sub">{snapshot?.model ?? "No data"}</div>
            </div>
            <div className="metric">
              <div className="metric-label">Brier score</div>
              <div className="metric-value">
                {formatMetric(snapshot?.brier, 3)}
              </div>
              <div className="metric-sub">{snapshot?.model ?? "No data"}</div>
            </div>
            <div className="metric">
              <div className="metric-label">ECE</div>
              <div className="metric-value">
                {formatMetric(snapshot?.ece, 3)}
              </div>
              <div className="metric-sub">{snapshot?.model ?? "No data"}</div>
            </div>
          </div>
          <div className="signal-panel">
            <div className="signal-header">
              <div className="signal-title">Calibration stability</div>
              <div className="signal-sub">Last {signalBars.length || 0} runs</div>
            </div>
            {signalBars.length ? (
              <div className="signal-bars">
                {signalBars.map((value, index) => (
                  <span
                    key={`${value}-${index}`}
                    style={{ height: `${value}%` }}
                  />
                ))}
              </div>
            ) : (
              <div className="empty-state">No signal history yet.</div>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
