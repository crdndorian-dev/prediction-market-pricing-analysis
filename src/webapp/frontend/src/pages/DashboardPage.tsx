import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import type { DashboardData } from "../api/dashboard";
import { fetchDashboard } from "../api/dashboard";
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
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;

    fetchDashboard()
      .then((payload) => {
        if (!isMounted) return;
        setData(payload);
      })
      .catch((err) => {
        if (!isMounted) return;
        console.warn(
          "Dashboard data fetch failed:",
          err instanceof Error ? err.message : err
        );
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
  const runQueue = data?.runQueue ?? [];
  const snapshot = data?.calibrationSnapshot;

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



  return (
    <section className="page dashboard">
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
            <Link className="button primary" to="/option-chain">
              Build option chain
            </Link>
            <Link className="button light" to="/polymarket-pipeline">
              Run history builder
            </Link>
          </div>
          <div className="hero-badges">
            <div className="badge-card">
              <div className="badge-label">Data freshness</div>
              <div className="badge-value">{dataFreshnessLabel}</div>
              <div className="badge-sub">
                {dataSourceLabel} - {dataFreshnessDate}
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
                <h2>Run queue</h2>
                <p>Watch active and scheduled jobs.</p>
              </div>
              <span className="status-pill running">
                {runQueue.length ? `${runQueue.length} jobs` : "Idle"}
              </span>
            </div>
            <ul className="queue">
              {runQueue.length ? (
                runQueue.map((item, index) => (
                  <li key={item.jobId ?? `${item.name}-${index}`}>
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
          </div>
          <div className="panel-card doc-card">
            <div className="card-header">
              <div>
                <h2>Documentation</h2>
                <p>Understand every page and run pipelines end-to-end.</p>
              </div>
              <span className="status-pill idle">Guides</span>
            </div>
            <p className="doc-paragraph">
              Use the docs to learn the dashboard flow, queue behavior, and how
              to run the pipeline end-to-end.
            </p>
            <div className="panel-actions">
              <Link className="button primary" to="/docs">
                Open Documentation
              </Link>
            </div>
          </div>
        </div>
      </section>

    </section>
  );
}
