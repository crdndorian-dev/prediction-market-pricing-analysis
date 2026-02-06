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

const formatCount = (value?: number | null) => {
  if (value === null || value === undefined) return "--";
  return value.toLocaleString();
};

const formatSize = (value?: number | null) => {
  if (value === null || value === undefined) return "--";
  return `${value.toFixed(2)} MB`;
};

const statusVariant = (status: string) => {
  const normalized = status.toLowerCase();
  if (normalized === "ready" || normalized === "success") return "success";
  if (normalized === "running" || normalized === "queued") return "running";
  if (normalized === "missing" || normalized === "needs review" || normalized === "failed") {
    return "failed";
  }
  return "idle";
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
  const runQueue = data?.runQueue ?? [];
  const snapshot = data?.calibrationSnapshot;
  const datasetSummary = data?.datasetSummary ?? null;
  const polymarketSummary = data?.polymarketSummary ?? null;
  const phatEdgeSummary = data?.phatEdgeSummary ?? null;
  const modelSummary = data?.modelSummary ?? null;
  const latestModel = modelSummary?.latestModel ?? null;

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

  const polymarketStatus =
    polymarketSummary?.latestRunId || polymarketSummary?.latestSnapshotDate
      ? "Ready"
      : "Missing";
  const polymarketDescription = polymarketSummary?.latestRunId
    ? `Latest run: ${polymarketSummary.latestRunId}`
    : polymarketSummary?.latestSnapshotDate
      ? `Latest snapshot: ${formatDate(polymarketSummary.latestSnapshotDate)}`
      : "No snapshots captured yet.";
  const polymarketMetaParts = [
    polymarketSummary?.fileCount
      ? `${formatCount(polymarketSummary.fileCount)} files`
      : null,
    polymarketSummary?.sizeMB !== null && polymarketSummary?.sizeMB !== undefined
      ? formatSize(polymarketSummary.sizeMB)
      : null,
  ].filter(Boolean);
  const polymarketMeta =
    polymarketMetaParts.length > 0
      ? polymarketMetaParts.join(" · ")
      : "Run snapshot to generate files.";
  const polymarketFileHint =
    polymarketSummary?.datasetFile ??
    polymarketSummary?.ppmFile ??
    polymarketSummary?.prnFile ??
    "Outputs stored under src/data/raw/polymarket.";

  const phatEdgeStatus = phatEdgeSummary ? "Ready" : "Missing";
  const phatEdgeDescription = phatEdgeSummary
    ? `Latest output: ${phatEdgeSummary.fileName}`
    : "No Edge output yet.";
  const phatEdgeMeta = phatEdgeSummary
    ? `${formatCount(phatEdgeSummary.rowCount)} rows · ${formatDateTime(
        phatEdgeSummary.lastModified,
      )}`
    : "Run inference to generate an Edge CSV.";
  const phatEdgeTop =
    phatEdgeSummary?.maxEdgeTicker && phatEdgeSummary.maxEdge !== null
      ? `Top edge: ${phatEdgeSummary.maxEdgeTicker} (${formatMetric(
          phatEdgeSummary.maxEdge,
          4,
        )})`
      : "Top edge not available yet.";

  const stageCards = [
    {
      key: "dataset",
      title: "Option chain build",
      status: datasetSummary ? "Ready" : "Missing",
      description: datasetSummary
        ? `${datasetSummary.fileName} - ${formatCount(
            datasetSummary.rowCount,
          )} rows`
        : "No dataset snapshot detected yet.",
      meta: datasetSummary
        ? `Updated ${formatDateTime(datasetSummary.lastModified)}`
        : "Run dataset builder to generate a CSV.",
      to: "/option-chain",
    },
    {
      key: "snapshots",
      title: "Polymarket",
      status: polymarketStatus,
      description: polymarketDescription,
      meta: polymarketMeta,
      hint: polymarketFileHint,
      to: "/polymarket",
    },
    {
      key: "calibration",
      title: "Calibrate models",
      status: modelSummary?.modelCount ? "Ready" : "Missing",
      description: latestModel?.id
        ? `Latest model: ${latestModel.id}`
        : "No calibration models found.",
      meta: latestModel?.modifiedAt
        ? `Updated ${formatDateTime(latestModel.modifiedAt)}`
        : "Run calibration to produce a model artifact.",
      to: "/calibrate-models",
    },
    {
      key: "edge",
      title: "Edge",
      status: phatEdgeStatus,
      description: phatEdgeDescription,
      meta: phatEdgeMeta,
      hint: phatEdgeTop,
      to: "/edge",
    },
  ];


  return (
    <section className="page dashboard">
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
            <Link className="button primary" to="/option-chain">
              Build option chain
            </Link>
            <Link className="button light" to="/polymarket">
              Run Polymarket snapshot
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
              <Link className="button ghost" to="/option-chain">
                Open run monitor
              </Link>
            </div>
          </div>
        </div>
      </section>

      <section className="dashboard-overview reveal delay-2">
          <div className="section-heading">
            <div>
              <h2>Pipeline overview</h2>
              <p>Jump into each stage and track readiness.</p>
            </div>
          </div>
        <div className="stage-grid">
          {stageCards.map((stage) => (
            <div key={stage.key} className="stage-card">
              <div className="stage-card-header">
                <h3 className="stage-title">{stage.title}</h3>
                <span
                  className={`status-pill ${statusVariant(stage.status)}`}
                >
                  {stage.status}
                </span>
              </div>
              <p className="stage-description">{stage.description}</p>
              <div className="stage-meta">{stage.meta}</div>
              {"hint" in stage && stage.hint ? (
                <div className="stage-hint">{stage.hint}</div>
              ) : null}
              <div className="stage-actions">
                <Link className="button ghost" to={stage.to}>
                  Open stage
                </Link>
              </div>
            </div>
          ))}
        </div>
      </section>

    </section>
  );
}
