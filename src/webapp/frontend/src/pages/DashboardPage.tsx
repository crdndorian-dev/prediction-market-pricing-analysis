import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import type { DashboardData } from "../api/dashboard";
import { fetchDashboard } from "../api/dashboard";
import "./DashboardPage.css";

export default function DashboardPage() {
  const [runQueue, setRunQueue] = useState<DashboardData["runQueue"]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;

    fetchDashboard()
      .then((payload) => {
        if (!isMounted) return;
        setRunQueue(payload.runQueue ?? []);
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

  return (
    <section className="page dashboard">
      {isLoading ? (
        <div className="dashboard-banner">Loading run queue...</div>
      ) : null}

      <section className="dashboard-hero reveal delay-1">
        <div className="hero-content">
          <span className="hero-eyebrow">Prediction market pricing pipeline</span>
          <h1>Benchmark Polymarket prices with option-implied probabilities.</h1>
          <p className="hero-lede">
            Run ingestion, calibration, and analysis locally to generate fair-value
            probability estimates and audit the comparison.
          </p>
          <div className="hero-actions">
            <Link className="button primary" to="/option-chain">
              Build option chain
            </Link>
            <Link className="button light" to="/polymarket-pipeline">
              Run history builder
            </Link>
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
                <p>Setup steps, run order, and output definitions.</p>
              </div>
              <span className="status-pill idle">Guides</span>
            </div>
            <p className="doc-paragraph">
              Open the docs for quick setup, pipeline steps, and output notes.
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
