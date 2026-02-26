import { useLayoutEffect, useRef } from "react";
import { BrowserRouter, NavLink, Navigate, Route, Routes } from "react-router-dom";

import "./App.css";
import DashboardPage from "./pages/DashboardPage";
import DatasetsPage from "./pages/DatasetsPage";
import CalibrateModelsPage from "./pages/CalibrateModelsPage";
import PolymarketPipelinePage from "./pages/PolymarketPipelinePage";
import BacktestsPage from "./pages/BacktestsPage";
import MarketsPage from "./pages/MarketsPage";
import DocumentationPage from "./pages/DocumentationPage";
import { DatasetJobProvider } from "./contexts/datasetJob";
import { CalibrationJobProvider } from "./contexts/calibrationJob";
import { PolymarketHistoryJobProvider } from "./contexts/polymarketHistoryJob";
import { MarketMapJobProvider } from "./contexts/marketMapJob";
import { MarketsJobProvider } from "./contexts/marketsJob";

const linkClass = ({ isActive }: { isActive: boolean }) =>
  `nav-link${isActive ? " active" : ""}`;

export default function App() {
  const navRef = useRef<HTMLElement | null>(null);

  useLayoutEffect(() => {
    const nav = navRef.current;
    if (!nav) return;

    const updateNavHeight = () => {
      document.documentElement.style.setProperty(
        "--app-nav-height",
        `${nav.offsetHeight}px`,
      );
    };

    updateNavHeight();

    let observer: ResizeObserver | null = null;
    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(updateNavHeight);
      observer.observe(nav);
    }

    window.addEventListener("resize", updateNavHeight);

    return () => {
      window.removeEventListener("resize", updateNavHeight);
      observer?.disconnect();
    };
  }, []);

  return (
    <BrowserRouter>
      <DatasetJobProvider>
        <CalibrationJobProvider>
          <PolymarketHistoryJobProvider>
            <MarketMapJobProvider>
              <MarketsJobProvider>
                <div className="app-shell">
                  <nav className="app-nav" ref={navRef}>
                    <div className="app-nav-inner">
                      <div className="app-brand">
                        <div>
                          <div className="brand-title">Polymarket Pricing Analysis</div>
                          <div className="brand-subtitle">Local Pipeline Workbench</div>
                        </div>
                      </div>
                      <div className="nav-meta">
                        <span className="status-dot" />
                        Localhost
                      </div>
                    </div>
                    <div className="nav-links">
                      <NavLink to="/" end className={linkClass}>
                        Dashboard
                      </NavLink>
                      <NavLink to="/option-chain-history-builder" className={linkClass}>
                        Option Chain History Builder
                      </NavLink>
                      <NavLink to="/polymarket-history-builder" className={linkClass}>
                        Polymarket History Builder
                      </NavLink>
                      <NavLink to="/calibrate" className={linkClass}>
                        Calibrate
                      </NavLink>
                      <NavLink to="/markets" className={linkClass}>
                        Markets
                      </NavLink>
                      <NavLink to="/backtests" className={linkClass}>
                        Backtests
                      </NavLink>
                      <NavLink to="/docs" className={linkClass}>
                        Documentation
                      </NavLink>
                    </div>
                  </nav>
                  <main className="app-main">
                    <Routes>
                      <Route path="/" element={<DashboardPage />} />
                      <Route
                        path="/option-chain-history-builder"
                        element={<DatasetsPage />}
                      />
                      <Route
                        path="/option-chain"
                        element={<Navigate to="/option-chain-history-builder" replace />}
                      />
                      <Route path="/calibrate" element={<CalibrateModelsPage />} />
                      <Route
                        path="/calibrate-models"
                        element={<Navigate to="/calibrate" replace />}
                      />
                      <Route path="/markets" element={<MarketsPage />} />
                      <Route
                        path="/polymarket-history-builder"
                        element={<PolymarketPipelinePage />}
                      />
                      <Route
                        path="/polymarket-pipeline"
                        element={<Navigate to="/polymarket-history-builder" replace />}
                      />
                      <Route path="/backtests" element={<BacktestsPage />} />
                      <Route path="/docs" element={<DocumentationPage />} />
                    </Routes>
                  </main>
                </div>
              </MarketsJobProvider>
            </MarketMapJobProvider>
          </PolymarketHistoryJobProvider>
        </CalibrationJobProvider>
      </DatasetJobProvider>
    </BrowserRouter>
  );
}
