import {
  BrowserRouter,
  NavLink,
  Navigate,
  Route,
  Routes,
} from "react-router-dom";

import "./App.css";
import DashboardPage from "./pages/DashboardPage";
import DatasetsPage from "./pages/DatasetsPage";
import CalibrateModelsPage from "./pages/CalibrateModelsPage";
import PHATEdgePage from "./pages/PHATEdgePage";
import PolymarketSnapshotsPage from "./pages/PolymarketSnapshotsPage";
import PolymarketPipelinePage from "./pages/PolymarketPipelinePage";
import BacktestsPage from "./pages/BacktestsPage";
import MarketsPage from "./pages/MarketsPage";
import DocumentationPage from "./pages/DocumentationPage";
import { DatasetJobProvider } from "./contexts/datasetJob";
import { CalibrationJobProvider } from "./contexts/calibrationJob";
import { PolymarketJobProvider } from "./contexts/polymarketJob";
import { PolymarketHistoryJobProvider } from "./contexts/polymarketHistoryJob";
import { PhatEdgeJobProvider } from "./contexts/phatEdgeJob";
import { MarketMapJobProvider } from "./contexts/marketMapJob";
import { MarketsJobProvider } from "./contexts/marketsJob";

const linkClass = ({ isActive }: { isActive: boolean }) =>
  `nav-link${isActive ? " active" : ""}`;

export default function App() {
  return (
    <BrowserRouter>
      <DatasetJobProvider>
        <CalibrationJobProvider>
          <PolymarketJobProvider>
            <PolymarketHistoryJobProvider>
              <MarketMapJobProvider>
                <MarketsJobProvider>
                <PhatEdgeJobProvider>
                  <div className="app-shell">
                                <nav className="app-nav">
                                  <div className="app-nav-inner">
                                    <div className="app-brand">
                                      <div>
                                        <div className="brand-title">PolyEdge</div>
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
                                    <NavLink to="/option-chain" className={linkClass}>
                                      Option Chain History Builder
                                    </NavLink>
                                    <NavLink to="/polymarket-pipeline" className={linkClass}>
                                      Polymarket History Builder
                                    </NavLink>
                                    <NavLink to="/calibrate-models" className={linkClass}>
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
                                      path="/option-chain"
                                      element={<DatasetsPage />}
                                    />
                                    <Route
                                      path="/calibrate-models"
                                      element={<CalibrateModelsPage />}
                                    />
                                    <Route path="/polymarket" element={<PolymarketSnapshotsPage />} />
                                    <Route path="/markets" element={<MarketsPage />} />
                                    <Route path="/polymarket-pipeline" element={<PolymarketPipelinePage />} />
                                    {/* Redirect old routes */}
                                    <Route path="/polymarket-subgraph" element={<Navigate to="/polymarket-pipeline" replace />} />
                                    <Route path="/market-map" element={<Navigate to="/polymarket-pipeline" replace />} />
                                    {/* Temporarily disabled pipeline routes */}
                                    <Route path="/calibrate-models-v2" element={<Navigate to="/calibrate-models" replace />} />
                                    <Route path="/backtests" element={<BacktestsPage />} />
                                    <Route path="/edge" element={<PHATEdgePage />} />
                                    <Route path="/docs" element={<DocumentationPage />} />
                                    <Route
                                      path="/polymarket-snapshots"
                                      element={<Navigate to="/polymarket" replace />}
                                    />
                                    <Route
                                      path="/subgraph"
                                      element={<Navigate to="/polymarket-subgraph" replace />}
                                    />
                                    <Route
                                      path="/datasets"
                                      element={<Navigate to="/option-chain" replace />}
                                    />
                                    <Route path="/phat-edge" element={<Navigate to="/edge" replace />} />
                                  </Routes>
                                </main>
                  </div>
                </PhatEdgeJobProvider>
                </MarketsJobProvider>
              </MarketMapJobProvider>
            </PolymarketHistoryJobProvider>
          </PolymarketJobProvider>
        </CalibrationJobProvider>
      </DatasetJobProvider>
    </BrowserRouter>
  );
}
