import {
  BrowserRouter,
  NavLink,
  Route,
  Routes,
} from "react-router-dom";

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
  return (
    <BrowserRouter>
      <DatasetJobProvider>
        <CalibrationJobProvider>
          <PolymarketHistoryJobProvider>
            <MarketMapJobProvider>
              <MarketsJobProvider>
                <div className="app-shell">
                  <nav className="app-nav">
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
                      <Route path="/option-chain" element={<DatasetsPage />} />
                      <Route path="/calibrate-models" element={<CalibrateModelsPage />} />
                      <Route path="/markets" element={<MarketsPage />} />
                      <Route path="/polymarket-pipeline" element={<PolymarketPipelinePage />} />
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
