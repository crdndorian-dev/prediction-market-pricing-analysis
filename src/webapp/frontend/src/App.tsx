import {
  BrowserRouter,
  NavLink,
  Navigate,
  Route,
  Routes,
} from "react-router-dom";

import "./App.css";
import CalibrateModelsPage from "./pages/CalibrateModelsPage";
import DashboardPage from "./pages/DashboardPage";
import DatasetsPage from "./pages/DatasetsPage";
import PHATEdgePage from "./pages/PHATEdgePage";
import PolymarketSnapshotsPage from "./pages/PolymarketSnapshotsPage";
import { DatasetJobProvider } from "./contexts/datasetJob";
import { CalibrationJobProvider } from "./contexts/calibrationJob";
import { PolymarketJobProvider } from "./contexts/polymarketJob";
import { PhatEdgeJobProvider } from "./contexts/phatEdgeJob";

const linkClass = ({ isActive }: { isActive: boolean }) =>
  `nav-link${isActive ? " active" : ""}`;

export default function App() {
  // #region agent log
  fetch('http://127.0.0.1:7243/ingest/fc71f522-f272-4d23-8f35-daabdd192e3f',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'App.tsx',message:'App component rendering',data:{},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'H6'})}).catch(()=>{});
  // #endregion
  return (
    <BrowserRouter>
      <DatasetJobProvider>
        <CalibrationJobProvider>
          <PolymarketJobProvider>
            <PhatEdgeJobProvider>
              <div className="app-shell">
                <nav className="app-nav">
                  <div className="app-nav-inner">
                    <div className="app-brand">
                      <span className="brand-mark">PE</span>
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
                      Option Chain
                    </NavLink>
                    <NavLink to="/calibrate-models" className={linkClass}>
                      Calibrate
                    </NavLink>
                    <NavLink to="/polymarket" className={linkClass}>
                      Polymarket
                    </NavLink>
                    <NavLink to="/edge" className={linkClass}>
                      Edge
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
                    <Route path="/edge" element={<PHATEdgePage />} />
                    <Route
                      path="/polymarket-snapshots"
                      element={<Navigate to="/polymarket" replace />}
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
          </PolymarketJobProvider>
        </CalibrationJobProvider>
      </DatasetJobProvider>
    </BrowserRouter>
  );
}
