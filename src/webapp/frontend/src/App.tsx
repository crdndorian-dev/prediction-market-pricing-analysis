import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";

import "./App.css";
import CalibrateModelsPage from "./pages/CalibrateModelsPage";
import DashboardPage from "./pages/DashboardPage";
import DatasetsPage from "./pages/DatasetsPage";
import PolymarketSnapshotsPage from "./pages/PolymarketSnapshotsPage";
import { DatasetJobProvider } from "./contexts/datasetJob";

const linkClass = ({ isActive }: { isActive: boolean }) =>
  `nav-link${isActive ? " active" : ""}`;

export default function App() {
  // #region agent log
  fetch('http://127.0.0.1:7243/ingest/fc71f522-f272-4d23-8f35-daabdd192e3f',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'App.tsx',message:'App component rendering',data:{},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'H6'})}).catch(()=>{});
  // #endregion
  return (
    <BrowserRouter>
      <DatasetJobProvider>
        <div className="app-shell">
          <nav className="app-nav">
            <div className="app-brand">
              <span className="brand-mark">PE</span>
              <div>
                <div className="brand-title">PolyEdge</div>
                <div className="brand-subtitle">Local Pipeline Workbench</div>
              </div>
            </div>
            <div className="nav-links">
              <NavLink to="/" end className={linkClass}>
                Dashboard
              </NavLink>
              <NavLink to="/polymarket-snapshots" className={linkClass}>
                Polymarket Snapshots
              </NavLink>
              <NavLink to="/datasets" className={linkClass}>
                Datasets
              </NavLink>
              <NavLink to="/calibrate-models" className={linkClass}>
                Calibrate Models
              </NavLink>
            </div>
            <div className="nav-meta">
              <span className="status-dot" />
              Localhost
            </div>
          </nav>
          <main className="app-main">
            <Routes>
              <Route path="/" element={<DashboardPage />} />
              <Route
                path="/polymarket-snapshots"
                element={<PolymarketSnapshotsPage />}
              />
              <Route path="/datasets" element={<DatasetsPage />} />
              <Route
                path="/calibrate-models"
                element={<CalibrateModelsPage />}
              />
            </Routes>
          </main>
        </div>
      </DatasetJobProvider>
    </BrowserRouter>
  );
}
