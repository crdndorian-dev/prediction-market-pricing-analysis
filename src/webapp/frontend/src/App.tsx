import { BrowserRouter, NavLink, Route, Routes } from "react-router-dom";

import "./App.css";
import DashboardPage from "./pages/DashboardPage";
import DatasetsPage from "./pages/DatasetsPage";
import PipelinePage from "./pages/PipelinePage";
import ResultsPage from "./pages/ResultsPage";

const linkClass = ({ isActive }: { isActive: boolean }) =>
  `nav-link${isActive ? " active" : ""}`;

export default function App() {
  return (
    <BrowserRouter>
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
            <NavLink to="/pipeline" className={linkClass}>
              Pipeline
            </NavLink>
            <NavLink to="/results" className={linkClass}>
              Results
            </NavLink>
            <NavLink to="/datasets" className={linkClass}>
              Datasets
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
            <Route path="/pipeline" element={<PipelinePage />} />
            <Route path="/results" element={<ResultsPage />} />
            <Route path="/datasets" element={<DatasetsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
