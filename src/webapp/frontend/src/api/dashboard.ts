export type DashboardData = {
  asOf: string;
  hero: {
    dataFreshnessDate: string | null;
    dataFreshnessDays: number | null;
    dataSourceLabel: string | null;
    calibrationEce: number | null;
    calibrationModel: string | null;
    calibrationSplit: string | null;
    lastRunId: string | null;
    lastRunTime: string | null;
    lastRunSummary: string | null;
  };
  readiness: Array<{
    title: string;
    detail: string;
    status: "Ready" | "Needs review" | "Missing" | string;
    progress: number;
  }>;
  runQueue: Array<{
    name: string;
    state: string;
    detail: string;
  }>;
  recentRuns: Array<{
    id: string;
    dataset: string;
    focus: string;
    status: string;
    time: string;
  }>;
  calibrationSnapshot: {
    logloss: number | null;
    brier: number | null;
    ece: number | null;
    model: string | null;
    split: string | null;
  };
  signalBars: number[];
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function fetchDashboard(): Promise<DashboardData> {
  const response = await fetch(`${API_BASE}/dashboard`);
  if (!response.ok) {
    throw new Error(`Dashboard request failed: ${response.status}`);
  }
  return response.json();
}
