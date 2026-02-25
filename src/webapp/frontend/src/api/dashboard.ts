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
    jobId?: string;
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
  datasetSummary?: {
    fileName: string;
    path: string;
    sizeMB: number;
    rowCount: number;
    columnCount: number;
    tickerCount: number;
    dateRange: {
      column: string | null;
      start: string | null;
      end: string | null;
    };
    lastModified: string;
  } | null;
  dropsSummary?: {
    fileName: string;
    path: string;
    rowCount: number;
  } | null;
  subgraphSummary?: {
    latestRunId: string | null;
    latestRunTime: string | null;
    latestQuery: string | null;
    totalEntities: number | null;
  } | null;
  marketMapSummary?: {
    fileName: string;
    path: string;
    rowCount: number | null;
    lastModified: string;
  } | null;
  barsSummary?: {
    barsDir: string;
    freqs: Record<string, number>;
    lastModified: string | null;
  } | null;
  featuresSummary?: {
    fileName: string;
    path: string;
    lastModified: string;
    featureCount: number | null;
    createdAtUtc: string | null;
  } | null;
  mixedModelSummary?: {
    runCount: number;
    latestRunId: string | null;
    latestRunTime: string | null;
    modelType: string | null;
    rowCount: number | null;
  } | null;
  backtestSummary?: {
    latestRunId: string | null;
    latestRunTime: string | null;
    metrics: {
      trades: number | null;
      hitRate: number | null;
      sharpeLike: number | null;
      maxDrawdown: number | null;
    } | null;
  } | null;
  signalsSummary?: {
    latestRunId: string | null;
    latestRunTime: string | null;
    rowCount: number | null;
  } | null;
  modelSummary?: {
    modelCount: number;
    latestModel: {
      id: string | null;
      calibration: string | null;
      dataset: string | null;
      modifiedAt: string | null;
      metrics: {
        logloss: number | null;
        brier: number | null;
        ece: number | null;
        model: string | null;
        split: string | null;
      } | null;
    };
    bestTest: {
      model: string | null;
      split: string | null;
      logloss: number | null;
      brier: number | null;
      ece: number | null;
    } | null;
    bestVal: {
      model: string | null;
      split: string | null;
      logloss: number | null;
      brier: number | null;
      ece: number | null;
    } | null;
  } | null;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function fetchDashboard(): Promise<DashboardData> {
  const response = await fetch(`${API_BASE}/dashboard`);
  if (!response.ok) {
    throw new Error(`Dashboard request failed: ${response.status}`);
  }
  return response.json();
}
