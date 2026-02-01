export type DatasetFileSummary = {
  name: string;
  path: string;
  size_bytes: number;
  last_modified: string | null;
};

export type DatasetListResponse = {
  base_dir: string;
  datasets: DatasetFileSummary[];
};

export type ModelRunSummary = {
  id: string;
  path: string;
  last_modified: string | null;
  has_metadata: boolean;
  has_metrics: boolean;
};

export type ModelListResponse = {
  base_dir: string;
  models: ModelRunSummary[];
};

export type CalibrateModelRunRequest = {
  csv: string;
  outName?: string;
  targetCol?: string;
  weekCol?: string;
  tickerCol?: string;
  weightCol?: string;
  features?: string;
  categoricalFeatures?: string;
  addInteractions?: boolean;
  calibrate?: "none" | "platt";
  cGrid?: string;
  trainDecayHalfLifeWeeks?: number;
  calibFracOfTrain?: number;
  fitWeightRenorm?: "none" | "mean1";
  testWeeks?: number;
  valWindows?: number;
  valWindowWeeks?: number;
  nBins?: number;
  randomState?: number;
};

export type CalibrateModelRunResponse = {
  ok: boolean;
  out_dir: string;
  stdout: string;
  stderr: string;
  duration_s: number;
  command: string[];
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function fetchCalibrationDatasets(): Promise<DatasetListResponse> {
  const response = await fetch(`${API_BASE}/calibrate-models/datasets`);
  if (!response.ok) {
    throw new Error(`Dataset list failed: ${response.status}`);
  }
  return response.json();
}

export async function fetchCalibrationModels(): Promise<ModelListResponse> {
  const response = await fetch(`${API_BASE}/calibrate-models/models`);
  if (!response.ok) {
    throw new Error(`Model list failed: ${response.status}`);
  }
  return response.json();
}

export async function deleteCalibrationModel(
  modelId: string,
): Promise<ModelRunSummary> {
  const response = await fetch(
    `${API_BASE}/calibrate-models/models/${encodeURIComponent(modelId)}`,
    {
      method: "DELETE",
    },
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Delete model failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function runCalibration(
  payload: CalibrateModelRunRequest,
): Promise<CalibrateModelRunResponse> {
  const response = await fetch(`${API_BASE}/calibrate-models/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      csv: payload.csv,
      out_name: payload.outName,
      target_col: payload.targetCol,
      week_col: payload.weekCol,
      ticker_col: payload.tickerCol,
      weight_col: payload.weightCol,
      features: payload.features,
      categorical_features: payload.categoricalFeatures,
      add_interactions: payload.addInteractions,
      calibrate: payload.calibrate,
      c_grid: payload.cGrid,
      train_decay_half_life_weeks: payload.trainDecayHalfLifeWeeks,
      calib_frac_of_train: payload.calibFracOfTrain,
      fit_weight_renorm: payload.fitWeightRenorm,
      test_weeks: payload.testWeeks,
      val_windows: payload.valWindows,
      val_window_weeks: payload.valWindowWeeks,
      n_bins: payload.nBins,
      random_state: payload.randomState,
    }),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Calibration request failed (${response.status}): ${detail || "unknown error"}`,
    );
  }

  return response.json();
}
