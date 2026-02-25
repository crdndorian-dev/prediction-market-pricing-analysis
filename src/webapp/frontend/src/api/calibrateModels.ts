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

export type ModelDetailResponse = {
  id: string;
  path: string;
  last_modified: string | null;
  has_metadata: boolean;
  has_metrics: boolean;
  files: string[];
  features_used?: string[] | null;
  categorical_features_used?: string[] | null;
  metrics_summary?: Record<string, SplitMetricSummary> | null;
  model_equation?: string | null;
  metadata?: Record<string, unknown> | null;
  feature_manifest?: Record<string, unknown> | null;
  two_stage_metrics?: TwoStageMetricRow[] | null;
};

export type ModelListResponse = {
  base_dir: string;
  models: ModelRunSummary[];
};

export type CalibrateModelRunRequest = {
  csv: string;
  outName?: string;
  modelKind?: "calibrate" | "mixed" | "both";
  targetCol?: string;
  weekCol?: string;
  tickerCol?: string;
  weightCol?: string;
  foundationTickers?: string;
  foundationWeight?: number;
  tickerIntercepts?: "none" | "all" | "non_foundation";
  tickerXInteractions?: boolean;
  tickerMinSupport?: number;
  tickerMinSupportInteractions?: number;
  trainTickers?: string;
  tdaysAllowed?: string;
  asofDowAllowed?: string;
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
  eceqBins?: number;
  selectionObjective?: "delta_vs_baseline";
  fallbackToBaselineIfWorse?: boolean;
  autoDropNearConstant?: boolean;
  randomState?: number;
  metricsTopTickers?: number;
  enableXAbsM?: boolean;
  groupReweight?: "none" | "chain";
  maxAbsLogm?: number;
  dropPrnExtremes?: boolean;
  prnEps?: number;
  bootstrapCi?: boolean;
  bootstrapB?: number;
  bootstrapSeed?: number;
  bootstrapGroup?: "auto" | "ticker_day" | "day" | "iid";
  mixedFeatures?: string;
  mixedOutDir?: string;
  mixedRunId?: string;
  mixedModel?: "residual" | "blend";
  mixedPmCol?: string;
  mixedPrnCol?: string;
  mixedLabelCol?: string;
  mixedFeaturesCols?: string;
  mixedTrainFrac?: number;
  mixedWalkForward?: boolean;
  mixedWfTrainDays?: number;
  mixedWfTestDays?: number;
  mixedWfStepDays?: number;
  mixedMaxSplits?: number;
  mixedEmbargoDays?: number;
  mixedMinTimeToResolutionDays?: number;
  mixedAlpha?: number;
  twoStageMode?: boolean;
  twoStagePrnCsv?: string;
  twoStagePmCsv?: string;
  twoStageLabelCol?: string;
};

export type MetricsSummaryStatus = "good" | "unusable";

export type SplitMetricSummary = {
  split: string;
  baseline_logloss: number;
  model_logloss: number;
  delta_model_minus_baseline: number;
  baseline_brier?: number | null;
  model_brier?: number | null;
  delta_brier?: number | null;
  baseline_ece?: number | null;
  model_ece?: number | null;
  delta_ece?: number | null;
  baseline_ece_q?: number | null;
  model_ece_q?: number | null;
  delta_ece_q?: number | null;
  delta_logloss_ci_lo?: number | null;
  delta_logloss_ci_hi?: number | null;
  delta_brier_ci_lo?: number | null;
  delta_brier_ci_hi?: number | null;
  delta_ece_ci_lo?: number | null;
  delta_ece_ci_hi?: number | null;
  delta_ece_q_ci_lo?: number | null;
  delta_ece_q_ci_hi?: number | null;
  bootstrap_n_groups?: number | null;
  bootstrap_B?: number | null;
  status: MetricsSummaryStatus;
  verdict: string;
};

export type CalibrateModelRunResponse = {
  ok: boolean;
  out_dir: string;
  stdout: string;
  stderr: string;
  duration_s: number;
  command: string[];
  files: string[];
  metrics_summary?: Record<string, SplitMetricSummary> | null;
  auto_out_dir?: string | null;
  features?: string[] | null;
  model_equation?: string | null;
  two_stage_metrics?: TwoStageMetricRow[] | null;
};

export type TwoStageMetricRow = {
  split: string;
  model: string;
  logloss: number;
  brier: number;
  n: number;
};

export type CalibrationJobStatus = {
  job_id: string;
  status: "queued" | "running" | "finished" | "failed";
  mode: "manual" | "auto";
  result: CalibrateModelRunResponse | null;
  error: string | null;
  started_at: string | null;
  finished_at: string | null;
  progress?: ProgressPayload | null;
};

export type ProgressPayload = {
  stage: string;
  trials_total: number;
  trials_done: number;
  trials_failed: number;
  best_score_so_far: number | null;
  last_error: string | null;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function fetchCalibrationDatasets(): Promise<DatasetListResponse> {
  const response = await fetch(`${API_BASE}/calibrate-models/datasets`);
  if (!response.ok) {
    throw new Error(`Dataset list failed: ${response.status}`);
  }
  return response.json();
}

export async function fetchPolymarketCalibrationDatasets(): Promise<DatasetListResponse> {
  const response = await fetch(`${API_BASE}/calibrate-models/polymarket-datasets`);
  if (!response.ok) {
    throw new Error(`Polymarket dataset list failed: ${response.status}`);
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

export async function fetchCalibrationModelDetail(
  modelId: string,
): Promise<ModelDetailResponse> {
  const response = await fetch(
    `${API_BASE}/calibrate-models/models/${encodeURIComponent(modelId)}`,
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Model detail failed (${response.status}): ${detail || "unknown error"}`,
    );
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
      model_kind: payload.modelKind,
      target_col: payload.targetCol,
      week_col: payload.weekCol,
      ticker_col: payload.tickerCol,
      weight_col: payload.weightCol,
      foundation_tickers: payload.foundationTickers,
      foundation_weight: payload.foundationWeight,
      ticker_intercepts: payload.tickerIntercepts,
      ticker_x_interactions: payload.tickerXInteractions,
      ticker_min_support: payload.tickerMinSupport,
      ticker_min_support_interactions: payload.tickerMinSupportInteractions,
      train_tickers: payload.trainTickers,
      tdays_allowed: payload.tdaysAllowed,
      asof_dow_allowed: payload.asofDowAllowed,
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
      eceq_bins: payload.eceqBins,
      selection_objective: payload.selectionObjective,
      fallback_to_baseline_if_worse: payload.fallbackToBaselineIfWorse,
      auto_drop_near_constant: payload.autoDropNearConstant,
      metrics_top_tickers: payload.metricsTopTickers,
      random_state: payload.randomState,
      enable_x_abs_m: payload.enableXAbsM,
      group_reweight: payload.groupReweight,
      max_abs_logm: payload.maxAbsLogm,
      drop_prn_extremes: payload.dropPrnExtremes,
      prn_eps: payload.prnEps,
      bootstrap_ci: payload.bootstrapCi,
      bootstrap_b: payload.bootstrapB,
      bootstrap_seed: payload.bootstrapSeed,
      bootstrap_group: payload.bootstrapGroup,
      mixed_features: payload.mixedFeatures,
      mixed_out_dir: payload.mixedOutDir,
      mixed_run_id: payload.mixedRunId,
      mixed_model: payload.mixedModel,
      mixed_pm_col: payload.mixedPmCol,
      mixed_prn_col: payload.mixedPrnCol,
      mixed_label_col: payload.mixedLabelCol,
      mixed_features_cols: payload.mixedFeaturesCols,
      mixed_train_frac: payload.mixedTrainFrac,
      mixed_walk_forward: payload.mixedWalkForward,
      mixed_wf_train_days: payload.mixedWfTrainDays,
      mixed_wf_test_days: payload.mixedWfTestDays,
      mixed_wf_step_days: payload.mixedWfStepDays,
      mixed_max_splits: payload.mixedMaxSplits,
      mixed_embargo_days: payload.mixedEmbargoDays,
      mixed_min_time_to_resolution_days: payload.mixedMinTimeToResolutionDays,
      mixed_alpha: payload.mixedAlpha,
      two_stage_mode: payload.twoStageMode,
      two_stage_prn_csv: payload.twoStagePrnCsv,
      two_stage_pm_csv: payload.twoStagePmCsv,
      two_stage_label_col: payload.twoStageLabelCol,
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

export type AutoModelRunRequest = {
  csv: string;
  mode?: "option_only" | "mixed";
  pmDatasetPath?: string;
  runName?: string;
  modelKind?: "calibrate" | "mixed" | "both";
  objective?: "logloss" | "roll_val_logloss" | "test_logloss";
  maxTrials?: number;
  seed?: number;
  parallel?: number;
  baselineArgs?: string;
  tdaysAllowed?: string;
  asofDowAllowed?: string;
  foundationTickers?: string;
  foundationWeight?: number;
  bootstrapCi?: boolean;
  bootstrapB?: number;
  bootstrapSeed?: number;
  bootstrapGroup?: "auto" | "ticker_day" | "day" | "iid";
  mixedFeatures?: string;
  mixedOutDir?: string;
  mixedRunId?: string;
  mixedModel?: "residual" | "blend";
  mixedPmCol?: string;
  mixedPrnCol?: string;
  mixedLabelCol?: string;
  mixedFeaturesCols?: string;
  mixedTrainFrac?: number;
  mixedWalkForward?: boolean;
  mixedWfTrainDays?: number;
  mixedWfTestDays?: number;
  mixedWfStepDays?: number;
  mixedMaxSplits?: number;
  mixedEmbargoDays?: number;
  mixedMinTimeToResolutionDays?: number;
  mixedAlpha?: number;
};

export async function runAutoModelSelection(
  payload: AutoModelRunRequest,
): Promise<CalibrateModelRunResponse> {
  const response = await fetch(`${API_BASE}/calibrate-models/run-auto`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      csv: payload.csv,
      mode: payload.mode,
      pm_dataset_path: payload.pmDatasetPath,
      run_name: payload.runName,
      model_kind: payload.modelKind,
      objective: payload.objective,
      max_trials: payload.maxTrials,
      seed: payload.seed,
      parallel: payload.parallel,
      baseline_args: payload.baselineArgs,
      tdays_allowed: payload.tdaysAllowed,
      asof_dow_allowed: payload.asofDowAllowed,
      foundation_tickers: payload.foundationTickers,
      foundation_weight: payload.foundationWeight,
      bootstrap_ci: payload.bootstrapCi,
      bootstrap_b: payload.bootstrapB,
      bootstrap_seed: payload.bootstrapSeed,
      bootstrap_group: payload.bootstrapGroup,
      mixed_features: payload.mixedFeatures,
      mixed_out_dir: payload.mixedOutDir,
      mixed_run_id: payload.mixedRunId,
      mixed_model: payload.mixedModel,
      mixed_pm_col: payload.mixedPmCol,
      mixed_prn_col: payload.mixedPrnCol,
      mixed_label_col: payload.mixedLabelCol,
      mixed_features_cols: payload.mixedFeaturesCols,
      mixed_train_frac: payload.mixedTrainFrac,
      mixed_walk_forward: payload.mixedWalkForward,
      mixed_wf_train_days: payload.mixedWfTrainDays,
      mixed_wf_test_days: payload.mixedWfTestDays,
      mixed_wf_step_days: payload.mixedWfStepDays,
      mixed_max_splits: payload.mixedMaxSplits,
      mixed_embargo_days: payload.mixedEmbargoDays,
      mixed_min_time_to_resolution_days: payload.mixedMinTimeToResolutionDays,
      mixed_alpha: payload.mixedAlpha,
    }),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Auto model selection request failed (${response.status}): ${
        detail || "unknown error"
      }`,
    );
  }

  return response.json();
}

export async function startCalibrationJob(
  payload: CalibrateModelRunRequest,
): Promise<CalibrationJobStatus> {
  const response = await fetch(`${API_BASE}/calibrate-models/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      csv: payload.csv,
      out_name: payload.outName,
      target_col: payload.targetCol,
      week_col: payload.weekCol,
      ticker_col: payload.tickerCol,
      weight_col: payload.weightCol,
      foundation_tickers: payload.foundationTickers,
      foundation_weight: payload.foundationWeight,
      ticker_intercepts: payload.tickerIntercepts,
      ticker_x_interactions: payload.tickerXInteractions,
      ticker_min_support: payload.tickerMinSupport,
      ticker_min_support_interactions: payload.tickerMinSupportInteractions,
      train_tickers: payload.trainTickers,
      tdays_allowed: payload.tdaysAllowed,
      asof_dow_allowed: payload.asofDowAllowed,
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
      eceq_bins: payload.eceqBins,
      selection_objective: payload.selectionObjective,
      fallback_to_baseline_if_worse: payload.fallbackToBaselineIfWorse,
      auto_drop_near_constant: payload.autoDropNearConstant,
      metrics_top_tickers: payload.metricsTopTickers,
      random_state: payload.randomState,
      enable_x_abs_m: payload.enableXAbsM,
      group_reweight: payload.groupReweight,
      max_abs_logm: payload.maxAbsLogm,
      drop_prn_extremes: payload.dropPrnExtremes,
      prn_eps: payload.prnEps,
      bootstrap_ci: payload.bootstrapCi,
      bootstrap_b: payload.bootstrapB,
      bootstrap_seed: payload.bootstrapSeed,
      bootstrap_group: payload.bootstrapGroup,
    }),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Calibration job start failed (${response.status}): ${detail || "unknown error"}`,
    );
  }

  return response.json();
}

export async function startAutoCalibrationJob(
  payload: AutoModelRunRequest,
): Promise<CalibrationJobStatus> {
  const response = await fetch(`${API_BASE}/calibrate-models/jobs-auto`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      csv: payload.csv,
      mode: payload.mode,
      pm_dataset_path: payload.pmDatasetPath,
      run_name: payload.runName,
      objective: payload.objective,
      max_trials: payload.maxTrials,
      seed: payload.seed,
      parallel: payload.parallel,
      baseline_args: payload.baselineArgs,
      tdays_allowed: payload.tdaysAllowed,
      asof_dow_allowed: payload.asofDowAllowed,
      foundation_tickers: payload.foundationTickers,
      foundation_weight: payload.foundationWeight,
      bootstrap_ci: payload.bootstrapCi,
      bootstrap_b: payload.bootstrapB,
      bootstrap_seed: payload.bootstrapSeed,
      bootstrap_group: payload.bootstrapGroup,
    }),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Auto job start failed (${response.status}): ${detail || "unknown error"}`,
    );
  }

  return response.json();
}

export async function getCalibrationJob(
  jobId: string,
): Promise<CalibrationJobStatus> {
  const response = await fetch(`${API_BASE}/calibrate-models/jobs/${jobId}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Calibration job not found (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function renameCalibrationModel(
  modelId: string,
  newName: string,
): Promise<ModelRunSummary> {
  const response = await fetch(
    `${API_BASE}/calibrate-models/models/${encodeURIComponent(modelId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ new_name: newName }),
    },
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Rename model failed (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export type ModelFileSummary = {
  name: string;
  size_bytes: number;
  is_viewable: boolean;
};

export type ModelFilesListResponse = {
  model_id: string;
  files: ModelFileSummary[];
};

export type ModelFileContentResponse = {
  model_id: string;
  filename: string;
  content: string;
  content_type: "json" | "csv" | "markdown" | "text";
  truncated: boolean;
};

export async function fetchModelFiles(
  modelId: string,
): Promise<ModelFilesListResponse> {
  const response = await fetch(
    `${API_BASE}/calibrate-models/models/${encodeURIComponent(modelId)}/files`,
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Failed to list model files (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function fetchModelFileContent(
  modelId: string,
  filename: string,
): Promise<ModelFileContentResponse> {
  const response = await fetch(
    `${API_BASE}/calibrate-models/models/${encodeURIComponent(modelId)}/files/${encodeURIComponent(filename)}`,
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Failed to fetch file content (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export type RegimePreviewRequest = {
  csv: string;
  tdaysAllowed?: string;
  asofDowAllowed?: string;
};

export type RegimePreviewResponse = {
  rows_before: number;
  rows_after: number;
  tickers_after: number;
  by_tdays: Record<string, number>;
};

export async function previewCalibrationRegime(
  payload: RegimePreviewRequest,
): Promise<RegimePreviewResponse> {
  const response = await fetch(`${API_BASE}/calibrate-models/preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      csv: payload.csv,
      tdays_allowed: payload.tdaysAllowed,
      asof_dow_allowed: payload.asofDowAllowed,
    }),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Regime preview failed (${response.status}): ${detail || "unknown error"}`,
    );
  }

  return response.json();
}

export type DatasetTickersResponse = {
  dataset: string;
  tickers: string[];
  count: number;
};

export type FeatureStat = {
  missing_pct: number;
  dtype: string;
  nunique: number;
};

export type RegimeInfo = {
  tdays_mode: number[] | null;
  is_weekly: boolean | null;
  is_daily: boolean | null;
};

export type DatasetFeaturesResponse = {
  dataset: string;
  available_columns: string[];
  feature_stats: Record<string, FeatureStat>;
  regime_info: RegimeInfo;
};

export async function fetchDatasetFeatures(
  dataset: string,
): Promise<DatasetFeaturesResponse> {
  const response = await fetch(
    `${API_BASE}/calibrate-models/datasets/features?dataset=${encodeURIComponent(dataset)}`,
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Failed to fetch dataset features (${response.status}): ${detail || "unknown error"}`,
    );
  }
  return response.json();
}

export async function fetchDatasetTickers(
  dataset: string,
): Promise<DatasetTickersResponse> {
  const response = await fetch(
    `${API_BASE}/calibrate-models/datasets/tickers?dataset=${encodeURIComponent(dataset)}`,
  );

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Failed to fetch dataset tickers (${response.status}): ${detail || "unknown error"}`,
    );
  }

  return response.json();
}
