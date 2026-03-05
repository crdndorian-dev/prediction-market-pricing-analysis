export type DatasetFileSummary = {
  name: string;
  path: string;
  size_bytes: number;
  last_modified: string | null;
  dataset_id?: string | null;
  rows?: number | null;
  date_col_used?: string | null;
  date_start?: string | null;
  date_end?: string | null;
  week_count?: number | null;
  ticker_count?: number | null;
  ticker_sample?: string[] | null;
  available_weight_columns?: string[] | null;
  available_grouping_keys?: string[] | null;
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
  dataset_id?: string | null;
  dataset_path?: string | null;
  train_date_start?: string | null;
  train_date_end?: string | null;
  tickers_summary?: string | null;
  dow_regime?: string | null;
  split_strategy?: string | null;
  c_value?: number | null;
  calibration_method?: string | null;
  weighting_mode?: string | null;
  is_two_stage?: boolean | null;
  run_type?: "manual" | "auto" | null;
  auto_status?: string | null;
  selected_trial_id?: number | null;
  has_selected_model?: boolean | null;
};

export type EquationTermContribution = {
  feature_name: string;
  ticker?: string | null;
  coef: number;
  latex: string;
};

export type EquationSpec = {
  compact_latex: string;
  expanded_latex?: string | null;
  linear_predictor_compact_latex?: string | null;
  linear_predictor_expanded_latex?: string | null;
  term_count?: number | null;
  intercept?: number | null;
  uses_transformed_features?: boolean;
  feature_name_source?: string | null;
  pm_feature_names?: string[] | null;
  ticker_intercepts?: EquationTermContribution[] | null;
  ticker_interactions?: EquationTermContribution[] | null;
  notes?: string[] | null;
  all_feature_names?: string[] | null;
  model_family?: string | null;
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
  split_row_counts?: Record<string, number> | null;
  split_group_counts?: Record<string, number> | null;
  model_equation?: string | null;
  model_equation_spec?: EquationSpec | null;
  metadata?: Record<string, unknown> | null;
  feature_manifest?: Record<string, unknown> | null;
  two_stage_metrics?: TwoStageMetricRow[] | null;
  is_two_stage?: boolean;
  stage1_equation?: string | null;
  stage1_equation_spec?: EquationSpec | null;
  two_stage_equation?: string | null;
  two_stage_equation_spec?: EquationSpec | null;
  combined_p_hat_equation?: string | null;
  combined_p_hat_equation_spec?: EquationSpec | null;
};

export type ModelListResponse = {
  base_dir: string;
  models: ModelRunSummary[];
};

export type CalibrateModelRunRequest = {
  csv: string;
  outName?: string;
  runMode?: "manual" | "auto_search";
  modelKind?: "calibrate" | "mixed" | "both";
  targetCol?: string;
  weekCol?: string;
  tickerCol?: string;
  weightCol?: string;
  weightColStrategy?: "auto" | "weight_final" | "sample_weight_final" | "uniform";
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
  selectionObjective?: "delta_vs_baseline" | "logloss" | "brier" | "ece_q";
  fallbackToBaselineIfWorse?: boolean;
  autoDropNearConstant?: boolean;
  randomState?: number;
  metricsTopTickers?: number;
  enableXAbsM?: boolean;
  groupReweight?: "none" | "chain" | "chain_snapshot";
  maxAbsLogm?: number;
  dropPrnExtremes?: boolean;
  prnEps?: number;
  prnBelow?: number;
  prnAbove?: number;
  bootstrapCi?: boolean;
  bootstrapB?: number;
  bootstrapSeed?: number;
  bootstrapGroup?: "auto" | "ticker_day" | "day" | "iid" | "contract_id" | "group_id";
  allowDefaults?: boolean;
  allowIidBootstrap?: boolean;
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
  split?: {
    strategy: "walk_forward" | "single_holdout";
    windowMode: "rolling" | "expanding";
    trainWindowWeeks?: number;
    validationFolds?: number;
    validationWindowWeeks?: number;
    testWindowWeeks?: number;
    embargoDays?: number;
  };
  regularization?: {
    cGrid?: string;
    calibrationMethod?: "none" | "platt";
    selectionObjective?: "logloss" | "brier" | "ece_q";
  };
  modelStructure?: {
    tradingUniverseTickers?: string;
    trainTickers?: string;
    foundationTickers?: string;
    foundationWeight?: number;
    tickerIntercepts?: "none" | "all" | "non_foundation";
    tickerXInteractions?: boolean;
    tickerMinSupport?: number;
    tickerMinSupportInteractions?: number;
  };
  weighting?: {
    baseWeightSource?: "dataset_weight" | "uniform";
    groupingKey?: string;
    groupEqualization?: boolean;
    renorm?: "mean1";
    tradingUniverseUpweight?: number;
    tickerBalanceMode?: "none" | "sqrt_inv_clipped";
  };
  bootstrap?: {
    bootstrapCi?: boolean;
    bootstrapGroup?: string;
    bootstrapB?: number;
    bootstrapSeed?: number;
    ciLevel?: 90 | 95 | 99;
    perSplitReporting?: boolean;
    perFoldReporting?: boolean;
    allowIidBootstrap?: boolean;
  };
  diagnostics?: {
    splitTimeline?: boolean;
    perFoldDeltaChart?: boolean;
    perGroupDeltaDistribution?: boolean;
  };
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
  split_row_counts?: Record<string, number> | null;
  split_group_counts?: Record<string, number> | null;
  auto_out_dir?: string | null;
  features?: string[] | null;
  model_equation?: string | null;
  model_equation_spec?: EquationSpec | null;
  two_stage_metrics?: TwoStageMetricRow[] | null;
  is_two_stage?: boolean;
  stage1_equation?: string | null;
  stage1_equation_spec?: EquationSpec | null;
  two_stage_equation?: string | null;
  two_stage_equation_spec?: EquationSpec | null;
  combined_p_hat_equation?: string | null;
  combined_p_hat_equation_spec?: EquationSpec | null;
  artifact_manifest?: Array<{
    name: string;
    type: string;
    path: string;
    relative_path?: string;
    section?: "selected_model" | "auto_search" | "legacy_root";
  }> | null;
  diagnostics_available?: Record<string, boolean> | null;
  warnings?: string[] | null;
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
  status: "queued" | "running" | "finished" | "failed" | "cancelled";
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
  phase?: string | null;
  candidate_index?: number | null;
  candidate_total?: number | null;
  fold_index?: number | null;
  fold_total?: number | null;
  message?: string | null;
  last_log_lines?: string[] | null;
  top_candidates?: Array<Record<string, any>> | null;
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
      run_mode: payload.runMode,
      model_kind: payload.modelKind,
      target_col: payload.targetCol,
      week_col: payload.weekCol,
      ticker_col: payload.tickerCol,
      weight_col: payload.weightCol,
      weight_col_strategy: payload.weightColStrategy,
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
      prn_below: payload.prnBelow,
      prn_above: payload.prnAbove,
      bootstrap_ci: payload.bootstrapCi,
      bootstrap_b: payload.bootstrapB,
      bootstrap_seed: payload.bootstrapSeed,
      bootstrap_group: payload.bootstrapGroup,
      allow_defaults: payload.allowDefaults,
      allow_iid_bootstrap: payload.allowIidBootstrap,
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
      split: payload.split
        ? {
            strategy: payload.split.strategy,
            window_mode: payload.split.windowMode,
            train_window_weeks: payload.split.trainWindowWeeks,
            validation_folds: payload.split.validationFolds,
            validation_window_weeks: payload.split.validationWindowWeeks,
            test_window_weeks: payload.split.testWindowWeeks,
            embargo_days: payload.split.embargoDays,
          }
        : undefined,
      regularization: payload.regularization
        ? {
            c_grid: payload.regularization.cGrid,
            calibration_method: payload.regularization.calibrationMethod,
            selection_objective: payload.regularization.selectionObjective,
          }
        : undefined,
      model_structure: payload.modelStructure
        ? {
            trading_universe_tickers: payload.modelStructure.tradingUniverseTickers,
            train_tickers: payload.modelStructure.trainTickers,
            foundation_tickers: payload.modelStructure.foundationTickers,
            foundation_weight: payload.modelStructure.foundationWeight,
            ticker_intercepts: payload.modelStructure.tickerIntercepts,
            ticker_x_interactions: payload.modelStructure.tickerXInteractions,
            ticker_min_support: payload.modelStructure.tickerMinSupport,
            ticker_min_support_interactions: payload.modelStructure.tickerMinSupportInteractions,
          }
        : undefined,
      weighting: payload.weighting
        ? {
            base_weight_source: payload.weighting.baseWeightSource,
            grouping_key: payload.weighting.groupingKey,
            group_equalization: payload.weighting.groupEqualization,
            renorm: payload.weighting.renorm,
            trading_universe_upweight: payload.weighting.tradingUniverseUpweight,
            ticker_balance_mode: payload.weighting.tickerBalanceMode,
          }
        : undefined,
      bootstrap: payload.bootstrap
        ? {
            bootstrap_ci: payload.bootstrap.bootstrapCi,
            bootstrap_group: payload.bootstrap.bootstrapGroup,
            bootstrap_b: payload.bootstrap.bootstrapB,
            bootstrap_seed: payload.bootstrap.bootstrapSeed,
            ci_level: payload.bootstrap.ciLevel,
            per_split_reporting: payload.bootstrap.perSplitReporting,
            per_fold_reporting: payload.bootstrap.perFoldReporting,
            allow_iid_bootstrap: payload.bootstrap.allowIidBootstrap,
          }
        : undefined,
      diagnostics: payload.diagnostics
        ? {
            split_timeline: payload.diagnostics.splitTimeline,
            per_fold_delta_chart: payload.diagnostics.perFoldDeltaChart,
            per_group_delta_distribution: payload.diagnostics.perGroupDeltaDistribution,
          }
        : undefined,
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

export type AutoSearchConfig = {
  featureSets: string[][];
  cValues: number[];
  calibrationMethods: Array<"none" | "platt">;
  tradingUniverseUpweight: number[];
  foundationWeight: number[];
  tickerIntercepts: Array<"off" | "on">;
  allowRiskyFeatures?: boolean;
  advancedInteractions?: boolean;
  maxTrials?: number;
  selectionRule?: "one_se" | "epsilon";
  epsilon?: number;
  outerFolds?: number;
  outerTestWeeks?: number;
  outerGapWeeks?: number;
  outerSelectionMetric?: "median_delta_logloss" | "worst_delta_logloss" | "mean_delta_logloss";
  outerMinImproveFraction?: number;
  outerMaxWorstDelta?: number;
  acceptDeltaThreshold?: number;
  minImproveFraction?: number;
  maxWorstDelta?: number;
};

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
  bootstrapGroup?: "auto" | "ticker_day" | "day" | "iid" | "contract_id" | "group_id";
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
  baseConfig?: CalibrateModelRunRequest;
  search?: AutoSearchConfig;
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
      base_config: payload.baseConfig ? serializeCalibrationPayload(payload.baseConfig) : undefined,
          search: payload.search
        ? {
            feature_sets: payload.search.featureSets,
            c_values: payload.search.cValues,
            calibration_methods: payload.search.calibrationMethods,
            trading_universe_upweight: payload.search.tradingUniverseUpweight,
            foundation_weight: payload.search.foundationWeight,
            ticker_intercepts: payload.search.tickerIntercepts,
            allow_risky_features: payload.search.allowRiskyFeatures,
            advanced_interactions: payload.search.advancedInteractions,
            max_trials: payload.search.maxTrials,
            selection_rule: payload.search.selectionRule,
            epsilon: payload.search.epsilon,
            outer_folds: payload.search.outerFolds,
            outer_test_weeks: payload.search.outerTestWeeks,
            outer_gap_weeks: payload.search.outerGapWeeks,
            outer_selection_metric: payload.search.outerSelectionMetric,
            outer_min_improve_fraction: payload.search.outerMinImproveFraction,
            outer_max_worst_delta: payload.search.outerMaxWorstDelta,
            accept_delta_threshold: payload.search.acceptDeltaThreshold,
            min_improve_fraction: payload.search.minImproveFraction,
            max_worst_delta: payload.search.maxWorstDelta,
          }
        : undefined,
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
  const serialized = serializeCalibrationPayload(payload);
  const response = await fetch(`${API_BASE}/calibrate-models/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(serialized),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Calibration job start failed (${response.status}): ${detail || "unknown error"}`,
    );
  }

  return response.json();
}

const serializeCalibrationPayload = (payload: CalibrateModelRunRequest) => ({
  csv: payload.csv,
  out_name: payload.outName,
  run_mode: payload.runMode,
  target_col: payload.targetCol,
  week_col: payload.weekCol,
  ticker_col: payload.tickerCol,
  weight_col: payload.weightCol,
  weight_col_strategy: payload.weightColStrategy,
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
  prn_below: payload.prnBelow,
  prn_above: payload.prnAbove,
  bootstrap_ci: payload.bootstrapCi,
  bootstrap_b: payload.bootstrapB,
  bootstrap_seed: payload.bootstrapSeed,
  bootstrap_group: payload.bootstrapGroup,
  allow_defaults: payload.allowDefaults,
  allow_iid_bootstrap: payload.allowIidBootstrap,
  split: payload.split
    ? {
        strategy: payload.split.strategy,
        window_mode: payload.split.windowMode,
        train_window_weeks: payload.split.trainWindowWeeks,
        validation_folds: payload.split.validationFolds,
        validation_window_weeks: payload.split.validationWindowWeeks,
        test_window_weeks: payload.split.testWindowWeeks,
        embargo_days: payload.split.embargoDays,
      }
    : undefined,
  regularization: payload.regularization
    ? {
        c_grid: payload.regularization.cGrid,
        calibration_method: payload.regularization.calibrationMethod,
        selection_objective: payload.regularization.selectionObjective,
      }
    : undefined,
  model_structure: payload.modelStructure
    ? {
        trading_universe_tickers: payload.modelStructure.tradingUniverseTickers,
        train_tickers: payload.modelStructure.trainTickers,
        foundation_tickers: payload.modelStructure.foundationTickers,
        foundation_weight: payload.modelStructure.foundationWeight,
        ticker_intercepts: payload.modelStructure.tickerIntercepts,
        ticker_x_interactions: payload.modelStructure.tickerXInteractions,
        ticker_min_support: payload.modelStructure.tickerMinSupport,
        ticker_min_support_interactions: payload.modelStructure.tickerMinSupportInteractions,
      }
    : undefined,
  weighting: payload.weighting
    ? {
        base_weight_source: payload.weighting.baseWeightSource,
        grouping_key: payload.weighting.groupingKey,
        group_equalization: payload.weighting.groupEqualization,
        renorm: payload.weighting.renorm,
        trading_universe_upweight: payload.weighting.tradingUniverseUpweight,
        ticker_balance_mode: payload.weighting.tickerBalanceMode,
      }
    : undefined,
  bootstrap: payload.bootstrap
    ? {
        bootstrap_ci: payload.bootstrap.bootstrapCi,
        bootstrap_group: payload.bootstrap.bootstrapGroup,
        bootstrap_b: payload.bootstrap.bootstrapB,
        bootstrap_seed: payload.bootstrap.bootstrapSeed,
        ci_level: payload.bootstrap.ciLevel,
        per_split_reporting: payload.bootstrap.perSplitReporting,
        per_fold_reporting: payload.bootstrap.perFoldReporting,
        allow_iid_bootstrap: payload.bootstrap.allowIidBootstrap,
      }
    : undefined,
  diagnostics: payload.diagnostics
    ? {
        split_timeline: payload.diagnostics.splitTimeline,
        per_fold_delta_chart: payload.diagnostics.perFoldDeltaChart,
        per_group_delta_distribution: payload.diagnostics.perGroupDeltaDistribution,
      }
    : undefined,
});

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
      base_config: payload.baseConfig ? serializeCalibrationPayload(payload.baseConfig) : undefined,
      search: payload.search
        ? {
            feature_sets: payload.search.featureSets,
            c_values: payload.search.cValues,
            calibration_methods: payload.search.calibrationMethods,
            trading_universe_upweight: payload.search.tradingUniverseUpweight,
            foundation_weight: payload.search.foundationWeight,
            ticker_intercepts: payload.search.tickerIntercepts,
            allow_risky_features: payload.search.allowRiskyFeatures,
            advanced_interactions: payload.search.advancedInteractions,
            max_trials: payload.search.maxTrials,
            selection_rule: payload.search.selectionRule,
            epsilon: payload.search.epsilon,
            outer_folds: payload.search.outerFolds,
            outer_test_weeks: payload.search.outerTestWeeks,
            outer_gap_weeks: payload.search.outerGapWeeks,
            outer_selection_metric: payload.search.outerSelectionMetric,
            outer_min_improve_fraction: payload.search.outerMinImproveFraction,
            outer_max_worst_delta: payload.search.outerMaxWorstDelta,
            accept_delta_threshold: payload.search.acceptDeltaThreshold,
            min_improve_fraction: payload.search.minImproveFraction,
            max_worst_delta: payload.search.maxWorstDelta,
          }
        : undefined,
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

export async function cancelCalibrationJob(
  jobId: string,
): Promise<CalibrationJobStatus> {
  const response = await fetch(
    `${API_BASE}/calibrate-models/jobs/${jobId}/cancel`,
    { method: "POST" },
  );
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Cancel calibration job failed (${response.status}): ${detail || "unknown error"}`,
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
  relative_path?: string | null;
  section?: "selected_model" | "auto_search" | "legacy_root" | null;
  kind?: "file" | "dir" | null;
};

export type ModelFilesListResponse = {
  model_id: string;
  files: ModelFileSummary[];
};

export type ModelFileContentResponse = {
  model_id: string;
  filename: string;
  relative_path?: string | null;
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

export async function fetchModelFileContentByPath(
  modelId: string,
  relativePath: string,
): Promise<ModelFileContentResponse> {
  const response = await fetch(
    `${API_BASE}/calibrate-models/models/${encodeURIComponent(modelId)}/file-content?path=${encodeURIComponent(relativePath)}`,
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

export type WeightingPreviewRequest = {
  csv: string;
  weightColStrategy?: "auto" | "weight_final" | "sample_weight_final" | "uniform";
  baseWeightSource?: "dataset_weight" | "uniform";
  groupingKey?: string;
  groupEqualization?: boolean;
  tradingUniverseTickers?: string;
  tradingUniverseUpweight?: number;
  tickerBalanceMode?: "none" | "sqrt_inv_clipped";
  splitStrategy?: "walk_forward" | "single_holdout";
  testWindowWeeks?: number;
  validationWindowWeeks?: number;
};

export type WeightingPreviewResponse = {
  selected_weight_column: string | null;
  min_weight: number;
  mean_weight: number;
  max_weight: number;
  group_sum_min: number | null;
  group_sum_mean: number | null;
  group_sum_max: number | null;
  split_group_counts: Record<string, number>;
  split_row_counts: Record<string, number>;
  warnings: string[];
};

export async function previewCalibrationWeighting(
  payload: WeightingPreviewRequest,
): Promise<WeightingPreviewResponse> {
  const response = await fetch(`${API_BASE}/calibrate-models/weighting-preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      csv: payload.csv,
      weight_col_strategy: payload.weightColStrategy,
      base_weight_source: payload.baseWeightSource,
      grouping_key: payload.groupingKey,
      group_equalization: payload.groupEqualization,
      trading_universe_tickers: payload.tradingUniverseTickers,
      trading_universe_upweight: payload.tradingUniverseUpweight,
      ticker_balance_mode: payload.tickerBalanceMode,
      split_strategy: payload.splitStrategy,
      test_window_weeks: payload.testWindowWeeks,
      validation_window_weeks: payload.validationWindowWeeks,
    }),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Weighting preview failed (${response.status}): ${detail || "unknown error"}`,
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
