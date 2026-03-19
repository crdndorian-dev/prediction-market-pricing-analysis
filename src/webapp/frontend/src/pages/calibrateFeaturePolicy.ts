export const RETIRED_OPTION_CHAIN_FEATURES = [
  "quality_issue_count",
  "dropped_liquidity",
  "dropped_intrinsic",
  "dropped_insane",
  "prn_monotone_adj_intervals",
  "prn_monotone_adj_targets",
  "flag_asof_close_fallback",
  "flag_expiry_close_fallback",
  "flag_expiry_saturday_fallback",
  "flag_quote_close_fallback",
  "flag_split_event_context",
  "flag_prn_monotone_adjusted",
  "flag_wide_rel_spread",
  "flag_low_chain_used",
  "flag_thin_prn_band",
  "flag_missing_rv",
  "flag_band_edge",
] as const;

type FeatureSelectionInputs = {
  selectedFeatures: string[];
  selectedCategoricalFeatures: string[];
};

type FeatureSelectionSanitization = {
  selectedFeatures: string[];
  selectedCategoricalFeatures: string[];
  removedFeatures: string[];
};

const RETIRED_OPTION_CHAIN_FEATURE_SET = new Set<string>(RETIRED_OPTION_CHAIN_FEATURES);
const DIAGNOSTICS_SKIP_WARNING_PREFIX = "Stata diagnostics skipped:";

const dedupeStrings = (values: string[]): string[] => {
  const seen = new Set<string>();
  const deduped: string[] = [];
  values.forEach((value) => {
    const cleaned = value.trim();
    if (!cleaned || seen.has(cleaned)) return;
    seen.add(cleaned);
    deduped.push(cleaned);
  });
  return deduped;
};

export const stripRetiredFeatureSelections = ({
  selectedFeatures,
  selectedCategoricalFeatures,
}: FeatureSelectionInputs): FeatureSelectionSanitization => {
  const numeric = dedupeStrings(selectedFeatures);
  const categorical = dedupeStrings(selectedCategoricalFeatures);
  const removedFeatures = dedupeStrings(
    [...numeric, ...categorical].filter((feature) => RETIRED_OPTION_CHAIN_FEATURE_SET.has(feature)),
  );
  return {
    selectedFeatures: numeric.filter((feature) => !RETIRED_OPTION_CHAIN_FEATURE_SET.has(feature)),
    selectedCategoricalFeatures: categorical.filter(
      (feature) => !RETIRED_OPTION_CHAIN_FEATURE_SET.has(feature),
    ),
    removedFeatures,
  };
};

export const buildRetiredFeatureNotice = (
  removedFeatures: string[],
  sourceLabel: string,
): string | null => {
  const removed = dedupeStrings(removedFeatures);
  if (!removed.length) return null;
  const plural = removed.length === 1 ? "feature" : "features";
  return `${sourceLabel} removed retired quality-derived ${plural}: ${removed.join(", ")}.`;
};

export const coerceWarningList = (value: unknown): string[] => {
  if (!Array.isArray(value)) return [];
  return value
    .map((entry) => (typeof entry === "string" ? entry.trim() : String(entry).trim()))
    .filter(Boolean);
};

export const findDiagnosticsSkipWarning = (
  warnings: readonly string[] | null | undefined,
): string | null => {
  if (!warnings?.length) return null;
  return warnings.find((warning) => warning.startsWith(DIAGNOSTICS_SKIP_WARNING_PREFIX)) ?? null;
};
