import { describe, expect, it } from "vitest";

import {
  buildRetiredFeatureNotice,
  coerceWarningList,
  findDiagnosticsSkipWarning,
  stripRetiredFeatureSelections,
} from "./calibrateFeaturePolicy";

describe("calibrateFeaturePolicy", () => {
  it("strips retired quality-derived features from stored selections", () => {
    expect(
      stripRetiredFeatureSelections({
        selectedFeatures: [
          "log_m_fwd",
          "quality_issue_count",
          "dropped_liquidity",
          "quality_issue_count",
        ],
        selectedCategoricalFeatures: ["spot_scale_used", "flag_asof_close_fallback"],
      }),
    ).toEqual({
      selectedFeatures: ["log_m_fwd"],
      selectedCategoricalFeatures: ["spot_scale_used"],
      removedFeatures: [
        "quality_issue_count",
        "dropped_liquidity",
        "flag_asof_close_fallback",
      ],
    });
  });

  it("builds a coherent retirement notice", () => {
    expect(
      buildRetiredFeatureNotice(
        ["quality_issue_count", "dropped_liquidity"],
        "Stored selection",
      ),
    ).toBe(
      "Stored selection removed retired quality-derived features: quality_issue_count, dropped_liquidity.",
    );
  });

  it("extracts the diagnostics skip warning from existing warning surfaces", () => {
    const warnings = coerceWarningList([
      "Validation split has fewer than 30 groups; CI estimates may be unstable.",
      "Stata diagnostics skipped: statsmodels dependency unavailable.",
    ]);
    expect(findDiagnosticsSkipWarning(warnings)).toBe(
      "Stata diagnostics skipped: statsmodels dependency unavailable.",
    );
  });
});
