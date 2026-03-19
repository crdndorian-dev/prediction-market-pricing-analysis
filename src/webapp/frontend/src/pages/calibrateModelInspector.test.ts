import { describe, expect, it } from "vitest";

import type { ModelFileSummary } from "../api/calibrateModels";
import { buildModelInspectorArtifactState } from "./calibrateModelInspector";

const file = (
  name: string,
  overrides: Partial<ModelFileSummary> = {},
): ModelFileSummary => ({
  name,
  size_bytes: 1024,
  is_viewable: true,
  section: "legacy_root",
  ...overrides,
});

describe("buildModelInspectorArtifactState", () => {
  it("keeps diagnostics selectable inside its section", () => {
    const state = buildModelInspectorArtifactState([
      file("metrics.csv", { section: "selected_model" }),
      file("diagnostics_table.json", { section: "selected_model" }),
      file("metadata.json", { section: "selected_model" }),
    ]);

    expect(state.sections).toHaveLength(1);
    expect(state.sections[0].id).toBe("selected_model");
    expect(state.sections[0].files.map((entry) => entry.name)).toEqual([
      "metrics.csv",
      "diagnostics_table.json",
      "metadata.json",
    ]);
  });

  it("filters hidden auto-search artifacts from the sidebar", () => {
    const state = buildModelInspectorArtifactState([
      file("auto_search_no_viable.json", { section: "auto_search" }),
      file("auto_search_summary.json", { section: "auto_search" }),
    ]);

    expect(state.sections).toHaveLength(1);
    expect(state.sections[0].id).toBe("auto_search");
    expect(state.sections[0].files.map((entry) => entry.name)).toEqual(["auto_search_summary.json"]);
  });
});
