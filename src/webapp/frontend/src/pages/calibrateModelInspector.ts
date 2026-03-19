import type { ModelFileSummary } from "../api/calibrateModels";

export type ModelArtifactGroupId = "selected_model" | "auto_search" | "legacy_root";

export type ModelArtifactSection = {
  id: ModelArtifactGroupId;
  title: string;
  files: ModelFileSummary[];
};

type ModelInspectorArtifactState = {
  sections: ModelArtifactSection[];
};

const HIDDEN_ARTIFACT_NAMES = new Set<string>(["auto_search_no_viable.json"]);
const GROUP_ORDER: ModelArtifactGroupId[] = ["selected_model", "auto_search", "legacy_root"];
const GROUP_TITLES: Record<ModelArtifactGroupId, string> = {
  selected_model: "Selected model",
  auto_search: "Auto search",
  legacy_root: "Legacy root",
};

const fileBaseName = (path: string | null | undefined): string => {
  if (!path) return "";
  const normalized = path.replace(/\\/g, "/");
  const parts = normalized.split("/");
  return parts[parts.length - 1] || normalized;
};

const artifactFilePath = (file: ModelFileSummary): string => file.relative_path ?? file.name;

const normalizeGroupId = (section: ModelFileSummary["section"]): ModelArtifactGroupId =>
  section === "selected_model" || section === "auto_search" ? section : "legacy_root";

const isHiddenArtifact = (file: ModelFileSummary): boolean =>
  HIDDEN_ARTIFACT_NAMES.has(fileBaseName(artifactFilePath(file)));

export const buildModelInspectorArtifactState = (
  files: ModelFileSummary[] | null | undefined,
): ModelInspectorArtifactState => {
  const sourceFiles = files ?? [];
  const groupedFiles: Record<ModelArtifactGroupId, ModelFileSummary[]> = {
    selected_model: [],
    auto_search: [],
    legacy_root: [],
  };

  for (const file of sourceFiles) {
    groupedFiles[normalizeGroupId(file.section)].push(file);
  }

  const sections = GROUP_ORDER.map((groupId) => {
    const visibleFiles = groupedFiles[groupId].filter((file) => !isHiddenArtifact(file));

    return {
      id: groupId,
      title: GROUP_TITLES[groupId],
      files: visibleFiles,
    };
  }).filter((section) => section.files.length > 0);

  return {
    sections,
  };
};
