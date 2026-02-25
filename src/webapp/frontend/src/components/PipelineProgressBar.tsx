type PipelineProgress = {
  total: number;
  completed: number;
  failed: number;
  status: "running" | "completed" | "failed";
};

type PipelineProgressBarProps = {
  title: string;
  progress: PipelineProgress | null;
  running: boolean;
  runningLabel: string;
  idleLabel?: string;
  unitLabel?: string;
  forceError?: boolean;
};

export default function PipelineProgressBar({
  title,
  progress,
  running,
  runningLabel,
  idleLabel = "Ready",
  unitLabel = "jobs",
  forceError = false,
}: PipelineProgressBarProps) {
  const total = progress?.total ?? 0;
  const completed = progress?.completed ?? 0;
  const failed = progress?.failed ?? 0;
  const hasProgress = total > 0;
  const percent = hasProgress ? Math.round((completed / total) * 100) : 0;

  const stats = hasProgress
    ? `${completed} / ${total} ${unitLabel} completed (${percent}%)${
        failed > 0 ? ` \u2022 ${failed} failed` : ""
      }`
    : running
      ? runningLabel
      : idleLabel;

  const showIndeterminate = running && !hasProgress;
  const hasError = failed > 0 || forceError;
  const fillStyle =
    hasProgress ? { width: `${percent}%` } : showIndeterminate ? undefined : { width: "0%" };

  return (
    <div className={`progress-section${hasError ? " error" : ""}`}>
      <div className="progress-header">
        <span className="progress-label">{title}</span>
        <span className="progress-stats">{stats}</span>
      </div>
      <div className={`progress-bar${showIndeterminate ? " indeterminate" : ""}`}>
        <div
          className="progress-bar-fill"
          style={fillStyle}
        />
      </div>
    </div>
  );
}
