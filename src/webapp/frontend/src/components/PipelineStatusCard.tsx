import { Link } from "react-router-dom";

type PipelineStatusCardProps = {
  className?: string;
  activeJobsCount: number;
};

export default function PipelineStatusCard({
  className,
  activeJobsCount,
}: PipelineStatusCardProps) {
  const statusLabel = activeJobsCount > 0 ? "Running" : "Idle";
  const statusClass = activeJobsCount > 0 ? "running" : "idle";
  const countLabel = `${activeJobsCount} job${
    activeJobsCount === 1 ? "" : "s"
  } running`;
  const classes = ["meta-card", "page-goal-card", "pipeline-status-card", className]
    .filter((value): value is string => Boolean(value))
    .join(" ");

  return (
    <div className={classes}>
      <span className="pipeline-group">
        <span className="pipeline-bracket">[</span>
        <span className={`status-pill ${statusClass}`}>{statusLabel}</span>
        <span className="pipeline-separator">:</span>
        <span className="pipeline-count">{countLabel}</span>
        <span className="pipeline-bracket">]</span>
      </span>
      <span className="pipeline-divider">|</span>
      <Link className="meta-pill meta-link" to="/">
        View run queue
      </Link>
    </div>
  );
}
