import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

type PipelineStatusCardProps = {
  className?: string;
  activeJobsCount: number;
};

export default function PipelineStatusCard({
  className,
  activeJobsCount,
}: PipelineStatusCardProps) {
  const [isScrolled, setIsScrolled] = useState(false);
  const statusLabel = activeJobsCount > 0 ? "Running" : "Idle";
  const statusClass = activeJobsCount > 0 ? "running" : "idle";
  const countLabel = `${activeJobsCount} job${
    activeJobsCount === 1 ? "" : "s"
  } running`;
  const classes = [
    "meta-card",
    "page-goal-card",
    "pipeline-status-card",
    isScrolled ? "is-dim" : null,
    className,
  ]
    .filter((value): value is string => Boolean(value))
    .join(" ");

  useEffect(() => {
    const handleScroll = () => {
      const next = window.scrollY > 80;
      setIsScrolled((prev) => (prev === next ? prev : next));
    };

    handleScroll();
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  return (
    <Link className={classes} to="/" aria-label="View run queue on dashboard">
      <div className="pipeline-status-main">
        <span className={`status-pill ${statusClass}`}>{statusLabel}</span>
        <span className="pipeline-count">{countLabel}</span>
      </div>
      <span className="pipeline-status-action">View run queue</span>
    </Link>
  );
}
