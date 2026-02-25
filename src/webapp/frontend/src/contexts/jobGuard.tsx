import { useMemo } from "react";

import { useCalibrationJob } from "./calibrationJob";
import { useDatasetJob } from "./datasetJob";
import { useMarketsJob } from "./marketsJob";
import { usePhatEdgeJob } from "./phatEdgeJob";
import { usePolymarketJob } from "./polymarketJob";
import { usePolymarketHistoryJob } from "./polymarketHistoryJob";
import { useMarketMapJob } from "./marketMapJob";

type JobDescriptor = {
  key: string;
  name: string;
  jobId: string | null;
  status: string | null;
};

const isRunningStatus = (status: string | null) =>
  status === "queued" || status === "running";

const parseMaxActiveJobs = (): number | null => {
  const raw = import.meta.env.VITE_MAX_ACTIVE_JOBS;
  if (!raw) return null;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) return null;
  const max = Math.floor(parsed);
  return max > 0 ? max : null;
};

export function useAnyJobRunning() {
  const dataset = useDatasetJob();
  const calibration = useCalibrationJob();
  const polymarket = usePolymarketJob();
  const polymarketHistory = usePolymarketHistoryJob();
  const phatEdge = usePhatEdgeJob();
  const marketMap = useMarketMapJob();
  const markets = useMarketsJob();

  const jobs: JobDescriptor[] = useMemo(
    () => [
      {
        key: "dataset",
        name: "Option chain dataset",
        jobId: dataset.jobId,
        status: dataset.jobStatus?.status ?? null,
      },
      {
        key: "calibration",
        name: "Calibrate models",
        jobId: calibration.jobId,
        status: calibration.jobStatus?.status ?? null,
      },
      {
        key: "polymarket",
        name: "Snapshot",
        jobId: polymarket.jobId,
        status: polymarket.jobStatus?.status ?? null,
      },
      {
        key: "polymarket-history",
        name: "Polymarket weekly history",
        jobId: polymarketHistory.jobId,
        status: polymarketHistory.jobStatus?.status ?? null,
      },
      {
        key: "phat-edge",
        name: "Edge compute",
        jobId: phatEdge.jobId,
        status: phatEdge.jobStatus?.status ?? null,
      },
      {
        key: "market-map",
        name: "Market map",
        jobId: marketMap.jobId,
        status: marketMap.jobStatus?.status ?? null,
      },
      {
        key: "markets",
        name: "Markets refresh",
        jobId: markets.jobId,
        status: markets.jobStatus?.status ?? null,
      },
    ],
    [
      dataset.jobId,
      dataset.jobStatus?.status,
      calibration.jobId,
      calibration.jobStatus?.status,
      polymarket.jobId,
      polymarket.jobStatus?.status,
      polymarketHistory.jobId,
      polymarketHistory.jobStatus?.status,
      phatEdge.jobId,
      phatEdge.jobStatus?.status,
      marketMap.jobId,
      marketMap.jobStatus?.status,
      markets.jobId,
      markets.jobStatus?.status,
    ],
  );

  const activeJobs = jobs.filter(
    (job) => job.jobId && isRunningStatus(job.status),
  );
  const maxActiveJobs = parseMaxActiveJobs();
  const anyJobRunning = maxActiveJobs ? activeJobs.length >= maxActiveJobs : false;
  const primaryJob = anyJobRunning ? activeJobs[0] ?? null : null;

  return { anyJobRunning, activeJobs, primaryJob, maxActiveJobs };
}
