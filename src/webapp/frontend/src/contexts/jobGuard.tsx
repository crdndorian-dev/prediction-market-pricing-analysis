import { useMemo } from "react";

import { useCalibrationJob } from "./calibrationJob";
import { useDatasetJob } from "./datasetJob";
import { usePhatEdgeJob } from "./phatEdgeJob";
import { usePolymarketJob } from "./polymarketJob";

type JobDescriptor = {
  key: string;
  name: string;
  jobId: string | null;
  status: string | null;
};

const isRunningStatus = (status: string | null) =>
  status === "queued" || status === "running";

export function useAnyJobRunning() {
  const dataset = useDatasetJob();
  const calibration = useCalibrationJob();
  const polymarket = usePolymarketJob();
  const phatEdge = usePhatEdgeJob();

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
        name: "Polymarket snapshot",
        jobId: polymarket.jobId,
        status: polymarket.jobStatus?.status ?? null,
      },
      {
        key: "phat-edge",
        name: "Edge compute",
        jobId: phatEdge.jobId,
        status: phatEdge.jobStatus?.status ?? null,
      },
    ],
    [
      dataset.jobId,
      dataset.jobStatus?.status,
      calibration.jobId,
      calibration.jobStatus?.status,
      polymarket.jobId,
      polymarket.jobStatus?.status,
      phatEdge.jobId,
      phatEdge.jobStatus?.status,
    ],
  );

  const activeJobs = jobs.filter(
    (job) => job.jobId && isRunningStatus(job.status),
  );
  const anyJobRunning = activeJobs.length > 0;
  const primaryJob = activeJobs[0] ?? null;

  return { anyJobRunning, activeJobs, primaryJob };
}
