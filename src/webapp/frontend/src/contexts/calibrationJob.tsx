import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { getCalibrationJob } from "../api/calibrateModels";

type CalibrationJobStatus = Awaited<ReturnType<typeof getCalibrationJob>>;

const STORAGE_KEY = "polyedgetool.calibration.job";

type CalibrationJobContextValue = {
  jobId: string | null;
  jobStatus: CalibrationJobStatus | null;
  setJobId: (value: string | null) => void;
  setJobStatus: (status: CalibrationJobStatus | null) => void;
  refreshJob: () => Promise<void>;
};

const CalibrationJobContext =
  createContext<CalibrationJobContextValue | undefined>(undefined);

const loadStoredJobId = (): string | null => {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
};

export function CalibrationJobProvider({ children }: { children: ReactNode }) {
  const [jobId, setJobIdState] = useState<string | null>(() =>
    loadStoredJobId(),
  );
  const [jobStatus, setJobStatusState] =
    useState<CalibrationJobStatus | null>(null);
  const [storageReady, setStorageReady] = useState(false);

  useEffect(() => {
    setStorageReady(true);
  }, []);

  const setJobId = useCallback(
    (value: string | null) => {
      setJobIdState(value);
      if (!storageReady) return;
      try {
        if (value) {
          localStorage.setItem(STORAGE_KEY, value);
        } else {
          localStorage.removeItem(STORAGE_KEY);
        }
      } catch {
        // ignore storage failures
      }
    },
    [storageReady],
  );

  const refreshJob = useCallback(async () => {
    if (!jobId) {
      setJobStatusState(null);
      return;
    }
    try {
      const status = await getCalibrationJob(jobId);
      setJobStatusState(status);
    } catch {
      setJobStatusState(null);
      setJobId(null);
      try {
        localStorage.removeItem(STORAGE_KEY);
      } catch {
        /* ignore */
      }
    }
  }, [jobId, setJobId]);

  useEffect(() => {
    if (!jobId) {
      setJobStatusState(null);
      return undefined;
    }

    let cancelled = false;
    const poll = async () => {
      try {
        const status = await getCalibrationJob(jobId);
        if (cancelled) return;
        setJobStatusState(status);
      } catch {
        if (cancelled) return;
        setJobStatusState(null);
        setJobId(null);
        try {
          localStorage.removeItem(STORAGE_KEY);
        } catch {
          /* ignore */
        }
      }
    };

    poll();
    const id = setInterval(poll, 2000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [jobId, setJobId]);

  const contextValue = useMemo(
    () => ({
      jobId,
      jobStatus,
      setJobId,
      setJobStatus: setJobStatusState,
      refreshJob,
    }),
    [jobId, jobStatus, setJobId, refreshJob],
  );

  return (
    <CalibrationJobContext.Provider value={contextValue}>
      {children}
    </CalibrationJobContext.Provider>
  );
}

export function useCalibrationJob() {
  const context = useContext(CalibrationJobContext);
  if (!context) {
    throw new Error(
      "useCalibrationJob must be used within CalibrationJobProvider",
    );
  }
  return context;
}
