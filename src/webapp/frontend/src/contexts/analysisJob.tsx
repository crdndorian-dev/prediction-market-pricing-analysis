import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { getAnalysisJob } from "../api/analysis";

type AnalysisJobStatus = Awaited<ReturnType<typeof getAnalysisJob>>;

const STORAGE_KEY = "polyedgetool.analysis.job";

type AnalysisJobContextValue = {
  jobId: string | null;
  jobStatus: AnalysisJobStatus | null;
  setJobId: (value: string | null) => void;
  setJobStatus: (status: AnalysisJobStatus | null) => void;
  refreshJob: () => Promise<void>;
};

const AnalysisJobContext = createContext<AnalysisJobContextValue | undefined>(
  undefined,
);

const loadStoredJobId = (): string | null => {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
};

export function AnalysisJobProvider({ children }: { children: ReactNode }) {
  const [jobId, setJobIdState] = useState<string | null>(() => loadStoredJobId());
  const [jobStatus, setJobStatusState] = useState<AnalysisJobStatus | null>(null);
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
      const status = await getAnalysisJob(jobId);
      setJobStatusState(status);
      if (status.status === "finished" || status.status === "failed" || status.status === "cancelled") {
        setJobId(null);
      }
    } catch {
      setJobStatusState(null);
      setJobId(null);
      try {
        localStorage.removeItem(STORAGE_KEY);
      } catch {
        // ignore storage failures
      }
    }
  }, [jobId, setJobId]);

  useEffect(() => {
    if (!jobId) {
      return undefined;
    }

    let cancelled = false;
    const poll = async () => {
      try {
        const status = await getAnalysisJob(jobId);
        if (cancelled) return;
        setJobStatusState(status);
        if (status.status === "finished" || status.status === "failed" || status.status === "cancelled") {
          setJobId(null);
        }
      } catch {
        if (cancelled) return;
        setJobStatusState(null);
        setJobId(null);
        try {
          localStorage.removeItem(STORAGE_KEY);
        } catch {
          // ignore storage failures
        }
      }
    };

    void poll();
    const id = window.setInterval(() => {
      void poll();
    }, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
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
    [jobId, jobStatus, refreshJob, setJobId],
  );

  return (
    <AnalysisJobContext.Provider value={contextValue}>
      {children}
    </AnalysisJobContext.Provider>
  );
}

export function useAnalysisJob() {
  const context = useContext(AnalysisJobContext);
  if (!context) {
    throw new Error("useAnalysisJob must be used within AnalysisJobProvider");
  }
  return context;
}
