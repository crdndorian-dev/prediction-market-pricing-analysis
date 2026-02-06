import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { getPhatEdgeJob } from "../api/phatEdge";

type PhatEdgeJobStatus = Awaited<ReturnType<typeof getPhatEdgeJob>>;

const STORAGE_KEY = "polyedgetool.phat-edge.job";

type PhatEdgeJobContextValue = {
  jobId: string | null;
  jobStatus: PhatEdgeJobStatus | null;
  setJobId: (value: string | null) => void;
  setJobStatus: (status: PhatEdgeJobStatus | null) => void;
  refreshJob: () => Promise<void>;
};

const PhatEdgeJobContext =
  createContext<PhatEdgeJobContextValue | undefined>(undefined);

const loadStoredJobId = (): string | null => {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
};

export function PhatEdgeJobProvider({ children }: { children: ReactNode }) {
  const [jobId, setJobIdState] = useState<string | null>(() =>
    loadStoredJobId(),
  );
  const [jobStatus, setJobStatusState] =
    useState<PhatEdgeJobStatus | null>(null);
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
      const status = await getPhatEdgeJob(jobId);
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
        const status = await getPhatEdgeJob(jobId);
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
    <PhatEdgeJobContext.Provider value={contextValue}>
      {children}
    </PhatEdgeJobContext.Provider>
  );
}

export function usePhatEdgeJob() {
  const context = useContext(PhatEdgeJobContext);
  if (!context) {
    throw new Error(
      "usePhatEdgeJob must be used within PhatEdgeJobProvider",
    );
  }
  return context;
}
