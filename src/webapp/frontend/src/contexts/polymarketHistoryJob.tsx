import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { getPolymarketHistoryJob } from "../api/polymarketHistory";

type PolymarketHistoryJobStatus = Awaited<ReturnType<typeof getPolymarketHistoryJob>>;

const STORAGE_KEY = "polyedgetool.polymarketHistory.job";

type PolymarketHistoryJobContextValue = {
  jobId: string | null;
  jobStatus: PolymarketHistoryJobStatus | null;
  setJobId: (value: string | null) => void;
  setJobStatus: (status: PolymarketHistoryJobStatus | null) => void;
  refreshJob: () => Promise<void>;
};

const PolymarketHistoryJobContext =
  createContext<PolymarketHistoryJobContextValue | undefined>(undefined);

const loadStoredJobId = (): string | null => {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
};

export function PolymarketHistoryJobProvider({ children }: { children: ReactNode }) {
  const [jobId, setJobIdState] = useState<string | null>(() => loadStoredJobId());
  const [jobStatus, setJobStatusState] = useState<PolymarketHistoryJobStatus | null>(null);
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
      const status = await getPolymarketHistoryJob(jobId);
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
        const status = await getPolymarketHistoryJob(jobId);
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
    const id = setInterval(poll, 500);
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
    <PolymarketHistoryJobContext.Provider value={contextValue}>
      {children}
    </PolymarketHistoryJobContext.Provider>
  );
}

export function usePolymarketHistoryJob() {
  const context = useContext(PolymarketHistoryJobContext);
  if (!context) {
    throw new Error(
      "usePolymarketHistoryJob must be used within PolymarketHistoryJobProvider",
    );
  }
  return context;
}
