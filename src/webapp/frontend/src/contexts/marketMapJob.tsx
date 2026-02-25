import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { getMarketMapJob } from "../api/marketMap";

type MarketMapJobStatus = Awaited<ReturnType<typeof getMarketMapJob>>;

const STORAGE_KEY = "polyedgetool.marketMap.job";

type MarketMapJobContextValue = {
  jobId: string | null;
  jobStatus: MarketMapJobStatus | null;
  setJobId: (value: string | null) => void;
  setJobStatus: (status: MarketMapJobStatus | null) => void;
  refreshJob: () => Promise<void>;
};

const MarketMapJobContext =
  createContext<MarketMapJobContextValue | undefined>(undefined);

const loadStoredJobId = (): string | null => {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
};

export function MarketMapJobProvider({ children }: { children: ReactNode }) {
  const [jobId, setJobIdState] = useState<string | null>(() =>
    loadStoredJobId(),
  );
  const [jobStatus, setJobStatusState] =
    useState<MarketMapJobStatus | null>(null);
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
      const status = await getMarketMapJob(jobId);
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
        const status = await getMarketMapJob(jobId);
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
    <MarketMapJobContext.Provider value={contextValue}>
      {children}
    </MarketMapJobContext.Provider>
  );
}

export function useMarketMapJob() {
  const context = useContext(MarketMapJobContext);
  if (!context) {
    throw new Error("useMarketMapJob must be used within MarketMapJobProvider");
  }
  return context;
}
