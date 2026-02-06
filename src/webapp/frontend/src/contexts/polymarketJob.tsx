import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { getPolymarketSnapshotJob } from "../api/polymarketSnapshots";

type PolymarketJobStatus = Awaited<ReturnType<typeof getPolymarketSnapshotJob>>;

const STORAGE_KEY = "polyedgetool.polymarket.job";

type PolymarketJobContextValue = {
  jobId: string | null;
  jobStatus: PolymarketJobStatus | null;
  setJobId: (value: string | null) => void;
  setJobStatus: (status: PolymarketJobStatus | null) => void;
  refreshJob: () => Promise<void>;
};

const PolymarketJobContext =
  createContext<PolymarketJobContextValue | undefined>(undefined);

const loadStoredJobId = (): string | null => {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
};

export function PolymarketJobProvider({ children }: { children: ReactNode }) {
  const [jobId, setJobIdState] = useState<string | null>(() =>
    loadStoredJobId(),
  );
  const [jobStatus, setJobStatusState] =
    useState<PolymarketJobStatus | null>(null);
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
      const status = await getPolymarketSnapshotJob(jobId);
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
        const status = await getPolymarketSnapshotJob(jobId);
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
    <PolymarketJobContext.Provider value={contextValue}>
      {children}
    </PolymarketJobContext.Provider>
  );
}

export function usePolymarketJob() {
  const context = useContext(PolymarketJobContext);
  if (!context) {
    throw new Error(
      "usePolymarketJob must be used within PolymarketJobProvider",
    );
  }
  return context;
}
