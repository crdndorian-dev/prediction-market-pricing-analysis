import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { getMarketsJob, type MarketsJobStatus } from "../api/markets";

const STORAGE_KEY = "polyedgetool.markets.job";

type MarketsJobContextValue = {
  jobId: string | null;
  jobStatus: MarketsJobStatus | null;
  setJobId: (value: string | null) => void;
  setJobStatus: (status: MarketsJobStatus | null) => void;
};

const MarketsJobContext = createContext<MarketsJobContextValue | undefined>(undefined);

const loadStoredJobId = (): string | null => {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
};

export function MarketsJobProvider({ children }: { children: ReactNode }) {
  const [jobId, setJobIdState] = useState<string | null>(() => loadStoredJobId());
  const [jobStatus, setJobStatusState] = useState<MarketsJobStatus | null>(null);
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

  // Polling lives here at the root level — survives page navigation
  useEffect(() => {
    if (!jobId) return;

    let cancelled = false;
    const poll = async () => {
      try {
        const status = await getMarketsJob(jobId);
        if (cancelled) return;
        setJobStatusState(status);
        // Auto-stop polling on terminal state; keep jobStatus so the page can react
        if (status.status === "finished" || status.status === "failed") {
          setJobId(null);
        }
      } catch {
        if (cancelled) return;
        setJobId(null);
      }
    };

    poll();
    const id = setInterval(poll, 1000);
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
    }),
    [jobId, jobStatus, setJobId],
  );

  return (
    <MarketsJobContext.Provider value={contextValue}>
      {children}
    </MarketsJobContext.Provider>
  );
}

export function useMarketsJob() {
  const context = useContext(MarketsJobContext);
  if (!context) {
    throw new Error("useMarketsJob must be used within MarketsJobProvider");
  }
  return context;
}
