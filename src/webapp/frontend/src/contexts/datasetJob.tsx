import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { getDatasetJob } from "../api/datasets";

type DatasetJobStatus = Awaited<ReturnType<typeof getDatasetJob>>;

const STORAGE_KEY = "polyedgetool.datasets.job";

type DatasetJobContextValue = {
  jobId: string | null;
  jobStatus: DatasetJobStatus | null;
  setJobId: (value: string | null) => void;
  setJobStatus: (status: DatasetJobStatus | null) => void;
  refreshJob: () => Promise<void>;
};

const DatasetJobContext = createContext<DatasetJobContextValue | undefined>(
  undefined,
);

const loadStoredJobId = (): string | null => {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
};

export function DatasetJobProvider({
  children,
}: {
  children: ReactNode;
}) {
  const [jobId, setJobIdState] = useState<string | null>(() => loadStoredJobId());
  const [jobStatus, setJobStatusState] =
    useState<DatasetJobStatus | null>(null);
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
      const status = await getDatasetJob(jobId);
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
        const status = await getDatasetJob(jobId);
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
    <DatasetJobContext.Provider value={contextValue}>
      {children}
    </DatasetJobContext.Provider>
  );
}

export function useDatasetJob() {
  const context = useContext(DatasetJobContext);
  if (!context) {
    throw new Error("useDatasetJob must be used within DatasetJobProvider");
  }
  return context;
}
