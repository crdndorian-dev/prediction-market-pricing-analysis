const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export type AnalysisJobStatus = {
  job_id: string;
  status: "queued" | "running" | "finished" | "failed" | "cancelled";
  started_at?: string | null;
  finished_at?: string | null;
  result?: unknown;
  error?: string | null;
};

export async function getAnalysisJob(jobId: string): Promise<AnalysisJobStatus> {
  const response = await fetch(`${API_BASE}/analysis/jobs/${encodeURIComponent(jobId)}`);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Analysis job lookup failed (${response.status}): ${detail || "unknown error"}`);
  }
  return response.json();
}
