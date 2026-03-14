import {
  Fragment,
  useCallback,
  useEffect,
  useMemo,
  useState,
  type FormEvent,
} from "react";

import {
  applyDatasetCleanup,
  auditDatasetFile,
  deleteDatasetRun,
  getDatasetFileUrl,
  killDatasetJob,
  listDatasetRuns,
  previewDatasetCleanup,
  previewDatasetFile,
  renameDatasetRun,
  startDatasetJob,
  type DatasetAuditDistribution,
  type DatasetAuditResponse,
  type DatasetAuditTickerSummary,
  type DatasetCleanupPreviewResponse,
  type DatasetJobGroupChecks,
  type DatasetJobStatus,
  type DatasetJobTelemetry,
  type DatasetJobTickerTelemetry,
  type DatasetFileSummary,
  type DatasetPreviewResponse,
  type DatasetRunResponse,
  type DatasetRunSummary,
} from "../api/datasets";
import PipelineStatusCard from "../components/PipelineStatusCard";
import { useDatasetJob } from "../contexts/datasetJob";
import { useAnyJobRunning } from "../contexts/jobGuard";
import "./DatasetsPage.css";

type DatasetFormState = {
  outDir: string;
  datasetName: string;
  scheduleMode: "weekly" | "expiry_range";
  expiryWeekdays: string;
  asofWeekdays: string;
  dteList: string;
  dteMin: string;
  dteMax: string;
  dteStep: string;
  writeSnapshot: boolean;
  writePrnView: boolean;
  writeTrainView: boolean;
  writeLegacy: boolean;
  prnVersion: string;
  prnConfigHash: string;
  tickers: string;
  start: string;
  end: string;
  thetaBaseUrl: string;
  stockSource: "yfinance" | "theta" | "auto";
  timeoutS: string;
  riskFreeRate: string;
  maxAbsLogm: string;
  maxAbsLogmCap: string;
  bandWidenStep: string;
  adaptiveBand: boolean;
  maxBandStrikes: string;
  minBandStrikes: string;
  minBandPrnStrikes: string;
  strikeRange: string;
  retryFullChain: boolean;
  saturdayExpiryFallback: boolean;
  threads: string;
  preferBidask: boolean;
  minTradeCount: string;
  minVolume: string;
  minChainUsedHard: string;
  maxRelSpreadMedianHard: string;
  hardDropCloseFallback: boolean;
  minPrnTrain: string;
  maxPrnTrain: string;
  splitAdjust: boolean;
  dividendSource: "yfinance" | "none";
  dividendLookbackDays: string;
  dividendYieldDefault: string;
  forwardMoneyness: boolean;
  groupWeights: boolean;
  tickerWeights: boolean;
  softQualityWeight: boolean;
  rvLookbackDays: string;
  cache: boolean;
  writeDrops: boolean;
  sanityReport: boolean;
  sanityDrop: boolean;
  sanityAbsLogmMax: string;
  sanityKOverSMin: string;
  sanityKOverSMax: string;
  verboseSkips: boolean;
};

const TRADING_UNIVERSE_TICKERS = [
  "AAPL",
  "GOOGL",
  "MSFT",
  "META",
  "AMZN",
  "PLTR",
  "NVDA",
  "TSLA",
  "NFLX",
  "OPEN",
];

const defaultForm: DatasetFormState = {
  outDir: "src/data/raw/option-chain",
  datasetName: "",
  scheduleMode: "weekly",
  expiryWeekdays: "fri",
  asofWeekdays: "mon,tue,wed,thu",
  dteList: "",
  dteMin: "",
  dteMax: "",
  dteStep: "1",
  writeSnapshot: true,
  writePrnView: true,
  writeTrainView: true,
  writeLegacy: true,
  prnVersion: "v1",
  prnConfigHash: "",
  tickers: TRADING_UNIVERSE_TICKERS.join(", "),
  start: "",
  end: "",
  thetaBaseUrl: "http://127.0.0.1:25503/v3",
  stockSource: "yfinance",
  timeoutS: "30",
  riskFreeRate: "0.03",
  maxAbsLogm: "0.06",
  maxAbsLogmCap: "0.10",
  bandWidenStep: "0.01",
  adaptiveBand: true,
  maxBandStrikes: "0",
  minBandStrikes: "10",
  minBandPrnStrikes: "7",
  strikeRange: "60",
  retryFullChain: true,
  saturdayExpiryFallback: true,
  threads: "6",
  preferBidask: true,
  minTradeCount: "0",
  minVolume: "0",
  minChainUsedHard: "0",
  maxRelSpreadMedianHard: "1000000000",
  hardDropCloseFallback: false,
  minPrnTrain: "0.10",
  maxPrnTrain: "0.90",
  splitAdjust: true,
  dividendSource: "yfinance",
  dividendLookbackDays: "365",
  dividendYieldDefault: "0.0",
  forwardMoneyness: true,
  groupWeights: true,
  tickerWeights: true,
  softQualityWeight: true,
  rvLookbackDays: "20",
  cache: true,
  writeDrops: false,
  sanityReport: false,
  sanityDrop: false,
  sanityAbsLogmMax: "0.40",
  sanityKOverSMin: "0.25",
  sanityKOverSMax: "4.0",
  verboseSkips: false,
};

const toKebabCase = (value: string) =>
  value
    .trim()
    .toLowerCase()
    .replace(/[\s_]+/g, "-")
    .replace(/[^a-z0-9-]/g, "")
    .replace(/-{2,}/g, "-")
    .replace(/^-|-$/g, "");

const STORAGE_KEY = "polyedgetool.datasets.form";
const CALIBRATE_STORAGE_KEY = "polyedgetool.calibrate.form";

const parseOptionalNumber = (value: string): number | undefined => {
  const trimmed = value.trim();
  if (!trimmed) return undefined;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : undefined;
};

const parseOptionalInt = (value: string): number | undefined => {
  const trimmed = value.trim();
  if (!trimmed) return undefined;
  const parsed = Number.parseInt(trimmed, 10);
  return Number.isFinite(parsed) ? parsed : undefined;
};

const loadStoredForm = (): Partial<DatasetFormState> | null => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    // Migration: convert old fields to datasetName
    if (!parsed.datasetName && parsed.runDirName) {
      parsed.datasetName = parsed.runDirName;
    }
    parsed.writeTrainView = true;
    delete parsed.outName;
    delete parsed.runDirName;
    delete parsed.trainViewName;
    delete parsed.trainingDataset;
    return parsed as Partial<DatasetFormState>;
  } catch {
    return null;
  }
};

const parseTickers = (raw: string): string[] | undefined => {
  const cleaned = raw
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  return cleaned.length > 0 ? cleaned : undefined;
};

const normalizeTickers = (values: string[]): string[] => {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values) {
    const normalized = value.trim().toUpperCase();
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
};

const orderTickers = (values: string[]): string[] => {
  const universeSet = new Set(TRADING_UNIVERSE_TICKERS);
  const universeOrdered = TRADING_UNIVERSE_TICKERS.filter((ticker) =>
    values.includes(ticker),
  );
  const extras = values.filter((ticker) => !universeSet.has(ticker));
  return [...universeOrdered, ...extras];
};

const formatTickerList = (values: string[]): string => values.join(", ");

const splitTickerInput = (raw: string): string[] => {
  return raw
    .split(/[\s,]+/)
    .map((value) => value.trim())
    .filter(Boolean);
};

const WEEKDAY_MAP: Record<string, number> = {
  mon: 1,
  monday: 1,
  tue: 2,
  tues: 2,
  tuesday: 2,
  wed: 3,
  weds: 3,
  wednesday: 3,
  thu: 4,
  thur: 4,
  thurs: 4,
  thursday: 4,
  fri: 5,
  friday: 5,
  sat: 6,
  saturday: 6,
  sun: 0,
  sunday: 0,
};

const parseWeekdays = (raw: string): number[] | null => {
  const cleaned = raw
    .split(",")
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean);
  if (cleaned.length === 0) return null;
  const out: number[] = [];
  const seen = new Set<number>();
  for (const item of cleaned) {
    let day: number | undefined;
    if (/^\d+$/.test(item)) {
      const scriptDay = Number.parseInt(item, 10);
      day = Number.isFinite(scriptDay) ? (scriptDay + 1) % 7 : undefined;
    } else {
      day = WEEKDAY_MAP[item];
    }
    if (day === undefined || Number.isNaN(day)) return null;
    if (!seen.has(day)) {
      seen.add(day);
      out.push(day);
    }
  }
  return out;
};

const countWeekdaysInRange = (
  start: string,
  end: string,
  weekdays: number[] | null,
): number | null => {
  if (!start || !end || !weekdays || weekdays.length === 0) return null;
  const startDate = new Date(`${start}T00:00:00Z`);
  const endDate = new Date(`${end}T00:00:00Z`);
  if (Number.isNaN(startDate.getTime()) || Number.isNaN(endDate.getTime())) {
    return null;
  }
  if (endDate < startDate) return null;
  const daySet = new Set(weekdays);
  let count = 0;
  const cursor = new Date(startDate);
  while (cursor <= endDate) {
    if (daySet.has(cursor.getUTCDay())) count += 1;
    cursor.setUTCDate(cursor.getUTCDate() + 1);
  }
  return count;
};

const resolveDteCount = (
  listRaw: string,
  minRaw: string,
  maxRaw: string,
  stepRaw: string,
): number | null => {
  const parts = listRaw
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  if (parts.length > 0) {
    const values = new Set<number>();
    for (const part of parts) {
      if (part.includes("-")) {
        const [startStr, endStr] = part.split("-", 2);
        const start = Number.parseInt(startStr, 10);
        const end = Number.parseInt(endStr, 10);
        if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
        const lo = Math.min(start, end);
        const hi = Math.max(start, end);
        for (let v = lo; v <= hi; v += 1) values.add(v);
      } else {
        const v = Number.parseInt(part, 10);
        if (!Number.isFinite(v)) return null;
        values.add(v);
      }
    }
    return values.size;
  }

  const minVal = minRaw.trim() ? Number.parseInt(minRaw, 10) : null;
  const maxVal = maxRaw.trim() ? Number.parseInt(maxRaw, 10) : null;
  if (minVal === null && maxVal === null) return null;
  const step = stepRaw.trim() ? Number.parseInt(stepRaw, 10) : 1;
  if (!Number.isFinite(step) || step <= 0) return null;
  const lo = minVal ?? 0;
  const hi = maxVal ?? lo;
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi < lo) return null;
  return Math.floor((hi - lo) / step) + 1;
};

const formatByteCount = (bytes?: number | null): string => {
  if (!bytes) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(1)} MB`;
};

const formatTimestamp = (value?: string | null): string =>
  value ? new Date(value).toLocaleString() : "Unknown";

const formatPercent = (value?: number | null): string =>
  value == null || !Number.isFinite(value) ? "—" : `${(value * 100).toFixed(1)}%`;

const formatNumeric = (value?: number | null, digits = 2): string =>
  value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits);

const shareBarWidth = (share?: number | null, minVisiblePercent = 6): string => {
  const percent = (share ?? 0) * 100;
  if (!Number.isFinite(percent) || percent <= 0) return "0%";
  return `${Math.max(minVisiblePercent, percent)}%`;
};

type HeatmapMetric = "rows" | "issues" | "flagged";

const HEATMAP_METRIC_OPTIONS: { value: HeatmapMetric; label: string }[] = [
  { value: "rows", label: "Coverage" },
  { value: "issues", label: "Issue load" },
  { value: "flagged", label: "Flagged share" },
];

type CleanupCriteriaFormState = {
  mode: "quality_buckets" | "flags";
  qualityBuckets: Array<"clean" | "watch" | "noisy">;
  selectedFlags: string[];
  flagMatchMode: "any" | "all";
};

type CleanupModalTarget = {
  runId: string;
  runDir: string;
  runName: string;
  trainingFile: DatasetFileSummary;
};

const DEFAULT_CLEANUP_CRITERIA_FORM: CleanupCriteriaFormState = {
  mode: "quality_buckets",
  qualityBuckets: ["noisy"],
  selectedFlags: [],
  flagMatchMode: "any",
};

type ProblemTickerCardItem = {
  ticker: string;
  row_count: number;
  snapshot_count?: number | null;
  avg_issue_count?: number | null;
  flagged_share?: number | null;
  fallback_share?: number | null;
  wide_spread_share?: number | null;
  clean_share?: number | null;
  watch_share?: number | null;
  noisy_share?: number | null;
};

const compareProblemTickerCardItems = (
  a: ProblemTickerCardItem,
  b: ProblemTickerCardItem,
): number => {
  const issueDiff = (b.avg_issue_count ?? -1) - (a.avg_issue_count ?? -1);
  if (issueDiff !== 0) return issueDiff;
  const flaggedDiff = (b.flagged_share ?? -1) - (a.flagged_share ?? -1);
  if (flaggedDiff !== 0) return flaggedDiff;
  const spreadDiff = (b.wide_spread_share ?? -1) - (a.wide_spread_share ?? -1);
  if (spreadDiff !== 0) return spreadDiff;
  if (b.row_count !== a.row_count) return b.row_count - a.row_count;
  return a.ticker.localeCompare(b.ticker);
};

const normalizeAuditProblemTicker = (
  item: DatasetAuditTickerSummary,
): ProblemTickerCardItem => ({
  ticker: item.ticker,
  row_count: item.row_count,
  snapshot_count: item.snapshot_count,
  avg_issue_count: item.avg_issue_count,
  flagged_share: item.flagged_share,
  fallback_share: item.fallback_share,
  wide_spread_share: item.wide_spread_share,
  clean_share: item.clean_share,
  watch_share: item.watch_share,
  noisy_share: item.noisy_share,
});

const normalizeLiveProblemTicker = (
  item: DatasetJobTickerTelemetry,
): ProblemTickerCardItem | null => {
  if (!item.rows || item.rows <= 0) return null;
  const rowCount = item.rows;
  return {
    ticker: item.ticker,
    row_count: rowCount,
    snapshot_count: item.kept_groups,
    avg_issue_count: rowCount > 0 ? item.issue_count_sum / rowCount : null,
    flagged_share: rowCount > 0 ? item.flagged_rows / rowCount : null,
    fallback_share: rowCount > 0 ? item.fallback_rows / rowCount : null,
    wide_spread_share: rowCount > 0 ? item.wide_spread_rows / rowCount : null,
    clean_share: rowCount > 0 ? item.clean_rows / rowCount : null,
    watch_share: rowCount > 0 ? item.watch_rows / rowCount : null,
    noisy_share: rowCount > 0 ? item.noisy_rows / rowCount : null,
  };
};

type ProblemTickerAuditCardProps = {
  subtitle: string;
  items: ProblemTickerCardItem[];
  emptyMessage: string;
  className?: string;
};

function ProblemTickerAuditCard({
  subtitle,
  items,
  emptyMessage,
  className,
}: ProblemTickerAuditCardProps) {
  const cardClassName = [
    "dataset-audit-card",
    "dataset-audit-problem-tickers-card",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <section className={cardClassName}>
      <div className="dataset-audit-card-header">
        <h3>Problem tickers</h3>
        <span>{subtitle}</span>
      </div>
      {items.length > 0 ? (
        <div className="dataset-audit-table dataset-audit-problem-ticker-table">
          <div className="dataset-audit-table-head">
            <span>Ticker</span>
            <span>Issue load</span>
            <span>Quality mix</span>
          </div>
          {items.map((ticker) => (
            <div key={ticker.ticker} className="dataset-audit-table-row">
              <div>
                <strong>{ticker.ticker}</strong>
                <span>
                  {ticker.row_count.toLocaleString()} rows ·{" "}
                  {ticker.snapshot_count?.toLocaleString() ?? "—"} snapshots
                </span>
              </div>
              <div>
                <strong>{formatNumeric(ticker.avg_issue_count, 2)}</strong>
                <span>{formatPercent(ticker.flagged_share)} flagged rows</span>
              </div>
              <div>
                <div className="dataset-audit-stack-bar" aria-label="Clean, watch, and noisy row mix">
                  <span
                    className="clean"
                    style={{ width: `${Math.max(0, (ticker.clean_share ?? 0) * 100)}%` }}
                  />
                  <span
                    className="watch"
                    style={{ width: `${Math.max(0, (ticker.watch_share ?? 0) * 100)}%` }}
                  />
                  <span
                    className="noisy"
                    style={{ width: `${Math.max(0, (ticker.noisy_share ?? 0) * 100)}%` }}
                  />
                </div>
                <div className="dataset-audit-mix-legend">
                  <span className="dataset-audit-mix-stat clean">
                    <span className="dataset-audit-mix-dot" />
                    Clean {formatPercent(ticker.clean_share)}
                  </span>
                  <span className="dataset-audit-mix-stat watch">
                    <span className="dataset-audit-mix-dot" />
                    Watch {formatPercent(ticker.watch_share)}
                  </span>
                  <span className="dataset-audit-mix-stat noisy">
                    <span className="dataset-audit-mix-dot" />
                    Noisy {formatPercent(ticker.noisy_share)}
                  </span>
                </div>
                <span className="dataset-audit-mix-meta">
                  Fallback {formatPercent(ticker.fallback_share)} · Wide spread{" "}
                  {formatPercent(ticker.wide_spread_share)}
                </span>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="dataset-preview-empty">{emptyMessage}</div>
      )}
    </section>
  );
}

const RV_LEVEL_FEATURES = ["rv5", "rv10", "rv20"] as const;
const RV_RATIO_FEATURES = ["rv5_over_rv10", "rv5_over_rv20", "rv10_over_rv20"] as const;
const RV_TERCILE_BUCKETS = ["low", "mid", "high"] as const;
const RV_CONTEXT_METRICS = [
  "rel_spread_median",
  "n_chain_used",
  "quality_issue_count",
] as const;

const formatAuditMetricValue = (metricName: string, value?: number | null): string => {
  if (metricName === "n_chain_used") return formatNumeric(value, 0);
  if (metricName === "quality_issue_count") return formatNumeric(value, 2);
  return formatNumeric(value, 3);
};

const railPositionPercent = (
  value?: number | null,
  min?: number | null,
  max?: number | null,
): number => {
  if (
    value == null ||
    min == null ||
    max == null ||
    !Number.isFinite(value) ||
    !Number.isFinite(min) ||
    !Number.isFinite(max)
  ) {
    return 50;
  }
  const span = max - min;
  if (!Number.isFinite(span) || Math.abs(span) < 1e-12) return 50;
  return Math.min(100, Math.max(0, ((value - min) / span) * 100));
};

const formatAuditRange = (
  metricName: string,
  min?: number | null,
  max?: number | null,
): string => {
  if (min == null || !Number.isFinite(min)) return "—";
  if (max == null || !Number.isFinite(max)) return formatAuditMetricValue(metricName, min);
  if (Math.abs(max - min) < 1e-12) return formatAuditMetricValue(metricName, min);
  return `${formatAuditMetricValue(metricName, min)} - ${formatAuditMetricValue(metricName, max)}`;
};

function RvSurfaceAuditCard({ auditResponse }: { auditResponse: DatasetAuditResponse }) {
  const availableFeatures = auditResponse.available_rv_features;
  const levels = RV_LEVEL_FEATURES.filter((feature) => availableFeatures.includes(feature));
  const ratios = RV_RATIO_FEATURES.filter((feature) => availableFeatures.includes(feature));
  const distributionMap = new Map(
    auditResponse.numeric_distributions.map((metric) => [metric.name, metric]),
  );
  const rvFeatureAuditMap = new Map(
    auditResponse.rv_feature_audit.map((item) => [item.feature, item]),
  );
  const contextMetrics = RV_CONTEXT_METRICS.map((metricName) => ({
    metricName,
    distribution: distributionMap.get(metricName),
  })).filter(
    (
      item,
    ): item is { metricName: (typeof RV_CONTEXT_METRICS)[number]; distribution: DatasetAuditDistribution } =>
      Boolean(item.distribution),
  );

  return (
    <section className="dataset-audit-card dataset-audit-card-wide dataset-audit-rv-card">
      <div className="dataset-audit-card-header">
        <h3>RV surface</h3>
        <span>{availableFeatures.length} features</span>
      </div>
      {availableFeatures.length > 0 ? (
        <>
          <div className="dataset-audit-rv-section">
            <div className="dataset-audit-rv-section-title">Feature families</div>
            <div className="dataset-audit-rv-group-grid">
              <div className="dataset-audit-rv-group">
                <span className="meta-label">Levels</span>
                <div className="dataset-audit-chip-row">
                  {levels.length > 0 ? (
                    levels.map((feature) => (
                      <span key={feature} className="dataset-audit-chip">
                        {feature}
                      </span>
                    ))
                  ) : (
                    <span className="dataset-audit-rv-empty-inline">No level RV features</span>
                  )}
                </div>
              </div>
              <div className="dataset-audit-rv-group">
                <span className="meta-label">Regime ratios</span>
                <div className="dataset-audit-chip-row">
                  {ratios.length > 0 ? (
                    ratios.map((feature) => (
                      <span key={feature} className="dataset-audit-chip">
                        {feature}
                      </span>
                    ))
                  ) : (
                    <span className="dataset-audit-rv-empty-inline">No RV ratio features</span>
                  )}
                </div>
                <p className="dataset-audit-rv-caption">
                  Ratios above 1 mean shorter-window RV is above the longer-window RV.
                </p>
              </div>
            </div>
          </div>

          <div className="dataset-audit-rv-section">
            <div className="dataset-audit-rv-section-title">Distribution range</div>
            <div className="dataset-audit-rv-rail-list">
              {availableFeatures.map((feature) => {
                const distribution = distributionMap.get(feature);
                if (!distribution) {
                  return (
                    <div key={feature} className="dataset-audit-rv-rail-row muted">
                      <div className="dataset-audit-rv-feature-label">
                        <code>{feature}</code>
                      </div>
                      <div className="dataset-preview-empty">No finite values available.</div>
                    </div>
                  );
                }

                const bandStart = railPositionPercent(
                  distribution.p05,
                  distribution.min,
                  distribution.max,
                );
                const bandEnd = railPositionPercent(
                  distribution.p95,
                  distribution.min,
                  distribution.max,
                );
                const medianPosition = railPositionPercent(
                  distribution.p50,
                  distribution.min,
                  distribution.max,
                );

                return (
                  <div key={feature} className="dataset-audit-rv-rail-row">
                    <div className="dataset-audit-rv-feature-label">
                      <code>{feature}</code>
                    </div>
                    <div className="dataset-audit-rv-rail-block">
                      <div className="dataset-audit-rv-rail-track">
                        <span
                          className="dataset-audit-rv-rail-band"
                          style={{
                            left: `${bandStart}%`,
                            width: `${Math.max(bandEnd - bandStart, 1)}%`,
                          }}
                        />
                        <span
                          className="dataset-audit-rv-rail-marker"
                          style={{ left: `${medianPosition}%` }}
                        />
                      </div>
                      <div className="dataset-audit-rv-rail-range">
                        <span>min {formatAuditMetricValue(feature, distribution.min)}</span>
                        <span>max {formatAuditMetricValue(feature, distribution.max)}</span>
                      </div>
                    </div>
                    <div className="dataset-audit-rv-rail-values">
                      <span>P05 {formatAuditMetricValue(feature, distribution.p05)}</span>
                      <span>P50 {formatAuditMetricValue(feature, distribution.p50)}</span>
                      <span>P95 {formatAuditMetricValue(feature, distribution.p95)}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="dataset-audit-rv-section">
            <div className="dataset-audit-rv-section-title">Quality by RV tercile</div>
            <div className="dataset-audit-rv-tercile-table-wrap">
              <table className="dataset-audit-rv-tercile-table">
                <colgroup>
                  <col className="dataset-audit-rv-tercile-col-feature" />
                  <col className="dataset-audit-rv-tercile-col-coverage" />
                  {RV_TERCILE_BUCKETS.map((label) => (
                    <Fragment key={`cols-${label}`}>
                      <col className="dataset-audit-rv-tercile-col-range" />
                      <col className="dataset-audit-rv-tercile-col-metric" />
                      <col className="dataset-audit-rv-tercile-col-metric" />
                    </Fragment>
                  ))}
                </colgroup>
                <thead>
                  <tr className="dataset-audit-rv-tercile-group-row">
                    <th scope="col" rowSpan={2}>
                      Feature
                    </th>
                    <th scope="col" rowSpan={2}>
                      Coverage
                    </th>
                    {RV_TERCILE_BUCKETS.map((label) => (
                      <th
                        key={label}
                        scope="colgroup"
                        colSpan={3}
                        className="dataset-audit-rv-tercile-group-head dataset-audit-rv-tercile-divider-col"
                      >
                        {label}
                      </th>
                    ))}
                  </tr>
                  <tr className="dataset-audit-rv-tercile-subhead-row">
                    {RV_TERCILE_BUCKETS.map((label) => (
                      <Fragment key={`subhead-${label}`}>
                        <th scope="col" className="dataset-audit-rv-tercile-divider-col">
                          Range
                        </th>
                        <th scope="col">Issues</th>
                        <th scope="col">Flagged</th>
                      </Fragment>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {availableFeatures.map((feature) => {
                    const featureAudit = rvFeatureAuditMap.get(feature);
                    const bucketMap = new Map(
                      (featureAudit?.buckets ?? []).map((bucket) => [bucket.label, bucket] as const),
                    );

                    return (
                      <tr key={feature}>
                        <th scope="row" className="dataset-audit-rv-tercile-feature-cell">
                          <code>{feature}</code>
                        </th>
                        <td className="dataset-audit-rv-tercile-coverage-cell">
                          {featureAudit
                            ? `${featureAudit.finite_row_count.toLocaleString()} finite rows · ${formatPercent(
                                featureAudit.finite_row_share,
                              )} coverage`
                            : "No RV audit summary"}
                        </td>
                        {RV_TERCILE_BUCKETS.map((label) => {
                          const bucket = bucketMap.get(label);
                          return (
                            <Fragment key={`${feature}-${label}`}>
                              <td className="dataset-audit-rv-tercile-range-cell dataset-audit-rv-tercile-divider-col">
                                {bucket
                                  ? formatAuditRange(feature, bucket.value_min, bucket.value_max)
                                  : "—"}
                              </td>
                              <td className="dataset-audit-rv-tercile-metric-cell">
                                {bucket
                                  ? formatAuditMetricValue(
                                      "quality_issue_count",
                                      bucket.avg_issue_count,
                                    )
                                  : "—"}
                              </td>
                              <td className="dataset-audit-rv-tercile-metric-cell">
                                {bucket ? formatPercent(bucket.flagged_share) : "—"}
                              </td>
                            </Fragment>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      ) : (
        <div className="dataset-preview-empty">No RV features detected.</div>
      )}

      {contextMetrics.length > 0 ? (
        <div className="dataset-audit-rv-section">
          <div className="dataset-audit-rv-section-title">Quality context</div>
          <div className="dataset-audit-rv-context-grid">
            {contextMetrics.map(({ metricName, distribution }) => (
              <div key={metricName} className="dataset-audit-rv-context-card">
                <code>{metricName}</code>
                <strong>P50 {formatAuditMetricValue(metricName, distribution.p50)}</strong>
                <span>
                  P05 {formatAuditMetricValue(metricName, distribution.p05)} · P95{" "}
                  {formatAuditMetricValue(metricName, distribution.p95)}
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}

type PreviewMode = "head" | "tail";

const PREVIEW_LIMIT_DEFAULT = 20;
const PREVIEW_LIMIT_OPTIONS = [20, 50, 100] as const;
const PREVIEW_MODE_OPTIONS: { value: PreviewMode; label: string }[] = [
  { value: "head", label: "First" },
  { value: "tail", label: "Last" },
];

type PreviewTarget = {
  label: string;
  path: string;
};

const dedupeFiles = (files: DatasetFileSummary[]): DatasetFileSummary[] => {
  const seen = new Set<string>();
  return files.filter((file) => {
    if (seen.has(file.path)) return false;
    seen.add(file.path);
    return true;
  });
};

const buildRunFiles = (run: DatasetRunSummary): DatasetFileSummary[] => {
  const listed = run.files?.filter(Boolean) ?? [];
  if (listed.length > 0) {
    return dedupeFiles(listed);
  }
  const fallback = [
    run.training_file,
    run.dataset_file,
    run.drops_file,
  ].filter(Boolean) as DatasetFileSummary[];
  return dedupeFiles(fallback);
};

const isCleanedDatasetFile = (file: DatasetFileSummary): boolean =>
  file.name.toLowerCase().endsWith("-cleaned.csv");

const RUN_CARD_TOGGLE_IGNORE_SELECTOR = [
  "button",
  "a",
  "input",
  "select",
  "textarea",
  "label",
  "[role='button']",
  "[role='link']",
  ".dataset-run-files-drawer",
].join(", ");

const shouldIgnoreRunCardToggle = (target: EventTarget | null): boolean =>
  target instanceof Element &&
  Boolean(target.closest(RUN_CARD_TOGGLE_IGNORE_SELECTOR));

const sortRunFiles = (
  files: DatasetFileSummary[],
  trainingPath?: string | null,
): DatasetFileSummary[] => {
  return [...files].sort((a, b) => {
    const aIsTraining = trainingPath && a.path === trainingPath;
    const bIsTraining = trainingPath && b.path === trainingPath;
    if (aIsTraining && !bIsTraining) return -1;
    if (!aIsTraining && bIsTraining) return 1;
    return a.name.localeCompare(b.name);
  });
};

const countMondaysInRange = (start: string, end: string): number | null => {
  if (!start || !end) return null;
  const startDate = new Date(`${start}T00:00:00Z`);
  const endDate = new Date(`${end}T00:00:00Z`);
  if (Number.isNaN(startDate.getTime()) || Number.isNaN(endDate.getTime())) {
    return null;
  }
  if (endDate < startDate) return null;
  let count = 0;
  const cursor = new Date(startDate);
  while (cursor <= endDate) {
    if (cursor.getUTCDay() === 1) count += 1;
    cursor.setUTCDate(cursor.getUTCDate() + 1);
  }
  return count;
};

const DATE_RANGE_ERROR = "End date must be on or after start date.";

const LIVE_PHASE_LABELS: Record<DatasetJobTelemetry["phase"], string> = {
  planning: "Planning",
  preloading_stock: "Preloading closes",
  preloading_dividends: "Preloading dividends",
  building: "Building snapshots",
  finalizing: "Finalizing dataset",
  writing_outputs: "Writing CSVs",
  finished: "Finished",
};

type QuickAuditCheckKey = keyof DatasetJobGroupChecks;

const QUICK_AUDIT_CHECKS: Array<{
  key: QuickAuditCheckKey;
  label: string;
  description: string;
}> = [
  {
    key: "asof_close_fallback",
    label: "As-of fallback",
    description: "Snapshot close had to fall forward from the target date.",
  },
  {
    key: "expiry_close_fallback",
    label: "Expiry fallback",
    description: "Expiry close had to fall backward from Friday.",
  },
  {
    key: "expiry_saturday_fallback",
    label: "Saturday expiry",
    description: "The option chain was pulled from Saturday instead of Friday.",
  },
  {
    key: "quote_close_fallback",
    label: "Close quote source",
    description: "Close prices were used instead of bid/ask mid quotes.",
  },
  {
    key: "low_chain_used",
    label: "Thin chain",
    description: "The kept group used fewer strikes than the comfort threshold.",
  },
  {
    key: "wide_rel_spread",
    label: "Wide spread",
    description: "Median relative spread crossed the warning threshold.",
  },
];

const formatDurationFromSeconds = (value?: number | null): string => {
  if (value == null || !Number.isFinite(value) || value < 0) return "—";
  const wholeSeconds = Math.round(value);
  if (wholeSeconds < 60) return `${wholeSeconds}s`;
  const minutes = Math.floor(wholeSeconds / 60);
  const seconds = wholeSeconds % 60;
  if (minutes < 60) return seconds > 0 ? `${minutes}m ${seconds}s` : `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  const remMinutes = minutes % 60;
  return remMinutes > 0 ? `${hours}h ${remMinutes}m` : `${hours}h`;
};

const countDropReasons = (dropReasons?: Record<string, number> | null): number =>
  Object.values(dropReasons ?? {}).reduce((sum, count) => sum + (count ?? 0), 0);

const topDropReason = (dropReasons?: Record<string, number> | null): string => {
  const entries = Object.entries(dropReasons ?? {});
  if (!entries.length) return "—";
  entries.sort((a, b) => {
    if (b[1] !== a[1]) return b[1] - a[1];
    return a[0].localeCompare(b[0]);
  });
  return entries[0][0].replace(/_/g, " ");
};

const getTickerDropShare = (item: DatasetJobTickerTelemetry): number => {
  if (!item.completed_jobs) return 0;
  return countDropReasons(item.drop_reasons) / item.completed_jobs;
};

const hasInvalidDateRange = (start: string, end: string): boolean => {
  if (!start || !end) return false;
  const startDate = new Date(`${start}T00:00:00Z`);
  const endDate = new Date(`${end}T00:00:00Z`);
  if (Number.isNaN(startDate.getTime()) || Number.isNaN(endDate.getTime())) {
    return false;
  }
  return endDate < startDate;
};

export default function DatasetsPage() {
  const [formState, setFormState] = useState<DatasetFormState>(defaultForm);
  const [isRunning, setIsRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [runResult, setRunResult] = useState<DatasetRunResponse | null>(null);
  const [workspaceTab, setWorkspaceTab] = useState<"run_job" | "run_directory">(
    "run_job",
  );
  const [runJobPanel, setRunJobPanel] = useState<"configuration" | "active_run">(
    "configuration",
  );
  const [storageReady, setStorageReady] = useState(false);
  const { jobStatus, jobId, setJobId, setJobStatus: setGlobalJobStatus } =
    useDatasetJob();
  const { anyJobRunning, primaryJob, activeJobs } = useAnyJobRunning();
  const [killLoading, setKillLoading] = useState(false);
  const [datasetRuns, setDatasetRuns] = useState<DatasetRunSummary[]>([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [previewTarget, setPreviewTarget] = useState<PreviewTarget | null>(null);
  const [previewResponse, setPreviewResponse] =
    useState<DatasetPreviewResponse | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [auditTargetPath, setAuditTargetPath] = useState<string | null>(null);
  const [auditResponse, setAuditResponse] =
    useState<DatasetAuditResponse | null>(null);
  const [auditError, setAuditError] = useState<string | null>(null);
  const [auditLoading, setAuditLoading] = useState(false);
  const [heatmapMetric, setHeatmapMetric] = useState<HeatmapMetric>("issues");
  const [activeRunAuditResponse, setActiveRunAuditResponse] =
    useState<DatasetAuditResponse | null>(null);
  const [activeRunAuditError, setActiveRunAuditError] = useState<string | null>(null);
  const [activeRunAuditLoading, setActiveRunAuditLoading] = useState(false);
  const [previewMode, setPreviewMode] = useState<PreviewMode>("head");
  const [previewLimit, setPreviewLimit] = useState<number>(
    PREVIEW_LIMIT_DEFAULT,
  );
  const [deleteConfirmRun, setDeleteConfirmRun] = useState<string | null>(null);
  const [deleteConfirmText, setDeleteConfirmText] = useState("");
  const [deleteLoadingRun, setDeleteLoadingRun] = useState<string | null>(null);
  const [renamingRunId, setRenamingRunId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState<string>("");
  const [renameError, setRenameError] = useState<string | null>(null);
  const [renameLoading, setRenameLoading] = useState(false);
  const [cleanupTarget, setCleanupTarget] = useState<CleanupModalTarget | null>(
    null,
  );
  const [cleanupCriteriaForm, setCleanupCriteriaForm] =
    useState<CleanupCriteriaFormState>(DEFAULT_CLEANUP_CRITERIA_FORM);
  const [cleanupPreviewResponse, setCleanupPreviewResponse] =
    useState<DatasetCleanupPreviewResponse | null>(null);
  const [cleanupPreviewError, setCleanupPreviewError] = useState<string | null>(
    null,
  );
  const [cleanupPreviewLoading, setCleanupPreviewLoading] = useState(false);
  const [cleanupConfirmText, setCleanupConfirmText] = useState("");
  const [cleanupApplyLoading, setCleanupApplyLoading] = useState(false);
  const [cleanupActionError, setCleanupActionError] = useState<string | null>(
    null,
  );
  // Accordion: tracks which run's CSV file list is expanded — null means all collapsed.
  // Single value ensures only one run can be open at a time without per-item flags.
  const [openRunId, setOpenRunId] = useState<string | null>(null);
  const [customTickerInput, setCustomTickerInput] = useState("");
  const refreshDatasetRuns = useCallback(async () => {
    setRunsLoading(true);
    setRunsError(null);
    try {
      const response = await listDatasetRuns();
      setDatasetRuns(response.runs);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setRunsError(message);
    } finally {
      setRunsLoading(false);
    }
  }, []);

  const selectedTickers = useMemo(() => {
    const parsed = normalizeTickers(parseTickers(formState.tickers) ?? []);
    return orderTickers(parsed);
  }, [formState.tickers]);
  const todayDateString = useMemo(
    () => new Date().toISOString().slice(0, 10),
    [],
  );
  const selectedTickerSet = useMemo(
    () => new Set(selectedTickers),
    [selectedTickers],
  );
  const customTickers = useMemo(
    () =>
      selectedTickers.filter(
        (ticker) => !TRADING_UNIVERSE_TICKERS.includes(ticker),
      ),
    [selectedTickers],
  );
  const customTickerCount = customTickers.length;
  const resolvedTickersCount = selectedTickers.length;
  const weeksCount = useMemo(
    () => countMondaysInRange(formState.start, formState.end),
    [formState.start, formState.end],
  );
  const dateRangeInvalid = useMemo(
    () => hasInvalidDateRange(formState.start, formState.end),
    [formState.start, formState.end],
  );
  const expiryWeekdays = useMemo(
    () => parseWeekdays(formState.expiryWeekdays),
    [formState.expiryWeekdays],
  );
  const asofWeekdays = useMemo(
    () => parseWeekdays(formState.asofWeekdays),
    [formState.asofWeekdays],
  );
  const expiriesCount = useMemo(() => {
    if (formState.scheduleMode === "weekly") return weeksCount;
    return countWeekdaysInRange(formState.start, formState.end, expiryWeekdays);
  }, [
    formState.scheduleMode,
    formState.start,
    formState.end,
    expiryWeekdays,
    weeksCount,
  ]);
  const dteCount = useMemo(
    () =>
      resolveDteCount(
        formState.dteList,
        formState.dteMin,
        formState.dteMax,
        formState.dteStep,
      ),
    [formState.dteList, formState.dteMin, formState.dteMax, formState.dteStep],
  );
  const asofCount = dteCount ?? (asofWeekdays ? asofWeekdays.length : null);
  const snapshotCountLabel = dteCount
    ? `${dteCount} DTEs`
    : asofWeekdays
      ? `${asofWeekdays.length} weekdays`
      : "—";
  const plannedJobs =
    expiriesCount && asofCount && resolvedTickersCount
      ? expiriesCount * asofCount * resolvedTickersCount
      : null;

  const updateTickers = useCallback((next: string[]) => {
    const normalized = normalizeTickers(next);
    if (normalized.length === 0) return;
    const ordered = orderTickers(normalized);
    setFormState((prev) => ({
      ...prev,
      tickers: formatTickerList(ordered),
    }));
  }, []);

  const updateJobState = (status: DatasetJobStatus) => {
    setGlobalJobStatus(status);
    setJobId(status.job_id);
    const running = status.status === "queued" || status.status === "running";
    setIsRunning(running);
    if (status.result) {
      setRunResult(status.result);
    }
    if (status.status === "failed" && status.error) {
      setRunError(status.error);
    } else if (status.status === "cancelled") {
      setRunError("Dataset creation was cancelled.");
    } else if (running) {
      setRunError(null);
    }
  };
  const isDatasetJobActive = Boolean(
    jobId && (!jobStatus || jobStatus.status === "running" || jobStatus.status === "queued"),
  );

  useEffect(() => {
    refreshDatasetRuns();
  }, [refreshDatasetRuns]);

  useEffect(() => {
    if (!isDatasetJobActive) return undefined;
    void refreshDatasetRuns();
    const intervalId = window.setInterval(() => {
      void refreshDatasetRuns();
    }, 1000);
    return () => window.clearInterval(intervalId);
  }, [isDatasetJobActive, refreshDatasetRuns]);

  useEffect(() => {
    if (selectedTickers.length === 0) {
      updateTickers(TRADING_UNIVERSE_TICKERS);
    }
  }, [selectedTickers, updateTickers]);

  useEffect(() => {
    if (!jobStatus) return;
    if (["finished", "failed", "cancelled"].includes(jobStatus.status)) {
      refreshDatasetRuns();
    }
  }, [jobStatus?.status, refreshDatasetRuns]);

  useEffect(() => {
    if (!jobStatus) return;
    if (jobStatus.status === "running" || jobStatus.status === "queued") {
      setRunJobPanel("active_run");
    }
  }, [jobStatus?.status]);

  useEffect(() => {
    if (isDatasetJobActive) {
      setRunJobPanel("active_run");
    }
  }, [isDatasetJobActive]);

  const resolvedRange =
    formState.start && formState.end
      ? `${formState.start} → ${formState.end}`
      : "Select a date range";
  const resolvedTickersLabel = customTickers.length
    ? `${resolvedTickersCount} tickers (${customTickers.length} custom)`
    : `${resolvedTickersCount} tickers`;
  const plannedJobsLabel = plannedJobs
    ? `${plannedJobs.toLocaleString()} jobs`
    : "Set a date range";
  const plannedWeeksLabel =
    expiriesCount !== null ? `${expiriesCount} expiries` : "Expiries pending";
  const jobProgress = jobStatus?.progress ?? null;
  const currentTelemetry = jobStatus?.telemetry ?? null;
  const progressPercent =
    jobProgress && jobProgress.total > 0
      ? Math.round((jobProgress.done / jobProgress.total) * 100)
      : jobStatus && (jobStatus.status === "running" || jobStatus.status === "queued")
      ? 0
      : null;
  const stdoutText =
    jobStatus?.stdout.join("") ||
    jobStatus?.result?.stdout ||
    runResult?.stdout ||
    "";
  const stderrText =
    jobStatus?.stderr.join("") ||
    jobStatus?.result?.stderr ||
    runResult?.stderr ||
    "";
  const recentWarnings = useMemo(
    () =>
      stdoutText
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line.startsWith("[WARN]"))
        .slice(-3)
        .reverse(),
    [stdoutText],
  );
  const currentResult = jobStatus?.result ?? runResult;
  const statusClass = jobStatus
    ? jobStatus.status === "running" || jobStatus.status === "queued"
      ? "running"
      : jobStatus.status === "finished"
        ? currentResult?.ok
          ? "success"
          : "failed"
        : "failed"
    : "running";
  const statusLabel = jobStatus
    ? jobStatus.status === "running" || jobStatus.status === "queued"
      ? "Running"
      : jobStatus.status === "finished"
        ? currentResult?.ok
          ? "Success"
          : "Failed"
        : "Cancelled"
    : "Idle";
  const isJobInFlight = Boolean(
    jobStatus && (jobStatus.status === "running" || jobStatus.status === "queued"),
  );
  const showNewJobButton = Boolean(jobStatus && !isJobInFlight);
  const datasetNameKebab = toKebabCase(formState.datasetName);
  const outputLocationLabel =
    currentResult?.output_file ??
    currentResult?.run_dir ??
    `${currentResult?.out_dir ?? formState.outDir}/${datasetNameKebab || "(pending)"}`;
  const trainingDatasetPath =
    currentResult?.training_file ??
    (currentResult?.out_dir && datasetNameKebab
      ? `${currentResult.out_dir}/${datasetNameKebab}/training-${datasetNameKebab}.csv`
      : null);
  const trainingDatasetEnabled = formState.writeTrainView;
  const activeRunAuditPath =
    jobStatus?.status === "finished" && currentResult?.ok
      ? currentResult?.training_file ?? null
      : null;
  const liveTickerItems = currentTelemetry?.tickers ?? [];
  const telemetryKeptGroups =
    jobProgress?.groups ??
    liveTickerItems.reduce((sum, item) => sum + item.kept_groups, 0);
  const elapsedSeconds = useMemo(() => {
    if (currentResult) return currentResult.duration_s;
    if (!jobStatus?.started_at) return null;
    const started = new Date(jobStatus.started_at).getTime();
    if (!Number.isFinite(started)) return null;
    const finished = jobStatus.finished_at
      ? new Date(jobStatus.finished_at).getTime()
      : Date.now();
    if (!Number.isFinite(finished) || finished < started) return null;
    return (finished - started) / 1000;
  }, [currentResult, jobStatus?.finished_at, jobStatus?.started_at]);
  const jobsPerMinute =
    jobProgress && elapsedSeconds && elapsedSeconds > 0
      ? (jobProgress.done / elapsedSeconds) * 60
      : null;
  const remainingJobs =
    jobProgress ? Math.max(0, jobProgress.total - jobProgress.done) : null;
  const etaSeconds =
    jobsPerMinute && remainingJobs != null && jobsPerMinute > 0
      ? (remainingJobs / jobsPerMinute) * 60
      : null;
  const keptGroupRate =
    jobProgress && jobProgress.done > 0
      ? telemetryKeptGroups / jobProgress.done
      : null;
  const rowsPerKeptGroup =
    jobProgress && telemetryKeptGroups > 0
      ? jobProgress.rows / telemetryKeptGroups
      : null;
  const phaseLabel = currentTelemetry
    ? LIVE_PHASE_LABELS[currentTelemetry.phase]
    : jobStatus?.status === "queued"
      ? LIVE_PHASE_LABELS.planning
      : "Waiting for telemetry";
  const quickAuditChecks = useMemo(
    () =>
      QUICK_AUDIT_CHECKS.map((item) => {
        const count = currentTelemetry?.group_checks?.[item.key] ?? 0;
        const share =
          telemetryKeptGroups > 0 ? count / Math.max(telemetryKeptGroups, 1) : null;
        return {
          ...item,
          count,
          share,
        };
      }),
    [currentTelemetry, telemetryKeptGroups],
  );
  const problemTickers = useMemo(() => {
    return [...liveTickerItems]
      .map(normalizeLiveProblemTicker)
      .filter((item): item is ProblemTickerCardItem => item !== null)
      .sort(compareProblemTickerCardItems)
      .slice(0, 6);
  }, [liveTickerItems]);
  const tickerProgressItems = useMemo(() => {
    return [...liveTickerItems]
      .filter((item) => item.completed_jobs > 0)
      .sort((a, b) => {
        const dropShareDiff = getTickerDropShare(b) - getTickerDropShare(a);
        if (dropShareDiff !== 0) return dropShareDiff;
        if (b.completed_jobs !== a.completed_jobs) {
          return b.completed_jobs - a.completed_jobs;
        }
        return a.ticker.localeCompare(b.ticker);
      })
      .slice(0, 6);
  }, [liveTickerItems]);
  const compactAuditFlags = activeRunAuditResponse?.quality_flags.slice(0, 5) ?? [];
  const compactProblemTickers = useMemo(
    () =>
      (activeRunAuditResponse?.top_problem_tickers ?? [])
        .slice(0, 5)
        .map(normalizeAuditProblemTicker),
    [activeRunAuditResponse],
  );
  const fullAuditProblemTickers = useMemo(
    () => (auditResponse?.top_problem_tickers ?? []).map(normalizeAuditProblemTicker),
    [auditResponse],
  );
  const cleanupTargetTrainingPath = cleanupTarget?.trainingFile.path ?? null;
  const cleanupAudit = useMemo(() => {
    if (!cleanupTargetTrainingPath) return null;
    return auditResponse?.file.path === cleanupTargetTrainingPath ? auditResponse : null;
  }, [auditResponse, cleanupTargetTrainingPath]);
  const cleanupAuditRequestError = useMemo(() => {
    if (!cleanupTargetTrainingPath || auditTargetPath !== cleanupTargetTrainingPath) {
      return null;
    }
    return auditError;
  }, [auditError, auditTargetPath, cleanupTargetTrainingPath]);
  const cleanupAuditRequestLoading =
    Boolean(cleanupTargetTrainingPath) &&
    auditTargetPath === cleanupTargetTrainingPath &&
    auditLoading;
  const cleanupCriteriaPayload = useMemo(
    () => ({
      qualityBuckets:
        cleanupCriteriaForm.mode === "quality_buckets"
          ? cleanupCriteriaForm.qualityBuckets
          : [],
      flagColumns:
        cleanupCriteriaForm.mode === "flags" ? cleanupCriteriaForm.selectedFlags : [],
      flagMatchMode:
        cleanupCriteriaForm.mode === "flags"
          ? cleanupCriteriaForm.flagMatchMode
          : undefined,
    }),
    [cleanupCriteriaForm],
  );
  const cleanupCanApply = Boolean(
    cleanupPreviewResponse &&
      cleanupConfirmText === "CLEAN" &&
      !cleanupPreviewLoading &&
      !cleanupApplyLoading &&
      cleanupPreviewResponse.rows_to_drop > 0 &&
      !cleanupPreviewResponse.would_drop_all,
  );
  const cleanupFlagOptions = cleanupAudit?.available_quality_flags ?? [];
  const cleanupModeHasSelection =
    cleanupCriteriaForm.mode === "quality_buckets"
      ? cleanupCriteriaForm.qualityBuckets.length > 0
      : cleanupCriteriaForm.selectedFlags.length > 0;
  const criticalErrorText = (
    stderrText ||
    jobStatus?.error ||
    runError ||
    (jobStatus?.status === "cancelled" ? "Dataset creation was cancelled." : "")
  ).trim();
  const showCriticalErrors = Boolean(
    stderrText.trim() ||
      jobStatus?.status === "failed" ||
      jobStatus?.status === "cancelled",
  );
  const heatmapTickers = useMemo(() => {
    if (!auditResponse) return [];
    return auditResponse.top_problem_tickers.map((item) => item.ticker);
  }, [auditResponse]);
  const heatmapCellMap = useMemo(() => {
    if (!auditResponse) {
      return new Map<string, NonNullable<DatasetAuditResponse["heatmap_cells"]>[number]>();
    }
    return new Map(
      auditResponse.heatmap_cells.map((cell) => [
        `${cell.ticker}__${cell.asof_date}`,
        cell,
      ]),
    );
  }, [auditResponse]);
  const heatmapMaxRows = useMemo(() => {
    if (!auditResponse || auditResponse.heatmap_cells.length === 0) return 1;
    return Math.max(...auditResponse.heatmap_cells.map((cell) => cell.row_count), 1);
  }, [auditResponse]);

  useEffect(() => {
    if (!jobStatus || jobStatus.status !== "finished") return;
    if (!currentResult?.ok) return;
    if (!trainingDatasetPath || !trainingDatasetEnabled) return;
    try {
      const raw = localStorage.getItem(CALIBRATE_STORAGE_KEY);
      const parsed = raw ? JSON.parse(raw) : {};
      const next = { ...parsed, datasetPath: trainingDatasetPath };
      localStorage.setItem(CALIBRATE_STORAGE_KEY, JSON.stringify(next));
    } catch {
      // ignore storage failures
    }
  }, [
    jobStatus?.status,
    currentResult?.ok,
    trainingDatasetPath,
    trainingDatasetEnabled,
  ]);

  useEffect(() => {
    if (!activeRunAuditPath) {
      setActiveRunAuditResponse(null);
      setActiveRunAuditError(null);
      setActiveRunAuditLoading(false);
      return;
    }
    let cancelled = false;
    setActiveRunAuditLoading(true);
    setActiveRunAuditError(null);
    auditDatasetFile(activeRunAuditPath)
      .then((result) => {
        if (cancelled) return;
        setActiveRunAuditResponse(result);
      })
      .catch((err) => {
        if (cancelled) return;
        setActiveRunAuditResponse(null);
        const message = err instanceof Error ? err.message : "Unknown error";
        setActiveRunAuditError(message);
      })
      .finally(() => {
        if (!cancelled) setActiveRunAuditLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [activeRunAuditPath]);

  useEffect(() => {
    if (!previewTarget) {
      setPreviewResponse(null);
      setPreviewError(null);
      setPreviewLoading(false);
      return;
    }
    let cancelled = false;
    setPreviewLoading(true);
    setPreviewError(null);
    previewDatasetFile(previewTarget.path, previewMode, previewLimit)
      .then((result) => {
        if (cancelled) return;
        setPreviewResponse(result);
      })
      .catch((err) => {
        if (cancelled) return;
        setPreviewResponse(null);
        const message = err instanceof Error ? err.message : "Unknown error";
        setPreviewError(message);
      })
      .finally(() => {
        if (!cancelled) setPreviewLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [previewTarget, previewMode, previewLimit]);

  useEffect(() => {
    if (!auditTargetPath) {
      setAuditResponse(null);
      setAuditError(null);
      setAuditLoading(false);
      return;
    }
    let cancelled = false;
    setAuditLoading(true);
    setAuditError(null);
    auditDatasetFile(auditTargetPath)
      .then((result) => {
        if (cancelled) return;
        setAuditResponse(result);
      })
      .catch((err) => {
        if (cancelled) return;
        setAuditResponse(null);
        const message = err instanceof Error ? err.message : "Unknown error";
        setAuditError(message);
      })
      .finally(() => {
        if (!cancelled) setAuditLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [auditTargetPath]);

  useEffect(() => {
    if (!cleanupTargetTrainingPath) {
      setCleanupPreviewResponse(null);
      setCleanupPreviewError(null);
      setCleanupPreviewLoading(false);
      setCleanupConfirmText("");
      setCleanupActionError(null);
      return;
    }
    if (auditTargetPath !== cleanupTargetTrainingPath) {
      setAuditTargetPath(cleanupTargetTrainingPath);
    }
  }, [auditTargetPath, cleanupTargetTrainingPath]);

  useEffect(() => {
    if (!cleanupTarget || !cleanupTargetTrainingPath) {
      return;
    }
    if (cleanupAuditRequestLoading) {
      setCleanupPreviewLoading(true);
      return;
    }
    if (cleanupAuditRequestError) {
      setCleanupPreviewResponse(null);
      setCleanupPreviewError(cleanupAuditRequestError);
      setCleanupPreviewLoading(false);
      return;
    }
    if (!cleanupAudit) {
      return;
    }
    if (!cleanupModeHasSelection) {
      setCleanupPreviewResponse(null);
      setCleanupPreviewError(null);
      setCleanupPreviewLoading(false);
      return;
    }

    let cancelled = false;
    const timeoutId = window.setTimeout(() => {
      setCleanupPreviewLoading(true);
      setCleanupPreviewError(null);
      setCleanupActionError(null);
      previewDatasetCleanup({
        runDir: cleanupTarget.runDir,
        criteria: cleanupCriteriaPayload,
      })
        .then((result) => {
          if (cancelled) return;
          setCleanupPreviewResponse(result);
        })
        .catch((err) => {
          if (cancelled) return;
          setCleanupPreviewResponse(null);
          const message = err instanceof Error ? err.message : "Unknown error";
          setCleanupPreviewError(message);
        })
        .finally(() => {
          if (!cancelled) setCleanupPreviewLoading(false);
        });
    }, 250);

    return () => {
      cancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, [
    cleanupAudit,
    cleanupAuditRequestError,
    cleanupAuditRequestLoading,
    cleanupModeHasSelection,
    cleanupCriteriaPayload,
    cleanupTarget,
    cleanupTargetTrainingPath,
  ]);

  useEffect(() => {
    const stored = loadStoredForm();
    if (stored) {
      setFormState((prev) => ({ ...prev, ...stored }));
    }
    setStorageReady(true);
  }, []);

  useEffect(() => {
    if (!formState.writeTrainView) {
      setFormState((prev) => ({ ...prev, writeTrainView: true }));
    }
  }, [formState.writeTrainView]);

  useEffect(() => {
    if (!storageReady) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(formState));
    } catch {
      // ignore storage failures
    }
  }, [formState, storageReady]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (anyJobRunning) {
      setRunError(
        `Another job is running (${primaryJob?.name ?? "unknown"}). Wait for it to finish.`,
      );
      return;
    }
    setRunError(null);
    setRunResult(null);
    setGlobalJobStatus(null);
    setJobId(null);
    setActiveRunAuditResponse(null);
    setActiveRunAuditError(null);
    setActiveRunAuditLoading(false);
    setIsRunning(true);

    try {
      if (!formState.datasetName.trim()) {
        setRunError("Dataset name is required.");
        setIsRunning(false);
        return;
      }
      if (dateRangeInvalid) {
        setRunError(DATE_RANGE_ERROR);
        setIsRunning(false);
        return;
      }
      const payload = {
        outDir: formState.outDir.trim() || undefined,
        datasetName: formState.datasetName.trim(),
        scheduleMode: formState.scheduleMode,
        expiryWeekdays: formState.expiryWeekdays.trim() || undefined,
        asofWeekdays: formState.asofWeekdays.trim() || undefined,
        dteList: formState.dteList.trim() || undefined,
        dteMin: parseOptionalInt(formState.dteMin),
        dteMax: parseOptionalInt(formState.dteMax),
        dteStep: parseOptionalInt(formState.dteStep),
        writeSnapshot: formState.writeSnapshot,
        writePrnView: formState.writePrnView,
        writeTrainView: formState.writeTrainView,
        writeLegacy: formState.writeLegacy,
        prnVersion: formState.prnVersion.trim() || undefined,
        prnConfigHash: formState.prnConfigHash.trim() || undefined,
        tickers: formState.tickers.trim() || undefined,
        start: formState.start,
        end: formState.end,
        thetaBaseUrl: formState.thetaBaseUrl.trim() || undefined,
        stockSource: formState.stockSource,
        timeoutS: parseOptionalInt(formState.timeoutS),
        riskFreeRate: parseOptionalNumber(formState.riskFreeRate),
        maxAbsLogm: parseOptionalNumber(formState.maxAbsLogm),
        maxAbsLogmCap: parseOptionalNumber(formState.maxAbsLogmCap),
        bandWidenStep: parseOptionalNumber(formState.bandWidenStep),
        noAdaptiveBand: !formState.adaptiveBand,
        maxBandStrikes: parseOptionalInt(formState.maxBandStrikes),
        minBandStrikes: parseOptionalInt(formState.minBandStrikes),
        minBandPrnStrikes: parseOptionalInt(formState.minBandPrnStrikes),
        strikeRange: parseOptionalInt(formState.strikeRange),
        noRetryFullChain: !formState.retryFullChain,
        noSatExpiryFallback: !formState.saturdayExpiryFallback,
        threads: parseOptionalInt(formState.threads),
        preferBidask: formState.preferBidask,
        minTradeCount: parseOptionalInt(formState.minTradeCount),
        minVolume: parseOptionalInt(formState.minVolume),
        minChainUsedHard: parseOptionalInt(formState.minChainUsedHard),
        maxRelSpreadMedianHard: parseOptionalNumber(
          formState.maxRelSpreadMedianHard,
        ),
        hardDropCloseFallback: formState.hardDropCloseFallback,
        minPrnTrain: parseOptionalNumber(formState.minPrnTrain),
        maxPrnTrain: parseOptionalNumber(formState.maxPrnTrain),
        noSplitAdjust: !formState.splitAdjust,
        dividendSource: formState.dividendSource,
        dividendLookbackDays: parseOptionalInt(formState.dividendLookbackDays),
        dividendYieldDefault: parseOptionalNumber(
          formState.dividendYieldDefault,
        ),
        noForwardMoneyness: !formState.forwardMoneyness,
        noGroupWeights: !formState.groupWeights,
        noTickerWeights: !formState.tickerWeights,
        noSoftQualityWeight: !formState.softQualityWeight,
        rvLookbackDays: parseOptionalInt(formState.rvLookbackDays),
        cache: formState.cache,
        writeDrops: formState.writeDrops,
        sanityReport: formState.sanityReport,
        sanityDrop: formState.sanityDrop,
        sanityAbsLogmMax: parseOptionalNumber(formState.sanityAbsLogmMax),
        sanityKOverSMin: parseOptionalNumber(formState.sanityKOverSMin),
        sanityKOverSMax: parseOptionalNumber(formState.sanityKOverSMax),
        verboseSkips: formState.verboseSkips,
      };

      const status = await startDatasetJob(payload);
      updateJobState(status);
      await refreshDatasetRuns();
      setRunJobPanel("active_run");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setRunError(message);
      setIsRunning(false);
    }
  };

  const handleKill = async () => {
    if (!jobId) return;
    setKillLoading(true);
    try {
      const status = await killDatasetJob(jobId);
      updateJobState(status);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setRunError(message);
    } finally {
      setKillLoading(false);
    }
  };

  const handlePreviewSelection = useCallback(
    (target: PreviewTarget, options?: { auditPath?: string | null }) => {
      setPreviewTarget(target);
      if (options && "auditPath" in options) {
        setAuditTargetPath(options.auditPath ?? null);
      }
    },
    [],
  );

  const handleDeleteRun = async (runId: string, runDir: string) => {
    setDeleteLoadingRun(runId);
    try {
      await deleteDatasetRun(runDir);
      setDeleteConfirmRun(null);
      setDeleteConfirmText("");
      if (previewTarget?.path.startsWith(runDir)) {
        setPreviewTarget(null);
      }
      if (auditTargetPath?.startsWith(runDir)) {
        setAuditTargetPath(null);
      }
      await refreshDatasetRuns();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setRunsError(message);
    } finally {
      setDeleteLoadingRun(null);
    }
  };

  const handleStartRename = (runId: string, currentName: string) => {
    setRenamingRunId(runId);
    setRenameValue(currentName);
    setRenameError(null);
  };

  const handleCancelRename = () => {
    setRenamingRunId(null);
    setRenameValue("");
    setRenameError(null);
  };

  const handleConfirmRename = async (runId: string, runDir: string) => {
    const trimmed = renameValue.trim();
    if (!trimmed) {
      setRenameError("Directory name cannot be empty.");
      return;
    }
    setRenameLoading(true);
    setRenameError(null);
    try {
      const updated = await renameDatasetRun(runDir, trimmed);
      setDatasetRuns((prev) =>
        prev.map((run) => (run.id === runId ? updated : run)),
      );
      if (previewTarget?.path.startsWith(runDir)) {
        setPreviewTarget(null);
      }
      if (auditTargetPath?.startsWith(runDir)) {
        setAuditTargetPath(null);
      }
      handleCancelRename();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setRenameError(message);
    } finally {
      setRenameLoading(false);
    }
  };

  const handleOpenCleanupModal = useCallback(
    (run: DatasetRunSummary, trainingFile: DatasetFileSummary) => {
      const runName = run.run_dir.split("/").pop() ?? run.id;
      setCleanupTarget({
        runId: run.id,
        runDir: run.run_dir,
        runName,
        trainingFile,
      });
      setCleanupCriteriaForm(DEFAULT_CLEANUP_CRITERIA_FORM);
      setCleanupPreviewResponse(null);
      setCleanupPreviewError(null);
      setCleanupPreviewLoading(true);
      setCleanupConfirmText("");
      setCleanupActionError(null);
      setAuditTargetPath(trainingFile.path);
    },
    [],
  );

  const handleCloseCleanupModal = useCallback(() => {
    setCleanupTarget(null);
    setCleanupPreviewResponse(null);
    setCleanupPreviewError(null);
    setCleanupPreviewLoading(false);
    setCleanupConfirmText("");
    setCleanupActionError(null);
  }, []);

  const toggleCleanupBucket = useCallback(
    (bucket: "clean" | "watch" | "noisy") => {
      setCleanupCriteriaForm((prev) => ({
        ...prev,
        qualityBuckets: prev.qualityBuckets.includes(bucket)
          ? prev.qualityBuckets.filter((value) => value !== bucket)
          : [...prev.qualityBuckets, bucket],
      }));
    },
    [],
  );

  const toggleCleanupFlag = useCallback((flag: string) => {
    setCleanupCriteriaForm((prev) => ({
      ...prev,
      selectedFlags: prev.selectedFlags.includes(flag)
        ? prev.selectedFlags.filter((value) => value !== flag)
        : [...prev.selectedFlags, flag],
    }));
  }, []);

  const handleApplyCleanup = async () => {
    if (!cleanupTarget || !cleanupCanApply) return;
    setCleanupApplyLoading(true);
    setCleanupActionError(null);
    try {
      const response = await applyDatasetCleanup({
        runDir: cleanupTarget.runDir,
        criteria: cleanupCriteriaPayload,
      });
      await refreshDatasetRuns();
      const cleanedFileName =
        response.cleaned_file.split("/").pop() ?? "training-cleaned.csv";
      setWorkspaceTab("run_directory");
      setOpenRunId(response.cleaned_run_dir);
      setPreviewTarget({
        label: `${cleanedFileName} (cleaned)`,
        path: response.cleaned_file,
      });
      setAuditTargetPath(response.cleaned_file);
      handleCloseCleanupModal();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setCleanupActionError(message);
    } finally {
      setCleanupApplyLoading(false);
    }
  };

  const handleStartDateChange = useCallback((value: string) => {
    setFormState((prev) => ({ ...prev, start: value }));
    setRunError((current) => (current === DATE_RANGE_ERROR ? null : current));
  }, []);

  const handleEndDateChange = useCallback((value: string) => {
    setFormState((prev) => ({ ...prev, end: value }));
    setRunError((current) => (current === DATE_RANGE_ERROR ? null : current));
  }, []);

  const toggleUniverseTicker = useCallback(
    (ticker: string) => {
      const next = selectedTickerSet.has(ticker)
        ? selectedTickers.filter((value) => value !== ticker)
        : [...selectedTickers, ticker];
      updateTickers(next);
    },
    [selectedTickers, selectedTickerSet, updateTickers],
  );

  const removeSelectedTicker = useCallback(
    (ticker: string) => {
      if (!selectedTickerSet.has(ticker)) return;
      const next = selectedTickers.filter((value) => value !== ticker);
      updateTickers(next);
    },
    [selectedTickers, selectedTickerSet, updateTickers],
  );

  const handleAddCustomTicker = useCallback(() => {
    const tokens = splitTickerInput(customTickerInput);
    if (!tokens.length) return;
    const next = [...selectedTickers, ...tokens];
    updateTickers(next);
    setCustomTickerInput("");
  }, [customTickerInput, selectedTickers, updateTickers]);

  const handleNewJob = useCallback(() => {
    if (isDatasetJobActive) return;
    setRunJobPanel("configuration");
    setWorkspaceTab("run_job");
  }, [isDatasetJobActive]);
  const handleViewLatestRun = useCallback(() => {
    setRunJobPanel("active_run");
    setWorkspaceTab("run_job");
  }, []);
  const toggleRunOpen = useCallback(
    (
      runId: string,
      runDir: string,
      fileCount: number,
      trainingPath: string | null,
      trainingFileName?: string | null,
    ) => {
      if (fileCount === 0) return;
      setOpenRunId((prev) => {
        const nextOpen = prev === runId ? null : runId;
        if (nextOpen === runId) {
          if (trainingPath) {
            setAuditTargetPath(trainingPath);
            setPreviewTarget((current) =>
              current && current.path.startsWith(runDir)
                ? current
                : {
                    label: `${trainingFileName ?? "training.csv"} (training)`,
                    path: trainingPath,
                  },
            );
          }
        } else if (auditTargetPath?.startsWith(runDir)) {
          setAuditTargetPath(null);
        }
        return nextOpen;
      });
    },
    [auditTargetPath],
  );
  const handleOpenFullAudit = useCallback(() => {
    if (!activeRunAuditPath) return;
    const trainingName = activeRunAuditPath.split("/").pop() ?? activeRunAuditPath;
    setWorkspaceTab("run_directory");
    setOpenRunId(currentResult?.run_dir ?? null);
    setPreviewTarget({
      label: `${trainingName} (training)`,
      path: activeRunAuditPath,
    });
    setAuditTargetPath(activeRunAuditPath);
  }, [activeRunAuditPath, currentResult?.run_dir]);
  const deleteTargetRun = deleteConfirmRun
    ? datasetRuns.find((run) => run.id === deleteConfirmRun) ?? null
    : null;
  const deleteTargetRunName = deleteTargetRun
    ? deleteTargetRun.run_dir.split("/").pop() ?? deleteTargetRun.id
    : null;

  return (
    <section className="page datasets-page">
      <PipelineStatusCard
        className="page-sticky-meta datasets-meta"
        activeJobsCount={activeJobs.length}
      />
      <header className="page-header datasets-page-header">
        <div className="datasets-title-row">
          <h1 className="page-title datasets-page-title">
            Option Chain Dataset Builder
          </h1>
        </div>
      </header>

      <div className="datasets-workspace">
        <div
          className="datasets-workspace-tabs"
          role="tablist"
          aria-label="Option chain dataset builder workspace"
        >
          <button
            id="datasets-tab-run-job"
            type="button"
            role="tab"
            aria-selected={workspaceTab === "run_job"}
            aria-controls="datasets-panel-run-job"
            className={`datasets-workspace-tab ${
              workspaceTab === "run_job" ? "active" : ""
            }`}
            onClick={() => setWorkspaceTab("run_job")}
          >
            Run job
          </button>
          <button
            id="datasets-tab-run-directory"
            type="button"
            role="tab"
            aria-selected={workspaceTab === "run_directory"}
            aria-controls="datasets-panel-run-directory"
            className={`datasets-workspace-tab ${
              workspaceTab === "run_directory" ? "active" : ""
            }`}
            onClick={() => setWorkspaceTab("run_directory")}
          >
            Datasets
          </button>
        </div>

        {workspaceTab === "run_job" ? (
          <div
            id="datasets-panel-run-job"
            role="tabpanel"
            aria-labelledby="datasets-tab-run-job"
            className="datasets-tab-panel"
          >
            <div className="datasets-grid">
              {runJobPanel === "configuration" && !isDatasetJobActive ? (
                <section className="panel">
          <div className="panel-header datasets-job-config-header">
            <div>
              <h2 className="datasets-job-config-title">Job Configuration</h2>
            </div>
            <div className="datasets-job-config-actions">
              <button
                className="button ghost datasets-config-action-button"
                type="button"
                disabled={isRunning}
                onClick={() => setFormState(defaultForm)}
              >
                Reset config
              </button>
              <button
                className="button ghost datasets-config-action-button"
                type="button"
                disabled={!currentResult && !jobStatus}
                onClick={handleViewLatestRun}
              >
                View Latest Run
              </button>
            </div>
          </div>
          <div className="config-summary">
            <div>
              <span className="meta-label">Date range</span>
              <span>{resolvedRange}</span>
            </div>
            <div>
              <span className="meta-label">Tickers</span>
              <span>{resolvedTickersLabel}</span>
            </div>
            <div>
              <span className="meta-label">Dataset</span>
              <span>
                {formState.outDir}/{datasetNameKebab || "(unnamed)"}
              </span>
            </div>
            <div>
              <span className="meta-label">Schedule</span>
              <span>
                {formState.scheduleMode}
                {dteCount
                  ? ` · DTE ${formState.dteList || `${formState.dteMin}-${formState.dteMax}`}`
                  : ` · asof ${formState.asofWeekdays}`}
              </span>
            </div>
            <div>
              <span className="meta-label">Planned workload</span>
              <span>{plannedJobsLabel}</span>
            </div>
          </div>
          <form className="panel-body" onSubmit={handleSubmit}>
            <div className="section-card dataset-section datasets-core-range-section">
              <h3>Core range</h3>
              <div className="datasets-date-range-fields">
                <div className="field">
                  <label htmlFor="datasetStart">Start date</label>
                  <input
                    id="datasetStart"
                    className={`input ${dateRangeInvalid ? "input-invalid" : ""}`}
                    type="date"
                    min="2023-06-01"
                    max={todayDateString}
                    required
                    value={formState.start}
                    aria-invalid={dateRangeInvalid}
                    onChange={(event) =>
                      handleStartDateChange(event.target.value)
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="datasetEnd">End date</label>
                  <input
                    id="datasetEnd"
                    className={`input ${dateRangeInvalid ? "input-invalid" : ""}`}
                    type="date"
                    min="2023-06-01"
                    max={todayDateString}
                    required
                    value={formState.end}
                    aria-invalid={dateRangeInvalid}
                    onChange={(event) =>
                      handleEndDateChange(event.target.value)
                    }
                  />
                </div>
              </div>
              {dateRangeInvalid ? (
                <p className="field-hint datasets-date-range-hint is-invalid">
                  {DATE_RANGE_ERROR}
                </p>
              ) : null}
              <div className="field datasets-ticker-universe-field">
                <label>Trading universe</label>
                <div className="ticker-grid">
                  {TRADING_UNIVERSE_TICKERS.map((ticker) => (
                    <button
                      key={ticker}
                      type="button"
                      className={`ticker-chip ${
                        selectedTickerSet.has(ticker) ? "selected" : ""
                      }`}
                      aria-pressed={selectedTickerSet.has(ticker)}
                      onClick={() => toggleUniverseTicker(ticker)}
                    >
                      {ticker}
                    </button>
                  ))}
                </div>
              </div>
              <div className="field">
                <label htmlFor="datasetCustomTicker">Add custom tickers</label>
                <div className="inline-fields compact">
                  <input
                    id="datasetCustomTicker"
                    className="input"
                    placeholder="e.g. AMD, INTC"
                    value={customTickerInput}
                    onChange={(event) => setCustomTickerInput(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        event.preventDefault();
                        handleAddCustomTicker();
                      }
                    }}
                  />
                  <button
                    type="button"
                    className="button ghost datasets-ticker-add-button"
                    onClick={handleAddCustomTicker}
                  >
                    Add
                  </button>
                </div>
              </div>
              <div className="field">
                <label>Selected tickers</label>
                <div className="ticker-selection">
                  {selectedTickers.map((ticker) => {
                    const isCustom = !TRADING_UNIVERSE_TICKERS.includes(ticker);
                    return (
                      <button
                        key={ticker}
                        type="button"
                        className={`ticker-chip selected ${
                          isCustom ? "custom" : ""
                        }`}
                        onClick={() => removeSelectedTicker(ticker)}
                        aria-label={`Remove ${ticker}`}
                      >
                        <span>{ticker}</span>
                        <span className="ticker-remove">×</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            <div className="section-card dataset-section">
              <h3>Output targets</h3>
              <div className="inline-fields">
                <div className="field">
                  <label htmlFor="datasetName">Dataset name</label>
                  <input
                    id="datasetName"
                    className="input"
                    required
                    value={formState.datasetName}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        datasetName: event.target.value,
                      }))
                    }
                  />
                </div>
              </div>
              <div className="inline-fields">
                <div className="field">
                  <label htmlFor="prnVersion">pRN version</label>
                  <input
                    id="prnVersion"
                    className="input"
                    value={formState.prnVersion}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        prnVersion: event.target.value,
                      }))
                    }
                  />
                </div>
              </div>
              <div className="inline-fields">
                <div className="field datasets-output-targets-field">
                  <label>Outputs to generate</label>
                  <div className="datasets-output-targets" role="group" aria-label="Outputs to generate">
                    <button
                      type="button"
                      className="ticker-chip datasets-output-chip selected locked"
                      disabled
                      title="Training output is always generated"
                    >
                      Training
                    </button>
                    <button
                      type="button"
                      className={`ticker-chip datasets-output-chip ${
                        formState.writeSnapshot ? "selected" : ""
                      }`}
                      aria-pressed={formState.writeSnapshot}
                      onClick={() =>
                        setFormState((prev) => ({
                          ...prev,
                          writeSnapshot: !prev.writeSnapshot,
                        }))
                      }
                    >
                      Snapshot
                    </button>
                    <button
                      type="button"
                      className={`ticker-chip datasets-output-chip ${
                        formState.writePrnView ? "selected" : ""
                      }`}
                      aria-pressed={formState.writePrnView}
                      onClick={() =>
                        setFormState((prev) => ({
                          ...prev,
                          writePrnView: !prev.writePrnView,
                        }))
                      }
                    >
                      pRN View
                    </button>
                    <button
                      type="button"
                      className={`ticker-chip datasets-output-chip ${
                        formState.writeLegacy ? "selected" : ""
                      }`}
                      aria-pressed={formState.writeLegacy}
                      onClick={() =>
                        setFormState((prev) => ({
                          ...prev,
                          writeLegacy: !prev.writeLegacy,
                        }))
                      }
                    >
                      Legacy
                    </button>
                    <button
                      type="button"
                      className={`ticker-chip datasets-output-chip ${
                        formState.writeDrops ? "selected" : ""
                      }`}
                      aria-pressed={formState.writeDrops}
                      onClick={() =>
                        setFormState((prev) => ({
                          ...prev,
                          writeDrops: !prev.writeDrops,
                        }))
                      }
                    >
                      Drops
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <div className="section-card dataset-section">
              <h3>Snapshot schedule</h3>
              <div className="inline-fields">
                <div className="field">
                  <label htmlFor="scheduleMode">How are start/end dates used?</label>
                  <select
                    id="scheduleMode"
                    className="input"
                    value={formState.scheduleMode}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        scheduleMode: event.target
                          .value as DatasetFormState["scheduleMode"],
                      }))
                    }
                  >
                    <option value="weekly">Weekly (generate Monday anchors)</option>
                    <option value="expiry_range">Expiry range (start/end are expiry dates)</option>
                  </select>
                </div>
                {formState.scheduleMode === "expiry_range" ? (
                  <div className="field">
                    <label htmlFor="expiryWeekdays">Expiry weekdays</label>
                    <input
                      id="expiryWeekdays"
                      className="input"
                      value={formState.expiryWeekdays}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          expiryWeekdays: event.target.value,
                        }))
                      }
                    />
                    <span className="field-hint">
                      e.g. fri or mon,fri (comma-separated).
                    </span>
                  </div>
                ) : null}
              </div>
              <div className="inline-fields">
                <div className="field">
                  <label htmlFor="asofWeekdays">Observation weekdays</label>
                  <input
                    id="asofWeekdays"
                    className="input"
                    value={formState.asofWeekdays}
                    disabled={!!formState.dteList.trim()}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        asofWeekdays: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="dteList">DTE list (overrides weekdays)</label>
                  <input
                    id="dteList"
                    className="input"
                    placeholder="e.g. 1,2,3 or 1-5"
                    value={formState.dteList}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        dteList: event.target.value,
                      }))
                    }
                  />
                </div>
              </div>
              {formState.dteList.trim() ? (
                <div className="inline-fields">
                  <div className="field">
                    <label htmlFor="dteMin">DTE min</label>
                    <input
                      id="dteMin"
                      className="input"
                      inputMode="numeric"
                      value={formState.dteMin}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          dteMin: event.target.value,
                        }))
                      }
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="dteMax">DTE max</label>
                    <input
                      id="dteMax"
                      className="input"
                      inputMode="numeric"
                      value={formState.dteMax}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          dteMax: event.target.value,
                        }))
                      }
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="dteStep">DTE step</label>
                    <input
                      id="dteStep"
                      className="input"
                      inputMode="numeric"
                      value={formState.dteStep}
                      onChange={(event) =>
                        setFormState((prev) => ({
                          ...prev,
                          dteStep: event.target.value,
                        }))
                      }
                    />
                  </div>
                </div>
              ) : null}
            </div>

            <div className="section-card dataset-section datasets-settings-section">
              <h3>Settings</h3>
              <details className="advanced">
              <summary>Market data & runtime</summary>
              <div className="fields-grid">
                <div className="field">
                  <label htmlFor="thetaBaseUrl">Theta base URL</label>
                  <input
                    id="thetaBaseUrl"
                    className="input"
                    value={formState.thetaBaseUrl}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        thetaBaseUrl: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="stockSource">Stock source</label>
                  <select
                    id="stockSource"
                    className="input"
                    value={formState.stockSource}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        stockSource: event.target.value as DatasetFormState["stockSource"],
                      }))
                    }
                  >
                    <option value="yfinance">yfinance</option>
                    <option value="theta">theta</option>
                    <option value="auto">auto</option>
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="timeoutS">Timeout (s)</label>
                  <input
                    id="timeoutS"
                    className="input"
                    inputMode="numeric"
                    value={formState.timeoutS}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        timeoutS: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="riskFreeRate">Risk-free rate (r)</label>
                  <input
                    id="riskFreeRate"
                    className="input"
                    inputMode="decimal"
                    value={formState.riskFreeRate}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        riskFreeRate: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="threads">Threads</label>
                  <input
                    id="threads"
                    className="input"
                    inputMode="numeric"
                    value={formState.threads}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        threads: event.target.value,
                      }))
                    }
                  />
                </div>
              </div>
            </details>

            <details className="advanced">
              <summary>Band selection & training</summary>
              <div className="fields-grid">
                <div className="field">
                  <label htmlFor="maxAbsLogm">Max abs log-m</label>
                  <input
                    id="maxAbsLogm"
                    className="input"
                    inputMode="decimal"
                    value={formState.maxAbsLogm}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        maxAbsLogm: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="maxAbsLogmCap">Max abs log-m cap</label>
                  <input
                    id="maxAbsLogmCap"
                    className="input"
                    inputMode="decimal"
                    value={formState.maxAbsLogmCap}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        maxAbsLogmCap: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="bandWidenStep">Band widen step</label>
                  <input
                    id="bandWidenStep"
                    className="input"
                    inputMode="decimal"
                    value={formState.bandWidenStep}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        bandWidenStep: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="maxBandStrikes">Max band strikes</label>
                  <input
                    id="maxBandStrikes"
                    className="input"
                    inputMode="numeric"
                    value={formState.maxBandStrikes}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        maxBandStrikes: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="minBandStrikes">Min band strikes</label>
                  <input
                    id="minBandStrikes"
                    className="input"
                    inputMode="numeric"
                    value={formState.minBandStrikes}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        minBandStrikes: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="minBandPrnStrikes">
                    Min band pRN strikes
                  </label>
                  <input
                    id="minBandPrnStrikes"
                    className="input"
                    inputMode="numeric"
                    value={formState.minBandPrnStrikes}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        minBandPrnStrikes: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="minPrnTrain">Min pRN train</label>
                  <input
                    id="minPrnTrain"
                    className="input"
                    inputMode="decimal"
                    value={formState.minPrnTrain}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        minPrnTrain: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="maxPrnTrain">Max pRN train</label>
                  <input
                    id="maxPrnTrain"
                    className="input"
                    inputMode="decimal"
                    value={formState.maxPrnTrain}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        maxPrnTrain: event.target.value,
                      }))
                    }
                  />
                </div>
              </div>
              <div className="toggle-grid">
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.adaptiveBand}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        adaptiveBand: event.target.checked,
                      }))
                    }
                  />
                  Adaptive band
                </label>
              </div>
            </details>

            <details className="advanced">
              <summary>Option chain & expiry</summary>
              <div className="fields-grid">
                <div className="field">
                  <label htmlFor="strikeRange">Strike range</label>
                  <input
                    id="strikeRange"
                    className="input"
                    inputMode="numeric"
                    value={formState.strikeRange}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        strikeRange: event.target.value,
                      }))
                    }
                  />
                </div>
              </div>
              <div className="toggle-grid">
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.retryFullChain}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        retryFullChain: event.target.checked,
                      }))
                    }
                  />
                  Retry full chain if band thin
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.saturdayExpiryFallback}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        saturdayExpiryFallback: event.target.checked,
                      }))
                    }
                  />
                  Saturday expiry fallback
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.splitAdjust}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        splitAdjust: event.target.checked,
                      }))
                    }
                  />
                  Apply split adjustment
                </label>
              </div>
            </details>

            <details className="advanced">
              <summary>Liquidity & filters</summary>
              <div className="fields-grid">
                <div className="field">
                  <label htmlFor="minTradeCount">Min trade count</label>
                  <input
                    id="minTradeCount"
                    className="input"
                    inputMode="numeric"
                    value={formState.minTradeCount}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        minTradeCount: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="minVolume">Min volume</label>
                  <input
                    id="minVolume"
                    className="input"
                    inputMode="numeric"
                    value={formState.minVolume}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        minVolume: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="minChainUsedHard">Min chain used hard</label>
                  <input
                    id="minChainUsedHard"
                    className="input"
                    inputMode="numeric"
                    value={formState.minChainUsedHard}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        minChainUsedHard: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="maxRelSpreadMedianHard">
                    Max rel spread median hard
                  </label>
                  <input
                    id="maxRelSpreadMedianHard"
                    className="input"
                    inputMode="decimal"
                    value={formState.maxRelSpreadMedianHard}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        maxRelSpreadMedianHard: event.target.value,
                      }))
                    }
                  />
                </div>
              </div>
              <div className="toggle-grid">
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.preferBidask}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        preferBidask: event.target.checked,
                      }))
                    }
                  />
                  Prefer bid/ask quotes
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.hardDropCloseFallback}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        hardDropCloseFallback: event.target.checked,
                      }))
                    }
                  />
                  Hard drop close fallback
                </label>
              </div>
            </details>

            <details className="advanced">
              <summary>Dividends, weights & volatility</summary>
              <div className="fields-grid">
                <div className="field">
                  <label htmlFor="dividendSource">Dividend source</label>
                  <select
                    id="dividendSource"
                    className="input"
                    value={formState.dividendSource}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        dividendSource: event.target.value as DatasetFormState["dividendSource"],
                      }))
                    }
                  >
                    <option value="yfinance">yfinance</option>
                    <option value="none">none</option>
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="dividendLookbackDays">
                    Dividend lookback days
                  </label>
                  <input
                    id="dividendLookbackDays"
                    className="input"
                    inputMode="numeric"
                    value={formState.dividendLookbackDays}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        dividendLookbackDays: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="dividendYieldDefault">
                    Dividend yield default
                  </label>
                  <input
                    id="dividendYieldDefault"
                    className="input"
                    inputMode="decimal"
                    value={formState.dividendYieldDefault}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        dividendYieldDefault: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="rvLookbackDays">RV lookback days</label>
                  <input
                    id="rvLookbackDays"
                    className="input"
                    inputMode="numeric"
                    value={formState.rvLookbackDays}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        rvLookbackDays: event.target.value,
                      }))
                    }
                  />
                </div>
              </div>
              <div className="toggle-grid">
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.forwardMoneyness}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        forwardMoneyness: event.target.checked,
                      }))
                    }
                  />
                  Use forward moneyness
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.groupWeights}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        groupWeights: event.target.checked,
                      }))
                    }
                  />
                  Add group weights
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.tickerWeights}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        tickerWeights: event.target.checked,
                      }))
                    }
                  />
                  Add ticker weights
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.softQualityWeight}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        softQualityWeight: event.target.checked,
                      }))
                    }
                  />
                  Soft quality weighting
                </label>
              </div>
            </details>

            <details className="advanced">
              <summary>Cache & sanity checks</summary>
              <div className="fields-grid">
                <div className="field">
                  <label htmlFor="sanityAbsLogmMax">Sanity abs log-m max</label>
                  <input
                    id="sanityAbsLogmMax"
                    className="input"
                    inputMode="decimal"
                    value={formState.sanityAbsLogmMax}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        sanityAbsLogmMax: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="sanityKOverSMin">Sanity K/S min</label>
                  <input
                    id="sanityKOverSMin"
                    className="input"
                    inputMode="decimal"
                    value={formState.sanityKOverSMin}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        sanityKOverSMin: event.target.value,
                      }))
                    }
                  />
                </div>
                <div className="field">
                  <label htmlFor="sanityKOverSMax">Sanity K/S max</label>
                  <input
                    id="sanityKOverSMax"
                    className="input"
                    inputMode="decimal"
                    value={formState.sanityKOverSMax}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        sanityKOverSMax: event.target.value,
                      }))
                    }
                  />
                </div>
              </div>
              <div className="toggle-grid">
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.cache}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        cache: event.target.checked,
                      }))
                    }
                  />
                  Enable cache
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.sanityReport}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        sanityReport: event.target.checked,
                      }))
                    }
                  />
                  Sanity report
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.sanityDrop}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        sanityDrop: event.target.checked,
                      }))
                    }
                  />
                  Drop rows failing sanity
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={formState.verboseSkips}
                    onChange={(event) =>
                      setFormState((prev) => ({
                        ...prev,
                        verboseSkips: event.target.checked,
                      }))
                    }
                  />
                  Verbose skips
                </label>
              </div>
              </details>
            </div>

            {runError ? <div className="error">{runError}</div> : null}

            <div className="actions">
              <button
                className="button primary datasets-config-action-button"
                type="submit"
                disabled={isRunning || anyJobRunning}
              >
                {isRunning ? "Running job..." : "Run job"}
              </button>
            </div>
          </form>
        </section>
              ) : (
                <section className="panel">
          <div className="panel-header">
            <div>
              <h2 className="datasets-job-config-title">Active Run</h2>
              <span className="panel-hint">
                Monitor build telemetry, quick audit checks, and critical stderr.
              </span>
            </div>
            {showNewJobButton ? (
              <button className="button light" type="button" onClick={handleNewJob}>
                New job
              </button>
            ) : null}
          </div>
          <div className="panel-body">
            <div className="run-shell">
              <aside className="run-shell-sidebar">
                <div className="run-progress-panel">
                  <div className="run-progress-heading">
                    <span className="meta-label">Run monitor</span>
                    {jobStatus ? (
                      <span className={`status-pill ${statusClass}`}>
                        {statusLabel}
                      </span>
                    ) : (
                      <span className="status-pill idle">Idle</span>
                    )}
                  </div>
                  {jobStatus ? (
                    <>
                      <div className="run-progress-grid">
                        <div>
                          <span className="meta-label">Weeks</span>
                          <span>{plannedWeeksLabel}</span>
                        </div>
                        <div>
                          <span className="meta-label">Tickers</span>
                          <span>
                            {resolvedTickersCount.toLocaleString()}
                            {customTickerCount ? ` (${customTickerCount} custom)` : ""}
                          </span>
                        </div>
                        <div>
                          <span className="meta-label">Snapshot days</span>
                          <span>{snapshotCountLabel} per expiry</span>
                        </div>
                        <div>
                          <span className="meta-label">Planned jobs</span>
                          <span>{plannedJobsLabel}</span>
                        </div>
                        <div>
                          <span className="meta-label">Progress</span>
                          <span>
                            {jobProgress
                              ? `${jobProgress.done.toLocaleString()}/${jobProgress.total.toLocaleString()} (${progressPercent ?? 0}%)`
                              : "Waiting for progress…"}
                          </span>
                        </div>
                        <div>
                          <span className="meta-label">Groups kept</span>
                          <span>
                            {jobProgress ? jobProgress.groups.toLocaleString() : "—"}
                          </span>
                        </div>
                        <div>
                          <span className="meta-label">Rows</span>
                          <span>
                            {jobProgress ? jobProgress.rows.toLocaleString() : "—"}
                          </span>
                        </div>
                        <div>
                          <span className="meta-label">Last job</span>
                          <span>
                            {jobProgress
                              ? `${jobProgress.lastTicker} · ${jobProgress.lastWeek} · ${jobProgress.lastAsof}`
                              : "—"}
                          </span>
                        </div>
                      </div>
                      <div className="progress-tracker">
                        <div className="progress-label">
                          <span>Progress</span>
                          <span>
                            {progressPercent !== null ? `${progressPercent}%` : "—"}
                          </span>
                        </div>
                        <div className="progress-bar">
                          <div
                            className="progress-fill"
                            style={{ width: `${progressPercent ?? 0}%` }}
                          />
                        </div>
                      </div>
                      {(jobStatus.status === "running" ||
                        jobStatus.status === "queued") ? (
                        <button
                          className="button ghost kill-button danger"
                          type="button"
                          disabled={killLoading}
                          onClick={handleKill}
                        >
                          {killLoading ? "Stopping…" : "Stop run"}
                        </button>
                      ) : null}
                    </>
                  ) : (
                    <div className="run-progress-empty">
                      <p className="meta-label">No dataset build running</p>
                      <p>Start a run to see progress here.</p>
                    </div>
                  )}
                </div>
              </aside>
              <div className="run-shell-main">
                {!jobStatus ? (
                  <div className="empty">No dataset run yet.</div>
                ) : (
                  <div className="run-output">
                    <div className="run-summary">
                      <div className="run-summary-header">
                        <div>
                          <span className="meta-label">Output</span>
                          <div className="run-id">{outputLocationLabel}</div>
                        </div>
                        <div className="run-summary-actions">
                          <span className={`status-pill ${statusClass}`}>
                            {statusLabel}
                          </span>
                        </div>
                      </div>
                      <div className="run-meta-grid">
                        <div>
                          <span className="meta-label">Duration</span>
                          <span>
                            {currentResult
                              ? `${currentResult.duration_s.toFixed(2)}s`
                              : jobStatus.status === "running" ||
                                jobStatus.status === "queued"
                              ? formatDurationFromSeconds(elapsedSeconds)
                              : "Pending"}
                          </span>
                        </div>
                        <div>
                          <span className="meta-label">Output dir</span>
                          <span>{currentResult?.run_dir ?? currentResult?.out_dir ?? formState.outDir}</span>
                        </div>
                        <div>
                          <span className="meta-label">Training dataset</span>
                          <span>
                            {trainingDatasetEnabled
                              ? trainingDatasetPath ?? "Pending"
                              : "Not written"}
                          </span>
                        </div>
                        <div>
                          <span className="meta-label">Drops file</span>
                          <span>
                            {currentResult?.drops_file ??
                              (formState.writeDrops ? "Pending" : "Not written")}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="run-insights-panel">
                      <div className="run-insights-header">
                        <div>
                          <span className="meta-label">Build insights</span>
                          <p className="run-insights-title">
                            {activeRunAuditPath
                              ? "Compact final audit snapshot"
                              : "Live build telemetry"}
                          </p>
                        </div>
                        <span className={`status-pill ${statusClass}`}>
                          {activeRunAuditPath ? "Audit ready" : phaseLabel}
                        </span>
                      </div>
                      {activeRunAuditPath ? (
                        activeRunAuditLoading ? (
                          <div className="dataset-preview-empty">
                            Loading final audit snapshot…
                          </div>
                        ) : activeRunAuditError ? (
                          <div className="error">{activeRunAuditError}</div>
                        ) : activeRunAuditResponse ? (
                          <>
                            <div className="dataset-audit-kpis run-final-audit-kpis">
                              <div>
                                <span className="meta-label">Rows</span>
                                <strong>{activeRunAuditResponse.row_count.toLocaleString()}</strong>
                              </div>
                              <div>
                                <span className="meta-label">Tickers</span>
                                <strong>{activeRunAuditResponse.ticker_count?.toLocaleString() ?? "—"}</strong>
                              </div>
                              <div>
                                <span className="meta-label">Snapshots</span>
                                <strong>{activeRunAuditResponse.snapshot_count?.toLocaleString() ?? "—"}</strong>
                              </div>
                              <div>
                                <span className="meta-label">Date range</span>
                                <strong>
                                  {activeRunAuditResponse.date_start &&
                                  activeRunAuditResponse.date_end
                                    ? `${activeRunAuditResponse.date_start} → ${activeRunAuditResponse.date_end}`
                                    : "—"}
                                </strong>
                              </div>
                              <div>
                                <span className="meta-label">Expiry range</span>
                                <strong>
                                  {activeRunAuditResponse.expiry_start &&
                                  activeRunAuditResponse.expiry_end
                                    ? `${activeRunAuditResponse.expiry_start} → ${activeRunAuditResponse.expiry_end}`
                                    : "—"}
                                </strong>
                              </div>
                            </div>
                            <div className="dataset-audit-grid run-final-audit-grid">
                              <section className="dataset-audit-card">
                                <div className="dataset-audit-card-header">
                                  <h3>Quality flags</h3>
                                  <span>Top 5 by share</span>
                                </div>
                                {compactAuditFlags.length > 0 ? (
                                  <div className="dataset-audit-list">
                                    {compactAuditFlags.map((flag) => {
                                      const share = activeRunAuditResponse.row_count
                                        ? flag.count / activeRunAuditResponse.row_count
                                        : 0;
                                      return (
                                        <div key={flag.name} className="dataset-audit-list-row">
                                          <div className="dataset-audit-list-label">
                                            <code>{flag.name}</code>
                                            <span>{flag.count.toLocaleString()} rows</span>
                                          </div>
                                          <div className="dataset-audit-bar-track">
                                            <div
                                              className="dataset-audit-bar-fill"
                                              style={{ width: shareBarWidth(share) }}
                                            />
                                          </div>
                                          <span className="dataset-audit-list-value">
                                            {formatPercent(flag.share)}
                                          </span>
                                        </div>
                                      );
                                    })}
                                  </div>
                                ) : (
                                  <div className="dataset-preview-empty">
                                    No quality flags were detected in the final training file.
                                  </div>
                                )}
                              </section>
                              <ProblemTickerAuditCard
                                subtitle="Top 5 by issue load"
                                items={compactProblemTickers}
                                emptyMessage="No ticker-level issues were detected."
                              />
                            </div>
                            <div className="run-insights-cta">
                              <span>
                                The full drill-through audit remains in the Run Directory tab.
                              </span>
                              <button
                                className="button light"
                                type="button"
                                onClick={handleOpenFullAudit}
                              >
                                Open full audit
                              </button>
                            </div>
                          </>
                        ) : (
                          <div className="dataset-preview-empty">
                            Final audit snapshot is not available yet.
                          </div>
                        )
                      ) : (
                        <div className="run-insights-grid">
                          <section className="run-insight-card">
                            <div className="run-insight-card-header">
                              <h3>Build overview</h3>
                              <span>{phaseLabel}</span>
                            </div>
                            <div className="run-insight-metrics">
                              <div>
                                <span className="meta-label">Status</span>
                                <strong>{statusLabel}</strong>
                              </div>
                              <div>
                                <span className="meta-label">Last job</span>
                                <strong>
                                  {jobProgress
                                    ? `${jobProgress.lastTicker} · ${jobProgress.lastWeek} · ${jobProgress.lastAsof}`
                                    : "Waiting for progress…"}
                                </strong>
                              </div>
                              <div className="run-insight-metric-wide">
                                <span className="meta-label">Output path</span>
                                <strong>{outputLocationLabel}</strong>
                              </div>
                              <div>
                                <span className="meta-label">Duration</span>
                                <strong>{formatDurationFromSeconds(elapsedSeconds)}</strong>
                              </div>
                              <div>
                                <span className="meta-label">Training dataset</span>
                                <strong>{trainingDatasetPath ?? "Pending"}</strong>
                              </div>
                            </div>
                          </section>
                          <section className="run-insight-card">
                            <div className="run-insight-card-header">
                              <h3>Build yield</h3>
                              <span>
                                {jobProgress
                                  ? `${jobProgress.done.toLocaleString()}/${jobProgress.total.toLocaleString()}`
                                  : "Waiting"}
                              </span>
                            </div>
                            <div className="run-insight-metrics">
                              <div>
                                <span className="meta-label">Progress</span>
                                <strong>
                                  {jobProgress
                                    ? `${progressPercent ?? 0}%`
                                    : "Waiting for progress…"}
                                </strong>
                              </div>
                              <div>
                                <span className="meta-label">Kept-group rate</span>
                                <strong>{formatPercent(keptGroupRate)}</strong>
                              </div>
                              <div>
                                <span className="meta-label">Rows</span>
                                <strong>{jobProgress?.rows.toLocaleString() ?? "—"}</strong>
                              </div>
                              <div>
                                <span className="meta-label">Rows / kept group</span>
                                <strong>{formatNumeric(rowsPerKeptGroup, 1)}</strong>
                              </div>
                              <div>
                                <span className="meta-label">Jobs / min</span>
                                <strong>
                                  {jobsPerMinute && Number.isFinite(jobsPerMinute)
                                    ? jobsPerMinute.toFixed(1)
                                    : "—"}
                                </strong>
                              </div>
                              <div>
                                <span className="meta-label">ETA</span>
                                <strong>
                                  {isJobInFlight
                                    ? formatDurationFromSeconds(etaSeconds)
                                    : "—"}
                                </strong>
                              </div>
                            </div>
                          </section>
                          <section className="run-insight-card run-insight-card-wide">
                            <div className="run-insight-card-header">
                              <h3>Quick audit checks</h3>
                              <span>{telemetryKeptGroups.toLocaleString()} kept groups</span>
                            </div>
                            <div className="run-check-grid">
                              {quickAuditChecks.map((item) => (
                                <div key={item.key} className="run-check-card" title={item.description}>
                                  <div className="run-check-topline">
                                    <span>{item.label}</span>
                                    <strong>{item.count.toLocaleString()}</strong>
                                  </div>
                                  <div className="dataset-audit-bar-track">
                                    <div
                                      className="dataset-audit-bar-fill"
                                      style={{ width: shareBarWidth(item.share) }}
                                    />
                                  </div>
                                  <span className="run-check-share">
                                    {item.share == null ? "Waiting for kept groups…" : formatPercent(item.share)}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </section>
                          <ProblemTickerAuditCard
                            className="dataset-audit-card"
                            subtitle="Top 6 so far · kept rows only"
                            items={problemTickers}
                            emptyMessage="Waiting for enough kept rows to rank ticker diagnostics."
                          />
                          <section className="run-insight-card">
                            <div className="run-insight-card-header">
                              <h3>Ticker progress</h3>
                              <span>
                                {tickerProgressItems.length
                                  ? "Top 6 by drop pressure"
                                  : "No drop pressure yet"}
                              </span>
                            </div>
                            {tickerProgressItems.length > 0 ? (
                              <div className="run-ticker-table">
                                <div className="run-ticker-table-head">
                                  <span>Ticker</span>
                                  <span>Done / planned</span>
                                  <span>Kept</span>
                                  <span>Rows</span>
                                  <span>Top drop reason</span>
                                </div>
                                {tickerProgressItems.map((item) => (
                                  <div key={item.ticker} className="run-ticker-table-row">
                                    <strong>{item.ticker}</strong>
                                    <span>
                                      {item.completed_jobs.toLocaleString()} /{" "}
                                      {item.planned_jobs.toLocaleString()}
                                    </span>
                                    <span>{item.kept_groups.toLocaleString()}</span>
                                    <span>{item.rows.toLocaleString()}</span>
                                    <span>
                                      {topDropReason(item.drop_reasons)}{" "}
                                      {countDropReasons(item.drop_reasons)
                                        ? `(${formatPercent(getTickerDropShare(item))})`
                                        : ""}
                                    </span>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="dataset-preview-empty">
                                Waiting for enough completed jobs to rank ticker risk.
                              </div>
                            )}
                          </section>
                          <section className="run-insight-card">
                            <div className="run-insight-card-header">
                              <h3>Recent warnings</h3>
                              <span>{recentWarnings.length ? "Newest 3" : "No warnings yet"}</span>
                            </div>
                            {recentWarnings.length > 0 ? (
                              <div className="run-warning-list">
                                {recentWarnings.map((line, index) => (
                                  <div key={`${line}-${index}`} className="run-warning-item">
                                    <code>{line}</code>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="dataset-preview-empty">
                                No high-value warnings have been emitted yet.
                              </div>
                            )}
                          </section>
                        </div>
                      )}
                    </div>
                    {showCriticalErrors ? (
                      <div className="run-error-panel">
                        <div className="run-insight-card-header">
                          <h3>Critical errors</h3>
                          <span>stderr</span>
                        </div>
                        <pre>{criticalErrorText || "No critical stderr captured."}</pre>
                      </div>
                    ) : null}
                    {currentResult?.command ? (
                      <details className="command-details">
                        <summary>Command used</summary>
                        <code>{currentResult.command.join(" ")}</code>
                      </details>
                    ) : null}
                  </div>
                )}
              </div>
            </div>
          </div>
                </section>
              )}
            </div>
          </div>
        ) : (
          <div
            id="datasets-panel-run-directory"
            role="tabpanel"
            aria-labelledby="datasets-tab-run-directory"
            className="datasets-tab-panel"
          >
            <section className="panel dataset-registry-panel">
        <div className="panel-header datasets-job-config-header">
          <div>
            <h2 className="datasets-job-config-title">Datasets</h2>
          </div>
        </div>
        <div className="panel-body dataset-registry-body">
          <div className="dataset-runs-list">
            {runsLoading ? (
              <div className="empty">Loading dataset exports…</div>
            ) : runsError ? (
              <div className="error">{runsError}</div>
            ) : datasetRuns.length === 0 ? (
              <div className="empty">
                No dataset exports yet. Run the builder to create a CSV snapshot.
              </div>
            ) : (
              datasetRuns.map((run) => {
                const runName = run.run_dir.split("/").pop() ?? run.id;
                const trainingFile = run.training_file ?? null;
                const trainingPath = trainingFile?.path ?? null;
                const isCreatingRun = run.status === "creating";
                const files = sortRunFiles(buildRunFiles(run), trainingPath);
                const fileCount = files.length;
                const filesLabel = fileCount
                  ? `${fileCount} CSV${fileCount === 1 ? "" : "s"}`
                  : "No CSV files";
                const trainingLabel = trainingFile
                  ? `Training: ${trainingFile.name}`
                  : "Training file missing";
                const isRenaming = renamingRunId === run.id;
                const isPreviewingRun = Boolean(
                  previewTarget?.path.startsWith(run.run_dir),
                );
                const auditPathForRun = auditTargetPath?.startsWith(run.run_dir)
                  ? auditTargetPath
                  : null;
                const isAuditingRun = Boolean(auditPathForRun);
                const auditedFileName =
                  auditPathForRun?.split("/").pop() ??
                  trainingFile?.name ??
                  "Training dataset";
                // Derived from single openRunId state — no per-item boolean flags
                const isOpen = openRunId === run.id;
                return (
                  <article
                    key={run.id}
                    className={`dataset-run-item${isOpen ? " is-open" : ""}${
                      isCreatingRun ? " is-creating" : ""
                    }`}
                    onClick={
                      isRenaming || fileCount === 0
                        ? undefined
                        : (event) => {
                            if (shouldIgnoreRunCardToggle(event.target)) {
                              return;
                            }
                            toggleRunOpen(
                              run.id,
                              run.run_dir,
                              fileCount,
                              trainingPath,
                              trainingFile?.name,
                            );
                          }
                    }
                  >
                    {/* ── Accordion Header ────────────────────────────────────────
                        Rename mode  : plain <div> — no toggle while input is active.
                        Display mode : native <button> acts as the accordion trigger;
                                       Enter/Space activation handled by the browser.
                        No chevron or expand behavior when there are no CSV files.
                    ─────────────────────────────────────────────────────────────── */}
                    {isRenaming ? (
                      <div className="dataset-run-main">
                        <div className="rename-input-wrapper">
                          <input
                            className="input rename-input"
                            type="text"
                            value={renameValue}
                            onChange={(event) =>
                              setRenameValue(event.target.value)
                            }
                            onKeyDown={(event) => {
                              if (event.key === "Enter") {
                                handleConfirmRename(run.id, run.run_dir);
                              } else if (event.key === "Escape") {
                                handleCancelRename();
                              }
                            }}
                            autoFocus
                          />
                          {renameError ? (
                            <div className="error">{renameError}</div>
                          ) : null}
                          <div className="rename-actions">
                            <button
                              className="button ghost small"
                              type="button"
                              onClick={() =>
                                handleConfirmRename(run.id, run.run_dir)
                              }
                              disabled={renameLoading}
                            >
                              {renameLoading ? "Saving…" : "Save"}
                            </button>
                            <button
                              className="button ghost small"
                              type="button"
                              onClick={handleCancelRename}
                              disabled={renameLoading}
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                        <div className="dataset-run-meta">
                          <span>{formatTimestamp(run.last_modified)}</span>
                          <span>{filesLabel}</span>
                          <span>{trainingLabel}</span>
                        </div>
                        <div className="dataset-run-path">{run.run_dir}</div>
                      </div>
                    ) : (
                      // Display mode: button is the accordion trigger.
                      // Clicking toggles this run open; clicking again collapses it.
                      // Only one run may be open at a time (single openRunId state).
                      <button
                        type="button"
                        className="dataset-run-toggle"
                        aria-expanded={fileCount > 0 ? isOpen : undefined}
                        aria-controls={
                          fileCount > 0 ? `run-files-${run.id}` : undefined
                        }
                        onClick={() =>
                          toggleRunOpen(
                            run.id,
                            run.run_dir,
                            fileCount,
                            trainingPath,
                            trainingFile?.name,
                          )
                        }
                      >
                        <div className="dataset-run-main">
                          <div className="dataset-run-heading">
                            <div className="dataset-run-title">{runName}</div>
                            {isCreatingRun ? (
                              <span className="status-pill running">Creating</span>
                            ) : null}
                          </div>
                          <div className="dataset-run-meta">
                            <span>{formatTimestamp(run.last_modified)}</span>
                            <span>{filesLabel}</span>
                            <span>{trainingLabel}</span>
                          </div>
                          <div className="dataset-run-path">{run.run_dir}</div>
                        </div>
                        {/* Chevron: ▸ rotates 90° via CSS when .is-open is on the article */}
                        {fileCount > 0 ? (
                          <span className="dataset-run-chevron" aria-hidden="true">
                            ▸
                          </span>
                        ) : null}
                      </button>
                    )}
                    <div className="dataset-run-actions">
                      {isCreatingRun ? (
                        <button
                          type="button"
                          className="button ghost danger small"
                          onClick={handleViewLatestRun}
                        >
                          Stop run
                        </button>
                      ) : (
                        <>
                          <button
                            type="button"
                            className="button light small"
                            onClick={() =>
                              trainingFile
                                ? handleOpenCleanupModal(run, trainingFile)
                                : undefined
                            }
                            disabled={!trainingFile}
                          >
                            Clean noisy rows
                          </button>
                          <button
                            type="button"
                            className="button light small"
                            onClick={() => handleStartRename(run.id, runName)}
                            disabled={isRenaming || renameLoading}
                          >
                            Rename dataset
                          </button>
                          <button
                            type="button"
                            className="button ghost danger small"
                            onClick={() => {
                              setDeleteConfirmRun(run.id);
                              setDeleteConfirmText("");
                            }}
                            disabled={deleteLoadingRun === run.id}
                          >
                            {deleteLoadingRun === run.id
                              ? "Deleting…"
                              : "Delete dataset"}
                          </button>
                        </>
                      )}
                    </div>
                    {/* ── Collapsible Drawer ────────────────────────────────────────
                        Hidden by default via the `hidden` attribute.
                        Not rendered when there are no CSV files in this run.
                        The id links to aria-controls on the button.
                    ─────────────────────────────────────────────────────────────── */}
                    {fileCount > 0 ? (
                      <div
                        id={`run-files-${run.id}`}
                        className={`dataset-run-files-drawer${isOpen ? " is-open" : ""}`}
                        hidden={!isOpen}
                      >
                        <div className="dataset-run-files">
                          {files.map((file) => {
                            const isTraining = trainingPath === file.path;
                            const isCleanedVariant =
                              isCleanedDatasetFile(file);
                            return (
                              <div key={file.path} className="dataset-run-file">
                                <div className="dataset-run-file-info">
                                  <div className="dataset-run-file-name">
                                    {file.name}
                                  </div>
                                  <div className="dataset-run-file-meta">
                                    {isTraining ? (
                                      <span className="dataset-run-file-tag">
                                        Training
                                      </span>
                                    ) : null}
                                    <span>{formatByteCount(file.size_bytes)}</span>
                                  </div>
                                </div>
                                <div className="dataset-run-file-actions">
                                  <button
                                    type="button"
                                    className="button light small"
                                    onClick={() =>
                                      handlePreviewSelection(
                                        {
                                          label: `${file.name}${
                                            isTraining
                                              ? " (training)"
                                              : isCleanedVariant
                                                ? " (cleaned)"
                                                : ""
                                          }`,
                                          path: file.path,
                                        },
                                        {
                                          auditPath:
                                            isTraining || isCleanedVariant
                                              ? file.path
                                              : trainingPath,
                                        },
                                      )
                                    }
                                  >
                                    Preview
                                  </button>
                                  <a
                                    href={getDatasetFileUrl(file.path)}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="button light small"
                                  >
                                    Open
                                  </a>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                        {isPreviewingRun ? (
                          <div className="dataset-preview-panel dataset-preview-inline">
                            <div className="dataset-preview-header">
                              <div>
                                <span className="meta-label">CSV preview</span>
                                <p className="dataset-preview-title">
                                  {previewTarget?.label ??
                                    "Select a CSV to peek at its rows."}
                                </p>
                              </div>
                              <div className="dataset-preview-controls">
                                <label className="dataset-preview-control">
                                  <span className="meta-label">Range</span>
                                  <select
                                    className="input"
                                    value={previewMode}
                                    onChange={(event) =>
                                      setPreviewMode(
                                        event.target.value as PreviewMode,
                                      )
                                    }
                                  >
                                    {PREVIEW_MODE_OPTIONS.map((option) => (
                                      <option
                                        key={option.value}
                                        value={option.value}
                                      >
                                        {option.label}
                                      </option>
                                    ))}
                                  </select>
                                </label>
                                <label className="dataset-preview-control">
                                  <span className="meta-label">Rows</span>
                                  <select
                                    className="input"
                                    value={previewLimit}
                                    onChange={(event) => {
                                      const next = Number.parseInt(
                                        event.target.value,
                                        10,
                                      );
                                      setPreviewLimit(
                                        Number.isFinite(next)
                                          ? next
                                          : PREVIEW_LIMIT_DEFAULT,
                                      );
                                    }}
                                  >
                                    {PREVIEW_LIMIT_OPTIONS.map((limit) => (
                                      <option key={limit} value={limit}>
                                        {limit} rows
                                      </option>
                                    ))}
                                  </select>
                                </label>
                              </div>
                            </div>
                            {previewLoading ? (
                              <div className="dataset-preview-empty">
                                Loading preview…
                              </div>
                            ) : previewError ? (
                              <div className="error">{previewError}</div>
                            ) : previewResponse ? (
                              <>
                                {previewResponse.headers.length > 0 ? (
                                  <div className="table-container">
                                    <table className="preview-table">
                                      <thead>
                                        <tr>
                                          {previewResponse.headers.map(
                                            (column) => (
                                              <th key={column}>{column}</th>
                                            ),
                                          )}
                                        </tr>
                                      </thead>
                                      <tbody>
                                        {previewResponse.rows.length > 0 ? (
                                          previewResponse.rows.map(
                                            (row, index) => (
                                              <tr key={index}>
                                                {previewResponse.headers.map(
                                                  (column) => (
                                                    <td key={column}>
                                                      {row[column] ?? ""}
                                                    </td>
                                                  ),
                                                )}
                                              </tr>
                                            ),
                                          )
                                        ) : (
                                          <tr>
                                            <td
                                              colSpan={
                                                previewResponse.headers
                                                  .length || 1
                                              }
                                            >
                                              No rows to display.
                                            </td>
                                          </tr>
                                        )}
                                      </tbody>
                                    </table>
                                  </div>
                                ) : (
                                  <div className="dataset-preview-empty">
                                    CSV preview did not include column headers.
                                  </div>
                                )}
                                <div className="dataset-preview-meta">
                                  <span className="meta-label">
                                    Showing{" "}
                                    {previewResponse.mode === "tail"
                                      ? "last"
                                      : "first"}{" "}
                                    ({previewResponse.limit} rows)
                                  </span>
                                  <span>
                                    {previewResponse.row_count
                                      ? `${previewResponse.row_count.toLocaleString()} total rows`
                                      : "Row count unknown"}
                                  </span>
                                </div>
                              </>
                            ) : null}
                          </div>
                        ) : null}
                        {isAuditingRun ? (
                          <div className="dataset-audit-panel">
                            <div className="dataset-audit-header">
                              <div>
                                <span className="meta-label">Dataset audit</span>
                                <p className="dataset-audit-title">
                                  {auditedFileName}
                                </p>
                              </div>
                              <div className="dataset-audit-meta">
                                <span>{auditPathForRun}</span>
                              </div>
                            </div>
                            {auditLoading ? (
                              <div className="dataset-preview-empty">
                                Loading audit summary…
                              </div>
                            ) : auditError ? (
                              <div className="error">{auditError}</div>
                            ) : auditResponse ? (
                              <>
                                <div className="dataset-audit-kpis">
                                  <div>
                                    <span className="meta-label">Rows</span>
                                    <strong>{auditResponse.row_count.toLocaleString()}</strong>
                                  </div>
                                  <div>
                                    <span className="meta-label">Columns</span>
                                    <strong>{auditResponse.column_count.toLocaleString()}</strong>
                                  </div>
                                  <div>
                                    <span className="meta-label">Tickers</span>
                                    <strong>{auditResponse.ticker_count?.toLocaleString() ?? "—"}</strong>
                                  </div>
                                  <div>
                                    <span className="meta-label">Snapshots</span>
                                    <strong>{auditResponse.snapshot_count?.toLocaleString() ?? "—"}</strong>
                                  </div>
                                  <div>
                                    <span className="meta-label">Date range</span>
                                    <strong>
                                      {auditResponse.date_start && auditResponse.date_end
                                        ? `${auditResponse.date_start} → ${auditResponse.date_end}`
                                        : "—"}
                                    </strong>
                                  </div>
                                  <div>
                                    <span className="meta-label">Expiry range</span>
                                    <strong>
                                      {auditResponse.expiry_start && auditResponse.expiry_end
                                        ? `${auditResponse.expiry_start} → ${auditResponse.expiry_end}`
                                        : "—"}
                                    </strong>
                                  </div>
                                </div>
                                <div className="dataset-audit-grid">
                                  <section className="dataset-audit-card dataset-audit-card-wide">
                                    <div className="dataset-audit-card-header">
                                      <h3>Quality flags</h3>
                                      <span>{auditResponse.available_quality_flags.length} active columns</span>
                                    </div>
                                    {auditResponse.quality_flags.length > 0 ? (
                                      <div className="dataset-audit-list">
                                    {auditResponse.quality_flags.map((flag) => {
                                          const share = auditResponse.row_count
                                            ? flag.count / auditResponse.row_count
                                            : 0;
                                          return (
                                            <div key={flag.name} className="dataset-audit-list-row">
                                              <div className="dataset-audit-list-label">
                                                <code>{flag.name}</code>
                                                <span>{flag.count.toLocaleString()} rows</span>
                                              </div>
                                              <div className="dataset-audit-bar-track">
                                                <div
                                                  className="dataset-audit-bar-fill"
                                                  style={{ width: shareBarWidth(share) }}
                                                />
                                              </div>
                                              <span className="dataset-audit-list-value">
                                                {formatPercent(flag.share)}
                                              </span>
                                            </div>
                                          );
                                        })}
                                      </div>
                                    ) : (
                                      <div className="dataset-preview-empty">
                                        No explicit `flag_*` columns found yet.
                                      </div>
                                )}
                              </section>
                              <ProblemTickerAuditCard
                                className="dataset-audit-card-wide"
                                subtitle={`${fullAuditProblemTickers.length} ranked`}
                                items={fullAuditProblemTickers}
                                emptyMessage="Ticker-level diagnostics are unavailable."
                              />
                              <RvSurfaceAuditCard auditResponse={auditResponse} />
                                  <section className="dataset-audit-card dataset-audit-card-wide">
                                    <div className="dataset-audit-card-header">
                                      <h3>Coverage heatmap</h3>
                                      <div className="dataset-preview-controls">
                                        <label className="dataset-preview-control">
                                          <span className="meta-label">Metric</span>
                                          <select
                                            className="input"
                                            value={heatmapMetric}
                                            onChange={(event) =>
                                              setHeatmapMetric(event.target.value as HeatmapMetric)
                                            }
                                          >
                                            {HEATMAP_METRIC_OPTIONS.map((option) => (
                                              <option key={option.value} value={option.value}>
                                                {option.label}
                                              </option>
                                            ))}
                                          </select>
                                        </label>
                                      </div>
                                    </div>
                                    {auditResponse.heatmap_dates.length > 0 &&
                                    heatmapTickers.length > 0 ? (
                                      <div className="dataset-audit-heatmap-wrap">
                                        <div
                                          className="dataset-audit-heatmap-grid"
                                          style={{
                                            gridTemplateColumns: `minmax(72px, auto) repeat(${auditResponse.heatmap_dates.length}, minmax(1.75rem, 1.75rem))`,
                                          }}
                                        >
                                          <div className="dataset-audit-heatmap-corner">Ticker</div>
                                          {auditResponse.heatmap_dates.map((date) => (
                                            <div
                                              key={date}
                                              className="dataset-audit-heatmap-date"
                                              title={date}
                                            >
                                              {date.slice(5)}
                                            </div>
                                          ))}
                                          {heatmapTickers.map((ticker) => (
                                            <div key={ticker} className="dataset-audit-heatmap-row">
                                              <div key={`${ticker}-label`} className="dataset-audit-heatmap-ticker">
                                                {ticker}
                                              </div>
                                              {auditResponse.heatmap_dates.map((date) => {
                                                const cell = heatmapCellMap.get(`${ticker}__${date}`);
                                                let intensity = 0;
                                                if (cell) {
                                                  if (heatmapMetric === "rows") {
                                                    intensity = cell.row_count / heatmapMaxRows;
                                                  } else if (heatmapMetric === "flagged") {
                                                    intensity = cell.flagged_share ?? 0;
                                                  } else {
                                                    intensity = Math.min((cell.avg_issue_count ?? 0) / 3, 1);
                                                  }
                                                }
                                                return (
                                                  <button
                                                    key={`${ticker}-${date}`}
                                                    type="button"
                                                    className={`dataset-audit-heatmap-cell${
                                                      cell?.quality_bucket ? ` ${cell.quality_bucket}` : ""
                                                    }`}
                                                    style={{ opacity: cell ? Math.max(0.18, intensity) : 0.08 }}
                                                    title={
                                                      cell
                                                        ? `${ticker} ${date}: ${cell.row_count} rows, ${formatNumeric(cell.avg_issue_count, 2)} avg issues, ${formatPercent(cell.flagged_share)} flagged`
                                                        : `${ticker} ${date}: no rows`
                                                    }
                                                  >
                                                    {cell ? cell.row_count : ""}
                                                  </button>
                                                );
                                              })}
                                            </div>
                                          ))}
                                        </div>
                                      </div>
                                    ) : (
                                      <div className="dataset-preview-empty">
                                        Heatmap coverage is unavailable for this dataset.
                                      </div>
                                    )}
                                  </section>
                                </div>
                              </>
                            ) : null}
                          </div>
                        ) : null}
                      </div>
                    ) : null}
                  </article>
                );
              })
            )}
          </div>
        </div>
            </section>
          </div>
        )}
      </div>
      {cleanupTarget ? (
        <div
          className="dataset-delete-modal-overlay"
          onClick={() => {
            if (cleanupApplyLoading) return;
            handleCloseCleanupModal();
          }}
        >
          <div
            className="dataset-cleanup-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="dataset-cleanup-modal-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="dataset-delete-modal-header">
              <h3 id="dataset-cleanup-modal-title">Create cleaned dataset</h3>
              <p>
                This creates a cleaned clone of{" "}
                <span className="dataset-delete-modal-code">
                  {cleanupTarget.runName}
                </span>
                . Standard dataset artifacts are copied into a new sibling run,
                and rows matching the criteria below are removed from the copied
                row-level CSVs.
              </p>
            </div>
            <div className="dataset-cleanup-modal-body">
              <div className="dataset-cleanup-note">
                Training file:{" "}
                <span className="dataset-delete-modal-code">
                  {cleanupTarget.trainingFile.name}
                </span>
              </div>
              {cleanupActionError ? (
                <div className="error">{cleanupActionError}</div>
              ) : null}
              <section className="dataset-cleanup-section">
                <div className="dataset-cleanup-section-header">
                  <h4>Criteria</h4>
                  <span>
                    Pick one cleanup mode: quality buckets or flag filters.
                  </span>
                </div>
                <div className="dataset-cleanup-form-grid">
                  <div className="field dataset-cleanup-mode-field">
                    <label>Cleanup mode</label>
                    <div className="dataset-cleanup-toggle-row dataset-cleanup-mode-toggle">
                      <button
                        type="button"
                        className={`button ghost small${
                          cleanupCriteriaForm.mode === "quality_buckets"
                            ? " is-selected"
                            : ""
                        }`}
                        onClick={() =>
                          setCleanupCriteriaForm((prev) => ({
                            ...prev,
                            mode: "quality_buckets",
                          }))
                        }
                      >
                        Quality buckets
                      </button>
                      <button
                        type="button"
                        className={`button ghost small${
                          cleanupCriteriaForm.mode === "flags" ? " is-selected" : ""
                        }`}
                        onClick={() =>
                          setCleanupCriteriaForm((prev) => ({
                            ...prev,
                            mode: "flags",
                          }))
                        }
                      >
                        Flag filters
                      </button>
                    </div>
                  </div>
                  {cleanupCriteriaForm.mode === "quality_buckets" ? (
                    <div className="field dataset-cleanup-mode-panel">
                      <label>Quality buckets</label>
                      <div className="dataset-cleanup-chip-row">
                        {(["clean", "watch", "noisy"] as const).map((bucket) => (
                          <button
                            key={bucket}
                            type="button"
                            className={`dataset-cleanup-chip quality-${bucket}${
                              cleanupCriteriaForm.qualityBuckets.includes(bucket)
                                ? " selected"
                                : ""
                            }`}
                            onClick={() => toggleCleanupBucket(bucket)}
                          >
                            {bucket}
                          </button>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="field dataset-cleanup-mode-panel">
                        <label>Flag filters</label>
                        {cleanupAuditRequestLoading && !cleanupAudit ? (
                          <div className="dataset-preview-empty">
                            Loading audit flags…
                          </div>
                        ) : cleanupFlagOptions.length > 0 ? (
                          <div className="dataset-cleanup-chip-row">
                            {cleanupFlagOptions.map((flag) => (
                              <button
                                key={flag}
                                type="button"
                                className={`dataset-cleanup-chip${
                                  cleanupCriteriaForm.selectedFlags.includes(flag)
                                    ? " selected"
                                    : ""
                                }`}
                                onClick={() => toggleCleanupFlag(flag)}
                              >
                                {flag}
                              </button>
                            ))}
                          </div>
                        ) : (
                          <div className="dataset-preview-empty">
                            No `flag_*` columns available for this training file.
                          </div>
                        )}
                      </div>
                      <div className="field">
                        <label>Flag match mode</label>
                        <div className="dataset-cleanup-toggle-row">
                          {(["any", "all"] as const).map((mode) => (
                            <button
                              key={mode}
                              type="button"
                              className={`button ghost small${
                                cleanupCriteriaForm.flagMatchMode === mode
                                  ? " is-selected"
                                  : ""
                              }`}
                              onClick={() =>
                                setCleanupCriteriaForm((prev) => ({
                                  ...prev,
                                  flagMatchMode: mode,
                                }))
                              }
                            >
                              Match {mode}
                            </button>
                          ))}
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </section>

              <section className="dataset-cleanup-section">
                <div className="dataset-cleanup-section-header">
                  <h4>Preview</h4>
                  <span>Preview impact for the active cleanup mode.</span>
                </div>
                {cleanupAuditRequestLoading && !cleanupAudit ? (
                  <div className="dataset-preview-empty">
                    Loading audit summary…
                  </div>
                ) : cleanupPreviewLoading ? (
                  <div className="dataset-preview-empty">
                    Refreshing cleanup preview…
                  </div>
                ) : !cleanupModeHasSelection ? (
                  <div className="dataset-preview-empty">
                    Select at least one{" "}
                    {cleanupCriteriaForm.mode === "quality_buckets"
                      ? "quality bucket"
                      : "flag"}{" "}
                    to preview cleanup impact.
                  </div>
                ) : cleanupPreviewError ? (
                  <div className="error">{cleanupPreviewError}</div>
                ) : cleanupPreviewResponse ? (
                  <>
                    <div className="dataset-cleanup-preview-grid">
                      <div>
                        <span className="meta-label">Rows before</span>
                        <strong>
                          {cleanupPreviewResponse.rows_before.toLocaleString()}
                        </strong>
                      </div>
                      <div>
                        <span className="meta-label">Rows dropped</span>
                        <strong>
                          {cleanupPreviewResponse.rows_to_drop.toLocaleString()}
                        </strong>
                      </div>
                      <div>
                        <span className="meta-label">Rows after</span>
                        <strong>
                          {cleanupPreviewResponse.rows_after.toLocaleString()}
                        </strong>
                      </div>
                      <div>
                        <span className="meta-label">Drop share</span>
                        <strong>
                          {(cleanupPreviewResponse.drop_share * 100).toFixed(1)}%
                        </strong>
                      </div>
                    </div>
                    {cleanupCriteriaForm.mode === "flags" ? (
                      <div className="dataset-cleanup-summary-grid">
                        <div className="dataset-cleanup-summary-card">
                          <span className="meta-label">Dropped buckets</span>
                          <div className="dataset-cleanup-chip-row">
                            {Object.entries(
                              cleanupPreviewResponse.dropped_bucket_counts,
                            ).map(([bucket, count]) => (
                              <span
                                key={bucket}
                                className={`dataset-cleanup-chip quality-${bucket} static`}
                              >
                                {bucket}: {count.toLocaleString()}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    ) : null}
                  </>
                ) : (
                  <div className="dataset-preview-empty">
                    Configure cleanup criteria to preview cleanup impact.
                  </div>
                )}
              </section>

              <section className="dataset-cleanup-section">
                <div className="dataset-cleanup-section-header">
                  <h4>Confirm</h4>
                  <span>Type CLEAN to create the cleaned clone.</span>
                </div>
                <label htmlFor="datasetCleanupConfirmInput">
                  Type <strong>CLEAN</strong> to confirm
                </label>
                <input
                  id="datasetCleanupConfirmInput"
                  className="input"
                  type="text"
                  value={cleanupConfirmText}
                  onChange={(event) => setCleanupConfirmText(event.target.value)}
                  placeholder="CLEAN"
                  autoFocus
                  disabled={cleanupApplyLoading}
                />
              </section>
            </div>
            <div className="dataset-delete-modal-actions">
              <button
                type="button"
                className="button ghost"
                onClick={handleCloseCleanupModal}
                disabled={cleanupApplyLoading}
              >
                Cancel
              </button>
              <button
                type="button"
                className="button danger"
                onClick={handleApplyCleanup}
                disabled={!cleanupCanApply}
              >
                {cleanupApplyLoading ? "Creating…" : "Create cleaned clone"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {deleteTargetRun ? (
        <div
          className="dataset-delete-modal-overlay"
          onClick={() => {
            if (deleteLoadingRun === deleteTargetRun.id) return;
            setDeleteConfirmRun(null);
            setDeleteConfirmText("");
          }}
        >
          <div
            className="dataset-delete-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="dataset-delete-modal-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="dataset-delete-modal-header">
              <h3 id="dataset-delete-modal-title">Delete dataset directory</h3>
              <p>
                This permanently deletes{" "}
                <span className="dataset-delete-modal-code">
                  {deleteTargetRunName}
                </span>{" "}
                and all exported files.
              </p>
            </div>
            <div className="dataset-delete-modal-body">
              <label htmlFor="datasetDeleteConfirmInput">
                Type <strong>DELETE</strong> to confirm
              </label>
              <input
                id="datasetDeleteConfirmInput"
                className="input"
                type="text"
                value={deleteConfirmText}
                onChange={(event) => setDeleteConfirmText(event.target.value)}
                placeholder="DELETE"
                autoFocus
                disabled={deleteLoadingRun === deleteTargetRun.id}
              />
            </div>
            <div className="dataset-delete-modal-actions">
              <button
                type="button"
                className="button ghost"
                onClick={() => {
                  setDeleteConfirmRun(null);
                  setDeleteConfirmText("");
                }}
                disabled={deleteLoadingRun === deleteTargetRun.id}
              >
                Cancel
              </button>
              <button
                type="button"
                className="button ghost danger"
                onClick={() =>
                  handleDeleteRun(deleteTargetRun.id, deleteTargetRun.run_dir)
                }
                disabled={
                  deleteConfirmText !== "DELETE" ||
                  deleteLoadingRun === deleteTargetRun.id
                }
              >
                {deleteLoadingRun === deleteTargetRun.id
                  ? "Deleting…"
                  : "Delete permanently"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
