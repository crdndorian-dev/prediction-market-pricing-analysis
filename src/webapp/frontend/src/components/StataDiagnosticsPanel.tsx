import { useMemo, useState, type ReactNode } from "react";

import type {
  CalibrationBinRow,
  CoefficientDiagnosticRow,
  DiagnosticScalar,
  DiagnosticSectionRow,
  MarginalEffectDiagnosticRow,
  ProductionCoefficientDiagnosticRow,
  StataDiagnosticsPayload,
} from "../api/calibrateModels";

const CHART_SIZE = 560;
const GENERAL_DIAGNOSTIC_NOTE_PREFIXES = [
  "inferential rows come from a shadow statsmodels binomial glm",
  "predictive, probabilistic, and calibration rows use production sklearn probabilities.",
  "displayed coefficient and marginal-effect rows describe the base logistic layer",
] as const;

type TableFilter = "core" | "ticker" | "all";
type CalibrationSeriesKey = "model" | "baseline";

type DiagnosticsColumn<Row> = {
  key: string;
  label: string;
  colClass: string;
  align?: "left" | "right";
  cellClassName?: string;
  render: (row: Row) => ReactNode;
};

const normalizeNote = (value: string): string =>
  value.trim().replace(/\s+/g, " ").toLowerCase();

const isGeneralDiagnosticNote = (note: string): boolean => {
  const normalized = normalizeNote(note);
  return GENERAL_DIAGNOSTIC_NOTE_PREFIXES.some((prefix) => normalized.startsWith(prefix));
};

const formatScalar = (
  value: DiagnosticScalar | undefined,
  format: DiagnosticSectionRow["format"],
): string => {
  if (value == null) return "--";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "string") return value;
  if (!Number.isFinite(value)) return "--";

  if (format === "count") {
    return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2);
  }
  if (format === "percent") {
    return `${(value * 100).toFixed(2)}%`;
  }
  if (format === "pvalue") {
    return value < 0.0001 ? "<0.0001" : value.toFixed(4);
  }
  if (Number.isInteger(value) && Math.abs(value) >= 100) {
    return value.toLocaleString();
  }
  return value.toFixed(4);
};

const formatMetric = (value?: number | null): string =>
  value == null || Number.isNaN(value) ? "--" : value.toFixed(4);

const formatCount = (value?: number | null): string =>
  value == null || Number.isNaN(value) ? "--" : value.toLocaleString();

const normalizeCalibrationRows = (
  rows: CalibrationBinRow[],
): Array<CalibrationBinRow & { series: CalibrationSeriesKey }> =>
  rows
    .map((row) => ({
      ...row,
      series: (row.series === "baseline" ? "baseline" : "model") as CalibrationSeriesKey,
    }))
    .filter((row) => Number.isFinite(row.pred_mean) && Number.isFinite(row.obs_rate));

const rowStatusClass = (status?: string): string => {
  if (status === "ok") return "ok";
  if (status === "warn") return "warn";
  if (status === "bad") return "bad";
  return "na";
};

const coefficientRowsForFilter = <
  T extends CoefficientDiagnosticRow | MarginalEffectDiagnosticRow | ProductionCoefficientDiagnosticRow,
>(
  rows: T[],
  filter: TableFilter,
): T[] => {
  if (filter === "all") return rows;
  if (filter === "ticker") return rows.filter((row) => row.feature_group === "ticker");
  return rows.filter((row) => row.feature_group !== "ticker");
};

const alignClass = (align: DiagnosticsColumn<unknown>["align"] = "left"): string =>
  align === "right" ? "stata-cell-right" : "stata-cell-left";

const FilterTabs = ({
  title,
  filter,
  onChange,
}: {
  title: string;
  filter: TableFilter;
  onChange: (value: TableFilter) => void;
}) => (
  <div className="stata-filter-group" role="tablist" aria-label={`${title} filter`}>
    {(["core", "ticker", "all"] as const).map((option) => (
      <button
        key={`${title}-${option}`}
        type="button"
        className={`stata-filter-chip ${filter === option ? "active" : ""}`}
        onClick={() => onChange(option)}
      >
        {option}
      </button>
    ))}
  </div>
);

const DiagnosticsTable = <Row,>({
  tableClass,
  columns,
  rows,
  rowKey,
}: {
  tableClass: string;
  columns: DiagnosticsColumn<Row>[];
  rows: Row[];
  rowKey: (row: Row, index: number) => string;
}) => (
  <div className="stata-table-wrap">
    <table className={`stata-diagnostics-table ${tableClass}`}>
      <colgroup>
        {columns.map((column) => (
          <col key={`${tableClass}-${column.key}`} className={column.colClass} />
        ))}
      </colgroup>
      <thead>
        <tr>
          {columns.map((column) => (
            <th
              key={`${tableClass}-${column.key}-header`}
              scope="col"
              className={alignClass(column.align)}
            >
              {column.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, index) => (
          <tr key={rowKey(row, index)}>
            {columns.map((column) => (
              <td
                key={`${rowKey(row, index)}-${column.key}`}
                className={`${alignClass(column.align)}${column.cellClassName ? ` ${column.cellClassName}` : ""}`}
              >
                {column.render(row)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

const DiagnosticsTableCard = ({
  title,
  subtitle,
  controls,
  children,
}: {
  title: string;
  subtitle?: string | null;
  controls?: ReactNode;
  children: ReactNode;
}) => (
  <div className="stata-detail-card">
    <div className="stata-subsection-header">
      <div className="stata-subsection-title-block">
        <span className="meta-label">{title}</span>
        {subtitle ? <span className="stata-subsection-copy">{subtitle}</span> : null}
      </div>
      {controls}
    </div>
    {children}
  </div>
);

export const CalibrationCurveCard = ({
  rows,
  split,
  baselineLabel,
  title = "Calibration curve",
  subtitle,
}: {
  rows: CalibrationBinRow[];
  split: string;
  baselineLabel: string;
  title?: string;
  subtitle?: string;
}) => {
  const normalizedRows = useMemo(() => normalizeCalibrationRows(rows), [rows]);
  const modelRows = useMemo(
    () => normalizedRows.filter((row) => row.series === "model").sort((left, right) => left.bin - right.bin),
    [normalizedRows],
  );
  const baselineRows = useMemo(
    () => normalizedRows.filter((row) => row.series === "baseline").sort((left, right) => left.bin - right.bin),
    [normalizedRows],
  );

  if (!normalizedRows.length) return null;

  const width = CHART_SIZE;
  const height = CHART_SIZE;
  const leftPad = 76;
  const rightPad = 32;
  const topPad = 32;
  const bottomPad = 76;
  const plotWidth = width - leftPad - rightPad;
  const plotHeight = height - topPad - bottomPad;
  const scaleX = (v: number) => leftPad + v * plotWidth;
  const scaleY = (v: number) => topPad + (1 - v) * plotHeight;
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const buildPath = (seriesRows: Array<CalibrationBinRow & { series: CalibrationSeriesKey }>) => seriesRows
    .map((row, idx) => `${idx === 0 ? "M" : "L"} ${scaleX(row.pred_mean)} ${scaleY(row.obs_rate)}`)
    .join(" ");
  const modelPath = buildPath(modelRows);
  const baselinePath = buildPath(baselineRows);

  return (
    <div className="stata-reliability-card">
      <div className="stata-subsection-header artifact-header-row">
        <div className="stata-subsection-title-block">
          <span className="meta-label">{title}</span>
          <span className="stata-subsection-copy">
            {subtitle ?? `${split.toUpperCase()} equal-mass bins`}
          </span>
        </div>
        <div className="artifact-legend">
          <span className="artifact-legend-item">
            <span className="artifact-swatch artifact-swatch-diagonal" />
            Perfect calibration
          </span>
          <span className="artifact-legend-item">
            <span className="artifact-swatch artifact-swatch-model" />
            Production model
          </span>
          {baselineRows.length ? (
            <span className="artifact-legend-item">
              <span className="artifact-swatch artifact-swatch-baseline" />
              {baselineLabel}
            </span>
          ) : null}
        </div>
      </div>
      <div className="artifact-chart-panel stata-calibration-chart-panel">
        <svg viewBox={`0 0 ${width} ${height}`} className="artifact-chart stata-calibration-chart">
          <rect x={0} y={0} width={width} height={height} className="chart-frame" />
          {ticks.map((tick) => {
            const x = scaleX(tick);
            const y = scaleY(tick);
            return (
              <g key={`stata-reliability-${tick}`}>
                <line x1={x} x2={x} y1={topPad} y2={height - bottomPad} className="artifact-grid-line" />
                <line x1={leftPad} x2={width - rightPad} y1={y} y2={y} className="artifact-grid-line" />
                <text x={x} y={height - bottomPad + 22} textAnchor="middle" className="artifact-tick-label">
                  {tick.toFixed(2)}
                </text>
                <text x={leftPad - 10} y={y + 4} textAnchor="end" className="artifact-tick-label">
                  {tick.toFixed(2)}
                </text>
              </g>
            );
          })}
          <line x1={leftPad} x2={width - rightPad} y1={height - bottomPad} y2={height - bottomPad} className="artifact-axis-line" />
          <line x1={leftPad} x2={leftPad} y1={topPad} y2={height - bottomPad} className="artifact-axis-line" />
          <line x1={leftPad} y1={height - bottomPad} x2={width - rightPad} y2={topPad} className="chart-midline" />
          {baselinePath ? <path d={baselinePath} className="chart-line chart-line-baseline" /> : null}
          {modelPath ? <path d={modelPath} className="chart-line chart-line-model" /> : null}
          {baselineRows.map((row) => (
            <circle
              key={`stata-calibration-baseline-bin-${row.bin}`}
              cx={scaleX(row.pred_mean)}
              cy={scaleY(row.obs_rate)}
              r={4}
              className="artifact-chart-point artifact-chart-point-baseline"
            />
          ))}
          {modelRows.map((row) => (
            <circle
              key={`stata-calibration-model-bin-${row.bin}`}
              cx={scaleX(row.pred_mean)}
              cy={scaleY(row.obs_rate)}
              r={4}
              className="artifact-chart-point artifact-chart-point-model"
            />
          ))}
          <text x={(leftPad + width - rightPad) / 2} y={height - 14} textAnchor="middle" className="artifact-axis-title">
            Mean predicted probability
          </text>
          <text
            x={22}
            y={topPad + plotHeight / 2}
            transform={`rotate(-90 22 ${topPad + plotHeight / 2})`}
            textAnchor="middle"
            className="artifact-axis-title"
          >
            Observed event rate
          </text>
        </svg>
      </div>
    </div>
  );
};

const ProductionCoefficientTable = ({
  rows,
  subtitle,
}: {
  rows: ProductionCoefficientDiagnosticRow[];
  subtitle?: string | null;
}) => {
  const [filter, setFilter] = useState<TableFilter>("core");
  const filtered = useMemo(() => coefficientRowsForFilter(rows, filter), [rows, filter]);

  if (!rows.length) return null;

  const columns: DiagnosticsColumn<ProductionCoefficientDiagnosticRow>[] = [
    {
      key: "variable",
      label: "Variable",
      colClass: "stata-col-variable",
      render: (row) => row.display_name,
    },
    {
      key: "coef",
      label: "Coef.",
      colClass: "stata-col-number-sm",
      align: "right",
      render: (row) => formatMetric(row.coefficient),
    },
    {
      key: "or",
      label: "OR",
      colClass: "stata-col-number-sm",
      align: "right",
      render: (row) => formatMetric(row.odds_ratio),
    },
    {
      key: "note",
      label: "Note",
      colClass: "stata-col-note",
      cellClassName: "stata-note-cell",
      render: (row) => row.note ?? "--",
    },
  ];

  return (
    <DiagnosticsTableCard
      title="Production coefficient detail"
      subtitle={subtitle}
      controls={<FilterTabs title="Production coefficient detail" filter={filter} onChange={setFilter} />}
    >
      <DiagnosticsTable
        tableClass="stata-production-table"
        columns={columns}
        rows={filtered}
        rowKey={(row) => `production-${row.feature_name}`}
      />
    </DiagnosticsTableCard>
  );
};

const ShadowCoefficientTable = ({
  rows,
  subtitle,
}: {
  rows: CoefficientDiagnosticRow[];
  subtitle?: string | null;
}) => {
  const [filter, setFilter] = useState<TableFilter>("core");
  const filtered = useMemo(() => coefficientRowsForFilter(rows, filter), [rows, filter]);

  if (!rows.length) return null;

  const columns: DiagnosticsColumn<CoefficientDiagnosticRow>[] = [
    {
      key: "variable",
      label: "Variable",
      colClass: "stata-col-variable",
      render: (row) => row.display_name,
    },
    {
      key: "coef",
      label: "Coef.",
      colClass: "stata-col-number-sm",
      align: "right",
      render: (row) => formatMetric(row.coefficient),
    },
    {
      key: "std_err",
      label: "Std. Err.",
      colClass: "stata-col-number-sm",
      align: "right",
      render: (row) => formatMetric(row.std_error),
    },
    {
      key: "z",
      label: "z",
      colClass: "stata-col-number-xs",
      align: "right",
      render: (row) => formatMetric(row.z_stat),
    },
    {
      key: "p_value",
      label: "P>|z|",
      colClass: "stata-col-number-sm",
      align: "right",
      render: (row) => (row.p_value == null ? "--" : formatScalar(row.p_value, "pvalue")),
    },
    {
      key: "ci",
      label: "95% CI",
      colClass: "stata-col-range",
      align: "right",
      render: (row) => (
        row.ci_low == null || row.ci_high == null ? "--" : `[${formatMetric(row.ci_low)}, ${formatMetric(row.ci_high)}]`
      ),
    },
    {
      key: "or",
      label: "OR",
      colClass: "stata-col-number-sm",
      align: "right",
      render: (row) => formatMetric(row.odds_ratio),
    },
    {
      key: "or_ci",
      label: "95% OR CI",
      colClass: "stata-col-range",
      align: "right",
      render: (row) => (
        row.odds_ratio_ci_low == null || row.odds_ratio_ci_high == null
          ? "--"
          : `[${formatMetric(row.odds_ratio_ci_low)}, ${formatMetric(row.odds_ratio_ci_high)}]`
      ),
    },
    {
      key: "note",
      label: "Note",
      colClass: "stata-col-note",
      cellClassName: "stata-note-cell",
      render: (row) => row.note ?? "--",
    },
  ];

  return (
    <DiagnosticsTableCard
      title="Shadow inference detail"
      subtitle={subtitle}
      controls={<FilterTabs title="Shadow inference detail" filter={filter} onChange={setFilter} />}
    >
      <DiagnosticsTable
        tableClass="stata-shadow-coefficient-table"
        columns={columns}
        rows={filtered}
        rowKey={(row) => `shadow-${row.feature_name}`}
      />
    </DiagnosticsTableCard>
  );
};

const MarginalEffectsTable = ({
  rows,
  subtitle,
}: {
  rows: MarginalEffectDiagnosticRow[];
  subtitle?: string | null;
}) => {
  const [filter, setFilter] = useState<TableFilter>("core");
  const filtered = useMemo(() => coefficientRowsForFilter(rows, filter), [rows, filter]);

  if (!rows.length) return null;

  const columns: DiagnosticsColumn<MarginalEffectDiagnosticRow>[] = [
    {
      key: "variable",
      label: "Variable",
      colClass: "stata-col-variable",
      render: (row) => row.display_name,
    },
    {
      key: "ame",
      label: "AME",
      colClass: "stata-col-number-sm",
      align: "right",
      render: (row) => formatMetric(row.ame),
    },
    {
      key: "std_err",
      label: "Std. Err.",
      colClass: "stata-col-number-sm",
      align: "right",
      render: (row) => formatMetric(row.std_error),
    },
    {
      key: "z",
      label: "z",
      colClass: "stata-col-number-xs",
      align: "right",
      render: (row) => formatMetric(row.z_stat),
    },
    {
      key: "p_value",
      label: "P>|z|",
      colClass: "stata-col-number-sm",
      align: "right",
      render: (row) => (row.p_value == null ? "--" : formatScalar(row.p_value, "pvalue")),
    },
    {
      key: "ci",
      label: "95% CI",
      colClass: "stata-col-range",
      align: "right",
      render: (row) => (
        row.ci_low == null || row.ci_high == null ? "--" : `[${formatMetric(row.ci_low)}, ${formatMetric(row.ci_high)}]`
      ),
    },
    {
      key: "note",
      label: "Note",
      colClass: "stata-col-note",
      cellClassName: "stata-note-cell",
      render: (row) => row.note ?? "--",
    },
  ];

  return (
    <DiagnosticsTableCard
      title="Shadow marginal effects"
      subtitle={subtitle}
      controls={<FilterTabs title="Shadow marginal effects" filter={filter} onChange={setFilter} />}
    >
      <DiagnosticsTable
        tableClass="stata-marginal-effects-table"
        columns={columns}
        rows={filtered}
        rowKey={(row) => `marginal-${row.feature_name}`}
      />
    </DiagnosticsTableCard>
  );
};

export const isStataDiagnosticsPayload = (value: unknown): value is StataDiagnosticsPayload => {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return typeof candidate.schema_version === "number" && Array.isArray(candidate.sections);
};

export function StataDiagnosticsPanel({
  diagnostics,
  hideGeneralNotes = true,
  showHeaderStrip = true,
  showCalibrationChart = true,
  showProductionCoefficientTable = true,
}: {
  diagnostics: StataDiagnosticsPayload;
  hideGeneralNotes?: boolean;
  showHeaderStrip?: boolean;
  showCalibrationChart?: boolean;
  showProductionCoefficientTable?: boolean;
}) {
  const sections = diagnostics.sections.filter((section) => section.rows.length > 0);
  const headerNotes = (diagnostics.header.notes ?? []).filter(
    (note): note is string =>
      typeof note === "string"
      && note.trim().length > 0
      && (!hideGeneralNotes || !isGeneralDiagnosticNote(note)),
  );

  return (
    <div className="stata-diagnostics">
      {showHeaderStrip ? (
        <div className="stata-header-strip">
          <div className="stata-header-item">
            <span className="meta-label">Model</span>
            <strong>{diagnostics.header.model_id}</strong>
          </div>
          <div className="stata-header-item">
            <span className="meta-label">Estimator</span>
            <strong>{diagnostics.header.estimator}</strong>
          </div>
          <div className="stata-header-item">
            <span className="meta-label">Inference</span>
            <strong>{diagnostics.header.inference_basis}</strong>
          </div>
          <div className="stata-header-item">
            <span className="meta-label">Regularization</span>
            <strong>{`penalty=${diagnostics.header.penalty}, C=${formatMetric(diagnostics.header.best_c ?? null)}`}</strong>
          </div>
          <div className="stata-header-item">
            <span className="meta-label">Baseline</span>
            <strong>{diagnostics.header.baseline_name}</strong>
          </div>
          <div className="stata-header-item">
            <span className="meta-label">Threshold</span>
            <strong>
              {diagnostics.header.threshold_operating != null
                ? `0.50 / ${formatMetric(diagnostics.header.threshold_operating)}`
                : formatMetric(diagnostics.header.threshold_default)}
            </strong>
          </div>
          <div className="stata-header-item">
            <span className="meta-label">Train-fit</span>
            <strong>{`${formatCount(diagnostics.header.split_counts.train_fit)} rows / ${formatCount(diagnostics.header.event_counts.train_fit)} events`}</strong>
          </div>
          <div className="stata-header-item">
            <span className="meta-label">Val</span>
            <strong>{`${formatCount(diagnostics.header.split_counts.val)} rows / ${formatCount(diagnostics.header.event_counts.val)} events`}</strong>
          </div>
          <div className="stata-header-item">
            <span className="meta-label">Test</span>
            <strong>{`${formatCount(diagnostics.header.split_counts.test)} rows / ${formatCount(diagnostics.header.event_counts.test)} events`}</strong>
          </div>
          <div className="stata-header-item">
            <span className="meta-label">Features</span>
            <strong>{`${diagnostics.header.feature_counts.numeric} num / ${diagnostics.header.feature_counts.categorical} cat / ${diagnostics.header.feature_counts.transformed} transformed`}</strong>
          </div>
        </div>
      ) : null}

      {headerNotes.length ? (
        <div className="stata-header-notes">
          {headerNotes.map((note, idx) => (
            <div key={`stata-header-note-${idx}`} className="stata-header-note">
              {note}
            </div>
          ))}
        </div>
      ) : null}

      {diagnostics.warnings?.length ? (
        <div className="stata-warning-stack">
          {diagnostics.warnings.map((warning, idx) => (
            <div key={`stata-warning-${idx}`} className="equation-note">
              {warning}
            </div>
          ))}
        </div>
      ) : null}

      <div className="stata-table-wrap stata-table-wrap-static">
        <table className="stata-diagnostics-table stata-summary-table">
          <colgroup>
            <col className="stata-col-family" />
            <col className="stata-col-statistic" />
            <col className="stata-col-number-sm" />
            <col className="stata-col-number-xs" />
            <col className="stata-col-number-xs" />
            <col className="stata-col-note-wide" />
          </colgroup>
          <thead>
            <tr>
              <th scope="col" className="stata-cell-left">Family</th>
              <th scope="col" className="stata-cell-left">Statistic</th>
              <th scope="col" className="stata-cell-right">Train-fit</th>
              <th scope="col" className="stata-cell-right">Val</th>
              <th scope="col" className="stata-cell-right">Test</th>
              <th scope="col" className="stata-cell-left">Note</th>
            </tr>
          </thead>
          <tbody>
            {sections.map((section) =>
              section.rows.map((row, idx) => (
                <tr key={`${section.id}-${row.key}`} className={`stata-summary-row ${rowStatusClass(row.status)}`}>
                  {idx === 0 ? (
                    <th rowSpan={section.rows.length} scope="rowgroup" className="stata-family-cell stata-cell-left">
                      {section.title}
                    </th>
                  ) : null}
                  <td className="stata-cell-left stata-statistic-cell">{row.label}</td>
                  <td className="stata-cell-right">{formatScalar(row.train_fit, row.format)}</td>
                  <td className="stata-cell-right">{formatScalar(row.val, row.format)}</td>
                  <td className="stata-cell-right">{formatScalar(row.test, row.format)}</td>
                  <td className="stata-cell-left stata-note-cell">
                    {[row.note, row.source].filter(Boolean).join(" ")}
                  </td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>

      {showCalibrationChart ? (
        <CalibrationCurveCard
          rows={diagnostics.calibration_curve?.rows ?? []}
          split={diagnostics.calibration_curve?.split ?? "val"}
          baselineLabel={diagnostics.header.baseline_name}
        />
      ) : null}

      {showProductionCoefficientTable && diagnostics.production_coefficient_table?.rows?.length ? (
        <ProductionCoefficientTable
          rows={diagnostics.production_coefficient_table.rows}
          subtitle={diagnostics.production_coefficient_table.subtitle}
        />
      ) : null}

      {diagnostics.coefficient_table?.rows?.length ? (
        <ShadowCoefficientTable
          rows={diagnostics.coefficient_table.rows}
          subtitle={diagnostics.coefficient_table.subtitle}
        />
      ) : null}

      {diagnostics.marginal_effects_table?.rows?.length ? (
        <MarginalEffectsTable
          rows={diagnostics.marginal_effects_table.rows}
          subtitle={diagnostics.marginal_effects_table.subtitle}
        />
      ) : null}
    </div>
  );
}
