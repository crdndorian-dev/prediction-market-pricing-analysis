import { useEffect, useRef } from "react";
import { Link, useLocation } from "react-router-dom";

import PipelineStatusCard from "../components/PipelineStatusCard";
import { useAnyJobRunning } from "../contexts/jobGuard";
import "./DocumentationPage.css";

type DocPageLink = {
  id: string;
  label: string;
  route: string;
};

const docPages: DocPageLink[] = [
  {
    id: "option-chain",
    label: "Option Chain History Builder",
    route: "/option-chain-history-builder",
  },
  {
    id: "polymarket-history",
    label: "Polymarket History Builder",
    route: "/polymarket-history-builder",
  },
  { id: "calibrate", label: "Calibrate", route: "/calibrate" },
  { id: "markets", label: "Markets", route: "/markets" },
  { id: "backtests", label: "Backtests", route: "/backtests" },
];

export const optionChainDoc = (
  <>
    <p>
      The Option Chain History Builder creates the historical option-chain dataset
      used by downstream calibration and research pages. It is a UI wrapper around
      the dataset build script and keeps the run configuration, progress, logs,
      and exported CSVs in one place.
    </p>

    <div className="docs-split">
      <div className="docs-panel">
        <h3>What It Builds</h3>
        <ul>
          <li>
            Historical option-chain snapshots for the tickers and date range you select.
          </li>
          <li>
            Multiple CSV views produced by the build script (training, snapshot,
            pRN view, legacy, optional drops file).
          </li>
          <li>
            Run metadata: stdout/stderr logs, the exact command used, and timing.
          </li>
        </ul>
      </div>
      <div className="docs-panel">
        <h3>Primary Outputs</h3>
        <ul>
          <li>
            Output directory: <code>src/data/raw/option-chain/&lt;dataset-name&gt;</code>
            (dataset name is normalized to kebab-case).
          </li>
          <li>
            <code>training-&lt;name&gt;.csv</code> (always written, used by calibration).
          </li>
          <li>
            <code>snapshot-&lt;name&gt;.csv</code>, <code>prn-view-&lt;name&gt;.csv</code>,
            <code>legacy-&lt;name&gt;.csv</code> (optional views).
          </li>
          <li>
            <code>drops-&lt;name&gt;.csv</code> (optional file with rows dropped by the script).
          </li>
        </ul>
      </div>
    </div>

    <h3>Page Layout</h3>
    <ul>
      <li>
        Page header: <code>Option Chain Dataset Builder</code> with a Documentation button that links back to this section.
      </li>
      <li>
        Run job tab: a Configuration panel (dates, tickers, outputs, advanced settings) with Reset config at the top and Run job at the bottom.
      </li>
      <li>
        Active Run panel replaces Configuration after a run starts and shows status, progress, stop controls, and stdout/stderr log toggles (hidden until selected).
      </li>
      <li>
        Run directory tab: browse, preview, rename, and delete past runs.
      </li>
    </ul>

    <h3>Typical Workflow</h3>
    <ol className="docs-steps">
      <li>Pick a start/end date and select tickers.</li>
      <li>Name the dataset and choose which CSV outputs to write.</li>
      <li>Adjust schedule and advanced settings if needed.</li>
      <li>Click Run job to switch from Configuration to Active Run and monitor progress.</li>
      <li>Select stdout or stderr to inspect logs (click again to hide the log view).</li>
      <li>When the run finishes, click New job to return to Configuration or switch to Run directory to inspect exports.</li>
      <li>Use the training CSV for calibration or export it elsewhere.</li>
    </ol>

    <h3>Run Configuration</h3>
    <p>
      The summary strip updates as you edit the form, showing the date range, ticker
      count, target output path, schedule mode, and planned workload. Start/end
      dates and a dataset name are required to run.
    </p>

    <div className="docs-split">
      <div className="docs-panel">
        <h3>Core Range</h3>
        <ul>
          <li>
            Start date and End date are required. The picker prevents dates before
            <code>2023-06-01</code> and after today, and it blocks end dates earlier
            than the start date.
          </li>
          <li>
            Trading universe chips: <code>AAPL</code>, <code>GOOGL</code>, <code>MSFT</code>,
            <code>META</code>, <code>AMZN</code>, <code>PLTR</code>, <code>NVDA</code>,
            <code>TSLA</code>, <code>NFLX</code>, <code>OPEN</code>. Click to toggle.
          </li>
          <li>
            Add custom tickers using commas or spaces, then click Add or press Enter.
          </li>
          <li>
            Selected tickers appear as chips. Click a chip to remove it.
          </li>
          <li>
            Tickers are normalized to uppercase, de-duplicated, and ordered with
            the core universe first.
          </li>
        </ul>
      </div>
      <div className="docs-panel">
        <h3>Output Targets</h3>
        <ul>
          <li>
            Output directory is fixed to <code>src/data/raw/option-chain</code>.
          </li>
          <li>
            Dataset name is required and becomes the run folder and file suffix
            (normalized to kebab-case for filenames).
          </li>
          <li>
            pRN version is stored with the outputs and forwarded to the script.
          </li>
          <li>
            Outputs to generate:
            <code>training</code> is always on, while <code>snapshot</code>,
            <code>prn-view</code>, <code>legacy</code>, and <code>drops</code>
            can be toggled.
          </li>
        </ul>
      </div>
      <div className="docs-panel">
        <h3>Snapshot Schedule</h3>
        <ul>
          <li>
            Schedule mode decides how the date range is interpreted:
            Weekly (anchored on Mondays) or Expiry range (start/end are expiry dates).
          </li>
          <li>
            Expiry weekdays appears only in expiry range mode (comma-separated, e.g. <code>mon,fri</code>).
          </li>
          <li>
            Observation weekdays are the days to sample the chain (e.g. <code>mon,tue,wed,thu</code>).
          </li>
          <li>
            DTE list overrides observation weekdays. You can enter explicit values
            or ranges (e.g. <code>1,2,3</code> or <code>1-5</code>).
          </li>
          <li>
            DTE min/max/step fields appear when a DTE list is set and are forwarded
            to the script.
          </li>
        </ul>
      </div>
    </div>

    <h3>Advanced Settings</h3>
    <p>
      Each accordion matches an advanced section of the CLI. If you do not expand
      or change these, the defaults are used.
    </p>

    <div className="docs-split">
      <div className="docs-panel">
        <h3>Market Data & Runtime</h3>
        <ul>
          <li>Theta base URL for option chain data (default local Theta endpoint).</li>
          <li>Stock source selector: <code>yfinance</code>, <code>theta</code>, or <code>auto</code>.</li>
          <li>Timeout in seconds for data fetches.</li>
          <li>Risk-free rate (<code>r</code>) used in calculations.</li>
          <li>Threads: parallelism for the build job.</li>
        </ul>
      </div>
      <div className="docs-panel">
        <h3>Band Selection & Training</h3>
        <ul>
          <li>Max abs log-m and max abs log-m cap.</li>
          <li>Band widen step.</li>
          <li>Max band strikes, min band strikes, min band pRN strikes.</li>
          <li>Min and max pRN train bounds.</li>
          <li>Adaptive band toggle (maps to <code>--no-adaptive-band</code> when off).</li>
        </ul>
      </div>
      <div className="docs-panel">
        <h3>Option Chain & Expiry</h3>
        <ul>
          <li>Strike range.</li>
          <li>Retry full chain if the band is thin.</li>
          <li>Saturday expiry fallback.</li>
          <li>Apply split adjustment.</li>
        </ul>
      </div>
    </div>

    <div className="docs-split">
      <div className="docs-panel">
        <h3>Liquidity & Filters</h3>
        <ul>
          <li>Min trade count and min volume filters.</li>
          <li>Min chain used hard threshold.</li>
          <li>Max relative spread median hard cap.</li>
          <li>Prefer bid/ask quotes toggle.</li>
          <li>Hard drop close fallback toggle.</li>
        </ul>
      </div>
      <div className="docs-panel">
        <h3>Dividends, Weights & Volatility</h3>
        <ul>
          <li>Dividend source, lookback days, and default yield.</li>
          <li>RV (realized volatility) lookback days.</li>
          <li>Use forward moneyness toggle.</li>
          <li>Add group weights, ticker weights, and soft quality weighting toggles.</li>
        </ul>
      </div>
      <div className="docs-panel">
        <h3>Cache & Sanity Checks</h3>
        <ul>
          <li>Sanity abs log-m max, sanity K/S min, sanity K/S max.</li>
          <li>Enable cache toggle.</li>
          <li>Sanity report and drop rows failing sanity toggles.</li>
          <li>Verbose skips toggle for detailed drop reasons.</li>
        </ul>
      </div>
    </div>

    <h3>Run Monitoring & Logs</h3>
    <ul>
      <li>
        The run monitor shows planned expiries, tickers, snapshot days per expiry,
        and total planned jobs based on your schedule.
      </li>
      <li>
        Progress includes jobs done/total, percent complete, groups kept, rows
        written, and the last ticker/week/as-of processed.
      </li>
      <li>
        A progress bar updates while running; progress updates are echoed in stdout
        every 100 jobs.
      </li>
      <li>
        Stop run is available only while a job is queued or running.
      </li>
      <li>
        The output summary lists duration, output directory, training dataset path,
        and the drops file path if enabled.
      </li>
      <li>
        Tabs let you switch between stdout and stderr. Failed runs default to stderr.
      </li>
      <li>
        The Command used drawer shows the exact CLI invocation for the last run.
      </li>
    </ul>

    <h3>Dataset Registry</h3>
    <ul>
      <li>
        Lists every run directory under <code>src/data/raw/option-chain</code>,
        with last modified time and CSV counts.
      </li>
      <li>
        Click a run to expand its CSV list. Only one run can be open at a time.
      </li>
      <li>
        Each file row includes size, a Training tag (when applicable), Preview,
        and Open (download/new tab) actions.
      </li>
      <li>
        Preview shows the first or last rows (20/50/100) with column headers and
        total row count when available.
      </li>
      <li>
        Rename dataset updates the run directory name; Delete dataset removes the
        entire run after confirmation.
      </li>
    </ul>

    <h3>CLI Preview</h3>
    <ul>
      <li>
        Mirrors the command that will run, using the same flags as the backend.
      </li>
      <li>
        Optional toggles are represented with <code>--flag</code> or
        <code>--no-flag</code> depending on their state.
      </li>
      <li>
        Use it to reproduce a run in a terminal or share settings with teammates.
      </li>
    </ul>

    <h3>Behavior & Persistence</h3>
    <ul>
      <li>
        Form values are saved locally and restored on reload; Reset returns to
        the defaults.
      </li>
      <li>
        The last job id is remembered so progress can resume after a refresh.
      </li>
      <li>
        A successful run writes the training dataset path into the calibration
        form so the Calibrate page is pre-filled.
      </li>
      <li>
        The Run button is disabled when another pipeline job is running beyond the
        configured concurrency limit.
      </li>
    </ul>

    <h3>Training CSV Feature Reference</h3>
    <p>
      Each row in <code>training-*.csv</code> corresponds to one call strike kept in the
      pRN band for a single ticker + as-of snapshot + expiry. Option chains are fetched
      from Theta Terminal <code>option/history/eod</code> and stock closes are fetched
      via yfinance (when available) or Theta <code>stock/history/eod</code>; dividends
      and splits are pulled via yfinance. Source:
      <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L335</code>,
      <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L327</code>,
      <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L399</code>,
      <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L638</code>,
      <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L418</code>.
    </p>

    <h4>Row Identity & Schedule</h4>
    <div className="docs-table-wrap">
      <table className="docs-table">
        <thead>
          <tr>
            <th>Field</th>
            <th>Meaning</th>
            <th>Fetch / Computation</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>row_id</code></td>
            <td>Stable row identifier used for joins and deduplication.</td>
            <td>
              SHA1 of <code>asof_ts|ticker|expiry_date_used|strike|option_type</code>.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L127</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1375</code>.
            </td>
          </tr>
          <tr>
            <td><code>asof_ts</code></td>
            <td>UTC ISO timestamp for the actual as-of close date used.</td>
            <td>
              Computed from <code>asof_date</code> via <code>iso_ts</code>.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L123</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1106</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1376</code>.
            </td>
          </tr>
          <tr>
            <td><code>option_type</code></td>
            <td>Option right; currently always <code>call</code>.</td>
            <td>
              Hard-coded when emitting rows.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1358</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1377</code>.
            </td>
          </tr>
          <tr>
            <td><code>ticker</code></td>
            <td>Underlying symbol for the row.</td>
            <td>
              Taken from the schedule/ticker list.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1379</code>.
            </td>
          </tr>
          <tr>
            <td><code>week_monday</code></td>
            <td>ISO-week Monday anchor for the schedule.</td>
            <td>
              Derived from the schedule builder.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1380</code>.
            </td>
          </tr>
          <tr>
            <td><code>week_friday</code></td>
            <td>ISO-week Friday anchor (event end / expiry request).</td>
            <td>
              Derived from the schedule builder; used to set the expiry target.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1110</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1381</code>.
            </td>
          </tr>
          <tr>
            <td><code>asof_target</code></td>
            <td>Scheduled snapshot date prior to fallback.</td>
            <td>
              Generated by the schedule builder (weekday or DTE selection).
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1382</code>.
            </td>
          </tr>
          <tr>
            <td><code>asof_date</code></td>
            <td>Actual close date used after forward fallback.</td>
            <td>
              Forward search in the close map (yfinance or Theta <code>stock/history/eod</code>).
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L679</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1070</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1104</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1383</code>.
            </td>
          </tr>
          <tr>
            <td><code>expiry_close_date_used</code></td>
            <td>Actual close date used for the expiry label after backward fallback.</td>
            <td>
              Backward search in the close map (yfinance or Theta <code>stock/history/eod</code>).
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L679</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1087</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1105</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1384</code>.
            </td>
          </tr>
          <tr>
            <td><code>option_expiration_requested</code></td>
            <td>Requested option-chain expiry (defaults to <code>week_friday</code>).</td>
            <td>
              Used to query Theta <code>option/history/eod</code>.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1155</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1387</code>.
            </td>
          </tr>
          <tr>
            <td><code>option_expiration_used</code></td>
            <td>Expiry actually used (Friday or Saturday fallback).</td>
            <td>
              Chooses Friday then optional Saturday fallback when fetching Theta
              <code>option/history/eod</code>.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1173</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1388</code>.
            </td>
          </tr>
          <tr>
            <td><code>expiry_convention</code></td>
            <td>Marker for which expiry was used: <code>FRI</code> or <code>SAT_FALLBACK</code>.</td>
            <td>
              Set alongside the expiry fetch logic.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1158</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1389</code>.
            </td>
          </tr>
          <tr>
            <td><code>T_days</code></td>
            <td>Calendar days between <code>asof_date</code> and <code>week_friday</code>.</td>
            <td>
              Computed as the schedule horizon in days.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1110</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1392</code>.
            </td>
          </tr>
          <tr>
            <td><code>T_years</code></td>
            <td>Year fraction for the horizon.</td>
            <td>
              <code>T_days / 365.25</code>.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1120</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1393</code>.
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <h4>Rates & Spot/Close Prices</h4>
    <div className="docs-table-wrap">
      <table className="docs-table">
        <thead>
          <tr>
            <th>Field</th>
            <th>Meaning</th>
            <th>Fetch / Computation</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>r</code></td>
            <td>Risk-free rate used for discounting and forward pricing.</td>
            <td>
              Config/CLI value.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L41</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1394</code>.
            </td>
          </tr>
          <tr>
            <td><code>S_asof_close_raw</code></td>
            <td>Raw as-of close price.</td>
            <td>
              From yfinance (Yahoo Finance) or Theta <code>stock/history/eod</code>,
              forward fallback to the next trading day.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L399</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L327</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1070</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1397</code>.
            </td>
          </tr>
          <tr>
            <td><code>S_expiry_close_raw</code></td>
            <td>Raw expiry close used for outcome labels.</td>
            <td>
              From yfinance or Theta <code>stock/history/eod</code>, backward fallback
              from <code>week_friday</code>.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L679</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1087</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1398</code>.
            </td>
          </tr>
          <tr>
            <td><code>S_asof_close_adj</code></td>
            <td>Split-adjusted as-of close price.</td>
            <td>
              Derived from raw closes using yfinance splits when enabled.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L418</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L527</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1073</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1399</code>.
            </td>
          </tr>
          <tr>
            <td><code>S_expiry_close_adj</code></td>
            <td>Split-adjusted expiry close for outcome labels.</td>
            <td>
              Derived from raw closes with the same split adjustment.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L679</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1090</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1400</code>.
            </td>
          </tr>
          <tr>
            <td><code>split_events_in_preload_range</code></td>
            <td>Count of split events observed in the preload window.</td>
            <td>
              Pulled via yfinance splits.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L418</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L533</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1401</code>.
            </td>
          </tr>
          <tr>
            <td><code>split_adjustment_applied</code></td>
            <td>Whether split adjustment was applied when producing adjusted closes.</td>
            <td>
              Boolean derived from config.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L87</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L527</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1402</code>.
            </td>
          </tr>
          <tr>
            <td><code>spot_scale_used</code></td>
            <td>Selected spot scale (<code>raw</code> vs <code>split_adj</code>).</td>
            <td>
              Chosen by curve scoring on strike coverage.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1215</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1243</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1412</code>.
            </td>
          </tr>
          <tr>
            <td><code>spot_scale_score_raw</code></td>
            <td>Score for the raw spot scale (more strikes inside band → higher).</td>
            <td>
              Computed by the spot-scale scoring function.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1034</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1244</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1413</code>.
            </td>
          </tr>
          <tr>
            <td><code>spot_scale_score_adj</code></td>
            <td>Score for the split-adjusted spot scale.</td>
            <td>
              Computed by the spot-scale scoring function.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1034</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1245</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1414</code>.
            </td>
          </tr>
          <tr>
            <td><code>S_asof_close</code></td>
            <td>Spot close used in calculations (raw or adjusted).</td>
            <td>
              Selected based on <code>spot_scale_used</code>.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1334</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1417</code>.
            </td>
          </tr>
          <tr>
            <td><code>S_expiry_close</code></td>
            <td>Expiry close used for outcome labels (raw or adjusted).</td>
            <td>
              Selected based on <code>spot_scale_used</code>.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1334</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1418</code>.
            </td>
          </tr>
          <tr>
            <td><code>asof_fallback_days</code></td>
            <td>Days moved forward from <code>asof_target</code> to find a close.</td>
            <td>
              Computed by the forward close fallback.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L679</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1070</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1422</code>.
            </td>
          </tr>
          <tr>
            <td><code>expiry_fallback_days</code></td>
            <td>Days moved backward from <code>week_friday</code> to find a close.</td>
            <td>
              Computed by the backward close fallback.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L679</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1087</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1423</code>.
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <h4>Dividends & Forward</h4>
    <div className="docs-table-wrap">
      <table className="docs-table">
        <thead>
          <tr>
            <th>Field</th>
            <th>Meaning</th>
            <th>Fetch / Computation</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>dividend_source_used</code></td>
            <td>Dividend source actually used (yfinance, default, or none).</td>
            <td>
              Dividends fetched via yfinance; otherwise defaults/none.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L638</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1125</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1405</code>.
            </td>
          </tr>
          <tr>
            <td><code>dividend_lookback_days</code></td>
            <td>Lookback window (days) for dividend aggregation.</td>
            <td>
              Config/CLI value.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L92</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1406</code>.
            </td>
          </tr>
          <tr>
            <td><code>dividend_sum_lookback</code></td>
            <td>Sum of dividends in the lookback window ending at <code>asof_date</code>.</td>
            <td>
              Computed from yfinance dividends over the lookback window.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1131</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1407</code>.
            </td>
          </tr>
          <tr>
            <td><code>dividend_yield_raw</code></td>
            <td>Annualized dividend yield using raw spot close.</td>
            <td>
              Derived from <code>dividend_sum_lookback</code> and raw spot.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1143</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1150</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1408</code>.
            </td>
          </tr>
          <tr>
            <td><code>dividend_yield_adj</code></td>
            <td>Annualized dividend yield using split-adjusted spot close.</td>
            <td>
              Derived from <code>dividend_sum_lookback</code> and adjusted spot.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1143</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1151</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1409</code>.
            </td>
          </tr>
          <tr>
            <td><code>dividend_yield</code></td>
            <td>Yield used for forward pricing and curve calculations.</td>
            <td>
              Selected based on <code>spot_scale_used</code>.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1334</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1419</code>.
            </td>
          </tr>
          <tr>
            <td><code>forward_price</code></td>
            <td>Forward price for moneyness and band selection.</td>
            <td>
              <code>spot * exp((r - q) * T_years)</code> using the chosen scale.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1210</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1346</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1420</code>.
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <h4>Strike, Moneyness & Volatility</h4>
    <div className="docs-table-wrap">
      <table className="docs-table">
        <thead>
          <tr>
            <th>Field</th>
            <th>Meaning</th>
            <th>Fetch / Computation</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>K</code></td>
            <td>Strike price for the call option.</td>
            <td>
              From the selected strikes in the Theta option-chain curve.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1357</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1426</code>.
            </td>
          </tr>
          <tr>
            <td><code>log_m</code></td>
            <td>Log moneyness <code>log(K / S_asof_close)</code>.</td>
            <td>
              Computed from the chosen spot close.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1427</code>.
            </td>
          </tr>
          <tr>
            <td><code>abs_log_m</code></td>
            <td>Absolute log moneyness <code>|log_m|</code>.</td>
            <td>
              Computed from <code>log_m</code>.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1428</code>.
            </td>
          </tr>
          <tr>
            <td><code>log_m_fwd</code></td>
            <td>Forward moneyness <code>log(K / forward_price)</code> when available.</td>
            <td>
              Computed when a finite forward price exists.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1429</code>.
            </td>
          </tr>
          <tr>
            <td><code>abs_log_m_fwd</code></td>
            <td>Absolute forward moneyness <code>|log_m_fwd|</code>.</td>
            <td>
              Computed from <code>log_m_fwd</code>.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1430</code>.
            </td>
          </tr>
          <tr>
            <td><code>rv20</code></td>
            <td>Annualized realized volatility proxy from recent closes.</td>
            <td>
              Computed from the latest lookback window of closes.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L701</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1122</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1334</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1433</code>.
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <h4>Risk-Neutral Probabilities & Labels</h4>
    <div className="docs-table-wrap">
      <table className="docs-table">
        <thead>
          <tr>
            <th>Field</th>
            <th>Meaning</th>
            <th>Fetch / Computation</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>pRN</code></td>
            <td>Risk-neutral probability for the strike.</td>
            <td>
              Computed from Theta <code>option/history/eod</code> call-curve slopes
              (Breeden-Litzenberger) with monotone adjustments.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L335</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L772</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L876</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1436</code>.
            </td>
          </tr>
          <tr>
            <td><code>qRN</code></td>
            <td>Complement of pRN.</td>
            <td>
              <code>1 - pRN</code>.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1437</code>.
            </td>
          </tr>
          <tr>
            <td><code>pRN_raw</code></td>
            <td>Raw interpolation to strikes before isotonic adjustments.</td>
            <td>
              Computed from call-curve slopes without the monotone pass.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L921</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1438</code>.
            </td>
          </tr>
          <tr>
            <td><code>qRN_raw</code></td>
            <td>Complement of <code>pRN_raw</code>.</td>
            <td>
              <code>1 - pRN_raw</code>.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1439</code>.
            </td>
          </tr>
          <tr>
            <td><code>outcome_ST_gt_K</code></td>
            <td>Realized outcome label (1 if expiry close is above strike).</td>
            <td>
              Computed from <code>S_expiry_close</code> and <code>K</code>.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1334</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1441</code>.
            </td>
          </tr>
          <tr>
            <td><code>prn_monotone_adj_intervals</code></td>
            <td>Whether isotonic adjustment was needed on interval pRN values.</td>
            <td>
              Computed during pRN curve cleaning.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L925</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1463</code>.
            </td>
          </tr>
          <tr>
            <td><code>prn_monotone_adj_targets</code></td>
            <td>Whether isotonic adjustment was needed on target strike values.</td>
            <td>
              Computed during pRN curve cleaning.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L934</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1464</code>.
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <h4>Band & Curve Diagnostics</h4>
    <div className="docs-table-wrap">
      <table className="docs-table">
        <thead>
          <tr>
            <th>Field</th>
            <th>Meaning</th>
            <th>Fetch / Computation</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>max_abs_logm_start</code></td>
            <td>Starting abs log-m band.</td>
            <td>
              Config/CLI value.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L52</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1445</code>.
            </td>
          </tr>
          <tr>
            <td><code>max_abs_logm_cap</code></td>
            <td>Maximum abs log-m cap for adaptive widening.</td>
            <td>
              Config/CLI value.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L53</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1446</code>.
            </td>
          </tr>
          <tr>
            <td><code>used_max_abs_logm</code></td>
            <td>Abs log-m band actually used after adaptive widening.</td>
            <td>
              Computed by <code>pick_band_strikes</code>.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L965</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1284</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1447</code>.
            </td>
          </tr>
          <tr>
            <td><code>n_band_raw</code></td>
            <td>Count of strikes inside the abs log-m band.</td>
            <td>
              Computed by band selection.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1287</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1448</code>.
            </td>
          </tr>
          <tr>
            <td><code>n_band_inside</code></td>
            <td>Count of strikes inside the band after min/max strike filtering.</td>
            <td>
              Computed by band selection.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1288</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1449</code>.
            </td>
          </tr>
          <tr>
            <td><code>moneyness_ref</code></td>
            <td>Moneyness reference used for band selection (<code>spot</code> or <code>forward</code>).</td>
            <td>
              Derived from whether forward moneyness is enabled and finite.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1282</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1450</code>.
            </td>
          </tr>
          <tr>
            <td><code>moneyness_ref_price</code></td>
            <td>Spot or forward price used as the moneyness reference.</td>
            <td>
              Derived from spot/forward selection.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1282</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1451</code>.
            </td>
          </tr>
          <tr>
            <td><code>calls_k_min</code></td>
            <td>Minimum strike in the call curve after cleaning.</td>
            <td>
              Derived from the cleaned call curve.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1280</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1452</code>.
            </td>
          </tr>
          <tr>
            <td><code>calls_k_max</code></td>
            <td>Maximum strike in the call curve after cleaning.</td>
            <td>
              Derived from the cleaned call curve.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1281</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1453</code>.
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <h4>Option Chain Quality Diagnostics</h4>
    <div className="docs-table-wrap">
      <table className="docs-table">
        <thead>
          <tr>
            <th>Field</th>
            <th>Meaning</th>
            <th>Fetch / Computation</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>theta_quote_source</code></td>
            <td>Quote source used to build call mids.</td>
            <td>
              Chosen between bid/ask mid vs close fallback from Theta chain.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L824</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L832</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1456</code>.
            </td>
          </tr>
          <tr>
            <td><code>n_chain_raw</code></td>
            <td>Number of option records in the raw chain.</td>
            <td>
              Count of rows from Theta <code>option/history/eod</code>.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L781</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1457</code>.
            </td>
          </tr>
          <tr>
            <td><code>n_chain_used</code></td>
            <td>Number of option records used after filters and cleaning.</td>
            <td>
              Count after liquidity/intrinsic/insane filters.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L856</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1458</code>.
            </td>
          </tr>
          <tr>
            <td><code>rel_spread_median</code></td>
            <td>Median relative spread of bid/ask mids when available.</td>
            <td>
              Computed from bid/ask quotes in the chain.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L859</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1459</code>.
            </td>
          </tr>
          <tr>
            <td><code>dropped_liquidity</code></td>
            <td>Count dropped due to min trade count or min volume filters.</td>
            <td>
              Computed during call-curve build.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L805</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L810</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1460</code>.
            </td>
          </tr>
          <tr>
            <td><code>dropped_intrinsic</code></td>
            <td>Count dropped for pricing below intrinsic bounds.</td>
            <td>
              Computed during call-curve build.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L842</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L850</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1461</code>.
            </td>
          </tr>
          <tr>
            <td><code>dropped_insane</code></td>
            <td>Count dropped for pricing above the insane price multiple cap.</td>
            <td>
              Computed during call-curve build.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L852</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L854</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1462</code>.
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <h4>Grouping & Strike Spacing</h4>
    <div className="docs-table-wrap">
      <table className="docs-table">
        <thead>
          <tr>
            <th>Field</th>
            <th>Meaning</th>
            <th>Fetch / Computation</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>group_id</code></td>
            <td>Group key <code>ticker|asof_date|week_friday</code>.</td>
            <td>
              Computed per snapshot group.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1478</code>.
            </td>
          </tr>
          <tr>
            <td><code>median_dK</code></td>
            <td>Median strike spacing within the group.</td>
            <td>
              Computed from strike distances in the group.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L996</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1484</code>.
            </td>
          </tr>
          <tr>
            <td><code>min_dK</code></td>
            <td>Minimum strike spacing within the group.</td>
            <td>
              Computed from strike distances in the group.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L996</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1485</code>.
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <h4>Weights</h4>
    <div className="docs-table-wrap">
      <table className="docs-table">
        <thead>
          <tr>
            <th>Field</th>
            <th>Meaning</th>
            <th>Fetch / Computation</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>quality_weight</code></td>
            <td>Soft quality weight based on quote source, spreads, and pRN adjustments.</td>
            <td>
              Computed and clipped to [0.01, 1].
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1011</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L1487</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L2351</code>.
            </td>
          </tr>
          <tr>
            <td><code>group_size</code></td>
            <td>Number of strikes in the group (when group weights enabled).</td>
            <td>
              Computed from <code>group_id</code> counts.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L2333</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L2335</code>.
            </td>
          </tr>
          <tr>
            <td><code>group_weight</code></td>
            <td>Per-row group weight (<code>1 / group_size</code> when enabled).</td>
            <td>
              Computed from group sizes; defaults to 1 if disabled.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L2333</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L2339</code>.
            </td>
          </tr>
          <tr>
            <td><code>ticker_group_count</code></td>
            <td>Number of groups for the ticker (when ticker weights enabled).</td>
            <td>
              Computed by counting groups per ticker.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L2342</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L2344</code>.
            </td>
          </tr>
          <tr>
            <td><code>ticker_weight</code></td>
            <td>Per-row ticker weight (<code>1 / ticker_group_count</code> when enabled).</td>
            <td>
              Computed from ticker group counts; defaults to 1 if disabled.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L2343</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L2348</code>.
            </td>
          </tr>
          <tr>
            <td><code>sample_weight_final</code></td>
            <td>Final sample weight.</td>
            <td>
              <code>group_weight * ticker_weight * quality_weight</code>.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L2351</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L2353</code>.
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <h4>Training View Provenance</h4>
    <div className="docs-table-wrap">
      <table className="docs-table">
        <thead>
          <tr>
            <th>Field</th>
            <th>Meaning</th>
            <th>Fetch / Computation</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>prn_version</code></td>
            <td>pRN version recorded in the training view.</td>
            <td>
              Passed through from the build configuration.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L2374</code>.
            </td>
          </tr>
          <tr>
            <td><code>prn_config_hash</code></td>
            <td>Hash of pRN configuration settings (or CLI override).</td>
            <td>
              Computed by hashing key config fields (or provided by CLI).
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L139</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L2375</code>.
            </td>
          </tr>
          <tr>
            <td><code>dataset_view</code></td>
            <td>Constant marker set to <code>train_view</code>.</td>
            <td>
              Set when writing the training view.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L2376</code>.
            </td>
          </tr>
          <tr>
            <td><code>build_version</code></td>
            <td>Script filename captured for provenance.</td>
            <td>
              Set to the build script filename.
              Source: <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L26</code>,
              <code>src/scripts/01-option-chain-build-historic-dataset-v1.0.py#L2377</code>.
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </>
);

const simplifyFetchComputationText = (raw: string): string => {
  let text = raw.replace(/\s+/g, " ").trim();

  // Drop implementation provenance references (file paths, line refs, etc.).
  text = text.replace(/\s*Source:\s*.*$/i, "");

  // Replace endpoint-level wording with plain-language descriptions.
  text = text.replace(
    /yfinance\s*\(Yahoo Finance\)\s*or\s*Theta\s*stock\/history\/eod/gi,
    "the configured stock price source",
  );
  text = text.replace(
    /yfinance\s*or\s*Theta\s*stock\/history\/eod/gi,
    "the configured stock price source",
  );
  text = text.replace(
    /Theta\s*stock\/history\/eod/gi,
    "the stock price source",
  );
  text = text.replace(
    /Theta\s*option\/history\/eod/gi,
    "the option-chain data source",
  );

  // Make some common implementation shorthand read more clearly.
  text = text.replace(/\bCLI\/config value\b/gi, "Configured value");
  text = text.replace(/\bConfig\/CLI value\b/gi, "Configured value");

  // Normalize spacing around punctuation after removals.
  text = text
    .replace(/\s+([,.;:])/g, "$1")
    .replace(/\(\s+/g, "(")
    .replace(/\s+\)/g, ")")
    .trim();

  return text;
};

const escapeHtml = (value: string): string =>
  value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

const escapeRegExp = (value: string): string =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const formatFetchComputationCellHtml = (
  text: string,
  fieldNames: string[],
): string => {
  let escaped = escapeHtml(text);

  const pipeSequencePlaceholders: string[] = [];
  escaped = escaped.replace(
    /\b[A-Za-z_][A-Za-z0-9_]*(?:\|[A-Za-z_][A-Za-z0-9_]*)+\b/g,
    (sequence) => {
      const placeholder = `__DOC_CODE_SEQ_${pipeSequencePlaceholders.length}__`;
      pipeSequencePlaceholders.push(`<code>${sequence}</code>`);
      return placeholder;
    },
  );

  if (fieldNames.length === 0) {
    return escaped.replace(/__DOC_CODE_SEQ_(\d+)__/g, (_match, index) => {
      return pipeSequencePlaceholders[Number(index)] ?? "";
    });
  }

  const sorted = [...fieldNames].sort((a, b) => b.length - a.length);
  const tokenPattern = sorted.map(escapeRegExp).join("|");
  const pattern = new RegExp(
    `(^|[^A-Za-z0-9_])(${tokenPattern})(?=$|[^A-Za-z0-9_])`,
    "g",
  );

  const withFieldCodes = escaped.replace(
    pattern,
    (_match, prefix: string, token: string) => {
      return `${prefix}<code>${token}</code>`;
    },
  );

  return withFieldCodes.replace(/__DOC_CODE_SEQ_(\d+)__/g, (_match, index) => {
    return pipeSequencePlaceholders[Number(index)] ?? "";
  });
};

export function OptionChainDocContent({ className }: { className?: string }) {
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;

    const rows = root.querySelectorAll<HTMLTableRowElement>(
      ".docs-table tbody tr",
    );
    const fieldNames = Array.from(rows)
      .map((row) =>
        row.querySelector<HTMLElement>("td:first-child code")
          ?.textContent?.trim(),
      )
      .filter((value): value is string => Boolean(value));

    rows.forEach((row) => {
      const cells = row.querySelectorAll<HTMLTableCellElement>("td");
      if (cells.length < 3) return;
      const fetchCell = cells[2];
      const simplified = simplifyFetchComputationText(fetchCell.textContent ?? "");
      fetchCell.innerHTML = formatFetchComputationCellHtml(simplified, fieldNames);
    });
  }, []);

  return (
    <div ref={rootRef} className={className}>
      {optionChainDoc}
    </div>
  );
}

const polymarketHistoryDoc = (
  <>
    <p>
      The Polymarket History Builder manages two related workflows: a weekly
      history backfill for closed markets (including CLOB history), and a
      lightweight market-map run that builds the <code>dim_market</code> mapping
      for live markets. It centralizes configuration, live run monitoring, and
      a browseable archive of past runs.
    </p>

    <div className="docs-split">
      <div className="docs-panel">
        <h3>When To Use It</h3>
        <ul>
          <li>
            Weekly history backfill: capture closed markets and historical prices
            over a date range.
          </li>
          <li>
            Market map run: refresh <code>dim_market</code> mappings for live markets.
          </li>
          <li>
            Optional: build decision features by joining Polymarket history with
            an option-chain pRN dataset.
          </li>
        </ul>
      </div>
      <div className="docs-panel">
        <h3>Primary Outputs</h3>
        <ul>
          <li>
            Weekly history run directory under <code>src/data/raw/polymarket/weekly_history</code>
            (plus artifacts recorded in the Past Runs browser).
          </li>
          <li>
            Market map output file path shown in the latest run output panel.
          </li>
          <li>
            Optional decision features output + manifest when feature building is enabled.
          </li>
        </ul>
      </div>
    </div>

    <h3>Page Layout</h3>
    <ul>
      <li>
        Run configuration: choose weekly backfill vs market-map, pick tickers,
        dates, and history options.
      </li>
      <li>
        Latest run output: live status, progress bars, and log output.
      </li>
      <li>
        Past runs browser: inspect, label, pin, activate, and delete run records.
      </li>
    </ul>

    <h3>Typical Workflow</h3>
    <ol className="docs-steps">
      <li>Decide whether you need weekly history or just a market-map refresh.</li>
      <li>Select tickers from the trading universe (only these are allowed).</li>
      <li>For weekly backfill, set start/end dates to auto-generate event URLs.</li>
      <li>Optionally enable subgraph ingestion or feature building.</li>
      <li>Run the pipeline and monitor progress in the output panel.</li>
      <li>Use the Past Runs table to activate or label the run for downstream use.</li>
    </ol>

    <h3>Run Configuration</h3>
    <p>
      The page defaults to weekly history backfill. Uncheck it to run the
      market-map pipeline instead. The Run button is disabled when another
      pipeline job is already running.
    </p>

    <div className="docs-split">
      <div className="docs-panel">
        <h3>Trading Universe</h3>
        <ul>
          <li>
            Allowed tickers are fixed: <code>AAPL</code>, <code>GOOGL</code>,
            <code>MSFT</code>, <code>META</code>, <code>AMZN</code>, <code>PLTR</code>,
            <code>NVDA</code>, <code>TSLA</code>, <code>NFLX</code>, <code>OPEN</code>.
          </li>
          <li>
            Click a ticker chip to toggle it; the selected list is shown beneath.
          </li>
          <li>
            If nothing is selected, the full universe is used automatically.
          </li>
        </ul>
      </div>
      <div className="docs-panel">
        <h3>Market Map Mode</h3>
        <ul>
          <li>
            Enabled when <em>Use weekly history backfill</em> is unchecked.
          </li>
          <li>
            Uses <code>config/polymarket_market_overrides.csv</code> to resolve tickers
            and thresholds.
          </li>
          <li>
            Strict mapping enforcement fails the run if any market lacks a mapping.
          </li>
          <li>
            Output path and row counts appear in the Latest Run Output panel.
          </li>
        </ul>
      </div>
    </div>

    <div className="docs-split">
      <div className="docs-panel">
        <h3>Weekly History Backfill</h3>
        <ul>
          <li>
            Start and end dates (UTC) are required to auto-generate event URLs.
          </li>
          <li>
            Event URLs are generated for every Friday in the range using:
            <code>
              https://polymarket.com/event/&lt;ticker&gt;-above-on-&lt;month&gt;-&lt;day&gt;-&lt;year&gt;
            </code>
          </li>
          <li>
            CLOB fidelity (minutes) controls sampling resolution for market history.
          </li>
          <li>
            Bar frequencies are comma-separated (e.g. <code>1h,1d</code>).
          </li>
          <li>
            A helper panel shows how many URLs and Fridays were generated.
          </li>
        </ul>
      </div>
      <div className="docs-panel">
        <h3>Optional History Enhancements</h3>
        <ul>
          <li>
            Attempt subgraph trade ingest if the subgraph is configured.
          </li>
          <li>
            Build decision features to support model calibration.
          </li>
          <li>
            Feature building requires a pRN dataset directory with a
            <code>training-*.csv</code> file from the Option Chain History Builder.
          </li>
          <li>
            The resolved training CSV path is shown after selection.
          </li>
        </ul>
      </div>
    </div>

    <h3>Run Monitoring & Logs</h3>
    <ul>
      <li>
        Weekly history shows two progress bars: stage 1 for markets/history and
        stage 2 for feature creation (only when enabled).
      </li>
      <li>
        Stage labels update as the pipeline moves through history, features, and finalizing.
      </li>
      <li>
        Progress counts only advance when a market finishes, so they never move backward.
      </li>
      <li>
        Market map runs show a single progress bar (no live telemetry yet).
      </li>
      <li>
        Logs show stdout by default, switching to stderr when errors appear.
      </li>
      <li>
        Stop run is available only for weekly history jobs.
      </li>
    </ul>

    <h3>Past Runs Browser</h3>
    <ul>
      <li>
        Lists all historical backfill runs with status, dates, market counts, and artifacts.
      </li>
      <li>
        Storage badge shows total runs and disk usage when expanded.
      </li>
      <li>
        Click a label cell to rename; changes are saved on blur or Enter.
      </li>
      <li>
        Activate marks a run as the active/default run for downstream workflows.
      </li>
      <li>
        Pin highlights a run without changing the active selection.
      </li>
      <li>
        Delete is disabled for the active run and requires typing <code>DELETE</code>.
      </li>
      <li>
        Error summaries appear inline for failed runs (hover for full text).
      </li>
    </ul>

    <h3>Behavior & Persistence</h3>
    <ul>
      <li>
        Form settings are stored locally and restored on reload.
      </li>
      <li>
        Run output refreshes automatically when a history job completes.
      </li>
      <li>
        The job guard prevents overlapping pipeline runs beyond the configured limit.
      </li>
    </ul>
  </>
);

const calibrateDoc = (
  <>
    <p>
      The Calibrate page trains probabilistic models that translate option‑derived
      risk‑neutral signals (pRN) into realized outcome probabilities. It supports
      a base calibration model, an optional two‑stage Polymarket overlay, and a
      mixed Polymarket + pRN model. Outputs are versioned under
      <code>src/data/models</code> and can be reviewed, compared, and reused in
      downstream workflows.
    </p>

    <div className="docs-split">
      <div className="docs-panel">
        <h3>Primary Inputs</h3>
        <ul>
          <li>
            Option‑chain training dataset (CSV) from the Option Chain History Builder.
          </li>
          <li>
            Optional Polymarket decision features dataset (<code>decision_features.csv</code>
            / <code>.parquet</code>) for mixed or two‑stage modes.
          </li>
          <li>
            Labels are required for any model training. The base model searches
            <code>label</code>, <code>outcome_ST_gt_K</code>, or <code>outcome</code>.
          </li>
        </ul>
      </div>
      <div className="docs-panel">
        <h3>Primary Outputs</h3>
        <ul>
          <li>
            Base calibration artifacts: <code>final_model.joblib</code>,
            <code>metadata.json</code>, <code>metrics.csv</code>.
          </li>
          <li>
            Two‑stage overlay artifacts (when enabled):
            <code>two_stage_model.joblib</code>, <code>two_stage_metrics.csv</code>,
            <code>two_stage_metadata.json</code>.
          </li>
          <li>
            Mixed model artifacts (when enabled) under
            <code>src/data/models/mixed/&lt;run-id&gt;</code>.
          </li>
        </ul>
      </div>
    </div>

    <h3>Run Modes & Layout</h3>
    <ul>
      <li>
        The configuration panel starts with a Manual run vs Auto-calibrate run toggle; only one mode is active at a
        time.
      </li>
      <li>
        Manual mode exposes the full model configuration, feature selection, and CLI preview. Auto-calibrate mode
        focuses on trial search settings plus regime/foundation/bootstrapping controls.
      </li>
      <li>
        Latest run output is stacked directly beneath the configuration panel for quick feedback.
      </li>
    </ul>

    <h3>Base Calibration Model (Stage A)</h3>
    <p>
      The base model is a logistic regression trained on the option‑chain dataset.
      It learns a mapping from pRN features to realized outcomes.
    </p>
    <ul>
      <li>
        Features are chosen from the training CSV (pRN, moneyness, volatility,
        time‑to‑expiry, and related engineered features). The UI lets you toggle
        additional interactions such as <code>x_logit_prn</code> with
        <code>T_days</code>, <code>rv20</code>, and log‑moneyness.
      </li>
      <li>
        Preprocessing: median imputation and standardization for numeric
        features; optional categorical handling (e.g. ticker intercepts).
      </li>
      <li>
        Regularization: a grid of <code>C</code> values is evaluated on a
        time‑ordered validation split (last 20% of train). Best log‑loss wins.
      </li>
      <li>
        Time split: the default uses <code>week_friday</code> to hold out the
        last N weeks; if unavailable, an 80/20 time split is used.
      </li>
      <li>
        Optional Platt scaling is applied when <code>calibrate=platt</code> and
        the validation set is large enough.
      </li>
      <li>
        Ticker intercepts can be disabled, enabled for all tickers, or enabled
        for non‑foundation tickers (foundation tickers are grouped).
      </li>
    </ul>

    <h3>Two‑Stage Polymarket Overlay (Stage B)</h3>
    <p>
      When enabled, the overlay trains a second logistic model that blends the
      base prediction with Polymarket signals. Stage B only runs where
      Polymarket features exist; otherwise the system falls back to Stage A.
    </p>
    <ul>
      <li>
        Merge keys are strict: <code>ticker</code>, <code>threshold</code>,
        <code>expiry_date</code>, and <code>snapshot_date</code>.
      </li>
      <li>
        Stage A is re‑trained on pre‑overlap data to avoid leakage. Stage B is
        trained only on rows where Polymarket probabilities are present.
      </li>
      <li>
        Stage B features include <code>p_base</code> plus Polymarket signals
        (e.g. <code>pm_mid</code>, spread, liquidity, momentum, volatility, and
        time‑to‑resolution). <code>T_days</code> and categorical features
        (ticker, snapshot day‑of‑week) are added when available.
      </li>
      <li>
        Prediction logic: <code>p_final = p_base</code> when PM is missing;
        otherwise <code>p_final = sigmoid(γ₀ + γᵀ·features)</code>.
      </li>
      <li>
        Metrics compare Stage A, the PM baseline, and the two‑stage output on
        the same test split. Edge statistics use <code>edge = p_final − pm_mid</code>.
      </li>
    </ul>

    <h3>Mixed Model (PM + pRN Blend)</h3>
    <p>
      Mixed mode trains a ridge regression on the decision‑features dataset to
      combine Polymarket and pRN information directly.
    </p>
    <ul>
      <li>
        Model types: <code>residual</code> learns <code>y − pm</code>, and
        <code>blend</code> learns a weight that mixes PM and pRN probabilities.
      </li>
      <li>
        The default feature set is numeric PM + pRN columns found in the
        decision‑features dataset (time columns are excluded).
      </li>
      <li>
        Splits can be a single train/test split or walk‑forward windows with an
        embargo and min time‑to‑resolution filter.
      </li>
    </ul>

    <h3>Economics Intuition</h3>
    <ul>
      <li>
        pRN probabilities are risk‑neutral and can be biased by risk premia or
        microstructure effects; calibration learns a mapping to realized outcomes.
      </li>
      <li>
        Polymarket provides an alternate probability signal; the overlay learns
        when to trust it based on liquidity, spreads, and momentum.
      </li>
      <li>
        The model surfaces an “edge” when the calibrated probability diverges
        from the market’s implied probability.
      </li>
    </ul>

    <h3>Common Failure Modes</h3>
    <ul>
      <li>
        Two‑stage overlay requires labels and overlapping dates between pRN and
        Polymarket datasets; otherwise it fails early.
      </li>
      <li>
        Mixed mode requires a decision‑features dataset; it is unavailable for
        daily/1DTE datasets in the UI.
      </li>
      <li>
        If the label column is missing or filtered to zero rows, training halts.
      </li>
    </ul>
  </>
);

const marketsDoc = (
  <>
    <p>
      The Markets page refreshes and visualizes weekly Polymarket market data.
      It fetches historical bid/ask curves and the option-chain-derived
      risk‑neutral probability (pRN) for the selected week, then lets you explore
      tickers and strikes in a focused chart.
    </p>

    <div className="docs-split">
      <div className="docs-panel">
        <h3>What It Produces</h3>
        <ul>
          <li>Weekly Polymarket bid/ask series by ticker and strike.</li>
          <li>Risk‑neutral probability (pRN) series aligned to the same timestamps.</li>
          <li>Summary metadata for the week: market counts, tickers, and timestamps.</li>
        </ul>
      </div>
      <div className="docs-panel">
        <h3>When To Use It</h3>
        <ul>
          <li>Inspect historical pricing for a specific weekly market week.</li>
          <li>Compare Polymarket quotes vs pRN for a single strike.</li>
          <li>Validate that the weekly refresh pipeline is producing data.</li>
        </ul>
      </div>
    </div>

    <h3>Page Layout</h3>
    <ul>
      <li>Run configuration: select the week, ticker, and strike.</li>
      <li>Latest run output: progress, status, and errors from the refresh job.</li>
      <li>Chart: a single, centered time‑series plot for the chosen strike.</li>
    </ul>

    <h3>Typical Workflow</h3>
    <ol className="docs-steps">
      <li>Pick the week‑ending Friday (UTC) to refresh.</li>
      <li>Click Refresh Now to fetch data for that week.</li>
      <li>Select a ticker from the trading universe list.</li>
      <li>Choose a strike to view its chart.</li>
      <li>Hover the chart to compare bid/ask and pRN at a point in time.</li>
    </ol>

    <h3>Run Configuration</h3>
    <p>
      The page auto‑selects the most recent Friday in the
      <code>America/New_York</code> timezone and can auto‑refresh on first load.
      The refresh job is blocked if another pipeline job is already running.
    </p>

    <div className="docs-split">
      <div className="docs-panel">
        <h3>Week Selection</h3>
        <ul>
          <li>
            Week Friday defines the week range. The chart shows Monday 00:00 UTC
            through Friday 23:59:59 UTC for that week.
          </li>
          <li>
            Changing the date updates the next refresh request.
          </li>
        </ul>
      </div>
      <div className="docs-panel">
        <h3>Trading Universe</h3>
        <ul>
          <li>
            The ticker list is loaded from the refreshed summary data.
          </li>
          <li>
            Click a ticker pill to load its strikes and series.
          </li>
          <li>
            If no tickers are available yet, the selector is disabled with a hint.
          </li>
        </ul>
      </div>
      <div className="docs-panel">
        <h3>Strike Selection</h3>
        <ul>
          <li>
            Strikes load after a ticker is selected.
          </li>
          <li>
            Each strike pill shows its threshold, market id (when available),
            and the number of points in the series.
          </li>
          <li>
            Selecting a strike renders the chart immediately.
          </li>
        </ul>
      </div>
    </div>

    <h3>Run Monitoring & Errors</h3>
    <ul>
      <li>
        The run monitor shows stage, week, market count, run id, and timestamps.
      </li>
      <li>
        Progress advances after each ticker is fully processed.
      </li>
      <li>
        Errors and warnings are captured in the output panel.
      </li>
      <li>
        Job polling continues across navigation until the run completes.
      </li>
    </ul>

    <h3>Chart Behavior</h3>
    <ul>
      <li>
        Lines: Polymarket bid, Polymarket ask, and pRN (risk‑neutral probability).
      </li>
      <li>
        Hover shows UTC timestamp, local time in ET, and values at the nearest point.
      </li>
      <li>
        Warning chips appear when Polymarket or pRN data is missing for a strike.
      </li>
      <li>
        If a series has fewer than two points, the corresponding line is not drawn.
      </li>
    </ul>

    <h3>Behavior & Persistence</h3>
    <ul>
      <li>
        The page auto‑refreshes on first visit unless a run is already active.
      </li>
      <li>
        The last job id is stored locally so status can resume after reload.
      </li>
      <li>
        When a refresh finishes, the summary and ticker list update automatically.
      </li>
    </ul>
  </>
);

const backtestsDoc = (
  <>
    <p>
      The Backtests page is an experimental price explorer for Polymarket data.
      It lets you inspect per‑strike price curves for a specific trading week and
      compare them to pRN overlays from option‑chain data. This page is still
      under active development, so its scope is intentionally narrow and focused
      on visual inspection rather than full strategy backtesting.
    </p>

    <div className="docs-split">
      <div className="docs-panel">
        <h3>What It Does Today</h3>
        <ul>
          <li>Loads per‑strike Polymarket price bars for a ticker and week.</li>
          <li>Overlays pRN dots computed from option‑chain history.</li>
          <li>Optionally merges Theta‑computed pRN to fill missing DTEs.</li>
          <li>Renders a single strike chart with bid/ask + pRN overlays.</li>
        </ul>
      </div>
      <div className="docs-panel">
        <h3>Work In Progress</h3>
        <ul>
          <li>No trade simulation, PnL, or strategy metrics yet.</li>
          <li>No export of results or formal report generation.</li>
          <li>UI/logic may change as the backtesting pipeline matures.</li>
        </ul>
      </div>
    </div>

    <h3>Page Layout</h3>
    <ul>
      <li>Ticker selector: choose one trading‑universe ticker.</li>
      <li>Strikes panel: select the strike to chart.</li>
      <li>Trading week calendar: pick a Mon–Fri week with data.</li>
      <li>Pipeline run selector: choose which historical run to use.</li>
      <li>Results: a single chart for the selected strike.</li>
    </ul>

    <h3>Typical Workflow</h3>
    <ol className="docs-steps">
      <li>Select a ticker from the trading universe.</li>
      <li>Pick a trading week from the calendar (Mon–Fri only).</li>
      <li>Choose a strike from the strikes list.</li>
      <li>The chart renders automatically once selections are valid.</li>
      <li>Hover the chart to inspect bid/ask and pRN values by time.</li>
    </ol>

    <h3>Controls & Inputs</h3>
    <div className="docs-split">
      <div className="docs-panel">
        <h3>Ticker Selection</h3>
        <ul>
          <li>Only the core trading universe is available.</li>
          <li>Changing ticker resets the selected week and strike.</li>
        </ul>
      </div>
      <div className="docs-panel">
        <h3>Trading Week Calendar</h3>
        <ul>
          <li>Weeks are Monday–Friday in UTC.</li>
          <li>Only weeks with available data are clickable.</li>
          <li>The range is limited to 5 days (one trading week).</li>
          <li>Navigation arrows are disabled outside the data range.</li>
        </ul>
      </div>
      <div className="docs-panel">
        <h3>Pipeline Run</h3>
        <ul>
          <li>Choose a historical bar run or use the active run.</li>
          <li>The run controls which historical dataset is queried.</li>
          <li>Run labels and active status are shown in the dropdown.</li>
        </ul>
      </div>
    </div>

    <h3>Strikes Panel</h3>
    <ul>
      <li>
        Strikes are loaded after a valid ticker and week are selected.
      </li>
      <li>
        Each strike pill shows strike price, market id (if available), and
        the number of points in the series.
      </li>
      <li>
        Selecting a strike auto‑runs the data fetch and renders the chart.
      </li>
    </ul>

    <h3>Data Sources & Overlays</h3>
    <ul>
      <li>
        Polymarket bars are the primary price series; bid/ask is synthesized
        from bar prices when only mid prices are available.
      </li>
      <li>
        A weekly markets series is loaded when available to supply bid/ask and
        a proxy pRN line.
      </li>
      <li>
        pRN overlay dots come from the option‑chain dataset; only strikes with
        DTEs {`{`}1,2,3,4{`}`} are plotted.
      </li>
      <li>
        Theta on‑demand pRN is queried after the initial fetch to fill missing
        strike/DTE gaps; the merge is non‑destructive and only fills gaps.
      </li>
    </ul>

    <h3>Chart Behavior</h3>
    <ul>
      <li>Chart time axis is UTC; tooltips show UTC and ET timestamps.</li>
      <li>Lines: Polymarket bid, Polymarket ask, and pRN (when available).</li>
      <li>Dots: pRN overlay points (from training data and/or Theta).</li>
      <li>
        If there are fewer than two points, the chart shows a “Not enough data”
        placeholder.
      </li>
      <li>
        Warning chips indicate missing Polymarket data or missing pRN overlays.
      </li>
    </ul>

    <h3>Progress & Errors</h3>
    <ul>
      <li>
        Progress steps include fetching markets, processing strikes, Theta pRN,
        and completion.
      </li>
      <li>
        Errors are shown inline and stop the run; non‑fatal overlay failures log
        warnings and continue.
      </li>
    </ul>

    <h3>Current Limitations (WIP)</h3>
    <ul>
      <li>No backtest metrics like returns, drawdown, or Sharpe.</li>
      <li>No trade execution logic or position accounting.</li>
      <li>No bulk comparison across strikes or tickers in a single view.</li>
      <li>Limited to a single trading week per run.</li>
    </ul>
  </>
);

export default function DocumentationPage() {
  const { activeJobs } = useAnyJobRunning();
  const location = useLocation();

  useEffect(() => {
    if (!location.hash) return;
    const targetId = decodeURIComponent(location.hash.slice(1));
    const scrollToTarget = () => {
      document.getElementById(targetId)?.scrollIntoView({ block: "start" });
    };
    scrollToTarget();
    const rafId = window.requestAnimationFrame(scrollToTarget);
    return () => window.cancelAnimationFrame(rafId);
  }, [location.hash]);

  return (
    <section className="page docs">
      <PipelineStatusCard className="page-sticky-meta" activeJobsCount={activeJobs.length} />
      <header className="page-header docs-header">
        <div>
          <p className="page-kicker">Documentation</p>
          <h1 className="page-title">Documentation</h1>
        </div>
      </header>

      <div className="docs-layout">
        <aside className="docs-toc">
          <div className="docs-toc-card">
            <div className="toc-title">Pages</div>
            <nav aria-label="Documentation sections">
              {docPages.map((page) => (
                <a key={page.id} className="toc-link" href={`#${page.id}`}>
                  {page.label}
                </a>
              ))}
            </nav>
          </div>
        </aside>

        <article className="docs-content">
          {docPages.map((page) => (
            <section key={page.id} id={page.id} className="docs-section">
              <div className="docs-section-header">
                <h2>{page.label}</h2>
                <Link className="button light" to={page.route}>
                  Open Page
                </Link>
              </div>
              {page.id === "option-chain" ? <OptionChainDocContent /> : null}
              {page.id === "polymarket-history" ? polymarketHistoryDoc : null}
              {page.id === "calibrate" ? calibrateDoc : null}
              {page.id === "markets" ? marketsDoc : null}
              {page.id === "backtests" ? backtestsDoc : null}
            </section>
          ))}
        </article>
      </div>
    </section>
  );
}
