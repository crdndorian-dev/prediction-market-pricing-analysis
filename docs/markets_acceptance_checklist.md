# Markets Page Acceptance Checklist

1. Weekly-only selection: for a given Friday, Markets refresh includes only events whose `event_endDate` local date equals that Friday and slug suffix matches `-above-on-<month>-<day>-<year>`.
2. No similar events: daily close/1DTE events are excluded even if ticker + date match.
3. No duplication: visiting `/markets` twice does not increase row counts for the same `(timestamp_utc, market_id, token_role)` in `price_history.csv` or `(timestamp_utc, ticker, threshold)` in `markets_prn_hourly.csv`.
4. Weekend behavior: opening the page on Saturday or Sunday produces charts only up to Friday 21:00 UTC and no weekend timestamps.
5. No leakage: for every pRN row, `spot_asof_utc <= timestamp_utc` and `rn_asof_utc <= timestamp_utc` and no rows exceed cutoff.
6. Alignment: pRN points align to the same hourly timestamps as Polymarket series for each strike.
7. Progress UX: on page load, a progress bar appears, advances through stages, and ends in a consistent UI state.
8. Failure modes: when CLOB is down, the page still renders pRN curves with a clear error banner; when yfinance is down, the page still renders Polymarket curves; partial market failures do not fail the whole job.
