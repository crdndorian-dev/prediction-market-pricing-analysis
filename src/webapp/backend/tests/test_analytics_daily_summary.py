from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


BACKEND_ROOT = Path(__file__).resolve().parents[1]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services import analytics  # noqa: E402


class DailyAnalyticsSummaryTests(unittest.TestCase):
    def _trade_row(
        self,
        trade_id: str,
        timestamp_utc: str,
        market_id: str,
        ticker: str,
        notional: float,
        size: float,
        *,
        token_role: str = "yes",
        threshold: float = 250,
        week_friday: str = "2026-03-06",
        event_endDate: str = "2026-03-06T21:00:00Z",
    ) -> dict:
        return {
            "trade_id": trade_id,
            "timestamp_utc": timestamp_utc,
            "market_id": market_id,
            "ticker": ticker,
            "notional": notional,
            "size": size,
            "token_role": token_role,
            "threshold": threshold,
            "week_friday": week_friday,
            "event_endDate": event_endDate,
        }

    def _write_run(
        self,
        root: Path,
        *,
        run_id: str = "test-run",
        manifest: dict | None = None,
        trades: pd.DataFrame | None = None,
    ) -> Path:
        run_dir = root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps(manifest or {}, indent=2))
        if trades is not None:
            trades.to_csv(run_dir / "trades.csv", index=False)
        return run_dir

    def _ready_manifest(self) -> dict:
        return {
            "pipeline_args": {
                "include_subgraph": True,
            },
            "trade_artifact": {
                "path": "trades.csv",
                "rows": 4,
                "trade_days": 2,
                "schema_version": "pm_run_trades_v1.0",
            },
        }

    def test_daily_summary_zero_fills_missing_days_and_compares_latest_to_prior_calendar_day(self) -> None:
        trades = pd.DataFrame([
            self._trade_row("trade-1", "2026-03-01T10:00:00Z", "market-1", "AAPL", 10, 20),
            self._trade_row("trade-2", "2026-03-01T11:00:00Z", "market-2", "MSFT", 15, 25),
            self._trade_row("trade-3", "2026-03-03T13:00:00Z", "market-1", "AAPL", 7, 5),
        ])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root, manifest=self._ready_manifest(), trades=trades)
            with patch.object(analytics, "RUNS_DIR", root):
                summary = analytics.get_run_daily_analytics_summary("test-run")

        self.assertEqual(summary.latest_trade_date, "2026-03-03")
        self.assertEqual(summary.comparison_date, "2026-03-02")
        self.assertEqual(len(summary.days), 3)
        self.assertEqual(summary.days[1].date, "2026-03-02")
        self.assertEqual(summary.days[1].daily_notional_volume, 0.0)
        self.assertEqual(summary.summary.latest_day.daily_notional_volume, 7.0)
        self.assertIsNotNone(summary.summary.comparison_day)
        assert summary.summary.comparison_day is not None
        self.assertEqual(summary.summary.comparison_day.daily_notional_volume, 0.0)
        self.assertIsNone(summary.summary.delta_notional_pct)
        self.assertEqual(summary.coverage.observed_trade_days, 2)
        self.assertEqual(summary.coverage.filled_days, 3)

    def test_daily_summary_supports_ticker_filters(self) -> None:
        trades = pd.DataFrame([
            self._trade_row("trade-1", "2026-03-01T10:00:00Z", "market-1", "AAPL", 10, 20),
            self._trade_row("trade-2", "2026-03-01T11:00:00Z", "market-2", "MSFT", 15, 25),
        ])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root, manifest=self._ready_manifest(), trades=trades)
            with patch.object(analytics, "RUNS_DIR", root):
                summary = analytics.get_run_daily_analytics_summary("test-run", tickers=["AAPL"])

        self.assertEqual(summary.coverage.requested_tickers, ["AAPL"])
        self.assertEqual(summary.coverage.filtered_trade_rows, 1)
        self.assertEqual(summary.summary.latest_day.daily_notional_volume, 10.0)
        self.assertEqual(summary.summary.latest_day.active_tickers, 1)

    def test_single_trade_day_returns_unavailable_comparison(self) -> None:
        trades = pd.DataFrame([
            self._trade_row("trade-1", "2026-03-01T10:00:00Z", "market-1", "AAPL", 10, 20),
        ])

        manifest = self._ready_manifest()
        manifest["trade_artifact"]["rows"] = 1
        manifest["trade_artifact"]["trade_days"] = 1

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root, manifest=manifest, trades=trades)
            with patch.object(analytics, "RUNS_DIR", root):
                summary = analytics.get_run_daily_analytics_summary("test-run")

        self.assertEqual(summary.latest_trade_date, "2026-03-01")
        self.assertIsNone(summary.comparison_date)
        self.assertIsNone(summary.summary.comparison_day)
        self.assertIn("Only one observed trade day", " ".join(summary.warnings))

    def test_not_ready_run_raises_value_error(self) -> None:
        manifest = {
            "pipeline_args": {
                "include_subgraph": False,
            },
        }
        trades = pd.DataFrame()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root, manifest=manifest, trades=trades)
            with patch.object(analytics, "RUNS_DIR", root):
                with self.assertRaises(ValueError):
                    analytics.get_run_daily_analytics_summary("test-run")

    def test_missing_required_columns_raises_value_error(self) -> None:
        trades = pd.DataFrame([
            {
                "trade_id": "trade-1",
                "timestamp_utc": "2026-03-01T10:00:00Z",
                "ticker": "AAPL",
                "notional": 10,
                "size": 20,
            },
        ])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root, manifest=self._ready_manifest(), trades=trades)
            with patch.object(analytics, "RUNS_DIR", root):
                with self.assertRaises(ValueError):
                    analytics.get_run_daily_analytics_summary("test-run")

    def test_daily_summary_computes_baseline_bands_and_structure(self) -> None:
        trades = pd.DataFrame([
            self._trade_row(f"trade-{idx}", f"2026-03-{idx:02d}T10:00:00Z", "market-1", "AAPL", 100 + idx, 10 + idx)
            for idx in range(1, 11)
        ])

        manifest = self._ready_manifest()
        manifest["trade_artifact"]["rows"] = len(trades)
        manifest["trade_artifact"]["trade_days"] = 10

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root, manifest=manifest, trades=trades)
            with patch.object(analytics, "RUNS_DIR", root):
                summary = analytics.get_run_daily_analytics_summary(
                    "test-run",
                    ci_level=90,
                    exclude_flagged=True,
                )

        self.assertEqual(summary.ci_level, 90)
        self.assertTrue(summary.exclude_flagged)
        self.assertEqual(summary.token_role, "all")
        self.assertTrue(summary.available_tickers)
        self.assertGreater(len(summary.structure.weekday), 0)
        latest = summary.summary.latest_day
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertIsNotNone(latest.expected_notional_volume)
        self.assertIsNotNone(latest.band_lo)
        self.assertIsNotNone(latest.band_hi)
        self.assertGreater(latest.expected_notional_volume or 0, 0)

    def test_breakdown_returns_rows_for_selected_day(self) -> None:
        trades = pd.DataFrame([
            self._trade_row("trade-1", "2026-03-01T10:00:00Z", "market-1", "AAPL", 100, 20, threshold=240),
            self._trade_row("trade-2", "2026-03-02T10:00:00Z", "market-1", "AAPL", 120, 20, threshold=240),
            self._trade_row("trade-3", "2026-03-02T11:00:00Z", "market-2", "MSFT", 80, 15, threshold=300),
            self._trade_row("trade-4", "2026-03-03T10:00:00Z", "market-1", "AAPL", 130, 25, threshold=240),
            self._trade_row("trade-5", "2026-03-03T12:00:00Z", "market-2", "MSFT", 95, 18, threshold=300),
            self._trade_row("trade-6", "2026-03-04T12:00:00Z", "market-2", "MSFT", 105, 18, threshold=300),
            self._trade_row("trade-7", "2026-03-05T12:00:00Z", "market-1", "AAPL", 110, 17, threshold=240),
            self._trade_row("trade-8", "2026-03-05T13:00:00Z", "market-2", "MSFT", 102, 16, threshold=300),
            self._trade_row("trade-9", "2026-03-06T12:00:00Z", "market-1", "AAPL", 108, 18, threshold=240),
            self._trade_row("trade-10", "2026-03-06T13:00:00Z", "market-2", "MSFT", 115, 18, threshold=300),
        ])

        manifest = self._ready_manifest()
        manifest["trade_artifact"]["rows"] = len(trades)
        manifest["trade_artifact"]["trade_days"] = 6

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_run(root, manifest=manifest, trades=trades)
            with patch.object(analytics, "RUNS_DIR", root):
                breakdown = analytics.get_run_daily_analytics_breakdown(
                    "test-run",
                    breakdown_date="2026-03-06",
                    group_by="ticker",
                    ci_level=90,
                )

        self.assertEqual(breakdown.group_by, "ticker")
        self.assertEqual(breakdown.date, "2026-03-06")
        self.assertGreaterEqual(len(breakdown.rows), 2)
        self.assertEqual(breakdown.rows[0].label, breakdown.rows[0].key)


if __name__ == "__main__":
    unittest.main()
