from __future__ import annotations

import importlib.util
import multiprocessing
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_SRC = Path(__file__).resolve().parents[3]
REPO_ROOT = PROJECT_SRC.parent

if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from scripts.polymarket import weekly_history_io as bars_io  # noqa: E402


def _load_features_module():
    module_path = PROJECT_SRC / "scripts" / "02-polymarket-build-features-v1.0.py"
    spec = importlib.util.spec_from_file_location("polymarket_build_features_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FEATURES_MODULE = _load_features_module()


def _concurrent_write_worker(bars_dir: str, rows: list[dict]) -> None:
    if str(PROJECT_SRC) not in sys.path:
        sys.path.insert(0, str(PROJECT_SRC))
    from scripts.polymarket.weekly_history_io import write_bars  # noqa: WPS433

    for row in rows:
        write_bars(pd.DataFrame([row]), Path(bars_dir), "1h")


class BarsHistoryMasterStorageTests(unittest.TestCase):
    def _bar_row(
        self,
        timestamp_utc: str,
        market_id: str,
        close: float,
        *,
        open_price: float | None = None,
        high: float | None = None,
        low: float | None = None,
    ) -> dict:
        value = close if open_price is None else open_price
        return {
            "timestamp_utc": timestamp_utc,
            "market_id": market_id,
            "open": value,
            "high": close if high is None else high,
            "low": close if low is None else low,
            "close": close,
            "volume": None,
            "trade_count": None,
            "schema_version": "pm_bars_history_v1.0",
        }

    def test_write_bars_upserts_overlapping_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bars_dir = Path(tmp)
            first = pd.DataFrame(
                [
                    self._bar_row("2026-03-01T00:00:00Z", "market-1", 0.10),
                    self._bar_row("2026-03-02T00:00:00Z", "market-1", 0.20),
                ]
            )
            second = pd.DataFrame(
                [
                    self._bar_row("2026-03-02T00:00:00Z", "market-1", 0.25),
                    self._bar_row("2026-03-03T00:00:00Z", "market-1", 0.30),
                ]
            )

            self.assertEqual(bars_io.write_bars(first, bars_dir, "1d"), 2)
            self.assertEqual(bars_io.write_bars(second, bars_dir, "1d"), 2)

            master = pd.read_csv(bars_dir / "daily_master.csv", dtype={"market_id": str})

        self.assertEqual(len(master), 3)
        self.assertEqual(master.iloc[1]["timestamp_utc"], "2026-03-02T00:00:00Z")
        self.assertAlmostEqual(float(master.iloc[1]["close"]), 0.25)
        self.assertEqual(list(master["market_id"]), ["market-1", "market-1", "market-1"])

    def test_write_bars_deduplicates_incoming_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bars_dir = Path(tmp)
            duplicate_batch = pd.DataFrame(
                [
                    self._bar_row("2026-03-01T01:00:00Z", "market-1", 0.10),
                    self._bar_row("2026-03-01T01:00:00Z", "market-1", 0.15),
                ]
            )

            written = bars_io.write_bars(duplicate_batch, bars_dir, "1h")
            master = pd.read_csv(bars_dir / "hourly_master.csv", dtype={"market_id": str})

        self.assertEqual(written, 1)
        self.assertEqual(len(master), 1)
        self.assertAlmostEqual(float(master.iloc[0]["close"]), 0.15)

    def test_master_writes_are_safe_under_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bars_dir = Path(tmp)
            rows_a = [
                self._bar_row(f"2026-03-01T{hour:02d}:00:00Z", "market-a", 0.10 + hour / 100.0)
                for hour in range(10)
            ]
            rows_b = [
                self._bar_row(f"2026-03-02T{hour:02d}:00:00Z", "market-b", 0.20 + hour / 100.0)
                for hour in range(10)
            ]

            ctx = multiprocessing.get_context("spawn")
            proc_a = ctx.Process(target=_concurrent_write_worker, args=(str(bars_dir), rows_a))
            proc_b = ctx.Process(target=_concurrent_write_worker, args=(str(bars_dir), rows_b))
            proc_a.start()
            proc_b.start()
            proc_a.join(30)
            proc_b.join(30)

            self.assertEqual(proc_a.exitcode, 0)
            self.assertEqual(proc_b.exitcode, 0)

            master = pd.read_csv(bars_dir / "hourly_master.csv", dtype={"market_id": str})

        self.assertEqual(len(master), 20)
        self.assertEqual(int(master.duplicated(subset=["market_id", "timestamp_utc"]).sum()), 0)

    def test_feature_loader_reads_master_files_with_market_and_date_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bars_dir = Path(tmp)
            daily = pd.DataFrame(
                [
                    self._bar_row("2026-03-01T00:00:00Z", "market-1", 0.10),
                    self._bar_row("2026-03-02T00:00:00Z", "market-1", 0.20),
                    self._bar_row("2026-03-02T00:00:00Z", "market-2", 0.30),
                    self._bar_row("2026-03-03T00:00:00Z", "market-1", 0.40),
                ]
            )
            bars_io.write_bars(daily, bars_dir, "1d")

            loaded = FEATURES_MODULE._load_bars(
                bars_dir,
                "1d",
                ["market-1"],
                "2026-03-02",
                "2026-03-03",
            )

        self.assertEqual(len(loaded), 2)
        self.assertEqual(set(loaded["market_id"].astype(str)), {"market-1"})
        self.assertEqual(
            set(loaded["timestamp_utc"].dt.strftime("%Y-%m-%d").tolist()),
            {"2026-03-02", "2026-03-03"},
        )

    def test_migration_builds_two_masters_and_cleans_legacy_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bars_dir = Path(tmp)
            legacy_hourly = bars_dir / "1h" / "market_id=market-1" / "date=2026-03-01"
            legacy_daily = bars_dir / "1d" / "market_id=market-1" / "date=2026-03-01"
            legacy_hourly.mkdir(parents=True, exist_ok=True)
            legacy_daily.mkdir(parents=True, exist_ok=True)

            hourly_rows = pd.DataFrame(
                [
                    self._bar_row("2026-03-01T00:00:00Z", "market-1", 0.10),
                    self._bar_row("2026-03-01T00:00:00Z", "market-1", 0.12),
                    self._bar_row("2026-03-01T01:00:00Z", "market-1", 0.15),
                ]
            )
            daily_rows = pd.DataFrame(
                [
                    self._bar_row("2026-03-01T00:00:00Z", "market-1", 0.20),
                    self._bar_row("2026-03-02T00:00:00Z", "market-1", 0.25),
                ]
            )
            hourly_rows.to_csv(legacy_hourly / "bars.csv", index=False)
            daily_rows.to_csv(legacy_daily / "bars.csv", index=False)

            hourly_result = bars_io.build_master_from_legacy_partitions(bars_dir, "1h")
            daily_result = bars_io.build_master_from_legacy_partitions(bars_dir, "1d")

            self.assertTrue((bars_dir / "1h").exists())
            self.assertTrue((bars_dir / "1d").exists())
            bars_io.cleanup_legacy_partition_dirs(bars_dir, freqs=["1h", "1d"])

            hourly_master = pd.read_csv(bars_dir / "hourly_master.csv", dtype={"market_id": str})
            daily_master = pd.read_csv(bars_dir / "daily_master.csv", dtype={"market_id": str})

            self.assertFalse((bars_dir / "1h").exists())
            self.assertFalse((bars_dir / "1d").exists())

        self.assertEqual(hourly_result["legacy_files"], 1)
        self.assertEqual(hourly_result["legacy_rows"], 3)
        self.assertEqual(hourly_result["unique_rows"], 2)
        self.assertEqual(daily_result["unique_rows"], 2)
        self.assertEqual(len(hourly_master), 2)
        self.assertEqual(len(daily_master), 2)

    def test_feature_build_smoke_works_with_master_daily_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bars_dir = root / "bars_history"
            out_dir = root / "features_out"
            dim_market_path = root / "dim_market.csv"
            prn_path = root / "prn.csv"

            bars_io.write_bars(
                pd.DataFrame(
                    [
                        self._bar_row("2026-03-01T00:00:00Z", "market-1", 0.20),
                        self._bar_row("2026-03-02T00:00:00Z", "market-1", 0.30),
                    ]
                ),
                bars_dir,
                "1d",
            )

            pd.DataFrame(
                [
                    {
                        "market_id": "market-1",
                        "condition_id": "condition-1",
                        "ticker": "AAPL",
                        "threshold": 250,
                        "expiry_date_utc": "2026-03-07T21:00:00Z",
                        "resolution_time_utc": "2026-03-07T21:00:00Z",
                    },
                ]
            ).to_csv(dim_market_path, index=False)

            pd.DataFrame(
                [
                    {
                        "ticker": "AAPL",
                        "K": 250,
                        "expiry_date": "2026-03-07",
                        "asof_date": "2026-03-01",
                        "pRN": 0.20,
                    },
                    {
                        "ticker": "AAPL",
                        "K": 250,
                        "expiry_date": "2026-03-07",
                        "asof_date": "2026-03-02",
                        "pRN": 0.30,
                    },
                ]
            ).to_csv(prn_path, index=False)

            script_path = PROJECT_SRC / "scripts" / "02-polymarket-build-features-v1.0.py"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--dim-market",
                    str(dim_market_path),
                    "--bars-dir",
                    str(bars_dir),
                    "--out-dir",
                    str(out_dir),
                    "--prn-dataset",
                    str(prn_path),
                    "--decision-freq",
                    "1d",
                    "--start-date",
                    "2026-03-01",
                    "--end-date",
                    "2026-03-02",
                    "--skip-subgraph-labels",
                ],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
            )

            if result.returncode != 0:
                self.fail(f"Feature build failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

            features_csv = out_dir / "decision_features.csv"
            features_parquet = out_dir / "decision_features.parquet"
            manifest_path = out_dir / "feature_manifest.json"

            features_path = features_csv if features_csv.exists() else features_parquet
            self.assertTrue(features_path.exists())
            self.assertTrue(manifest_path.exists())

            if features_path.suffix == ".csv":
                features_df = pd.read_csv(features_path)
            else:
                features_df = pd.read_parquet(features_path)

        self.assertFalse(features_df.empty)
        self.assertIn("market_id", features_df.columns)


if __name__ == "__main__":
    unittest.main()
