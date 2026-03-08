from __future__ import annotations

import importlib.util
import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parents[2]
SCRIPT_PATH = REPO_ROOT / "src" / "scripts" / "09-polymarket-analysis-refresh-v1.0.py"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

spec = importlib.util.spec_from_file_location("analysis_refresh_script", SCRIPT_PATH)
analysis_refresh = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules[spec.name] = analysis_refresh
spec.loader.exec_module(analysis_refresh)


class AnalysisRefreshPipelineTests(unittest.TestCase):
    def test_materialize_market_day_keeps_has_trade_data_separate_from_coverage(self) -> None:
        market_meta = pd.DataFrame(
            [
                {
                    "market_id": "m1",
                    "ticker": "AAPL",
                    "stock_name": "Apple",
                    "condition_id": "c1",
                    "event_id": "e1",
                    "yes_token_id": "y1",
                    "no_token_id": "n1",
                    "threshold": 200,
                    "week_monday": date(2026, 3, 2),
                    "week_friday": date(2026, 3, 6),
                    "resolution_time_utc": "2026-03-06T21:00:00Z",
                }
            ]
        )
        prices_daily = pd.DataFrame(
            [
                {
                    "market_id": "m1",
                    "trade_date_ny": date(2026, 3, 2),
                    "hours_observed": 4,
                    "open_prob": 0.4,
                    "high_prob": 0.5,
                    "low_prob": 0.3,
                    "close_prob": 0.45,
                    "close_prob_1600_et": 0.45,
                    "range_prob": 0.2,
                    "rv_logit_intraday": 0.1,
                    "gap_logit": 0.0,
                    "distance_to_boundary": 0.45,
                    "price_complete_flag": True,
                    "has_price_data": True,
                }
            ]
        )
        trades_daily = pd.DataFrame(
            [
                {
                    "market_id": "m1",
                    "trade_date_ny": date(2026, 3, 2),
                    "trade_count": 2,
                    "contract_volume": 10.0,
                    "notional_volume": 4.0,
                    "buy_volume": 6.0,
                    "sell_volume": 4.0,
                    "avg_trade_size": 5.0,
                    "buy_sell_imbalance": 0.2,
                    "has_trade_data": True,
                }
            ]
        )

        result = analysis_refresh.materialize_market_day(
            market_meta=market_meta,
            prices_daily=prices_daily,
            trades_daily=trades_daily,
            refresh_id="refresh-1",
            trade_coverage_dates=set(),
        )

        row = result.iloc[0]
        self.assertTrue(bool(row["has_trade_data"]))
        self.assertFalse(bool(row["volume_complete_flag"]))

    def test_materialize_stock_day_avg_trade_size_uses_contracts_per_trade(self) -> None:
        market_day = pd.DataFrame(
            [
                {
                    "trade_date_ny": date(2026, 3, 2),
                    "ticker": "AAPL",
                    "stock_name": "Apple",
                    "listed_flag": True,
                    "observed_flag": True,
                    "traded_flag": True,
                    "trade_count": 4.0,
                    "contract_volume": 20.0,
                    "notional_volume": 100.0,
                    "buy_volume": 12.0,
                    "sell_volume": 8.0,
                    "hours_observed": 5.0,
                    "close_prob": 0.55,
                    "days_to_resolution": 2.0,
                    "rv_logit_intraday": 0.2,
                    "has_price_data": True,
                    "volume_complete_flag": True,
                }
            ]
        )

        result = analysis_refresh.materialize_stock_day(market_day, refresh_id="refresh-1")

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertAlmostEqual(float(row["avg_trade_size"]), 5.0)


if __name__ == "__main__":
    unittest.main()
