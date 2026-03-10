from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_SRC = Path(__file__).resolve().parents[3]

if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from scripts.polymarket.trade_artifact import (  # noqa: E402
    RUN_TRADES_SCHEMA_VERSION,
    build_trade_artifact_metadata,
    enrich_trades_for_run,
)


class TradeArtifactTests(unittest.TestCase):
    def test_enrich_trades_for_run_maps_token_roles_and_notional(self) -> None:
        trades = pd.DataFrame([
            {
                "trade_id": "trade-1",
                "block_number": 100,
                "timestamp_utc": "2026-03-05T12:00:00Z",
                "market_id": "market-1",
                "outcome_token_id": "yes-token",
                "price": 0.42,
                "size": 100,
                "side": "buy",
                "tx_hash": "0xabc",
            },
            {
                "trade_id": "trade-2",
                "block_number": 101,
                "timestamp_utc": "2026-03-06T12:00:00Z",
                "market_id": "market-1",
                "outcome_token_id": "no-token",
                "price": 0.58,
                "size": 50,
                "side": "sell",
                "tx_hash": "0xdef",
            },
        ])
        markets = pd.DataFrame([
            {
                "market_id": "market-1",
                "event_id": "event-1",
                "ticker": "AAPL",
                "threshold": 250,
                "week_friday": "2026-03-06",
                "event_endDate": "2026-03-06T21:00:00Z",
                "yes_token_id": "yes-token",
                "no_token_id": "no-token",
            },
        ])

        enriched = enrich_trades_for_run(trades, markets)

        self.assertEqual(list(enriched["token_role"]), ["yes", "no"])
        self.assertEqual(list(enriched["ticker"]), ["AAPL", "AAPL"])
        self.assertAlmostEqual(float(enriched.iloc[0]["notional"]), 42.0)
        self.assertAlmostEqual(float(enriched.iloc[1]["notional"]), 29.0)
        self.assertTrue((enriched["schema_version"] == RUN_TRADES_SCHEMA_VERSION).all())

    def test_build_trade_artifact_metadata_counts_rows_and_trade_days(self) -> None:
        trades = pd.DataFrame([
            {
                "trade_id": "trade-1",
                "timestamp_utc": "2026-03-05T12:00:00Z",
            },
            {
                "trade_id": "trade-2",
                "timestamp_utc": "2026-03-05T15:30:00Z",
            },
            {
                "trade_id": "trade-3",
                "timestamp_utc": "2026-03-06T09:00:00Z",
            },
        ])

        metadata = build_trade_artifact_metadata(trades, path="trades.csv")

        self.assertIsNotNone(metadata)
        assert metadata is not None
        self.assertEqual(metadata["path"], "trades.csv")
        self.assertEqual(metadata["rows"], 3)
        self.assertEqual(metadata["trade_days"], 2)
        self.assertEqual(metadata["schema_version"], RUN_TRADES_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
