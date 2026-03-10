from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.analytics import assess_run_volume_analytics_readiness  # noqa: E402


class AnalyticsReadinessTests(unittest.TestCase):
    def test_run_without_subgraph_is_not_ready(self) -> None:
        readiness = assess_run_volume_analytics_readiness({
            "pipeline_args": {
                "include_subgraph": False,
            },
            "subgraph": {},
        })

        self.assertFalse(readiness["analytics_ready"])
        self.assertFalse(readiness["has_trades"])
        self.assertIn("without subgraph trade ingestion", readiness["warning"])

    def test_run_with_trade_artifact_is_ready(self) -> None:
        readiness = assess_run_volume_analytics_readiness({
            "pipeline_args": {
                "include_subgraph": True,
            },
            "subgraph": {
                "ok": True,
                "total_entities": 1250,
                "partitions": 9,
            },
            "trade_artifact": {
                "path": "trades.csv",
                "rows": 1250,
                "trade_days": 9,
                "schema_version": "pm_run_trades_v1.0",
            },
        })

        self.assertTrue(readiness["analytics_ready"])
        self.assertTrue(readiness["has_trades"])
        self.assertEqual(readiness["trade_entities"], 1250)
        self.assertEqual(readiness["trade_days"], 9)
        self.assertEqual(readiness["trade_artifact_path"], "trades.csv")
        self.assertIsNone(readiness["warning"])

    def test_run_with_subgraph_metadata_but_no_trade_artifact_stays_blocked(self) -> None:
        readiness = assess_run_volume_analytics_readiness({
            "pipeline_args": {
                "include_subgraph": True,
            },
            "subgraph": {
                "ok": True,
                "total_entities": 1250,
                "partitions": 9,
            },
        })

        self.assertFalse(readiness["analytics_ready"])
        self.assertFalse(readiness["has_trades"])
        self.assertIn("no run-scoped trades artifact", readiness["warning"])


if __name__ == "__main__":
    unittest.main()
