from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services import analysis  # noqa: E402


class AnalysisContractTests(unittest.TestCase):
    def test_rejects_unsupported_analysis_variable(self) -> None:
        with self.assertRaises(ValueError):
            analysis._require_volume_variable("drop_table")

    def test_rejects_ticker_filter_for_threshold_rule(self) -> None:
        with self.assertRaises(ValueError):
            analysis._analysis_table_query_parts(
                table_name="threshold_rule",
                ticker="AAPL",
                variable_name="notional_volume",
            )

    def test_accepts_supported_filters_for_outlier_flag(self) -> None:
        table, where_sql, params, order_sql = analysis._analysis_table_query_parts(
            table_name="outlier_flag",
            ticker="AAPL",
            variable_name="notional_volume",
        )

        self.assertEqual(table.name, "outlier_flag")
        self.assertIn("ticker = :ticker", where_sql)
        self.assertIn("variable_name = :variable_name", where_sql)
        self.assertEqual(params["ticker"], "AAPL")
        self.assertEqual(params["variable_name"], "notional_volume")
        self.assertIn("flag_id DESC", order_sql)


if __name__ == "__main__":
    unittest.main()
