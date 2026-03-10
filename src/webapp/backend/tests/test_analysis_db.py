from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core import analysis_db  # noqa: E402


class AnalysisDbTests(unittest.TestCase):
    def test_default_upsert_update_columns_preserves_raw_ingest_timestamps(self) -> None:
        columns = analysis_db.default_upsert_update_columns(
            analysis_db.pm_raw_price_history,
            ["run_id", "timestamp_utc", "market_id", "token_id", "token_role"],
        )

        self.assertIn("price", columns)
        self.assertNotIn("ingested_at_utc", columns)

    def test_default_upsert_update_columns_preserves_refresh_metadata_timestamps(self) -> None:
        columns = analysis_db.default_upsert_update_columns(
            analysis_db.pm_raw_run_manifest,
            ["run_id"],
        )

        self.assertIn("status", columns)
        self.assertNotIn("last_refreshed_at_utc", columns)


if __name__ == "__main__":
    unittest.main()
