"""Service for loading pRN overlay data from option-chain training CSVs."""

import csv
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.models.bars import PrnOverlayResponse, PrnPoint, PrnStrikeSeries

log = logging.getLogger("prn_overlay")

BASE_DIR = Path(__file__).resolve().parents[5]
OPTION_CHAIN_DIR = BASE_DIR / "src" / "data" / "raw" / "option-chain"

# DTE values we serve (weekly options: Mon=4, Tue=3, Wed=2, Thu=1)
ALLOWED_DTES = {1, 2, 3, 4}


def _find_training_csvs() -> List[Path]:
    """Discover training CSVs in option-chain directories."""
    if not OPTION_CHAIN_DIR.exists():
        return []
    results: List[Path] = []
    for sub in sorted(OPTION_CHAIN_DIR.iterdir()):
        if not sub.is_dir():
            continue
        for f in sub.iterdir():
            if f.name.startswith("training-") and f.name.endswith(".csv"):
                results.append(f)
    return results


def _asof_date_to_eod_ms(date_str: str) -> Optional[int]:
    """Convert YYYY-MM-DD to US-market-close UTC ms (21:00 UTC) for chart placement.

    pRN is computed from EOD option chains finalised at ~16:00 ET / 21:00 UTC,
    so the chart dot must not appear before that time.
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
            hour=21, minute=0, second=0, tzinfo=timezone.utc
        )
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError):
        return None


def _normalize_strike(val: float) -> float:
    """Round strike to 2 decimal places for safe matching."""
    return round(val, 2)


def get_prn_overlay(
    ticker: str,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
) -> PrnOverlayResponse:
    """Load pRN data for a ticker within a date range.

    Scans training CSVs for matching rows, groups by strike, and returns
    sorted pRN points for DTE in {4, 3, 2, 1}.
    """
    # Parse date bounds (date-only, no time component needed)
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    if time_min:
        date_min = time_min[:10]  # "2026-02-03T00:00:00Z" -> "2026-02-03"
    if time_max:
        date_max = time_max[:10]

    csv_files = _find_training_csvs()
    if not csv_files:
        log.warning("No training CSVs found in %s", OPTION_CHAIN_DIR)
        return PrnOverlayResponse(ticker=ticker, strikes=[], metadata={"error": "no_training_csvs"})

    # strike -> list of PrnPoint
    groups: Dict[float, List[PrnPoint]] = {}
    rows_scanned = 0
    rows_matched = 0
    dataset_path: Optional[str] = None

    for csv_path in csv_files:
        dataset_path = str(csv_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows_scanned += 1

                if row.get("ticker") != ticker:
                    continue

                asof_date = row.get("asof_date", "")
                if not asof_date:
                    continue

                # Date range filter
                if date_min and asof_date < date_min:
                    continue
                if date_max and asof_date > date_max:
                    continue

                # DTE filter
                try:
                    dte = int(row.get("T_days", ""))
                except (ValueError, TypeError):
                    continue
                if dte not in ALLOWED_DTES:
                    continue

                # Parse strike
                try:
                    strike = float(row.get("K", ""))
                except (ValueError, TypeError):
                    continue
                if not math.isfinite(strike):
                    continue

                # Parse pRN
                try:
                    prn = float(row.get("pRN", ""))
                except (ValueError, TypeError):
                    continue
                if not math.isfinite(prn):
                    continue

                asof_ms = _asof_date_to_eod_ms(asof_date)
                if asof_ms is None:
                    continue

                point = PrnPoint(
                    asof_date=asof_date,
                    asof_date_ms=asof_ms,
                    dte=dte,
                    pRN=prn,
                )
                groups.setdefault(_normalize_strike(strike), []).append(point)
                rows_matched += 1

    # Build sorted output, deduplicating by (asof_date, dte) per strike
    strikes_list: List[PrnStrikeSeries] = []
    for strike in sorted(groups.keys()):
        seen: set = set()
        deduped: List[PrnPoint] = []
        for p in sorted(groups[strike], key=lambda p: p.asof_date_ms):
            key = (p.asof_date, p.dte)
            if key not in seen:
                seen.add(key)
                deduped.append(p)
        label = str(int(strike)) if strike == int(strike) else f"{strike:.2f}"
        strikes_list.append(PrnStrikeSeries(
            strike=strike,
            strike_label=label,
            points=deduped,
        ))

    log.info(
        "prn_overlay: ticker=%s date_range=%s..%s scanned=%d matched=%d strikes=%d",
        ticker, date_min, date_max, rows_scanned, rows_matched, len(strikes_list),
    )

    return PrnOverlayResponse(
        ticker=ticker,
        dataset_path=dataset_path,
        strikes=strikes_list,
        metadata={
            "rows_scanned": rows_scanned,
            "rows_matched": rows_matched,
            "strikes_count": len(strikes_list),
        },
    )
