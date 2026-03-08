#!/usr/bin/env python3
"""
backfill_gamma_volume.py

One-time backfill: adds gamma_volume and gamma_volume_24hr columns to an
existing weekly_markets.csv by querying the Gamma API for each market.

Usage:
    python src/scripts/backfill_gamma_volume.py \
        --csv src/data/raw/polymarket/weekly_history/runs/main-polymarket-2/weekly_markets.csv

    # Dry-run (prints what would change, writes nothing):
    python src/scripts/backfill_gamma_volume.py --csv <path> --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
REQUEST_TIMEOUT = 20
BATCH_SIZE = 50
SLEEP_BETWEEN_BATCHES = 0.3


def _build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _fetch_volume_for_market(
    session: requests.Session,
    market_id: str,
) -> Tuple[Optional[float], Optional[float]]:
    """Fetch volume and volume24hr for a single market from the Gamma API."""
    try:
        resp = session.get(
            f"{GAMMA_MARKETS_URL}/{market_id}",
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 404:
            return None, None
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            return None, None

        gamma_volume = None
        for vol_key in ("volume", "volumeNum"):
            raw_vol = data.get(vol_key)
            if raw_vol is not None:
                try:
                    gamma_volume = float(raw_vol)
                except (ValueError, TypeError):
                    pass
                if gamma_volume is not None:
                    break

        gamma_volume_24hr = None
        raw_vol_24hr = data.get("volume24hr") or data.get("volume_24hr")
        if raw_vol_24hr is not None:
            try:
                gamma_volume_24hr = float(raw_vol_24hr)
            except (ValueError, TypeError):
                pass

        return gamma_volume, gamma_volume_24hr
    except Exception as exc:
        print(f"  [WARN] Failed to fetch market_id={market_id}: {exc}", flush=True)
        return None, None


def backfill(csv_path: Path, dry_run: bool = False) -> None:
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Read existing CSV
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        original_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    already_has_volume = "gamma_volume" in original_fieldnames
    if already_has_volume:
        filled = sum(1 for r in rows if r.get("gamma_volume", "").strip())
        empty = len(rows) - filled
        print(f"CSV already has gamma_volume column ({filled} filled, {empty} empty).")
        if empty == 0:
            print("Nothing to backfill — all rows already have gamma_volume.")
            return

    # Collect unique market_ids that need volume data
    market_ids_needing_volume: Dict[str, bool] = {}
    for row in rows:
        mid = row.get("market_id", "").strip()
        if not mid:
            continue
        if already_has_volume and row.get("gamma_volume", "").strip():
            continue
        market_ids_needing_volume[mid] = True

    unique_ids = list(market_ids_needing_volume.keys())
    print(f"Markets to fetch: {len(unique_ids)} (total rows: {len(rows)})")

    if dry_run:
        print("[DRY RUN] Would fetch volume for these market_ids:")
        for mid in unique_ids[:20]:
            print(f"  {mid}")
        if len(unique_ids) > 20:
            print(f"  ... and {len(unique_ids) - 20} more")
        return

    # Fetch volumes from Gamma API
    session = _build_session()
    volume_map: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    fetched = 0
    errors = 0
    t0 = time.time()

    for i, mid in enumerate(unique_ids):
        vol, vol_24hr = _fetch_volume_for_market(session, mid)
        volume_map[mid] = (vol, vol_24hr)
        fetched += 1
        if vol is None:
            errors += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(unique_ids):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  [{i + 1}/{len(unique_ids)}] "
                f"fetched={fetched} errors={errors} "
                f"rate={rate:.1f}/s elapsed={elapsed:.0f}s",
                flush=True,
            )
            if (i + 1) % BATCH_SIZE == 0 and (i + 1) < len(unique_ids):
                time.sleep(SLEEP_BETWEEN_BATCHES)

    # Build updated fieldnames
    out_fieldnames = list(original_fieldnames)
    if "gamma_volume" not in out_fieldnames:
        schema_idx = out_fieldnames.index("schema_version") if "schema_version" in out_fieldnames else len(out_fieldnames)
        out_fieldnames.insert(schema_idx, "gamma_volume")
        out_fieldnames.insert(schema_idx + 1, "gamma_volume_24hr")

    # Write back
    backup_path = csv_path.with_suffix(".csv.bak")
    csv_path.rename(backup_path)
    print(f"Backup saved: {backup_path}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames, extrasaction="ignore")
        writer.writeheader()
        rows_updated = 0
        for row in rows:
            mid = row.get("market_id", "").strip()
            if mid and mid in volume_map:
                vol, vol_24hr = volume_map[mid]
                if vol is not None:
                    row["gamma_volume"] = str(vol)
                    rows_updated += 1
                elif "gamma_volume" not in row:
                    row["gamma_volume"] = ""
                if vol_24hr is not None:
                    row["gamma_volume_24hr"] = str(vol_24hr)
                elif "gamma_volume_24hr" not in row:
                    row["gamma_volume_24hr"] = ""
            else:
                row.setdefault("gamma_volume", "")
                row.setdefault("gamma_volume_24hr", "")
            writer.writerow(row)

    print(f"Done. Updated {rows_updated}/{len(rows)} rows with gamma_volume.")
    print(f"Written: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill gamma_volume into an existing weekly_markets.csv",
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the weekly_markets.csv to backfill",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making changes",
    )
    args = parser.parse_args()
    backfill(Path(args.csv), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
