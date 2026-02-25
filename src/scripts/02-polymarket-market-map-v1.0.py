#!/usr/bin/env python3
"""
02-polymarket-market-map-v1.0.py

Build a stable Polymarket market identity map from the Graph subgraph.
Outputs dim_market with ticker/threshold/expiry fields required for joins.
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "src" / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from polymarket.subgraph_client import SubgraphClient

SCRIPT_VERSION = "1.0.0"
SCHEMA_VERSION = "pm_dim_market_v1.0"

DEFAULT_OUT_PATH = REPO_ROOT / "src" / "data" / "models" / "polymarket" / "dim_market.parquet"
DEFAULT_OVERRIDES_PATH = REPO_ROOT / "config" / "polymarket_market_overrides.csv"

MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

# Company name to ticker symbol mappings for improved inference
COMPANY_NAME_MAP = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "amazon": "AMZN",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "amd": "AMD",
    "intel": "INTC",
    "paypal": "PYPL",
    "visa": "V",
    "mastercard": "MA",
    "walmart": "WMT",
    "disney": "DIS",
    "nike": "NKE",
    "mcdonalds": "MCD",
    "coca-cola": "KO",
    "pepsi": "PEP",
    "boeing": "BA",
    "exxon": "XOM",
    "chevron": "CVX",
    "pfizer": "PFE",
    "johnson & johnson": "JNJ",
    "jpmorgan": "JPM",
    "bank of america": "BAC",
    "wells fargo": "WFC",
    "goldman sachs": "GS",
    "morgan stanley": "MS",
}


# ----------------------------
# Profiling infrastructure
# ----------------------------

@dataclass
class ProfileStats:
    """Track performance metrics for the pipeline."""
    stage_times: Dict[str, float] = field(default_factory=dict)
    stage_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    api_calls: int = 0
    start_time: float = field(default_factory=time.perf_counter)

    def record_stage(self, stage: str, duration: float) -> None:
        self.stage_times[stage] = duration

    def increment_count(self, key: str, value: int = 1) -> None:
        self.stage_counts[key] += value

    def print_summary(self) -> None:
        total = time.perf_counter() - self.start_time
        print("\n" + "="*70)
        print("[PROFILE] Performance Summary")
        print("="*70)
        print(f"Total runtime: {total:.2f}s")
        print("\nStage breakdown:")
        for stage, duration in sorted(self.stage_times.items(), key=lambda x: -x[1]):
            pct = (duration / total * 100) if total > 0 else 0
            print(f"  {stage:40s} {duration:8.2f}s  ({pct:5.1f}%)")
        print("\nCounts:")
        for key, count in sorted(self.stage_counts.items()):
            print(f"  {key:40s} {count:>12,}")
        if self.api_calls:
            print(f"  {'API calls':40s} {self.api_calls:>12,}")
        print("="*70 + "\n")


class ProfileContext:
    """Context manager for timing pipeline stages."""
    def __init__(self, stats: Optional[ProfileStats], stage: str):
        self.stats = stats
        self.stage = stage
        self.start = None

    def __enter__(self):
        if self.stats:
            self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.stats and self.start:
            duration = time.perf_counter() - self.start
            self.stats.record_stage(self.stage, duration)
            print(f"[PROFILE] {self.stage}: {duration:.2f}s")


@dataclass
class Config:
    out_path: Path = DEFAULT_OUT_PATH
    overrides_path: Path = DEFAULT_OVERRIDES_PATH
    run_dir: Optional[Path] = None
    run_id: Optional[str] = None
    tickers: Optional[List[str]] = None
    prn_dataset: Optional[Path] = None
    strict: bool = True
    profile: bool = False
    profile_output: Optional[Path] = None


# ----------------------------
# Parsing helpers
# ----------------------------

def _slugify_ticker(ticker: str) -> str:
    return ticker.strip().lower().replace(".", "").replace("/", "-").replace(" ", "")


def _parse_slug_prefix(slug: str) -> Optional[str]:
    if not isinstance(slug, str) or not slug:
        return None
    m = re.match(r"^([a-z0-9]+)(?:-close)?-above", slug)
    if m:
        return m.group(1)
    return None


def _parse_threshold(question: str) -> Optional[float]:
    if not isinstance(question, str):
        return None

    def _to_float(value: str) -> Optional[float]:
        try:
            return float(value.replace(",", ""))
        except Exception:
            return None

    m = re.search(r"\$([0-9][0-9,]*(?:\.\d+)?)", question)
    if m:
        return _to_float(m.group(1))
    m = re.search(r"\babove\s+([0-9][0-9,]*(?:\.\d+)?)\b", question, flags=re.IGNORECASE)
    if m:
        return _to_float(m.group(1))
    return None


def _parse_date_from_slug(slug: str) -> Optional[date]:
    if not isinstance(slug, str) or not slug:
        return None
    m = re.search(r"(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|"
                  r"september|sep|sept|october|oct|november|nov|december|dec)-(\d{1,2})-(\d{4})",
                  slug, flags=re.IGNORECASE)
    if not m:
        return None
    month = MONTHS.get(m.group(1).lower())
    day = int(m.group(2))
    year = int(m.group(3))
    if not month:
        return None
    try:
        return date(year, month, day)
    except Exception:
        return None


def _parse_date_from_question(question: str) -> Optional[date]:
    if not isinstance(question, str) or not question:
        return None

    # Pattern 1: Standard date format "February 14, 2025" or "Feb 14th, 2025"
    m = re.search(
        r"(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|"
        r"september|sep|sept|october|oct|november|nov|december|dec)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+(\d{4})",
        question,
        flags=re.IGNORECASE,
    )
    if m:
        month = MONTHS.get(m.group(1).lower())
        day = int(m.group(2))
        year = int(m.group(3))
        if month:
            try:
                return date(year, month, day)
            except Exception:
                pass

    # Pattern 2: Week patterns - "week of February 10-14, 2025" or "week ending February 14, 2025"
    # Matches "week of/ending [month] [day][-day], [year]"
    m = re.search(
        r"week\s+(?:of|ending)\s+(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|"
        r"september|sep|sept|october|oct|november|nov|december|dec)\s+(\d{1,2})(?:\-\d{1,2})?(?:,)?\s+(\d{4})",
        question,
        flags=re.IGNORECASE,
    )
    if m:
        month = MONTHS.get(m.group(1).lower())
        day = int(m.group(2))  # Use first date in range as expiry
        year = int(m.group(3))
        if month:
            try:
                return date(year, month, day)
            except Exception:
                pass

    return None


def _infer_ticker(
    question: str,
    slug: str,
    allowlist: Optional[List[str]],
) -> Tuple[Optional[str], str]:
    allowset = {t.upper() for t in allowlist} if allowlist else set()
    slug_prefix = _parse_slug_prefix(slug)

    # Priority 1: Match slug prefix to ticker allowlist
    if slug_prefix and allowlist:
        slug_map = {_slugify_ticker(t): t.upper() for t in allowlist}
        if slug_prefix in slug_map:
            return slug_map[slug_prefix], "slug"

    # Priority 2: Direct ticker symbol match in question
    if allowlist and question:
        q = question.upper()
        for t in allowlist:
            if re.search(rf"\b{re.escape(t.upper())}\b", q):
                return t.upper(), "question"

    # Priority 3: Company name match (new enhancement)
    if allowlist and question:
        q_lower = question.lower()
        for company_name, ticker in COMPANY_NAME_MAP.items():
            # Only match if the ticker is in the allowlist
            if ticker.upper() in allowset and re.search(rf"\b{re.escape(company_name)}\b", q_lower):
                return ticker.upper(), "company_name"

    # Priority 4: Slug prefix fallback (no allowlist validation)
    if slug_prefix:
        return slug_prefix.upper(), "slug_fallback"

    # Priority 5: Extract uppercase tokens from question
    if question:
        tokens = re.findall(r"\b[A-Z]{1,6}\b", question)
        if tokens:
            return tokens[0].upper(), "question_fallback"

    return None, "none"


def _utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _expiry_iso(d: Optional[date]) -> Optional[str]:
    if not d:
        return None
    return _utc_iso(datetime.combine(d, dt_time(0, 0), tzinfo=timezone.utc))


def _resolution_iso(d: Optional[date]) -> Optional[str]:
    if not d:
        return None
    return _utc_iso(datetime.combine(d, dt_time(23, 59, 59), tzinfo=timezone.utc))


def _parse_resolved_time(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            ts = float(value)
            if ts > 1e12:
                ts /= 1000.0
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            return _utc_iso(dt)
        if isinstance(value, str):
            s = value.strip()
            if s and re.fullmatch(r"\d+(?:\.\d+)?", s):
                ts = float(s)
                if ts > 1e12:
                    ts /= 1000.0
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                return _utc_iso(dt)
        dt = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(dt):
            return None
        dt = dt.to_pydatetime()
        return _utc_iso(dt)
    except Exception:
        return None


def _compute_confidence(
    ticker: Optional[str],
    threshold: Optional[float],
    expiry_date: Optional[date],
    source: str,
) -> float:
    if source == "manual":
        return 1.0
    score = 0.0
    if ticker:
        score += 0.4
    if threshold is not None:
        score += 0.4
    if expiry_date:
        score += 0.2
    return min(0.9, score)


# ----------------------------
# Data loading
# ----------------------------

def _load_overrides(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df.columns = [c.strip() for c in df.columns]
    return df


def _load_prn_tickers(path: Optional[Path]) -> Optional[List[str]]:
    if not path or not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["ticker"])
    except ValueError as exc:
        print(f"[dim_market] Warning: pRN dataset missing 'ticker' column: {path} ({exc})")
        return None
    except Exception as exc:
        print(f"[dim_market] Warning: Failed to read pRN dataset {path}: {exc}")
        return None
    tickers = sorted({str(t).upper() for t in df["ticker"].dropna().unique() if str(t).strip()})
    return tickers or None


def _load_gamma_markets(run_id: Optional[str] = None) -> Tuple[List[dict], str]:
    """Fetch markets from Gamma API as a fallback when subgraph is unavailable."""
    import requests
    from uuid import uuid4

    GAMMA_URL = "https://gamma-api.polymarket.com/markets"
    GAMMA_PAGE_SIZE = 1000
    GAMMA_MAX_PAGES = 2000

    print("[dim_market] Fetching markets from Gamma API...")

    entities: List[dict] = []
    offset = 0

    with requests.Session() as session:
        for page_idx in range(GAMMA_MAX_PAGES):
            params = {
                "limit": GAMMA_PAGE_SIZE,
                "offset": offset,
                "closed": "false",  # Only fetch active (non-closed) markets
            }
            data = None
            for attempt in range(3):
                try:
                    resp = session.get(GAMMA_URL, params=params, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception as exc:
                    if attempt == 2:
                        print(f"[dim_market] Warning: Gamma API request failed after 3 attempts: {exc}")
                    else:
                        time.sleep(0.5 * (2 ** attempt))
            if data is None:
                break

            if not isinstance(data, list):
                print(f"[dim_market] Warning: Unexpected Gamma API response format")
                break

            # Normalize Gamma API response to match subgraph format
            for item in data:
                normalized = {
                    "id": item.get("id"),
                    "conditionId": item.get("conditionId") or item.get("condition_id"),
                    "question": item.get("question") or item.get("title") or "",
                    "slug": item.get("slug") or "",
                    "resolvedTime": item.get("resolvedTime") or item.get("resolvedAt"),
                    "createdAt": item.get("createdAt"),
                    "outcomeTokenIds": item.get("clobTokenIds") or item.get("outcomeTokenIds"),
                    "clobTokenIds": item.get("clobTokenIds"),
                    "endDate": item.get("endDate") or item.get("endDateIso"),
                }
                entities.append(normalized)

            print(f"[dim_market] Fetched page {page_idx + 1}: {len(data)} markets (total: {len(entities)})")

            if len(data) < GAMMA_PAGE_SIZE:
                break
            offset += GAMMA_PAGE_SIZE

    run_id = run_id or f"gamma-markets-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    print(f"[dim_market] Gamma API fetch complete: {len(entities)} markets")
    return entities, f"gamma:{run_id}"


def _load_subgraph_entities(cfg: Config) -> Tuple[List[dict], Optional[str]]:
    if cfg.run_dir:
        from polymarket.subgraph_client import SubgraphClient as _Client
        entities = _Client.entities_from_run(cfg.run_dir)
        return entities, str(cfg.run_dir)

    # Try to pull from subgraph, fallback to Gamma API if markets query not available
    try:
        client = SubgraphClient()
        result = client.pull("markets", run_id=cfg.run_id)
        entities = client.entities_from_run(result.run_dir)
        return entities, str(result.run_dir)
    except (RuntimeError, ValueError) as exc:
        # If markets query fails (e.g., GraphQL schema doesn't have markets field),
        # fallback to fetching from Gamma API
        error_msg = str(exc).lower()
        if "graphql" in error_msg or "markets" in error_msg or "field" in error_msg:
            print(f"[dim_market] Subgraph markets query unavailable, using Gamma API fallback: {exc}")
            return _load_gamma_markets(cfg.run_id)
        raise


# ----------------------------
# Overrides
# ----------------------------

def _apply_overrides(df: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    if overrides.empty:
        return df

    overrides = overrides.copy()
    overrides = overrides.replace({np.nan: None})

    by_market: Dict[str, Dict[str, Any]] = {}
    by_condition: Dict[str, Dict[str, Any]] = {}
    by_slug: Dict[str, Dict[str, Any]] = {}

    for _, row in overrides.iterrows():
        record = {k: row[k] for k in row.index}
        market_id = str(record.get("market_id") or "").strip()
        condition_id = str(record.get("condition_id") or "").strip()
        slug = str(record.get("slug") or "").strip()
        if market_id:
            by_market[market_id] = record
        if condition_id:
            by_condition[condition_id] = record
        if slug:
            by_slug[slug] = record

    def _merge_row(row: pd.Series) -> pd.Series:
        record = None
        if row.get("market_id") in by_market:
            record = by_market[row.get("market_id")]
        elif row.get("condition_id") in by_condition:
            record = by_condition[row.get("condition_id")]
        elif row.get("slug") in by_slug:
            record = by_slug[row.get("slug")]
        if record:
            for key, val in record.items():
                if key in row.index and val is not None and val != "":
                    row[key] = val
            row["source"] = "manual"
            if not record.get("mapping_confidence"):
                row["mapping_confidence"] = 1.0
        return row

    return df.apply(_merge_row, axis=1)


# ----------------------------
# Main build
# ----------------------------

def build_dim_market(
    entities: Iterable[dict],
    allowlist: Optional[List[str]],
    overrides: pd.DataFrame,
    strict: bool,
) -> pd.DataFrame:
    rows: List[dict] = []

    for m in entities:
        question = str(m.get("question") or "").strip()
        slug = str(m.get("slug") or "").strip()
        market_id = str(m.get("id") or "").strip() or None
        condition_id = str(m.get("conditionId") or m.get("condition_id") or "").strip() or None

        ticker, ticker_source = _infer_ticker(question, slug, allowlist)
        threshold = _parse_threshold(question)

        expiry_date = _parse_date_from_question(question) or _parse_date_from_slug(slug)
        expiry_date_utc = _expiry_iso(expiry_date)

        resolved_time = _parse_resolved_time(m.get("resolvedTime"))
        resolution_time_utc = resolved_time or _resolution_iso(expiry_date)

        token_ids = m.get("outcomeTokenIds")
        token_ids = token_ids if isinstance(token_ids, list) else []
        yes_token = str(token_ids[0]) if len(token_ids) >= 1 and token_ids[0] else None
        no_token = str(token_ids[1]) if len(token_ids) >= 2 and token_ids[1] else None

        source = "auto"
        confidence = _compute_confidence(ticker, threshold, expiry_date, source)

        rows.append(
            {
                "market_id": market_id,
                "condition_id": condition_id,
                "question": question or None,
                "ticker": ticker,
                "threshold": threshold,
                "expiry_date_utc": expiry_date_utc,
                "resolution_time_utc": resolution_time_utc,
                "outcome_yes_token_id": yes_token,
                "outcome_no_token_id": no_token,
                "slug": slug or None,
                "source": source,
                "mapping_confidence": confidence,
                "ticker_source": ticker_source,
                "schema_version": SCHEMA_VERSION,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = _apply_overrides(df, overrides)

    if allowlist:
        allowset = {t.upper() for t in allowlist}
        df = df[df["ticker"].isin(allowset)].copy()

    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    df["mapping_confidence"] = pd.to_numeric(df["mapping_confidence"], errors="coerce")

    if strict:
        missing = df[df["ticker"].isna() | df["threshold"].isna()]
        if not missing.empty:
            sample = missing[["market_id", "condition_id", "slug", "question"]].head(5)
            raise ValueError(
                "Failed to parse ticker/threshold for some markets and no override exists. "
                f"Sample:\n{sample.to_string(index=False)}"
            )

    # Column order (spec)
    cols = [
        "market_id",
        "condition_id",
        "question",
        "ticker",
        "threshold",
        "expiry_date_utc",
        "resolution_time_utc",
        "outcome_yes_token_id",
        "outcome_no_token_id",
        "slug",
        "source",
        "mapping_confidence",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = None

    extra_cols = [c for c in df.columns if c not in cols]
    df = df[cols + extra_cols]

    return df


# ----------------------------
# IO helpers
# ----------------------------

def _write_output(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(out_path, index=False)
            return out_path
        except Exception:
            csv_path = out_path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            return csv_path
    df.to_csv(out_path, index=False)
    return out_path


def _find_latest_prn_dataset() -> Optional[Path]:
    base = REPO_ROOT / "src" / "data" / "raw" / "option-chain"
    candidates = list(base.rglob("dataset-n1.csv"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dim_market mapping from Polymarket subgraph markets.")
    parser.add_argument("--run-dir", type=str, default=None, help="Use an existing subgraph run dir instead of fetching.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run_id when fetching markets.")
    parser.add_argument("--overrides", type=str, default=str(DEFAULT_OVERRIDES_PATH), help="CSV overrides file.")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated ticker allowlist.")
    parser.add_argument("--prn-dataset", type=str, default=None, help="Path to pRN dataset CSV (to infer tickers).")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_PATH), help="Output dim_market path (parquet or csv).")
    parser.add_argument("--strict", action="store_true", help="Fail if any target market lacks ticker or threshold.")
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True)
    parser.add_argument("--profile", action="store_true", help="Enable profiling mode with detailed timing.")
    parser.add_argument("--profile-output", type=str, default=None, help="Write cProfile stats to file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        out_path=Path(args.out),
        overrides_path=Path(args.overrides),
        run_dir=Path(args.run_dir) if args.run_dir else None,
        run_id=args.run_id,
        prn_dataset=Path(args.prn_dataset) if args.prn_dataset else None,
        strict=args.strict,
        profile=args.profile,
        profile_output=Path(args.profile_output) if args.profile_output else None,
    )

    # Initialize profiling
    stats = ProfileStats() if cfg.profile else None
    profiler = cProfile.Profile() if cfg.profile_output else None
    if profiler:
        profiler.enable()

    tickers = None
    if args.tickers:
        # Explicit tickers provided via command line
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        print(f"[dim_market] Using explicit ticker list: {','.join(tickers)}")
    else:
        # Auto-load tickers from pRN dataset (trading universe)
        with ProfileContext(stats, "load_prn_tickers"):
            prn_path = cfg.prn_dataset or _find_latest_prn_dataset()
            if prn_path:
                tickers = _load_prn_tickers(prn_path)
                if tickers:
                    print(f"[dim_market] Auto-loaded {len(tickers)} tickers from trading universe: {prn_path}")
                else:
                    print(f"[dim_market] No tickers found in {prn_path}")
            else:
                print("[dim_market] No pRN dataset found; processing all markets.")

    cfg.tickers = tickers

    with ProfileContext(stats, "load_overrides"):
        overrides = _load_overrides(cfg.overrides_path)
    if stats and not overrides.empty:
        stats.increment_count("override_rows", len(overrides))

    if not cfg.tickers and cfg.strict:
        print("[dim_market] No ticker allowlist available; disabling strict mode.")
        cfg.strict = False

    with ProfileContext(stats, "fetch_markets"):
        entities, source_run = _load_subgraph_entities(cfg)
    if stats:
        stats.api_calls += 1
        stats.increment_count("markets_fetched", len(entities))
    if not entities:
        print("[dim_market] No markets returned from subgraph.")
        return

    with ProfileContext(stats, "build_dim_market"):
        dim_market = build_dim_market(entities, cfg.tickers, overrides, cfg.strict)

    if dim_market.empty:
        print("[dim_market] 0 rows after filtering.")
        return

    with ProfileContext(stats, "write_output"):
        out_path = _write_output(dim_market, cfg.out_path)
    if stats:
        stats.increment_count("dim_market_rows", len(dim_market))

    print("[dim_market] rows=", len(dim_market))
    print("[dim_market] output=", out_path)
    if source_run:
        print("[dim_market] source_run=", source_run)
    if cfg.tickers:
        print("[dim_market] tickers=", ",".join(cfg.tickers))

    # Profiling summary
    if profiler:
        profiler.disable()
        profile_stats = pstats.Stats(profiler)
        profile_stats.dump_stats(str(cfg.profile_output))
        print(f"[PROFILE] cProfile stats written to {cfg.profile_output}")

    if stats:
        stats.print_summary()


if __name__ == "__main__":
    main()
