from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


RUN_TRADES_SCHEMA_VERSION = "pm_run_trades_v1.0"
RUN_TRADES_FILENAME = "trades.csv"
RUN_TRADES_COLUMNS = [
    "trade_id",
    "block_number",
    "timestamp_utc",
    "market_id",
    "event_id",
    "ticker",
    "threshold",
    "week_friday",
    "event_endDate",
    "outcome_token_id",
    "token_role",
    "price",
    "size",
    "notional",
    "side",
    "tx_hash",
    "schema_version",
]


def _clean_string(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .replace("", pd.NA)
    )


def _series_or_empty(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")


def enrich_trades_for_run(trades: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """Join normalized subgraph trades to weekly market metadata for run-local analytics."""
    if trades.empty or markets.empty:
        return pd.DataFrame(columns=RUN_TRADES_COLUMNS)

    trades_df = trades.copy()
    trades_df["trade_id"] = _clean_string(_series_or_empty(trades_df, "trade_id"))
    trades_df["market_id"] = _clean_string(_series_or_empty(trades_df, "market_id"))
    trades_df["outcome_token_id"] = _clean_string(_series_or_empty(trades_df, "outcome_token_id"))
    trades_df["side"] = _clean_string(_series_or_empty(trades_df, "side"))
    trades_df["tx_hash"] = _clean_string(_series_or_empty(trades_df, "tx_hash"))
    trades_df["block_number"] = pd.to_numeric(_series_or_empty(trades_df, "block_number"), errors="coerce")
    trades_df["price"] = pd.to_numeric(_series_or_empty(trades_df, "price"), errors="coerce")
    trades_df["size"] = pd.to_numeric(_series_or_empty(trades_df, "size"), errors="coerce")
    trades_df["timestamp_utc"] = pd.to_datetime(
        _series_or_empty(trades_df, "timestamp_utc"),
        utc=True,
        errors="coerce",
    )
    trades_df = trades_df.dropna(
        subset=[
            "trade_id",
            "market_id",
            "outcome_token_id",
            "timestamp_utc",
            "price",
            "size",
        ]
    ).copy()
    trades_df = trades_df.drop_duplicates(subset=["trade_id"], keep="first")

    if trades_df.empty:
        return pd.DataFrame(columns=RUN_TRADES_COLUMNS)

    markets_df = markets.copy()
    markets_df["market_id"] = _clean_string(_series_or_empty(markets_df, "market_id"))
    markets_df["yes_token_id"] = _clean_string(_series_or_empty(markets_df, "yes_token_id"))
    markets_df["no_token_id"] = _clean_string(_series_or_empty(markets_df, "no_token_id"))
    market_cols = [
        "market_id",
        "event_id",
        "ticker",
        "threshold",
        "week_friday",
        "event_endDate",
        "yes_token_id",
        "no_token_id",
    ]
    available_cols = [col for col in market_cols if col in markets_df.columns]
    markets_df = markets_df[available_cols].dropna(subset=["market_id"]).copy()
    markets_df = markets_df.drop_duplicates(subset=["market_id"], keep="first")

    out = trades_df.merge(markets_df, on="market_id", how="left")
    out["token_role"] = np.where(
        out["outcome_token_id"] == out.get("yes_token_id"),
        "yes",
        np.where(out["outcome_token_id"] == out.get("no_token_id"), "no", pd.NA),
    )
    out["notional"] = out["price"] * out["size"]
    out["timestamp_utc"] = out["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["schema_version"] = RUN_TRADES_SCHEMA_VERSION
    return out.reindex(columns=RUN_TRADES_COLUMNS)


def build_trade_artifact_metadata(
    trades: pd.DataFrame,
    *,
    path: str = RUN_TRADES_FILENAME,
) -> Optional[Dict[str, Any]]:
    if trades.empty:
        return None
    timestamps = pd.to_datetime(trades["timestamp_utc"], utc=True, errors="coerce")
    trade_days = int(timestamps.dt.strftime("%Y-%m-%d").nunique(dropna=True))
    return {
        "path": path,
        "rows": int(len(trades)),
        "trade_days": trade_days,
        "schema_version": RUN_TRADES_SCHEMA_VERSION,
    }


def write_run_trades_csv(trades: pd.DataFrame, path: Path) -> Optional[Dict[str, Any]]:
    if trades.empty:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(path, index=False)
    return build_trade_artifact_metadata(trades, path=path.name)
