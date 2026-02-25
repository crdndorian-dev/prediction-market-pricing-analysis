"""
Step 0 – Truth Layers and Anti-Leak Contract
=============================================

Two truth sources are defined and must **never** be mixed:

1. **Settlement truth**: on-chain events (fills, positions, liquidity,
   resolution).  These are immutable facts recorded on the blockchain.

2. **Decision truth**: best bid/ask or mid at the moment a trading
   decision is evaluated (CLOB).  When historical orderbook data is
   unavailable, use trade-derived mid with a conservative slippage model.

Anti-leak rules enforced by this module
---------------------------------------
* All features for a decision time ``t`` must be computed using data
  **strictly before** ``t``.
* Labels use resolution outcome at expiry / settlement time only.
* Backtests must include latency and spread/slippage.

Every downstream script that builds features, trains models, or runs
backtests should import and use the validators below.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Truth layer enum
# ---------------------------------------------------------------------------

class TruthLayer(str, Enum):
    """Identifies which truth source a column/row originates from."""
    SETTLEMENT = "settlement"
    DECISION   = "decision"


# ---------------------------------------------------------------------------
# Slippage defaults (conservative)
# ---------------------------------------------------------------------------

DEFAULT_SLIPPAGE_BPS: int = 200          # 2 % one-way
DEFAULT_FEE_BPS: int = 100               # 1 % taker fee
DEFAULT_LATENCY_MS: int = 2_000          # 2 s latency buffer
MIN_SPREAD_FOR_TRADE: float = 0.0        # spread floor (0 = any)
MIN_LIQUIDITY_USD_FOR_TRADE: float = 50  # skip illiquid markets


# ---------------------------------------------------------------------------
# Feature-time validators
# ---------------------------------------------------------------------------

def validate_feature_times(
    df: pd.DataFrame,
    decision_col: str = "timestamp_utc",
    feature_time_cols: list[str] | None = None,
) -> pd.Series:
    """Return a boolean Series that is True where *all* feature timestamps
    are strictly before the decision timestamp.

    Parameters
    ----------
    df : pd.DataFrame
        The feature dataset.
    decision_col : str
        Column holding the decision timestamp (must be tz-aware UTC or
        tz-naive interpreted as UTC).
    feature_time_cols : list[str] | None
        Columns whose values are timestamps that must precede *decision_col*.
        If ``None``, every column whose name ends with ``_ts`` or ``_time``
        is checked automatically.

    Returns
    -------
    pd.Series[bool]
        True where the row passes the leak check.
    """
    if decision_col not in df.columns:
        raise KeyError(f"Decision column '{decision_col}' not in DataFrame")

    t_decision = pd.to_datetime(df[decision_col], utc=True)

    if feature_time_cols is None:
        feature_time_cols = [
            c for c in df.columns
            if c != decision_col and (c.endswith("_ts") or c.endswith("_time"))
        ]

    if not feature_time_cols:
        # No feature-time columns detected → trivially passes
        return pd.Series(True, index=df.index)

    mask = pd.Series(True, index=df.index)
    for col in feature_time_cols:
        t_feat = pd.to_datetime(df[col], utc=True)
        mask &= t_feat < t_decision

    return mask


def assert_no_leaks(
    df: pd.DataFrame,
    decision_col: str = "timestamp_utc",
    feature_time_cols: list[str] | None = None,
    label: str = "dataset",
) -> None:
    """Raise ``ValueError`` if any row violates the anti-leak contract."""
    passed = validate_feature_times(df, decision_col, feature_time_cols)
    n_bad = int((~passed).sum())
    if n_bad:
        raise ValueError(
            f"Anti-leak violation in '{label}': {n_bad:,} / {len(df):,} rows "
            f"have feature timestamps >= decision time."
        )


# ---------------------------------------------------------------------------
# Label attachment validator
# ---------------------------------------------------------------------------

def validate_labels(
    df: pd.DataFrame,
    decision_col: str = "timestamp_utc",
    resolution_col: str = "resolution_time_utc",
) -> pd.Series:
    """Return True where labels are attached only *after* resolution.

    This ensures labels come from settlement truth and are not available
    at decision time.
    """
    t_decision = pd.to_datetime(df[decision_col], utc=True)
    t_resolution = pd.to_datetime(df[resolution_col], utc=True)
    return t_decision < t_resolution


# ---------------------------------------------------------------------------
# Slippage helpers
# ---------------------------------------------------------------------------

def apply_slippage(
    price: float | np.ndarray,
    side: str,
    slippage_bps: int = DEFAULT_SLIPPAGE_BPS,
    fee_bps: int = DEFAULT_FEE_BPS,
) -> float | np.ndarray:
    """Return the execution price after applying slippage + fees.

    Parameters
    ----------
    price : float or array
        Quoted mid or last price (0-1 scale for prediction markets).
    side : str
        ``"buy"`` or ``"sell"``.
    slippage_bps : int
        One-way slippage in basis points.
    fee_bps : int
        Taker fee in basis points.

    Returns
    -------
    float or array
        Adjusted execution price.
    """
    slip = slippage_bps / 10_000
    fee  = fee_bps / 10_000
    if side == "buy":
        return np.clip(price * (1 + slip + fee), 0, 1)
    elif side == "sell":
        return np.clip(price * (1 - slip - fee), 0, 1)
    else:
        raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")


def apply_latency_buffer(
    decision_time: dt.datetime,
    latency_ms: int = DEFAULT_LATENCY_MS,
) -> dt.datetime:
    """Return the effective execution time after adding latency."""
    return decision_time + dt.timedelta(milliseconds=latency_ms)


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def contract_summary() -> str:
    """Return a human-readable summary of the anti-leak contract."""
    return (
        "Anti-Leak Contract\n"
        "==================\n"
        "1. Features for decision time t use data strictly before t.\n"
        "2. Labels use resolution outcome only (settlement truth).\n"
        "3. Backtests apply latency, spread, and slippage.\n"
        f"   - Default slippage : {DEFAULT_SLIPPAGE_BPS} bps\n"
        f"   - Default fee      : {DEFAULT_FEE_BPS} bps\n"
        f"   - Default latency  : {DEFAULT_LATENCY_MS} ms\n"
    )
