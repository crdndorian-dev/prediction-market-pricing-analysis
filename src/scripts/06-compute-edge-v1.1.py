#!/usr/bin/env python3
"""
06-compute-edge-v1.1.py

Load a saved calibrator artifact, enrich a Polymarket snapshot with the same feature pipeline used in training,
compute pHAT for each row, and emit diagnostics/edges.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration.calibrate_common import (
    FinalModelBundle,
    dedupe_preserve_order,
    ensure_engineered_features,
)

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def add_asof_dow_column(snapshot: pd.DataFrame, asof_col: str) -> None:
    dt = pd.to_datetime(snapshot[asof_col], errors="coerce", utc=True)
    if dt.isna().any():
        bad = int(dt.isna().sum())
        raise ValueError(f"{asof_col} has {bad} NaT values; cannot derive asof_dow.")
    snapshot["asof_dow"] = dt.dt.weekday.map(lambda idx: DAY_NAMES[int(idx)])

EDGE_COLUMN_PRIORITY = [
    ("pPM_buy", "yes contract"),
    ("qPM_buy", "no contract"),
]


def parse_bool_flag(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean flag: {value}")


def parse_exclude_tickers(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [ticker.strip().upper() for ticker in value.split(",") if ticker.strip()]


def load_artifact(model_path: Path) -> (FinalModelBundle, Dict[str, Any]):
    raw = joblib.load(model_path)
    if isinstance(raw, dict):
        bundle = raw.get("bundle")
        metadata = raw.get("metadata", {})
    else:
        bundle = raw
        metadata = {}
    if not isinstance(bundle, FinalModelBundle):
        raise ValueError("Model artifact did not resolve to a FinalModelBundle.")
    return bundle, metadata


def resolve_prn_train_range(metadata: Dict[str, Any]) -> tuple[Optional[float], Optional[float], Optional[str]]:
    for min_key, max_key in (
        ("min_prn_train", "max_prn_train"),
        ("prn_train_min", "prn_train_max"),
    ):
        min_val = metadata.get(min_key)
        max_val = metadata.get(max_key)
        if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
            if 0.0 <= float(min_val) < float(max_val) <= 1.0:
                return float(min_val), float(max_val), f"metadata:{min_key}/{max_key}"
    range_val = metadata.get("prn_train_range")
    if (
        isinstance(range_val, (list, tuple))
        and len(range_val) == 2
        and all(isinstance(v, (int, float)) for v in range_val)
    ):
        min_val, max_val = float(range_val[0]), float(range_val[1])
        if 0.0 <= min_val < max_val <= 1.0:
            return min_val, max_val, "metadata:prn_train_range"

    opt = metadata.get("optional_filters") if isinstance(metadata.get("optional_filters"), dict) else None
    if opt and opt.get("drop_prn_extremes"):
        eps = opt.get("prn_eps", 1e-4)
        if isinstance(eps, (int, float)) and 0.0 < float(eps) < 0.5:
            eps = float(eps)
            return eps, 1.0 - eps, "metadata:optional_filters"
    return None, None, None


def ensure_required_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    strict: bool,
) -> List[str]:
    missing = [col for col in required_columns if col not in df.columns]
    if not missing:
        return []
    msg = f"Missing required columns in snapshot: {missing}"
    print(f"[WARN] {msg}")
    for col in missing:
        if col not in df.columns:
            df[col] = np.nan
    if strict:
        raise ValueError(msg)
    return missing


def describe_top_edges(
    df: pd.DataFrame,
    exclude: List[str],
    top_n: int = 10,
) -> None:
    if "edge" not in df.columns:
        print("[INFO] Edge column not available, skipping edge summary.")
        return
    subset = df.dropna(subset=["edge"]).copy()
    if not subset.empty and exclude:
        subset = subset[~subset["ticker"].str.upper().isin(exclude)]
    if subset.empty:
        print("[INFO] No rows with edge data after filtering.")
        return
    top_edges = subset.sort_values("edge", ascending=False).head(top_n)
    cols = ["ticker", "K", "pHAT"]
    if "pRN" in top_edges.columns:
        cols.append("pRN")
    if "pPM_buy" in top_edges.columns:
        cols.append("pPM_buy")
    if "qHAT" in top_edges.columns:
        cols.append("qHAT")
    if "qRN" in top_edges.columns:
        cols.append("qRN")
    if "edge_source" in top_edges.columns:
        cols.append("edge_source")
    for extra in ["qPM_buy"]:
        if extra in top_edges.columns:
            cols.append(extra)
    if "edge" in top_edges.columns:
        cols.append("edge")
    cols = [c for c in cols if c in top_edges.columns]
    header = "Top edges"
    if "edge_source" in top_edges.columns:
        sources = top_edges["edge_source"].dropna().astype(str).unique()
        if len(sources) > 0:
            header = f"{header} (computed vs {', '.join(sources)})"
    print(f"\n=== {header} ===")
    print(top_edges[cols].to_string(index=False))


def summarize_distribution(values: pd.Series) -> None:
    stats = values.dropna()
    if stats.empty:
        print("[INFO] pHAT series is empty.")
        return
    print("\n=== pHAT distribution ===")
    print(f"count: {int(stats.count())}")
    print(f"mean : {float(stats.mean()):.5f}")
    print(f"min  : {float(stats.min()):.5f}")
    print(f"max  : {float(stats.max()):.5f}")


def build_required_columns(
    manifest: Optional[Dict[str, Any]],
    bundle: FinalModelBundle,
) -> List[str]:
    requested = []
    if manifest and isinstance(manifest.get("required_columns"), list):
        requested.extend(manifest["required_columns"])
    else:
        requested.extend(bundle.numeric_features)
        requested.extend(bundle.categorical_features)
        requested.append(bundle.ticker_col)
        if bundle.ticker_feature_col:
            requested.append(bundle.ticker_feature_col)
        if bundle.interaction_ticker_col:
            requested.append(bundle.interaction_ticker_col)
        requested.extend(["pRN", "pRN_raw", "K", "S", "T_days", "event_endDate", "snapshot_time_utc"])
    return dedupe_preserve_order([str(col) for col in requested if col])


def log_snapshot_sanity(snapshot: pd.DataFrame) -> None:
    rows = len(snapshot)
    cols = snapshot.shape[1]
    print(f"[SnapshotSanity] rows={rows}, columns={cols}")
    if "ticker" in snapshot.columns:
        unique_tickers = snapshot["ticker"].nunique(dropna=True)
        print(f"[SnapshotSanity] unique tickers={unique_tickers}")
    if "pPM_buy" in snapshot.columns:
        coverage = float(snapshot["pPM_buy"].notna().mean())
        print(f"[SnapshotSanity] pPM_buy available in {coverage:.1%} of rows.")
    nan_frac = snapshot.isna().mean()
    nan_frac = nan_frac[nan_frac > 0]
    if not nan_frac.empty:
        high_nan = nan_frac.sort_values(ascending=False).head(5)
        print("[SnapshotSanity] highest NaN fractions:")
        for col, frac in high_nan.items():
            print(f"  {col}: {frac:.1%}")


def _run_main() -> None:
    parser = argparse.ArgumentParser(description="Apply pHAT calibrator over a Polymarket snapshot.")
    parser.add_argument("--model-path", required=True, type=Path, help="Path to joblib artifact with bundle + metadata.")
    parser.add_argument("--snapshot-csv", required=True, type=Path, help="Polymarket snapshot CSV path.")
    parser.add_argument("--out-csv", required=True, type=Path, help="Path to write the enriched CSV.")
    parser.add_argument("--exclude-tickers", type=str, default="", help="Comma-separated tickers to exclude from summaries.")
    parser.add_argument("--require-columns-strict", type=parse_bool_flag, default=True, help="True to fail when required columns missing.")
    parser.add_argument("--compute-edge", type=parse_bool_flag, default=True, help="Compute pHAT - buy_price when buy side pricing is available.")
    parser.add_argument(
        "--skip-edge-outside-prn-range",
        type=parse_bool_flag,
        default=True,
        help="Skip edge when pRN falls outside the model's training range (if available).",
    )
    args = parser.parse_args()

    print(
        "[INFO] Starting pHAT inference",
        f"model={args.model_path}",
        f"snapshot={args.snapshot_csv}",
        f"require_columns_strict={args.require_columns_strict}",
    )

    bundle, metadata = load_artifact(args.model_path)
    feature_manifest = metadata.get("feature_manifest") if isinstance(metadata.get("feature_manifest"), dict) else None
    numeric_features = (
        feature_manifest.get("numeric_features") if feature_manifest and isinstance(feature_manifest.get("numeric_features"), list) else bundle.numeric_features
    )
    if not numeric_features:
        numeric_features = bundle.numeric_features
    categorical_features = (
        feature_manifest.get("categorical_features") if feature_manifest and isinstance(feature_manifest.get("categorical_features"), list) else bundle.categorical_features
    )
    if not categorical_features:
        categorical_features = bundle.categorical_features

    snapshot = pd.read_csv(args.snapshot_csv, parse_dates=["snapshot_time_utc", "event_endDate"], low_memory=False)
    snapshot["ticker"] = snapshot["ticker"].astype("string").fillna("UNKNOWN")
    if "pm_mid" not in snapshot.columns and "pPM_mid" in snapshot.columns:
        snapshot["pm_mid"] = pd.to_numeric(snapshot["pPM_mid"], errors="coerce")

    asof_date_col = None
    if feature_manifest and isinstance(feature_manifest.get("asof_date_col"), str):
        asof_date_col = feature_manifest.get("asof_date_col")
    if "asof_dow" in categorical_features:
        if not asof_date_col:
            asof_date_col = "snapshot_time_utc" if "snapshot_time_utc" in snapshot.columns else None
        if not asof_date_col:
            raise ValueError("Cannot derive asof_dow: asof_date column missing in snapshot.")
        add_asof_dow_column(snapshot, asof_date_col)

    log_snapshot_sanity(snapshot)

    # Validate snapshot schema against baseline contract
    print("\n=== VALIDATING SNAPSHOT SCHEMA ===")
    try:
        from calibration.calibrate_common import validate_snapshot_schema
        all_required_features = list(set(
            list(numeric_features) + list(categorical_features) +
            ["pRN", "K", "S", "T_days", "ticker"]
        ))
        validation_result = validate_snapshot_schema(
            snapshot,
            required_features=all_required_features,
            strict=False  # Don't fail yet, just collect diagnostics
        )

        if not validation_result["valid"]:
            print(f"[ERROR] Snapshot schema validation failed:")
            for err in validation_result["errors"]:
                print(f"  - {err}")

        if validation_result["warnings"]:
            print(f"[WARNING] Snapshot schema issues detected:")
            for warn in validation_result["warnings"]:
                print(f"  - {warn}")

        # Check for critical NaN fractions
        critical_nan = []
        for feat in numeric_features:
            if feat in snapshot.columns:
                nan_frac = snapshot[feat].isna().mean()
                if nan_frac > 0.9:
                    critical_nan.append(f"{feat} ({nan_frac:.1%} NaN)")

        if critical_nan:
            print(f"\n[CRITICAL WARNING] Features with >90% NaN values:")
            for item in critical_nan:
                print(f"  - {item}")
            print(f"\nPredictions may be unreliable due to missing feature data.")
            print(f"Consider using a model trained without these features,")
            print(f"or join historical data to populate missing columns.\n")

    except Exception as e:
        print(f"[WARN] Could not validate snapshot schema: {e}")
        print(f"       Continuing with inference, but results may be unreliable.")

    missing = ensure_required_columns(snapshot, build_required_columns(feature_manifest, bundle), args.require_columns_strict)

    engineered = ensure_engineered_features(snapshot, numeric_features)
    if args.require_columns_strict is False and missing:
        # fill any remaining columns introduced by manifest
        for col in missing:
            if col not in engineered.columns:
                engineered[col] = np.nan

    p_hat = bundle.predict_proba_from_df(engineered)
    result = snapshot.copy()
    result["pHAT"] = p_hat
    result["qHAT"] = 1 - result["pHAT"]

    edge_source_label: Optional[str] = None
    prn_in_range_mask: Optional[pd.Series] = None
    prn_min, prn_max, prn_source = resolve_prn_train_range(metadata)
    if args.skip_edge_outside_prn_range:
        if prn_min is not None and prn_max is not None:
            prn_vals = pd.to_numeric(result.get("pRN"), errors="coerce")
            prn_in_range_mask = prn_vals.notna() & (prn_vals >= prn_min) & (prn_vals <= prn_max)
            out_count = int((~prn_in_range_mask).sum())
            print(
                "[INFO] Edge filter: skipping",
                f"{out_count}/{len(result)} rows outside pRN range [{prn_min:.4f}, {prn_max:.4f}]",
                f"(source={prn_source}).",
            )
        else:
            print(
                "[INFO] Edge filter: pRN training range unavailable; "
                "no rows will be skipped. (Set range in model metadata to enable.)"
            )
    if args.compute_edge:
        yes_edge = None
        no_edge = None
        if "pPM_buy" in result.columns:
            yes_price = pd.to_numeric(result["pPM_buy"], errors="coerce")
            yes_edge = result["pHAT"] - yes_price
        if "qPM_buy" in result.columns:
            no_price = pd.to_numeric(result["qPM_buy"], errors="coerce")
            no_edge = result["qHAT"] - no_price
        if prn_in_range_mask is not None:
            if yes_edge is not None:
                yes_edge = yes_edge.where(prn_in_range_mask, np.nan)
            if no_edge is not None:
                no_edge = no_edge.where(prn_in_range_mask, np.nan)

        if yes_edge is not None and no_edge is not None:
            yes_valid = yes_edge.notna()
            no_valid = no_edge.notna()
            choose_yes = yes_valid & (~no_valid | (yes_edge >= no_edge))
            choose_no = no_valid & (~yes_valid | (no_edge > yes_edge))
            result["edge"] = np.where(choose_yes, yes_edge, np.where(choose_no, no_edge, np.nan))
            edge_source = pd.Series(pd.NA, index=result.index, dtype="string")
            edge_source.loc[choose_yes] = "yes contract"
            edge_source.loc[choose_no] = "no contract"
            result["edge_source"] = edge_source
            edge_source_label = "best of yes/no"
        elif yes_edge is not None:
            result["edge"] = yes_edge
            result["edge_source"] = pd.Series("yes contract", index=result.index, dtype="string")
            edge_source_label = "yes contract"
        elif no_edge is not None:
            result["edge"] = no_edge
            result["edge_source"] = pd.Series("no contract", index=result.index, dtype="string")
            edge_source_label = "no contract"
        else:
            result["edge_source"] = pd.Series(pd.NA, index=result.index, dtype="string")
            print("[INFO] No buy price column found; edge will not be computed.")
        if prn_in_range_mask is not None:
            out_of_range = ~prn_in_range_mask
            if out_of_range.any():
                result.loc[out_of_range, "edge"] = np.nan
                result.loc[out_of_range, "edge_source"] = "pRN_out_of_range"
    else:
        result["edge_source"] = pd.Series(pd.NA, index=result.index, dtype="string")
        print("[INFO] Edge calculation disabled by flag.")

    result["model_version"] = metadata.get("model_version") or args.model_path.stem
    result["asof_timestamp"] = datetime.now(timezone.utc).isoformat()

    exclude = parse_exclude_tickers(args.exclude_tickers)
    summarize_distribution(result["pHAT"])
    if args.compute_edge and "edge" in result.columns:
        describe_top_edges(result, exclude)

    edge_rows = int(result["edge"].notna().sum()) if "edge" in result.columns else 0
    p_hat_rows = int(result["pHAT"].notna().sum())
    ppm_rows = int(result["pPM_buy"].notna().sum()) if "pPM_buy" in result.columns else 0
    print(
        "[Metrics]",
        f"rows={len(result)}",
        f"pHAT_present={p_hat_rows}",
        f"edge_present={edge_rows}",
        f"pPM_buy_present={ppm_rows}",
        f"edge_source={edge_source_label or 'none'}",
        f"missing_columns={len(missing)}",
        f"engineered_features={len(engineered.columns)}",
    )

    result.to_csv(args.out_csv, index=False)
    print(f"\nWritten {len(result)} rows to {args.out_csv}")


def main() -> None:
    try:
        _run_main()
    except Exception as exc:
        print(f"[FATAL] {exc}")
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)


if __name__ == "__main__":
    main()
