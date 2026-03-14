#!/usr/bin/env python3
"""
03-train-mixed-model-v1.0.py

Train a mixed Polymarket + pRN model using the decision dataset.
Supports residual modeling and blending-weight modeling.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "src" / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from polymarket.mixed_model import MixedModelBundle, save_bundle

SCRIPT_VERSION = "1.0.0"
SCHEMA_VERSION = "pm_mixed_model_v1.0"

DEFAULT_FEATURES_PATH = REPO_ROOT / "src" / "data" / "models" / "polymarket" / "decision_features.parquet"
DEFAULT_OUT_DIR = REPO_ROOT / "src" / "data" / "models" / "mixed"

PRN_COL_CANDIDATES = [
    "pRN",
    "pRN_raw",
    "qRN",
    "qRN_raw",
    "rv20",
    "log_m",
    "abs_log_m",
    "log_m_fwd",
    "abs_log_m_fwd",
    "x_logit_prn",
    "T_days",
    "S_asof_close",
    "forward_price",
    "dividend_yield",
]

PM_COL_CANDIDATES = [
    "pm_mid",
    "pm_last",
    "pm_bid",
    "pm_ask",
    "pm_spread",
    "pm_liquidity_proxy",
    "pm_momentum_5m",
    "pm_momentum_1h",
    "pm_momentum_1d",
    "pm_volatility",
    "pm_time_to_resolution",
]


@dataclass
class Config:
    features_path: Path
    out_dir: Path
    run_id: str
    model_type: str
    pm_col: str
    prn_col: str
    label_col: str
    features: Optional[List[str]]
    train_frac: float
    walk_forward: bool
    wf_train_days: int
    wf_test_days: int
    wf_step_days: int
    max_splits: int
    embargo_days: int
    min_time_to_resolution_days: float
    alpha: float


# ----------------------------
# IO helpers
# ----------------------------

def _load_features(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet" and path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path)
            raise
    if path.exists():
        return pd.read_csv(path)

    # fallback to csv if parquet missing
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(f"Decision features not found: {path}")


def _default_feature_cols(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in PM_COL_CANDIDATES + PRN_COL_CANDIDATES:
        if c in df.columns:
            cols.append(c)
    # keep only numeric columns and drop time-like cols
    out = []
    for c in cols:
        if c.endswith("_time") or c.endswith("_ts"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def _prepare_dataset(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()

    df[cfg.label_col] = pd.to_numeric(df.get(cfg.label_col), errors="coerce")
    df = df[df[cfg.label_col].isin([0, 1])]

    df[cfg.pm_col] = pd.to_numeric(df.get(cfg.pm_col), errors="coerce")
    df = df[(df[cfg.pm_col].notna()) & (df[cfg.pm_col] >= 0) & (df[cfg.pm_col] <= 1)]

    if cfg.model_type == "blend":
        df[cfg.prn_col] = pd.to_numeric(df.get(cfg.prn_col), errors="coerce")
        df = df[(df[cfg.prn_col].notna()) & (df[cfg.prn_col] >= 0) & (df[cfg.prn_col] <= 1)]

    if cfg.min_time_to_resolution_days > 0 and "pm_time_to_resolution" in df.columns:
        df = df[df["pm_time_to_resolution"] >= cfg.min_time_to_resolution_days]

    df["timestamp_utc"] = pd.to_datetime(df.get("timestamp_utc"), utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])

    df = df.sort_values("timestamp_utc")
    return df


def _build_target(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    y = df[cfg.label_col].to_numpy(dtype=float)
    pm = df[cfg.pm_col].to_numpy(dtype=float)

    if cfg.model_type == "residual":
        return y - pm

    prn = df[cfg.prn_col].to_numpy(dtype=float)
    denom = prn - pm
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(np.abs(denom) < 1e-6, 0.5, (y - pm) / denom)
    w = np.clip(w, 0.0, 1.0)
    return w


def _build_pipeline(alpha: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("ridge", Ridge(alpha=alpha)),
    ])


def _compute_metrics(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    y_true = y_true.astype(float)
    brier = float(np.mean((p - y_true) ** 2))
    mae = float(np.mean(np.abs(p - y_true)))
    rmse = float(np.sqrt(np.mean((p - y_true) ** 2)))
    logloss = float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
    return {
        "brier": brier,
        "mae": mae,
        "rmse": rmse,
        "logloss": logloss,
    }


def _single_split(df: pd.DataFrame, cfg: Config) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    n = len(df)
    split_idx = int(n * cfg.train_frac)
    split_ts = df["timestamp_utc"].iloc[split_idx]
    start_ts = df["timestamp_utc"].min()
    end_ts = df["timestamp_utc"].max()
    return [(start_ts, split_ts, split_ts, end_ts)]


def _walk_forward_splits(df: pd.DataFrame, cfg: Config) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    splits: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    start = df["timestamp_utc"].min().normalize()
    end = df["timestamp_utc"].max()

    train_delta = timedelta(days=cfg.wf_train_days)
    test_delta = timedelta(days=cfg.wf_test_days)
    step_delta = timedelta(days=cfg.wf_step_days)

    cursor = start
    while cursor + train_delta + test_delta <= end:
        train_start = cursor
        train_end = cursor + train_delta
        test_start = train_end
        test_end = train_end + test_delta
        splits.append((train_start, train_end, test_start, test_end))
        cursor = cursor + step_delta
        if cfg.max_splits and len(splits) >= cfg.max_splits:
            break

    return splits


def _filter_split(df: pd.DataFrame, train_start: pd.Timestamp, train_end: pd.Timestamp, test_start: pd.Timestamp, test_end: pd.Timestamp, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    embargo = timedelta(days=cfg.embargo_days)

    train_mask = (df["timestamp_utc"] >= train_start) & (df["timestamp_utc"] < train_end - embargo)
    test_mask = (df["timestamp_utc"] >= test_start) & (df["timestamp_utc"] < test_end)

    return df[train_mask], df[test_mask]


# ----------------------------
# Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a mixed Polymarket + pRN model.")
    parser.add_argument("--features", type=str, default=str(DEFAULT_FEATURES_PATH), help="Decision features path.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output root directory.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id.")
    parser.add_argument("--model", type=str, default="residual", choices=["residual", "blend"], help="Model type.")
    parser.add_argument("--pm-col", type=str, default="pm_mid", help="Column for Polymarket mid.")
    parser.add_argument("--prn-col", type=str, default="pRN", help="Column for pRN.")
    parser.add_argument("--label-col", type=str, default="label", help="Label column.")
    parser.add_argument("--features-cols", type=str, default=None, help="Comma-separated feature columns.")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Train fraction for single split.")
    parser.add_argument("--walk-forward", action="store_true", help="Enable walk-forward splits.")
    parser.add_argument("--wf-train-days", type=int, default=180, help="Walk-forward train window (days).")
    parser.add_argument("--wf-test-days", type=int, default=30, help="Walk-forward test window (days).")
    parser.add_argument("--wf-step-days", type=int, default=30, help="Walk-forward step (days).")
    parser.add_argument("--max-splits", type=int, default=6, help="Max walk-forward splits.")
    parser.add_argument("--embargo-days", type=int, default=2, help="Embargo window (days) before test start.")
    parser.add_argument("--min-time-to-resolution-days", type=float, default=0.0, help="Drop rows with too little time to resolution.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_id = args.run_id or f"mixed-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    cfg = Config(
        features_path=Path(args.features),
        out_dir=Path(args.out_dir) / run_id,
        run_id=run_id,
        model_type=args.model,
        pm_col=args.pm_col,
        prn_col=args.prn_col,
        label_col=args.label_col,
        features=[c.strip() for c in args.features_cols.split(",") if c.strip()] if args.features_cols else None,
        train_frac=args.train_frac,
        walk_forward=args.walk_forward,
        wf_train_days=args.wf_train_days,
        wf_test_days=args.wf_test_days,
        wf_step_days=args.wf_step_days,
        max_splits=args.max_splits,
        embargo_days=args.embargo_days,
        min_time_to_resolution_days=args.min_time_to_resolution_days,
        alpha=args.alpha,
    )

    df = _load_features(cfg.features_path)
    if df.empty:
        print("[mixed] Decision dataset is empty.")
        return

    df = _prepare_dataset(df, cfg)
    if df.empty:
        print("[mixed] No usable rows after filtering.")
        return

    feature_cols = cfg.features or _default_feature_cols(df)
    feature_cols = [c for c in feature_cols if c in df.columns]

    if not feature_cols:
        raise ValueError("No usable feature columns found.")

    splits = _walk_forward_splits(df, cfg) if cfg.walk_forward else _single_split(df, cfg)
    if not splits:
        raise ValueError("No valid splits generated. Check date coverage or window sizes.")

    metrics_rows: List[Dict[str, object]] = []

    for idx, (train_start, train_end, test_start, test_end) in enumerate(splits):
        train_df, test_df = _filter_split(df, train_start, train_end, test_start, test_end, cfg)
        if train_df.empty or test_df.empty:
            continue

        pipeline = _build_pipeline(cfg.alpha)
        target = _build_target(train_df, cfg)
        pipeline.fit(train_df[feature_cols], target)

        bundle = MixedModelBundle(
            model_type=cfg.model_type,
            model=pipeline,
            feature_cols=feature_cols,
            pm_col=cfg.pm_col,
            prn_col=cfg.prn_col,
        )

        y_true = test_df[cfg.label_col].to_numpy(dtype=float)
        p_pred = bundle.predict_p(test_df)
        metrics = _compute_metrics(y_true, p_pred)

        metrics_rows.append({
            "split": idx,
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            **metrics,
        })

    # Fit final model on all data
    final_pipeline = _build_pipeline(cfg.alpha)
    final_target = _build_target(df, cfg)
    final_pipeline.fit(df[feature_cols], final_target)

    bundle = MixedModelBundle(
        model_type=cfg.model_type,
        model=final_pipeline,
        feature_cols=feature_cols,
        pm_col=cfg.pm_col,
        prn_col=cfg.prn_col,
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.out_dir / "model.joblib"
    save_bundle(bundle, str(model_path))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_type": cfg.model_type,
        "features": feature_cols,
        "pm_col": cfg.pm_col,
        "prn_col": cfg.prn_col,
        "label_col": cfg.label_col,
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
    }
    (cfg.out_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))

    meta = {
        "run_id": cfg.run_id,
        "trained_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_type": cfg.model_type,
        "features": feature_cols,
        "pm_col": cfg.pm_col,
        "prn_col": cfg.prn_col,
        "label_col": cfg.label_col,
        "rows": int(len(df)),
        "embargo_days": cfg.embargo_days,
        "min_time_to_resolution_days": cfg.min_time_to_resolution_days,
        "walk_forward": cfg.walk_forward,
        "splits": metrics_rows,
        "model_path": model_path.name,
    }
    (cfg.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(cfg.out_dir / "metrics.csv", index=False)

    print("[mixed] run_id=", cfg.run_id)
    print("[mixed] rows=", len(df))
    print("[mixed] model=", model_path)
    if metrics_rows:
        print("[mixed] metrics=", cfg.out_dir / "metrics.csv")


if __name__ == "__main__":
    main()
