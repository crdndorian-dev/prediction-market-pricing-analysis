#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from option_chain_weighting_v3 import (
    WEIGHTING_VERSION,
    apply_weighting_v3,
    drop_weight_columns,
)


DEFAULT_INPUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "option-chain"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "option-chain-v3"


def _discover_files(input_root: Path, glob_pattern: str) -> List[Path]:
    if glob_pattern.strip() == "**/*.{csv,parquet}":
        csv_files = list(input_root.rglob("*.csv"))
        parquet_files = list(input_root.rglob("*.parquet"))
        return sorted([p for p in csv_files + parquet_files if p.is_file()])
    return sorted([p for p in input_root.glob(glob_pattern) if p.is_file()])


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
        return
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate option-chain datasets to weighting v3.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--glob", type=str, default="**/*.{csv,parquet}")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--ticker-reweight-mode", choices=["none", "sqrt_inv"], default="none")
    parser.add_argument("--ticker-reweight-alpha-min", type=float, default=0.5)
    parser.add_argument("--ticker-reweight-alpha-max", type=float, default=2.0)
    parser.add_argument("--trade-focus-beta", type=float, default=1.0)
    parser.add_argument("--trade-focus-tickers", type=str, default="")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not input_root.exists():
        raise SystemExit(f"input-root does not exist: {input_root}")
    if args.ticker_reweight_alpha_min <= 0 or args.ticker_reweight_alpha_max <= 0:
        raise SystemExit("ticker reweight alpha bounds must be > 0")
    if args.ticker_reweight_alpha_max < args.ticker_reweight_alpha_min:
        raise SystemExit("ticker-reweight-alpha-max must be >= ticker-reweight-alpha-min")
    if args.trade_focus_beta <= 0:
        raise SystemExit("trade-focus-beta must be > 0")

    files = _discover_files(input_root, args.glob)
    if not files:
        print(f"[RESULT] No files matched under {input_root} with glob={args.glob!r}")
        return

    print(
        f"[MIGRATE] weighting_version={WEIGHTING_VERSION} files={len(files)} input_root={input_root} "
        f"mode={'in-place' if args.in_place else 'versioned-output'} dry_run={bool(args.dry_run)}"
    )

    report_rows: List[dict] = []
    processed = 0
    skipped = 0
    failed = 0

    for path in files:
        rel = path.relative_to(input_root)
        target_path = path if args.in_place else (output_root / rel)
        row_report = {
            "file_in": str(path),
            "file_out": str(target_path),
            "status": "pending",
            "rows": 0,
            "unique_weight_groups": 0,
            "weight_final_min": None,
            "weight_final_mean": None,
            "weight_final_max": None,
            "warnings": [],
        }
        try:
            df = _read_frame(path)
            row_report["rows"] = int(len(df))
            base = drop_weight_columns(df)
            migrated = apply_weighting_v3(
                base,
                ticker_reweight_mode=str(args.ticker_reweight_mode),
                ticker_reweight_alpha_min=float(args.ticker_reweight_alpha_min),
                ticker_reweight_alpha_max=float(args.ticker_reweight_alpha_max),
                trade_focus_beta=float(args.trade_focus_beta),
                trade_focus_tickers=str(args.trade_focus_tickers),
                strict=True,
            )
            row_report["rows"] = int(len(migrated))
            row_report["unique_weight_groups"] = int(migrated["weight_group_key"].nunique(dropna=False))
            row_report["weight_final_min"] = float(migrated["weight_final"].min())
            row_report["weight_final_mean"] = float(migrated["weight_final"].mean())
            row_report["weight_final_max"] = float(migrated["weight_final"].max())

            if target_path.exists() and (not args.overwrite) and (not args.dry_run):
                skipped += 1
                row_report["status"] = "skipped_exists"
                row_report["warnings"].append("target exists; use --overwrite to replace")
            else:
                if not args.dry_run:
                    _write_frame(migrated, target_path)
                processed += 1
                row_report["status"] = "processed"
        except Exception as exc:
            # Non-training files (drops/snapshot/prn-view) are expected to fail key resolution.
            skipped += 1
            row_report["status"] = "skipped_unfit"
            row_report["warnings"].append(str(exc))
        report_rows.append(row_report)

    summary_df = pd.DataFrame(report_rows)
    total = len(summary_df)
    skipped = int((summary_df["status"].str.startswith("skipped")).sum())
    failed = int((summary_df["status"] == "failed").sum())
    processed = int((summary_df["status"] == "processed").sum())

    print(
        f"[SUMMARY] total={total} processed={processed} skipped={skipped} failed={failed} "
        f"version={WEIGHTING_VERSION}"
    )
    cols = [
        "status",
        "file_in",
        "file_out",
        "rows",
        "unique_weight_groups",
        "weight_final_min",
        "weight_final_mean",
        "weight_final_max",
    ]
    print(summary_df[cols].to_string(index=False))

    warnings = summary_df[summary_df["warnings"].map(bool)][["file_in", "warnings"]]
    if not warnings.empty:
        print("[WARNINGS]")
        for _, row in warnings.iterrows():
            joined = " | ".join(row["warnings"])
            print(f"- {row['file_in']}: {joined}")

    report_payload = {
        "weighting_version": WEIGHTING_VERSION,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "in_place": bool(args.in_place),
        "dry_run": bool(args.dry_run),
        "total_files": total,
        "processed_files": processed,
        "skipped_files": skipped,
        "failed_files": failed,
        "rows": report_rows,
    }
    report_base = input_root if args.in_place else output_root
    if not args.dry_run:
        report_base.mkdir(parents=True, exist_ok=True)
        report_path = report_base / "weighting_v3_migration_report.json"
        report_path.write_text(json.dumps(report_payload, indent=2))
        print(f"[WRITE] report: {report_path}")


if __name__ == "__main__":
    main()
