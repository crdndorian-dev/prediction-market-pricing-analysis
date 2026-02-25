#!/usr/bin/env python3
"""
03-auto-calibrate-logit-model-v1.1.py

Auto-calibration orchestrator that enumerates feature/hyperparameter combos and
invokes the v2.0 calibrator as a subprocess for each trial.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_PATH = REPO_ROOT / "config" / "autocalibrate_feature_catalog.json"
DEFAULT_TEMPLATE_PATH = REPO_ROOT / "config" / "metadata_schema_template.json"
DEFAULT_CALIBRATOR_SCRIPT = REPO_ROOT / "src" / "scripts" / "03-calibrate-logit-model-v2.0.py"


def _build_env() -> Dict[str, str]:
    env = os.environ.copy()
    root = str(REPO_ROOT)
    src = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH")
    base = f"{root}{os.pathsep}{src}"
    env["PYTHONPATH"] = f"{base}{os.pathsep}{existing}" if existing else base
    return env


def load_catalog(catalog_path: Path) -> Dict[str, Any]:
    return json.loads(catalog_path.read_text())


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _detect_pm_label_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isin([0, 1]).any():
            return col
    return None


def _prune_pm_features(
    catalog: Dict[str, Any],
    pm_df: pd.DataFrame,
) -> Dict[str, List[str]]:
    removed: Dict[str, List[str]] = {}

    def _pm_feature_has_values(col: str) -> bool:
        if col in pm_df.columns:
            return pd.to_numeric(pm_df[col], errors="coerce").notna().any()
        if col == "pm_mid" and "pPM_mid" in pm_df.columns:
            return pd.to_numeric(pm_df["pPM_mid"], errors="coerce").notna().any()
        return False

    for group in catalog.get("groups", []):
        name = group.get("name", "")
        if not name.startswith("pm_"):
            continue
        options = list(group.get("options", []))
        keep: List[str] = []
        dropped: List[str] = []
        for opt in options:
            if _pm_feature_has_values(opt):
                keep.append(opt)
            else:
                dropped.append(opt)
        if dropped:
            group["options"] = keep
            removed[name] = dropped
    return removed


def _prune_hyperparameters(
    catalog: Dict[str, Any],
    *,
    calibrator_version: Optional[str],
    foundation_tickers: str,
) -> Dict[str, List[Any]]:
    pruned: Dict[str, List[Any]] = {}
    hyper = catalog.get("hyperparameters", {})

    # v2.0 calibrator ignores train_decay_half_life_weeks + foundation_weight
    if calibrator_version and calibrator_version.startswith("v2."):
        decay_vals = list(hyper.get("train_decay_half_life_weeks", []))
        if len(decay_vals) > 1:
            pruned["train_decay_half_life_weeks"] = decay_vals
            hyper["train_decay_half_life_weeks"] = [decay_vals[0]]

        fweights = list(hyper.get("foundation_weight", []))
        if len(fweights) > 1:
            pruned["foundation_weight"] = fweights
            hyper["foundation_weight"] = [1.0]

    # If no foundation tickers supplied, foundation on/off has no effect
    if not foundation_tickers:
        f_enabled = list(hyper.get("foundation_enabled", []))
        if len(f_enabled) > 1:
            pruned.setdefault("foundation_enabled", f_enabled)
        hyper["foundation_enabled"] = [False]
        fweights = list(hyper.get("foundation_weight", []))
        if len(fweights) > 1:
            pruned.setdefault("foundation_weight", fweights)
        if "foundation_weight" in hyper:
            hyper["foundation_weight"] = [1.0]

    return pruned


def _powerset(options: List[str]) -> List[List[str]]:
    out: List[List[str]] = [[]]
    for opt in options:
        out += [subset + [opt] for subset in out]
    return out


def _group_choices(group: Dict[str, Any]) -> List[List[str]]:
    policy = group.get("policy", "any_subset")
    options = list(group.get("options", []))

    if policy == "at_most_one":
        return [[]] + [[opt] for opt in options]
    if policy == "any_subset":
        return _powerset(options)
    if policy == "required_if_mixed":
        return [options]
    if policy == "required":
        return [options]
    raise ValueError(f"Unknown policy: {policy}")


def _apply_dependencies(
    combos: List[Dict[str, List[str]]],
    dependencies: Dict[str, List[str]],
) -> List[Dict[str, List[str]]]:
    if not dependencies:
        return combos

    def _valid(combo: Dict[str, List[str]]) -> bool:
        selected = {feat for items in combo.values() for feat in items}
        for feat, reqs in dependencies.items():
            if feat in selected:
                for req in reqs:
                    if req not in selected:
                        return False
        return True

    return [combo for combo in combos if _valid(combo)]


def enumerate_feature_combos(catalog: Dict[str, Any], mode: str) -> Tuple[List[Dict[str, List[str]]], Dict[str, Any]]:
    groups = catalog.get("groups", [])
    all_groups = list(groups)
    dependencies: Dict[str, List[str]] = {}
    for group in groups:
        deps = group.get("dependencies") or {}
        dependencies.update({k: list(v) for k, v in deps.items()})

    # Count combos before mode filtering
    pre_mode_counts = 1
    for group in all_groups:
        pre_mode_counts *= len(_group_choices(group))

    allowed_groups = [g for g in all_groups if mode in g.get("modes", [])]
    post_mode_counts = 1
    for group in allowed_groups:
        post_mode_counts *= len(_group_choices(group))

    constraints_applied: List[Dict[str, Any]] = []
    if pre_mode_counts != post_mode_counts:
        constraints_applied.append({
            "name": "pm_features_excluded_in_option_only",
            "combos_eliminated": pre_mode_counts - post_mode_counts,
        })

    combos: List[Dict[str, List[str]]] = [{}]
    for group in allowed_groups:
        choices = _group_choices(group)
        new_combos: List[Dict[str, List[str]]] = []
        for combo in combos:
            for choice in choices:
                next_combo = dict(combo)
                next_combo[group["name"]] = list(choice)
                new_combos.append(next_combo)
        combos = new_combos

    # Apply dependencies sequentially for logging
    for feat, reqs in dependencies.items():
        filtered = _apply_dependencies(combos, {feat: reqs})
        eliminated = len(combos) - len(filtered)
        if eliminated:
            constraints_applied.append({
                "name": f"{feat}_requires_{'_'.join(reqs)}",
                "combos_eliminated": eliminated,
            })
        combos = filtered
    after_dep = len(combos)

    constraint_log = {
        "total_combos_before_constraints": pre_mode_counts,
        "constraints_applied": constraints_applied,
        "total_combos_after_constraints": after_dep,
    }

    return combos, constraint_log


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _split_features_by_mode(
    feature_choices: Dict[str, List[str]],
) -> Tuple[List[str], List[str]]:
    pm_features: List[str] = []
    option_features: List[str] = []
    for group, feats in feature_choices.items():
        if group.startswith("pm_"):
            pm_features.extend(feats)
        else:
            option_features.extend(feats)
    return _dedupe_preserve_order(option_features), _dedupe_preserve_order(pm_features)


def enumerate_trial_grid(
    feature_combos: List[Dict[str, List[str]]],
    catalog: Dict[str, Any],
    mode: str,
) -> List[Dict[str, Any]]:
    hyper = catalog.get("hyperparameters", {})
    ticker_intercepts = list(hyper.get("ticker_intercepts", []))
    decay_vals = list(hyper.get("train_decay_half_life_weeks", []))
    calibrations = list(hyper.get("calibration", []))
    foundation_enabled_vals = list(hyper.get("foundation_enabled", []))
    foundation_weights = list(hyper.get("foundation_weight", [1.0]))

    base_features = list(catalog.get("base_features_always_included", []))

    trials: List[Dict[str, Any]] = []
    for combo in feature_combos:
        option_feats, pm_feats = _split_features_by_mode(combo)
        option_feats = _dedupe_preserve_order(base_features + option_feats)
        for ti in ticker_intercepts:
            for decay in decay_vals:
                for cal in calibrations:
                    for foundation_enabled in foundation_enabled_vals:
                        weights = foundation_weights if foundation_enabled else [1.0]
                        for fweight in weights:
                            trials.append({
                                "mode": mode,
                                "feature_choices": combo,
                                "option_features": option_feats,
                                "pm_features": pm_feats,
                                "ticker_intercepts": ti,
                                "train_decay_half_life_weeks": decay,
                                "calibrate": cal,
                                "foundation_enabled": foundation_enabled,
                                "foundation_weight": fweight,
                            })

    def _sort_key(item: Dict[str, Any]) -> Tuple[str, str, float, str, bool, float]:
        feat_sig = json.dumps(item["feature_choices"], sort_keys=True)
        pm_sig = ",".join(item.get("pm_features") or [])
        return (
            feat_sig + pm_sig,
            item["ticker_intercepts"],
            float(item["train_decay_half_life_weeks"]),
            item["calibrate"],
            bool(item["foundation_enabled"]),
            float(item["foundation_weight"]),
        )

    trials.sort(key=_sort_key)
    return trials


def make_trial_id(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def _format_feat_token(prefix: str, feats: List[str]) -> Optional[str]:
    if not feats:
        return None
    return f"{prefix}:{'+'.join(feats)}"


def _build_feat_label(option_feats: List[str], pm_feats: List[str], base_features: List[str]) -> str:
    optional = [f for f in option_feats if f not in base_features]
    if not optional and not pm_feats:
        return "min"

    tokens: List[str] = []
    # infer groups from feature names
    vol = [f for f in optional if f in {"rv20", "rv20_sqrtT"}]
    mon = [f for f in optional if f in {"log_m_fwd", "abs_log_m_fwd"}]
    interactions = [f for f in optional if f in {"x_m", "x_abs_m"}]
    other = [f for f in optional if f not in set(vol + mon + interactions)]

    for prefix, feats in [
        ("v", vol),
        ("m", mon),
        ("x", interactions),
    ]:
        token = _format_feat_token(prefix, feats)
        if token:
            tokens.append(token)
    if other:
        tokens.append("o:" + "+".join(other))
    if pm_feats:
        tokens.append("pm:" + "+".join(pm_feats))

    return "+".join(tokens) if tokens else "min"


def make_trial_dir_name(trial_index: int, config: Dict[str, Any], base_features: List[str]) -> str:
    option_feats = config.get("option_features", [])
    pm_feats = config.get("pm_features", [])
    feat_label = _build_feat_label(option_feats, pm_feats, base_features)
    ti = config.get("ticker_intercepts", "none")
    decay = config.get("train_decay_half_life_weeks", 0)
    cal = config.get("calibrate", "none")
    foundation_enabled = config.get("foundation_enabled", False)
    fweight = config.get("foundation_weight", 1.0)

    def _format_number(value: Any) -> str:
        try:
            num = float(value)
        except Exception:
            return str(value)
        if num.is_integer():
            return str(int(num))
        return str(num)

    parts = [
        f"C=auto",
        f"feat={feat_label}",
        f"ti={ti}",
        "tx=0",
        f"decay={_format_number(decay)}",
        f"cal={cal}",
        f"foundation={'on' if foundation_enabled else 'off'}",
    ]
    if foundation_enabled:
        parts.append(f"fweight={_format_number(fweight)}")

    trial_name = "__".join(parts)
    trial_id = make_trial_id(config)
    return f"{trial_index:03d}__{trial_name}__{trial_id}"


def build_subprocess_command(
    config: Dict[str, Any],
    *,
    calibrator_script: Path,
    base_args: List[str],
    dataset_path: Path,
    pm_dataset_path: Optional[Path],
    out_dir: Path,
    seed: int,
    trial_index: int,
    model_kind: Optional[str] = None,
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(calibrator_script),
        "--csv",
        str(dataset_path),
        "--out-dir",
        str(out_dir),
    ]
    if base_args:
        cmd.extend(base_args)

    option_feats = config.get("option_features", [])
    if option_feats:
        cmd.extend(["--features", ",".join(option_feats)])

    cmd.extend(["--ticker-intercepts", str(config.get("ticker_intercepts", "none"))])
    cmd.extend(["--calibrate", str(config.get("calibrate", "none"))])
    cmd.extend(["--train-decay-half-life-weeks", str(config.get("train_decay_half_life_weeks", 0))])

    foundation_enabled = bool(config.get("foundation_enabled", False))
    foundation_tickers = config.get("foundation_tickers", "") or ""
    foundation_weight = config.get("foundation_weight", 1.0)
    if foundation_enabled:
        cmd.extend(["--foundation-tickers", foundation_tickers])
        cmd.extend(["--foundation-weight", str(foundation_weight)])
    else:
        cmd.extend(["--foundation-tickers", ""])
        cmd.extend(["--foundation-weight", "1.0"])

    if model_kind:
        cmd.extend(["--model-kind", model_kind])

    # Mixed mode: two-stage overlay
    if config.get("mode") == "mixed":
        cmd.append("--two-stage-mode")
        if pm_dataset_path is not None:
            cmd.extend(["--two-stage-pm-csv", str(pm_dataset_path)])
        pm_feats = config.get("pm_features") or []
        if pm_feats:
            cmd.extend(["--pm-features", ",".join(pm_feats)])

    cmd.extend(["--random-state", str(seed + trial_index)])
    return cmd


def _parse_metrics_summary(trial_dir: Path) -> Dict[str, Any]:
    summary_path = trial_dir / "metrics_summary.json"
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text())
        except Exception:
            return {}
    return {}


def _parse_rolling_summary(trial_dir: Path) -> Dict[str, Any]:
    rolling_path = trial_dir / "rolling_summary.csv"
    if not rolling_path.exists():
        return {}
    try:
        import csv
        with rolling_path.open() as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        if not rows:
            return {}
        row = rows[0]
        return {
            "avg_roll_logloss_model": float(row.get("avg_roll_logloss_model", "nan")),
            "avg_roll_logloss_baseline": float(row.get("avg_roll_logloss_baseline", "nan")),
            "avg_roll_delta": float(row.get("avg_roll_delta", "nan")),
        }
    except Exception:
        return {}


def _parse_rolling_windows(trial_dir: Path) -> List[Dict[str, Any]]:
    path = trial_dir / "rolling_windows.csv"
    if not path.exists():
        return []
    try:
        import csv
        with path.open() as handle:
            reader = csv.DictReader(handle)
            rows = []
            for row in reader:
                rows.append({
                    "window": row.get("window"),
                    "window_start": row.get("window_start"),
                    "window_end": row.get("window_end"),
                    "logloss": float(row.get("logloss", "nan")),
                    "baseline_logloss": float(row.get("baseline_logloss", "nan")),
                })
            return rows
    except Exception:
        return []


def _parse_test_metrics(trial_dir: Path) -> Dict[str, Any]:
    metrics_path = trial_dir / "metrics.csv"
    if not metrics_path.exists():
        return {}
    try:
        import csv
        with metrics_path.open() as handle:
            reader = csv.DictReader(handle)
            baseline_logloss = None
            test_metrics = None
            for row in reader:
                if row.get("split") != "test":
                    continue
                model = row.get("model")
                if model == "baseline_pRN":
                    try:
                        baseline_logloss = float(row.get("logloss", "nan"))
                    except Exception:
                        baseline_logloss = None
                elif model == "logit":
                    try:
                        test_metrics = {
                            "logloss": float(row.get("logloss", "nan")),
                            "brier": float(row.get("brier", "nan")),
                            "ece": float(row.get("ece", "nan")),
                            "ece_q": float(row.get("ece_q", "nan")),
                        }
                    except Exception:
                        test_metrics = None
            if test_metrics is None:
                return {}
            test_metrics["baseline_logloss"] = baseline_logloss
            return test_metrics
    except Exception:
        return {}


def _extract_weeks_ranges(trial_dir: Path) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    meta_path = trial_dir / "metadata.json"
    if not meta_path.exists():
        return None, None
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return None, None
    splits = meta.get("splits") or {}
    train_range = splits.get("train_fit_weeks_range")
    test_range = splits.get("test_weeks_range")
    return train_range, test_range


def _resolve_objective_value(
    objective_name: str,
    *,
    metrics_summary: Dict[str, Any],
    rolling_summary: Dict[str, Any],
    test_metrics: Dict[str, Any],
) -> Optional[float]:
    if objective_name in {"test_logloss", "test"}:
        return test_metrics.get("logloss")
    # default: prefer rolling/validation logloss if available
    if metrics_summary.get("val_logloss_mean") is not None:
        return metrics_summary.get("val_logloss_mean")
    if rolling_summary.get("avg_roll_logloss_model") is not None:
        return rolling_summary.get("avg_roll_logloss_model")
    return test_metrics.get("logloss")


def run_trial(
    trial_dir: Path,
    cmd: List[str],
    timeout_s: int,
    objective_name: str = "logloss",
) -> Dict[str, Any]:
    result_path = trial_dir / "trial_result.json"
    if result_path.exists():
        try:
            cached = json.loads(result_path.read_text())
            if cached.get("status") == "success":
                return cached
        except Exception:
            pass

    trial_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=str(REPO_ROOT),
            env=_build_env(),
            timeout=timeout_s if timeout_s > 0 else None,
        )
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except Exception as exc:
        return_code = 1
        stdout = ""
        stderr = str(exc)

    runtime_seconds = time.monotonic() - start

    metrics_summary = _parse_metrics_summary(trial_dir)
    rolling_summary = _parse_rolling_summary(trial_dir)
    rolling_windows = _parse_rolling_windows(trial_dir)
    test_metrics = _parse_test_metrics(trial_dir)
    train_range, test_range = _extract_weeks_ranges(trial_dir)

    val_logloss_std = metrics_summary.get("val_logloss_std")
    val_logloss_by_window = metrics_summary.get("val_logloss_by_window") or rolling_windows

    objective_value = _resolve_objective_value(
        objective_name,
        metrics_summary=metrics_summary,
        rolling_summary=rolling_summary,
        test_metrics=test_metrics,
    )

    status = "success" if return_code == 0 else "failed"
    result = {
        "status": status,
        "return_code": return_code,
        "runtime_seconds": runtime_seconds,
        "objective": objective_value,
        "score": objective_value,
        "delta_brier": None,
        "delta_ece": None,
        "val_logloss_std": val_logloss_std,
        "val_logloss_by_window": val_logloss_by_window,
        "test_metrics": test_metrics or None,
        "train_weeks_range": train_range,
        "test_weeks_range": test_range,
        "command": cmd,
        "stdout": stdout,
        "stderr": stderr,
    }

    result_path.write_text(json.dumps(result, indent=2))
    return result


def write_progress(
    out_dir: Path,
    trials_total: int,
    trials_done: int,
    trials_failed: int,
    best_score: Optional[float],
    last_error: Optional[str],
    stage: str = "training_trials",
) -> None:
    payload = {
        "stage": stage,
        "trials_total": trials_total,
        "trials_done": trials_done,
        "trials_failed": trials_failed,
        "best_score_so_far": best_score,
        "last_error": last_error,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f".progress.{os.getpid()}.{int(time.time() * 1000)}.{uuid.uuid4().hex}.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, out_dir / "progress.json")


def select_best_trial(out_dir: Path, objective: str) -> Optional[Path]:
    trials_dir = out_dir / "trials"
    if not trials_dir.exists():
        return None

    best_path: Optional[Path] = None
    best_score: Optional[float] = None
    for result_path in trials_dir.rglob("trial_result.json"):
        try:
            result = json.loads(result_path.read_text())
        except Exception:
            continue
        if result.get("status") != "success":
            continue
        score = result.get("objective")
        if score is None:
            continue
        if best_score is None or score < best_score:
            best_score = score
            best_path = result_path.parent

    return best_path


def promote_best_trial(best_trial_dir: Path, out_dir: Path, template_path: Path) -> None:
    from app.services.parity_check import check_artifact_parity

    out_dir.mkdir(parents=True, exist_ok=True)
    for item in best_trial_dir.iterdir():
        if not item.is_file():
            continue
        if item.name in {"trial_result.json", "trial_config.json"}:
            continue
        shutil.copy2(item, out_dir / item.name)

    warnings = check_artifact_parity(best_trial_dir, template_path)
    for warning in warnings:
        print(f"[parity] {warning}", file=sys.stderr)


def write_leaderboard(out_dir: Path) -> None:
    trials_dir = out_dir / "trials"
    if not trials_dir.exists():
        return

    rows: List[Dict[str, Any]] = []
    for result_path in trials_dir.rglob("trial_result.json"):
        try:
            result = json.loads(result_path.read_text())
        except Exception:
            continue
        config_path = result_path.parent / "trial_config.json"
        config = {}
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text()).get("config", {})
            except Exception:
                config = {}
        rows.append({
            "trial_id": result_path.parent.name,
            "trial_dir": str(result_path.parent),
            "C": config.get("C"),
            "features": ",".join(config.get("option_features", []) or []),
            "decay": config.get("train_decay_half_life_weeks"),
            "ticker_intercepts": config.get("ticker_intercepts"),
            "calibration": config.get("calibrate"),
            "objective_score": result.get("objective"),
            "status": result.get("status"),
        })

    if not rows:
        return

    import csv

    leaderboard_path = out_dir / "leaderboard.csv"
    with leaderboard_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_run_config(
    out_dir: Path,
    config: Dict[str, Any],
    trial_grid: List[Dict[str, Any]],
    constraint_log: Dict[str, Any],
    catalog_path: Path,
    calibrator_script: Path,
    dataset_path: Path,
    pm_dataset_path: Optional[Path],
    n_workers: int,
    objective: str,
    max_trials: Optional[int],
    pruned_hyperparameters: Optional[Dict[str, List[Any]]] = None,
    pruned_pm_features: Optional[Dict[str, List[str]]] = None,
    pm_label_col: Optional[str] = None,
) -> None:
    calibrator_version = None
    try:
        text = calibrator_script.read_text()
        for line in text.splitlines():
            if line.strip().startswith("SCRIPT_VERSION"):
                _, value = line.split("=", 1)
                calibrator_version = value.strip().strip("'\"")
                break
    except Exception:
        calibrator_version = None

    run_config = {
        "invoked_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": config.get("mode"),
        "seed": config.get("seed"),
        "calibrator_script": str(calibrator_script),
        "calibrator_version": calibrator_version,
        "catalog_path": str(catalog_path),
        "dataset_path": str(dataset_path),
        "pm_dataset_path": str(pm_dataset_path) if pm_dataset_path else None,
        "n_workers": n_workers,
        "objective": objective,
        "max_trials": max_trials,
        "pruned_hyperparameters": pruned_hyperparameters,
        "pruned_pm_features": pruned_pm_features,
        "pm_label_col": pm_label_col,
        "constraints": constraint_log,
        "baseline_args": config.get("baseline_args"),
        "trial_count": len(trial_grid),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))


def write_constraint_log(out_dir: Path, constraint_log: Dict[str, Any]) -> None:
    (out_dir / "constraint_log.json").write_text(json.dumps(constraint_log, indent=2))


def _load_config_from_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    return payload


def _resolve_objective(value: Optional[str]) -> str:
    if not value:
        return "logloss"
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-calibrate logit model by enumerating feature combos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-json", default=None, help="Path to config JSON payload.")
    parser.add_argument("--catalog-path", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--calibrator-script", default=str(DEFAULT_CALIBRATOR_SCRIPT))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--pm-dataset-path", default=None)
    parser.add_argument("--mode", choices=["option_only", "mixed"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=None, help="Alias for --n-workers")
    parser.add_argument("--objective", default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--baseline-args", default=None)
    parser.add_argument("--tdays-allowed", default=None)
    parser.add_argument("--asof-dow-allowed", default=None)
    parser.add_argument("--foundation-tickers", default=None)
    parser.add_argument("--model-kind", default=None)
    parser.add_argument("--timeout-s", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config: Dict[str, Any] = {}
    if args.config_json:
        config = _load_config_from_json(Path(args.config_json))

    def _get_config(key: str, default: Any = None) -> Any:
        return config.get(key, default)

    dataset_value = args.csv if args.csv is not None else _get_config("csv")
    if not dataset_value:
        raise SystemExit("--csv is required (or include csv in config-json)")
    dataset_path = Path(dataset_value)
    if not dataset_path.is_absolute():
        dataset_path = (REPO_ROOT / dataset_path).resolve()

    pm_value = args.pm_dataset_path if args.pm_dataset_path is not None else _get_config("pm_dataset_path")
    if pm_value:
        pm_dataset_path = Path(pm_value)
        if not pm_dataset_path.is_absolute():
            pm_dataset_path = (REPO_ROOT / pm_dataset_path).resolve()
    else:
        pm_dataset_path = None

    mode = args.mode if args.mode is not None else _get_config("mode", "option_only")
    if mode == "mixed" and not pm_dataset_path:
        raise SystemExit("mode=mixed requires --pm-dataset-path")

    seed = args.seed if args.seed is not None else int(_get_config("seed", 42))
    n_workers = args.n_workers if args.n_workers is not None else args.parallel
    if n_workers is None:
        n_workers = int(_get_config("parallel", _get_config("n_workers", 1)) or 1)

    objective = _resolve_objective(args.objective if args.objective is not None else _get_config("objective"))
    max_trials = args.max_trials if args.max_trials is not None else _get_config("max_trials")

    catalog_path = Path(args.catalog_path)
    if not catalog_path.is_absolute():
        catalog_path = (REPO_ROOT / catalog_path).resolve()

    calibrator_script = Path(args.calibrator_script)
    if not calibrator_script.is_absolute():
        calibrator_script = (REPO_ROOT / calibrator_script).resolve()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    calibrator_version = None
    try:
        text = calibrator_script.read_text()
        for line in text.splitlines():
            if line.strip().startswith("SCRIPT_VERSION"):
                _, value = line.split("=", 1)
                calibrator_version = value.strip().strip("'\"")
                break
    except Exception:
        calibrator_version = None

    catalog = load_catalog(catalog_path)

    pm_label_col = None
    pruned_pm_features: Optional[Dict[str, List[str]]] = None
    if mode == "mixed" and pm_dataset_path is not None:
        pm_df = _load_dataset(pm_dataset_path)
        pm_label_col = _detect_pm_label_col(pm_df, ["label", "outcome_ST_gt_K", "outcome"])
        if not pm_label_col:
            prn_df = _load_dataset(dataset_path)
            pm_label_col = _detect_pm_label_col(prn_df, ["label", "outcome_ST_gt_K", "outcome"])
        if not pm_label_col:
            raise SystemExit(
                "No valid label column found in PM or pRN dataset (expected values in {0,1}). "
                "Populate label/outcome_ST_gt_K or pass a different dataset."
            )
        pruned_pm_features = _prune_pm_features(catalog, pm_df)
        if pruned_pm_features:
            print(f"[auto] Pruned PM features with no data: {pruned_pm_features}")
        for group in catalog.get("groups", []):
            if group.get("name") == "pm_signal" and not group.get("options"):
                raise SystemExit(
                    "Polymarket dataset missing pm_mid (or pPM_mid); cannot run mixed auto-calibrate."
                )
    foundation_tickers = args.foundation_tickers or _get_config("foundation_tickers") or ""
    pruned_hyperparameters = _prune_hyperparameters(
        catalog,
        calibrator_version=calibrator_version,
        foundation_tickers=foundation_tickers,
    )
    if pruned_hyperparameters:
        print(f"[auto] Pruned hyperparameters: {pruned_hyperparameters}")

    feature_combos, constraint_log = enumerate_feature_combos(catalog, mode)
    trial_grid = enumerate_trial_grid(feature_combos, catalog, mode)

    if max_trials is not None:
        trial_grid = trial_grid[: int(max_trials)]

    write_constraint_log(out_dir, constraint_log)

    config.setdefault("mode", mode)
    config.setdefault("seed", seed)
    config.setdefault("baseline_args", args.baseline_args or _get_config("baseline_args"))

    write_run_config(
        out_dir,
        config,
        trial_grid,
        constraint_log,
        catalog_path,
        calibrator_script,
        dataset_path,
        pm_dataset_path,
        n_workers,
        objective,
        max_trials,
        pruned_hyperparameters=pruned_hyperparameters or None,
        pruned_pm_features=pruned_pm_features or None,
        pm_label_col=pm_label_col,
    )

    write_progress(out_dir, len(trial_grid), 0, 0, None, None, stage="enumerating")

    baseline_args: List[str] = []
    if config.get("baseline_args"):
        baseline_args.extend(shlex.split(str(config.get("baseline_args"))))

    if args.tdays_allowed or _get_config("tdays_allowed"):
        baseline_args.extend(["--tdays-allowed", str(args.tdays_allowed or _get_config("tdays_allowed"))])
    if args.asof_dow_allowed or _get_config("asof_dow_allowed"):
        baseline_args.extend(["--asof-dow-allowed", str(args.asof_dow_allowed or _get_config("asof_dow_allowed"))])

    trials_dir = out_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    best_score: Optional[float] = None
    trials_done = 0
    trials_failed = 0
    last_error = None

    base_features = list(catalog.get("base_features_always_included", []))

    def _attach_config(
        trial_dir: Path,
        result: Dict[str, Any],
        trial_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        if result.get("config") != trial_config:
            result["config"] = trial_config
            (trial_dir / "trial_result.json").write_text(json.dumps(result, indent=2))
        return result

    trial_items: List[Dict[str, Any]] = []
    for idx, trial in enumerate(trial_grid):
        trial_config = dict(trial)
        trial_config["foundation_tickers"] = foundation_tickers
        trial_config["option_features"] = trial.get("option_features")
        trial_config["pm_features"] = trial.get("pm_features")
        trial_config["enable_x_m"] = "x_m" in (trial_config["option_features"] or [])
        trial_config["enable_x_abs_m"] = "x_abs_m" in (trial_config["option_features"] or [])
        trial_config["ticker_x_interactions"] = False
        trial_config["group_reweight"] = "none"
        trial_config["C"] = None

        trial_dir = trials_dir / make_trial_dir_name(idx + 1, trial_config, base_features)
        trial_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_subprocess_command(
            trial_config,
            calibrator_script=calibrator_script,
            base_args=baseline_args,
            dataset_path=dataset_path,
            pm_dataset_path=pm_dataset_path,
            out_dir=trial_dir,
            seed=seed,
            trial_index=idx,
            model_kind=args.model_kind or _get_config("model_kind"),
        )

        name_parts = trial_dir.name.split("__")
        trial_name = "__".join(name_parts[1:-1]) if len(name_parts) > 2 else trial_dir.name
        trial_payload = {
            "trial_id": idx + 1,
            "trial_name": trial_name,
            "config": trial_config,
            "baseline_args": baseline_args,
            "objective": objective,
        }
        (trial_dir / "trial_config.json").write_text(json.dumps(trial_payload, indent=2))

        trial_items.append({
            "index": idx,
            "trial_dir": trial_dir,
            "cmd": cmd,
            "trial_config": trial_config,
        })

    if n_workers and n_workers > 1:
        futures: Dict[Any, Dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for item in trial_items:
                future = executor.submit(
                    run_trial,
                    item["trial_dir"],
                    item["cmd"],
                    args.timeout_s,
                    objective,
                )
                futures[future] = item
            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    trials_done += 1
                    trials_failed += 1
                    last_error = f"Trial {item['index'] + 1:03d}: {exc}"
                    write_progress(out_dir, len(trial_grid), trials_done, trials_failed, best_score, last_error)
                    continue

                result = _attach_config(item["trial_dir"], result, item["trial_config"])
                trials_done += 1
                if result.get("status") != "success":
                    trials_failed += 1
                    last_error = f"Trial {item['index'] + 1:03d}: {result.get('stderr') or 'Trial failed'}"
                else:
                    score = result.get("objective")
                    if score is not None:
                        if best_score is None or score < best_score:
                            best_score = score
                write_progress(out_dir, len(trial_grid), trials_done, trials_failed, best_score, last_error)
    else:
        for item in trial_items:
            result = run_trial(item["trial_dir"], item["cmd"], args.timeout_s, objective)
            result = _attach_config(item["trial_dir"], result, item["trial_config"])
            trials_done += 1
            if result.get("status") != "success":
                trials_failed += 1
                last_error = f"Trial {item['index'] + 1:03d}: {result.get('stderr') or 'Trial failed'}"
            else:
                score = result.get("objective")
                if score is not None:
                    if best_score is None or score < best_score:
                        best_score = score
            write_progress(out_dir, len(trial_grid), trials_done, trials_failed, best_score, last_error)

    best_trial = select_best_trial(out_dir, objective)
    if best_trial:
        # Add backend to sys.path for parity check import
        backend_root = REPO_ROOT / "src" / "webapp" / "backend"
        if str(backend_root) not in sys.path:
            sys.path.insert(0, str(backend_root))
        promote_best_trial(best_trial, out_dir, DEFAULT_TEMPLATE_PATH)

    write_leaderboard(out_dir)
    write_progress(out_dir, len(trial_grid), trials_done, trials_failed, best_score, last_error, stage="done")


if __name__ == "__main__":
    main()
