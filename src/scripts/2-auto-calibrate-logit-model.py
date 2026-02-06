#!/usr/bin/env python3
"""
2-auto-calibrate-logit-model.py

Beam search auto-tuner for 2-calibrate-logit-model-v1.5.py.

Uses beam search (default width=5) instead of greedy coordinate descent to explore
hyperparameter combinations while staying bounded (default max 250 trials).

Key features:
- FEATURE_CHOICES: pick one variant per family (e.g., rv20 OR rv20_sqrtT, not both)
- Interaction features: x_m (auto) and x_abs_m (--enable-x-abs-m) when moneyness is present
- Calibration: none, platt
- Group reweighting: none, chain
- Composite scoring: logloss + complexity penalty + tie-breakers (brier, ECE)
- Caching: deduplication + persistent cache for reruns
- Outputs: best_config.json, leaderboard.csv (top 25)

Only the best model's artifacts are exported to the final output directory,
matching the structure produced by 2-calibrate-logit-model-v1.5.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import logging
import math
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_C_GRID = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

# FEATURE_CHOICES: each family allows picking one option (list of features) from the list
# This avoids adding collinear features together (e.g., rv20 AND rv20_sqrtT)
# Note: Features must exist in the calibrator's default feature list to work
FEATURE_CHOICES = {
    "volatility": [[], ["rv20"], ["rv20_sqrtT"]],
    "moneyness": [[], ["log_m_fwd"], ["abs_log_m_fwd"], ["log_m_fwd", "abs_log_m_fwd"]],
    # quality and trend features may not exist in all datasets - use with caution
    # "quality": [[], ["log_rel_spread"]],
    # "trend": [[], ["dividend_yield"]],
}

DECAY_OPTIONS_WEEKS = [0.0, 4.0, 8.0, 13.0, 26.0]
TICKER_INTERCEPT_OPTIONS = ["none", "all", "non_foundation"]
TICKER_INTERACTION_OPTIONS = [False, True]
# Note: isotonic is gated behind --allow-isotonic and requires calibrator support
# Currently calibrator only supports ["none", "platt"]
CALIBRATION_OPTIONS = ["none", "platt"]
GROUP_REWEIGHT_OPTIONS = ["none", "chain"]
FOUNDATION_WEIGHT_OPTIONS = [1.0, 1.5, 2.0]

# Beam search defaults
DEFAULT_BEAM_WIDTH = 5
DEFAULT_MAX_TRIALS = 250

OBJECTIVE_KEY = "val_logloss_mean"

# Complexity penalty weights for tie-breaking
COMPLEXITY_PENALTY_FEATURE = 0.0005
COMPLEXITY_PENALTY_TICKER_INTERACTIONS = 0.001


@dataclass
class TrialOutcome:
    config: Dict[str, Any]
    trial_id: int
    trial_name: str
    trial_dir: Path
    status: str
    objective_value: Optional[float]
    val_logloss_std: Optional[float]
    val_logloss_by_window: List[Dict[str, Any]]
    test_metrics: Dict[str, Optional[float]]
    train_weeks_range: Optional[List[str]]
    test_weeks_range: Optional[List[str]]
    runtime_seconds: float
    return_code: Optional[int]
    command: List[str]
    stdout: str
    stderr: str
    # Additional metrics for tie-breaking
    delta_brier: Optional[float] = None
    delta_ece: Optional[float] = None
    score: Optional[float] = None  # Composite score including penalties

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "trial_name": self.trial_name,
            "status": self.status,
            "objective": self.objective_value,
            "score": self.score,
            "delta_brier": self.delta_brier,
            "delta_ece": self.delta_ece,
            "val_logloss_std": self.val_logloss_std,
            "val_logloss_by_window": self.val_logloss_by_window,
            "test_metrics": self.test_metrics,
            "train_weeks_range": self.train_weeks_range,
            "test_weeks_range": self.test_weeks_range,
            "runtime_seconds": self.runtime_seconds,
            "return_code": self.return_code,
            "command": self.command,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "config": self.config,
        }


class AutoModelSelector:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dataset_path = Path(args.csv).expanduser()
        self.out_dir = Path(args.out_dir).expanduser()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.trials_dir = self.out_dir / "trials"
        self.trials_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_args = shlex.split(args.baseline_args or "")
        if args.tdays_allowed:
            self.baseline_args += ["--tdays-allowed", args.tdays_allowed]
        if args.asof_dow_allowed:
            self.baseline_args += ["--asof-dow-allowed", args.asof_dow_allowed]
        if getattr(args, 'train_tickers', ''):
            self.baseline_args += ["--train-tickers", args.train_tickers]
        self.calibrator_script = Path(args.calibrator_script).expanduser()
        if not self.calibrator_script.exists():
            raise FileNotFoundError(f"Calibrator not found at {self.calibrator_script}")
        self.parallel = max(1, args.parallel)
        self.max_trials = args.max_trials if args.max_trials is not None else DEFAULT_MAX_TRIALS
        self.beam_width = args.beam_width if hasattr(args, 'beam_width') else DEFAULT_BEAM_WIDTH
        self.runs_used = 0
        self.next_trial_id = 1
        self.history: Dict[str, TrialOutcome] = {}
        self.results_cache: Dict[str, Dict[str, Any]] = {}  # Persistent cache
        self.budget_exhausted = False
        self.foundation_tickers = args.foundation_tickers.strip()
        self.foundation_weight_base = args.foundation_weight
        self.seed = args.seed
        self.objective = args.objective
        if self.objective != "logloss":
            raise ValueError("Only logloss objective is currently supported")
        # Load cached results if available
        self._load_cache()

    def run(self) -> TrialOutcome:
        """Beam search tuner: explores combinations across families while staying bounded."""
        base_config = self._make_base_config()
        start = time.time()

        # Evaluate base config
        base_results = self._ensure_configs([("base", base_config)])
        base_outcome = base_results.get(self._config_key(base_config))
        if base_outcome is None:
            raise RuntimeError("Unable to evaluate base configuration")
        logging.info("Base configuration evaluated: objective=%s, score=%s",
                     base_outcome.objective_value, base_outcome.score)

        # Initialize beam with base config
        beam: List[TrialOutcome] = [base_outcome]
        families = self._family_generators()
        previous_best_score: Optional[float] = base_outcome.score

        for family_name, generator in families:
            if self.budget_exhausted:
                logging.warning("Budget exhausted; stopping family expansion")
                break

            logging.info("Beam search: expanding family '%s' (beam size=%d)", family_name, len(beam))

            # Generate all candidate variants from current beam
            all_candidates: List[Tuple[str, Dict[str, Any]]] = []
            seen_keys: set = set()

            for beam_outcome in beam:
                candidates = generator(beam_outcome.config)
                for label, update in candidates:
                    next_config = self._merge_config(beam_outcome.config, update)
                    key = self._config_key(next_config)
                    if key in seen_keys or key in self.history:
                        continue
                    seen_keys.add(key)
                    all_candidates.append((f"{family_name}:{label}", next_config))

            if not all_candidates:
                logging.info("No new candidates for family %s", family_name)
                continue

            # Evaluate all unique candidates
            results = self._ensure_configs(all_candidates)

            # Combine beam outcomes with new results, keeping top K by score
            combined: List[TrialOutcome] = list(beam)
            for _, config in all_candidates:
                key = self._config_key(config)
                outcome = results.get(key)
                if outcome and outcome.score is not None:
                    combined.append(outcome)

            # Sort by score (lower is better) and keep top beam_width
            combined.sort(key=lambda o: o.score if o.score is not None else float('inf'))
            beam = combined[:self.beam_width]

            current_best = beam[0] if beam else None
            if current_best:
                logging.info("Family '%s' done. Best score: %s (objective: %s)",
                             family_name, current_best.score, current_best.objective_value)

            # Early stopping if no improvement across full pass
            if current_best and previous_best_score is not None:
                if current_best.score >= previous_best_score:
                    logging.info("No improvement from family '%s'; continuing to next family", family_name)
            previous_best_score = current_best.score if current_best else previous_best_score

        duration = time.time() - start
        best_result = beam[0] if beam else base_outcome
        logging.info("Beam search complete in %.1f seconds after %d trials", duration, self.runs_used)

        # Write outputs
        self._write_leaderboard(beam)
        self._write_best_config(best_result)
        self._save_cache()

        return best_result

    def _family_generators(self) -> List[Tuple[str, callable]]:
        return [
            ("C_grid", self._generate_c_candidates),
            ("features", self._generate_feature_candidates),
            ("interactions", self._generate_interaction_candidates),
            ("ticker_intercepts", self._generate_ticker_intercept_candidates),
            ("ticker_interactions", self._generate_ticker_interaction_candidates),
            ("recency_decay", self._generate_decay_candidates),
            ("calibration", self._generate_calibration_candidates),
            ("group_reweight", self._generate_group_reweight_candidates),
            ("foundation_mode", self._generate_foundation_mode_candidates),
            ("foundation_weight", self._generate_foundation_weight_candidates),
        ]

    def _generate_c_candidates(self, base_config: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        return [
            (f"C={c:g}", {"C": c})
            for c in DEFAULT_C_GRID
            if not math.isclose(c, base_config["C"])
        ]

    def _generate_feature_candidates(self, base_config: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate candidates by varying one choice family at a time."""
        candidates = []
        current_choices = base_config.get("feature_choices", {})

        for family, options in FEATURE_CHOICES.items():
            current_option = current_choices.get(family, [])
            for option in options:
                if option == current_option:
                    continue
                new_choices = dict(current_choices)
                new_choices[family] = option
                label = f"{family}={'|'.join(option) if option else 'none'}"
                candidates.append((label, {"feature_choices": new_choices}))

        return candidates

    def _generate_interaction_candidates(self, base_config: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate candidates for interaction features (x_m, x_abs_m)."""
        candidates = []
        current_x_m = base_config.get("enable_x_m", False)
        current_x_abs_m = base_config.get("enable_x_abs_m", False)
        feature_choices = base_config.get("feature_choices", {})

        # x_m only makes sense when moneyness is present
        has_moneyness = bool(feature_choices.get("moneyness", []))

        if has_moneyness:
            # Toggle x_m
            if not current_x_m:
                candidates.append(("x_m=on", {"enable_x_m": True}))
            else:
                candidates.append(("x_m=off", {"enable_x_m": False}))

            # Toggle x_abs_m (optional, default False)
            if not current_x_abs_m:
                candidates.append(("x_abs_m=on", {"enable_x_abs_m": True}))
            elif current_x_abs_m:
                candidates.append(("x_abs_m=off", {"enable_x_abs_m": False}))
        else:
            # No moneyness, ensure interactions are disabled
            if current_x_m:
                candidates.append(("x_m=off", {"enable_x_m": False}))
            if current_x_abs_m:
                candidates.append(("x_abs_m=off", {"enable_x_abs_m": False}))

        return candidates

    def _generate_group_reweight_candidates(self, base_config: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate candidates for group reweighting options."""
        candidates = []
        current = base_config.get("group_reweight", "none")
        for option in GROUP_REWEIGHT_OPTIONS:
            if option != current:
                candidates.append((f"reweight={option}", {"group_reweight": option}))
        return candidates

    def _generate_ticker_intercept_candidates(self, base_config: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        return [
            (f"ti={option}", {"ticker_intercepts": option})
            for option in TICKER_INTERCEPT_OPTIONS
            if option != base_config["ticker_intercepts"]
        ]

    def _generate_ticker_interaction_candidates(self, base_config: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        return [
            (f"tx={int(option)}", {"ticker_x_interactions": option})
            for option in TICKER_INTERACTION_OPTIONS
            if option != base_config["ticker_x_interactions"]
        ]

    def _generate_decay_candidates(self, base_config: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        return [
            (f"decay={int(val) if val else 0}", {"train_decay_half_life_weeks": val})
            for val in DECAY_OPTIONS_WEEKS
            if not math.isclose(val, base_config["train_decay_half_life_weeks"])
        ]

    def _generate_calibration_candidates(self, base_config: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        candidates = []
        for option in CALIBRATION_OPTIONS:
            if option == base_config["calibrate"]:
                continue
            candidates.append((f"calibrate={option}", {"calibrate": option}))
        return candidates

    def _generate_foundation_mode_candidates(self, base_config: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        candidates: List[Tuple[str, Dict[str, Any]]] = []
        if base_config["foundation_enabled"]:
            candidates.append(
                ("foundation=off", {"foundation_enabled": False, "foundation_tickers": "", "foundation_weight": self.foundation_weight_base})
            )
        if self.foundation_tickers:
            candidates.append(
                (
                    "foundation=on",
                    {
                        "foundation_enabled": True,
                        "foundation_tickers": self.foundation_tickers,
                        "foundation_weight": self.foundation_weight_base,
                    },
                )
            )
        return candidates

    def _generate_foundation_weight_candidates(self, base_config: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        if not base_config["foundation_enabled"]:
            return []
        return [
            (f"fweight={weight:g}", {"foundation_weight": weight})
            for weight in FOUNDATION_WEIGHT_OPTIONS
            if not math.isclose(weight, base_config["foundation_weight"])
        ]

    def _make_base_config(self) -> Dict[str, Any]:
        return {
            "C": DEFAULT_C_GRID[0],
            "feature_choices": {family: [] for family in FEATURE_CHOICES},
            "enable_x_m": False,
            "enable_x_abs_m": False,
            "ticker_intercepts": "non_foundation",
            "ticker_x_interactions": False,
            "train_decay_half_life_weeks": 0.0,
            "calibrate": "none",
            "group_reweight": "none",
            "foundation_enabled": False,
            "foundation_weight": self.foundation_weight_base,
            "foundation_tickers": "",
        }

    def _merge_config(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in updates.items():
            if isinstance(value, list):
                merged[key] = list(value)
            else:
                merged[key] = value
        return merged

    def _config_key(self, config: Dict[str, Any]) -> str:
        return json.dumps(config, sort_keys=True)

    def _hash_config(self, config: Dict[str, Any]) -> str:
        return hashlib.sha1(self._config_key(config).encode("utf-8")).hexdigest()[:8]

    def _build_features_from_choices(self, config: Dict[str, Any]) -> List[str]:
        """Build feature list from choice selections. Always includes x_logit_prn.

        Note: x_m is auto-added by the calibrator when moneyness features are present.
        x_abs_m requires --enable-x-abs-m flag (handled in _build_calibrator_command).
        """
        features: List[str] = ["x_logit_prn"]
        feature_choices = config.get("feature_choices", {})

        for family in FEATURE_CHOICES:
            selected = feature_choices.get(family, [])
            for feat in selected:
                if feat not in features:
                    features.append(feat)

        # Note: x_m and x_abs_m are NOT added here - they are handled by calibrator flags
        return features

    def _feature_list_for_config(self, config: Dict[str, Any]) -> List[str]:
        """Get full feature list including interactions (for display/logging)."""
        features = self._build_features_from_choices(config)
        # Add interaction indicators for display (calibrator auto-adds x_m, uses --enable-x-abs-m)
        feature_choices = config.get("feature_choices", {})
        has_moneyness = bool(feature_choices.get("moneyness", []))
        if has_moneyness and config.get("enable_x_m", False):
            features.append("x_m*")  # * indicates auto-added by calibrator
        if config.get("enable_x_abs_m", False):
            features.append("x_abs_m*")
        return features

    def _build_trial_name(self, config: Dict[str, Any]) -> str:
        feature_choices = config.get("feature_choices", {})
        feat_parts = []
        for family, selected in feature_choices.items():
            if selected:
                feat_parts.append(f"{family[0]}:{'+'.join(selected)}")
        feat_label = "min" if not feat_parts else "|".join(feat_parts)

        parts = [
            f"C={config['C']:g}",
            f"feat={feat_label}",
        ]
        if config.get("enable_x_m"):
            parts.append("xm=1")
        if config.get("enable_x_abs_m"):
            parts.append("xam=1")
        parts.extend([
            f"ti={config['ticker_intercepts']}",
            f"tx={int(config['ticker_x_interactions'])}",
            f"decay={int(config['train_decay_half_life_weeks'])}",
            f"cal={config['calibrate']}",
        ])
        if config.get("group_reweight", "none") != "none":
            parts.append(f"rw={config['group_reweight']}")
        if config["foundation_enabled"]:
            parts.append("foundation=on")
            parts.append(f"fweight={config['foundation_weight']:g}")
        else:
            parts.append("foundation=off")
        return "__".join(parts)

    def _build_calibrator_command(self, config: Dict[str, Any], trial_dir: Path) -> List[str]:
        features_arg = ",".join(self._feature_list_for_config(config))
        core = [
            sys.executable,
            str(self.calibrator_script),
            "--csv",
            str(self.dataset_path),
            "--out-dir",
            str(trial_dir),
        ]
        tuning_args = [
            "--random-state",
            str(self.seed),
            "--C-grid",
            str(config["C"]),
            "--features",
            features_arg,
            "--ticker-intercepts",
            config["ticker_intercepts"],
            "--train-decay-half-life-weeks",
            str(config["train_decay_half_life_weeks"]),
            "--calibrate",
            config["calibrate"],
            "--foundation-tickers",
            config["foundation_tickers"],
            "--foundation-weight",
            str(config["foundation_weight"]),
        ]
        if config["ticker_x_interactions"]:
            tuning_args.append("--ticker-x-interactions")
        # Add group reweight if not "none"
        if config.get("group_reweight", "none") != "none":
            tuning_args.extend(["--group-reweight", config["group_reweight"]])
        # Add x_abs_m interaction feature flag (x_m is auto-added by calibrator when moneyness present)
        if config.get("enable_x_abs_m", False):
            tuning_args.append("--enable-x-abs-m")
        return core + self.baseline_args + tuning_args

    def _truncate(self, text: Optional[str], limit: int = 2000) -> str:
        if not text:
            return ""
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        with path.open() as f:
            return json.load(f)

    def _score_trial(self, config: Dict[str, Any], metrics: Dict[str, Any]) -> Tuple[float, Optional[float], Optional[float]]:
        """
        Compute composite score for ranking trials.
        Primary: delta logloss (lower is better)
        Tie-breakers: delta brier, delta ECE
        Complexity penalty: small penalty for feature count and ticker interactions
        Returns: (score, delta_brier, delta_ece)
        """
        val_logloss = metrics.get(OBJECTIVE_KEY)
        if val_logloss is None:
            return (float('inf'), None, None)

        # Base score is the validation logloss (delta vs baseline if available)
        base_score = val_logloss

        # Compute delta metrics if baseline is available
        val_metrics = metrics.get("val", {})
        delta_brier = val_metrics.get("delta_brier")
        delta_ece = val_metrics.get("delta_ece")

        # Complexity penalty
        features = self._feature_list_for_config(config)
        n_features = len(features) - 1  # Exclude x_logit_prn (always present)
        ticker_interactions_penalty = COMPLEXITY_PENALTY_TICKER_INTERACTIONS if config.get("ticker_x_interactions") else 0
        complexity = COMPLEXITY_PENALTY_FEATURE * n_features + ticker_interactions_penalty

        score = base_score + complexity
        return (score, delta_brier, delta_ece)

    def _load_cache(self) -> None:
        """Load cached results from disk if available."""
        cache_path = self.out_dir / "results_cache.jsonl"
        if cache_path.exists():
            try:
                with cache_path.open() as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        key = entry.get("key")
                        if key:
                            self.results_cache[key] = entry
                logging.info("Loaded %d cached results", len(self.results_cache))
            except Exception as e:
                logging.warning("Failed to load cache: %s", e)

    def _save_cache(self) -> None:
        """Persist results cache to disk."""
        cache_path = self.out_dir / "results_cache.jsonl"
        try:
            with cache_path.open("w") as f:
                for key, entry in self.results_cache.items():
                    f.write(json.dumps(entry) + "\n")
            logging.info("Saved %d results to cache", len(self.results_cache))
        except Exception as e:
            logging.warning("Failed to save cache: %s", e)

    def _write_leaderboard(self, beam: List[TrialOutcome]) -> None:
        """Write top N trials to leaderboard.csv."""
        leaderboard_path = self.out_dir / "leaderboard.csv"
        top_n = 25
        rows = []
        for outcome in sorted(beam, key=lambda o: o.score if o.score is not None else float('inf'))[:top_n]:
            rows.append({
                "trial_id": outcome.trial_id,
                "trial_name": outcome.trial_name,
                "score": outcome.score,
                "objective": outcome.objective_value,
                "delta_brier": outcome.delta_brier,
                "delta_ece": outcome.delta_ece,
                "status": outcome.status,
                "runtime_seconds": outcome.runtime_seconds,
                "config": json.dumps(outcome.config),
            })
        if rows:
            with leaderboard_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            logging.info("Leaderboard written to %s with %d entries", leaderboard_path, len(rows))

    def _write_best_config(self, outcome: TrialOutcome) -> None:
        """Write best configuration to best_config.json."""
        best_config_path = self.out_dir / "best_config.json"
        best_data = {
            "trial_id": outcome.trial_id,
            "trial_name": outcome.trial_name,
            "score": outcome.score,
            "objective": outcome.objective_value,
            "delta_brier": outcome.delta_brier,
            "delta_ece": outcome.delta_ece,
            "config": outcome.config,
            "features": self._feature_list_for_config(outcome.config),
        }
        best_config_path.write_text(json.dumps(best_data, indent=2))
        logging.info("Best config written to %s", best_config_path)

    def _ensure_configs(self, entries: Sequence[Tuple[str, Dict[str, Any]]]) -> Dict[str, TrialOutcome]:
        results: Dict[str, TrialOutcome] = {}
        to_run: List[Tuple[str, Dict[str, Any]]] = []
        for label, config in entries:
            key = self._config_key(config)
            if key in self.history:
                results[key] = self.history[key]
            else:
                to_run.append((label, config))
        if not to_run:
            return results
        futures: List[Tuple[concurrent.futures.Future, Dict[str, Any]]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel) as executor:
            for label, config in to_run:
                if not self._can_schedule_more():
                    self.budget_exhausted = True
                    logging.warning("Max trials reached; skipping remaining candidates")
                    break
                trial_id = self.next_trial_id
                self.next_trial_id += 1
                trial_name = self._build_trial_name(config)
                future = executor.submit(self._run_trial, config, trial_id, trial_name)
                futures.append((future, config))
                self.runs_used += 1
            for future, config in futures:
                outcome = future.result()
                key = self._config_key(config)
                self.history[key] = outcome
                results[key] = outcome
        return results

    def _can_schedule_more(self) -> bool:
        return self.runs_used < self.max_trials

    def _run_trial(self, config: Dict[str, Any], trial_id: int, trial_name: str) -> TrialOutcome:
        trial_dir = self.trials_dir / f"{trial_id:03d}__{trial_name}__{self._hash_config(config)}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        cmd = self._build_calibrator_command(config, trial_dir)
        trial_config = {
            "trial_id": trial_id,
            "trial_name": trial_name,
            "config": config,
            "baseline_args": self.baseline_args,
            "objective": self.objective,
        }
        (trial_dir / "trial_config.json").write_text(json.dumps(trial_config, indent=2))
        start = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            status = "success"
            return_code = proc.returncode
        except subprocess.CalledProcessError as exc:
            proc = exc
            status = "failed"
            return_code = exc.returncode
        runtime = time.time() - start
        metrics_summary = self._load_json(trial_dir / "metrics_summary.json") if status == "success" else None
        metadata = self._load_json(trial_dir / "metadata.json")
        objective_value: Optional[float] = None
        val_std: Optional[float] = None
        val_windows: List[Dict[str, Any]] = []
        test_metrics: Dict[str, Optional[float]] = {"logloss": None, "brier": None, "ece": None, "baseline_logloss": None}
        score: Optional[float] = None
        delta_brier: Optional[float] = None
        delta_ece: Optional[float] = None
        if metrics_summary:
            objective_value = metrics_summary.get(OBJECTIVE_KEY)
            val_std = metrics_summary.get("val_logloss_std")
            val_windows = metrics_summary.get("val_logloss_by_window", [])
            test_metrics = metrics_summary.get("test", test_metrics)
            # Compute composite score
            score, delta_brier, delta_ece = self._score_trial(config, metrics_summary)
        train_range = metadata.get("splits", {}).get("train_fit_weeks_range") if metadata else None
        test_range = metadata.get("splits", {}).get("test_weeks_range") if metadata else None
        outcome = TrialOutcome(
            config=config,
            trial_id=trial_id,
            trial_name=trial_name,
            trial_dir=trial_dir,
            status=status,
            objective_value=objective_value,
            val_logloss_std=val_std,
            val_logloss_by_window=val_windows,
            test_metrics=test_metrics,
            train_weeks_range=train_range,
            test_weeks_range=test_range,
            runtime_seconds=runtime,
            return_code=return_code,
            command=cmd,
            stdout=self._truncate(getattr(proc, "stdout", "")),
            stderr=self._truncate(getattr(proc, "stderr", "")),
            delta_brier=delta_brier,
            delta_ece=delta_ece,
            score=score,
        )
        trial_result = {
            "status": outcome.status,
            "return_code": outcome.return_code,
            "runtime_seconds": outcome.runtime_seconds,
            "objective": outcome.objective_value,
            "score": outcome.score,
            "delta_brier": outcome.delta_brier,
            "delta_ece": outcome.delta_ece,
            "val_logloss_std": outcome.val_logloss_std,
            "val_logloss_by_window": outcome.val_logloss_by_window,
            "test_metrics": outcome.test_metrics,
            "train_weeks_range": outcome.train_weeks_range,
            "test_weeks_range": outcome.test_weeks_range,
            "command": outcome.command,
            "stdout": outcome.stdout,
            "stderr": outcome.stderr,
            "config": outcome.config,
        }
        # Store in persistent cache
        cache_key = self._config_key(config)
        self.results_cache[cache_key] = {"key": cache_key, **trial_result}
        (trial_dir / "trial_result.json").write_text(json.dumps(trial_result, indent=2))
        logging.info(
            "Completed trial %s (%s): status=%s, objective=%s", trial_id, trial_name, outcome.status, outcome.objective_value
        )
        return outcome

    def write_best_report(self, outcome: TrialOutcome) -> None:
        report_path = self.out_dir / "best_model_report.md"
        features = self._feature_list_for_config(outcome.config)
        val_loss = outcome.objective_value
        val_std = outcome.val_logloss_std
        test_metrics = outcome.test_metrics
        lines: List[str] = [
            "# Best Auto-Selected Model",
            "",
            f"- **score** (composite with complexity penalty): {outcome.score if outcome.score is not None else 'N/A'}",
            f"- **objective** (rolling VAL logloss mean): {val_loss if val_loss is not None else 'N/A'}",
            f"- **delta_brier**: {outcome.delta_brier if outcome.delta_brier is not None else 'N/A'}",
            f"- **delta_ece**: {outcome.delta_ece if outcome.delta_ece is not None else 'N/A'}",
            f"- **validation logloss std across windows**: {val_std if val_std is not None else 'N/A'}",
            f"- **test metrics (not used for selection)**: logloss={test_metrics.get('logloss')}, brier={test_metrics.get('brier')}, ece={test_metrics.get('ece')}",
            "",
            "## Final configuration",
            "",
            f"- C: {outcome.config['C']}",
            f"- Features: {features}",
            f"- Feature choices: {outcome.config.get('feature_choices', {})}",
            f"- Interactions: x_m={outcome.config.get('enable_x_m', False)}, x_abs_m={outcome.config.get('enable_x_abs_m', False)}",
            f"- Ticker intercepts: {outcome.config['ticker_intercepts']}",
            f"- Ticker interactions: {outcome.config['ticker_x_interactions']}",
            f"- Recency decay half-life (weeks): {outcome.config['train_decay_half_life_weeks']}",
            f"- Calibration: {outcome.config['calibrate']}",
            f"- Group reweight: {outcome.config.get('group_reweight', 'none')}",
            f"- Foundation enabled: {outcome.config['foundation_enabled']}",
            f"- Foundation tickers: {outcome.config['foundation_tickers']}",
            f"- Foundation weight: {outcome.config['foundation_weight']}",
            "",
            "## Parameter family choices",
            "",
        ]
        family_values = [
            ("C", outcome.config["C"]),
            ("features", features),
            ("feature_choices", outcome.config.get("feature_choices", {})),
            ("enable_x_m", outcome.config.get("enable_x_m", False)),
            ("enable_x_abs_m", outcome.config.get("enable_x_abs_m", False)),
            ("ticker_intercepts", outcome.config["ticker_intercepts"]),
            ("ticker_interactions", outcome.config["ticker_x_interactions"]),
            ("recency_decay_weeks", outcome.config["train_decay_half_life_weeks"]),
            ("calibration", outcome.config["calibrate"]),
            ("group_reweight", outcome.config.get("group_reweight", "none")),
            ("foundation_enabled", outcome.config["foundation_enabled"]),
            ("foundation_weight", outcome.config["foundation_weight"]),
        ]
        for idx, (name, value) in enumerate(family_values, start=1):
            lines.append(f"{idx}. {name}: {value}")
        lines.extend([
            "",
            "## Trial provenance",
            f"- trial id: {outcome.trial_id}",
            f"- trial dir: {outcome.trial_dir}",
            "- top coefficients summary: not captured (run calibrator manually for coefficient inspection)",
        ])
        report_path.write_text("\n".join(lines))
        logging.info("Best model report written to %s", report_path)

    def export_best_model(self, best_outcome: TrialOutcome) -> Path:
        """Copy best model artifacts to final output directory matching 2-calibrate-logit-model-v1.5.py structure."""
        dest_root = self.out_dir
        dest_root.mkdir(parents=True, exist_ok=True)

        # Expected artifact files from calibrator (matches 2-calibrate-logit-model-v1.5.py output)
        expected_files = [
            "base_pipeline.joblib",
            "feature_manifest.json",
            "final_model.joblib",
            "metadata.json",
            "metrics.csv",
            "metrics_groups.csv",
            "metrics_summary.json",
            "model.joblib",
            "reliability_bins.csv",
            "rolling_summary.csv",
            "rolling_windows.csv",
        ]

        copied_count = 0
        missing = []
        for fname in expected_files:
            src = best_outcome.trial_dir / fname
            if not src.exists():
                missing.append(fname)
                continue
            dest = dest_root / fname
            shutil.copy2(src, dest)
            copied_count += 1
            logging.info("Copied %s", fname)

        if missing:
            logging.warning("Best trial missing files: %s", missing)

        logging.info("Best model exported to %s (%d files)", dest_root, copied_count)

        # Validate that critical files exist
        critical_files = ["model.joblib", "final_model.joblib", "metadata.json", "metrics_summary.json"]
        missing_critical = [f for f in critical_files if not (dest_root / f).exists()]
        if missing_critical:
            raise RuntimeError(f"Best model export missing critical files: {missing_critical}")

        return dest_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Beam search auto-tuner for the pHAT calibrator")
    parser.add_argument("--csv", required=True, help="Dataset CSV path")
    parser.add_argument("--out-dir", required=True, help="Root output directory for trials and report")
    parser.add_argument(
        "--calibrator-script",
        default="src/scripts/2-calibrate-logit-model-v1.5.py",
        help="Path to the logistic calibrator script",
    )
    parser.add_argument("--objective", default="logloss", choices=["logloss"], help="Only logloss supported")
    parser.add_argument(
        "--max-trials",
        type=int,
        default=DEFAULT_MAX_TRIALS,
        help=f"Cap on total calibrator trials (default: {DEFAULT_MAX_TRIALS})",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=DEFAULT_BEAM_WIDTH,
        help=f"Beam width for beam search (default: {DEFAULT_BEAM_WIDTH})",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed propagated to calibrator runs")
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="How many calibrator trials may run at the same time (1=sequential)",
    )
    parser.add_argument(
        "--baseline-args",
        default="",
        help="Extra CLI args that are always passed to the calibrator (e.g. --mode pooled)",
    )
    parser.add_argument(
        "--tdays-allowed",
        default="",
        help="Comma-separated list of allowed T_days values (ints). Fixed during tuning.",
    )
    parser.add_argument(
        "--asof-dow-allowed",
        default="",
        help="Comma-separated list of allowed as-of weekdays (Mon..Sun or 0..6 where 0=Mon). Fixed during tuning.",
    )
    parser.add_argument(
        "--train-tickers",
        default="",
        help="Comma-separated tickers for training. Fixed during tuning.",
    )
    parser.add_argument(
        "--foundation-tickers",
        default="SPY,QQQ,IWM",
        help="Comma-delimited foundation tickers used when foundation mode is activated",
    )
    parser.add_argument(
        "--foundation-weight",
        type=float,
        default=1.0,
        help="Baseline foundation weight when foundation mode is toggled on",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    args = parser.parse_args()
    if args.parallel < 1:
        raise ValueError("--parallel must be >= 1")
    if args.max_trials is not None and args.max_trials < 1:
        raise ValueError("--max-trials must be >= 1")
    if args.beam_width < 1:
        raise ValueError("--beam-width must be >= 1")
    logging.basicConfig(level=getattr(logging, args.log_level, logging.INFO), format="[%(levelname)s] %(message)s")
    return args


def main() -> None:
    args = parse_args()
    selector = AutoModelSelector(args)
    best_outcome = selector.run()
    selector.write_best_report(best_outcome)

    # Print winner summary
    print("\n" + "="*80)
    print("WINNER SUMMARY (Beam Search)")
    print("="*80)
    print(f"Trial ID: {best_outcome.trial_id}")
    print(f"Trial Name: {best_outcome.trial_name}")
    print(f"Status: {best_outcome.status}")
    print(f"Score (composite): {best_outcome.score:.6f}" if best_outcome.score else "Score: N/A")
    print(f"Objective (rolling VAL logloss mean): {best_outcome.objective_value:.6f}" if best_outcome.objective_value else "Objective: N/A")
    print(f"Delta Brier: {best_outcome.delta_brier:.6f}" if best_outcome.delta_brier else "Delta Brier: N/A")
    print(f"Delta ECE: {best_outcome.delta_ece:.6f}" if best_outcome.delta_ece else "Delta ECE: N/A")
    print(f"Test Logloss: {best_outcome.test_metrics.get('logloss'):.6f}" if best_outcome.test_metrics.get('logloss') else "Test Logloss: N/A")
    features = selector._feature_list_for_config(best_outcome.config)
    print(f"Features: {features}")

    # Only export model if best trial succeeded
    if best_outcome.status == "success":
        best_model_dir = selector.export_best_model(best_outcome)
        print(f"Best Model Dir: {best_model_dir}")
        logging.info("Auto-tuning complete. Best model exported to %s", best_model_dir)
    else:
        print(f"WARNING: Best trial FAILED - no model exported")
        print(f"Trial dir: {best_outcome.trial_dir}")
        print(f"Stderr (truncated): {best_outcome.stderr[:500] if best_outcome.stderr else 'N/A'}")
        logging.error("Auto-tuning complete but best trial failed. Check trial logs at %s", best_outcome.trial_dir)
        sys.exit(1)

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
