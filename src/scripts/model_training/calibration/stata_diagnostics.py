from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    _STATSMODELS_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:
    sm = None
    variance_inflation_factor = None
    _STATSMODELS_IMPORT_ERROR = exc

from support.calibrate_common import EPS, ece_equal_mass

COEFFICIENT_COLUMNS = [
    "feature_name",
    "display_name",
    "feature_group",
    "coefficient",
    "std_error",
    "z_stat",
    "p_value",
    "ci_low",
    "ci_high",
    "odds_ratio",
    "odds_ratio_ci_low",
    "odds_ratio_ci_high",
    "note",
]

MARGINAL_EFFECT_COLUMNS = [
    "feature_name",
    "display_name",
    "feature_group",
    "ame",
    "std_error",
    "z_stat",
    "p_value",
    "ci_low",
    "ci_high",
    "note",
]

RELIABILITY_BIN_COLUMNS = [
    "split",
    "series",
    "bin",
    "n",
    "pred_mean",
    "obs_rate",
    "abs_gap",
    "pred_min",
    "pred_max",
]

PRODUCTION_COEFFICIENT_SUBTITLE = (
    "Production final sklearn model coefficients. Matches the displayed equation exactly."
)
SHADOW_COEFFICIENT_SUBTITLE = (
    "Shadow statsmodels inference on the transformed train_fit design matrix. "
    "Inferential only; coefficients are not expected to numerically match the production equation."
)
SHADOW_MARGINAL_EFFECTS_SUBTITLE = (
    "Shadow statsmodels marginal effects on the transformed train_fit design matrix."
)

STATSMODELS_MISSING_WARNING = (
    "Stata diagnostics skipped: statsmodels dependency unavailable. "
    "Install backend requirements to enable diagnostics."
)


def _ensure_statsmodels_available() -> None:
    if sm is not None and variance_inflation_factor is not None:
        return
    detail = ""
    if _STATSMODELS_IMPORT_ERROR is not None:
        detail = f" ({type(_STATSMODELS_IMPORT_ERROR).__name__}: {_STATSMODELS_IMPORT_ERROR})"
    raise RuntimeError(f"{STATSMODELS_MISSING_WARNING}{detail}")


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        out = int(value)
    except Exception:
        return None
    return out


def _clean_array(values: Sequence[Any], *, dtype: Any = float) -> np.ndarray:
    return np.asarray(values, dtype=dtype).ravel()


def _valid_probability_sample(
    y_true: Sequence[Any],
    p_pred: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray]:
    y = _clean_array(y_true, dtype=float)
    p = np.clip(_clean_array(p_pred, dtype=float), EPS, 1.0 - EPS)
    valid = np.isfinite(y) & np.isfinite(p)
    return y[valid], p[valid]


def _positive_weights(sample_weight: Optional[Sequence[Any]], n_obs: int) -> Optional[np.ndarray]:
    if sample_weight is None:
        return None
    w = _clean_array(sample_weight, dtype=float)
    if len(w) != n_obs:
        return None
    valid = np.isfinite(w) & (w > 0)
    if not valid.any():
        return None
    out = np.ones(n_obs, dtype=float)
    out[valid] = w[valid]
    return out


def _observed_event_rate(y_true: np.ndarray) -> Optional[float]:
    if y_true.size == 0:
        return None
    return _safe_float(np.mean(y_true))


def _compute_logloss(y_true: np.ndarray, p_pred: np.ndarray) -> Optional[float]:
    y, p = _valid_probability_sample(y_true, p_pred)
    if y.size == 0:
        return None
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _compute_brier(y_true: np.ndarray, p_pred: np.ndarray) -> Optional[float]:
    y, p = _valid_probability_sample(y_true, p_pred)
    if y.size == 0:
        return None
    return float(np.mean((p - y) ** 2))


def _equal_width_bin_ids(p_pred: np.ndarray, n_bins: int) -> np.ndarray:
    bins = np.linspace(0.0, 1.0, int(max(1, n_bins)) + 1)
    ids = np.digitize(p_pred, bins[1:-1], right=False)
    return ids.astype(int)


def _compute_ece(y_true: np.ndarray, p_pred: np.ndarray, *, n_bins: int = 10) -> Optional[float]:
    y, p = _valid_probability_sample(y_true, p_pred)
    if y.size == 0:
        return None
    bin_ids = _equal_width_bin_ids(p, max(1, n_bins))
    out = 0.0
    for idx in range(int(max(1, n_bins))):
        mask = bin_ids == idx
        if not np.any(mask):
            continue
        out += (float(mask.sum()) / float(len(y))) * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return float(out)


def _compute_ece_q(y_true: np.ndarray, p_pred: np.ndarray, *, n_bins: int = 10) -> Optional[float]:
    y, p = _valid_probability_sample(y_true, p_pred)
    if y.size == 0:
        return None
    return float(ece_equal_mass(y, p, n_bins=int(max(1, n_bins))))


def _compute_mce(y_true: np.ndarray, p_pred: np.ndarray, *, n_bins: int = 10) -> Optional[float]:
    y, p = _valid_probability_sample(y_true, p_pred)
    if y.size == 0:
        return None
    bin_ids = _equal_width_bin_ids(p, max(1, n_bins))
    errors: List[float] = []
    for idx in range(int(max(1, n_bins))):
        mask = bin_ids == idx
        if not np.any(mask):
            continue
        errors.append(abs(float(np.mean(p[mask])) - float(np.mean(y[mask]))))
    return float(max(errors)) if errors else None


def _build_equal_mass_bins(
    *,
    y_true: Sequence[Any],
    p_pred: Sequence[Any],
    split: str,
    series: str,
    n_bins: int,
) -> List[Dict[str, Any]]:
    y, p = _valid_probability_sample(y_true, p_pred)
    if y.size == 0:
        return []
    n = len(y)
    bin_count = int(max(1, min(int(max(1, n_bins)), n)))
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n, dtype=int)
    bin_ids = np.minimum((ranks * bin_count) // n, bin_count - 1)

    rows: List[Dict[str, Any]] = []
    for idx in range(bin_count):
        mask = bin_ids == idx
        if not np.any(mask):
            continue
        pred_slice = p[mask]
        obs_slice = y[mask]
        pred_mean = float(np.mean(pred_slice))
        obs_rate = float(np.mean(obs_slice))
        rows.append(
            {
                "split": split,
                "series": series,
                "bin": int(idx + 1),
                "n": int(mask.sum()),
                "pred_mean": pred_mean,
                "obs_rate": obs_rate,
                "abs_gap": float(abs(pred_mean - obs_rate)),
                "pred_min": float(np.min(pred_slice)),
                "pred_max": float(np.max(pred_slice)),
            }
        )
    return rows


def _compute_brier_decomposition(
    y_true: Sequence[Any],
    p_pred: Sequence[Any],
    *,
    n_bins: int = 10,
) -> Dict[str, Optional[float]]:
    y, p = _valid_probability_sample(y_true, p_pred)
    if y.size == 0:
        return {"reliability": None, "resolution": None, "uncertainty": None}

    rows = _build_equal_mass_bins(y_true=y, p_pred=p, split="na", series="model", n_bins=n_bins)
    if not rows:
        return {"reliability": None, "resolution": None, "uncertainty": None}

    base_rate = float(np.mean(y))
    n_total = float(len(y))
    reliability = 0.0
    resolution = 0.0
    for row in rows:
        weight = float(row["n"]) / n_total
        pred_mean = float(row["pred_mean"])
        obs_rate = float(row["obs_rate"])
        reliability += weight * ((pred_mean - obs_rate) ** 2)
        resolution += weight * ((obs_rate - base_rate) ** 2)

    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(base_rate * (1.0 - base_rate)),
    }


def _compute_classification_metrics(
    y_true: Sequence[Any],
    p_pred: Sequence[Any],
    *,
    threshold: float = 0.5,
) -> Dict[str, Optional[float]]:
    y, p = _valid_probability_sample(y_true, p_pred)
    if y.size == 0:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "specificity": None,
            "f1": None,
            "balanced_accuracy": None,
            "tp": None,
            "fp": None,
            "tn": None,
            "fn": None,
            "roc_auc": None,
            "pr_auc": None,
        }

    pred = (p >= float(threshold)).astype(int)
    actual = y.astype(int)
    tp = int(np.sum((pred == 1) & (actual == 1)))
    fp = int(np.sum((pred == 1) & (actual == 0)))
    tn = int(np.sum((pred == 0) & (actual == 0)))
    fn = int(np.sum((pred == 0) & (actual == 1)))

    accuracy = float(np.mean(pred == actual))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else None
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    f1 = (
        float((2.0 * precision * recall) / (precision + recall))
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    balanced_accuracy = (
        float(((recall or 0.0) + (specificity or 0.0)) / 2.0)
        if recall is not None and specificity is not None
        else None
    )

    roc_auc = None
    pr_auc = None
    if len(np.unique(actual)) >= 2:
        try:
            roc_auc = float(roc_auc_score(actual, p))
        except Exception:
            roc_auc = None
        try:
            pr_auc = float(average_precision_score(actual, p))
        except Exception:
            pr_auc = None

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def _compute_calibration_stats(
    y_true: Sequence[Any],
    p_pred: Sequence[Any],
) -> Dict[str, Optional[float]]:
    y, p = _valid_probability_sample(y_true, p_pred)
    if y.size == 0:
        return {
            "calibration_intercept": None,
            "calibration_slope": None,
            "calibration_in_the_large": None,
        }
    if len(np.unique(y.astype(int))) < 2:
        return {
            "calibration_intercept": None,
            "calibration_slope": None,
            "calibration_in_the_large": float(np.mean(p) - np.mean(y)),
        }

    logits = np.log(np.clip(p, EPS, 1.0 - EPS) / np.clip(1.0 - p, EPS, 1.0))
    out: Dict[str, Optional[float]] = {
        "calibration_intercept": None,
        "calibration_slope": None,
        "calibration_in_the_large": float(np.mean(p) - np.mean(y)),
    }

    try:
        offset_model = sm.GLM(
            y,
            np.ones((len(y), 1), dtype=float),
            family=sm.families.Binomial(),
            offset=logits,
        )
        offset_result = offset_model.fit()
        out["calibration_intercept"] = _safe_float(offset_result.params[0])
    except Exception:
        pass

    try:
        slope_exog = sm.add_constant(logits, prepend=True, has_constant="add")
        slope_model = sm.GLM(y, slope_exog, family=sm.families.Binomial())
        slope_result = slope_model.fit()
        out["calibration_slope"] = _safe_float(slope_result.params[1])
    except Exception:
        pass

    return out


def _classify_feature_group(feature_name: str) -> str:
    name = str(feature_name).strip().lower()
    if name in {"const", "_cons"}:
        return "core"
    if "ticker" in name or "_ticker_" in name:
        return "ticker"
    return "core"


def _display_feature_name(feature_name: str) -> str:
    raw = str(feature_name).strip()
    if raw == "const":
        return "_cons"
    for prefix in ("num__", "cat__", "ticker_x__"):
        if raw.startswith(prefix):
            raw = raw[len(prefix) :]
    if "_ticker_feature_" in raw:
        left, _, right = raw.partition("_ticker_feature_")
        if left:
            return f"{left} / ticker={right}"
        return f"ticker={right}"
    return raw


def build_production_coefficient_table(
    *,
    coefficients: Sequence[Any],
    intercept: Optional[Any],
    feature_names: Sequence[str],
) -> Optional[Dict[str, Any]]:
    cleaned_feature_names = [str(name) for name in feature_names if str(name).strip()]
    coefficient_values = [_safe_float(value) for value in coefficients]
    if intercept is None and (not cleaned_feature_names or not coefficient_values):
        return None

    n_pairs = min(len(cleaned_feature_names), len(coefficient_values))
    rows: List[Dict[str, Any]] = []

    intercept_value = _safe_float(intercept)
    if intercept_value is not None:
        rows.append(
            {
                "feature_name": "const",
                "display_name": _display_feature_name("const"),
                "feature_group": _classify_feature_group("const"),
                "coefficient": intercept_value,
                "odds_ratio": None,
                "note": "Intercept from the final production sklearn model.",
            }
        )

    for idx in range(n_pairs):
        feature_name = cleaned_feature_names[idx]
        coefficient = coefficient_values[idx]
        if coefficient is None:
            continue
        rows.append(
            {
                "feature_name": feature_name,
                "display_name": _display_feature_name(feature_name),
                "feature_group": _classify_feature_group(feature_name),
                "coefficient": coefficient,
                "odds_ratio": math.exp(coefficient),
                "note": "Matches the displayed production equation.",
            }
        )

    if not rows:
        return None

    return {
        "basis": "production_sklearn_final_model",
        "fit_scope": "train",
        "subtitle": PRODUCTION_COEFFICIENT_SUBTITLE,
        "rows": rows,
    }


def _fit_shadow_glm(
    *,
    exog: Sequence[Any],
    endog: Sequence[Any],
    sample_weight: Optional[Sequence[Any]],
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    x = np.asarray(exog, dtype=float)
    y = _clean_array(endog, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if len(y) != len(x):
        raise ValueError("Shadow GLM input length mismatch.")

    weights = _positive_weights(sample_weight, len(y))
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if weights is not None:
        valid = valid & np.isfinite(weights) & (weights > 0)
    x = x[valid]
    y = y[valid]
    weights = weights[valid] if weights is not None else None
    if len(y) == 0:
        raise ValueError("No valid rows for shadow GLM.")
    if len(np.unique(y.astype(int))) < 2:
        raise ValueError("Shadow GLM requires both outcome classes.")

    exog_full = sm.add_constant(x, prepend=True, has_constant="add")
    exog_null = np.ones((len(y), 1), dtype=float)
    warnings_seen: List[str] = []

    def _fit(model: sm.GLM) -> Any:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.fit(maxiter=200, disp=0)
        for item in caught:
            message = str(item.message).strip()
            if message and message not in warnings_seen:
                warnings_seen.append(message)
        return result

    full_model = sm.GLM(y, exog_full, family=sm.families.Binomial(), freq_weights=weights)
    full_result = _fit(full_model)
    null_model = sm.GLM(y, exog_null, family=sm.families.Binomial(), freq_weights=weights)
    null_result = _fit(null_model)
    design_names = ["const", *[str(name) for name in feature_names]]

    return {
        "full_result": full_result,
        "null_result": null_result,
        "design_names": design_names,
        "warnings": warnings_seen,
        "exog_full": exog_full,
        "x_no_const": x,
        "y": y,
    }


def _coefficient_rows_from_result(full_result: Any, design_names: Sequence[str]) -> List[Dict[str, Any]]:
    conf_raw = full_result.conf_int()
    if hasattr(conf_raw, "iloc"):
        conf = conf_raw
    else:
        conf = np.asarray(conf_raw, dtype=float)
    rows: List[Dict[str, Any]] = []
    for idx, name in enumerate(design_names):
        coef = _safe_float(full_result.params[idx])
        std_error = _safe_float(full_result.bse[idx]) if hasattr(full_result, "bse") else None
        zvalues = getattr(full_result, "tvalues", getattr(full_result, "zvalues", None))
        z_stat = _safe_float(zvalues[idx]) if zvalues is not None else None
        p_value = _safe_float(full_result.pvalues[idx]) if hasattr(full_result, "pvalues") else None
        if hasattr(conf, "iloc"):
            ci_low = _safe_float(conf.iloc[idx, 0]) if len(conf) > idx else None
            ci_high = _safe_float(conf.iloc[idx, 1]) if len(conf) > idx else None
        else:
            ci_low = _safe_float(conf[idx, 0]) if conf.shape[0] > idx else None
            ci_high = _safe_float(conf[idx, 1]) if conf.shape[0] > idx else None
        odds_ratio = math.exp(coef) if coef is not None and name != "const" else None
        odds_ratio_ci_low = math.exp(ci_low) if ci_low is not None and name != "const" else None
        odds_ratio_ci_high = math.exp(ci_high) if ci_high is not None and name != "const" else None
        note = None
        if std_error is not None and std_error > 5.0:
            note = "Large standard error; interpret cautiously."
        rows.append(
            {
                "feature_name": str(name),
                "display_name": _display_feature_name(str(name)),
                "feature_group": _classify_feature_group(str(name)),
                "coefficient": coef,
                "std_error": std_error,
                "z_stat": z_stat,
                "p_value": p_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "odds_ratio": odds_ratio,
                "odds_ratio_ci_low": odds_ratio_ci_low,
                "odds_ratio_ci_high": odds_ratio_ci_high,
                "note": note,
            }
        )
    return rows


def _marginal_effect_rows_from_result(
    full_result: Any,
    design_names: Sequence[str],
    x_no_const: np.ndarray,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        margeff = full_result.get_margeff(at="overall", method="dydx")
        frame = margeff.summary_frame()
        rename_map = {str(col).strip().lower(): col for col in frame.columns}

        def _lookup_column(*candidates: str) -> Optional[str]:
            for candidate in candidates:
                column = rename_map.get(candidate)
                if column is not None:
                    return column
            return None

        dy_dx_col = _lookup_column("dy/dx")
        std_err_col = _lookup_column("std. err.")
        z_col = _lookup_column("z")
        pvalue_col = _lookup_column("pr(>|z|)")
        ci_low_col = _lookup_column("conf. int. low")
        ci_high_col = _lookup_column("conf. int. hi.", "cont. int. hi.")
        rows: List[Dict[str, Any]] = []
        for idx, feature_name in enumerate(design_names[1:]):
            row_key = feature_name if feature_name in frame.index else frame.index[idx]
            row = frame.loc[row_key]
            rows.append(
                {
                    "feature_name": str(feature_name),
                    "display_name": _display_feature_name(str(feature_name)),
                    "feature_group": _classify_feature_group(str(feature_name)),
                    "ame": _safe_float(row.get(dy_dx_col)),
                    "std_error": _safe_float(row.get(std_err_col)),
                    "z_stat": _safe_float(row.get(z_col)),
                    "p_value": _safe_float(row.get(pvalue_col)),
                    "ci_low": _safe_float(row.get(ci_low_col)),
                    "ci_high": _safe_float(row.get(ci_high_col)),
                    "note": None,
                }
            )
        return rows, None
    except Exception:
        pass

    rows = []
    params = np.asarray(full_result.params, dtype=float)[1:]
    linear_term = np.asarray(full_result.predict(linear=True), dtype=float)
    for idx, feature_name in enumerate(design_names[1:]):
        column = x_no_const[:, idx]
        beta = _safe_float(params[idx])
        ame = None
        note = "AME estimated without variance due to marginal-effects fallback."
        if beta is not None:
            unique = np.unique(np.round(column, 12))
            if len(unique) <= 2 and set(unique.tolist()).issubset({0.0, 1.0}):
                eta_zero = linear_term - (beta * column)
                eta_one = eta_zero + beta
                ame = float(np.mean(1.0 / (1.0 + np.exp(-eta_one)) - 1.0 / (1.0 + np.exp(-eta_zero))))
            else:
                p = 1.0 / (1.0 + np.exp(-linear_term))
                ame = float(beta * np.mean(p * (1.0 - p)))
        rows.append(
            {
                "feature_name": str(feature_name),
                "display_name": _display_feature_name(str(feature_name)),
                "feature_group": _classify_feature_group(str(feature_name)),
                "ame": ame,
                "std_error": None,
                "z_stat": None,
                "p_value": None,
                "ci_low": None,
                "ci_high": None,
                "note": note,
            }
        )
    return rows, "Marginal-effect inference unavailable; falling back to AME point estimates only."


def _numeric_vif_summary(frame: pd.DataFrame, numeric_features: Sequence[str]) -> Dict[str, Any]:
    cols = [str(col) for col in numeric_features if str(col) in frame.columns]
    if len(cols) < 2:
        return {"max_vif": None, "n_high_vif": None, "note": "Need at least two numeric features for VIF."}

    data = frame[cols].copy()
    for col in cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna()
    if len(data) < 5:
        return {"max_vif": None, "n_high_vif": None, "note": "Too few complete numeric rows for VIF."}

    try:
        x = data.to_numpy(dtype=float)
        vif_rows = []
        for idx, col in enumerate(cols):
            vif_value = float(variance_inflation_factor(x, idx))
            if math.isfinite(vif_value):
                vif_rows.append((col, vif_value))
        if not vif_rows:
            return {"max_vif": None, "n_high_vif": None, "note": "No finite VIF values."}
        max_name, max_value = max(vif_rows, key=lambda item: item[1])
        n_high = int(sum(1 for _, value in vif_rows if value >= 5.0))
        return {
            "max_vif": float(max_value),
            "n_high_vif": n_high,
            "note": f"Highest numeric-feature VIF: {max_name}.",
        }
    except Exception as exc:
        return {"max_vif": None, "n_high_vif": None, "note": f"VIF failed: {exc}"}


def _missingness_summary(frame: pd.DataFrame, features: Sequence[str]) -> Dict[str, Any]:
    rows: List[Tuple[str, float]] = []
    for feature in features:
        if feature not in frame.columns:
            rows.append((str(feature), 1.0))
            continue
        missing_share = float(frame[feature].isna().mean())
        rows.append((str(feature), missing_share))
    if not rows:
        return {"max_missing_share": None, "features_with_missing": None, "note": "No selected features."}
    max_feature, max_share = max(rows, key=lambda item: item[1])
    features_with_missing = [name for name, share in rows if share > 0]
    note = None
    if features_with_missing:
        preview = ", ".join(features_with_missing[:6])
        if len(features_with_missing) > 6:
            preview += ", ..."
        note = f"Features with missing values: {preview}"
    else:
        note = "No missing values in selected features."
    return {
        "max_missing_share": float(max_share),
        "features_with_missing": int(len(features_with_missing)),
        "note": f"{note} Max missing feature: {max_feature}.",
    }


def _row(
    key: str,
    label: str,
    *,
    train_fit: Any = None,
    val: Any = None,
    test: Any = None,
    note: Optional[str] = None,
    status: str = "na",
    format_kind: str = "metric",
    source: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "train_fit": train_fit,
        "val": val,
        "test": test,
        "note": note,
        "status": status,
        "format": format_kind,
        "source": source,
    }


def _section(section_id: str, title: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"id": section_id, "title": title, "rows": rows}


def _fit_warning_summary(messages: Sequence[str]) -> Tuple[Optional[str], str]:
    if not messages:
        return None, "ok"
    joined = "; ".join(str(message) for message in messages if str(message).strip())
    if not joined:
        return None, "ok"
    if "separation" in joined.lower() or "perfect" in joined.lower():
        return joined, "warn"
    return joined, "warn"


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: Sequence[str]) -> None:
    pd.DataFrame(rows, columns=list(columns)).to_csv(path, index=False)


def build_and_write_stata_diagnostics(
    *,
    out_dir: Path,
    model_id: str,
    estimator: str,
    baseline_name: str,
    best_c: Optional[float],
    penalty: str,
    solver: str,
    threshold_default: float,
    threshold_operating: Optional[float],
    production_coefficients: Optional[Sequence[Any]],
    production_intercept: Optional[Any],
    production_feature_names: Optional[Sequence[str]],
    shadow_exog: Sequence[Any],
    shadow_endog: Sequence[Any],
    shadow_sample_weight: Optional[Sequence[Any]],
    shadow_feature_names: Sequence[str],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    requested_numeric_features: Sequence[str],
    requested_categorical_features: Sequence[str],
    train_fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_fit_pred: Optional[Sequence[Any]],
    val_pred: Optional[Sequence[Any]],
    test_pred: Optional[Sequence[Any]],
    target_col: str,
    n_bins: int,
    eceq_bins: int,
    fold_delta_rows: Sequence[Dict[str, Any]],
    trainer_warnings: Sequence[str],
) -> Dict[str, Any]:
    _ensure_statsmodels_available()

    split_frames: Dict[str, Tuple[pd.DataFrame, Optional[Sequence[Any]]]] = {
        "train_fit": (train_fit_df, train_fit_pred),
        "val": (val_df, val_pred),
        "test": (test_df, test_pred),
    }

    split_counts: Dict[str, Optional[int]] = {}
    event_counts: Dict[str, Optional[int]] = {}
    split_metrics: Dict[str, Dict[str, Any]] = {}
    reliability_rows: List[Dict[str, Any]] = []

    for split, (frame, predictions) in split_frames.items():
        if frame.empty or predictions is None or target_col not in frame.columns:
            split_counts[split] = None
            event_counts[split] = None
            split_metrics[split] = {}
            continue

        y_true = frame[target_col].to_numpy(dtype=float)
        p_pred = np.asarray(predictions, dtype=float)
        p_baseline = pd.to_numeric(frame.get(baseline_name, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        split_counts[split] = int(len(y_true))
        event_counts[split] = int(np.sum(y_true))

        model_logloss = _compute_logloss(y_true, p_pred)
        model_brier = _compute_brier(y_true, p_pred)
        model_ece = _compute_ece(y_true, p_pred, n_bins=n_bins)
        model_ece_q = _compute_ece_q(y_true, p_pred, n_bins=eceq_bins)
        model_mce = _compute_mce(y_true, p_pred, n_bins=n_bins)
        model_brier_parts = _compute_brier_decomposition(y_true, p_pred, n_bins=eceq_bins)
        classification = _compute_classification_metrics(y_true, p_pred, threshold=threshold_default)
        calibration = _compute_calibration_stats(y_true, p_pred)

        baseline_logloss = _compute_logloss(y_true, p_baseline)
        baseline_brier = _compute_brier(y_true, p_baseline)
        baseline_ece_q = _compute_ece_q(y_true, p_baseline, n_bins=eceq_bins)

        split_metrics[split] = {
            "model_logloss": model_logloss,
            "model_brier": model_brier,
            "model_ece": model_ece,
            "model_ece_q": model_ece_q,
            "model_mce": model_mce,
            "brier_reliability": model_brier_parts.get("reliability"),
            "brier_resolution": model_brier_parts.get("resolution"),
            "brier_uncertainty": model_brier_parts.get("uncertainty"),
            "baseline_logloss": baseline_logloss,
            "baseline_brier": baseline_brier,
            "baseline_ece_q": baseline_ece_q,
            "delta_logloss_vs_baseline": (
                model_logloss - baseline_logloss if model_logloss is not None and baseline_logloss is not None else None
            ),
            "delta_brier_vs_baseline": (
                model_brier - baseline_brier if model_brier is not None and baseline_brier is not None else None
            ),
            "delta_ece_q_vs_baseline": (
                model_ece_q - baseline_ece_q if model_ece_q is not None and baseline_ece_q is not None else None
            ),
            **classification,
            **calibration,
        }

        reliability_rows.extend(_build_equal_mass_bins(y_true=y_true, p_pred=p_pred, split=split, series="model", n_bins=eceq_bins))
        if np.isfinite(p_baseline).any():
            reliability_rows.extend(
                _build_equal_mass_bins(y_true=y_true, p_pred=p_baseline, split=split, series="baseline", n_bins=eceq_bins)
            )

    shadow_warnings: List[str] = []
    shadow_failure_note: Optional[str] = None
    coefficient_rows: List[Dict[str, Any]] = []
    marginal_effect_rows: List[Dict[str, Any]] = []
    global_rows: List[Dict[str, Any]] = []
    coefficient_summary_rows: List[Dict[str, Any]] = []
    goodness_rows: List[Dict[str, Any]] = []

    try:
        shadow = _fit_shadow_glm(
            exog=shadow_exog,
            endog=shadow_endog,
            sample_weight=shadow_sample_weight,
            feature_names=shadow_feature_names,
        )
        shadow_warnings.extend(list(shadow.get("warnings") or []))
        full_result = shadow["full_result"]
        null_result = shadow["null_result"]
        design_names = list(shadow["design_names"])
        coefficient_rows = _coefficient_rows_from_result(full_result, design_names)
        marginal_effect_rows, marginal_note = _marginal_effect_rows_from_result(
            full_result,
            design_names,
            shadow["x_no_const"],
        )
        if marginal_note:
            shadow_warnings.append(marginal_note)

        llf = _safe_float(getattr(full_result, "llf", None))
        llnull = _safe_float(getattr(null_result, "llf", None))
        df_model = max(0, len(design_names) - 1)
        lr_stat = None
        lr_p = None
        pseudo_r2 = None
        if llf is not None and llnull is not None:
            lr_stat = float(max(0.0, 2.0 * (llf - llnull)))
            lr_p = float(chi2.sf(lr_stat, df_model)) if df_model > 0 else None
            pseudo_r2 = float(1.0 - (llf / llnull)) if llnull not in (None, 0.0) else None

        wald_stat = None
        wald_p = None
        score_stat = None
        score_p = None
        score_note = None
        if df_model > 0:
            try:
                restriction = np.eye(len(design_names), dtype=float)[1:, :]
                wald = full_result.wald_test(restriction, scalar=True)
                wald_stat = _safe_float(getattr(wald, "statistic", None))
                wald_p = _safe_float(getattr(wald, "pvalue", None))
            except Exception as exc:
                shadow_warnings.append(f"Global Wald test unavailable: {exc}")
            try:
                score_result = null_result.score_test(exog_extra=np.asarray(shadow["x_no_const"], dtype=float))
                if isinstance(score_result, tuple) and len(score_result) >= 2:
                    score_stat = _safe_float(np.asarray(score_result[0]).squeeze())
                    score_p = _safe_float(np.asarray(score_result[1]).squeeze())
            except Exception as exc:
                score_note = f"Score / LM test unavailable: {exc}"

        global_rows = [
            _row(
                "lr_chi2",
                f"LR chi2({df_model})",
                train_fit=lr_stat,
                note="Likelihood-ratio test versus intercept-only shadow GLM.",
                status="ok" if lr_p is not None and lr_p < 0.05 else "warn",
            ),
            _row(
                "lr_prob",
                "Prob > chi2",
                train_fit=lr_p,
                format_kind="pvalue",
                status="ok" if lr_p is not None and lr_p < 0.05 else "warn",
            ),
            _row(
                "wald_chi2",
                "Global Wald chi2",
                train_fit=wald_stat,
                note="Joint Wald test on non-intercept shadow coefficients.",
                status="ok" if wald_p is not None and wald_p < 0.05 else "warn",
            ),
            _row(
                "wald_prob",
                "Wald p-value",
                train_fit=wald_p,
                format_kind="pvalue",
                status="ok" if wald_p is not None and wald_p < 0.05 else "warn",
            ),
            _row(
                "score_chi2",
                "Score / LM chi2",
                train_fit=score_stat,
                note=score_note,
                status="ok" if score_p is not None and score_p < 0.05 else ("na" if score_stat is None else "warn"),
            ),
            _row(
                "score_prob",
                "Score / LM p-value",
                train_fit=score_p,
                format_kind="pvalue",
                status="ok" if score_p is not None and score_p < 0.05 else ("na" if score_p is None else "warn"),
            ),
        ]

        significant_coefficients = int(
            sum(1 for row in coefficient_rows if row["feature_name"] != "const" and row.get("p_value") is not None and float(row["p_value"]) < 0.05)
        )
        unstable_coefficients = int(
            sum(
                1
                for row in coefficient_rows
                if row["feature_name"] != "const"
                and (
                    row.get("std_error") is None
                    or row.get("coefficient") is None
                    or (row.get("std_error") is not None and float(row["std_error"]) > 5.0)
                )
            )
        )
        coefficient_summary_rows = [
            _row(
                "significant_coefficients",
                "Significant coefficients (p < 0.05)",
                train_fit=float(significant_coefficients),
                format_kind="count",
                note="Excludes the intercept row.",
                status="ok" if significant_coefficients > 0 else "warn",
            ),
            _row(
                "unstable_coefficients",
                "Unstable / weakly identified coefficients",
                train_fit=float(unstable_coefficients),
                format_kind="count",
                note="Large standard errors or missing inferential statistics.",
                status="warn" if unstable_coefficients > 0 else "ok",
            ),
            _row(
                "marginal_effect_rows",
                "Marginal effects rows",
                train_fit=float(len(marginal_effect_rows)),
                format_kind="count",
                note="Average marginal effects are reported in a separate table.",
                status="ok" if marginal_effect_rows else "na",
            ),
        ]

        bic_value = _safe_float(getattr(full_result, "bic_llf", None))
        if bic_value is None:
            bic_value = _safe_float(getattr(full_result, "bic", None))
        goodness_rows = [
            _row("log_likelihood", "Log-likelihood", train_fit=llf, status="ok"),
            _row("null_log_likelihood", "Null log-likelihood", train_fit=llnull, status="ok"),
            _row("pseudo_r2_mcfadden", "Pseudo-R² (McFadden)", train_fit=pseudo_r2, status="ok"),
            _row("aic", "AIC", train_fit=_safe_float(getattr(full_result, "aic", None)), status="ok"),
            _row("bic", "BIC", train_fit=bic_value, status="ok"),
            _row("deviance", "Deviance", train_fit=_safe_float(getattr(full_result, "deviance", None)), status="ok"),
        ]
    except Exception as exc:
        shadow_failure_note = f"Shadow statsmodels GLM unavailable: {exc}"
        shadow_warnings.append(shadow_failure_note)
        global_rows = [
            _row("lr_chi2", "LR chi2", note=shadow_failure_note),
            _row("lr_prob", "Prob > chi2", note=shadow_failure_note, format_kind="pvalue"),
            _row("wald_chi2", "Global Wald chi2", note=shadow_failure_note),
            _row("wald_prob", "Wald p-value", note=shadow_failure_note, format_kind="pvalue"),
            _row("score_chi2", "Score / LM chi2", note=shadow_failure_note),
            _row("score_prob", "Score / LM p-value", note=shadow_failure_note, format_kind="pvalue"),
        ]
        coefficient_summary_rows = [
            _row("significant_coefficients", "Significant coefficients (p < 0.05)", note=shadow_failure_note, format_kind="count"),
            _row("unstable_coefficients", "Unstable / weakly identified coefficients", note=shadow_failure_note, format_kind="count"),
            _row("marginal_effect_rows", "Marginal effects rows", note=shadow_failure_note, format_kind="count"),
        ]
        goodness_rows = [
            _row("log_likelihood", "Log-likelihood", note=shadow_failure_note),
            _row("null_log_likelihood", "Null log-likelihood", note=shadow_failure_note),
            _row("pseudo_r2_mcfadden", "Pseudo-R² (McFadden)", note=shadow_failure_note),
            _row("aic", "AIC", note=shadow_failure_note),
            _row("bic", "BIC", note=shadow_failure_note),
            _row("deviance", "Deviance", note=shadow_failure_note),
        ]

    predictive_rows = [
        _row(
            "classification_threshold",
            "Classification threshold",
            train_fit=float(threshold_default),
            val=float(threshold_default),
            test=float(threshold_default) if split_counts.get("test") else None,
            status="ok",
        ),
        _row(
            "accuracy_050",
            "Accuracy @ 0.50",
            train_fit=split_metrics.get("train_fit", {}).get("accuracy"),
            val=split_metrics.get("val", {}).get("accuracy"),
            test=split_metrics.get("test", {}).get("accuracy"),
            status="ok",
        ),
        _row(
            "precision_050",
            "Precision @ 0.50",
            train_fit=split_metrics.get("train_fit", {}).get("precision"),
            val=split_metrics.get("val", {}).get("precision"),
            test=split_metrics.get("test", {}).get("precision"),
            status="ok",
        ),
        _row(
            "recall_050",
            "Recall / Sensitivity @ 0.50",
            train_fit=split_metrics.get("train_fit", {}).get("recall"),
            val=split_metrics.get("val", {}).get("recall"),
            test=split_metrics.get("test", {}).get("recall"),
            status="ok",
        ),
        _row(
            "specificity_050",
            "Specificity @ 0.50",
            train_fit=split_metrics.get("train_fit", {}).get("specificity"),
            val=split_metrics.get("val", {}).get("specificity"),
            test=split_metrics.get("test", {}).get("specificity"),
            status="ok",
        ),
        _row(
            "f1_050",
            "F1-score @ 0.50",
            train_fit=split_metrics.get("train_fit", {}).get("f1"),
            val=split_metrics.get("val", {}).get("f1"),
            test=split_metrics.get("test", {}).get("f1"),
            status="ok",
        ),
        _row(
            "balanced_accuracy_050",
            "Balanced accuracy @ 0.50",
            train_fit=split_metrics.get("train_fit", {}).get("balanced_accuracy"),
            val=split_metrics.get("val", {}).get("balanced_accuracy"),
            test=split_metrics.get("test", {}).get("balanced_accuracy"),
            status="ok",
        ),
        _row(
            "tp_050",
            "TP @ 0.50",
            train_fit=split_metrics.get("train_fit", {}).get("tp"),
            val=split_metrics.get("val", {}).get("tp"),
            test=split_metrics.get("test", {}).get("tp"),
            format_kind="count",
            status="ok",
        ),
        _row(
            "fp_050",
            "FP @ 0.50",
            train_fit=split_metrics.get("train_fit", {}).get("fp"),
            val=split_metrics.get("val", {}).get("fp"),
            test=split_metrics.get("test", {}).get("fp"),
            format_kind="count",
            status="ok",
        ),
        _row(
            "tn_050",
            "TN @ 0.50",
            train_fit=split_metrics.get("train_fit", {}).get("tn"),
            val=split_metrics.get("val", {}).get("tn"),
            test=split_metrics.get("test", {}).get("tn"),
            format_kind="count",
            status="ok",
        ),
        _row(
            "fn_050",
            "FN @ 0.50",
            train_fit=split_metrics.get("train_fit", {}).get("fn"),
            val=split_metrics.get("val", {}).get("fn"),
            test=split_metrics.get("test", {}).get("fn"),
            format_kind="count",
            status="ok",
        ),
        _row(
            "roc_auc",
            "ROC-AUC",
            train_fit=split_metrics.get("train_fit", {}).get("roc_auc"),
            val=split_metrics.get("val", {}).get("roc_auc"),
            test=split_metrics.get("test", {}).get("roc_auc"),
            status="ok",
        ),
        _row(
            "pr_auc",
            "PR-AUC",
            train_fit=split_metrics.get("train_fit", {}).get("pr_auc"),
            val=split_metrics.get("val", {}).get("pr_auc"),
            test=split_metrics.get("test", {}).get("pr_auc"),
            status="ok",
        ),
    ]

    probabilistic_rows = [
        _row(
            "logloss",
            "Log loss",
            train_fit=split_metrics.get("train_fit", {}).get("model_logloss"),
            val=split_metrics.get("val", {}).get("model_logloss"),
            test=split_metrics.get("test", {}).get("model_logloss"),
            status="ok",
        ),
        _row(
            "delta_logloss_vs_baseline",
            f"Delta log loss vs {baseline_name}",
            train_fit=split_metrics.get("train_fit", {}).get("delta_logloss_vs_baseline"),
            val=split_metrics.get("val", {}).get("delta_logloss_vs_baseline"),
            test=split_metrics.get("test", {}).get("delta_logloss_vs_baseline"),
            note="Negative values improve on the baseline.",
            status="ok",
        ),
        _row(
            "brier",
            "Brier score",
            train_fit=split_metrics.get("train_fit", {}).get("model_brier"),
            val=split_metrics.get("val", {}).get("model_brier"),
            test=split_metrics.get("test", {}).get("model_brier"),
            status="ok",
        ),
        _row(
            "delta_brier_vs_baseline",
            f"Delta Brier vs {baseline_name}",
            train_fit=split_metrics.get("train_fit", {}).get("delta_brier_vs_baseline"),
            val=split_metrics.get("val", {}).get("delta_brier_vs_baseline"),
            test=split_metrics.get("test", {}).get("delta_brier_vs_baseline"),
            note="Negative values improve on the baseline.",
            status="ok",
        ),
        _row(
            "brier_reliability",
            "Brier decomposition: reliability",
            train_fit=split_metrics.get("train_fit", {}).get("brier_reliability"),
            val=split_metrics.get("val", {}).get("brier_reliability"),
            test=split_metrics.get("test", {}).get("brier_reliability"),
            status="ok",
        ),
        _row(
            "brier_resolution",
            "Brier decomposition: resolution",
            train_fit=split_metrics.get("train_fit", {}).get("brier_resolution"),
            val=split_metrics.get("val", {}).get("brier_resolution"),
            test=split_metrics.get("test", {}).get("brier_resolution"),
            status="ok",
        ),
        _row(
            "brier_uncertainty",
            "Brier decomposition: uncertainty",
            train_fit=split_metrics.get("train_fit", {}).get("brier_uncertainty"),
            val=split_metrics.get("val", {}).get("brier_uncertainty"),
            test=split_metrics.get("test", {}).get("brier_uncertainty"),
            status="ok",
        ),
        _row(
            "ece",
            "ECE",
            train_fit=split_metrics.get("train_fit", {}).get("model_ece"),
            val=split_metrics.get("val", {}).get("model_ece"),
            test=split_metrics.get("test", {}).get("model_ece"),
            status="ok",
        ),
        _row(
            "ece_q",
            "ECE-Q",
            train_fit=split_metrics.get("train_fit", {}).get("model_ece_q"),
            val=split_metrics.get("val", {}).get("model_ece_q"),
            test=split_metrics.get("test", {}).get("model_ece_q"),
            status="ok",
        ),
        _row(
            "delta_ece_q_vs_baseline",
            f"Delta ECE-Q vs {baseline_name}",
            train_fit=split_metrics.get("train_fit", {}).get("delta_ece_q_vs_baseline"),
            val=split_metrics.get("val", {}).get("delta_ece_q_vs_baseline"),
            test=split_metrics.get("test", {}).get("delta_ece_q_vs_baseline"),
            note="Negative values improve on the baseline.",
            status="ok",
        ),
        _row(
            "mce",
            "MCE",
            train_fit=split_metrics.get("train_fit", {}).get("model_mce"),
            val=split_metrics.get("val", {}).get("model_mce"),
            test=split_metrics.get("test", {}).get("model_mce"),
            status="ok",
        ),
    ]

    calibration_rows = [
        _row(
            "calibration_intercept",
            "Calibration intercept",
            train_fit=split_metrics.get("train_fit", {}).get("calibration_intercept"),
            val=split_metrics.get("val", {}).get("calibration_intercept"),
            test=split_metrics.get("test", {}).get("calibration_intercept"),
            note="Estimated via intercept-only recalibration with slope fixed at 1.",
            status="ok",
        ),
        _row(
            "calibration_slope",
            "Calibration slope",
            train_fit=split_metrics.get("train_fit", {}).get("calibration_slope"),
            val=split_metrics.get("val", {}).get("calibration_slope"),
            test=split_metrics.get("test", {}).get("calibration_slope"),
            status="ok",
        ),
        _row(
            "calibration_in_the_large",
            "Calibration-in-the-large",
            train_fit=split_metrics.get("train_fit", {}).get("calibration_in_the_large"),
            val=split_metrics.get("val", {}).get("calibration_in_the_large"),
            test=split_metrics.get("test", {}).get("calibration_in_the_large"),
            note="Mean predicted probability minus observed event rate.",
            status="ok",
        ),
        _row(
            "mean_abs_bin_gap",
            "Mean absolute bin gap",
            train_fit=_safe_float(np.mean([row["abs_gap"] for row in reliability_rows if row["split"] == "train_fit" and row["series"] == "model"])) if reliability_rows else None,
            val=_safe_float(np.mean([row["abs_gap"] for row in reliability_rows if row["split"] == "val" and row["series"] == "model"])) if reliability_rows else None,
            test=_safe_float(np.mean([row["abs_gap"] for row in reliability_rows if row["split"] == "test" and row["series"] == "model"])) if reliability_rows else None,
            note="Equal-mass reliability bins.",
            status="ok",
        ),
    ]

    fold_values = [float(row.get("delta_logloss")) for row in fold_delta_rows if _safe_float(row.get("delta_logloss")) is not None]
    val_gap = None
    if split_metrics.get("train_fit", {}).get("model_logloss") is not None and split_metrics.get("val", {}).get("model_logloss") is not None:
        val_gap = float(split_metrics["val"]["model_logloss"] - split_metrics["train_fit"]["model_logloss"])
    test_gap = None
    if split_metrics.get("test", {}).get("model_logloss") is not None and split_metrics.get("val", {}).get("model_logloss") is not None:
        test_gap = float(split_metrics["test"]["model_logloss"] - split_metrics["val"]["model_logloss"])

    robustness_rows = [
        _row(
            "val_minus_train_fit_logloss",
            "Val minus train-fit log loss",
            train_fit=val_gap,
            note="Positive values indicate deterioration out of sample.",
            status="warn" if val_gap is not None and val_gap > 0 else "ok",
        ),
        _row(
            "test_minus_val_logloss",
            "Test minus val log loss",
            train_fit=test_gap,
            note="Positive values indicate deterioration from validation to test.",
            status="warn" if test_gap is not None and test_gap > 0 else "ok",
        ),
        _row(
            "fold_mean_delta_logloss",
            "Fold mean delta log loss",
            train_fit=float(np.mean(fold_values)) if fold_values else None,
            note="Per-fold model minus baseline on validation folds.",
            status="ok" if fold_values and float(np.mean(fold_values)) < 0 else ("na" if not fold_values else "warn"),
        ),
        _row(
            "fold_std_delta_logloss",
            "Fold std delta log loss",
            train_fit=float(np.std(fold_values, ddof=1)) if len(fold_values) > 1 else None,
            status="ok" if len(fold_values) > 1 else "na",
        ),
        _row(
            "fold_worst_delta_logloss",
            "Worst fold delta log loss",
            train_fit=float(np.max(fold_values)) if fold_values else None,
            status="warn" if fold_values and float(np.max(fold_values)) > 0 else ("na" if not fold_values else "ok"),
        ),
        _row(
            "folds_improved",
            "Improved folds",
            train_fit=float(sum(1 for value in fold_values if value < 0)) if fold_values else None,
            format_kind="count",
            note="Count of folds where model log loss beats the baseline.",
            status="ok" if fold_values else "na",
        ),
    ]

    requested_features = [str(name) for name in requested_numeric_features] + [str(name) for name in requested_categorical_features]
    used_features = {str(name) for name in numeric_features} | {str(name) for name in categorical_features}
    dropped_features = sorted([name for name in requested_features if name not in used_features])
    missingness = _missingness_summary(train_fit_df, list(used_features))
    vif = _numeric_vif_summary(train_fit_df, numeric_features)
    event_rate = (
        float(event_counts["train_fit"]) / float(split_counts["train_fit"])
        if split_counts.get("train_fit") and event_counts.get("train_fit") is not None
        else None
    )
    fit_warning_note, fit_warning_status = _fit_warning_summary([*trainer_warnings, *shadow_warnings])
    data_quality_rows = [
        _row(
            "n_obs",
            "Number of observations",
            train_fit=float(split_counts.get("train_fit")) if split_counts.get("train_fit") is not None else None,
            val=float(split_counts.get("val")) if split_counts.get("val") is not None else None,
            test=float(split_counts.get("test")) if split_counts.get("test") is not None else None,
            format_kind="count",
            status="ok",
        ),
        _row(
            "n_events",
            "Number of events",
            train_fit=float(event_counts.get("train_fit")) if event_counts.get("train_fit") is not None else None,
            val=float(event_counts.get("val")) if event_counts.get("val") is not None else None,
            test=float(event_counts.get("test")) if event_counts.get("test") is not None else None,
            format_kind="count",
            status="ok",
        ),
        _row(
            "n_non_events",
            "Number of non-events",
            train_fit=(float(split_counts["train_fit"] - event_counts["train_fit"]) if split_counts.get("train_fit") is not None and event_counts.get("train_fit") is not None else None),
            val=(float(split_counts["val"] - event_counts["val"]) if split_counts.get("val") is not None and event_counts.get("val") is not None else None),
            test=(float(split_counts["test"] - event_counts["test"]) if split_counts.get("test") is not None and event_counts.get("test") is not None else None),
            format_kind="count",
            status="ok",
        ),
        _row(
            "event_rate",
            "Event rate",
            train_fit=event_rate,
            val=(float(event_counts["val"]) / float(split_counts["val"]) if split_counts.get("val") and event_counts.get("val") is not None else None),
            test=(float(event_counts["test"]) / float(split_counts["test"]) if split_counts.get("test") and event_counts.get("test") is not None else None),
            format_kind="percent",
            status="warn" if event_rate is not None and (event_rate < 0.1 or event_rate > 0.9) else "ok",
        ),
        _row(
            "features_with_missing",
            "Features with missing values",
            train_fit=float(missingness["features_with_missing"]) if missingness.get("features_with_missing") is not None else None,
            format_kind="count",
            note=missingness.get("note"),
            status="warn" if missingness.get("features_with_missing") else "ok",
        ),
        _row(
            "max_missing_share",
            "Max missing share",
            train_fit=missingness.get("max_missing_share"),
            format_kind="percent",
            status="warn" if missingness.get("max_missing_share") is not None and float(missingness["max_missing_share"]) > 0.1 else "ok",
        ),
        _row(
            "dropped_features",
            "Dropped requested features",
            train_fit=float(len(dropped_features)),
            format_kind="count",
            note=", ".join(dropped_features[:8]) if dropped_features else "None.",
            status="warn" if dropped_features else "ok",
        ),
        _row(
            "max_numeric_vif",
            "Max numeric-feature VIF",
            train_fit=vif.get("max_vif"),
            note=vif.get("note"),
            status="warn" if vif.get("max_vif") is not None and float(vif["max_vif"]) >= 5.0 else "ok",
        ),
        _row(
            "n_high_vif",
            "Numeric features with VIF >= 5",
            train_fit=float(vif.get("n_high_vif")) if vif.get("n_high_vif") is not None else None,
            format_kind="count",
            status="warn" if vif.get("n_high_vif") else "ok",
        ),
        _row(
            "fit_warnings",
            "Fit / specification warnings",
            train_fit=float(len([message for message in [*trainer_warnings, *shadow_warnings] if str(message).strip()])),
            format_kind="count",
            note=fit_warning_note,
            status=fit_warning_status,
        ),
    ]

    preferred_split = "test" if split_counts.get("test") else "val"
    preferred_curve_rows = [
        {
            "bin": int(row["bin"]),
            "n": int(row["n"]),
            "series": str(row["series"]),
            "pred_mean": float(row["pred_mean"]),
            "obs_rate": float(row["obs_rate"]),
            "abs_gap": float(row["abs_gap"]),
        }
        for row in reliability_rows
        if row["split"] == preferred_split and row["series"] in {"model", "baseline"}
    ]

    header_notes = [
        "Inferential rows come from a shadow statsmodels Binomial GLM on the transformed train_fit design matrix.",
        "Predictive, probabilistic, and calibration rows use production sklearn probabilities.",
    ]
    if estimator == "sklearn_logit_platt":
        header_notes.append("Displayed coefficient and marginal-effect rows describe the base logistic layer, not the downstream Platt transform.")
    if shadow_failure_note:
        header_notes.append(shadow_failure_note)
    production_coefficient_table = build_production_coefficient_table(
        coefficients=production_coefficients or [],
        intercept=production_intercept,
        feature_names=production_feature_names or [],
    )

    payload = {
        "schema_version": 1,
        "header": {
            "model_id": str(model_id),
            "estimator": str(estimator),
            "inference_basis": "shadow_statsmodels_glm",
            "baseline_name": str(baseline_name),
            "best_c": _safe_float(best_c),
            "penalty": str(penalty),
            "solver": str(solver),
            "threshold_default": float(threshold_default),
            "threshold_operating": _safe_float(threshold_operating),
            "split_counts": split_counts,
            "event_counts": event_counts,
            "feature_counts": {
                "numeric": int(len(numeric_features)),
                "categorical": int(len(categorical_features)),
                "transformed": int(len(shadow_feature_names)),
            },
            "notes": header_notes,
        },
        "sections": [
            _section("global_significance", "Global model significance", global_rows),
            _section("coefficient_quality", "Coefficient quality", coefficient_summary_rows),
            _section("goodness_of_fit", "Goodness of fit", goodness_rows),
            _section("predictive_quality", "Predictive / classification", predictive_rows),
            _section("probabilistic_quality", "Probabilistic quality", probabilistic_rows),
            _section("calibration", "Calibration", calibration_rows),
            _section("stability_robustness", "Stability / robustness", robustness_rows),
            _section("data_specification_quality", "Data / specification quality", data_quality_rows),
        ],
        "production_coefficient_table": production_coefficient_table,
        "coefficient_table": {
            "basis": "shadow_statsmodels_glm",
            "fit_scope": "train_fit",
            "subtitle": SHADOW_COEFFICIENT_SUBTITLE,
            "rows": coefficient_rows,
        } if coefficient_rows else None,
        "marginal_effects_table": {
            "basis": "shadow_statsmodels_glm",
            "fit_scope": "train_fit",
            "subtitle": SHADOW_MARGINAL_EFFECTS_SUBTITLE,
            "rows": marginal_effect_rows,
        } if marginal_effect_rows else None,
        "calibration_curve": {
            "split": preferred_split,
            "binning": "equal_mass",
            "rows": preferred_curve_rows,
        } if preferred_curve_rows else None,
        "warnings": [str(message) for message in [*trainer_warnings, *shadow_warnings] if str(message).strip()],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diagnostics_table.json").write_text(json.dumps(payload, indent=2))
    _write_csv(out_dir / "coefficient_diagnostics.csv", coefficient_rows, COEFFICIENT_COLUMNS)
    _write_csv(out_dir / "marginal_effects.csv", marginal_effect_rows, MARGINAL_EFFECT_COLUMNS)
    _write_csv(out_dir / "reliability_bins.csv", reliability_rows, RELIABILITY_BIN_COLUMNS)
    return payload
