from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Optional, Sequence


FeatureKind = Literal["numeric", "categorical"]

OPTION_CHAIN_BASE_FEATURE = "x_logit_prn"


@dataclass(frozen=True)
class OptionChainSelectableFeature:
    name: str
    label: str
    kind: FeatureKind
    group: str
    order: int
    default_selected: bool = False
    mutex_group: Optional[str] = None


OPTION_CHAIN_SELECTABLE_FEATURES: tuple[OptionChainSelectableFeature, ...] = (
    OptionChainSelectableFeature("log_m", "Spot log moneyness", "numeric", "Moneyness", 10, mutex_group="spot_m"),
    OptionChainSelectableFeature("abs_log_m", "Spot abs log moneyness", "numeric", "Moneyness", 20, mutex_group="spot_m"),
    OptionChainSelectableFeature(
        "log_m_fwd",
        "Forward log moneyness",
        "numeric",
        "Moneyness",
        30,
        default_selected=True,
        mutex_group="forward_m",
    ),
    OptionChainSelectableFeature(
        "abs_log_m_fwd",
        "Forward abs log moneyness",
        "numeric",
        "Moneyness",
        40,
        mutex_group="forward_m",
    ),
    OptionChainSelectableFeature("rv5", "RV5", "numeric", "Volatility", 50),
    OptionChainSelectableFeature("rv10", "RV10", "numeric", "Volatility", 60),
    OptionChainSelectableFeature("rv20", "RV20", "numeric", "Volatility", 70, default_selected=True),
    OptionChainSelectableFeature("rv5_over_rv10", "RV5 / RV10", "numeric", "Volatility", 80),
    OptionChainSelectableFeature("rv5_over_rv20", "RV5 / RV20", "numeric", "Volatility", 90),
    OptionChainSelectableFeature("rv10_over_rv20", "RV10 / RV20", "numeric", "Volatility", 100),
    OptionChainSelectableFeature(
        "rel_spread_median",
        "Relative spread median",
        "numeric",
        "Pricing",
        110,
        default_selected=True,
    ),
    OptionChainSelectableFeature(
        "dividend_yield",
        "Dividend yield",
        "numeric",
        "Pricing",
        120,
        default_selected=True,
    ),
    OptionChainSelectableFeature(
        "spot_scale_used",
        "Spot split scale adjusted",
        "categorical",
        "Categorical",
        300,
    ),
)

OPTION_CHAIN_FEATURE_METADATA = {feature.name: feature for feature in OPTION_CHAIN_SELECTABLE_FEATURES}
OPTION_CHAIN_NUMERIC_FEATURES = tuple(
    feature.name for feature in OPTION_CHAIN_SELECTABLE_FEATURES if feature.kind == "numeric"
)
OPTION_CHAIN_CATEGORICAL_FEATURES = tuple(
    feature.name for feature in OPTION_CHAIN_SELECTABLE_FEATURES if feature.kind == "categorical"
)
OPTION_CHAIN_DEFAULT_OPTIONAL_FEATURES = tuple(
    feature.name for feature in OPTION_CHAIN_SELECTABLE_FEATURES if feature.default_selected
)
OPTION_CHAIN_DEFAULT_FEATURES = (
    OPTION_CHAIN_BASE_FEATURE,
    *OPTION_CHAIN_DEFAULT_OPTIONAL_FEATURES,
)
OPTION_CHAIN_AUTO_FEATURE_SETS = (
    (OPTION_CHAIN_BASE_FEATURE,),
    (OPTION_CHAIN_BASE_FEATURE, "rv20"),
    (OPTION_CHAIN_BASE_FEATURE, "abs_log_m_fwd"),
    (OPTION_CHAIN_BASE_FEATURE, "rv20", "abs_log_m_fwd"),
    (OPTION_CHAIN_BASE_FEATURE, "rv20", "abs_log_m_fwd", "rel_spread_median"),
)


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def option_chain_selectable_feature_payloads(
    available_columns: Sequence[str],
) -> list[dict[str, object]]:
    available = set(str(column).strip() for column in available_columns)
    return [
        asdict(feature)
        for feature in OPTION_CHAIN_SELECTABLE_FEATURES
        if feature.name in available
    ]


def normalize_option_chain_numeric_features(
    features: Iterable[str],
    *,
    include_base_feature: bool = True,
) -> list[str]:
    allowed = set(OPTION_CHAIN_NUMERIC_FEATURES)
    if include_base_feature:
        allowed.add(OPTION_CHAIN_BASE_FEATURE)
    return [feature for feature in _dedupe_preserve_order(features) if feature in allowed]


def normalize_option_chain_categorical_features(features: Iterable[str]) -> list[str]:
    allowed = set(OPTION_CHAIN_CATEGORICAL_FEATURES)
    return [feature for feature in _dedupe_preserve_order(features) if feature in allowed]


def validate_option_chain_feature_selection(
    *,
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    available_columns: Sequence[str],
    allow_base_feature: bool = True,
) -> tuple[list[str], list[str]]:
    requested_numeric = _dedupe_preserve_order(numeric_features)
    requested_categorical = _dedupe_preserve_order(categorical_features)
    allowed_numeric = set(OPTION_CHAIN_NUMERIC_FEATURES)
    if allow_base_feature:
        allowed_numeric.add(OPTION_CHAIN_BASE_FEATURE)
    allowed_categorical = set(OPTION_CHAIN_CATEGORICAL_FEATURES)
    available = set(str(column).strip() for column in available_columns)

    invalid_numeric = [feature for feature in requested_numeric if feature not in allowed_numeric]
    invalid_categorical = [
        feature for feature in requested_categorical if feature not in allowed_categorical
    ]
    if invalid_numeric:
        raise ValueError(
            "Unsupported numeric features requested: " + ", ".join(invalid_numeric) + "."
        )
    if invalid_categorical:
        raise ValueError(
            "Unsupported categorical features requested: " + ", ".join(invalid_categorical) + "."
        )

    mutex_conflicts: list[list[str]] = []
    mutex_members: dict[str, list[str]] = {}
    for feature in requested_numeric:
        metadata = OPTION_CHAIN_FEATURE_METADATA.get(feature)
        if metadata is None or not metadata.mutex_group:
            continue
        members = mutex_members.setdefault(metadata.mutex_group, [])
        members.append(feature)
    for members in mutex_members.values():
        if len(members) > 1:
            mutex_conflicts.append(members)
    if mutex_conflicts:
        raise ValueError(
            "Mutually exclusive numeric features requested: "
            + "; ".join(", ".join(conflict) for conflict in mutex_conflicts)
            + "."
        )

    missing_numeric = [
        feature
        for feature in requested_numeric
        if feature != OPTION_CHAIN_BASE_FEATURE and feature not in available
    ]
    missing_categorical = [
        feature for feature in requested_categorical if feature not in available
    ]
    if missing_numeric:
        raise ValueError(
            "Requested numeric features are not present in the training CSV: "
            + ", ".join(missing_numeric)
            + "."
        )
    if missing_categorical:
        raise ValueError(
            "Requested categorical features are not present in the training CSV: "
            + ", ".join(missing_categorical)
            + "."
        )

    return requested_numeric, requested_categorical
