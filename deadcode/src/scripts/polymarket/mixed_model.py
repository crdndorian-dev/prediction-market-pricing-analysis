"""
Helpers for mixed pRN + Polymarket models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import joblib


@dataclass
class MixedModelBundle:
    model_type: str               # "residual" or "blend"
    model: Any                    # sklearn pipeline/regressor
    feature_cols: List[str]
    pm_col: str = "pm_mid"
    prn_col: str = "pRN"
    clip: Tuple[float, float] = (0.0, 1.0)

    def predict_p(self, df: pd.DataFrame) -> np.ndarray:
        if self.model_type not in {"residual", "blend"}:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        X = df[self.feature_cols]
        pred = self.model.predict(X)

        if self.model_type == "residual":
            base = pd.to_numeric(df[self.pm_col], errors="coerce").to_numpy()
            out = base + pred
        else:
            w = np.clip(pred, 0.0, 1.0)
            prn = pd.to_numeric(df[self.prn_col], errors="coerce").to_numpy()
            pm = pd.to_numeric(df[self.pm_col], errors="coerce").to_numpy()
            out = w * prn + (1.0 - w) * pm

        return np.clip(out, self.clip[0], self.clip[1])


def save_bundle(bundle: MixedModelBundle, path: str) -> None:
    joblib.dump(bundle, path)


def load_bundle(path: str) -> MixedModelBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, MixedModelBundle):
        raise ValueError("Model artifact did not resolve to MixedModelBundle.")
    return bundle
