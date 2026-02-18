# src/eyebrow/correction.py
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge


def compute_d0(df_control: pd.DataFrame, t0: float, cols: List[str]) -> Dict[str, float]:
    quiet = df_control[(df_control["time_s"] >= 0.0) & (df_control["time_s"] < t0)].copy()
    d0: Dict[str, float] = {}
    for c in cols:
        if c not in quiet.columns:
            d0[c] = float("nan")
            continue
        s = pd.to_numeric(quiet[c], errors="coerce").dropna()
        d0[c] = float(s.mean()) if len(s) else float("nan")
    return d0


def fit_pose_bias(
    df_control: pd.DataFrame,
    t0: float,
    d_cols: List[str],
    feature_cols: List[str],
    degree: int = 2,
    ridge_alpha: float = 1e-3,
    min_rows: int = 20,
) -> Dict[str, Tuple[PolynomialFeatures, Ridge]]:
    move = df_control[df_control["time_s"] >= t0].copy()
    models: Dict[str, Tuple[PolynomialFeatures, Ridge]] = {}

    for c in d_cols:
        if c not in move.columns:
            continue

        sub = move.dropna(subset=[c] + feature_cols).copy()
        if len(sub) < min_rows:
            raise RuntimeError(f"Not enough valid rows to fit model for {c} (got {len(sub)}).")

        X = sub[feature_cols].values.astype(np.float64)
        y = sub[c].values.astype(np.float64)

        poly = PolynomialFeatures(degree=degree, include_bias=False)
        Xp = poly.fit_transform(X)

        model = Ridge(alpha=ridge_alpha, fit_intercept=False)
        model.fit(Xp, y)

        models[c] = (poly, model)

    return models


def apply_correction(
    df: pd.DataFrame,
    d0: Dict[str, float],
    models: Dict[str, Tuple[PolynomialFeatures, Ridge]],
    d_cols: List[str],
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Adds:
      - {c}_d0   = c - d0[c]
      - {c}_corr = (c - y_hat) + d0[c]
    """
    out = df.copy()

    X = out[feature_cols].values.astype(np.float64) if all(fc in out.columns for fc in feature_cols) else None
    valid = np.zeros(len(out), dtype=bool) if X is None else np.all(np.isfinite(X), axis=1)

    for c in d_cols:
        out[c + "_d0"] = np.nan
        out[c + "_corr"] = np.nan

        if c not in out.columns or c not in models:
            continue
        if c not in d0 or not np.isfinite(float(d0[c])):
            continue

        out.loc[:, c + "_d0"] = out[c] - float(d0[c])

        poly, model = models[c]
        y_hat = np.full(len(out), np.nan, dtype=np.float64)
        if X is not None:
            y_hat[valid] = model.predict(poly.transform(X[valid]))

        out.loc[valid, c + "_corr"] = (out.loc[valid, c].values - y_hat[valid]) + float(d0[c])

    return out
