# src/eyebrow/units.py
from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd


def mm_per_from_quiet(df_control: pd.DataFrame, t0: float, outer_eye_mm: float, scale_col: str) -> float:
    quiet = df_control[(df_control["time_s"] >= 0.0) & (df_control["time_s"] < t0)].copy()
    if scale_col not in quiet.columns:
        raise RuntimeError(f"Missing '{scale_col}' in control extraction.")
    s = pd.to_numeric(quiet[scale_col], errors="coerce").dropna()
    if len(s) == 0:
        raise RuntimeError(f"No valid '{scale_col}' values in quiet segment; cannot compute constant scale.")
    mean_scale = float(s.mean())
    if not np.isfinite(mean_scale) or mean_scale <= 1e-12:
        raise RuntimeError(f"Invalid mean {scale_col} in quiet segment; cannot compute constant scale.")
    return float(outer_eye_mm) / mean_scale


def add_scaled(df: pd.DataFrame, cols: List[str], factor: float, out_suffix: str) -> pd.DataFrame:
    """
    Adds: <col><out_suffix> = <col> * factor
    Example: L_all_mean_norm -> L_all_mean_mm_const  (if you name it that way outside)
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        out[c + out_suffix] = out[c].astype(float) * float(factor)
    return out


def add_scaled_replace_suffix(
    df: pd.DataFrame,
    cols: List[str],
    factor: float,
    replace_from: str,
    replace_to: str,
) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        out[c.replace(replace_from, replace_to)] = out[c].astype(float) * float(factor)
    return out
