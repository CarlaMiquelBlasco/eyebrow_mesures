# src/eyebrow/agg.py
from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd
from .metrics_common import safe_std, safe_range


def _safe_mean(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.mean()) if len(s) else float("nan")


def corrected_col_name(measure: str) -> str:
    """
    Convention:
      raw:  X
      corr: X_corr

    Special for mm_const:
      raw:  X_mm_const
      corr: X_corr_mm_const
    """
    if measure.endswith("_mm_const"):
        return measure.replace("_mm_const", "_corr_mm_const")
    return measure + "_corr"


def build_agg_experiment_1(
    df: pd.DataFrame,
    measures: List[str],
    d0_map: Dict[str, float],
) -> pd.DataFrame:
    """
    Exp1 columns:
      Measure, d0, dist_raw, dist_corrected, |dist_raw−d0|, |dist_corrected−d0|,
      std_raw, std_corrected, range_raw, range_corrected
    """
    rows = []
    for m in measures:
        if m not in df.columns:
            continue
        mc = corrected_col_name(m)
        if mc not in df.columns:
            continue

        d0 = float(d0_map.get(m, float("nan")))
        raw = df[m].astype(float)
        corr = df[mc].astype(float)

        dist_raw = _safe_mean(raw)
        dist_corr = _safe_mean(corr)

        rows.append(
            {
                "Measure": m,
                "d0": d0,
                "dist_raw": dist_raw,
                "dist_corrected": dist_corr,
                "|dist_raw−d0|": abs(dist_raw - d0) if np.isfinite(dist_raw) and np.isfinite(d0) else float("nan"),
                "|dist_corrected−d0|": abs(dist_corr - d0) if np.isfinite(dist_corr) and np.isfinite(d0) else float("nan"),
                "std_raw": safe_std(raw),
                "std_corrected": safe_std(corr),
                "range_raw": safe_range(raw),
                "range_corrected": safe_range(corr),
            }
        )
    return pd.DataFrame(rows)


def build_agg_experiment_2(
    df: pd.DataFrame,
    measures: List[str],
    d0_map: Dict[str, float],
) -> pd.DataFrame:
    """
    Exp2 columns:
      Measure, d0, dist_raw, dist_corrected, dist_factor,
      std_raw, std_corrected, std_factor,
      range_raw, range_corrected, range_factor

    NOTE: matches your current behavior: dist_factor = dist_raw / dist_corrected
    """
    rows = []
    for m in measures:
        if m not in df.columns:
            continue
        mc = corrected_col_name(m)
        if mc not in df.columns:
            continue

        d0 = float(d0_map.get(m, float("nan")))
        raw = df[m].astype(float)
        corr = df[mc].astype(float)

        dist_raw = _safe_mean(raw)
        dist_corr = _safe_mean(corr)

        std_raw = safe_std(raw)
        std_corr = safe_std(corr)
        range_raw = safe_range(raw)
        range_corr = safe_range(corr)

        rows.append(
            {
                "Measure": m,
                "d0": d0,
                "dist_raw": dist_raw,
                "dist_corrected": dist_corr,
                "dist_factor": (dist_raw / dist_corr) if np.isfinite(dist_raw) and np.isfinite(dist_corr) and abs(dist_corr) > 1e-12 else float("nan"),
                "std_raw": std_raw,
                "std_corrected": std_corr,
                "std_factor": (std_raw / std_corr) if np.isfinite(std_raw) and np.isfinite(std_corr) and abs(std_corr) > 1e-12 else float("nan"),
                "range_raw": range_raw,
                "range_corrected": range_corr,
                "range_factor": (range_raw / range_corr) if np.isfinite(range_raw) and np.isfinite(range_corr) and abs(range_corr) > 1e-12 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_agg_experiment_3(
    df: pd.DataFrame,
    measures_mm_const: List[str],
    pose_features: List[str],
    min_rows: int = 20,
) -> pd.DataFrame:
    """
    Your latest exp3:
      mean_abs_corr_raw        = | mean_f corr(raw, f) |
      mean_abs_corr_corrected  = | mean_f corr(corrected, f) |
      metric                   = mean_abs_corr_raw - mean_abs_corr_corrected
    """
    rows = []
    for m in measures_mm_const:
        if m not in df.columns:
            continue
        mc = corrected_col_name(m)
        if mc not in df.columns:
            continue

        corr_raw_list = []
        corr_corr_list = []

        for f in pose_features:
            if f not in df.columns:
                continue
            sub = df[[m, mc, f]].dropna()
            if len(sub) < min_rows:
                continue

            cr = float(sub[m].corr(sub[f]))
            cc = float(sub[mc].corr(sub[f]))
            if np.isfinite(cr):
                corr_raw_list.append(cr)
            if np.isfinite(cc):
                corr_corr_list.append(cc)

        if len(corr_raw_list) == 0 or len(corr_corr_list) == 0:
            mean_abs_raw = float("nan")
            mean_abs_corr = float("nan")
            metric = float("nan")
            n_used = 0
        else:
            mean_abs_raw = abs(float(np.mean(corr_raw_list)))
            mean_abs_corr = abs(float(np.mean(corr_corr_list)))
            metric = mean_abs_raw - mean_abs_corr
            n_used = min(len(corr_raw_list), len(corr_corr_list))

        rows.append(
            {
                "Measure": m,
                "N_features_used": int(n_used),
                "mean_abs_corr_raw": mean_abs_raw,
                "mean_abs_corr_corrected": mean_abs_corr,
                "metric": metric,
            }
        )

    return pd.DataFrame(rows)
