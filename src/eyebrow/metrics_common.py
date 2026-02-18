# src/eyebrow/metrics_common.py
from __future__ import annotations
import math
from typing import List, Tuple
import numpy as np
import pandas as pd


def safe_mean(values: List[float]) -> float:
    values = [v for v in values if np.isfinite(v)]
    return float("nan") if len(values) == 0 else float(np.mean(values))


def safe_std(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.std()) if len(s) else float("nan")


def safe_range(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.max() - s.min()) if len(s) else float("nan")


def split_inner_outer(dists: List[float]) -> Tuple[List[float], List[float]]:
    """
    Split list ordered inner->outer into two halves (ceil(n/2) each):
      inner = first k
      outer = last k
    """
    n = len(dists)
    if n <= 1:
        return dists, dists
    k = (n + 1) // 2
    return dists[:k], dists[-k:]


def summarize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].dropna()
        if len(s) == 0:
            continue
        rows.append(
            {
                "col": c,
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
                "range": float(s.max() - s.min()),
            }
        )
    return pd.DataFrame(rows)
