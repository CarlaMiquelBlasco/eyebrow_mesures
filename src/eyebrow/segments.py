# src/eyebrow/segments.py
from __future__ import annotations
import re
from typing import Dict, List
import numpy as np
import pandas as pd


def parse_motion_segments(spec: str) -> List[Dict[str, float]]:
    """
    Parse segments like:
      "pitch:7-12,yaw:12-17,roll:17-22"
    Returns list of dicts: {"label": "pitch", "t_start": 7.0, "t_end": 12.0}
    """
    if spec is None or str(spec).strip() == "":
        return []

    out = []
    items = [x.strip() for x in spec.split(",") if x.strip()]
    pat = re.compile(r"^(pitch|yaw|roll)\s*:\s*([0-9]*\.?[0-9]+)\s*-\s*([0-9]*\.?[0-9]+)\s*$", re.IGNORECASE)

    for it in items:
        m = pat.match(it)
        if not m:
            print(f"[WARNING] Could not parse motion segment '{it}'. Expected like 'pitch:7-12'. Skipping.")
            continue
        label = m.group(1).lower()
        t_start = float(m.group(2))
        t_end = float(m.group(3))
        if not (np.isfinite(t_start) and np.isfinite(t_end)):
            continue
        if t_end <= t_start:
            continue
        out.append({"label": label, "t_start": t_start, "t_end": t_end})

    out.sort(key=lambda d: d["t_start"])
    return out


def slice_segment(df: pd.DataFrame, t_start: float, t_end: float) -> pd.DataFrame:
    if "time_s" not in df.columns:
        return df.iloc[0:0].copy()
    return df[(df["time_s"] >= float(t_start)) & (df["time_s"] < float(t_end))].copy()
