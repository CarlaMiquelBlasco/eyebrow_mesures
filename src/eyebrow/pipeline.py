# src/eyebrow/pipeline.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import re
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any

from .correction import compute_d0, fit_pose_bias, apply_correction
from .units import mm_per_from_quiet, add_scaled_replace_suffix
from .metrics_common import summarize
from .agg import build_agg_experiment_1, build_agg_experiment_2, build_agg_experiment_3
from .segments import parse_motion_segments, slice_segment


@dataclass
class RunConfig:
    t0: float
    degree: int
    ridge_alpha: float
    outer_eye_mm: float
    experiment: int
    motion_segments: str = ""


@dataclass
class BackendSpec:
    name: str
    extract: Callable[[str], pd.DataFrame]

    # Train + correct ONLY on these norm measures
    d_cols_norm: List[str]

    # Features used for correction
    feature_cols_norm: List[str]

    # How to turn norm -> mm_const_from_norm
    # Either:
    #   (A) factor from quiet segment (2D): factor = outer_eye_mm / mean(scale_norm quiet)
    #   (B) direct factor (3D): factor = outer_eye_mm
    mm_factor_mode: str                 # "quiet_scale" or "direct"
    mm_scale_col: Optional[str] = None  # required if mm_factor_mode == "quiet_scale"
    mm_direct_factor: Optional[float] = None  # if mm_factor_mode == "direct", typically set at runtime to outer_eye_mm

    # Naming: replace "_norm" -> "_mm_const"
    norm_to_mm_replace: Tuple[str, str] = ("_norm", "_mm_const")
    extract_kwargs: Dict[str, Any] = field(default_factory=dict)


def run_pipeline(
    control_video: str,
    target_video: str,
    out_csv: str,
    backend: BackendSpec,
    cfg: RunConfig,
) -> None:
    # 1) Extract
    df_control = backend.extract(control_video, **backend.extract_kwargs)
    df_target = backend.extract(target_video, **backend.extract_kwargs)

    # 2) d0 (norm only, mm only for reporting)
    d0_norm = compute_d0(df_control, cfg.t0, backend.d_cols_norm)

    # 3) mm factor for norm -> mm_const_from_norm
    if backend.mm_factor_mode == "quiet_scale":
        if not backend.mm_scale_col:
            raise RuntimeError(f"[{backend.name}] mm_scale_col is required for quiet_scale mode.")
        mm_factor = mm_per_from_quiet(df_control, cfg.t0, cfg.outer_eye_mm, backend.mm_scale_col)
    elif backend.mm_factor_mode == "direct":
        # for 3D_norm: mm = norm * outer_eye_mm
        mm_factor = float(cfg.outer_eye_mm)
    else:
        raise RuntimeError(f"[{backend.name}] Unknown mm_factor_mode: {backend.mm_factor_mode}")

    print(f"[{backend.name}] mm_factor for norm->mm_const: {mm_factor:.10f}")

    # 4) Fit pose bias model (norm only)
    models_norm = fit_pose_bias(
        df_control,
        t0=cfg.t0,
        d_cols=backend.d_cols_norm,
        feature_cols=backend.feature_cols_norm,
        degree=cfg.degree,
        ridge_alpha=cfg.ridge_alpha,
    )

    # 5) Apply correction (norm only)
    df_out = apply_correction(
        df=df_target,
        d0=d0_norm,
        models=models_norm,
        d_cols=backend.d_cols_norm,
        feature_cols=backend.feature_cols_norm,
    )

    # 6) Add mm_const_from_norm for raw and corrected norm
    # raw norm -> mm_const
    df_out = add_scaled_replace_suffix(
        df_out,
        cols=backend.d_cols_norm,
        factor=mm_factor,
        replace_from=backend.norm_to_mm_replace[0],
        replace_to=backend.norm_to_mm_replace[1],
    )
    # corrected norm -> corr_mm_const
    df_out = add_scaled_replace_suffix(
        df_out,
        cols=[c + "_corr" for c in backend.d_cols_norm],
        factor=mm_factor,
        replace_from=backend.norm_to_mm_replace[0] + "_corr",
        replace_to=backend.norm_to_mm_replace[1].replace("_mm_const", "_corr_mm_const"),
    )

    # 7) Save framewise CSV
    df_out.to_csv(out_csv, index=False)
    print(f"[{backend.name}] Saved framewise CSV: {out_csv}")

    # 8) Aggregated CSV (only norm + mm_const_from_norm)
    agg_out_path = re.sub(r"\_F.csv$", "", out_csv, flags=re.IGNORECASE) + f"_G.csv"

    measures_exp12 = []
    for c in backend.d_cols_norm:
        if c in df_out.columns:
            measures_exp12.append(c)
        mmc = c.replace("_norm", "_mm_const")
        if mmc in df_out.columns:
            measures_exp12.append(mmc)

    measures_exp3 = [m for m in measures_exp12 if m.endswith("_mm_const")]

    # Build d0 map for measures we report:
    d0_map = {}
    for c in backend.d_cols_norm:
        d0_map[c] = float(d0_norm.get(c, float("nan")))
        d0_map[c.replace("_norm", "_mm_const")] = float(d0_norm.get(c, float("nan"))) * mm_factor

    if cfg.experiment == 1:
        agg_df = build_agg_experiment_1(df_out, measures_exp12, d0_map)
    elif cfg.experiment == 2:
        agg_df = build_agg_experiment_2(df_out, measures_exp12, d0_map)
    else:
        pose_features = [c for c in ["pitch", "yaw", "roll", "scale", "scale_px", "scale_3d"] if c in df_out.columns]
        agg_df = build_agg_experiment_3(df_out, measures_exp3, pose_features)

    agg_df.to_csv(agg_out_path, index=False)
    print(f"[{backend.name}] Saved aggregated CSV: {agg_out_path}")

    # 9) Print summary (only norm + mm_const_from_norm)
    report_cols = []
    for c in backend.d_cols_norm:
        report_cols += [c, c + "_corr", c.replace("_norm", "_mm_const"), c.replace("_norm", "_corr_mm_const")]
    report_cols = [c for c in report_cols if c in df_out.columns]

    print(f"\n[{backend.name}] Summary (norm + mm_const_from_norm):")
    print(summarize(df_out, report_cols).to_string(index=False))

    # 10) Per-motion segments optional
    segments = parse_motion_segments(cfg.motion_segments)
    if segments:
        print(f"\n[{backend.name}] PER-MOTION reporting")
        for seg in segments:
            df_seg = slice_segment(df_out, seg["t_start"], seg["t_end"])
            print(f"\n  Segment '{seg['label']}' t=[{seg['t_start']}, {seg['t_end']}) frames={len(df_seg)}")
            print(summarize(df_seg, report_cols).to_string(index=False))
