# scripts/run_mp2d.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from eyebrow.pipeline import BackendSpec, RunConfig, run_pipeline
from eyebrow.backends.mp2d import extract_video_mp2d


def parse_people(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return ["p1", "p2", "p3", "p4"]
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()

    # Batch mode
    ap.add_argument("--experiment", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--people", type=str, default="p2")

    # Base directories
    ap.add_argument("--control_dir", type=str, default="datasets/own_data/controlled_video")
    ap.add_argument("--target_dir", type=str, default="datasets/own_data/target_video_exp{X}")
    ap.add_argument("--out_dir", type=str, default="results/exp{X}")

    # Hyperparameters
    ap.add_argument("--t0", type=float, default=7.0)
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--ridge_alpha", type=float, default=1e-3)
    ap.add_argument("--outer_eye_mm", type=float, default=90.0)
    ap.add_argument("--motion_segments", type=str, default="")

    args = ap.parse_args()

    people = parse_people(args.people)
    exp = args.experiment

    control_dir = Path(args.control_dir)
    target_dir = Path(args.target_dir.format(X=exp))
    out_dir = Path(args.out_dir.format(X=exp))
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RunConfig(
        t0=args.t0,
        degree=args.degree,
        ridge_alpha=args.ridge_alpha,
        outer_eye_mm=args.outer_eye_mm,
        experiment=exp,
        motion_segments=args.motion_segments,
    )

    # Distance columns
    d_cols_px = [
        "L_all_mean", "R_all_mean",
        "L_inner_mean", "L_outer_mean",
        "R_inner_mean", "R_outer_mean"
    ]
    d_cols_norm = [c + "_norm" for c in d_cols_px]

    # Two variants: F (no scale feature) and G (with scale feature)
    use_scale = False

    for p in people:

        ext = 'mov' if p=='p1' else 'mp4'
        control_video = control_dir / f"{p}_control.{ext}"
        target_video = target_dir / f"{p}_exp{exp}.{ext}"

        if not control_video.exists():
            print(f"[WARN] Control not found: {control_video} (skipping {p})")
            continue
        if not target_video.exists():
            print(f"[WARN] Target not found: {target_video} (skipping {p})")
            continue

        feature_cols_norm = ["pitch", "yaw", "roll", "scale"] if use_scale else ["pitch", "yaw", "roll"]

        backend = BackendSpec(
            name="MP2D",
            extract=extract_video_mp2d,
            d_cols_norm=d_cols_norm,
            feature_cols_norm=feature_cols_norm,
            mm_factor_mode="quiet_scale",
            mm_scale_col="scale_norm",
            norm_to_mm_replace=("_norm", "_mm_const"),
        )

        out_csv = out_dir / f"{p}_exp{exp}_MPP2D_F.csv"

        print(f"\n=== {p} | exp{exp}  ===")
        print(f"control: {control_video}")
        print(f"target : {target_video}")
        print(f"out    : {out_csv}")

        run_pipeline(
            control_video=str(control_video),
            target_video=str(target_video),
            out_csv=str(out_csv),
            backend=backend,
            cfg=cfg,
        )


if __name__ == "__main__":
    main()


'''
cd /Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code_refactor

PYTHONPATH=src nohup python scripts/run_mp2d.py --experiment 3 \
  > results/logs/run_mp2d_exp3.log 2>&1 &


PYTHONPATH=src python scripts/run_mp2d.py --experiment 1 \



  --control_dir datasets/own_data/controlled_video/p2_control.mp4 \
  --target_dir datasets/own_data/target_video_exp1/p2_exp1.mp4 \
  --outer_eye_mm 90.0 \
  --experiment 1 \
  --out_dir results/exp1/p2_exp1_MP222D_F.csv


   --norm_use_scale \
'''