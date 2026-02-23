#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from eyebrow.pipeline import BackendSpec, RunConfig, run_pipeline
from eyebrow.backends.tddfa3d import extract_video_3ddfa3d_norm


def parse_people(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return ["p1", "p2", "p3", "p4"]
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()

    # 3DDFA settings
    ap.add_argument(
        "--tddfa_repo",
        required=False,
        default=None,
        help="Path to 3DDFA_V2 repo (or set env TDDFA_V2_REPO).",
    )
    ap.add_argument("--no_onnx", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--fail_examples_dir", type=str, default="")
    ap.add_argument("--max_fail_examples_per_status", type=int, default=30)

    # Batch mode
    ap.add_argument("--experiment", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--people", type=str, default="p1,p2,p3,p4")

    # Base directories
    ap.add_argument("--control_dir", type=str, default="/Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code_refactor/datasets/own_data/controlled_video")
    ap.add_argument("--target_dir", type=str, default="/Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code_refactor/datasets/own_data/target_video_exp{X}")
    ap.add_argument("--out_dir", type=str, default="/Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code_refactor/results/exp{X}_v6")

    # Pipeline params
    ap.add_argument("--t0", type=float, default=7.0)
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--ridge_alpha", type=float, default=1e-3)
    ap.add_argument("--outer_eye_mm", type=float, default=90.0)
    ap.add_argument("--motion_segments", type=str, default="")

    # Optional features
    ap.add_argument(
        "--use_scale3d_feature",
        action="store_true",
        help="If set, include scale_3d as a feature in correction model.",
    )
    ap.add_argument(
        "--use_scale_px_feature",
        action="store_true",
        help="If set, include scale_px as a feature in correction model.",
    )

    args = ap.parse_args()

    use_onnx = not args.no_onnx
    people = parse_people(args.people)
    exp = args.experiment

    control_dir = Path(args.control_dir)
    target_dir = Path(args.target_dir.format(X=exp))
    out_dir = Path(args.out_dir.format(X=exp))
    out_dir.mkdir(parents=True, exist_ok=True)

    d_cols_3d_norm = [
        "L_all_mean_3d_norm", "R_all_mean_3d_norm",
        "L_inner_mean_3d_norm", "L_outer_mean_3d_norm",
        "R_inner_mean_3d_norm", "R_outer_mean_3d_norm",
    ]

    feature_cols = ["pitch", "yaw", "roll"]
    if args.use_scale3d_feature:
        feature_cols.append("scale_3d")
    if args.use_scale_px_feature:
        feature_cols.append("scale_px")

    backend = BackendSpec(
        name="TDDFA_3D",
        extract=extract_video_3ddfa3d_norm,
        extract_kwargs={
            "tddfa_repo": args.tddfa_repo,
            "use_onnx": use_onnx,
            "device": args.device,
            "fail_examples_dir": (args.fail_examples_dir.strip() or None),
            "max_fail_examples_per_status": args.max_fail_examples_per_status,
        },
        d_cols_norm=d_cols_3d_norm,
        feature_cols_norm=feature_cols,
        mm_factor_mode="direct",          # mm = norm * outer_eye_mm
        mm_scale_col=None,
        norm_to_mm_replace=("_3d_norm", "_3d_mm_const"),
    )

    cfg = RunConfig(
        t0=args.t0,
        degree=args.degree,
        ridge_alpha=args.ridge_alpha,
        outer_eye_mm=args.outer_eye_mm,
        experiment=exp,
        motion_segments=args.motion_segments,
    )

    for p in people:
        ext = ".mov" if p == "p1" else ".mp4"

        control_video = control_dir / f"{p}_control{ext}"
        target_video = target_dir / f"{p}_exp{exp}{ext}"

        if not control_video.exists():
            print(f"[WARN] Control not found: {control_video} (skipping {p})")
            continue
        if not target_video.exists():
            print(f"[WARN] Target not found: {target_video} (skipping {p})")
            continue

        out_csv = out_dir / f"{p}_exp{exp}_TDDFA_3D_F.csv"

        print(f"\n=== {p} | exp{exp} | TDDFA_3D ===")
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
cd /Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code/3DDFAV2/3DDFA_V2

source /Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code/3DDFAV2/3ddfav2/bin/activate


PYTHONPATH=/Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code_refactor/src  nohup python /Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code_refactor/scripts/run_3ddfa3d.py \
  --tddfa_repo . \
  --experiment 1 \
  --outer_eye_mm 90 \
  > /Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code_refactor/results/logs/run_3ddfa3d_exp1_v6.log 2>&1 &



PYTHONPATH=/Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code_refactor/src \
python /Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code_refactor/scripts/run_3ddfa3d.py \
  --tddfa_repo . \
  --control_video /Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code_refactor/datasets/own_data/controlled_video/p1_control.mov \
  --target_video /Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code_refactor/datasets/own_data/target_video_exp3/p1_exp3.mov \
  --t0 7 --degree 2 --ridge_alpha 1e-3 \
  --outer_eye_mm 90 \
  --experiment 3 \
  --out_csv /Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code_refactor/results/exp3/p1_exp3_TDDFA_3D_F.csv

    --use_scale_norm_feature \
  '''