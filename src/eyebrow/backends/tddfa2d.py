#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3DDFA_V2 backend (2D): extract eyebrow-eye distances in NORMALIZED coords only.

Outputs per frame:
- time_s, frame, w, h
- pitch, yaw, roll
- scale_norm  (outer-eye distance in normalized coords)  <-- used for mm_from_norm (quiet scaling)
- scale_px    (optional debug)
- L/R distances in *_norm columns

Runs from anywhere if you pass tddfa_repo or set env TDDFA_V2_REPO.
"""

from __future__ import annotations

import os
import sys
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Numpy compatibility (some 3DDFA setups still reference np.int/np.long)
# ---------------------------------------------------------------------
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "long"):
    np.long = np.int64

# =============================================================================
# 68-landmark indices (common 68-point facial landmark scheme)
# =============================================================================
EYE_68 = {"L_OUTER": 36, "L_INNER": 39, "R_INNER": 42, "R_OUTER": 45}
BROW_68 = {"L": [17, 18, 19, 20, 21], "R": [22, 23, 24, 25, 26]}


# =============================================================================
# Geometry helpers
# =============================================================================
def signed_point_line_distance(p: Tuple[float, float],
                               a: Tuple[float, float],
                               b: Tuple[float, float]) -> float:
    """Signed distance from point p to directed line a->b in 2D."""
    (px, py), (ax, ay), (bx, by) = p, a, b
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = math.hypot(vx, vy)
    if denom < 1e-12:
        return float("nan")
    return (vx * wy - vy * wx) / denom


def safe_mean(values: List[float]) -> float:
    values = [v for v in values if np.isfinite(v)]
    return float("nan") if len(values) == 0 else float(np.mean(values))


def split_inner_outer(dists: List[float]) -> Tuple[List[float], List[float]]:
    """Split list ordered inner->outer into halves (ceil(n/2))."""
    n = len(dists)
    if n <= 1:
        return dists, dists
    k = (n + 1) // 2
    return dists[:k], dists[-k:]


@dataclass
class BrowMeasuresNorm:
    L_inner_mean_norm: float
    L_outer_mean_norm: float
    L_all_mean_norm: float
    R_inner_mean_norm: float
    R_outer_mean_norm: float
    R_all_mean_norm: float


def compute_brow_measures_norm(lm_norm68: np.ndarray) -> BrowMeasuresNorm:
    """
    Compute eyebrow-eye-line distances in normalized coords (x/w, y/h).
    lm_norm68: (68,2)
    """
    L_a = tuple(lm_norm68[EYE_68["L_INNER"]])
    L_b = tuple(lm_norm68[EYE_68["L_OUTER"]])
    R_a = tuple(lm_norm68[EYE_68["R_INNER"]])
    R_b = tuple(lm_norm68[EYE_68["R_OUTER"]])

    L_pts = [tuple(lm_norm68[i]) for i in BROW_68["L"]]
    R_pts = [tuple(lm_norm68[i]) for i in BROW_68["R"]]

    L_d = [signed_point_line_distance(p, L_a, L_b) for p in L_pts]
    R_d = [signed_point_line_distance(p, R_a, R_b) for p in R_pts]

    L_in, L_out = split_inner_outer(L_d)
    R_in, R_out = split_inner_outer(R_d)

    return BrowMeasuresNorm(
        L_inner_mean_norm=safe_mean(L_in),
        L_outer_mean_norm=safe_mean(L_out),
        L_all_mean_norm=safe_mean(L_d),
        R_inner_mean_norm=safe_mean(R_in),
        R_outer_mean_norm=safe_mean(R_out),
        R_all_mean_norm=safe_mean(R_d),
    )


# =============================================================================
# 3DDFA_V2 wrapper with repo-root injection
# =============================================================================
def _resolve_repo_root(tddfa_repo: Optional[str]) -> str:
    env = os.environ.get("TDDFA_V2_REPO", "").strip()
    root = (tddfa_repo or env or "").strip()
    if not root:
        raise RuntimeError(
            "Missing 3DDFA_V2 repo path. Provide --tddfa_repo /path/to/3DDFA_V2 "
            "or set env var TDDFA_V2_REPO."
        )
    root = os.path.abspath(root)
    cfg = os.path.join(root, "configs", "mb1_120x120.yml")
    if not os.path.isfile(cfg):
        raise RuntimeError(f"Invalid 3DDFA_V2 repo root (missing configs): {cfg}")
    return root


class ThreeDDFA2D:
    """
    Returns sparse vertices (N,2) as 2D landmarks and pose (pitch,yaw,roll).

    Works from anywhere if repo root is provided.
    """

    def __init__(self, tddfa_repo: Optional[str], use_onnx: bool = True, device: str = "cpu"):
        repo_root = _resolve_repo_root(tddfa_repo)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from FaceBoxes import FaceBoxes
        from TDDFA import TDDFA
        from utils.pose import calc_pose
        import yaml

        cfg_path = os.path.join(repo_root, "configs", "mb1_120x120.yml")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.face_boxes = FaceBoxes()
        self.tddfa = TDDFA(gpu_mode=(device != "cpu"), **cfg, use_onnx=use_onnx)
        self.calc_pose = calc_pose

    def infer_one(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float]]]:
        boxes = self.face_boxes(frame_bgr)
        if boxes is None or len(boxes) == 0:
            return None, None

        boxes = np.array(boxes)
        best_idx = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
        boxes = boxes[best_idx:best_idx + 1]

        param_lst, roi_box_lst = self.tddfa(frame_bgr, boxes)
        if param_lst is None or len(param_lst) == 0:
            return None, None

        ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
        ver = ver_lst[0]          # (3, N)
        lm_px = ver[:2, :].T      # (N, 2)

        _, pose = self.calc_pose(param_lst[0])
        yaw, pitch, roll = float(pose[0]), float(pose[1]), float(pose[2])
        return lm_px.astype(np.float32), (pitch, yaw, roll)


# =============================================================================
# Video extraction
# =============================================================================
def extract_video_3ddfa2d_norm(
    video_path: str,
    tddfa_repo: Optional[str],
    use_onnx: bool = True,
    device: str = "cpu",
) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    model = ThreeDDFA2D(tddfa_repo=tddfa_repo, use_onnx=use_onnx, device=device)

    rows = []
    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        h, w = frame_bgr.shape[:2]
        t = (frame_idx / fps) if fps > 0 else float("nan")

        lm_px, pose = model.infer_one(frame_bgr)

        if lm_px is None or pose is None or lm_px.shape[0] < 68:
            rows.append({"frame": frame_idx, "time_s": t, "w": int(w), "h": int(h)})
            frame_idx += 1
            continue

        pitch, yaw, roll = pose

        lm_px68 = lm_px[:68, :].astype(np.float64)
        lm_norm68 = lm_px68.copy()
        lm_norm68[:, 0] /= float(w)
        lm_norm68[:, 1] /= float(h)

        brow = compute_brow_measures_norm(lm_norm68=lm_norm68)

        # scale in pixels (debug / optional feature)
        lx, ly = lm_px68[EYE_68["L_OUTER"]]
        rx, ry = lm_px68[EYE_68["R_OUTER"]]
        scale_px = float(math.hypot(rx - lx, ry - ly))

        # scale in normalized coords (used for mm_from_norm via quiet-segment constant)
        lxn, lyn = lm_norm68[EYE_68["L_OUTER"]]
        rxn, ryn = lm_norm68[EYE_68["R_OUTER"]]
        scale_norm = float(math.hypot(rxn - lxn, ryn - lyn))

        rows.append(
            {
                "frame": frame_idx,
                "time_s": t,
                "w": int(w),
                "h": int(h),
                "pitch": float(pitch),
                "yaw": float(yaw),
                "roll": float(roll),
                "scale_px": scale_px,
                "scale_norm": scale_norm,
                "L_inner_mean_norm": brow.L_inner_mean_norm,
                "L_outer_mean_norm": brow.L_outer_mean_norm,
                "L_all_mean_norm": brow.L_all_mean_norm,
                "R_inner_mean_norm": brow.R_inner_mean_norm,
                "R_outer_mean_norm": brow.R_outer_mean_norm,
                "R_all_mean_norm": brow.R_all_mean_norm,
            }
        )

        frame_idx += 1

    cap.release()
    return pd.DataFrame(rows)
