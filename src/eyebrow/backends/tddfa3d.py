#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3DDFA_V2 backend (3D): extract eyebrow-eye distance in 3D point-to-plane, report only *_3d_norm.

Outputs per frame:
- time_s, frame, w, h
- pitch, yaw, roll
- scale_3d (interocular_3d)
- (optional) scale_px
- 3D normalized distances: *_3d_norm

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
# Numpy compatibility
# ---------------------------------------------------------------------
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "long"):
    np.long = np.int64

# =============================================================================
# 68-landmark indices
# =============================================================================
EYE_68 = {"L_OUTER": 36, "L_INNER": 39, "R_INNER": 42, "R_OUTER": 45}
BROW_68 = {"L": [17, 18, 19, 20, 21], "R": [22, 23, 24, 25, 26]}


def safe_mean(values: List[float]) -> float:
    values = [v for v in values if np.isfinite(v)]
    return float("nan") if len(values) == 0 else float(np.mean(values))


def split_inner_outer(dists: List[float]) -> Tuple[List[float], List[float]]:
    n = len(dists)
    if n <= 1:
        return dists, dists
    k = (n + 1) // 2
    return dists[:k], dists[-k:]


def fit_plane_svd(points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit plane n·(x-p0)=0 via SVD. Returns (p0,n_unit)."""
    pts = np.asarray(points_3d, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        raise ValueError("Need at least 3 points (N,3) to fit plane.")
    p0 = pts.mean(axis=0)
    A = pts - p0
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    n = vh[-1, :]
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        raise ValueError("Degenerate plane fit.")
    return p0, (n / n_norm)


def orient_normal_toward_brows(n: np.ndarray, eye_center: np.ndarray, brow_center: np.ndarray) -> np.ndarray:
    if float(np.dot(n, (brow_center - eye_center))) < 0.0:
        return -n
    return n


def point_plane_signed_distance(p: np.ndarray, p0: np.ndarray, n: np.ndarray) -> float:
    return float(np.dot(n, (p - p0)))


@dataclass
class BrowMeasures3DNorm:
    L_inner_mean_3d_norm: float
    L_outer_mean_3d_norm: float
    L_all_mean_3d_norm: float
    R_inner_mean_3d_norm: float
    R_outer_mean_3d_norm: float
    R_all_mean_3d_norm: float


def compute_brow_measures_point_plane_3d_norm(lm3d68: np.ndarray) -> Tuple[BrowMeasures3DNorm, float]:
    """
    lm3d68: (68,3) 3DDFA vertices in its coordinate system.
    Returns:
      measures_norm (dimensionless)
      scale_3d = interocular_3d (outer corners distance)
    """
    eye_idx = [EYE_68["L_OUTER"], EYE_68["L_INNER"], EYE_68["R_INNER"], EYE_68["R_OUTER"]]
    eye_pts = np.array([lm3d68[i] for i in eye_idx], dtype=np.float64)

    L_brow = np.array([lm3d68[i] for i in BROW_68["L"]], dtype=np.float64)
    R_brow = np.array([lm3d68[i] for i in BROW_68["R"]], dtype=np.float64)
    brow_all = np.vstack([L_brow, R_brow])

    eye_center, n = fit_plane_svd(eye_pts)
    brow_center = brow_all.mean(axis=0)
    n = orient_normal_toward_brows(n, eye_center, brow_center)

    L_d = [point_plane_signed_distance(p, eye_center, n) for p in L_brow]
    R_d = [point_plane_signed_distance(p, eye_center, n) for p in R_brow]

    # scale_3d = interocular distance (outer corners)
    L_outer = lm3d68[EYE_68["L_OUTER"]]
    R_outer = lm3d68[EYE_68["R_OUTER"]]
    scale_3d = float(np.linalg.norm(R_outer - L_outer))
    if not np.isfinite(scale_3d) or scale_3d < 1e-12:
        scale_3d = float("nan")

    def norm_list(vals: List[float], s: float) -> List[float]:
        if not np.isfinite(s) or s < 1e-12:
            return [float("nan")] * len(vals)
        return [float(v) / float(s) if np.isfinite(v) else float("nan") for v in vals]

    L_dn = norm_list(L_d, scale_3d)
    R_dn = norm_list(R_d, scale_3d)

    L_in, L_out = split_inner_outer(L_dn)
    R_in, R_out = split_inner_outer(R_dn)

    measures = BrowMeasures3DNorm(
        L_inner_mean_3d_norm=safe_mean(L_in),
        L_outer_mean_3d_norm=safe_mean(L_out),
        L_all_mean_3d_norm=safe_mean(L_dn),
        R_inner_mean_3d_norm=safe_mean(R_in),
        R_outer_mean_3d_norm=safe_mean(R_out),
        R_all_mean_3d_norm=safe_mean(R_dn),
    )
    return measures, scale_3d


# =============================================================================
# repo root injection
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


class ThreeDDFA3D:
    """Returns lm3d (N,3) and pose (pitch,yaw,roll)."""

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
        ver = ver_lst[0]   # (3, N)
        lm3d = ver.T       # (N, 3)

        _, pose = self.calc_pose(param_lst[0])
        yaw, pitch, roll = float(pose[0]), float(pose[1]), float(pose[2])
        return lm3d.astype(np.float32), (pitch, yaw, roll)


def extract_video_3ddfa3d_norm(
    video_path: str,
    tddfa_repo: Optional[str],
    use_onnx: bool = True,
    device: str = "cpu",
) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    model = ThreeDDFA3D(tddfa_repo=tddfa_repo, use_onnx=use_onnx, device=device)

    rows = []
    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        h, w = frame_bgr.shape[:2]
        t = (frame_idx / fps) if fps > 0 else float("nan")

        lm3d, pose = model.infer_one(frame_bgr)
        if lm3d is None or pose is None or lm3d.shape[0] < 68:
            rows.append({"frame": frame_idx, "time_s": t, "w": int(w), "h": int(h)})
            frame_idx += 1
            continue

        pitch, yaw, roll = pose
        lm3d68 = lm3d[:68, :].astype(np.float64)

        try:
            brow, scale_3d = compute_brow_measures_point_plane_3d_norm(lm3d68)
        except Exception:
            brow, scale_3d = None, float("nan")

        # optional pixel scale proxy (from x,y projection)
        lm_px = lm3d68[:, :2].copy()
        lx, ly = lm_px[EYE_68["L_OUTER"]]
        rx, ry = lm_px[EYE_68["R_OUTER"]]
        scale_px = float(math.hypot(rx - lx, ry - ly))

        row = {
            "frame": frame_idx,
            "time_s": t,
            "w": int(w),
            "h": int(h),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "roll": float(roll),
            "scale_3d": float(scale_3d),
            "scale_px": float(scale_px),
        }

        if brow is not None:
            row.update(
                {
                    "L_inner_mean_3d_norm": brow.L_inner_mean_3d_norm,
                    "L_outer_mean_3d_norm": brow.L_outer_mean_3d_norm,
                    "L_all_mean_3d_norm": brow.L_all_mean_3d_norm,
                    "R_inner_mean_3d_norm": brow.R_inner_mean_3d_norm,
                    "R_outer_mean_3d_norm": brow.R_outer_mean_3d_norm,
                    "R_all_mean_3d_norm": brow.R_all_mean_3d_norm,
                }
            )

        rows.append(row)
        frame_idx += 1

    cap.release()
    return pd.DataFrame(rows)
