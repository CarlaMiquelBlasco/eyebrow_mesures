# src/eyebrow/backends/mp2d.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

from typing import Optional
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from ..mp_pose import EYE, estimate_pose_euler
from ..metrics_common import safe_mean, split_inner_outer
from ..utils.orientation import infer_from_path, fix_orientation, save_debug_frame_once


BROW_POINTS = {
    "L": [107, 66, 105, 63, 70],
    "R": [336, 296, 334, 293, 300],
}


@dataclass
class BrowMeasures2D:
    # Pixel distances (signed)
    L_inner_mean: float
    L_outer_mean: float
    L_all_mean: float
    R_inner_mean: float
    R_outer_mean: float
    R_all_mean: float

    # Normalized-coordinate distances (signed) in [0,1] image coords
    L_inner_mean_norm: float
    L_outer_mean_norm: float
    L_all_mean_norm: float
    R_inner_mean_norm: float
    R_outer_mean_norm: float
    R_all_mean_norm: float


def point_xy(landmarks, idx: int) -> Tuple[float, float]:
    p = landmarks[idx]
    return float(p.x), float(p.y)  # normalized [0,1]


def point_xy_px(landmarks, idx: int, w: int, h: int) -> Tuple[float, float]:
    x, y = point_xy(landmarks, idx)
    return x * w, y * h


def signed_point_line_distance(p, a, b) -> float:
    (px, py), (ax, ay), (bx, by) = p, a, b
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    denom = math.hypot(vx, vy)
    if denom < 1e-9:
        return float("nan")
    return (vx * wy - vy * wx) / denom


def compute_brow_measures(face_landmarks, w: int, h: int) -> BrowMeasures2D:
    lm = face_landmarks.landmark

    # pixel eye line points
    L_a_px = point_xy_px(lm, EYE["L_INNER"], w, h)
    L_b_px = point_xy_px(lm, EYE["L_OUTER"], w, h)
    R_a_px = point_xy_px(lm, EYE["R_INNER"], w, h)
    R_b_px = point_xy_px(lm, EYE["R_OUTER"], w, h)

    # brow points px
    L_pts_px = [point_xy_px(lm, idx, w, h) for idx in BROW_POINTS["L"]]
    R_pts_px = [point_xy_px(lm, idx, w, h) for idx in BROW_POINTS["R"]]

    L_d_px = [signed_point_line_distance(p, L_a_px, L_b_px) for p in L_pts_px]
    R_d_px = [signed_point_line_distance(p, R_a_px, R_b_px) for p in R_pts_px]

    L_in_px, L_out_px = split_inner_outer(L_d_px)
    R_in_px, R_out_px = split_inner_outer(R_d_px)

    # normalized eye line points
    L_a_n = point_xy(lm, EYE["L_INNER"])
    L_b_n = point_xy(lm, EYE["L_OUTER"])
    R_a_n = point_xy(lm, EYE["R_INNER"])
    R_b_n = point_xy(lm, EYE["R_OUTER"])

    # brow points norm
    L_pts_n = [point_xy(lm, idx) for idx in BROW_POINTS["L"]]
    R_pts_n = [point_xy(lm, idx) for idx in BROW_POINTS["R"]]

    L_d_n = [signed_point_line_distance(p, L_a_n, L_b_n) for p in L_pts_n]
    R_d_n = [signed_point_line_distance(p, R_a_n, R_b_n) for p in R_pts_n]

    L_in_n, L_out_n = split_inner_outer(L_d_n)
    R_in_n, R_out_n = split_inner_outer(R_d_n)

    return BrowMeasures2D(
        L_inner_mean=safe_mean(L_in_px),
        L_outer_mean=safe_mean(L_out_px),
        L_all_mean=safe_mean(L_d_px),
        R_inner_mean=safe_mean(R_in_px),
        R_outer_mean=safe_mean(R_out_px),
        R_all_mean=safe_mean(R_d_px),
        L_inner_mean_norm=safe_mean(L_in_n),
        L_outer_mean_norm=safe_mean(L_out_n),
        L_all_mean_norm=safe_mean(L_d_n),
        R_inner_mean_norm=safe_mean(R_in_n),
        R_outer_mean_norm=safe_mean(R_out_n),
        R_all_mean_norm=safe_mean(R_d_n),
    )

def _maybe_save_fail_frame(
    frame_bgr: np.ndarray,
    out_dir: Optional[str],
    status: str,
    frame_idx: int,
    max_per_status: int,
    counters: Dict[str, int],
) -> None:
    '''
    save the frame if it is not valid
    '''
    if out_dir is None:
        return
    if counters.get(status, 0) >= max_per_status:
        return
    os.makedirs(out_dir, exist_ok=True)
    subdir = os.path.join(out_dir, status)
    os.makedirs(subdir, exist_ok=True)
    out_path = os.path.join(subdir, f"frame_{frame_idx:06d}.png")
    cv2.imwrite(out_path, frame_bgr)
    counters[status] = counters.get(status, 0) + 1


def extract_video_mp2d(
    video_path: str,
    fail_examples_dir: Optional[str] = None,
    max_fail_examples_per_status: int = 30,) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    participant, video_type = infer_from_path(video_path)
    print("[ORIENT]", video_path, "->", participant, video_type)
    rows = []
    frame_idx = 0
    fail_counters: Dict[str, int] = {}
    while True:
        ok, frame_bgr = cap.read()
        frame_bgr = fix_orientation(frame_bgr, participant, video_type)
        # Save one debug frame per video (important to check because frames may be read rotated and it can break the whole experiment)
        if frame_idx == 0:
            save_debug_frame_once(frame_bgr, video_path)
        if not ok:
            break

        h, w = frame_bgr.shape[:2]
        t = (frame_idx / fps) if fps > 0 else float("nan")

        status = "exception"
        row_base = {"frame": frame_idx, "time_s": t, "w": int(w), "h": int(h)}

        try:
            res = face_mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

            if not res.multi_face_landmarks:
                status = "no_face"
                _maybe_save_fail_frame(frame_bgr, fail_examples_dir, status, frame_idx,
                                       max_fail_examples_per_status, fail_counters)
                rows.append({**row_base, "status": status})
                frame_idx += 1
                continue

            fl = res.multi_face_landmarks[0]

            # scale_norm (outer-eye distance in normalized coords)
            lm = fl.landmark
            lxo, lyo = float(lm[EYE["L_OUTER"]].x), float(lm[EYE["L_OUTER"]].y)
            rxo, ryo = float(lm[EYE["R_OUTER"]].x), float(lm[EYE["R_OUTER"]].y)
            scale_norm = float(math.hypot(rxo - lxo, ryo - lyo))

            # pose
            try:
                pitch, yaw, roll, scale_px = estimate_pose_euler(fl, w, h)
            except Exception:
                status = "pose_fail"
                _maybe_save_fail_frame(frame_bgr, fail_examples_dir, status, frame_idx,
                                       max_fail_examples_per_status, fail_counters)
                rows.append({**row_base, "status": status})
                frame_idx += 1
                continue

            # measures
            try:
                brow = compute_brow_measures(fl, w, h)
            except Exception:
                status = "measure_fail"
                _maybe_save_fail_frame(frame_bgr, fail_examples_dir, status, frame_idx,
                                       max_fail_examples_per_status, fail_counters)
                rows.append({
                    **row_base,
                    "status": status,
                    "pitch": float(pitch),
                    "yaw": float(yaw),
                    "roll": float(roll),
                    "scale": float(scale_px),
                    "scale_norm": float(scale_norm),
                })
                frame_idx += 1
                continue

            status = "ok"
            rows.append(
                {
                    **row_base,
                    "status": status,
                    "pitch": float(pitch),
                    "yaw": float(yaw),
                    "roll": float(roll),
                    "scale": float(scale_px),
                    "scale_norm": float(scale_norm),

                    "L_inner_mean": brow.L_inner_mean,
                    "L_outer_mean": brow.L_outer_mean,
                    "L_all_mean": brow.L_all_mean,
                    "R_inner_mean": brow.R_inner_mean,
                    "R_outer_mean": brow.R_outer_mean,
                    "R_all_mean": brow.R_all_mean,

                    "L_inner_mean_norm": brow.L_inner_mean_norm,
                    "L_outer_mean_norm": brow.L_outer_mean_norm,
                    "L_all_mean_norm": brow.L_all_mean_norm,
                    "R_inner_mean_norm": brow.R_inner_mean_norm,
                    "R_outer_mean_norm": brow.R_outer_mean_norm,
                    "R_all_mean_norm": brow.R_all_mean_norm,
                }
            )

        except Exception:
            status = "exception"
            _maybe_save_fail_frame(frame_bgr, fail_examples_dir, status, frame_idx,
                                   max_fail_examples_per_status, fail_counters)
            rows.append({**row_base, "status": status})

        frame_idx += 1

    cap.release()
    face_mesh.close()
    return pd.DataFrame(rows)
