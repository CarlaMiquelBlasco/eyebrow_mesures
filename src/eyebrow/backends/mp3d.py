# src/eyebrow/backends/mp3d.py
from __future__ import annotations
import numpy as np
import cv2
import pandas as pd
import mediapipe as mp
from typing import List, Tuple

from ..mp_pose import EYE, estimate_pose_euler
from ..metrics_common import safe_mean, split_inner_outer


BROW_POINTS = {
    "L": [107, 66, 105, 63, 70],
    "R": [336, 296, 334, 293, 300],
}


def mp_lm_xyz(face_landmarks, idx: int) -> np.ndarray:
    lm = face_landmarks.landmark[idx]
    return np.array([float(lm.x), float(lm.y), float(lm.z)], dtype=np.float64)


def fit_plane_svd(points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_3d, dtype=np.float64)
    p0 = pts.mean(axis=0)
    A = pts - p0
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    n = vh[-1, :]
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        raise ValueError("Degenerate plane fit.")
    return p0, n / n_norm


def orient_normal_toward_brows(n: np.ndarray, eye_center: np.ndarray, brow_center: np.ndarray) -> np.ndarray:
    if float(np.dot(n, (brow_center - eye_center))) < 0.0:
        return -n
    return n


def point_plane_signed_distance(p: np.ndarray, p0: np.ndarray, n: np.ndarray) -> float:
    return float(np.dot(n, (p - p0)))


def compute_brow_measures_point_plane_mp3d(face_landmarks):
    # eye plane points
    eye_idx = [EYE["L_OUTER"], EYE["L_INNER"], EYE["R_INNER"], EYE["R_OUTER"]]
    eye_pts = np.stack([mp_lm_xyz(face_landmarks, i) for i in eye_idx], axis=0)

    # brow points
    L_brow = np.stack([mp_lm_xyz(face_landmarks, i) for i in BROW_POINTS["L"]], axis=0)
    R_brow = np.stack([mp_lm_xyz(face_landmarks, i) for i in BROW_POINTS["R"]], axis=0)
    brow_all = np.vstack([L_brow, R_brow])

    eye_center, n = fit_plane_svd(eye_pts)
    n = orient_normal_toward_brows(n, eye_center, brow_all.mean(axis=0))

    L_d = [point_plane_signed_distance(p, eye_center, n) for p in L_brow]
    R_d = [point_plane_signed_distance(p, eye_center, n) for p in R_brow]

    L_in, L_out = split_inner_outer(L_d)
    R_in, R_out = split_inner_outer(R_d)

    # interocular_3d outer corners
    L_outer = mp_lm_xyz(face_landmarks, EYE["L_OUTER"])
    R_outer = mp_lm_xyz(face_landmarks, EYE["R_OUTER"])
    scale_3d = float(np.linalg.norm(R_outer - L_outer))
    if not np.isfinite(scale_3d) or scale_3d < 1e-12:
        scale_3d = float("nan")

    def norm_list(vals: List[float], s: float) -> List[float]:
        if not np.isfinite(s) or s < 1e-12:
            return [float("nan")] * len(vals)
        return [float(v) / float(s) if np.isfinite(v) else float("nan") for v in vals]

    L_d_n = norm_list(L_d, scale_3d)
    R_d_n = norm_list(R_d, scale_3d)
    L_in_n, L_out_n = split_inner_outer(L_d_n)
    R_in_n, R_out_n = split_inner_outer(R_d_n)

    measures = {
        "L_inner_mean_3d": safe_mean(L_in),
        "L_outer_mean_3d": safe_mean(L_out),
        "L_all_mean_3d": safe_mean(L_d),
        "R_inner_mean_3d": safe_mean(R_in),
        "R_outer_mean_3d": safe_mean(R_out),
        "R_all_mean_3d": safe_mean(R_d),
        "L_inner_mean_3d_norm": safe_mean(L_in_n),
        "L_outer_mean_3d_norm": safe_mean(L_out_n),
        "L_all_mean_3d_norm": safe_mean(L_d_n),
        "R_inner_mean_3d_norm": safe_mean(R_in_n),
        "R_outer_mean_3d_norm": safe_mean(R_out_n),
        "R_all_mean_3d_norm": safe_mean(R_d_n),
    }
    return measures, scale_3d


def extract_video_mp3d(video_path: str) -> pd.DataFrame:
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

    rows = []
    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        h, w = frame_bgr.shape[:2]
        t = (frame_idx / fps) if fps > 0 else float("nan")

        res = face_mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        if res.multi_face_landmarks:
            fl = res.multi_face_landmarks[0]

            pitch, yaw, roll, scale_px = estimate_pose_euler(fl, w, h)

            try:
                measures, scale_3d = compute_brow_measures_point_plane_mp3d(fl)
            except Exception:
                measures, scale_3d = None, float("nan")

            row = {
                "frame": frame_idx,
                "time_s": t,
                "w": int(w),
                "h": int(h),
                "pitch": float(pitch),
                "yaw": float(yaw),
                "roll": float(roll),
                "scale_px": float(scale_px),
                "scale_3d": float(scale_3d),
            }
            if measures is not None:
                row.update(measures)

            rows.append(row)
        else:
            rows.append({"frame": frame_idx, "time_s": t, "w": int(w), "h": int(h)})

        frame_idx += 1

    cap.release()
    face_mesh.close()
    return pd.DataFrame(rows)
