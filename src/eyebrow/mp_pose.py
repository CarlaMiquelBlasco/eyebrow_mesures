# src/eyebrow/mp_pose.py
from __future__ import annotations
import math
from typing import Tuple
import cv2
import numpy as np

# MediaPipe indices (FaceMesh)
POSE_LM = {
    "NOSE_TIP": 1,
    "CHIN": 152,
    "L_EYE_OUTER": 33,
    "R_EYE_OUTER": 263,
    "L_MOUTH": 61,
    "R_MOUTH": 291,
}

# Approximate 3D model points for solvePnP (units arbitrary but consistent)
MODEL_3D = np.array(
    [
        [0.0, 0.0, 0.0],        # nose tip
        [0.0, -63.6, -12.5],    # chin
        [-43.3, 32.7, -26.0],   # left eye outer
        [43.3, 32.7, -26.0],    # right eye outer
        [-28.9, -28.9, -24.1],  # left mouth corner
        [28.9, -28.9, -24.1],   # right mouth corner
    ],
    dtype=np.float32,
)

EYE = {"L_INNER": 133, "L_OUTER": 33, "R_INNER": 362, "R_OUTER": 263}


def estimate_pose_euler(face_landmarks, frame_w: int, frame_h: int) -> Tuple[float, float, float, float]:
    """
    Return (pitch_deg, yaw_deg, roll_deg, scale_px).
    scale_px = outer-eye distance (px) between landmarks 33 and 263.
    """
    lm = face_landmarks.landmark

    def px(idx):
        x, y = float(lm[idx].x), float(lm[idx].y)
        return [x * frame_w, y * frame_h]

    image_points = np.array(
        [
            px(POSE_LM["NOSE_TIP"]),
            px(POSE_LM["CHIN"]),
            px(POSE_LM["L_EYE_OUTER"]),
            px(POSE_LM["R_EYE_OUTER"]),
            px(POSE_LM["L_MOUTH"]),
            px(POSE_LM["R_MOUTH"]),
        ],
        dtype=np.float32,
    )

    focal_length = float(frame_w)
    center = (frame_w / 2.0, frame_h / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]],
        dtype=np.float32,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    ok, rvec, _tvec = cv2.solvePnP(
        MODEL_3D, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return float("nan"), float("nan"), float("nan"), float("nan")

    rmat, _ = cv2.Rodrigues(rvec)

    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = 0.0

    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)
    roll_deg = math.degrees(roll)

    # scale proxy: outer-eye distance in px
    lx, ly = image_points[2]
    rx, ry = image_points[3]
    scale_px = float(math.hypot(rx - lx, ry - ly))

    return pitch_deg, yaw_deg, roll_deg, scale_px
