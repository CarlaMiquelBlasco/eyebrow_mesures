from __future__ import annotations

from typing import Optional, Tuple
import os
import re

import cv2
import numpy as np


# Verified from manual inspection
ROTATION_MAP = {
    "p1": {"control": cv2.ROTATE_90_COUNTERCLOCKWISE,
           "target":  cv2.ROTATE_90_COUNTERCLOCKWISE},

    "p2": {"control": None,
           "target":  cv2.ROTATE_90_CLOCKWISE},

    "p3": {"control": cv2.ROTATE_90_COUNTERCLOCKWISE,
           "target":  cv2.ROTATE_90_COUNTERCLOCKWISE},

    "p4": {"control": None,
           "target":  None},
}


def fix_orientation(
    frame: np.ndarray,
    participant: Optional[str],
    video_type: Optional[str],
) -> np.ndarray:
    """Deterministic orientation fix."""
    if participant is None or video_type is None:
        return frame
    rotate_code = ROTATION_MAP.get(participant, {}).get(video_type, None)
    if rotate_code is None:
        return frame
    return cv2.rotate(frame, rotate_code)


_P_RE = re.compile(r"\b(p[1-4])\b", re.IGNORECASE)


_P_RE = re.compile(r"(?:^|[_\-\s])(p[1-4])(?:[_\-\.\s]|$)", re.IGNORECASE)

def infer_from_path(video_path: str):
    base = os.path.basename(video_path).lower()

    m = _P_RE.search(base)
    participant = m.group(1).lower() if m else None

    video_type = "control" if "_control" in base else "target"
    return participant, video_type

# -------------------------------------------------
# Debug: save one oriented frame per video
# -------------------------------------------------

def save_debug_frame_once(
    frame: np.ndarray,
    video_path: str,
    out_dir: str = "tmp/video_debug_frames_3d",
) -> None:
    """
    Saves ONE frame per video (first call only).
    Filename = <video_basename>.png
    """
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.basename(video_path)
    name = os.path.splitext(base)[0]
    out_path = os.path.join(out_dir, f"{name}.png")

    cv2.imwrite(out_path, frame)