#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import cv2

# =========================
# CONFIG
# =========================

EXP = 1

CONTROL_DIR = Path("/Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code_refactor/datasets/own_data/controlled_video")
TARGET_DIR_TEMPLATE = Path("/Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code_refactor/datasets/own_data/target_video_exp{X}")

OUT_DIR = Path("tmp/orientation_correct")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARTICIPANTS = ["p1", "p2", "p3", "p4"]

# =========================
# FIXED ROTATION MAP
# =========================

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

# =========================
# Helpers
# =========================

def resolve_paths(participant):
    ext = ".mov" if participant == "p1" else ".mp4"
    control = CONTROL_DIR / f"{participant}_control{ext}"
    target = Path(str(TARGET_DIR_TEMPLATE).format(X=EXP)) / f"{participant}_exp{EXP}{ext}"
    return control, target


def apply_rotation(frame, rotate_code):
    if rotate_code is None:
        return frame
    return cv2.rotate(frame, rotate_code)


def save_one_frame(video_path: Path, rotate_code, out_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open {video_path}")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print(f"[WARN] Could not read frame from {video_path}")
        return

    frame = apply_rotation(frame, rotate_code)
    cv2.imwrite(str(out_path), frame)


# =========================
# Main
# =========================

def main():
    for p in PARTICIPANTS:
        control_path, target_path = resolve_paths(p)

        # Control
        control_rot = ROTATION_MAP[p]["control"]
        out_control = OUT_DIR / f"{p}_control_correct.png"
        save_one_frame(control_path, control_rot, out_control)

        # Target
        target_rot = ROTATION_MAP[p]["target"]
        out_target = OUT_DIR / f"{p}_target_correct.png"
        save_one_frame(target_path, target_rot, out_target)

        print(f"{p} done.")

    print(f"\nSaved corrected frames to: {OUT_DIR}")


if __name__ == "__main__":
    main()