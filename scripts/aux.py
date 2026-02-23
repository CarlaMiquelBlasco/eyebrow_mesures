#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combined script:
1) Compute validity stats per file and aggregated by model (using `status` if present).
2) Save example failure frames by reading the *_F.csv files and grabbing frames from the
   corresponding target videos.

Default behavior:
- Stats: runs for all models in {MPP2D, MP3D, TDDFA_2D, TDDFA_3D}
- Fail examples: runs ONLY for TDDFA models by default (can enable MP too)

Edit the CONFIG section to match your paths.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
import sys

# =============================================================================
# CONFIG
# =============================================================================

# Folder containing p{i}_exp1_{model}_F.csv




RESULTS_DIR = Path("results/exp1_v4")

# Experiment number in filenames / video folder template
EXP = 1

# Output stats CSVs
OUT_FILE_PER_FILE = RESULTS_DIR / "validity_per_file.csv"
OUT_FILE_BY_MODEL = RESULTS_DIR / "validity_by_model.csv"

# Where your original videos live
CONTROL_DIR = Path("/Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code_refactor/datasets/own_data/target_video_exp{X}")
TARGET_DIR_TEMPLATE = Path("/Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code_refactor/datasets/own_data/target_video_exp{X}")

# Where to save failure examples (frames)
FAIL_EXAMPLES_ROOT = Path("tmp/fail_examples") / f"exp{EXP}"

# Consider only these models (ignore any other file naming)
ALLOWED_MODELS_ALL: Set[str] = {"MP2D", "MP3D", "TDDFA_2D", "TDDFA_3D"}

# Which models to export failure-frame examples for
# (default: only TDDFA, since MP rarely fails; set to ALLOWED_MODELS_ALL if you want all)
MODELS_FOR_FAIL_EXAMPLES: Set[str] = {"TDDFA_2D", "TDDFA_3D","MP2D", "MP3D"}

# Strict filename pattern: only read exact matches
FILENAME_RE = re.compile(r"^(p[1-4])_exp1_(MP2D|MP3D|TDDFA_2D|TDDFA_3D)_F\.csv$")

# Failure statuses of interest; others will be grouped as "other_fail"
FAIL_STATUSES = ["no_face", "no_fit", "pose_fail", "exception", "bad_landmarks", "plane_fail", "measure_fail"]

# Max images to save per (participant, model, status)
MAX_PER_STATUS = 40

# Random seed for reproducible sampling
RNG_SEED = 123


# =============================================================================
# Stats helpers
# =============================================================================
def compute_valid_mask(df: pd.DataFrame) -> pd.Series:
    """
    Prefer status if present.
    Valid frame definition:
      - if 'status' exists: valid iff status == 'ok'
      - else: fallback to (pose finite) AND (at least one brow mean finite)
    """
    if "status" in df.columns:
        return df["status"].astype(str).str.strip().str.lower().eq("ok")

    # Fallback (legacy)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    pose_cols = ["pitch", "yaw", "roll"]
    brow_cols = [c for c in df.columns if "_mean_" in c and not c.endswith("_corr")]

    if not brow_cols:
        raise RuntimeError("No eyebrow mean columns found (legacy fallback).")

    pose_valid = df[pose_cols].notna().all(axis=1)
    brow_valid = df[brow_cols].notna().any(axis=1)
    return pose_valid & brow_valid


def status_counts(df: pd.DataFrame) -> dict:
    """
    Returns a dict with counts for ok and failure types.
    If status missing, returns empty breakdown.
    """
    if "status" not in df.columns:
        return {}

    s = df["status"].astype(str).str.strip().str.lower()
    vc = s.value_counts(dropna=False).to_dict()

    out: Dict[str, int] = {}
    out["ok"] = int(vc.get("ok", 0))

    for st in FAIL_STATUSES:
        out[st] = int(vc.get(st, 0))

    known = set(["ok"] + FAIL_STATUSES)
    other_fail = 0
    for k, v in vc.items():
        if k not in known:
            other_fail += int(v)

    out["other_fail"] = int(other_fail)
    out["fail_total"] = int(len(df) - out["ok"])
    return out


def compute_and_save_stats() -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    for filename in os.listdir(RESULTS_DIR):
        m = FILENAME_RE.match(filename)
        if not m:
            continue

        participant, model = m.groups()
        if model not in ALLOWED_MODELS_ALL:
            continue

        filepath = RESULTS_DIR / filename
        df = pd.read_csv(filepath)

        total_frames = len(df)
        valid_mask = compute_valid_mask(df)

        valid_frames = int(valid_mask.sum())
        invalid_frames = total_frames - valid_frames

        valid_pct = 100.0 * valid_frames / total_frames if total_frames > 0 else np.nan
        invalid_pct = 100.0 - valid_pct if total_frames > 0 else np.nan

        row = {
            "file": filename,
            "participant": participant,
            "model": model,
            "total_frames": total_frames,
            "valid_frames": valid_frames,
            "invalid_frames": invalid_frames,
            "valid_%": valid_pct,
            "invalid_%": invalid_pct,
        }

        sc = status_counts(df)
        if sc:
            for k in ["ok"] + FAIL_STATUSES + ["other_fail", "fail_total"]:
                row[f"n_{k}"] = sc.get(k, 0)
                row[f"pct_{k}"] = (100.0 * row[f"n_{k}"] / total_frames) if total_frames > 0 else np.nan

        rows.append(row)
        print(f"[STATS] {filename} → {valid_pct:.2f}% valid")

    if not rows:
        raise RuntimeError(f"No matching files found in {RESULTS_DIR}")

    df_files = pd.DataFrame(rows).sort_values(["model", "participant"])
    df_files.to_csv(OUT_FILE_PER_FILE, index=False)

    # Aggregate by model
    agg_dict = {"total_frames": "sum", "valid_frames": "sum", "invalid_frames": "sum"}
    for k in ["ok"] + FAIL_STATUSES + ["other_fail", "fail_total"]:
        col = f"n_{k}"
        if col in df_files.columns:
            agg_dict[col] = "sum"

    df_model = df_files.groupby("model").agg(agg_dict).reset_index()
    df_model["valid_%"] = 100.0 * df_model["valid_frames"] / df_model["total_frames"]
    df_model["invalid_%"] = 100.0 - df_model["valid_%"]

    for k in ["ok"] + FAIL_STATUSES + ["other_fail", "fail_total"]:
        col = f"n_{k}"
        if col in df_model.columns:
            df_model[f"pct_{k}"] = 100.0 * df_model[col] / df_model["total_frames"]

    df_model.to_csv(OUT_FILE_BY_MODEL, index=False)

    print("\n[STATS] Saved:")
    print(" -", OUT_FILE_PER_FILE)
    print(" -", OUT_FILE_BY_MODEL)

    return df_files, df_model

def fix_orientation(frame: np.ndarray) -> np.ndarray:
    """
    Normalize video orientation.
    If frame is landscape (w > h), rotate 90° CCW to make portrait.
    """
    h, w = frame.shape[:2]
    if w > h:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

# =============================================================================
# Failure examples helpers
# =============================================================================
def resolve_video_paths(participant: str) -> Dict[str, Path]:
    """
    Returns dict with keys 'control' and 'target'
    Assumes p1 uses .mov and others .mp4 like your runner scripts.
    """
    ext = ".mov" if participant == "p1" else ".mp4"
    control = CONTROL_DIR / f"{participant}_{ext}"
    target = Path(str(TARGET_DIR_TEMPLATE).format(X=EXP)) / f"{participant}_exp{EXP}{ext}"
    return {"control": control, "target": target}


def grab_and_save_frames(video_path: Path, frames: List[int], out_dir: Path) -> int:
    """
    Saves the given frame indices from video into out_dir.
    Returns number saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return 0

    saved = 0
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        out_path = out_dir / f"frame_{int(fi):06d}.png"
        frame = fix_orientation(frame)
        cv2.imwrite(str(out_path), frame)
        saved += 1

    cap.release()
    return saved


def save_fail_examples_from_csv() -> None:
    rng = np.random.default_rng(RNG_SEED)

    csv_files = sorted([p for p in RESULTS_DIR.iterdir() if p.is_file() and p.name.endswith("_F.csv")])
    if not csv_files:
        raise RuntimeError(f"No _F.csv files found in {RESULTS_DIR}")

    FAIL_EXAMPLES_ROOT.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_files:
        m = FILENAME_RE.match(csv_path.name)
        if not m:
            continue

        participant, model = m.groups()
        if model not in MODELS_FOR_FAIL_EXAMPLES:
            continue

        df = pd.read_csv(csv_path)
        if "status" not in df.columns or "frame" not in df.columns:
            print(f"[WARN] Missing status/frame in {csv_path.name}, skipping.")
            continue

        df["status"] = df["status"].astype(str).str.strip().str.lower()
        df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")

        # Pipeline output corresponds to target video frames
        vids = resolve_video_paths(participant)
        video_path = vids["target"]

        if not video_path.exists():
            print(f"[WARN] Target video not found for {participant}: {video_path}")
            continue

        print(f"\n[EXAMPLES] === {csv_path.name} ===")
        print(f"[EXAMPLES] Video: {video_path}")

        for st in FAIL_STATUSES:
            sub = df[df["status"] == st].dropna(subset=["frame"])
            frame_ids = sub["frame"].astype(int).tolist()

            if not frame_ids:
                continue

            # Sample up to MAX_PER_STATUS
            if len(frame_ids) > MAX_PER_STATUS:
                frame_ids = rng.choice(frame_ids, size=MAX_PER_STATUS, replace=False).tolist()
                frame_ids.sort()

            out_dir = FAIL_EXAMPLES_ROOT / model / participant / st
            saved = grab_and_save_frames(video_path, frame_ids, out_dir)
            print(f"[EXAMPLES]   {st}: found {len(sub)} frames, saved {saved} → {out_dir}")

    print(f"\n[EXAMPLES] Done. Examples saved under: {FAIL_EXAMPLES_ROOT}")


# =============================================================================
# Main
# =============================================================================
def main():
    compute_and_save_stats()
    save_fail_examples_from_csv()


if __name__ == "__main__":
    main()