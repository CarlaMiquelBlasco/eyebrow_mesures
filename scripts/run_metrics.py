#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from typing import Dict, List

import pandas as pd


# ----------------------------
# Column matching (robust to unicode minus)
# ----------------------------
def norm_col(s: str) -> str:
    s = str(s).strip().replace("−", "-")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def find_matching_column(columns: List[str], targets_norm: List[str]) -> str:
    norm_map = {norm_col(c): c for c in columns}
    for t in targets_norm:
        if t in norm_map:
            return norm_map[t]
    return ""


# ----------------------------
# Measure filters
# ----------------------------
def keep_measures_for_model(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    MP3D -> *_3d_mm_const
    MP2D, TDDFA_2D -> *_mm_const but NOT *_3d_mm_const
    """
    m = model.upper()
    measures = df["Measure"].astype(str)

    if m in ("MP3D,TDDFA_3D"):
        return df[measures.str.endswith("_3d_mm_const")].copy()

    if m in ("MP2D", "TDDFA_2D"):
        return df[measures.str.endswith("_mm_const") & ~measures.str.endswith("_3d_mm_const")].copy()

    raise ValueError(f"Unexpected model: {model}")


def canonical_measure_name(measure: str) -> str:
    """
    Only for MODEL aggregation: make 2D and 3D comparable by collapsing:
      *_3d_mm_const -> *_mm_const
    """
    if measure.endswith("_3d_mm_const"):
        return measure.replace("_3d_mm_const", "_mm_const")
    return measure


# ----------------------------
# Aggregation helpers
# ----------------------------
def build_participant_table(
    source: Dict[str, Dict[str, Dict[str, float]]],
    participants: List[str],
    models: List[str],
    measures_per_model: Dict[str, List[str]],
) -> pd.DataFrame:
    cols: List[str] = []
    for model in models:
        for measure in measures_per_model.get(model, []):
            cols.append(f"{model}__{measure}")

    out = pd.DataFrame(index=participants, columns=cols, dtype=float)

    for model in models:
        for pi in participants:
            for measure in measures_per_model.get(model, []):
                out.loc[pi, f"{model}__{measure}"] = source.get(model, {}).get(pi, {}).get(measure, float("nan"))

    out.loc["average"] = out.mean(axis=0, skipna=True)
    return out


def build_model_table(
    source: Dict[str, Dict[str, Dict[str, float]]],
    participants: List[str],
    models: List[str],
    measures_union: List[str],
) -> pd.DataFrame:
    out = pd.DataFrame(index=models, columns=measures_union, dtype=float)

    for model in models:
        for measure in measures_union:
            vals = []
            for pi in participants:
                v = source.get(model, {}).get(pi, {}).get(measure, float("nan"))
                if pd.notna(v):
                    vals.append(v)
            out.loc[model, measure] = (sum(vals) / len(vals)) if vals else float("nan")

    # do not write columns that are entirely empty
    out = out.dropna(axis=1, how="all")
    return out


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_dir",
        required=True,
        help="Directory containing experiment-1 CSVs.",
    )
    ap.add_argument(
        "--participants",
        default="p1,p2,p3,p4",
        help="Comma-separated list: p1,p2,p3,p4",
    )
    ap.add_argument(
        "--models",
        default="MP3D,MP2D,TDDFA_2D,TDDFA_3D",
        help="Comma-separated list: MP3D,MP2D,TDDFA_2D,TDDFA_3D",
    )
    ap.add_argument("--out_xlsx1", default="exp1_mmconst_by_participant.xlsx")
    ap.add_argument("--out_xlsx2", default="exp1_mmconst_by_model.xlsx")
    args = ap.parse_args()

    results_dir = args.results_dir
    participants = [x.strip() for x in args.participants.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]

    # Targets for error columns
    raw_targets_n = [norm_col(x) for x in ["|dist_raw-d0|", "|dist_raw−d0|"]]
    corr_targets_n = [norm_col(x) for x in ["|dist_corrected-d0|", "|dist_corrected−d0|"]]

    # -------------------------
    # Two parallel aggregations:
    #   - *_part: used for FILE 1 (by participant) -> KEEP ORIGINAL measure names
    #   - *_model: used for FILE 2 (by model) -> CANONICALIZE measure names
    # -------------------------
    raw_abs_part: Dict[str, Dict[str, Dict[str, float]]] = {}
    corr_abs_part: Dict[str, Dict[str, Dict[str, float]]] = {}

    raw_abs_model: Dict[str, Dict[str, Dict[str, float]]] = {}
    corr_abs_model: Dict[str, Dict[str, Dict[str, float]]] = {}

    # Track measures per model for FILE 1 (must be original names so MP3D shows *_3d_mm_const)
    measures_per_model_part: Dict[str, List[str]] = {}

    missing = []
    for pi in participants:
        for model in models:
            fname = f"{pi}_exp1_{model}_G.csv"
            path = os.path.join(results_dir, fname)
            if not os.path.exists(path):
                missing.append(fname)
                continue

            df = pd.read_csv(path)
            if "Measure" not in df.columns:
                raise ValueError(f"Missing 'Measure' column in {fname}")

            raw_col = find_matching_column(df.columns.tolist(), raw_targets_n)
            corr_col = find_matching_column(df.columns.tolist(), corr_targets_n)
            if not raw_col or not corr_col:
                raise ValueError(
                    f"Cannot find error columns in {fname}. "
                    f"Expected '|dist_raw−d0|' and '|dist_corrected−d0|'. "
                    f"Got: {list(df.columns)}"
                )

            df_keep = keep_measures_for_model(df, model)
            if df_keep.empty:
                print(f"[WARN] No matching mm_const measures in {fname} for model={model}")
                continue

            # Ensure dict structure exists
            raw_abs_part.setdefault(model, {}).setdefault(pi, {})
            corr_abs_part.setdefault(model, {}).setdefault(pi, {})
            raw_abs_model.setdefault(model, {}).setdefault(pi, {})
            corr_abs_model.setdefault(model, {}).setdefault(pi, {})

            for _, row in df_keep.iterrows():
                measure_original = str(row["Measure"])            # for FILE 1
                measure_canon = canonical_measure_name(measure_original)  # for FILE 2

                raw_val = pd.to_numeric(row[raw_col], errors="coerce")
                corr_val = pd.to_numeric(row[corr_col], errors="coerce")

                raw_f = float(raw_val) if pd.notna(raw_val) else float("nan")
                corr_f = float(corr_val) if pd.notna(corr_val) else float("nan")

                # FILE 1 (by participant): keep original measure name (MP3D keeps *_3d_mm_const)
                raw_abs_part[model][pi][measure_original] = raw_f
                corr_abs_part[model][pi][measure_original] = corr_f

                # FILE 2 (by model): canonicalize so *_3d_mm_const == *_mm_const
                raw_abs_model[model][pi][measure_canon] = raw_f
                corr_abs_model[model][pi][measure_canon] = corr_f

                # Track measures per model for FILE 1 columns
                measures_per_model_part.setdefault(model, [])
                if measure_original not in measures_per_model_part[model]:
                    measures_per_model_part[model].append(measure_original)

    # Keep FILE 1 measures sorted per model (nice column order)
    for m in list(measures_per_model_part.keys()):
        measures_per_model_part[m] = sorted(measures_per_model_part[m])

    if missing:
        print("[WARN] Missing files (skipped):")
        for m in missing:
            print("   -", m)

    # FILE 1 measures: derived from measures_per_model_part (already original names)
    any_measures_part = any(len(v) for v in measures_per_model_part.values())
    if not any_measures_part:
        raise SystemExit("No measures found for participant tables. Check filenames and measure filters.")

    # FILE 2 measures: union of canonical measures actually present (prevents blank 3d columns entirely)
    measures_union_model = sorted({
        meas
        for model in models
        for pi in participants
        for meas in raw_abs_model.get(model, {}).get(pi, {}).keys()
    } | {
        meas
        for model in models
        for pi in participants
        for meas in corr_abs_model.get(model, {}).get(pi, {}).keys()
    })
    if not measures_union_model:
        raise SystemExit("No measures found for model tables. Check canonicalization and measure filters.")

    # -------- File 1: by participant (two sheets) --------
    t1_raw = build_participant_table(raw_abs_part, participants, models, measures_per_model_part)
    t1_corr = build_participant_table(corr_abs_part, participants, models, measures_per_model_part)

    out1_path = os.path.join(results_dir, args.out_xlsx1)
    with pd.ExcelWriter(out1_path, engine="openpyxl") as writer:
        t1_raw.to_excel(writer, sheet_name="ABS_RAW", index=True)
        t1_corr.to_excel(writer, sheet_name="ABS_CORRECTED", index=True)

    # -------- File 2: by model (two sheets) --------
    t2_raw = build_model_table(raw_abs_model, participants, models, measures_union_model)
    t2_corr = build_model_table(corr_abs_model, participants, models, measures_union_model)

    out2_path = os.path.join(results_dir, args.out_xlsx2)
    with pd.ExcelWriter(out2_path, engine="openpyxl") as writer:
        t2_raw.to_excel(writer, sheet_name="ABS_RAW", index=True)
        t2_corr.to_excel(writer, sheet_name="ABS_CORRECTED", index=True)

    print(f"OK -> {out1_path}")
    print(f"OK -> {out2_path}")


if __name__ == "__main__":
    main()


'''
cd /Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code_refactor

PYTHONPATH=src python scripts/run_metrics.py \
  --results_dir "/Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code_refactor/results/exp1_v6" \
  --out_xlsx1 "/Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code_refactor/results/exp1_v6/metrics/exp1_mmconst_by_participant_v6.xlsx" \
  --out_xlsx2 "/Users/carlamiquelblasco/Desktop/NONMANUAL/eyebrows/code_refactor/results/exp1_v6/metrics/exp1_mmconst_by_model_v6.xlsx"

'''