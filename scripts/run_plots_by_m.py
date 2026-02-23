#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot ABS_RAW vs ABS_CORRECTED from a *by-model* Excel file with two sheets:
  - ABS_RAW
  - ABS_CORRECTED

Rows (index): MP3D, MP2D, TDDFA_2D, TDDFA_3D
Columns:
  L_all_mean_mm_const
  L_inner_mean_mm_const
  L_outer_mean_mm_const
  R_all_mean_mm_const
  R_inner_mean_mm_const
  R_outer_mean_mm_const

Outputs several figures (png) into an output directory.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Parsing columns (by-model file)
# ----------------------------

COL_RE = re.compile(
    r"^(?P<side>L|R)_(?P<region>all|inner|outer)_mean_mm_const$"
)

MODEL_ORDER = ["MP3D", "MP2D", "TDDFA_2D", "TDDFA_3D"]

MODEL_META: Dict[str, Dict[str, str]] = {
    "MP3D": {"backend": "MP", "approach": "3D"},
    "MP2D": {"backend": "MP", "approach": "2D"},
    "TDDFA_2D": {"backend": "TDDFA", "approach": "2D"},
    "TDDFA_3D": {"backend": "TDDFA", "approach": "3D"},
}


def parse_col(col: str) -> Optional[dict]:
    m = COL_RE.match(col)
    if not m:
        return None
    return {
        "side": m.group("side"),
        "region": m.group("region"),
    }


def wide_to_long_by_model(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """
    Convert wide df with index=model and columns like L_all_mean_mm_const into long tidy format.
    """
    if df.index.name is None:
        df.index.name = "model"
    df = df.copy()
    df.index = df.index.astype(str)

    records = []
    for col in df.columns:
        meta = parse_col(str(col))
        if meta is None:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        tmp = pd.DataFrame({
            "model": df.index,
            value_name: s.values,
            **meta
        })
        records.append(tmp)

    if not records:
        raise ValueError("No columns matched expected pattern. Check column names / regex.")

    out = pd.concat(records, ignore_index=True)

    # add backend/approach from model
    out["backend"] = out["model"].map(lambda m: MODEL_META.get(m, {}).get("backend", "UNKNOWN"))
    out["approach"] = out["model"].map(lambda m: MODEL_META.get(m, {}).get("approach", "UNKNOWN"))

    return out


# ----------------------------
# Plot helpers
# ----------------------------

def ensure_outdir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def models_order(df: pd.DataFrame) -> List[str]:
    present = [m for m in MODEL_ORDER if m in set(df["model"].unique())]
    # include any extra models (unexpected) at end
    extra = sorted([m for m in df["model"].unique() if m not in present])
    return present + extra


def plot_grouped_bars_models(
    df: pd.DataFrame,
    out_path: str,
    region: str,
    side: str,
    title_prefix: str = "",
) -> None:
    """
    Grouped bars raw vs corrected per model (single panel).
    Expects columns: abs_raw, abs_corrected, model, region, side.
    """
    d = df[(df["region"] == region) & (df["side"] == side)].copy()
    if d.empty:
        return

    mods = models_order(d)
    d["model"] = pd.Categorical(d["model"], categories=mods, ordered=True)
    d = d.sort_values("model")

    raw_vals = []
    cor_vals = []
    for m in mods:
        row = d[d["model"] == m]
        raw_vals.append(float(row["abs_raw"].iloc[0]) if len(row) else np.nan)
        cor_vals.append(float(row["abs_corrected"].iloc[0]) if len(row) else np.nan)

    x = np.arange(len(mods))
    width = 0.38

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.bar(x - width/2, raw_vals, width, label="abs_raw")
    ax.bar(x + width/2, cor_vals, width, label="abs_corrected")

    ax.set_xticks(x)
    ax.set_xticklabels(mods)
    ax.set_ylabel("Absolute error (mm_const)")
    ax.grid(True, axis="y", alpha=0.3)

    ttl = f"{title_prefix} Raw vs Corrected | region={region} | side={side}"
    ax.set_title(ttl)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_scatter_raw_vs_corrected_models(
    df: pd.DataFrame,
    out_path: str,
    region: str,
    side: str,
    title_prefix: str = "",
) -> None:
    """
    Scatter x=raw y=corrected with diagonal y=x (single panel), labeled by model.
    """
    d = df[(df["region"] == region) & (df["side"] == side)].copy()
    if d.empty:
        return

    vals = pd.concat([d["abs_raw"], d["abs_corrected"]], ignore_index=True)
    vmax = float(np.nanmax(vals.values)) if np.isfinite(np.nanmax(vals.values)) else 1.0
    lim = (0.0, vmax * 1.05)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
    ax.scatter(d["abs_raw"], d["abs_corrected"])

    for _, r in d.iterrows():
        ax.annotate(str(r["model"]), (r["abs_raw"], r["abs_corrected"]), fontsize=10, alpha=0.85)

    ax.plot(lim, lim, linestyle="--")  # y=x
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("abs_raw")
    ax.set_ylabel("abs_corrected")
    ax.grid(True, alpha=0.3)

    ttl = f"{title_prefix} Scatter raw vs corrected | region={region} | side={side}"
    ax.set_title(ttl)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_improvement_models(
    df: pd.DataFrame,
    out_path: str,
    region: str,
    side: str,
    title_prefix: str = "",
) -> None:
    """
    Plot improvement = raw - corrected (positive is better), per model (single panel).
    """
    d = df[(df["region"] == region) & (df["side"] == side)].copy()
    if d.empty:
        return

    d["improvement"] = d["abs_raw"] - d["abs_corrected"]

    mods = models_order(d)
    d["model"] = pd.Categorical(d["model"], categories=mods, ordered=True)
    d = d.sort_values("model")

    y = []
    for m in mods:
        row = d[d["model"] == m]
        y.append(float(row["improvement"].iloc[0]) if len(row) else np.nan)

    x = np.arange(len(mods))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.bar(x, y)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(mods)
    ax.set_ylabel("Improvement = abs_raw - abs_corrected")
    ax.grid(True, axis="y", alpha=0.3)

    ttl = f"{title_prefix} Improvement (raw - corrected) | region={region} | side={side}"
    ax.set_title(ttl)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Input .xlsx with sheets ABS_RAW and ABS_CORRECTED (by-model).")
    ap.add_argument("--out_dir", default="plots_abs_by_model", help="Output directory for figures.")
    ap.add_argument("--regions", default="all,inner,outer",
                    help="Comma-separated regions to plot (default: all,inner,outer).")
    ap.add_argument("--sides", default="L,R",
                    help="Comma-separated sides to plot (default: L,R).")
    args = ap.parse_args()

    ensure_outdir(args.out_dir)

    # Read excel (by-model: index_col=0 is model)
    df_raw = pd.read_excel(args.xlsx, sheet_name="ABS_RAW", index_col=0)
    df_cor = pd.read_excel(args.xlsx, sheet_name="ABS_CORRECTED", index_col=0)

    raw_long = wide_to_long_by_model(df_raw, "abs_raw").rename(columns={"abs_raw": "value"})
    cor_long = wide_to_long_by_model(df_cor, "abs_corrected").rename(columns={"abs_corrected": "value"})

    raw_w = raw_long.rename(columns={"value": "abs_raw"})
    cor_w = cor_long.rename(columns={"value": "abs_corrected"})

    key_cols = ["model", "backend", "approach", "side", "region"]
    df = pd.merge(raw_w[key_cols + ["abs_raw"]], cor_w[key_cols + ["abs_corrected"]], on=key_cols, how="outer")

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    sides = [s.strip() for s in args.sides.split(",") if s.strip()]

    # Make plots
    for region in regions:
        for side in sides:
            base = f"region_{region}_side_{side}"

            plot_grouped_bars_models(
                df, os.path.join(args.out_dir, f"{base}__bars_raw_vs_corrected.png"),
                region=region, side=side,
                title_prefix="ABS (by model)"
            )

            plot_scatter_raw_vs_corrected_models(
                df, os.path.join(args.out_dir, f"{base}__scatter_raw_vs_corrected.png"),
                region=region, side=side,
                title_prefix="ABS (by model)"
            )

            plot_improvement_models(
                df, os.path.join(args.out_dir, f"{base}__improvement.png"),
                region=region, side=side,
                title_prefix="ABS (by model)"
            )

    # Save tidy data used for plotting
    df.to_csv(os.path.join(args.out_dir, "tidy_abs_by_model_raw_vs_corrected.csv"), index=False)
    print(f"Saved plots and tidy CSV to: {args.out_dir}")


if __name__ == "__main__":
    main()


    """
Example:
python scripts/run_plots_by_m.py \
  --xlsx results/exp1_v6/metrics/exp1_mmconst_by_model_v6.xlsx \
  --out_dir results/exp1_v6/plots_by_model
"""