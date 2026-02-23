#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot ABS_RAW vs ABS_CORRECTED from an Excel file with two sheets:
  - ABS_RAW
  - ABS_CORRECTED

Rows: p1, p2, p3, p4, average (optional)
Columns: e.g.
  MP3D__L_all_mean_3d_mm_const
  MP2D__R_outer_mean_mm_const
  TDDFA_2D__L_inner_mean_mm_const
  TDDFA_3D__R_all_mean_3d_mm_const

Outputs several figures (png) into an output directory.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Parsing column names
# ----------------------------

COL_RE = re.compile(
    r"^(?P<model>MP3D|MP2D|TDDFA_2D|TDDFA_3D)__"
    r"(?P<side>L|R)_(?P<region>all|inner|outer)_mean"
    r"(?:_3d)?_mm_const$"
)

def parse_col(col: str) -> Optional[dict]:
    m = COL_RE.match(col)
    if not m:
        return None
    model = m.group("model")
    side = m.group("side")
    region = m.group("region")

    if model.startswith("MP"):
        backend = "MP"
        approach = "3D" if model == "MP3D" else "2D"
    elif model.startswith("TDDFA"):
        backend = "TDDFA"
        approach = "3D" if model == "TDDFA_3D" else "2D"
    else:
        backend = "UNKNOWN"
        approach = "UNKNOWN"

    return {
        "model": model,
        "backend": backend,
        "approach": approach,
        "side": side,
        "region": region,
    }


def wide_to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """
    Convert wide df with index=person and columns=metrics into long tidy format.
    """
    # Ensure person index
    if df.index.name is None:
        df.index.name = "person"
    df = df.copy()
    df.index = df.index.astype(str)

    records = []
    for col in df.columns:
        meta = parse_col(str(col))
        if meta is None:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        tmp = pd.DataFrame({
            "person": df.index,
            value_name: s.values,
            **meta
        })
        records.append(tmp)

    if not records:
        raise ValueError("No columns matched expected pattern. Check column names / regex.")

    out = pd.concat(records, ignore_index=True)
    return out


# ----------------------------
# Plot helpers
# ----------------------------

def ensure_outdir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def persons_order(df: pd.DataFrame, include_average_row: bool) -> List[str]:
    people = sorted([p for p in df["person"].unique() if p.lower().startswith("p")])
    if include_average_row and "average" in [p.lower() for p in df["person"].unique()]:
        # Preserve exact label (could be "average" or "Average")
        avg_label = next(p for p in df["person"].unique() if p.lower() == "average")
        people.append(avg_label)
    return people


def facet_keys() -> List[Tuple[str, str]]:
    # Order for consistent layout
    return [("MP", "2D"), ("MP", "3D"), ("TDDFA", "2D"), ("TDDFA", "3D")]


def plot_grouped_bars(
    df: pd.DataFrame,
    out_path: str,
    region: str,
    side: str,
    include_average_row: bool,
    title_prefix: str = "",
) -> None:
    """
    Grouped bars raw vs corrected per person, faceted by backend×approach (2x2).
    Expects columns: abs_raw, abs_corrected, backend, approach, person, region, side.
    """
    d = df[(df["region"] == region) & (df["side"] == side)].copy()
    if d.empty:
        return

    people = persons_order(d, include_average_row)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    axes = axes.ravel()

    for i, (backend, approach) in enumerate(facet_keys()):
        ax = axes[i]
        dd = d[(d["backend"] == backend) & (d["approach"] == approach)].copy()

        # Align to ordered people
        dd["person"] = pd.Categorical(dd["person"], categories=people, ordered=True)
        dd = dd.sort_values("person")

        # If some person missing, keep placeholders
        raw_vals = []
        cor_vals = []
        for p in people:
            row = dd[dd["person"] == p]
            if len(row) == 0:
                raw_vals.append(np.nan)
                cor_vals.append(np.nan)
            else:
                raw_vals.append(float(row["abs_raw"].iloc[0]))
                cor_vals.append(float(row["abs_corrected"].iloc[0]))

        x = np.arange(len(people))
        width = 0.38

        ax.bar(x - width/2, raw_vals, width, label="abs_raw")
        ax.bar(x + width/2, cor_vals, width, label="abs_corrected")

        ax.set_title(f"{backend} {approach}")
        ax.set_xticks(x)
        ax.set_xticklabels(people, rotation=0)
        ax.grid(True, axis="y", alpha=0.3)

        if i in (0, 2):
            ax.set_ylabel("Absolute error (mm_const)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    ttl = f"{title_prefix} Raw vs Corrected | region={region} | side={side}"
    fig.suptitle(ttl, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_scatter_raw_vs_corrected(
    df: pd.DataFrame,
    out_path: str,
    region: str,
    side: str,
    include_average_row: bool,
    title_prefix: str = "",
) -> None:
    """
    Scatter x=raw y=corrected with diagonal y=x, faceted by backend×approach (2x2).
    """
    d = df[(df["region"] == region) & (df["side"] == side)].copy()
    if not include_average_row:
        d = d[d["person"].str.lower() != "average"].copy()
    if d.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.ravel()

    # global limits
    vals = pd.concat([d["abs_raw"], d["abs_corrected"]], ignore_index=True)
    vmax = float(np.nanmax(vals.values)) if np.isfinite(np.nanmax(vals.values)) else 1.0
    lim = (0.0, vmax * 1.05)

    for i, (backend, approach) in enumerate(facet_keys()):
        ax = axes[i]
        dd = d[(d["backend"] == backend) & (d["approach"] == approach)].copy()

        ax.scatter(dd["abs_raw"], dd["abs_corrected"])
        # annotate with person labels
        for _, r in dd.iterrows():
            ax.annotate(str(r["person"]), (r["abs_raw"], r["abs_corrected"]), fontsize=9, alpha=0.8)

        ax.plot(lim, lim, linestyle="--")  # y=x
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_title(f"{backend} {approach}")
        ax.grid(True, alpha=0.3)
        if i in (2, 3):
            ax.set_xlabel("abs_raw")
        if i in (0, 2):
            ax.set_ylabel("abs_corrected")

    ttl = f"{title_prefix} Scatter raw vs corrected | region={region} | side={side}"
    fig.suptitle(ttl, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_improvement(
    df: pd.DataFrame,
    out_path: str,
    region: str,
    side: str,
    include_average_row: bool,
    title_prefix: str = "",
) -> None:
    """
    Plot improvement = raw - corrected (positive is better), faceted by backend×approach.
    """
    d = df[(df["region"] == region) & (df["side"] == side)].copy()
    d["improvement"] = d["abs_raw"] - d["abs_corrected"]

    if not include_average_row:
        d = d[d["person"].str.lower() != "average"].copy()
    if d.empty:
        return

    people = persons_order(d, include_average_row)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    axes = axes.ravel()

    for i, (backend, approach) in enumerate(facet_keys()):
        ax = axes[i]
        dd = d[(d["backend"] == backend) & (d["approach"] == approach)].copy()
        dd["person"] = pd.Categorical(dd["person"], categories=people, ordered=True)
        dd = dd.sort_values("person")

        y = []
        for p in people:
            row = dd[dd["person"] == p]
            y.append(float(row["improvement"].iloc[0]) if len(row) else np.nan)

        x = np.arange(len(people))
        ax.bar(x, y)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_title(f"{backend} {approach}")
        ax.set_xticks(x)
        ax.set_xticklabels(people)
        ax.grid(True, axis="y", alpha=0.3)
        if i in (0, 2):
            ax.set_ylabel("Improvement = abs_raw - abs_corrected")

    ttl = f"{title_prefix} Improvement (raw - corrected) | region={region} | side={side}"
    fig.suptitle(ttl, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ----------------------------
# Aggregation
# ----------------------------

def add_average_over_people(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average across persons starting with 'p' (p1..), ignoring any existing 'average' row.
    Adds a new row with person='average_calc'.
    """
    d = df_long.copy()
    d = d[d["person"].str.lower().str.startswith("p")].copy()
    gcols = ["model", "backend", "approach", "side", "region"]
    avg = d.groupby(gcols, as_index=False)["value"].mean()
    avg["person"] = "average_calc"
    return pd.concat([df_long, avg], ignore_index=True)


# ----------------------------
# Main
# ----------------------------

'''
python scripts/run_plots_by_p.py --xlsx results/exp1_v6/metrics/exp1_mmconst_by_participant_v6.xlsx --out_dir results/exp1_v6/plots_by_p --add_average_calc
'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Input .xlsx with sheets ABS_RAW and ABS_CORRECTED.")
    ap.add_argument("--out_dir", default="plots_abs", help="Output directory for figures.")
    ap.add_argument("--include_average_row", action="store_true",
                    help="Include the Excel 'average' row in plots (if present).")
    ap.add_argument("--add_average_calc", action="store_true",
                    help="Compute average over p* rows and add as 'average_calc' (recommended).")
    ap.add_argument("--regions", default="all,inner,outer",
                    help="Comma-separated regions to plot (default: all,inner,outer).")
    ap.add_argument("--sides", default="L,R",
                    help="Comma-separated sides to plot (default: L,R).")
    args = ap.parse_args()

    ensure_outdir(args.out_dir)

    # Read excel
    df_raw = pd.read_excel(args.xlsx, sheet_name="ABS_RAW", index_col=0)
    df_cor = pd.read_excel(args.xlsx, sheet_name="ABS_CORRECTED", index_col=0)

    raw_long = wide_to_long(df_raw, "abs_raw").rename(columns={"abs_raw": "value"})
    cor_long = wide_to_long(df_cor, "abs_corrected").rename(columns={"abs_corrected": "value"})

    raw_long["metric"] = "abs_raw"
    cor_long["metric"] = "abs_corrected"

    # Merge raw/corrected into one table
    raw_w = raw_long.rename(columns={"value": "abs_raw"}).drop(columns=["metric"])
    cor_w = cor_long.rename(columns={"value": "abs_corrected"}).drop(columns=["metric"])

    key_cols = ["person", "model", "backend", "approach", "side", "region"]
    df = pd.merge(raw_w, cor_w, on=key_cols, how="outer")

    # Optionally add average over people
    if args.add_average_calc:
        # to reuse add_average_over_people, we create a long "value" form per metric
        tmp_long = pd.concat([
            df[key_cols + ["abs_raw"]].rename(columns={"abs_raw": "value"}).assign(metric="abs_raw"),
            df[key_cols + ["abs_corrected"]].rename(columns={"abs_corrected": "value"}).assign(metric="abs_corrected"),
        ], ignore_index=True)

        tmp_long = add_average_over_people(tmp_long)

        df = (
            tmp_long.pivot_table(
                index=key_cols,
                columns="metric",
                values="value",
                aggfunc="first"
            )
            .reset_index()
        )

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    sides = [s.strip() for s in args.sides.split(",") if s.strip()]

    # Make plots
    for region in regions:
        for side in sides:
            base = f"region_{region}_side_{side}"

            plot_grouped_bars(
                df, os.path.join(args.out_dir, f"{base}__bars_raw_vs_corrected.png"),
                region=region, side=side,
                include_average_row=args.include_average_row,
                title_prefix="ABS"
            )

            plot_scatter_raw_vs_corrected(
                df, os.path.join(args.out_dir, f"{base}__scatter_raw_vs_corrected.png"),
                region=region, side=side,
                include_average_row=args.include_average_row,
                title_prefix="ABS"
            )

            plot_improvement(
                df, os.path.join(args.out_dir, f"{base}__improvement.png"),
                region=region, side=side,
                include_average_row=args.include_average_row,
                title_prefix="ABS"
            )

    # Save tidy data used for plotting
    df.to_csv(os.path.join(args.out_dir, "tidy_abs_raw_vs_corrected.csv"), index=False)
    print(f"Saved plots and tidy CSV to: {args.out_dir}")


if __name__ == "__main__":
    main()
