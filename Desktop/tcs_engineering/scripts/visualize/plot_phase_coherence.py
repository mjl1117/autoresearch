#!/usr/bin/env python3
"""Heptad register phase coherence heatmap and cluster diversity plot.

Two panels:
  1. Phase coherence heatmap: clusters × phase (0–6), colour = fraction of members
     in each phase. Clusters sorted by coherence (most coherent first).
     Working system clusters highlighted with gold border.

  2. Cluster size distribution: histogram of HK cluster sizes with overlay
     showing where phase-compatible candidates fall.

Output: results/visualization/phase_coherence_heatmap.pdf + .png
        results/visualization/cluster_size_distribution.pdf + .png

Dependencies: matplotlib, seaborn, pandas, numpy
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


PHASE_CMAP = "Blues"
WORKING_BORDER = "#F0E442"


def build_phase_matrix(candidates: pd.DataFrame) -> pd.DataFrame:
    """Build (cluster × phase) matrix of phase membership fractions.

    Uses cluster_dominant_phase + cluster_phase_coherence to reconstruct
    the approximate phase distribution: coherence fraction in dominant phase,
    remainder distributed evenly across other 6 phases.
    """
    rows = []
    for cluster_id, grp in candidates.groupby("protein_id"):
        row = grp.iloc[0]
        dom = row.get("cluster_dominant_phase", None)
        coh = float(row.get("cluster_phase_coherence", 0) or 0)
        if pd.isna(dom):
            continue
        fracs = np.ones(7) * (1 - coh) / 6
        fracs[int(dom)] = coh
        rows.append({
            "protein_id": cluster_id,
            "dominant_phase": int(dom),
            "coherence": coh,
            "known_tcs": str(row.get("known_tcs_system", "")),
            "working": bool(row.get("working_in_user_system", False)),
            **{f"phase_{i}": fracs[i] for i in range(7)}
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("coherence", ascending=False)
    return df


def plot_phase_heatmap(phase_df: pd.DataFrame, outdir: Path, top_n: int = 50):
    """Heatmap: rows = clusters (top N by coherence), columns = phases 0–6."""
    if phase_df.empty:
        print("  WARNING: no phase data available")
        return

    top = phase_df.head(top_n)
    matrix = top[[f"phase_{i}" for i in range(7)]].values

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.22)))

    im = ax.imshow(matrix, cmap=PHASE_CMAP, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(7))
    ax.set_xticklabels([f"Phase {i}" for i in range(7)], fontsize=9)
    ax.set_xlabel("Heptad register (0–6)", fontsize=10)

    ylabels = []
    for _, row in top.iterrows():
        label = str(row["protein_id"])[:20]
        if row["known_tcs"] and row["known_tcs"] != "nan":
            label = f"★ {row['known_tcs']}"  # known system
        ylabels.append(label)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(ylabels, fontsize=7)

    # Highlight working systems with gold border
    for i, (_, row) in enumerate(top.iterrows()):
        if row["working"]:
            j = int(row["dominant_phase"])
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                  fill=False, edgecolor=WORKING_BORDER,
                                  linewidth=3)
            ax.add_patch(rect)

    plt.colorbar(im, ax=ax, fraction=0.04, label="Fraction of cluster in phase")
    ax.set_title(
        f"Heptad register coherence — top {len(top)} clusters by coherence\n"
        "★ = user-confirmed working chimera system",
        fontsize=10
    )
    plt.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = outdir / f"phase_coherence_heatmap.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_cluster_sizes(candidates: pd.DataFrame, outdir: Path):
    """Histogram of HK cluster sizes, overlaid with phase-compatible candidates."""
    if candidates.empty or "cluster_size" not in candidates.columns:
        return

    hk = candidates[candidates["chimera_type"] == "HK_sensor_swap"].copy()
    if hk.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    sizes = hk["cluster_size"].dropna().astype(float)
    bins = np.logspace(np.log10(max(1, sizes.min())),
                       np.log10(sizes.max() + 1), 40)

    ax.hist(sizes, bins=bins, color="#AAAAAA", alpha=0.7, label="All HK clusters")

    compat = hk[hk["linker_phase_compatible"] == True]["cluster_size"].dropna()
    if not compat.empty:
        ax.hist(compat, bins=bins, color="#2166AC", alpha=0.8,
                label=f"Phase-compatible (n={len(compat)})")

    working = hk[hk["working_in_user_system"] == True]["cluster_size"].dropna()
    if not working.empty:
        for cs in working:
            ax.axvline(cs, color="#F0E442", linewidth=2.5, linestyle="--",
                       label="User working system" if cs == working.iloc[0] else "")

    ax.set_xscale("log")
    ax.set_xlabel("Cluster size (log scale)", fontsize=11)
    ax.set_ylabel("Number of clusters", fontsize=11)
    ax.set_title(
        "HK Cluster Size Distribution\n"
        "Large clusters → conserved kinase core → better chimera candidates",
        fontsize=10
    )
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = outdir / f"cluster_size_distribution.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chimera", required=True)
    parser.add_argument("--outdir",  required=True)
    parser.add_argument("--top_n",   type=int, default=50)
    args = parser.parse_args()

    if not Path(args.chimera).exists():
        print(f"ERROR: {args.chimera} not found", file=sys.stderr)
        import sys; sys.exit(1)

    candidates = pd.read_csv(args.chimera, sep="\t")
    phase_df = build_phase_matrix(candidates)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_phase_heatmap(phase_df, outdir, args.top_n)
    plot_cluster_sizes(candidates, outdir)

    # Ensure output files exist even when data is absent (Snakemake requires them)
    for fname in ("phase_coherence_heatmap.png", "phase_coherence_heatmap.pdf",
                  "cluster_size_distribution.png", "cluster_size_distribution.pdf"):
        p = outdir / fname
        if not p.exists():
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No phase data available", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.axis("off")
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Placeholder saved: {p}")

    print("Done.")


if __name__ == "__main__":
    main()
