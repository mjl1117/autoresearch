#!/usr/bin/env python3
"""UMAP embedding of TCS sequence space.

Builds a UMAP from MMseqs2 pairwise identity scores (hk_homology.m8 +
rr_homology.m8), converting identity to distance (1 - pident/100).
Sparse matrix → UMAP 2D embedding → publication figure.

Coloring layers:
  - Primary: protein type (HK blue / RR orange)
  - Secondary: DBD family (OmpR_PhoB, NarL_FixJ, NtrC, CheY, other)
    shown as marker shape
  - Size: cluster_size (larger = more conserved)
  - Labels: top-ranked chimera candidates + user working systems

Output: results/visualization/tcs_umap.pdf + .png

Dependencies: umap-learn, scipy, matplotlib, seaborn, pandas, numpy
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


HK_COLOR = "#2166AC"
RR_COLOR = "#D6604D"

DBD_PALETTE = {
    "OmpR_PhoB": "#1A9641",
    "NarL_FixJ": "#A6D96A",
    "NtrC_AAA":  "#FDAE61",
    "CheY":      "#D7191C",
    "Spo0A":     "#762A83",
    "other":     "#BBBBBB",
}

DBD_MARKERS = {
    "OmpR_PhoB": "o",
    "NarL_FixJ": "s",
    "NtrC_AAA":  "^",
    "CheY":      "D",
    "Spo0A":     "P",
    "other":     ".",
}

WORKING_SYSTEMS = {"NarXL", "PhoRB", "NarX", "NarL", "PhoR", "PhoB"}


def load_homology(hk_m8: str, rr_m8: str) -> pd.DataFrame:
    """Load both homology files, tag protein type, return unified edge list."""
    cols = ["query", "target", "pident", "length", "mismatch", "gapopen",
            "qstart", "qend", "tstart", "tend", "evalue", "bitscore"]
    frames = []
    for path, ptype in [(hk_m8, "HK"), (rr_m8, "RR")]:
        if not Path(path).exists():
            print(f"  WARNING: {path} not found — skipping")
            continue
        df = pd.read_csv(path, sep="\t", header=None, names=cols)
        df["ptype"] = ptype
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_distance_matrix(edges: pd.DataFrame):
    """Build sparse distance matrix from pairwise identity edges.

    Returns (protein_ids array, scipy sparse matrix).
    """
    from scipy.sparse import csr_matrix

    proteins = pd.unique(pd.concat([edges["query"], edges["target"]]))
    idx = {p: i for i, p in enumerate(proteins)}
    n = len(proteins)

    rows = edges["query"].map(idx).values
    cols = edges["target"].map(idx).values
    # Distance = 1 - identity/100; clip to [0, 1]
    dists = np.clip(1.0 - edges["pident"].values / 100.0, 0, 1)

    mat = csr_matrix((dists, (rows, cols)), shape=(n, n))
    # Symmetrise (take minimum distance for each pair)
    mat = mat.minimum(mat.T)
    return proteins, mat


def run_umap(dist_matrix, n_neighbors: int = 15, min_dist: float = 0.1):
    """Run UMAP on a precomputed distance matrix."""
    import umap
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="precomputed",
        random_state=42,
        n_components=2,
        low_memory=True,
    )
    return reducer.fit_transform(dist_matrix.toarray())


def plot_umap(embedding: np.ndarray, proteins: np.ndarray,
              chimera_df: pd.DataFrame, hk_ids: set, outdir: Path):
    """Publication-quality UMAP scatter plot with annotation layers."""

    n = len(proteins)
    ptypes = np.array(["HK" if p in hk_ids else "RR" for p in proteins])

    # DBD family from chimera_df
    dbd_map = {}
    if chimera_df is not None and "dbd_family" in chimera_df.columns:
        dbd_map = chimera_df.set_index("protein_id")["dbd_family"].fillna("other").to_dict()

    dbd = np.array([dbd_map.get(p, "other") for p in proteins])

    # Cluster size for point size
    size_map = {}
    if chimera_df is not None and "cluster_size" in chimera_df.columns:
        size_map = chimera_df.set_index("protein_id")["cluster_size"].to_dict()
    sizes = np.array([np.clip(size_map.get(p, 5), 2, 200) for p in proteins])
    sizes = np.sqrt(sizes) * 2  # sqrt scale to prevent huge points

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax_idx, (ax, color_by) in enumerate(zip(axes, ["type", "dbd"])):
        if color_by == "type":
            colors = np.where(ptypes == "HK", HK_COLOR, RR_COLOR)
            title = "Coloured by protein type (HK / RR)"
        else:
            colors = np.array([DBD_PALETTE.get(d, "#BBBBBB") for d in dbd])
            title = "Coloured by DBD family"

        # Background points
        ax.scatter(embedding[:, 0], embedding[:, 1],
                   c=colors, s=sizes, alpha=0.45, linewidths=0,
                   rasterized=True)

        # Highlight working systems
        if chimera_df is not None and "known_tcs_system" in chimera_df.columns:
            working = chimera_df[chimera_df["working_in_user_system"] == True]
            for _, row in working.iterrows():
                pid = row["protein_id"]
                if pid in {p: i for i, p in enumerate(proteins)}:
                    pidx = np.where(proteins == pid)[0]
                    if len(pidx):
                        i = pidx[0]
                        ax.scatter(embedding[i, 0], embedding[i, 1],
                                   c="gold", s=120, marker="*",
                                   edgecolors="black", linewidths=0.5, zorder=5)
                        ax.annotate(
                            row.get("known_tcs_system", pid),
                            (embedding[i, 0], embedding[i, 1]),
                            fontsize=7, fontweight="bold",
                            xytext=(5, 5), textcoords="offset points",
                            arrowprops=dict(arrowstyle="-", color="black", lw=0.5),
                        )

        ax.set_xlabel("UMAP 1", fontsize=10)
        ax.set_ylabel("UMAP 2", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legends
    type_legend = [
        mpatches.Patch(color=HK_COLOR, label=f"HK (n={np.sum(ptypes=='HK'):,})"),
        mpatches.Patch(color=RR_COLOR, label=f"RR (n={np.sum(ptypes=='RR'):,})"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                   markeredgecolor="black", markersize=10,
                   label="User confirmed working chimera"),
    ]
    axes[0].legend(handles=type_legend, fontsize=8, loc="lower left", framealpha=0.9)

    dbd_legend = [
        mpatches.Patch(color=DBD_PALETTE[d], label=d)
        for d in DBD_PALETTE if d != "other"
    ] + [mpatches.Patch(color=DBD_PALETTE["other"], label="other / unknown")]
    axes[1].legend(handles=dbd_legend, fontsize=8, loc="lower left",
                   framealpha=0.9, ncol=2)

    fig.suptitle(
        f"TCS Sequence Space — UMAP of pairwise MMseqs2 identity\n"
        f"(n = {n:,} cluster representatives; distance = 1 − pident)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = outdir / f"tcs_umap.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hk_homology", required=True)
    parser.add_argument("--rr_homology", required=True)
    parser.add_argument("--chimera",     required=True)
    parser.add_argument("--hk_ann",      required=True)
    parser.add_argument("--outdir",      required=True)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist",    type=float, default=0.1)
    args = parser.parse_args()

    print("Loading homology data...")
    edges = load_homology(args.hk_homology, args.rr_homology)
    if edges.empty:
        print("ERROR: no homology data found", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(edges):,} pairwise alignments → building distance matrix")
    proteins, dist_mat = build_distance_matrix(edges)
    print(f"  {len(proteins):,} proteins × {len(proteins):,} distance matrix")

    print("Running UMAP...")
    embedding = run_umap(dist_mat, args.n_neighbors, args.min_dist)

    # Load annotation
    hk_ids = set()
    if Path(args.hk_ann).exists():
        ann = pd.read_csv(args.hk_ann, sep="\t", header=None,
                          names=["qseqid","sseqid","pident","length",
                                 "qcovhsp","evalue","bitscore","stitle"])
        hk_ids = set(ann["qseqid"].unique())

    chimera_df = None
    if Path(args.chimera).exists():
        chimera_df = pd.read_csv(args.chimera, sep="\t")

    print("Plotting UMAP...")
    plot_umap(embedding, proteins, chimera_df, hk_ids, Path(args.outdir))
    print("Done.")


if __name__ == "__main__":
    main()
