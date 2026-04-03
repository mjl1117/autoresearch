#!/usr/bin/env python3
"""UMAP embedding of TCS sequence space — separated by protein class.

Two-panel figure:
  Left:  HK sensor kinases only — coloured by sensor architecture
         (HAMP / PAS / GAF / PHY / other), sized by cluster_size.
  Right: RR response regulators only — coloured by DBD family
         (OmpR_PhoB / NarL_FixJ / NtrC_AAA / CheY / Spo0A / other).

Stars mark user-confirmed working chimera systems (NarXL, PhoRB).
Labels added for all annotated known TCS systems (EnvZ-OmpR, PhoQP, etc.).
Chimera candidate labels include swap type and junction position.

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


# ─── Colour palettes ─────────────────────────────────────────────────────────

HK_SENSOR_PALETTE = {
    "HAMP":  "#2166AC",   # canonical membrane sensor → blue
    "PAS":   "#74ADD1",   # cytoplasmic/periplasmic PAS → light blue
    "GAF":   "#ABD9E9",   # GAF sensor → pale blue
    "PHY":   "#E0F3F8",   # phytochrome → very pale blue
    "other": "#BBBBBB",
}

RR_DBD_PALETTE = {
    "OmpR_PhoB": "#1A9641",   # green — most common
    "NarL_FixJ": "#A6D96A",   # light green
    "NtrC_AAA":  "#FDAE61",   # orange
    "CheY":      "#D7191C",   # red
    "Spo0A":     "#762A83",   # purple
    "other":     "#BBBBBB",
}

WORKING_COLOR = "gold"

# ─── Keyword → class mappings ─────────────────────────────────────────────────

# Sensor type inference from DIAMOND stitle
HK_SENSOR_KEYWORDS = {
    "PHY":  ["phytochrome", "bphp", "cph1", "bphq"],
    "GAF":  ["gaf domain", "nreb", "nrec", "gas sensor"],
    "PAS":  ["pas domain", "pas sensor", "ntrb", "dcus", "dpib", "arce", "arcs",
             "nitrogen regulation", "citrate sensor"],
    "HAMP": ["hamp", "histidine kinase", "narx", "narq", "phor", "phoq", "envz",
             "kdpd", "cpxa", "baes", "rstb", "evgs", "tors", "rese", "ccka",
             "sensor histidine"],
}

# DBD family inference from DIAMOND stitle
RR_DBD_KEYWORDS = {
    "Spo0A":     ["spo0a", "sporulation", "stage 0"],
    "CheY":      ["chey", "chemotaxis response regulator", "chea"],
    "NtrC_AAA":  ["ntrc", "nifa", "nifa", "luxo", "sigma-54", "aaa+",
                  "enhancer-binding", "ntr regulator"],
    "NarL_FixJ": ["narl", "narp", "fixj", "gera", "coma", "flhd", "luxr-type",
                  "narL", "response regulator narl", "transcriptional regulatory protein narl",
                  "probable transcriptional regulatory protein narl"],
    "OmpR_PhoB": ["ompr", "phob", "phop", "rsca", "cpxr", "baer", "kdpe",
                  "rsta", "evga", "tcrr", "rega", "resd", "degu", "ctra",
                  "vpst", "gaca", "nrec", "rcsb", "uvry", "arca", "torr",
                  "ompR", "phob", "response regulator phob", "response regulator ompr"],
}

# Known system label lookup — keyword in stitle → display label
KNOWN_SYSTEM_LABELS = {
    "narx":       "NarXL",     "narl":   "NarXL",
    "narq":       "NarQP",     "narp":   "NarQP",
    "phor ":      "PhoRB",     "phob":   "PhoRB",
    "envz":       "EnvZ-OmpR", "ompr":   "EnvZ-OmpR",
    "phoq":       "PhoQP",     "phop":   "PhoQP",
    "kdpd":       "KdpDE",     "kdpe":   "KdpDE",
    "rstb":       "RstBA",     "rsta":   "RstBA",
    "ntrb":       "NtrBC",     "ntrc":   "NtrBC",
    "chea":       "CheAY",     "chey":   "CheAY",
    "cpxa":       "CpxAR",     "cpxr":   "CpxAR",
    "kina":       "KinA-Spo0A","spo0a":  "KinA-Spo0A",
    "ccka":       "CckA-CtrA", "ctra":   "CckA-CtrA",
    "bphp":       "BphP1",
    "cph1":       "Cph1-Rcp1",
}


def classify_from_stitle(stitle: str, keyword_map: dict, default: str = "other") -> str:
    """Return class label by scanning stitle against keyword groups (order = priority)."""
    t = str(stitle).lower()
    for label, keywords in keyword_map.items():
        if any(k in t for k in keywords):
            return label
    return default


def known_system_label(stitle: str) -> str | None:
    """Return known system shorthand if stitle matches, else None."""
    t = str(stitle).lower()
    for keyword, label in KNOWN_SYSTEM_LABELS.items():
        if keyword in t:
            return label
    return None


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


def build_distance_matrix(edges: pd.DataFrame, ptype_filter: str = None):
    """Build sparse distance matrix from pairwise identity edges.

    If ptype_filter is 'HK' or 'RR', only include edges of that type.
    Returns (protein_ids array, scipy sparse matrix).
    """
    from scipy.sparse import csr_matrix

    if ptype_filter:
        edges = edges[edges["ptype"] == ptype_filter]

    proteins = pd.unique(pd.concat([edges["query"], edges["target"]]))
    idx = {p: i for i, p in enumerate(proteins)}
    n = len(proteins)

    rows = edges["query"].map(idx).values
    cols = edges["target"].map(idx).values
    dists = np.clip(1.0 - edges["pident"].values / 100.0, 0, 1)

    mat = csr_matrix((dists, (rows, cols)), shape=(n, n))
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


def chimera_label(row: pd.Series) -> str:
    """Build chimera annotation label with swap type and junction info."""
    system = str(row.get("known_tcs_system", "") or "")
    ctype = str(row.get("chimera_type", "") or "")
    hamp = row.get("hamp_start", None)

    if "sensor_swap" in ctype:
        if hamp and pd.notna(hamp):
            suffix = f" [sensor↔HAMP@{int(hamp)}]"
        else:
            suffix = " [sensor swap]"
    elif "DBD_swap" in ctype:
        suffix = " [DBD swap]"
    else:
        suffix = ""

    return f"{system}{suffix}" if system else row["protein_id"][:12]


def plot_panel(ax, embedding, proteins, ann_df, chimera_df,
               ptype: str, color_by: str, palette: dict, title: str,
               keyword_map: dict = None):
    """Draw one UMAP panel for a single protein type."""

    # Classify each protein
    stitle_map = {}
    if ann_df is not None:
        stitle_map = ann_df.set_index("qseqid")["stitle"].to_dict()

    kmap = keyword_map if keyword_map is not None else (
        HK_SENSOR_KEYWORDS if ptype == "HK" else RR_DBD_KEYWORDS
    )
    classes = []
    for p in proteins:
        stitle = stitle_map.get(p, "")
        cls = classify_from_stitle(stitle, kmap)
        classes.append(cls)
    classes = np.array(classes)

    # Sizes from cluster_size
    size_map = {}
    if chimera_df is not None and "cluster_size" in chimera_df.columns:
        size_map = chimera_df.set_index("protein_id")["cluster_size"].to_dict()
    sizes = np.array([np.clip(np.sqrt(size_map.get(p, 5)), 1, 14) * 2
                      for p in proteins])

    # Draw background scatter by class
    drawn_labels = set()
    for cls, color in palette.items():
        mask = classes == cls
        if not mask.any():
            continue
        label = cls if cls not in drawn_labels else "_nolegend_"
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=color, s=sizes[mask], alpha=0.5, linewidths=0,
                   label=label, rasterized=True)
        drawn_labels.add(cls)

    # Grey for "other" if not in palette
    other_mask = np.array([c not in palette for c in classes])
    if other_mask.any():
        ax.scatter(embedding[other_mask, 0], embedding[other_mask, 1],
                   c="#BBBBBB", s=sizes[other_mask], alpha=0.3, linewidths=0,
                   label="other", rasterized=True)

    # Annotate known systems from annotation
    labeled_systems = set()
    protein_to_idx = {p: i for i, p in enumerate(proteins)}
    if ann_df is not None:
        for _, row in ann_df.iterrows():
            pid = row["qseqid"]
            if pid not in protein_to_idx:
                continue
            sys_label = known_system_label(row.get("stitle", ""))
            if sys_label and sys_label not in labeled_systems:
                i = protein_to_idx[pid]
                ax.scatter(embedding[i, 0], embedding[i, 1],
                           c="#444444", s=30, marker="D",
                           edgecolors="white", linewidths=0.5, zorder=4)
                ax.annotate(sys_label,
                            (embedding[i, 0], embedding[i, 1]),
                            fontsize=6.5, fontweight="bold", color="#222222",
                            xytext=(6, 4), textcoords="offset points",
                            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.4))
                labeled_systems.add(sys_label)

    # Gold stars for user working systems — with chimera swap label
    if chimera_df is not None and "working_in_user_system" in chimera_df.columns:
        working = chimera_df[chimera_df["working_in_user_system"] == True]
        for _, row in working.iterrows():
            pid = row["protein_id"]
            if pid not in protein_to_idx:
                continue
            i = protein_to_idx[pid]
            ax.scatter(embedding[i, 0], embedding[i, 1],
                       c=WORKING_COLOR, s=150, marker="*",
                       edgecolors="black", linewidths=0.6, zorder=6)
            lbl = chimera_label(row)
            ax.annotate(lbl,
                        (embedding[i, 0], embedding[i, 1]),
                        fontsize=7, fontweight="bold", color="#1a1a1a",
                        xytext=(7, 5), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color="black", lw=0.5))

    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_umap(hk_embedding, hk_proteins,
              rr_embedding, rr_proteins,
              hk_ann_df, rr_ann_df, chimera_df, outdir: Path):
    """Two-panel UMAP: HK (sensor type) | RR (DBD family)."""

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Left panel: HK coloured by sensor class
    plot_panel(axes[0], hk_embedding, hk_proteins, hk_ann_df, chimera_df,
               ptype="HK", color_by="sensor",
               palette=HK_SENSOR_PALETTE,
               keyword_map=HK_SENSOR_KEYWORDS,
               title=f"Sensor histidine kinases (n={len(hk_proteins):,})\nColoured by sensor architecture")

    # Right panel: RR coloured by DBD family
    plot_panel(axes[1], rr_embedding, rr_proteins, rr_ann_df, chimera_df,
               ptype="RR", color_by="dbd",
               palette=RR_DBD_PALETTE,
               keyword_map=RR_DBD_KEYWORDS,
               title=f"Response regulators (n={len(rr_proteins):,})\nColoured by DBD family")

    # Legends
    hk_legend = [mpatches.Patch(color=c, label=l) for l, c in HK_SENSOR_PALETTE.items()]
    hk_legend += [plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=WORKING_COLOR,
                              markeredgecolor="black", markersize=11,
                              label="Working chimera (confirmed)"),
                  plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#444444",
                              markersize=7, label="Known TCS system")]
    axes[0].legend(handles=hk_legend, fontsize=8, loc="lower left", framealpha=0.9)

    rr_legend = [mpatches.Patch(color=c, label=l) for l, c in RR_DBD_PALETTE.items()]
    rr_legend += [plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=WORKING_COLOR,
                              markeredgecolor="black", markersize=11,
                              label="Working chimera (confirmed)"),
                  plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#444444",
                              markersize=7, label="Known TCS system")]
    axes[1].legend(handles=rr_legend, fontsize=8, loc="lower left", framealpha=0.9, ncol=2)

    fig.suptitle(
        "TCS Sequence Space — Separate UMAP embeddings from MMseqs2 pairwise identity\n"
        "(distance = 1 − pident; labels show chimera swap type and junction position)",
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
    parser.add_argument("--rr_ann",      required=True)
    parser.add_argument("--outdir",      required=True)
    parser.add_argument("--n_neighbors", type=int,   default=15)
    parser.add_argument("--min_dist",    type=float, default=0.1)
    args = parser.parse_args()

    print("Loading homology data...")
    edges = load_homology(args.hk_homology, args.rr_homology)
    if edges.empty:
        print("ERROR: no homology data found", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(edges):,} pairwise alignments")

    print("Building separate HK and RR distance matrices...")
    hk_proteins, hk_mat = build_distance_matrix(edges, ptype_filter="HK")
    rr_proteins, rr_mat = build_distance_matrix(edges, ptype_filter="RR")
    print(f"  HK: {len(hk_proteins):,} proteins | RR: {len(rr_proteins):,} proteins")

    print("Running UMAP (HK)...")
    hk_embedding = run_umap(hk_mat, args.n_neighbors, args.min_dist)
    print("Running UMAP (RR)...")
    rr_embedding = run_umap(rr_mat, args.n_neighbors, args.min_dist)

    ann_cols = ["qseqid", "sseqid", "pident", "length",
                "qcovhsp", "evalue", "bitscore", "stitle"]

    hk_ann_df = None
    if Path(args.hk_ann).exists():
        hk_ann_df = pd.read_csv(args.hk_ann, sep="\t", header=None, names=ann_cols)

    rr_ann_df = None
    if Path(args.rr_ann).exists():
        rr_ann_df = pd.read_csv(args.rr_ann, sep="\t", header=None, names=ann_cols)

    chimera_df = None
    if Path(args.chimera).exists():
        chimera_df = pd.read_csv(args.chimera, sep="\t")

    print("Plotting UMAP...")
    plot_umap(hk_embedding, hk_proteins,
              rr_embedding, rr_proteins,
              hk_ann_df, rr_ann_df, chimera_df, Path(args.outdir))
    print("Done.")


if __name__ == "__main__":
    main()
