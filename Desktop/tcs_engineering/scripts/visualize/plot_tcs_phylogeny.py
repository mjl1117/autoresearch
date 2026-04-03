#!/usr/bin/env python3
"""Publication-quality TCS phylogenetic tree visualization.

Reads the FastTree ML tree of TCS cluster representatives and annotates with:
  - Tip color: HK (blue) vs RR (orange)
  - Tip shape: DBD family (OmpR_PhoB, NarL_FixJ, NtrC, CheY, other)
  - Branch support shading (SH-like values from FastTree)
  - Labeled highlights for user confirmed working systems (NarXL, PhoRB)
  - Clade rectangles for major TCS superfamilies

Output: results/visualization/tcs_phylogeny.pdf + .png

Dependencies: matplotlib, biopython
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from Bio import Phylo
from io import StringIO


# ─── Visual constants ────────────────────────────────────────────────────────

HK_COLOR  = "#2166AC"   # Blue  — IPCC / colorblind-safe
RR_COLOR  = "#D6604D"   # Red-orange
EDGE_GRAY = "#AAAAAA"

DBD_MARKERS = {
    "OmpR_PhoB": "o",
    "NarL_FixJ": "s",
    "NtrC_AAA":  "^",
    "CheY":      "D",
    "Spo0A":     "P",
    "other":     ".",
}

DBD_COLORS = {
    "OmpR_PhoB": "#1A9641",
    "NarL_FixJ": "#A6D96A",
    "NtrC_AAA":  "#FDAE61",
    "CheY":      "#D7191C",
    "Spo0A":     "#762A83",
    "other":     "#CCCCCC",
}

# Confirmed working chimera systems to label on tree
HIGHLIGHT_SYSTEMS = {"NarXL", "PhoRB", "NarX", "NarL", "PhoR", "PhoB", "EnvZ", "OmpR"}


def classify_tip(name: str, annotation_df) -> tuple:
    """Return (protein_type, dbd_family) for a tip label."""
    ptype = "HK" if name in annotation_df.get("hk_ids", set()) else "RR"
    dbd = "other"
    if annotation_df is not None and "dbd" in annotation_df:
        row = annotation_df["dbd"].get(name, "")
        for fam in DBD_MARKERS:
            if fam.lower() in row.lower():
                dbd = fam
                break
    return ptype, dbd


def draw_tree(tree_path: str, annotation: dict, outdir: Path, max_tips: int = 500):
    """Draw cladogram with tip annotations."""
    tree = Phylo.read(tree_path, "newick")
    terminals = tree.get_terminals()

    if len(terminals) > max_tips:
        print(f"  Tree has {len(terminals)} tips — sampling {max_tips} for readability")
        # Keep tips that match known systems; sample rest
        keep = [t for t in terminals if any(h.lower() in t.name.lower() for h in HIGHLIGHT_SYSTEMS)]
        rest = [t for t in terminals if t not in keep]
        rng = np.random.default_rng(42)
        n_sample = min(max_tips - len(keep), len(rest))
        idx = rng.choice(len(rest), n_sample, replace=False)
        sampled = [rest[i] for i in idx]
        keep_names = {t.name for t in keep + sampled}
        for t in [t for t in terminals if t.name not in keep_names]:
            tree.prune(t)
        terminals = tree.get_terminals()

    n_tips = len(terminals)
    fig_height = max(6, n_tips * 0.12)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    Phylo.draw(tree, axes=ax, do_show=False,
               label_func=lambda x: "",
               branch_labels=lambda c: "",
               show_confidence=False)

    # Overlay colored tip points
    tip_y = {t.name: i + 1 for i, t in enumerate(terminals)}
    hk_ids  = annotation.get("hk_ids", set())
    dbd_map = annotation.get("dbd_map", {})

    for tip in terminals:
        name = tip.name
        ptype = "HK" if name in hk_ids else "RR"
        color = HK_COLOR if ptype == "HK" else RR_COLOR
        dbd   = dbd_map.get(name, "other")
        marker = DBD_MARKERS.get(dbd, ".")

        # Get x position from drawn line
        xmax = ax.get_xlim()[1]
        y    = tip_y[name]
        ax.plot(xmax * 1.001, y, marker=marker, color=color,
                markersize=5, alpha=0.85, linewidth=0, clip_on=False)

        # Label highlighted systems
        if any(h.lower() in name.lower() for h in HIGHLIGHT_SYSTEMS):
            short = name[:25]
            ax.text(xmax * 1.01, y, f"  {short}", fontsize=5,
                    va="center", color=color, fontweight="bold")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=HK_COLOR,  label="Histidine kinase (HK)"),
        mpatches.Patch(facecolor=RR_COLOR,  label="Response regulator (RR)"),
    ] + [
        plt.Line2D([0], [0], marker=DBD_MARKERS[d], color="k",
                   label=d, markersize=6, linewidth=0)
        for d in DBD_MARKERS if d != "other"
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=7,
              framealpha=0.9, ncol=2)

    ax.set_xlabel("Substitutions per site", fontsize=9)
    ax.set_title("TCS Phylogeny — cluster representatives\n"
                 "(coloured by type; shape by DBD family)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = outdir / f"tcs_phylogeny.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


_RR_DBD_KEYWORDS = {
    "Spo0A":     ["spo0a", "sporulation", "stage 0"],
    "CheY":      ["chey ", "chemotaxis response regulator", " chey"],
    "NtrC_AAA":  ["ntrc", "nifa", "luxo", "sigma-54", "aaa+", "enhancer-binding",
                  "ntr regulator", "nitrogen regulatory protein"],
    "NarL_FixJ": ["narl", "narp", "fixj", "gera", "coma", "flhd",
                  "transcriptional regulatory protein narl",
                  "probable transcriptional regulatory protein narl"],
    "OmpR_PhoB": ["ompr", "phob", "phop", "rsca", "cpxr", "baer", "kdpe",
                  "rsta", "evga", "tcrr", "rega", "resd", "degu", "ctra",
                  "vpst", "gaca", "nrec", "rcsb", "uvry", "arca", "torr"],
}


def _classify_dbd(stitle: str) -> str:
    t = str(stitle).lower()
    for fam, keywords in _RR_DBD_KEYWORDS.items():
        if any(k in t for k in keywords):
            return fam
    return "other"


def build_annotation(chimera_tsv: str, hk_ann: str, rr_ann: str) -> dict:
    """Build annotation dict from pipeline outputs."""
    import pandas as pd

    hk_ids, dbd_map = set(), {}
    ann_cols = ["qseqid", "sseqid", "pident", "length",
                "qcovhsp", "evalue", "bitscore", "stitle"]

    if Path(hk_ann).exists():
        df = pd.read_csv(hk_ann, sep="\t", header=None, names=ann_cols)
        hk_ids = set(df["qseqid"].unique())

    # Build DBD map from RR annotation stitle (covers all RR proteins)
    if Path(rr_ann).exists():
        df = pd.read_csv(rr_ann, sep="\t", header=None, names=ann_cols)
        for _, row in df.iterrows():
            dbd_map[row["qseqid"]] = _classify_dbd(row["stitle"])

    # Chimera candidates override with curated dbd_family where available
    if Path(chimera_tsv).exists():
        df = pd.read_csv(chimera_tsv, sep="\t")
        if "dbd_family" in df.columns and "protein_id" in df.columns:
            for pid, fam in df.set_index("protein_id")["dbd_family"].items():
                if pd.notna(fam) and fam != "N/A":
                    dbd_map[pid] = fam

    return {"hk_ids": hk_ids, "dbd_map": dbd_map}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tree",       required=True, help="results/phylogeny/tcs_tree.treefile")
    parser.add_argument("--hk_ann",     required=True, help="results/annotation/hk_annotation.tsv")
    parser.add_argument("--rr_ann",     required=True, help="results/annotation/rr_annotation.tsv")
    parser.add_argument("--chimera",    required=True, help="results/chimera_targets/chimera_candidates.tsv")
    parser.add_argument("--outdir",     required=True, help="results/visualization/")
    parser.add_argument("--max_tips",   type=int, default=500)
    args = parser.parse_args()

    annotation = build_annotation(args.chimera, args.hk_ann, args.rr_ann)
    draw_tree(args.tree, annotation, Path(args.outdir), args.max_tips)


if __name__ == "__main__":
    main()
