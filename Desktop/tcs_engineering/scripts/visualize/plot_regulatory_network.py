#!/usr/bin/env python3
"""TCS regulatory network visualization.

Builds a bipartite graph of HK–RR cognate pairs detected by operon analysis
(≤500 bp on same strand) and annotated by DIAMOND. Highlights:
  - Confirmed working chimera pairs (gold star border)
  - Known TCS systems (named nodes)
  - Edge weight: pident of HK→RR annotation match
  - Node size: cluster_size (representation across dataset)
  - Node color: signal type from well_characterized_tcs.tsv

Output: results/visualization/tcs_regulatory_network.pdf + .png

Dependencies: networkx, matplotlib, pandas
"""

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd


SIGNAL_COLORS = {
    "Nitrate":         "#2166AC",
    "Phosphate":       "#4DAC26",
    "Osmolarity":      "#762A83",
    "Mg2+":            "#E66101",
    "Chemotaxis":      "#D7191C",
    "Oxygen":          "#ABDDA4",
    "Light":           "#F0E442",
    "Quorum sensing":  "#D6604D",
    "Redox":           "#74ADD1",
    "Nitrogen":        "#A6D96A",
    "unknown":         "#CCCCCC",
}

WORKING_SYSTEMS = {"NarXL", "PhoRB"}


def load_operons(operon_dir: str) -> pd.DataFrame:
    """Load all per-genome operon TSVs into one DataFrame."""
    frames = []
    for path in glob.glob(f"{operon_dir}/*.tsv"):
        try:
            df = pd.read_csv(path, sep="\t")
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_graph(operons: pd.DataFrame,
                hk_ann: pd.DataFrame,
                rr_ann: pd.DataFrame,
                chimera_df: pd.DataFrame,
                ref_tcs: pd.DataFrame) -> nx.DiGraph:
    """Build directed HK → RR graph from operon data."""
    G = nx.DiGraph()

    # HK annotation map: qseqid → (system_name, signal)
    hk_system = {}
    if hk_ann is not None:
        for _, row in hk_ann.iterrows():
            hk_system[row["qseqid"]] = row.get("stitle", "")

    # RR annotation map
    rr_system = {}
    if rr_ann is not None:
        for _, row in rr_ann.iterrows():
            rr_system[row["qseqid"]] = row.get("stitle", "")

    # Reference signal lookup
    signal_map = {}
    if ref_tcs is not None:
        for _, row in ref_tcs.iterrows():
            for gene in [row.get("hk_gene", ""), row.get("rr_gene", "")]:
                if gene:
                    signal_map[gene.lower()] = row.get("signal", "unknown")

    # Working system set
    working_pairs = set()
    if chimera_df is not None and "known_tcs_system" in chimera_df.columns:
        working_pairs = set(chimera_df[chimera_df["working_in_user_system"] == True]
                             ["known_tcs_system"].dropna().unique())

    # Required operon columns
    hk_col = next((c for c in operons.columns if "hk" in c.lower() and "id" in c.lower()), None)
    rr_col = next((c for c in operons.columns if "rr" in c.lower() and "id" in c.lower()), None)
    if hk_col is None or rr_col is None:
        # Try positional columns
        cols = list(operons.columns)
        if len(cols) >= 2:
            hk_col, rr_col = cols[0], cols[1]
        else:
            return G

    # Sample to top 200 operon pairs for readability
    sample = operons.dropna(subset=[hk_col, rr_col]).drop_duplicates(
        subset=[hk_col, rr_col]
    ).head(200)

    cluster_sizes = {}
    if chimera_df is not None and "cluster_size" in chimera_df.columns:
        cluster_sizes = chimera_df.set_index("protein_id")["cluster_size"].to_dict()

    for _, row in sample.iterrows():
        hk_id = str(row[hk_col])
        rr_id = str(row[rr_col])

        hk_title = hk_system.get(hk_id, "")
        rr_title = rr_system.get(rr_id, "")

        # Resolve signal colour
        signal = "unknown"
        for gene_hint, sig in signal_map.items():
            if gene_hint in hk_title.lower() or gene_hint in rr_title.lower():
                signal = sig
                break

        G.add_node(hk_id, ptype="HK", signal=signal, size=cluster_sizes.get(hk_id, 5))
        G.add_node(rr_id, ptype="RR", signal=signal, size=cluster_sizes.get(rr_id, 5))
        G.add_edge(hk_id, rr_id)

    return G


def draw_network(G: nx.DiGraph, chimera_df: pd.DataFrame, outdir: Path):
    """Draw bipartite TCS regulatory network."""
    if len(G.nodes) == 0:
        print("  WARNING: empty graph — no operon pairs loaded")
        return

    # Layout: HKs on left, RRs on right
    hk_nodes = [n for n, d in G.nodes(data=True) if d.get("ptype") == "HK"]
    rr_nodes = [n for n, d in G.nodes(data=True) if d.get("ptype") == "RR"]

    pos = {}
    for i, n in enumerate(hk_nodes):
        pos[n] = (-1, i / max(len(hk_nodes), 1))
    for i, n in enumerate(rr_nodes):
        pos[n] = (1, i / max(len(rr_nodes), 1))

    fig, ax = plt.subplots(figsize=(16, max(10, len(G.nodes) * 0.3)))

    # Node colors from signal
    hk_colors = [SIGNAL_COLORS.get(G.nodes[n].get("signal", "unknown"), "#CCCCCC")
                 for n in hk_nodes]
    rr_colors = [SIGNAL_COLORS.get(G.nodes[n].get("signal", "unknown"), "#CCCCCC")
                 for n in rr_nodes]

    node_sizes = [max(50, np.sqrt(G.nodes[n].get("size", 5)) * 20)
                  for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, nodelist=hk_nodes, node_color=hk_colors,
                           node_size=[max(50, np.sqrt(G.nodes[n].get("size", 5)) * 20)
                                      for n in hk_nodes],
                           ax=ax, alpha=0.85, node_shape="s")  # squares for HK
    nx.draw_networkx_nodes(G, pos, nodelist=rr_nodes, node_color=rr_colors,
                           node_size=[max(50, np.sqrt(G.nodes[n].get("size", 5)) * 20)
                                      for n in rr_nodes],
                           ax=ax, alpha=0.85, node_shape="o")  # circles for RR

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.5,
                           edge_color="#888888",
                           arrows=True, arrowsize=10,
                           connectionstyle="arc3,rad=0.05")

    # Labels only for known systems
    working_ids = set()
    if chimera_df is not None and "protein_id" in chimera_df.columns:
        working_ids = set(chimera_df[chimera_df["working_in_user_system"] == True]
                           ["protein_id"].dropna().unique())

    labels = {n: n[:15] for n in G.nodes if n in working_ids}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7,
                            font_color="black", font_weight="bold")

    # Legend
    signal_legend = [
        mpatches.Patch(color=c, label=s)
        for s, c in SIGNAL_COLORS.items()
        if s != "unknown" and any(G.nodes[n].get("signal") == s for n in G.nodes)
    ]
    type_legend = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#555555",
                   markersize=10, label="Histidine kinase (HK)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
                   markersize=10, label="Response regulator (RR)"),
    ]
    ax.legend(handles=signal_legend + type_legend, fontsize=8,
              loc="upper right", framealpha=0.9, ncol=2)

    ax.set_title(
        f"TCS Regulatory Network — cognate HK→RR pairs (operons)\n"
        f"({len(hk_nodes)} HKs, {len(rr_nodes)} RRs, {len(G.edges)} pairs; "
        f"colour = signal type; size = cluster size)",
        fontsize=10
    )
    ax.axis("off")

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = outdir / f"tcs_regulatory_network.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--operon_dir",   required=True)
    parser.add_argument("--hk_ann",       required=True)
    parser.add_argument("--rr_ann",       required=True)
    parser.add_argument("--chimera",      required=True)
    parser.add_argument("--reference_tcs",required=True)
    parser.add_argument("--outdir",       required=True)
    args = parser.parse_args()

    print("Loading operons...")
    operons = load_operons(args.operon_dir)
    print(f"  {len(operons):,} operon pairs loaded")

    ann_cols = ["qseqid", "sseqid", "pident", "length", "qcovhsp", "evalue", "bitscore", "stitle"]
    hk_ann = pd.read_csv(args.hk_ann, sep="\t", header=None, names=ann_cols) \
        if Path(args.hk_ann).exists() else None
    rr_ann = pd.read_csv(args.rr_ann, sep="\t", header=None, names=ann_cols) \
        if Path(args.rr_ann).exists() else None
    chimera_df = pd.read_csv(args.chimera, sep="\t") \
        if Path(args.chimera).exists() else None
    ref_tcs = pd.read_csv(args.reference_tcs, sep="\t") \
        if Path(args.reference_tcs).exists() else None

    print("Building graph...")
    G = build_graph(operons, hk_ann, rr_ann, chimera_df, ref_tcs)
    print(f"  {len(G.nodes)} nodes, {len(G.edges)} edges")

    print("Drawing network...")
    draw_network(G, chimera_df, Path(args.outdir))
    print("Done.")


if __name__ == "__main__":
    main()
