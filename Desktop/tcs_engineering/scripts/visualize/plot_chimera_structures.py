#!/usr/bin/env python3
"""Chimera structural feasibility visualization.

For each top chimera candidate with an AF2 structure, produces:
  1. Domain architecture diagram (linear cartoon) coloured by domain type
     - Sensor domain (grey)
     - HAMP linker (yellow — the swap junction)
     - DHp helix bundle (blue)
     - CA/ATPase domain (green)
     - pLDDT confidence as a continuous band below the cartoon
  2. Phase coherence wheel: heptad register (0–6) radial plot showing
     what fraction of cluster members are in each phase

  3. Summary panel: top-15 candidates ranked by composite score,
     showing linker_phase_compatible, working_in_user_system, cluster_size,
     and pLDDT as a heatmap

Output: results/visualization/chimera_*.png per candidate
        results/visualization/chimera_candidates_heatmap.pdf

Dependencies: matplotlib, numpy, pandas, biopython
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


DOMAIN_COLORS = {
    "sensor":   "#AAAAAA",
    "hamp":     "#F0E442",   # Yellow — the engineering junction
    "dhp":      "#2166AC",   # Blue
    "ca":       "#1A9641",   # Green
    "linker":   "#F0E442",
    "unknown":  "#EEEEEE",
}

PLDDT_CMAP = "RdYlGn"   # Red (low) → Yellow → Green (high pLDDT)


def parse_pdb_plddt(pdb_path: str) -> np.ndarray:
    """Extract pLDDT (B-factor column) from an AF2 PDB."""
    plddt = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                try:
                    plddt.append(float(line[60:66].strip()))
                except ValueError:
                    pass
    return np.array(plddt)


def domain_architecture_panel(ax, row: pd.Series, plddt: np.ndarray):
    """Draw linear domain cartoon with pLDDT confidence track."""
    total_len = int(row.get("chain_length", len(plddt) or 500))
    hamp_start = int(row.get("hamp_start", total_len // 3))
    sensor_end = max(1, hamp_start - 1)
    dhp_start  = hamp_start + 50   # approximate HAMP length
    ca_start   = dhp_start + 60

    domains = [
        ("Sensor",  1,          sensor_end, DOMAIN_COLORS["sensor"]),
        ("HAMP",    hamp_start, dhp_start,  DOMAIN_COLORS["hamp"]),
        ("DHp",     dhp_start,  ca_start,   DOMAIN_COLORS["dhp"]),
        ("CA",      ca_start,   total_len,  DOMAIN_COLORS["ca"]),
    ]

    # Draw domain rectangles
    for label, start, end, color in domains:
        width = (end - start) / total_len
        x = (start - 1) / total_len
        rect = mpatches.FancyBboxPatch(
            (x, 0.35), width, 0.30,
            boxstyle="round,pad=0.02",
            linewidth=0.8, edgecolor="black",
            facecolor=color, alpha=0.9,
        )
        ax.add_patch(rect)
        if width > 0.04:
            ax.text(x + width / 2, 0.50, label,
                    ha="center", va="center", fontsize=7, fontweight="bold",
                    color="black")

    # pLDDT confidence track (below cartoon)
    if len(plddt) > 0:
        xs = np.linspace(0, 1, len(plddt))
        # Smooth with running mean
        window = max(1, len(plddt) // 50)
        smoothed = np.convolve(plddt, np.ones(window) / window, mode="same")
        # Color by pLDDT value
        from matplotlib.collections import LineCollection
        points = np.array([xs, smoothed / 100]).T.reshape(-1, 1, 2)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segs, cmap=PLDDT_CMAP, norm=plt.Normalize(0, 1))
        lc.set_array(smoothed / 100)
        lc.set_linewidth(3)
        ax.add_collection(lc)
        ax.set_ylim(-0.2, 0.9)
        ax.text(0.01, 0.05, "pLDDT", fontsize=6, color="gray", va="center")
    else:
        ax.set_ylim(0, 0.9)

    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels([1, total_len // 4, total_len // 2,
                         3 * total_len // 4, total_len], fontsize=7)
    ax.set_xlabel("Residue position", fontsize=8)
    ax.set_yticks([])
    name = str(row.get("known_tcs_system", row.get("protein_id", "")))
    ax.set_title(name, fontsize=9, fontweight="bold", pad=4)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def phase_wheel_panel(ax, row: pd.Series):
    """Radial plot of heptad phase (0–6) coherence for the candidate's cluster."""
    phase_dom = row.get("cluster_dominant_phase", None)
    coherence = float(row.get("cluster_phase_coherence", 0) or 0)
    candidate_phase = row.get("hamp_phase", None)

    # Construct uniform 1/7 base + coherence spike at dominant phase
    fracs = np.ones(7) / 7
    if pd.notna(phase_dom):
        extra = coherence - 1 / 7
        fracs[int(phase_dom)] += extra

    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 14

    bars = ax.bar(angles, fracs, width=2 * np.pi / 7 * 0.85,
                  bottom=0.0, alpha=0.75,
                  color=["#2166AC" if i == int(phase_dom or -1) else "#AAAAAA"
                         for i in range(7)],
                  edgecolor="white", linewidth=0.5)

    # Mark candidate's own phase
    if pd.notna(candidate_phase):
        cp = int(candidate_phase)
        ax.bar([angles[cp]], [fracs[cp]], width=2 * np.pi / 7 * 0.85,
               bottom=0, color="#F0E442", edgecolor="black", linewidth=1.0,
               alpha=1.0, label="Candidate phase")

    ax.set_xticks(angles)
    ax.set_xticklabels([str(i) for i in range(7)], fontsize=7)
    ax.set_yticks([])
    ax.set_title("Heptad\nregister (0–6)", fontsize=8, pad=6)
    compat = row.get("linker_phase_compatible", None)
    color = "#1A9641" if compat is True else ("#D7191C" if compat is False else "#AAAAAA")
    ax.text(0, 0, f"{'✓' if compat else '?'}\ncoherence\n{coherence:.2f}",
            ha="center", va="center", fontsize=8, color=color, fontweight="bold")


def candidate_heatmap(candidates: pd.DataFrame, plddt_df: pd.DataFrame, outdir: Path):
    """Summary heatmap of top 15 candidates."""
    top = candidates.head(15).copy()

    # Merge pLDDT
    if plddt_df is not None and "protein_id" in plddt_df.columns:
        top = top.merge(
            plddt_df[["protein_id", "plddt_hamp_mean"]].rename(
                columns={"plddt_hamp_mean": "hamp_plddt"}),
            on="protein_id", how="left"
        )
    else:
        top["hamp_plddt"] = np.nan

    cols_to_show = {
        "cluster_size":            "Cluster size",
        "pident":                  "Swiss-Prot\n% identity",
        "cluster_phase_coherence": "Phase\ncoherence",
        "hamp_plddt":              "HAMP pLDDT",
    }
    present = {k: v for k, v in cols_to_show.items() if k in top.columns}

    matrix = top[[k for k in present]].values.astype(float)
    # Normalise each column to 0–1 for heatmap display
    vmin = np.nanmin(matrix, axis=0)
    vmax = np.nanmax(matrix, axis=0)
    norm_matrix = (matrix - vmin) / np.where(vmax - vmin > 0, vmax - vmin, 1)

    fig, (ax_heat, ax_flags) = plt.subplots(
        1, 2, figsize=(14, 8),
        gridspec_kw={"width_ratios": [3, 1]}
    )

    im = ax_heat.imshow(norm_matrix, cmap="viridis", aspect="auto",
                        vmin=0, vmax=1)
    ax_heat.set_xticks(range(len(present)))
    ax_heat.set_xticklabels(list(present.values()), rotation=30, ha="right", fontsize=9)
    ylabels = []
    for _, row in top.iterrows():
        label = str(row.get("protein_id", ""))
        system = str(row.get("known_tcs_system", ""))
        if system and system != "nan":
            label = f"{system} ({label[:12]})"
        ylabels.append(label)
    ax_heat.set_yticks(range(len(top)))
    ax_heat.set_yticklabels(ylabels, fontsize=8)

    # Annotate cells with actual values
    for i in range(len(top)):
        for j, col in enumerate(present):
            val = matrix[i, j]
            if not np.isnan(val):
                txt = f"{val:.0f}" if col == "cluster_size" else f"{val:.2f}"
                ax_heat.text(j, i, txt, ha="center", va="center",
                             fontsize=7, color="white" if norm_matrix[i, j] > 0.5 else "black")

    plt.colorbar(im, ax=ax_heat, fraction=0.04, label="Normalised value")
    ax_heat.set_title("Top chimera candidates — key metrics", fontsize=11)

    # Boolean flags panel
    flag_cols = {
        "linker_phase_compatible": "Phase\ncompatible",
        "working_in_user_system":  "Working in\nuser system",
    }
    flag_present = {k: v for k, v in flag_cols.items() if k in top.columns}
    if flag_present:
        flag_matrix = np.zeros((len(top), len(flag_present)))
        for j, col in enumerate(flag_present):
            flag_matrix[:, j] = top[col].map(
                {True: 1.0, False: 0.0, "yes": 1.0, "no": 0.0}
            ).fillna(0.5).values

        ax_flags.imshow(flag_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax_flags.set_xticks(range(len(flag_present)))
        ax_flags.set_xticklabels(list(flag_present.values()),
                                  rotation=30, ha="right", fontsize=9)
        ax_flags.set_yticks([])
        ax_flags.set_title("Boolean flags", fontsize=11)

        for i in range(len(top)):
            for j, col in enumerate(flag_present):
                v = flag_matrix[i, j]
                sym = "✓" if v == 1 else ("✗" if v == 0 else "?")
                ax_flags.text(j, i, sym, ha="center", va="center",
                               fontsize=10, fontweight="bold",
                               color="white" if v > 0.5 else "black")

    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = outdir / f"chimera_candidates_heatmap.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chimera",    required=True)
    parser.add_argument("--af2_dir",    required=True, help="results/alphafold/")
    parser.add_argument("--plddt_tsv",  required=True, help="results/deepcoil/af2_plddt_analysis.tsv")
    parser.add_argument("--outdir",     required=True)
    parser.add_argument("--top_n",      type=int, default=8)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not Path(args.chimera).exists():
        print(f"ERROR: {args.chimera} not found", file=sys.stderr)
        sys.exit(1)

    candidates = pd.read_csv(args.chimera, sep="\t")
    plddt_df = pd.read_csv(args.plddt_tsv, sep="\t") if Path(args.plddt_tsv).exists() else None

    # Candidate heatmap (always)
    candidate_heatmap(candidates, plddt_df, outdir)

    # Per-candidate structure panels (top N)
    top = candidates.head(args.top_n)
    for _, row in top.iterrows():
        pid = str(row.get("protein_id", ""))
        # Find AF2 PDB
        uniprot = str(row.get("best_hit", "")).split("|")[1] if "|" in str(row.get("best_hit", "")) else ""
        pdb_path = Path(args.af2_dir) / f"{uniprot}.pdb"
        plddt = parse_pdb_plddt(str(pdb_path)) if pdb_path.exists() else np.array([])

        fig = plt.figure(figsize=(14, 4))
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3, 1])
        ax_arch  = fig.add_subplot(gs[0])
        ax_wheel = fig.add_subplot(gs[1], polar=True)

        domain_architecture_panel(ax_arch, row, plddt)
        phase_wheel_panel(ax_wheel, row)

        system = str(row.get("known_tcs_system", pid))
        fig.suptitle(
            f"Chimera candidate: {system}  |  {row.get('chimera_type', '')}  "
            f"|  cluster_size={int(row.get('cluster_size', 0) or 0)}",
            fontsize=10, fontweight="bold"
        )
        plt.tight_layout()

        safe_name = system.replace("/", "_").replace(" ", "_")[:40]
        for ext in ("pdf", "png"):
            path = outdir / f"chimera_{safe_name}.{ext}"
            fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {outdir}/chimera_{safe_name}.*")

    print("Done.")


if __name__ == "__main__":
    main()
