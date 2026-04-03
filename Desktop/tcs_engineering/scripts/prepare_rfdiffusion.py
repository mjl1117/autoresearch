#!/usr/bin/env python3
"""Prepare RFDiffusion linker design inputs for top chimera candidates.

For each HK_sensor_swap candidate:
  1. Look up its AF2 PDB structure
  2. Get HAMP domain boundaries from domtblout
  3. Estimate sensor domain end position (last non-kinase domain before HAMP)
  4. Write RFDiffusion contig specification:
       A1-{sensor_end}/5-20/A{hamp_start}-{total_length}
       (fix sensor → design 5-20 linker residues → fix HAMP onward)

The contig syntax fixes the flanking domains and diffuses new linker residues.
Linker length range 5-20 encompasses the observed natural range, but per
Hatstat 2025, the designed linker length must be congruent mod 7 with the
original — so post-design filtering keeps only in-phase designs.

Requires: AF2 PDB structures (results/alphafold/) and domtblout
  (results/domains/hk_reps_domtbl.txt) to be available.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


# Domains that mark the sensor / pre-kinase region (N-terminal to HAMP)
SENSOR_DOMAINS = {
    "PAS", "PAS_4", "PAS_3", "PAS_9", "GAF", "CHASE", "CHASE2", "CHASE3",
    "Cache_1", "Cache_2", "Cache_3", "MASE1", "MASE2", "PDC", "7tm_2",
    "TM_helix", "CSS-motif",
}

# Domains that mark the kinase core (C-terminal, should be fixed)
KINASE_DOMAINS = {"HAMP", "HisKA", "HisKA_3", "HisK_5", "HATPase_c"}


def parse_domtbl_full(domtbl_path):
    """Parse HMMER --domtblout → per-protein list of (domain, ali_from, ali_to, score)."""
    hits = {}
    with open(domtbl_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 17:
                continue
            protein = parts[0]
            domain = parts[2]
            score = float(parts[11])
            ali_from = int(parts[15])
            ali_to = int(parts[16])
            hits.setdefault(protein, []).append((domain, ali_from, ali_to, score))
    return hits


def get_sensor_end_hamp_start(domain_hits):
    """Return (sensor_end, hamp_start) for a protein.

    sensor_end: ali_to of the last sensor-region domain before HAMP
    hamp_start: ali_from of the highest-scoring HAMP hit
    """
    hamp_hits = [(d, f, t, s) for d, f, t, s in domain_hits if d == "HAMP"]
    if not hamp_hits:
        return None, None
    hamp_start = min(f for _, f, _, _ in hamp_hits)

    # Sensor end: last domain that ends before HAMP starts
    pre_hamp = [(d, f, t, s) for d, f, t, s in domain_hits if t < hamp_start]
    if pre_hamp:
        sensor_end = max(t for _, f, t, _ in pre_hamp)
    else:
        # No detected sensor domain — estimate as hamp_start - 10
        sensor_end = max(1, hamp_start - 10)
    return sensor_end, hamp_start


def get_pdb_chain_length(pdb_path):
    """Return the number of residues in chain A of a PDB file."""
    max_res = 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[21] == "A":
                try:
                    resnum = int(line[22:26].strip())
                    max_res = max(max_res, resnum)
                except ValueError:
                    pass
    return max_res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True,
                        help="chimera_candidates.tsv from identify_chimera_targets.py")
    parser.add_argument("--af2_manifest", required=True,
                        help="af2_manifest.tsv from download_alphafold.py")
    parser.add_argument("--domtbl", required=True,
                        help="results/domains/hk_reps_domtbl.txt")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n_candidates", type=int, default=10,
                        help="Number of top HK_sensor_swap candidates to prepare")
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.candidates, sep="\t")
    af2 = pd.read_csv(args.af2_manifest, sep="\t")
    domain_hits = parse_domtbl_full(args.domtbl)

    # Focus on HK_sensor_swap with AF2 structures and known HAMP boundaries
    hk_candidates = (
        candidates[candidates["chimera_type"] == "HK_sensor_swap"]
        .sort_values("cluster_size", ascending=False)
        .merge(af2[["protein_id", "uniprot_id", "af2_pdb"]],
               on="protein_id", how="left")
        .dropna(subset=["af2_pdb"])
        .head(args.n_candidates)
    )

    rows = []
    for _, row in hk_candidates.iterrows():
        pid = row["protein_id"]
        pdb_path = row["af2_pdb"]

        hits = domain_hits.get(pid, [])
        sensor_end, hamp_start = get_sensor_end_hamp_start(hits)
        if hamp_start is None:
            print(f"  Skip {pid}: no HAMP domain in domtbl")
            continue

        chain_len = get_pdb_chain_length(pdb_path)
        if chain_len == 0:
            print(f"  Skip {pid}: PDB chain A empty or unreadable")
            continue

        # RFDiffusion contig: fix sensor → design linker (5-20 aa) → fix HAMP+kinase
        # Linker range chosen to span natural range; post-filter by mod-7 phase
        linker_min = max(5, sensor_end - hamp_start + 3)
        linker_max = min(30, sensor_end - hamp_start + 15)
        contig = f"A1-{sensor_end}/{linker_min}-{linker_max}/A{hamp_start}-{chain_len}"

        rows.append({
            "protein_id": pid,
            "uniprot_id": row["uniprot_id"],
            "af2_pdb": pdb_path,
            "sensor_end": sensor_end,
            "hamp_start": hamp_start,
            "chain_length": chain_len,
            "linker_length_original": hamp_start - sensor_end,
            "hamp_phase": hamp_start % 7,
            "linker_phase_compatible": row.get("linker_phase_compatible", None),
            "cluster_size": row["cluster_size"],
            "rfdiffusion_contig": contig,
        })

    out_df = pd.DataFrame(rows)
    out_path = Path(args.outdir) / "candidates_for_design.tsv"
    out_df.to_csv(out_path, sep="\t", index=False)

    print(f"\nRFDiffusion inputs prepared: {len(out_df)} candidates")
    print(f"  Output: {out_path}")
    if not out_df.empty:
        print(out_df[["protein_id", "sensor_end", "hamp_start",
                       "linker_length_original", "rfdiffusion_contig"]].to_string(index=False))


if __name__ == "__main__":
    main()
