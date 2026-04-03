#!/usr/bin/env python3
"""Extract per-residue pLDDT scores for HAMP regions from AlphaFold2 PDB files.

AlphaFold2 stores per-residue confidence (pLDDT) in the B-factor column of PDB
ATOM records. For chimera design, we care about:
  - pLDDT > 70 in HAMP helix  → reliable structure for register inference
  - pLDDT < 50 in linker       → disordered connection (design risk flag)

Outputs a per-protein TSV with plddt_hamp_mean, plddt_linker_mean, and
hamp_high_confidence. If no AF2 structures are available, writes an empty
table (pipeline continues; visualizations will skip pLDDT track).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_domtbl_hamp(domtbl_path):
    """Parse HMMER domtblout → {protein_id: (hamp_start, hamp_end)}.

    domtblout columns (0-indexed):
      0:  target name (protein ID)
      3:  query name (Pfam profile name, e.g. "HAMP")
      13: domain score
      17: ali_from (sequence coordinate, 1-indexed)
      18: ali_to   (sequence coordinate, 1-indexed)
    """
    boundaries = {}
    with open(domtbl_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 23:
                continue
            protein = parts[0]
            domain = parts[3]
            score = float(parts[13])
            ali_from = int(parts[17])
            ali_to = int(parts[18])
            if domain in ("HAMP", "HAMP_2", "CovS-like_HAMP"):
                if protein not in boundaries or score > boundaries[protein][2]:
                    boundaries[protein] = (ali_from, ali_to, score)
    return {p: (v[0], v[1]) for p, v in boundaries.items()}


def extract_plddt_from_pdb(pdb_path, hamp_start, hamp_end, linker_upstream=30):
    """Read B-factor (pLDDT) per residue from an AF2 PDB file.

    Returns (plddt_hamp_mean, plddt_linker_mean) or (None, None) on failure.
    Only reads CA atoms to get one value per residue.
    """
    residue_plddt = {}  # {residue_number: plddt}
    try:
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue
                try:
                    resnum = int(line[22:26].strip())
                    bfactor = float(line[60:66].strip())
                    residue_plddt[resnum] = bfactor
                except (ValueError, IndexError):
                    continue
    except OSError:
        return None, None

    if not residue_plddt:
        return None, None

    # HAMP domain residues
    hamp_vals = [residue_plddt[r] for r in range(hamp_start, hamp_end + 1)
                 if r in residue_plddt]

    # Pre-HAMP linker residues
    linker_start = max(1, hamp_start - linker_upstream)
    linker_vals = [residue_plddt[r] for r in range(linker_start, hamp_start)
                   if r in residue_plddt]

    plddt_hamp = sum(hamp_vals) / len(hamp_vals) if hamp_vals else None
    plddt_linker = sum(linker_vals) / len(linker_vals) if linker_vals else None
    return plddt_hamp, plddt_linker


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--af2_manifest", required=True)
    parser.add_argument("--domtbl",       required=True)
    parser.add_argument("--output",       required=True)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.af2_manifest, sep="\t")

    hamp_bounds = parse_domtbl_hamp(args.domtbl)
    print(f"HAMP boundaries loaded for {len(hamp_bounds)} proteins")

    rows = []
    n_with_pdb = manifest["af2_pdb"].notna().sum() if "af2_pdb" in manifest.columns else 0
    print(f"Manifest entries: {len(manifest)}, with PDB: {n_with_pdb}")

    for _, row in manifest.iterrows():
        protein_id = row["protein_id"]
        uniprot_id = row.get("uniprot_id", "")
        pdb_path = row.get("af2_pdb", None)

        hamp_start, hamp_end = hamp_bounds.get(protein_id, (None, None))

        entry = {
            "protein_id":          protein_id,
            "uniprot_id":          uniprot_id,
            "hamp_start":          hamp_start,
            "hamp_end":            hamp_end,
            "plddt_hamp_mean":     None,
            "plddt_linker_mean":   None,
            "hamp_high_confidence": False,
        }

        if pdb_path and Path(str(pdb_path)).exists() and hamp_start is not None:
            plddt_hamp, plddt_linker = extract_plddt_from_pdb(
                pdb_path, hamp_start, hamp_end
            )
            entry["plddt_hamp_mean"] = round(plddt_hamp, 2) if plddt_hamp is not None else None
            entry["plddt_linker_mean"] = round(plddt_linker, 2) if plddt_linker is not None else None
            entry["hamp_high_confidence"] = bool(plddt_hamp is not None and plddt_hamp > 70)

        rows.append(entry)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, sep="\t", index=False)

    n_conf = out_df["hamp_high_confidence"].sum()
    print(f"Written: {args.output}")
    print(f"  {n_conf} proteins with high-confidence HAMP (pLDDT > 70)")
    if n_with_pdb == 0:
        print("  NOTE: No AF2 PDB files available. Run download_alphafold.py to fetch structures.")


if __name__ == "__main__":
    main()
