#!/usr/bin/env python3
"""Extract HAMP domain + flanking linker regions from HK cluster representatives.

HAMP domains are the signaling helix bundle that couples sensor input to kinase
core output. The linker between the sensor domain C-terminus and HAMP N-terminus
sets the heptad register (Hatstat 2025). We extract a window spanning
[HAMP_start - upstream : HAMP_start + downstream] to capture:
  - The sensor→HAMP linker (variable, sets register)
  - The HAMP domain itself (coiled-coil, structured)
This window is the input for DeepCoil coiled-coil register prediction.
"""

import argparse
import pandas as pd
from pathlib import Path
from Bio import SeqIO


def parse_domtbl(domtbl_path, domain_name="HAMP"):
    """Parse HMMER --domtblout → per-protein best domain hit coordinates.

    domtblout columns (1-indexed):
      1:  target name (protein ID)
      3:  query name (domain/HMM name)
      12: domain score
      16: ali coord from (sequence residue where alignment starts)
      17: ali coord to (sequence residue where alignment ends)
    Returns dict: protein_id -> (ali_from, ali_to)
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
            domain = parts[3]   # query name (Pfam profile name, e.g. "HAMP")
            score = float(parts[13])  # domain score
            ali_from = int(parts[17])  # alignment start on sequence
            ali_to = int(parts[18])    # alignment end on sequence
            if domain == domain_name:
                # Keep highest-scoring hit if a protein has multiple HAMP instances
                if protein not in boundaries or score > boundaries[protein][2]:
                    boundaries[protein] = (ali_from, ali_to, score)
    return {p: (v[0], v[1]) for p, v in boundaries.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hk_reps", required=True, help="HK representative FASTA")
    parser.add_argument("--domtbl", required=True, help="HMMER --domtblout for HK reps")
    parser.add_argument("--output", required=True, help="Output FASTA of HAMP linker windows")
    parser.add_argument("--upstream", type=int, default=30,
                        help="Residues before HAMP start to include")
    parser.add_argument("--downstream", type=int, default=50,
                        help="Residues after HAMP start (into domain) to include")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    hamp_boundaries = parse_domtbl(args.domtbl, domain_name="HAMP")
    print(f"HAMP boundaries found for {len(hamp_boundaries)} proteins")

    written = 0
    skipped_no_hamp = 0
    skipped_too_short = 0

    with open(args.output, "w") as out:
        for rec in SeqIO.parse(args.hk_reps, "fasta"):
            if rec.id not in hamp_boundaries:
                skipped_no_hamp += 1
                continue
            hamp_from, hamp_to = hamp_boundaries[rec.id]
            # Convert 1-indexed HMMER positions to 0-indexed Python slices
            start = max(0, hamp_from - 1 - args.upstream)
            end = min(len(rec.seq), hamp_from - 1 + args.downstream)
            linker_rec = rec[start:end]
            if len(linker_rec.seq) < 20:
                skipped_too_short += 1
                continue
            linker_rec.id = f"{rec.id}_HAMP_{hamp_from}"
            linker_rec.description = (
                f"HAMP_start={hamp_from} extracted_residues={start+1}-{end}"
            )
            SeqIO.write(linker_rec, out, "fasta")
            written += 1

    print(f"Extracted {written} HAMP linker windows")
    print(f"  Skipped (no HAMP): {skipped_no_hamp}")
    print(f"  Skipped (window < 20 aa): {skipped_too_short}")


if __name__ == "__main__":
    main()
