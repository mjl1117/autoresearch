#!/usr/bin/env python3
"""Subsample promoter sequences for MEME using RR cluster representatives.

Strategy: for each RR cluster representative, find and keep its promoter
sequence (if extracted). This gives ≤871 sequences covering the full
diversity of response regulator regulatory contexts without redundancy.
Falls back to random sampling if representative IDs don't match.
"""

import argparse
import random
from Bio import SeqIO
from pathlib import Path


def load_rep_ids(rep_fasta):
    return {rec.id for rec in SeqIO.parse(rep_fasta, "fasta")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--promoters", nargs="+", required=True)
    parser.add_argument("--rr_reps", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_seqs", type=int, default=600)
    args = parser.parse_args()

    rep_ids = load_rep_ids(args.rr_reps)

    # First pass: collect promoters whose protein ID matches an RR rep
    matched = []
    all_seqs = []
    for fasta_path in args.promoters:
        for rec in SeqIO.parse(fasta_path, "fasta"):
            all_seqs.append(rec)
            # Header format: >promoter_N_PROTEIN_ID
            protein_id = rec.id.split("_", 2)[-1] if rec.id.count("_") >= 2 else ""
            if protein_id in rep_ids:
                matched.append(rec)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if len(matched) >= 10:
        chosen = matched[:args.max_seqs]
        print(f"Selected {len(chosen)} rep-matched promoters from {len(matched)} candidates")
    else:
        # Fallback: random subsample
        random.seed(42)
        chosen = random.sample(all_seqs, min(args.max_seqs, len(all_seqs)))
        print(f"Fallback: randomly selected {len(chosen)} promoters from {len(all_seqs)}")

    with open(args.output, "w") as out:
        SeqIO.write(chosen, out, "fasta")


if __name__ == "__main__":
    main()
