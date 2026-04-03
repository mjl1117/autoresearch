#!/usr/bin/env python3
"""Extract one representative sequence per MMseqs2 cluster.

MMseqs2 clusters.tsv format: representative_id <TAB> member_id
The representative is the first column; we keep one sequence per unique rep.
"""

import argparse
from Bio import SeqIO
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", required=True, help="MMseqs2 clusters.tsv")
    parser.add_argument("--fasta", required=True, help="Input FASTA (HK or RR)")
    parser.add_argument("--output", required=True, help="Output FASTA of representatives")
    args = parser.parse_args()

    # Collect the set of representative IDs (column 1)
    reps = set()
    with open(args.clusters) as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                reps.add(parts[0])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Write exactly one sequence per rep ID (WP_ accessions recur across genomes)
    written = set()
    with open(args.output, "w") as out:
        for rec in SeqIO.parse(args.fasta, "fasta"):
            if rec.id in reps and rec.id not in written:
                SeqIO.write(rec, out, "fasta")
                written.add(rec.id)

    print(f"Wrote {len(written)} cluster representatives to {args.output}")


if __name__ == "__main__":
    main()
