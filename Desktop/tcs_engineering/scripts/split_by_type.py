#!/usr/bin/env python3
"""Split a merged TCS FASTA into separate HK and RR FASTA files.

Reads per-genome classification CSVs produced by 01_detect_domains.py to
build a protein-type lookup, then partitions the merged sequences accordingly.
"""

import argparse
import pandas as pd
from Bio import SeqIO
from pathlib import Path


def build_type_map(csv_paths):
    frames = []
    for p in csv_paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            pass  # empty or missing CSV for a genome with no TCS hits
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True)
    return dict(zip(combined["protein"], combined["type"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--faa", required=True)
    parser.add_argument("--csvs", nargs="+", required=True)
    parser.add_argument("--hk_out", required=True)
    parser.add_argument("--rr_out", required=True)
    args = parser.parse_args()

    type_map = build_type_map(args.csvs)

    Path(args.hk_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.rr_out).parent.mkdir(parents=True, exist_ok=True)

    hk_count = rr_count = 0
    with open(args.hk_out, "w") as hk_f, open(args.rr_out, "w") as rr_f:
        for rec in SeqIO.parse(args.faa, "fasta"):
            ptype = type_map.get(rec.id)
            if ptype == "HK":
                SeqIO.write(rec, hk_f, "fasta")
                hk_count += 1
            elif ptype == "RR":
                SeqIO.write(rec, rr_f, "fasta")
                rr_count += 1

    print(f"Split complete: {hk_count} HK proteins, {rr_count} RR proteins")


if __name__ == "__main__":
    main()
