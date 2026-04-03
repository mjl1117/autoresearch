#!/usr/bin/env python3

import pandas as pd
import argparse

def load_gff(gff):
    rows = []
    with open(gff) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue  # Skip malformed lines
            attrs = parts[8]
            if "protein_id=" not in attrs:
                continue
            try:
                protein = attrs.split("protein_id=")[1].split(";")[0]
                rows.append({
                    "protein": protein,
                    "start": int(parts[3]),
                    "end": int(parts[4]),
                    "strand": parts[6],
                    "seqid": parts[0]  # Add contig/chromosome for better sorting
                })
            except (IndexError, ValueError):
                continue  # Skip if parsing fails
    return pd.DataFrame(rows)

def pair_neighbors(class_df, gff_df):
    merged = class_df.merge(gff_df, on="protein")
    # Sort by seqid, strand, then start
    merged = merged.sort_values(["seqid", "strand", "start"])
    pairs = []
    for i in range(len(merged) - 1):
        a = merged.iloc[i]
        b = merged.iloc[i + 1]
        # Only pair if same seqid, same strand, HK then RR, and positive distance
        if (a.seqid == b.seqid and a.strand == b.strand and
            a.type == "HK" and b.type == "RR" and b.start > a.end):
            pairs.append({
                "HK": a.protein,
                "RR": b.protein,
                "distance": b.start - a.end
            })
    return pd.DataFrame(pairs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proteins", required=True)
    parser.add_argument("--gff", required=True)
    parser.add_argument("--out", required=True)  # Make output required
    args = parser.parse_args()
    class_df = pd.read_csv(args.proteins)
    gff_df = load_gff(args.gff)
    pairs = pair_neighbors(class_df, gff_df)
    pairs.to_csv(args.out, index=False, sep='\t')  # Changed to TSV format

if __name__ == "__main__":
    main()
