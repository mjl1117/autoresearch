#!/usr/bin/env python3

import subprocess
import pandas as pd
from pathlib import Path
import argparse
from Bio import SeqIO

# HisKA domain families that contain the phosphorylatable histidine - the
# definitive feature distinguishing histidine kinases from other GHKL ATPases
# (HtpG/Hsp90, GyrB, MutL all have HATPase_c but NOT a HisKA-family domain).
# A protein must carry at least one HisKA-family domain to be classified HK;
# HATPase_c alone is insufficient and a known source of false positives.
HK_DOMAINS_REQUIRED = {"HisKA", "HisKA_3", "HisK_5", "HA1_5", "HisK_3KD"}
# HATPase_c is not required for all HKs but is noted when present
HATP_DOMAINS = {"HATPase_c"}
RR_DOMAINS = {"Response_reg", "REC"}

def write_sequences(ids, protein_fasta, outfile):

    ids = set(ids)

    with open(outfile, "w") as out:

        for rec in SeqIO.parse(protein_fasta, "fasta"):

            if rec.id in ids:
                SeqIO.write(rec, out, "fasta")

EVALUE_THRESHOLD = 1e-3  # HMMER's default inclusion threshold; excludes noise hits
                         # (e.g. GrpE with spurious HisKA_3 hit at E=0.36)


def parse_domtbl(file):
    rows = []
    with open(file) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            parts = line.split()

            # HMMER tblout format:
            # target name [0], accession [1], query name [2], accession [3]...
            # We want target (protein) and query (domain)
            evalue = float(parts[4])
            if evalue > EVALUE_THRESHOLD:
                continue
            rows.append({
                "protein": parts[0],
                "domain": parts[2],
                "evalue": evalue,
            })

    return pd.DataFrame(rows)


def classify(df):

    protein_domains = df.groupby("protein")["domain"].apply(set)

    results = []

    for protein, domains in protein_domains.items():

        # HK: must carry a HisKA-family domain (phosphorylatable histidine helix)
        # HATPase_c alone matches Hsp90, GyrB, MutL — exclude those false positives
        if HK_DOMAINS_REQUIRED & domains:
            ptype = "HK"

        elif RR_DOMAINS & domains:
            ptype = "RR"

        else:
            continue

        results.append({
            "protein": protein,
            "type": ptype,
            "domains": ";".join(sorted(domains))
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proteins", required=True)
    parser.add_argument("--hmm_table", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv_output")  # New optional flag
    args = parser.parse_args()

    # Convert the output path to a Path object to handle parent directories
    output_file = Path(args.output)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    # Parse the existing HMM table
    df = parse_domtbl(args.hmm_table)

    csv_path = Path(args.csv_output) if args.csv_output else output_file.parent / f"{output_file.stem}_proteins.csv"
    csv_path.parent.mkdir(exist_ok=True, parents=True)

    if df.empty:
        print(f"No domains found in {args.hmm_table}")
        open(args.output, 'a').close()
        pd.DataFrame(columns=["protein", "type", "domains"]).to_csv(csv_path, index=False)
        return

    classified = classify(df)

    if classified.empty or "type" not in classified.columns:
        print(f"Warning: No HK or RR domains identified in {args.hmm_table}")
        open(args.output, 'w').close()
        pd.DataFrame(columns=["protein", "type", "domains"]).to_csv(csv_path, index=False)
        return

    # Now this won't crash
    hk_ids = classified[classified["type"] == "HK"]["protein"].tolist()

    classified.to_csv(csv_path, index=False)

    # Define hk_ids and rr_ids from classified DataFrame
    hk_ids = classified[classified["type"] == "HK"]["protein"].tolist()
    rr_ids = classified[classified["type"] == "RR"]["protein"].tolist()
    
    # Write the FASTA file
    write_sequences(hk_ids + rr_ids, args.proteins, args.output)
    print(f"Extracted {len(hk_ids)} HKs and {len(rr_ids)} RRs to {args.output}")


if __name__ == "__main__":
    main()
