#!/usr/bin/env python3

import subprocess
import pandas as pd
from pathlib import Path
import argparse

############################
# DOMAIN DEFINITIONS
############################

HK_DOMAINS = {"HisKA", "HATPase_c"}
RR_DOMAINS = {"Response_reg", "REC"}

############################
# Parse HMMER output
############################

def parse_domtbl(domtbl_file):

    records = []

    with open(domtbl_file) as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.split()

            target = parts[0]
            domain = parts[3]
            evalue = float(parts[6])

            records.append({
                "protein": target,
                "domain": domain,
                "evalue": evalue
            })

    return pd.DataFrame(records)

############################
# Classify proteins
############################

def classify_proteins(df):

    protein_domains = df.groupby("protein")["domain"].apply(set)

    classifications = []

    for protein, domains in protein_domains.items():

        if HK_DOMAINS & domains:
            cls = "histidine_kinase"

        elif RR_DOMAINS & domains:
            cls = "response_regulator"

        else:
            continue

        classifications.append({
            "protein": protein,
            "type": cls
        })

    return pd.DataFrame(classifications)

############################
# Pair HK and RR genes
############################

def pair_systems(class_df):

    hks = class_df[class_df.type == "histidine_kinase"]
    rrs = class_df[class_df.type == "response_regulator"]

    pairs = []

    for _, hk in hks.iterrows():
        for _, rr in rrs.iterrows():

            # simple placeholder pairing
            pairs.append({
                "histidine_kinase": hk.protein,
                "response_regulator": rr.protein
            })

    return pd.DataFrame(pairs)

############################
# Main
############################

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--proteins", required=True)
    parser.add_argument("--pfam", required=True)
    parser.add_argument("--outdir", default="results")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    domtbl = outdir / "pfam_hits.domtbl"

    df = parse_domtbl(domtbl)

    class_df = classify_proteins(df)

    pairs = pair_systems(class_df)

    class_df.to_csv(outdir / "tcs_proteins.csv", index=False)
    pairs.to_csv(outdir / "tcs_pairs.csv", index=False)

    print("Finished.")
    print(f"Proteins classified: {len(class_df)}")
    print(f"Candidate pairs: {len(pairs)}")

if __name__ == "__main__":
    main()
