#!/usr/bin/env python3

import pandas as pd
import argparse
from Bio import SeqIO

def load_ids(fasta_file):
    ids = set()
    for record in SeqIO.parse(fasta_file, "fasta"):
        ids.add(record.id)
    return ids

def extract_protein_id(attr):
    # Prioritise protein_id= (e.g. WP_xxxxx.1) over ID= (e.g. cds-WP_xxxxx.1)
    # so the returned value matches the accession used in proteins.faa
    fields = {f.split("=")[0]: f.split("=")[1] for f in attr.split(";") if "=" in f}
    return fields.get("protein_id") or fields.get("ID")


def extract_promoters(genome, gff, pairs, length):
    genome_dict = SeqIO.to_dict(SeqIO.parse(genome, "fasta"))

    gff_df = pd.read_csv(
        gff,
        sep="\t",
        comment="#",
        header=None,
        names=[
            "seqid", "source", "type", "start", "end",
            "score", "strand", "phase", "attributes"
        ]
    )

    gff_df["protein_id"] = gff_df["attributes"].apply(extract_protein_id)

    ids = load_ids(pairs)

    promoters = []

    for rr in ids:

        gene = gff_df[gff_df["protein_id"] == rr]

        if gene.empty:
            continue

        gene = gene.iloc[0]
        
        if gene.seqid not in genome_dict:
            continue

        contig = genome_dict[gene.seqid].seq

        if gene.strand == "+":
            start = max(0, gene.start - 1 - length)
            end = gene.start - 1
            seq = contig[start:end]

        else:
            start = gene.end
            end = gene.end + length
            seq = contig[start:end].reverse_complement()

        promoters.append({
            "RR": rr,
            "sequence": str(seq)
        })

    return promoters


def main():
    
    parser = argparse.ArgumentParser()
    # Update these to match what the .smk file is sending
    parser.add_argument("--fna", dest="genome") 
    parser.add_argument("--gff", required=True)
    parser.add_argument("--ids", dest="pairs") 
    parser.add_argument("--upstream", type=int, default=300)
    parser.add_argument("--output", dest="out")
    args = parser.parse_args()

    promoters = extract_promoters(
        args.genome,
        args.gff,
        args.pairs,
        args.upstream
    )

    with open(args.out, "w") as f:

        for i, p in enumerate(promoters):

            f.write(f">promoter_{i}_{p['RR']}\n")
            f.write(p["sequence"] + "\n")


if __name__ == "__main__":
    main()
