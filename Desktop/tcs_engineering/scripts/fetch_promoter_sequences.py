#!/usr/bin/env python3
"""Fetch 300 bp upstream sequences for characterized TCS output promoters from NCBI Entrez.

For each row in characterized_promoters.tsv (excluding CheY):
  1. Use Entrez efetch on the source genome accession (GenBank format)
  2. Locate the gene by gene name or locus_tag
  3. Extract upstream_bp upstream of the CDS start (strand-aware)
  4. Write FASTA to data/reference/promoter_sequences/{promoter_name}.fasta

Usage:
    python scripts/fetch_promoter_sequences.py \
        --promoters data/reference/characterized_promoters.tsv \
        --outdir data/reference/promoter_sequences \
        --upstream_bp 300 \
        --email your@email.com
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq


def fetch_gene_location(genome_acc: str, gene_name: str, email: str, record=None):
    """Return (cds_start, strand, record) for gene_name on genome_acc.

    cds_start is the 0-based coordinate of the transcription start:
      - strand +1: leftmost base of CDS
      - strand -1: rightmost base of CDS (= upstream on minus strand)
    Returns the genome SeqRecord so the caller can cache it.
    """
    Entrez.email = email
    if record is None:
        # gbwithparts is required for chromosomal records — 'gb' returns only source feature
        handle = Entrez.efetch(db="nuccore", id=genome_acc, rettype="gbwithparts", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()
        time.sleep(0.4)

    gene_name_lower = gene_name.lower()
    for feature in record.features:
        if feature.type not in ("CDS", "gene"):
            continue
        qualifiers = feature.qualifiers
        names = []
        names += qualifiers.get("gene", [])
        names += qualifiers.get("locus_tag", [])
        if any(gene_name_lower == n.lower() for n in names):
            loc = feature.location
            strand = loc.strand
            parts = list(loc.parts)
            if strand == 1:
                cds_start = int(parts[0].start)
            else:
                cds_start = int(parts[-1].end)
            return cds_start, strand, record

    raise ValueError(f"Gene '{gene_name}' not found on {genome_acc}")


def extract_upstream(record, cds_start: int, strand: int, upstream_bp: int) -> str:
    """Return upstream_bp bases immediately upstream of cds_start, strand-aware."""
    genome_len = len(record.seq)
    if strand == 1:
        fetch_start = max(0, cds_start - upstream_bp)
        fetch_end = cds_start
        seq = str(record.seq[fetch_start:fetch_end])
    else:
        # cds_start is the end of the CDS on the + strand (= upstream on - strand)
        fetch_start = cds_start
        fetch_end = min(genome_len, cds_start + upstream_bp)
        seq = str(record.seq[fetch_start:fetch_end].reverse_complement())
    return seq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promoters",   required=True,
                        default="data/reference/characterized_promoters.tsv")
    parser.add_argument("--outdir",      required=True,
                        default="data/reference/promoter_sequences")
    parser.add_argument("--upstream_bp", type=int, default=300)
    parser.add_argument("--email",       required=True,
                        help="Email required by NCBI Entrez (NCBI policy)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.promoters, sep="\t")
    # Skip hard-excluded and non-recommended entries
    df = df[df["recommended"].astype(str).str.lower() == "true"]
    df = df[df["target_gene"].astype(str) != "none"]
    df = df[df["gene_name_ncbi"].astype(str) != "none"]

    # Cache genome records to avoid redundant NCBI downloads
    genome_cache: dict = {}

    for _, row in df.iterrows():
        pname = row["promoter_name"]
        gene  = row["gene_name_ncbi"]
        acc   = row["ncbi_genome_accession"]
        out_fasta = outdir / f"{pname}.fasta"

        if out_fasta.exists():
            print(f"  [skip] {pname} already exists")
            continue

        print(f"  Fetching {pname} ({gene} on {acc}) ...")
        try:
            cached_record = genome_cache.get(acc)
            cds_start, strand, record = fetch_gene_location(
                acc, gene, args.email, record=cached_record
            )
            genome_cache[acc] = record

            upstream_seq = extract_upstream(record, cds_start, strand, args.upstream_bp)
            if len(upstream_seq) < 50:
                print(f"  WARNING: {pname} upstream very short ({len(upstream_seq)} bp) — near genome edge?")

            sr = SeqRecord(
                Seq(upstream_seq),
                id=pname,
                description=(
                    f"gene={gene} organism={row['source_organism']} "
                    f"genome={acc} upstream_bp={args.upstream_bp} "
                    f"signal={row['inducing_signal']}"
                )
            )
            with open(out_fasta, "w") as fh:
                SeqIO.write(sr, fh, "fasta")
            print(f"  Saved: {out_fasta} ({len(upstream_seq)} bp)")

        except Exception as e:
            print(f"  ERROR fetching {pname}: {e}")

        time.sleep(0.4)  # NCBI rate limit: max 3 requests/second without API key

    print(f"\nDone. Sequences in {outdir}/")


if __name__ == "__main__":
    main()
