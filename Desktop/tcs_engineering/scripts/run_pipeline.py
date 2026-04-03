#!/usr/bin/env python3

import subprocess
from pathlib import Path
import argparse
import sys

############################
# Helper
############################

def run(cmd):

    print("\nRunning:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

def detect_files(genome_dir):

    fna_files = list(genome_dir.glob("*.fna"))
    faa_files = list(genome_dir.glob("*.faa"))
    gff_files = list(genome_dir.glob("*.gff"))

    if not fna_files or not faa_files or not gff_files:
        raise FileNotFoundError("Missing required genome/protein/gff files in {}".format(genome_dir))

    genome = fna_files[0]
    proteins = faa_files[0]
    gff = gff_files[0]

    return genome, proteins, gff

############################
# Pipeline
############################

def run_pipeline(genome_dir, pfam, results_dir):

    genome_dir = Path(genome_dir)
    results_dir = Path(results_dir)

    genome, proteins, gff = detect_files(genome_dir)

    # stage directories
    domain_dir = results_dir / "domains"
    pair_dir = results_dir / "pairs"
    cluster_dir = results_dir / "clusters"
    phylo_dir = results_dir / "phylogeny"
    promoter_dir = results_dir / "promoters"
    motif_dir = results_dir / "motifs"

    for d in [
        domain_dir,
        pair_dir,
        cluster_dir,
        phylo_dir,
        promoter_dir,
        motif_dir
    ]:
        d.mkdir(parents=True, exist_ok=True)

    ############################
    # 1 DOMAIN DETECTION
    ############################

    run([
        sys.executable,
        "scripts/01_detect_domains.py",
        "--proteins",
        str(proteins),
        "--pfam",
        pfam,
        "--outdir",
        str(domain_dir)
    ])

    ############################
    # 2 GENE PAIRING
    ############################

    run([
        sys.executable,
        "scripts/02_pair_adjacent_genes.py",
        "--proteins",
        str(domain_dir / "tcs_proteins.csv"),
        "--gff",
        str(gff),
        "--out",
        str(pair_dir / "tcs_pairs.csv")
    ])

    ############################
    # 3 CLUSTERING
    ############################

    run([
        sys.executable,
        "scripts/03_cluster_sequences.py",
        "--hk_fasta",
        str(domain_dir / "histidine_kinases.faa"),
        "--rr_fasta",
        str(domain_dir / "response_regulators.faa"),
        "--outdir",
        str(cluster_dir)
    ])

    ############################
    # 4 PHYLOGENY
    ############################

    # MAFFT alignment for HK
    hk_fasta = domain_dir / "histidine_kinases.faa"
    hk_aligned = phylo_dir / "hk" / "hk_aligned.faa"
    (phylo_dir / "hk").mkdir(exist_ok=True)
    subprocess.run([
        "mafft",
        "--anysymbol",
        "--auto",
        str(hk_fasta)
    ], stdout=open(hk_aligned, "w"), check=True)

    # MAFFT alignment for RR
    rr_fasta = domain_dir / "response_regulators.faa"
    rr_aligned = phylo_dir / "rr" / "rr_aligned.faa"
    (phylo_dir / "rr").mkdir(exist_ok=True)
    subprocess.run([
        "mafft",
        "--anysymbol",
        "--auto",
        str(rr_fasta)
    ], stdout=open(rr_aligned, "w"), check=True)

    # Build phylogeny
    run([
        sys.executable,
        "scripts/04_build_phylogeny.py",
        "--fasta",
        str(hk_aligned),
        "--outdir",
        str(phylo_dir / "hk")
    ])

    run([
        sys.executable,
        "scripts/04_build_phylogeny.py",
        "--fasta",
        str(rr_aligned),
        "--outdir",
        str(phylo_dir / "rr")
    ])

    ############################
    # 5 PROMOTERS
    ############################

    run([
        sys.executable,
        "scripts/05_extract_promoters.py",
        "--genome",
        str(genome),
        "--gff",
        str(gff),
        "--pairs",
        str(pair_dir / "tcs_pairs.csv"),
        "--out",
        str(promoter_dir / "promoters.fasta")
    ])

    ############################
    # 6 MOTIF DISCOVERY
    ############################

    run([
        sys.executable,
        "scripts/06_discover_motifs.py",
        "--promoters",
        str(promoter_dir / "promoters.fasta"),
        "--outdir",
        str(motif_dir)
    ])

    print("\nPipeline complete.")


############################
# CLI
############################

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--genome_dir",
        required=True,
        help="directory containing genome.fna, proteins.faa, genome.gff"
    )

    parser.add_argument(
        "--pfam",
        required=True,
        help="Pfam-A.hmm file"
    )

    parser.add_argument(
        "--results",
        default="results"
    )

    args = parser.parse_args()

    run_pipeline(
        args.genome_dir,
        args.pfam,
        args.results
    )


if __name__ == "__main__":
    main()
