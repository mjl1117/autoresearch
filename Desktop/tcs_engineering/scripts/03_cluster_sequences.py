#!/usr/bin/env python3

import argparse
import subprocess
import shutil
from pathlib import Path


def run_mmseqs(fasta, outdir):

    outdir = Path(outdir)
    seqdb = outdir / "seqdb"
    clusterdb = outdir / "clusterdb"
    tmp = outdir / "tmp"

    if outdir.exists():
        shutil.rmtree(outdir)

    outdir.mkdir(parents=True)

    subprocess.run([
        "mmseqs",
        "createdb",
        fasta,
        seqdb
    ], check=True)

    subprocess.run([
        "mmseqs",
        "cluster",
        seqdb,
        clusterdb,
        tmp,
        "--min-seq-id",
        "0.4"
    ], check=True)

    subprocess.run([
        "mmseqs",
        "createtsv",
        seqdb,
        seqdb,
        clusterdb,
        outdir / "clusters.tsv"
    ], check=True)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--hk_fasta")
    parser.add_argument("--rr_fasta")
    parser.add_argument("--outdir")

    args = parser.parse_args()

    run_mmseqs(args.hk_fasta, Path(args.outdir) / "hk_clusters")

    run_mmseqs(args.rr_fasta, Path(args.outdir) / "rr_clusters")


if __name__ == "__main__":
    main()
