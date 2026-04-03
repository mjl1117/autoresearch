#!/usr/bin/env python3

import subprocess
import argparse
from pathlib import Path


def align_sequences(fasta, out):

    subprocess.run([
        "mafft",
        "--auto",
        fasta
    ], stdout=open(out, "w"), check=True)


def build_tree(alignment, outprefix):

    subprocess.run([
        "iqtree3",
        "-s",
        alignment,
        "-m",
        "MFP",
        "-bb",
        "1000",
        "-nt",
        "AUTO",
        "-pre",
        outprefix
    ], check=True)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--fasta", required=True)
    parser.add_argument("--outdir", required=True)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    aln = outdir / "alignment.fasta"

    align_sequences(args.fasta, aln)

    build_tree(aln, str(outdir / "tree"))


if __name__ == "__main__":
    main()
