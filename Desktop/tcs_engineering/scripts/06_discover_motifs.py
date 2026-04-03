#!/usr/bin/env python3

import subprocess
import argparse


def run_meme(promoters, outdir):

    cmd = [
        "meme",
        promoters,
        "-dna",
        "-oc",
        outdir,
        "-nmotifs",
        "5",
        "-minw",
        "6",
        "-maxw",
        "25"
    ]

    subprocess.run(cmd, check=True)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--promoters")
    parser.add_argument("--outdir")

    args = parser.parse_args()

    run_meme(args.promoters, args.outdir)


if __name__ == "__main__":
    main()
