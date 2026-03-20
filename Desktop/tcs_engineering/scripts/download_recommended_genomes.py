#!/usr/bin/env python3
"""Download RECOMMENDED_GENOMES from NCBI RefSeq.

Uses tcs_constants.RECOMMENDED_GENOMES accession list to look up FTP paths
in assembly_summary.txt and download proteins.faa + genomic.fna + genomic.gff
for each organism. These are the 12 core reference organisms that must be
in any rigorous TCS survey.

Usage:
  python scripts/download_recommended_genomes.py \
      --assembly_summary data/metadata/assembly_summary.txt \
      --genome_dir data/genomes \
      [--threads 4]
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# Add scripts/ to path for tcs_constants import
sys.path.insert(0, str(Path(__file__).parent))
from tcs_constants import RECOMMENDED_GENOMES


def load_ftp_paths(summary_path: str) -> dict:
    """Return {accession: ftp_path} from assembly_summary.txt."""
    df = pd.read_csv(
        summary_path, sep="\t", skiprows=2, header=None,
        usecols=[0, 15, 19],
        names=["accession", "asm_name", "ftp_path"],
        low_memory=False,
    )
    df = df[df["ftp_path"] != "na"]
    return df.set_index("accession")["ftp_path"].to_dict()


def download_genome(accession: str, ftp_base: str, genome_dir: Path) -> str:
    """Download proteins.faa, genomic.fna, genomic.gff for one accession."""
    ftp_base = ftp_base.rstrip("/")   # NCBI paths sometimes have trailing slash
    prefix = os.path.basename(ftp_base)
    out_dir = genome_dir / f"{accession}_{prefix.split('_', 2)[-1]}"

    # Check if already complete
    if (out_dir / "proteins.faa").exists() and \
       (out_dir / "genome.fna").exists() and \
       (out_dir / "genome.gff").exists():
        return f"  SKIP (already exists): {out_dir.name}"

    out_dir.mkdir(parents=True, exist_ok=True)
    errors = []

    files = [
        (f"{ftp_base}/{prefix}_protein.faa.gz",    out_dir / "proteins.faa"),
        (f"{ftp_base}/{prefix}_genomic.fna.gz",    out_dir / "genome.fna"),
        (f"{ftp_base}/{prefix}_genomic.gff.gz",    out_dir / "genome.gff"),
    ]

    for url, dest in files:
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        try:
            result = subprocess.run(
                f"curl -L --fail --silent '{url}' | gunzip > '{dest}'",
                shell=True, capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0 or (dest.exists() and dest.stat().st_size < 100):
                errors.append(f"{dest.name}: download failed")
                dest.unlink(missing_ok=True)
        except subprocess.TimeoutExpired:
            errors.append(f"{dest.name}: timeout")
            dest.unlink(missing_ok=True)

    if errors:
        return f"  ERROR {accession}: {'; '.join(errors)}"
    return f"  OK: {out_dir.name}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assembly_summary", required=True)
    parser.add_argument("--genome_dir", required=True)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()

    genome_dir = Path(args.genome_dir)
    genome_dir.mkdir(parents=True, exist_ok=True)

    print("Loading FTP paths from assembly_summary.txt...")
    ftp_map = load_ftp_paths(args.assembly_summary)

    tasks = []
    missing = []
    for name, acc in RECOMMENDED_GENOMES.items():
        if acc in ftp_map:
            tasks.append((acc, name, ftp_map[acc]))
        else:
            missing.append(f"  {acc}  {name}")

    if missing:
        print(f"\nWARNING: {len(missing)} accessions not found in assembly_summary.txt:")
        print("\n".join(missing))
        print("These may have been superseded — check NCBI for current accessions.\n")

    print(f"\nDownloading {len(tasks)} reference genomes with {args.threads} threads...\n")

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {
            pool.submit(download_genome, acc, ftp, genome_dir): (acc, name)
            for acc, name, ftp in tasks
        }
        for future in as_completed(futures):
            acc, name = futures[future]
            try:
                msg = future.result()
                print(f"[{name[:35]:<35}] {msg}")
            except Exception as e:
                print(f"[{name[:35]:<35}]   EXCEPTION: {e}")

    print("\nDone. Run curate_genome_set.py next, then start the pipeline.")


if __name__ == "__main__":
    main()
