#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

REFSEQ_SUMMARY = "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/assembly_summary.txt"

THREADS = 10


def ensure_dirs(base):

    base = Path(base)
    genome_dir = base / "genomes"
    metadata_dir = base / "metadata"

    genome_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    return genome_dir, metadata_dir


def download_file(url, outfile):

    subprocess.run(
        ["curl", "-L", "--fail", "-o", str(outfile), url],
        check=True
    )

    if outfile.stat().st_size < 10000:
        outfile.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed or truncated: {url}")


def download_assembly_summary(metadata_dir):

    outfile = metadata_dir / "assembly_summary.txt"

    if not outfile.exists():

        print("Downloading RefSeq assembly summary...")

        download_file(REFSEQ_SUMMARY, outfile)

    return outfile


def parse_summary(summary_file, max_genomes):

    df = pd.read_csv(summary_file, sep="\t", skiprows=1, dtype=str)

    df = df[df["assembly_level"] == "Complete Genome"]
    df = df[df["ftp_path"] != "na"]

    return df.head(max_genomes)


def download_genome(row, genome_dir):

    ftp = row["ftp_path"].rstrip("/")
    assembly = ftp.split("/")[-1]

    outdir = genome_dir / assembly
    outdir.mkdir(exist_ok=True)

    files = {
        "genome.fna": f"{assembly}_genomic.fna.gz",
        "proteins.faa": f"{assembly}_protein.faa.gz",
        "genome.gff": f"{assembly}_genomic.gff.gz"
    }

    for name, filename in files.items():

        outfile = outdir / name

        if outfile.exists():
            continue

        gz = outfile.with_suffix(outfile.suffix + ".gz")
        url = f"{ftp}/{filename}"

        print("Downloading", assembly, filename)

        download_file(url, gz)

        subprocess.run(["gunzip", "-f", str(gz)], check=True)

    return outdir


def detect_tcs(genome_dir):

    proteins = genome_dir / "proteins.faa"

    if not proteins.exists():
        return False

    result = subprocess.run(
        [
            "hmmsearch",
            "--tblout",
            "/dev/stdout",
            "pfam_tcs.hmm",
            str(proteins)
        ],
        capture_output=True,
        text=True
    )

    lines = [l for l in result.stdout.splitlines() if not l.startswith("#")]

    return len(lines) > 0


def write_metadata(df, metadata_dir):

    outfile = metadata_dir / "genome_metadata.tsv"

    df[[
        "assembly_accession",
        "organism_name",
        "taxid",
        "assembly_level",
        "ftp_path"
    ]].to_csv(outfile, sep="\t", index=False)

    print("Metadata written:", outfile)


def _download_from_manifest(manifest_path: str, genome_dir: Path) -> None:
    """Download genomes listed in a two-column TSV (full_id, ftp_path).

    Skips entries whose folder already exists in genome_dir (idempotent).
    Uses the same download_genome() logic as the standard path.
    Does NOT run TCS detection — manifested genomes are pre-selected.
    """
    rows = []
    with open(manifest_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                print(f"  Skipping malformed manifest line: {line!r}")
                continue
            full_id, ftp_path = parts[0], parts[1]
            if (genome_dir / full_id).exists():
                print(f"  Already present, skipping: {full_id}")
                continue
            rows.append({"full_id": full_id, "ftp_path": ftp_path})

    print(f"Manifest: {len(rows)} genomes to download")

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = {
            executor.submit(download_genome, row, genome_dir): row
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                future.result()
                print(f"  Downloaded: {row['full_id']}")
            except Exception as e:
                print(f"  FAILED: {row['full_id']}: {e}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--max_genomes", type=int, default=20)
    parser.add_argument("--manifest", default=None,
                        help="TSV manifest from select_expansion_genomes.py "
                             "(columns: full_id, ftp_path). When provided, "
                             "skips assembly_summary download and TCS detection.")

    args = parser.parse_args()

    genome_dir, metadata_dir = ensure_dirs(args.data_dir)

    if args.manifest:
        _download_from_manifest(args.manifest, genome_dir)
        return

    summary = download_assembly_summary(metadata_dir)

    df = parse_summary(summary, args.max_genomes)

    print("Selected genomes:", len(df))

    downloaded = []

    with ThreadPoolExecutor(max_workers=THREADS) as executor:

        futures = {
            executor.submit(download_genome, row, genome_dir): row
            for _, row in df.iterrows()
        }

        for future in as_completed(futures):

            row = futures[future]

            try:

                gdir = future.result()

                if detect_tcs(gdir):

                    downloaded.append(row.to_dict())

                    print("TCS detected:", row["organism_name"])

                else:

                    print("No TCS:", row["organism_name"])

                    shutil.rmtree(gdir)

            except Exception as e:

                print("Download failed:", e)

    metadata_df = pd.DataFrame(downloaded)

    write_metadata(metadata_df, metadata_dir)


if __name__ == "__main__":
    main()
