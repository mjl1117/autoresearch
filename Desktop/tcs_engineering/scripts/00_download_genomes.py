#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

REFSEQ_SUMMARY = "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/bacteria/assembly_summary.txt"

THREADS = 4  # NCBI rate-limits heavy parallel FTP requests


def ensure_dirs(base):

    base = Path(base)
    genome_dir = base / "genomes"
    metadata_dir = base / "metadata"

    genome_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    return genome_dir, metadata_dir


import re


def download_file(url, outfile):

    subprocess.run(
        ["curl", "-L", "--fail", "--max-time", "300", "--connect-timeout", "30",
         "-o", str(outfile), url],
        check=True
    )

    if outfile.stat().st_size < 10000:
        outfile.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed or truncated: {url}")


def list_ftp_directory(ftp_url: str) -> list[str] | None:
    """Return filenames in an NCBI FTP directory, or None if inaccessible.

    NCBI serves FTP over HTTPS which returns Apache-style HTML directory pages.
    Parses href attributes from the HTML to extract actual filenames.
    Returns None on any error (404, timeout, etc).
    """
    result = subprocess.run(
        ["curl", "-s", "--fail", "--max-time", "20", "--connect-timeout", "10",
         ftp_url.rstrip("/") + "/"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    # Parse Apache-style HTML directory listing — extract href values that
    # look like filenames (not parent dir "..", sort query strings "?...", etc.)
    hrefs = re.findall(r'href="([^"]+)"', result.stdout)
    filenames = [h for h in hrefs if not h.startswith("?") and h != "../" and h != "/"]
    return filenames if filenames else None


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
    """Download genome, protein, and GFF files for one assembly.

    Uses FTP directory listing to discover actual filenames — handles naming
    variations and fast-fails (with a clear error) if the directory is gone.
    """
    ftp = row["ftp_path"].rstrip("/")
    assembly = ftp.split("/")[-1]

    outdir = genome_dir / assembly
    outdir.mkdir(exist_ok=True)

    # Fetch directory listing once; fail fast if the path doesn't exist on NCBI.
    listing = list_ftp_directory(ftp)
    if listing is None:
        raise RuntimeError(f"FTP directory not found (404 or timeout): {ftp}")

    wanted = [
        ("genome.fna",   "_genomic.fna.gz"),
        ("proteins.faa", "_protein.faa.gz"),
        ("genome.gff",   "_genomic.gff.gz"),
    ]

    for outname, suffix in wanted:
        outfile = outdir / outname
        if outfile.exists():
            continue

        # Find the actual filename in the listing. Exclude derived files
        # (_cds_from_genomic.fna.gz, _rna_from_genomic.fna.gz) that share
        # the _genomic.fna.gz suffix but contain only CDS/RNA sequences.
        match = next(
            (f for f in listing
             if f.endswith(suffix)
             and "_cds_from_" not in f
             and "_rna_from_" not in f),
            None,
        )
        if match is None:
            raise RuntimeError(f"No {suffix} file in {ftp}; listing: {listing[:5]}")

        gz = outfile.with_suffix(outfile.suffix + ".gz")
        filename = match.split("/")[-1]
        print("Downloading", assembly, filename)
        download_file(f"{ftp}/{filename}", gz)
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


BLACKLIST_PATH = "data/reference/ftp_blacklist.txt"


def _append_blacklist(full_id: str, reason: str) -> None:
    """Record a permanently-broken assembly so future selection skips it."""
    Path(BLACKLIST_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(BLACKLIST_PATH, "a") as fh:
        fh.write(f"{full_id}\t{reason}\n")


def _load_blacklist() -> set[str]:
    """Return the set of full_ids known to have broken FTP paths."""
    p = Path(BLACKLIST_PATH)
    if not p.exists():
        return set()
    return {line.split("\t")[0] for line in p.read_text().splitlines() if line.strip()}


def _download_from_manifest(manifest_path: str, genome_dir: Path) -> None:
    """Download genomes listed in a two-column TSV (full_id, ftp_path).

    Skips entries whose folder already exists (idempotent).
    Skips entries in ftp_blacklist.txt (confirmed bad FTP paths).
    Failed downloads are added to the blacklist so future runs skip them.
    """
    blacklist = _load_blacklist()

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
            if full_id in blacklist:
                continue  # silently skip; was printed in detail when first blacklisted
            assembly = ftp_path.rstrip("/").split("/")[-1]
            if (genome_dir / full_id).exists() or (genome_dir / assembly).exists():
                print(f"  Already present, skipping: {full_id}")
                continue
            rows.append({"full_id": full_id, "ftp_path": ftp_path})

    print(f"Manifest: {len(rows)} new genomes to download")

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
                _append_blacklist(row["full_id"], f"download_failed: {e}")
                print(f"  FAILED (blacklisted): {row['full_id']}: {e}")


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
