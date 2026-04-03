#!/usr/bin/env python3
"""Select ~200 genomes from NCBI assembly_summary.txt for TCS pipeline expansion.

Selection strategy:
  1. Seed: always include RECOMMENDED_GENOMES accessions (from tcs_constants).
  2. Priority fill: guarantee 1 genome per priority genus not already in seed
     (currently only Pseudomonas — others are covered by RECOMMENDED_GENOMES).
  3. Diversity fill: 5/genus cap, Complete Genome preferred, until target_n reached.
     Priority genera are excluded from the diversity fill.

Outputs:
  expansion_download_manifest.txt  — new genomes only (tab-sep: full_id, ftp_path)
  curated_genome_list.txt          — all selected full_ids (one per line)

Usage:
  python scripts/select_expansion_genomes.py \\
      --assembly_summary data/metadata/assembly_summary.txt \\
      --genome_dir       data/genomes \\
      --output_manifest  data/reference/expansion_download_manifest.txt \\
      --output_list      data/reference/curated_genome_list.txt \\
      --target_n         200
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from tcs_constants import ASSEMBLY_LEVEL_PRIORITY, RECOMMENDED_GENOMES

# Priority genera: must have at least 1 representative in the final set.
# Myxococcus, Streptomyces, Anabaena, Caulobacter, Rhodobacter are already
# in RECOMMENDED_GENOMES; only Pseudomonas needs explicit priority selection.
PRIORITY_GENERA = [
    "Myxococcus", "Streptomyces", "Pseudomonas",
    "Anabaena", "Caulobacter", "Rhodobacter",
]


def get_downloaded_dirs(genome_dir: str) -> set[str]:
    """Return the set of already-downloaded genome directory names (GCF_* only)."""
    return {d for d in os.listdir(genome_dir) if d.startswith("GCF_")}


def parse_assembly_summary(path: str, existing_dirs: set[str]) -> pd.DataFrame:
    """Load assembly_summary.txt; add derived columns used for selection.

    Follows curate_genome_set.py conventions exactly:
      skiprows=2, header=None, usecols=[0,5,7,10,11,15,19].

    Filters to version_status == "latest" and assembly_level in
    {"Complete Genome", "Chromosome"} to exclude suppressed assemblies and
    Scaffold/Contig entries whose FTP paths are frequently stale.
    """
    df = pd.read_csv(
        path, sep="\t", skiprows=2, header=None,
        usecols=[0, 5, 7, 10, 11, 15, 19],
        names=["accession", "taxid", "organism_name",
               "version_status", "assembly_level", "asm_name", "ftp_path"],
        low_memory=False,
        dtype=str,
    )
    # Keep only current, accessible assemblies
    df = df[df["ftp_path"] != "na"].copy()
    df = df[df["version_status"] == "latest"].copy()
    df = df[df["assembly_level"].isin(["Complete Genome", "Chromosome"])].copy()
    df["full_id"] = df["accession"] + "_" + df["asm_name"]
    df["genus"] = df["organism_name"].str.split().str[0]
    df["level_rank"] = df["assembly_level"].map(ASSEMBLY_LEVEL_PRIORITY).fillna(99)
    df["already_downloaded"] = df["full_id"].isin(existing_dirs)
    return df


def select_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows matching RECOMMENDED_GENOMES accessions."""
    protected = set(RECOMMENDED_GENOMES.values())
    return df[df["accession"].isin(protected)].copy()


def select_priority(df: pd.DataFrame, priority_genera: list[str]) -> pd.DataFrame:
    """Return best 1 genome per priority genus not already in seed.

    Sort key: already_downloaded DESC (primary), level_rank ASC (secondary).
    Logs a warning for any genus absent from the catalog.
    """
    rows = []
    for genus in priority_genera:
        subset = df[df["genus"] == genus]
        if subset.empty:
            print(f"  WARNING: priority genus '{genus}' not found in assembly_summary")
            continue
        best = subset.sort_values(
            ["already_downloaded", "level_rank"],
            ascending=[False, True],
        ).iloc[[0]]
        rows.append(best)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=df.columns)


def select_diversity(
    df: pd.DataFrame,
    exclude_genera: set[str],
    exclude_full_ids: set[str],
    target_n: int,
    max_per_genus: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Return up to target_n genomes with genus cap, excluding specified genera/ids.

    Selection maximizes genus-level diversity: genera are drawn in random order
    (fixed seed for reproducibility) so that head(target_n) samples uniformly
    across the full taxonomic spread rather than biasing alphabetically-early
    genera.  Within each genus the best-quality assembly is taken first.

    Logs a warning if catalog exhausted before target_n.
    """
    import numpy as np

    pool = df[
        ~df["genus"].isin(exclude_genera) &
        ~df["full_id"].isin(exclude_full_ids)
    ].copy()

    # Assign each genus a random draw-order index so that when we round-robin
    # across genera and then call head(target_n), we sample genera uniformly
    # rather than front-loading whichever genera sort first alphabetically.
    rng = np.random.default_rng(seed)
    unique_genera = pool["genus"].unique()
    genus_order = {g: i for i, g in enumerate(rng.permutation(unique_genera))}
    pool["_genus_order"] = pool["genus"].map(genus_order)

    # Within each genus rank by quality (Complete Genome before Chromosome).
    # Round-robin: sort by (_rank, _genus_order) so all genera contribute their
    # best genome before any genus contributes a second.
    selected = (
        pool.sort_values(["level_rank", "full_id"])   # best quality first within genus
        .assign(_rank=lambda d: d.groupby("genus").cumcount())
        .query("_rank < @max_per_genus")
        .sort_values(["_rank", "_genus_order"])        # round-robin with random genus order
        .drop(columns=["_rank", "_genus_order"])
        .reset_index(drop=True)
    )

    if len(selected) < target_n:
        print(f"  WARNING: only {len(selected)} diversity genomes available "
              f"(target was {target_n})")
    return selected.head(target_n)


def write_outputs(
    selected: pd.DataFrame,
    existing_dirs: set[str],
    manifest_path: str,
    curated_list_path: str,
) -> None:
    """Write manifest (new genomes only) and curated list (all selected)."""
    new_genomes = selected[~selected["full_id"].isin(existing_dirs)]

    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    new_genomes[["full_id", "ftp_path"]].to_csv(
        manifest_path, sep="\t", index=False, header=False,
    )
    print(f"Manifest: {len(new_genomes)} new genomes → {manifest_path}")

    Path(curated_list_path).parent.mkdir(parents=True, exist_ok=True)
    Path(curated_list_path).write_text(
        "# Curated genome list — generated by select_expansion_genomes.py\n"
        f"# target_n=200  priority_genera={','.join(PRIORITY_GENERA)}\n"
        "# Format: full_id (accession_asmname) — matches data/genomes/ folder names\n"
        + "\n".join(selected["full_id"].tolist())
        + "\n"
    )
    print(f"Curated list: {len(selected)} genomes → {curated_list_path}")


def load_blacklist(path: str) -> set[str]:
    """Return full_ids that had broken FTP paths and should never be selected."""
    from pathlib import Path as _Path
    p = _Path(path)
    if not p.exists():
        return set()
    return {line.split("\t")[0] for line in p.read_text().splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assembly_summary", required=True)
    parser.add_argument("--genome_dir",       required=True)
    parser.add_argument("--output_manifest",  required=True)
    parser.add_argument("--output_list",      required=True)
    parser.add_argument("--target_n",         type=int, default=200)
    parser.add_argument("--blacklist",
                        default="data/reference/ftp_blacklist.txt",
                        help="TSV of full_ids with broken FTP paths to exclude")
    args = parser.parse_args()

    blacklisted = load_blacklist(args.blacklist)
    if blacklisted:
        print(f"  Blacklist: excluding {len(blacklisted)} known-broken assemblies")

    print("Loading assembly summary...")
    existing_dirs = get_downloaded_dirs(args.genome_dir)
    df = parse_assembly_summary(args.assembly_summary, existing_dirs)
    df = df[~df["full_id"].isin(blacklisted)].copy()  # exclude permanently broken
    print(f"  {len(df):,} assemblies with valid FTP paths")

    # Step 1: seed from RECOMMENDED_GENOMES
    seed = select_seed(df)
    print(f"  Seed (RECOMMENDED_GENOMES): {len(seed)} genomes")

    # Step 2: priority fill — 1/genus for genera not already in seed
    seed_accessions = set(seed["accession"])
    seed_genera = set(seed["genus"])
    priority_missing = [g for g in PRIORITY_GENERA if g not in seed_genera]
    priority_pool = df[~df["accession"].isin(seed_accessions)]
    priority = select_priority(priority_pool, priority_missing)
    print(f"  Priority fill ({', '.join(priority_missing) if priority_missing else 'none'}): {len(priority)} genome(s)")

    # Step 3: diversity fill
    already_selected = pd.concat([seed, priority], ignore_index=True)
    already_ids = set(already_selected["full_id"])
    diversity_target = args.target_n - len(already_selected)
    if diversity_target <= 0:
        print(f"  Seed + priority already meets target ({len(already_selected)} >= {args.target_n}); skipping diversity fill")
        diversity = pd.DataFrame(columns=df.columns)
    else:
        diversity = select_diversity(
            df,
            exclude_genera=set(PRIORITY_GENERA),
            exclude_full_ids=already_ids,
            target_n=diversity_target,
            max_per_genus=5,
        )
    print(f"  Diversity fill: {len(diversity)} genomes")

    all_selected = pd.concat([already_selected, diversity], ignore_index=True)
    print(f"  Total selected: {len(all_selected)} genomes")

    write_outputs(all_selected, existing_dirs, args.output_manifest, args.output_list)


if __name__ == "__main__":
    main()
