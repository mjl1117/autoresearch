#!/usr/bin/env python3
"""Audit taxonomic composition and well-characterized TCS coverage of the genome set.

This script is the reproducibility checkpoint for genome sampling. It runs as a
required pipeline step and produces a structured report used to:

  1. Document exactly which organisms are in the analysis (reproducibility)
  2. Identify over-represented genera (pseudo-replication risk)
  3. Report coverage of well-characterized TCS systems (scientific validity check)
  4. Recommend additional genomes to improve taxonomic balance

Run at pipeline start before any per-genome processing, and re-run after any
change to the genome set. The output TSVs are versioned alongside results.

Pseudo-replication risk threshold (hard-coded): any genus contributing >10% of
the total genome count is flagged. The current dataset has Pseudomonas (32%) and
Campylobacter (32%) well above this threshold.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from tcs_constants import (
    ASSEMBLY_LEVEL_PRIORITY,
    MAX_GENUS_FRACTION,
    MIN_SPECIES_COUNT,
    RECOMMENDED_GENOMES,
)

ASSEMBLY_LEVELS_OK = {"Complete Genome", "Chromosome"}  # Warn on Scaffold/Contig only


def load_assembly_summary(path):
    """Load NCBI assembly_summary.txt, return DataFrame."""
    return pd.read_csv(
        path, sep="\t", skiprows=2, header=None,
        usecols=[0, 7, 11, 15, 5],
        names=["accession", "organism_name", "assembly_level", "asm_name", "taxid"],
        low_memory=False,
    )


def get_downloaded_genomes(genome_dir):
    """Return list of genome folder names in data/genomes/."""
    return [d for d in os.listdir(genome_dir) if d.startswith("GCF_")]


def match_to_assembly_summary(downloaded, asm_df):
    """Match downloaded genome folders to assembly_summary rows."""
    asm_df = asm_df.copy()
    asm_df["full_id"] = asm_df["accession"] + "_" + asm_df["asm_name"]
    downloaded_set = set(downloaded)
    matched = asm_df[
        asm_df["accession"].isin(downloaded_set) |
        asm_df["full_id"].isin(downloaded_set)
    ].copy()
    matched["genus"] = matched["organism_name"].str.split().str[0]
    matched["species"] = matched["organism_name"].str.split().str[:2].str.join(" ")
    return matched


def report_taxonomic_composition(matched_df, outdir):
    """Write genus/species breakdown TSVs and print summary."""
    n = len(matched_df)

    genus_counts = (
        matched_df.groupby("genus")
        .size()
        .reset_index(name="genome_count")
        .sort_values("genome_count", ascending=False)
    )
    genus_counts["fraction"] = genus_counts["genome_count"] / n
    genus_counts["overrepresented"] = genus_counts["fraction"] > MAX_GENUS_FRACTION

    species_counts = (
        matched_df.groupby("species")
        .size()
        .reset_index(name="genome_count")
        .sort_values("genome_count", ascending=False)
    )

    assembly_level_counts = matched_df["assembly_level"].value_counts().reset_index()
    assembly_level_counts.columns = ["assembly_level", "count"]

    genus_counts.to_csv(outdir / "diversity_genus_counts.tsv", sep="\t", index=False)
    species_counts.to_csv(outdir / "diversity_species_counts.tsv", sep="\t", index=False)
    assembly_level_counts.to_csv(outdir / "diversity_assembly_levels.tsv", sep="\t", index=False)

    print(f"\n{'='*60}")
    print(f"GENOME SET COMPOSITION  (n = {n})")
    print(f"{'='*60}")
    print(f"\nDistinct genera:  {genus_counts['genus'].nunique()}")
    print(f"Distinct species: {species_counts['species'].nunique()}")
    print(f"\nTop genera:")
    print(genus_counts.head(15).to_string(index=False))

    overrep = genus_counts[genus_counts["overrepresented"]]
    if not overrep.empty:
        print(f"\n⚠  PSEUDO-REPLICATION RISK — genera >10% of dataset:")
        for _, row in overrep.iterrows():
            print(f"   {row['genus']:25s}  {row['genome_count']:5d}  ({row['fraction']*100:.1f}%)")
        print(f"   These inflate cluster sizes and phase coherence scores artificially.")
        print(f"   Recommendation: cap at 3 genomes per genus (max ANI diversity).")

    non_ok = matched_df[~matched_df["assembly_level"].isin(ASSEMBLY_LEVELS_OK)]
    if not non_ok.empty:
        print(f"\n⚠  {len(non_ok)} genomes at Scaffold/Contig level — promoter extraction may fail")

    return genus_counts, species_counts


def check_reference_tcs_coverage(reference_tcs_path, hk_annotation_path, rr_annotation_path, outdir):
    """Report which well-characterized TCS systems are detected in DIAMOND output."""
    ref = pd.read_csv(reference_tcs_path, sep="\t")

    results = []
    for path, ptype in [(hk_annotation_path, "HK"), (rr_annotation_path, "RR")]:
        if not Path(path).exists():
            print(f"  Annotation not yet available: {path} (run after DIAMOND)")
            continue
        ann = pd.read_csv(path, sep="\t", header=None,
                          names=["qseqid","sseqid","pident","length",
                                 "qcovhsp","evalue","bitscore","stitle"])
        ann["stitle_lower"] = ann["stitle"].str.lower()

        for _, row in ref.iterrows():
            if ptype == "HK":
                keywords = str(row["hk_swiss_prot_keywords"]).lower().split(";")
            else:
                keywords = str(row["rr_swiss_prot_keywords"]).lower().split(";")
            if keywords == ["null"] or keywords == ["nan"]:
                continue

            hits = ann[ann["stitle_lower"].apply(
                lambda t: any(k.strip() in t for k in keywords)
            )]
            results.append({
                "system": row["system_name"],
                "type": ptype,
                "working_in_user_system": row["working_in_user_system"],
                "signal": row["signal"],
                "dbd_family": row["dbd_family"],
                "n_hits": len(hits),
                "top_pident": hits["pident"].max() if not hits.empty else None,
                "top_hit": hits.iloc[0]["qseqid"] if not hits.empty else None,
            })

    if results:
        cov_df = pd.DataFrame(results)
        cov_df.to_csv(outdir / "reference_tcs_coverage.tsv", sep="\t", index=False)
        detected = cov_df[cov_df["n_hits"] > 0]
        missing = cov_df[cov_df["n_hits"] == 0]
        print(f"\n{'='*60}")
        print(f"WELL-CHARACTERIZED TCS COVERAGE")
        print(f"{'='*60}")
        print(f"  Detected in dataset: {len(detected)} / {len(cov_df)}")
        working_detected = detected[detected["working_in_user_system"] == "yes"]
        print(f"  User working systems detected: {len(working_detected)}")
        if not missing.empty:
            print(f"\n  NOT detected (consider adding genomes):")
            print(missing[["system","signal","dbd_family"]].to_string(index=False))


def check_recommended_genomes(matched_df, outdir):
    """Report which recommended reference organisms are in the dataset."""
    present_orgs = set(matched_df["organism_name"].str.lower())
    present_acc = set(matched_df["accession"])

    rows = []
    for org, acc in RECOMMENDED_GENOMES.items():
        org_lower = org.lower().split()[0]  # match on genus at minimum
        in_dataset = (
            acc in present_acc or
            any(org_lower in o for o in present_orgs)
        )
        rows.append({"organism": org, "accession": acc, "in_dataset": in_dataset})

    rec_df = pd.DataFrame(rows)
    rec_df.to_csv(outdir / "recommended_genomes_status.tsv", sep="\t", index=False)

    missing = rec_df[~rec_df["in_dataset"]]
    print(f"\n{'='*60}")
    print(f"RECOMMENDED REFERENCE ORGANISMS")
    print(f"{'='*60}")
    print(f"  Present: {rec_df['in_dataset'].sum()} / {len(rec_df)}")
    if not missing.empty:
        print(f"\n  Missing (download to improve coverage):")
        for _, r in missing.iterrows():
            print(f"   {r['accession']}   {r['organism']}")
        print(f"\n  Download command (ncbi-datasets-cli):")
        accs = " ".join(missing["accession"].tolist())
        print(f"   datasets download genome accession {accs} --include gff3,protein,genome")

    return rec_df


def write_summary_report(n_genomes, genus_counts, species_counts, rec_df, outdir):
    """Write a single human-readable summary file."""
    lines = [
        "# Genome Set Diversity Audit",
        f"# Generated by audit_species_diversity.py",
        "",
        f"Total genomes:          {n_genomes}",
        f"Distinct genera:        {genus_counts['genus'].nunique()}",
        f"Distinct species:       {species_counts['species'].nunique()}",
        f"Recommended present:    {rec_df['in_dataset'].sum()} / {len(rec_df)}",
        "",
        "## Pseudo-replication risk (genera >10% of dataset)",
    ]
    overrep = genus_counts[genus_counts["overrepresented"]]
    if overrep.empty:
        lines.append("  None — dataset is well-balanced.")
    else:
        for _, row in overrep.iterrows():
            lines.append(f"  {row['genus']:25s}  {row['genome_count']:5d}  ({row['fraction']*100:.1f}%)")
    lines += ["", "## Top 20 genera", genus_counts.head(20).to_string(index=False)]
    report_path = outdir / "diversity_audit_summary.txt"
    report_path.write_text("\n".join(lines))
    print(f"\nFull audit written to: {outdir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genome_dir", required=True, help="data/genomes/")
    parser.add_argument("--assembly_summary", required=True)
    parser.add_argument("--reference_tcs", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--hk_annotation", default=None,
                        help="results/annotation/hk_annotation.tsv (optional; checked if present)")
    parser.add_argument("--rr_annotation", default=None,
                        help="results/annotation/rr_annotation.tsv (optional; checked if present)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    asm_df = load_assembly_summary(args.assembly_summary)
    downloaded = get_downloaded_genomes(args.genome_dir)
    matched = match_to_assembly_summary(downloaded, asm_df)

    if len(matched) == 0:
        print("ERROR: No genomes matched between data/genomes/ and assembly_summary.txt",
              file=sys.stderr)
        sys.exit(1)

    genus_counts, species_counts = report_taxonomic_composition(matched, outdir)
    rec_df = check_recommended_genomes(matched, outdir)

    if args.hk_annotation and args.rr_annotation:
        check_reference_tcs_coverage(
            args.reference_tcs, args.hk_annotation, args.rr_annotation, outdir
        )

    write_summary_report(len(matched), genus_counts, species_counts, rec_df, outdir)

    # Exit non-zero only if a CRITICAL issue is detected: >50% from one genus
    # (not just a warning >10%). This allows the pipeline to proceed but flags severe bias.
    max_frac = genus_counts["fraction"].max()
    if max_frac > 0.50:
        print(f"\nCRITICAL: {genus_counts.iloc[0]['genus']} is {max_frac*100:.0f}% of dataset.",
              file=sys.stderr)
        print("Results will be dominated by one genus. Strongly recommend genome curation.",
              file=sys.stderr)
        # Don't sys.exit(1) — warn but don't block pipeline


if __name__ == "__main__":
    main()
