#!/usr/bin/env python3
"""Cross-species functional analog search for TCS chimera candidates.

Maps uncharacterized or poorly annotated HK/RR proteins to well-studied
analogs in model organisms using DIAMOND BLASTp. This finds the closest
functionally characterized relative even when Swiss-Prot annotation is absent
or generic ("sensor histidine kinase").

Reference proteomes used:
  E. coli K-12 MG1655  (GCF_000005845.2) — TCS gold standard, ~100 TCS pairs
  B. subtilis 168       (GCF_000009045.1) — sporulation + stress response TCS
  P. aeruginosa PAO1    (GCF_000006765.1) — environmental sensing, GacS/GacA etc.

PAO1 is downloaded if not already present; E. coli and B. subtilis are
expected to already be in data/genomes/ (they are part of the core genome set).

Search strategy:
  1. Concatenate model organism proteomes → build DIAMOND database
  2. BLASTp each chimera candidate protein against this reference DB
  3. Report top hit per query with: source organism, gene name, product,
     percent identity, coverage, e-value
  4. Extract gene name from NCBI GFF to label hits (e.g., "narX", "phoQ")
  5. Merge with chimera_functions.tsv to produce final annotated table

Cross-species mapping value:
  Even at low identity (30-40%), structural homologs in model organisms share
  the same signaling mechanism. A 32% hit to NarX in E. coli tells you the
  protein senses nitrate/nitrite via a periplasmic sensor domain — far more
  informative than "sensor histidine kinase". This is essential for predicting
  what stimulus the acceptor/donor responds to and designing the appropriate
  assay.
"""

import argparse
import re
import subprocess
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Reference genome configurations
# ---------------------------------------------------------------------------

REFERENCE_GENOMES = [
    {
        "name":    "Ecoli_K12",
        "ftp":     "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845"
                   "/GCF_000005845.2_ASM584v2",
        "local":   "data/genomes/GCF_000005845.2_ASM584v2",
        "faa":     "proteins.faa",
        "gff":     "genome.gff",
        "org":     "Escherichia coli K-12 MG1655",
    },
    {
        "name":    "Bsubtilis_168",
        "ftp":     "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/009/045"
                   "/GCF_000009045.1_ASM904v1",
        "local":   "data/genomes/GCF_000009045.1_ASM904v1",
        "faa":     "proteins.faa",
        "gff":     "genome.gff",
        "org":     "Bacillus subtilis 168",
    },
    {
        "name":    "Paeruginosa_PAO1",
        "ftp":     "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/006/765"
                   "/GCF_000006765.1_ASM676v1",
        "local":   "data/genomes/GCF_000006765.1_ASM676v1",
        "faa":     "proteins.faa",
        "gff":     "genome.gff",
        "org":     "Pseudomonas aeruginosa PAO1",
    },
]


# ---------------------------------------------------------------------------
# GFF gene name extraction
# ---------------------------------------------------------------------------

def parse_gene_names_from_gff(gff_path: str) -> dict[str, str]:
    """Return {protein_id: gene_name} from a GFF3 file.

    Parses CDS features and extracts Name= or gene= from attributes,
    then matches to Dbxref=GenBank: to get the protein accession.
    """
    gene_map: dict[str, str] = {}
    if not Path(gff_path).exists():
        return gene_map

    with open(gff_path) as fh:
        for line in fh:
            if line.startswith("#") or "\tCDS\t" not in line:
                continue
            attrs = line.split("\t")[-1] if "\t" in line else ""
            # Extract protein_id from Dbxref
            pid_m = re.search(r'protein_id=([^;]+)', attrs)
            if not pid_m:
                continue
            pid = pid_m.group(1).strip()
            # Extract gene name
            gene_m = re.search(r'(?:^|;)gene=([^;]+)', attrs)
            name_m = re.search(r'(?:^|;)Name=([^;]+)', attrs)
            gene = (gene_m.group(1) if gene_m else
                    name_m.group(1) if name_m else "")
            if gene:
                gene_map[pid] = gene.strip()
    return gene_map


# ---------------------------------------------------------------------------
# Reference DB construction
# ---------------------------------------------------------------------------

def build_reference_db(ref_genomes: list[dict], outdir: Path,
                        threads: int = 4) -> tuple[Path, dict[str, dict]]:
    """Concatenate reference proteomes and build DIAMOND database.

    Returns (db_path, protein_metadata) where protein_metadata maps each
    protein_id to {organism, gene}.
    """
    fasta_out = outdir / "reference_proteomes.faa"
    metadata: dict[str, dict] = {}

    with open(fasta_out, "w") as fout:
        for ref in ref_genomes:
            faa = Path(ref["local"]) / ref["faa"]
            if not faa.exists():
                print(f"  [skip] {ref['name']}: {faa} not found")
                continue

            # Parse gene names from GFF if available
            gff = Path(ref["local"]) / ref["gff"]
            gene_map = parse_gene_names_from_gff(str(gff))

            # Write renamed headers: >ORGNAME|protein_id gene_name product
            pid = None
            with open(faa) as fin:
                for line in fin:
                    if line.startswith(">"):
                        pid = line[1:].split()[0]
                        gene = gene_map.get(pid, "")
                        # Tag header with organism for hit parsing
                        new_header = f">{ref['name']}|{pid}"
                        if gene:
                            new_header += f" gene={gene}"
                        fout.write(new_header + "\n")
                        metadata[f"{ref['name']}|{pid}"] = {
                            "organism": ref["org"],
                            "gene":     gene,
                        }
                    else:
                        fout.write(line)

            print(f"  Added {ref['name']}: {len(gene_map)} gene names loaded")

    # Build DIAMOND database
    db_path = outdir / "reference_proteomes.dmnd"
    if not db_path.exists():
        subprocess.run(
            ["diamond", "makedb", "--in", str(fasta_out),
             "--db", str(outdir / "reference_proteomes"), "--threads", str(threads)],
            check=True,
        )
    print(f"  DIAMOND DB: {db_path}")
    return db_path, metadata


# ---------------------------------------------------------------------------
# DIAMOND search
# ---------------------------------------------------------------------------

def run_diamond_search(query_faa: str, db_path: Path,
                       outdir: Path, threads: int = 4,
                       evalue: float = 1e-5,
                       min_pident: float = 25.0) -> Path:
    """Run DIAMOND BLASTp against reference proteomes.

    Uses sensitive mode; reports top hit per query; filters by e-value and
    minimum percent identity to exclude spurious distant matches.
    """
    hits_path = outdir / "reference_analog_hits.m8"
    subprocess.run(
        [
            "diamond", "blastp",
            "--query",           query_faa,
            "--db",              str(db_path),
            "--out",             str(hits_path),
            "--outfmt",          "6",
            "qseqid", "sseqid", "pident", "length",
            "qcovhsp", "evalue", "bitscore", "stitle",
            "--max-target-seqs", "1",
            "--evalue",          str(evalue),
            "--threads",         str(threads),
            "--sensitive",
        ],
        check=True,
    )
    return hits_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--screen_tsv",   required=True,
                        help="results/chimera_screen/chimera_functions.tsv")
    parser.add_argument("--hk_reps",      required=True,
                        help="FASTA of HK representative sequences")
    parser.add_argument("--outdir",       required=True)
    parser.add_argument("--threads",      type=int, default=4)
    parser.add_argument("--min_pident",   type=float, default=25.0,
                        help="Minimum %% identity to report an analog (default: 25)")
    parser.add_argument("--output",       required=True,
                        help="Output TSV (chimera screen enriched with analog info)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Build reference DIAMOND database from model organism proteomes
    # ------------------------------------------------------------------
    print("Building reference proteome DIAMOND database ...")
    db_path, metadata = build_reference_db(REFERENCE_GENOMES, outdir, args.threads)
    print(f"  {len(metadata)} reference proteins indexed")

    # ------------------------------------------------------------------
    # 2. Extract unique chimera candidate sequences into query FASTA
    # ------------------------------------------------------------------
    screen_df = pd.read_csv(args.screen_tsv, sep="\t")
    query_ids = set(screen_df["acceptor"]) | set(screen_df["donor"])

    query_faa = outdir / "chimera_candidates.faa"
    written = 0
    pid = None
    buf: list[str] = []

    def flush(fout):
        nonlocal written
        if pid and pid in query_ids:
            fout.write(f">{pid}\n{''.join(buf)}\n")
            written += 1

    with open(query_faa, "w") as fout, open(args.hk_reps) as fin:
        for line in fin:
            line = line.rstrip()
            if line.startswith(">"):
                flush(fout)
                pid = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        flush(fout)

    print(f"Query FASTA: {written} sequences written")

    # ------------------------------------------------------------------
    # 3. Run DIAMOND search
    # ------------------------------------------------------------------
    print("Running DIAMOND BLASTp against model organism proteomes ...")
    hits_path = run_diamond_search(
        str(query_faa), db_path, outdir, args.threads, min_pident=args.min_pident
    )

    # ------------------------------------------------------------------
    # 4. Parse hits and attach organism + gene annotations
    # ------------------------------------------------------------------
    cols = ["qseqid", "sseqid", "pident", "length",
            "qcovhsp", "evalue", "bitscore", "stitle"]
    hits_df = pd.read_csv(hits_path, sep="\t", header=None, names=cols)
    hits_df = hits_df[hits_df["pident"] >= args.min_pident]

    # Parse organism name and gene from sseqid (format: ORGNAME|protein_id)
    hits_df["ref_org_tag"] = hits_df["sseqid"].str.split("|").str[0]
    hits_df["ref_pid"]     = hits_df["sseqid"].str.split("|").str[1]

    def lookup_meta(row):
        key = f"{row['ref_org_tag']}|{row['ref_pid']}"
        meta = metadata.get(key, {})
        return pd.Series({
            "ref_organism": meta.get("organism", row["ref_org_tag"]),
            "ref_gene":     meta.get("gene", ""),
        })

    meta_cols = hits_df.apply(lookup_meta, axis=1)
    hits_df = pd.concat([hits_df, meta_cols], axis=1)

    # Rename for merging
    hits_df = hits_df.rename(columns={
        "qseqid":    "protein_id",
        "pident":    "analog_pident",
        "qcovhsp":   "analog_qcov",
        "evalue":    "analog_evalue",
        "bitscore":  "analog_bitscore",
        "stitle":    "analog_stitle",
    })[["protein_id", "ref_gene", "ref_organism",
        "analog_pident", "analog_qcov", "analog_evalue", "analog_stitle"]]

    print(f"\n=== Reference analog hits ===")
    print(hits_df[hits_df["ref_gene"] != ""].to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Merge into chimera screen TSV (acceptor + donor separately)
    # ------------------------------------------------------------------
    ann_a = hits_df.rename(columns={c: f"acceptor_{c}" for c in hits_df.columns
                                    if c != "protein_id"}).rename(
                            columns={"protein_id": "acceptor"})
    ann_b = hits_df.rename(columns={c: f"donor_{c}" for c in hits_df.columns
                                    if c != "protein_id"}).rename(
                            columns={"protein_id": "donor"})

    merged = screen_df.merge(ann_a, on="acceptor", how="left") \
                      .merge(ann_b, on="donor", how="left")

    merged.to_csv(args.output, sep="\t", index=False)
    print(f"\nWritten: {args.output}")

    # Summary
    n_acceptor_hit = merged["acceptor_ref_gene"].notna().sum()
    n_donor_hit    = merged["donor_ref_gene"].notna().sum()
    print(f"\n=== Summary ===")
    print(f"  Acceptors with model organism analog: {n_acceptor_hit}/{len(merged)}")
    print(f"  Donors with model organism analog:    {n_donor_hit}/{len(merged)}")

    # Show pairs where both have named analogs
    named = merged[
        merged["acceptor_ref_gene"].notna() & merged["donor_ref_gene"].notna()
    ]
    if len(named):
        print(f"\n=== Pairs with named analogs in both partners ===")
        show_cols = ["acceptor", "donor", "junction_identity",
                     "acceptor_ref_gene", "acceptor_ref_organism",
                     "donor_ref_gene", "donor_ref_organism",
                     "acceptor_analog_pident", "donor_analog_pident"]
        show_cols = [c for c in show_cols if c in named.columns]
        print(named[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
