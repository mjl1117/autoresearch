#!/usr/bin/env python3
"""Download AlphaFold2 predicted structures from EBI for chimera candidate proteins.

Uses DIAMOND best hits (Swiss-Prot sseqid) to extract UniProt accessions,
then fetches PDB files from the EBI AlphaFold database:
  https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb

pLDDT per-residue confidence scores in the B-factor column provide an
independent measure of HAMP helix structural confidence — useful for
identifying whether the linker region is well-predicted (pLDDT > 70)
or disordered (pLDDT < 50), which informs chimera design feasibility.
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests


def extract_uniprot_id(sseqid):
    """Extract UniProt accession from Swiss-Prot sseqid.

    Handles formats like:
      sp|P23837|PHOQ_ECOLI  →  P23837
      P23837                →  P23837
    """
    if pd.isna(sseqid):
        return None
    m = re.match(r"sp\|(\w+)\|", str(sseqid))
    if m:
        return m.group(1)
    # Bare accession fallback
    if re.match(r"^[A-Z][0-9][A-Z0-9]{3}[0-9]$", str(sseqid)):
        return str(sseqid)
    return None


def download_af2_structure(uniprot_id, outdir, version=4, retries=2):
    """Download AF2 PDB from EBI AlphaFold database. Returns True on success."""
    outfile = Path(outdir) / f"{uniprot_id}.pdb"
    if outfile.exists() and outfile.stat().st_size > 0:
        return True
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v{version}.pdb"
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                outfile.write_text(resp.text)
                return True
            elif resp.status_code == 404:
                return False  # Structure not in AF2 DB, no point retrying
        except requests.RequestException:
            pass
        if attempt < retries:
            time.sleep(2 ** attempt)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chimera_candidates", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max_structures", type=int, default=150,
                        help="Download AF2 structures for top N candidates only")
    parser.add_argument("--af2_version", type=int, default=4)
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.chimera_candidates, sep="\t")
    df["uniprot_id"] = df["best_hit"].apply(extract_uniprot_id)

    # Prioritise HK_sensor_swap (largest clusters first), then RR_DBD_swap
    df_sorted = df.sort_values(
        ["chimera_type", "cluster_size"], ascending=[True, False]
    )
    uniprot_ids = (
        df_sorted["uniprot_id"].dropna().unique()[: args.max_structures]
    )

    print(f"Downloading AF2 structures for {len(uniprot_ids)} UniProt entries...")

    downloaded, failed = 0, []
    for uid in uniprot_ids:
        success = download_af2_structure(uid, args.outdir, version=args.af2_version)
        if success:
            downloaded += 1
        else:
            failed.append(uid)
        time.sleep(0.1)  # Respect EBI rate limits

    print(f"Downloaded: {downloaded} / {len(uniprot_ids)}")
    if failed:
        print(f"Not in AF2 DB ({len(failed)}): {failed[:10]}")

    # Write manifest linking chimera candidates → AF2 PDB paths
    manifest = df_sorted[
        ["protein_id", "best_hit", "uniprot_id", "chimera_type", "cluster_size",
         "linker_phase_compatible", "linker_validation_required"]
    ].dropna(subset=["uniprot_id"]).drop_duplicates(subset=["uniprot_id"])

    manifest["af2_pdb"] = manifest["uniprot_id"].apply(
        lambda uid: str(Path(args.outdir) / f"{uid}.pdb")
        if (Path(args.outdir) / f"{uid}.pdb").exists()
        else None
    )
    manifest_path = Path(args.outdir) / "af2_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    print(f"Manifest: {manifest_path} ({len(manifest)} entries, "
          f"{manifest['af2_pdb'].notna().sum()} with structures)")


if __name__ == "__main__":
    main()
