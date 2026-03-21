#!/usr/bin/env python3
"""Prepare and (optionally) run AlphaFold3 protein-DNA jobs for RR DBD + promoter screening.

For each top RR DBD-swap candidate × recommended promoter pair:
  1. Extract C-terminal DBD sequence using family-specific boundary heuristics
  2. Pair with the promoter binding site DNA sequence (double-stranded, from literature)
  3. Write AF3 JSON input file (compatible with local run_alphafold.py and web server)
  4. If --run_af3 and local AF3 is installed: execute prediction
  5. If output exists: parse ipTM and pLDDT at DNA-interface residues

DBD boundary heuristics (literature-based):
  OmpR_PhoB: C-terminal ~90 aa (winged-HTH, Krell 2010)
  NarL_FixJ: C-terminal ~65 aa (HTH, Maris 2002)
  NtrC_AAA:  C-terminal ~55 aa (HTH, De Carlo 2006)

Promoter binding site DNA sequences (core motifs, from literature):
  Pho box: Makino 1988 — 20 bp consensus, OmpR/PhoB binding
  OmpR box: Rampersaud 1989 — F1 box, 20 bp
  NarL box: Darwin 1997 — two heptanucleotide half-sites, 20 bp
  NtrC UAS: Reitzer 1989 — glnAp2 UAS element, 19 bp

AF3 ipTM threshold: > 0.5 = confident interaction predicted (Abramson 2024).

Usage (prepare only — upload JSONs to https://alphafoldserver.com):
    python scripts/af3_dbd_promoter.py \
        --candidates results/chimera_targets/chimera_candidates.tsv \
        --promoters  data/reference/characterized_promoters.tsv \
        --rr_fasta   results/representatives/rr_reps.faa \
        --outdir     results/af3_screening \
        --top_n      10

Usage (prepare + run local AF3):
    python scripts/af3_dbd_promoter.py ... --run_af3 --af3_dir ~/alphafold3

Output:
    results/af3_screening/inputs/{pair_id}.json      AF3 input JSON
    results/af3_screening/outputs/{pair_id}/         AF3 output directory
    results/af3_screening/af3_dbd_promoter_scores.tsv  Summary table
"""
import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
from Bio import SeqIO


# ── DBD boundary heuristics ────────────────────────────────────────────────
DBD_CTERMINAL_LENGTH = {
    "OmpR_PhoB": 90,
    "NarL_FixJ": 65,
    "NtrC_AAA":  55,
}

# ── Promoter binding site DNA (sense + antisense, 5'→3') ──────────────────
# Selected core binding sites from literature for use in AF3 protein-dsDNA jobs.
# Sequences are the minimal binding element; extend if longer contact surface needed.
PROMOTER_BINDING_SITES = {
    "PphoA":   ("CTGTCATAAAGCCTGTCATA", "TATGACAGGCTTTATGACAG"),  # Pho box consensus
    "PpstS":   ("CTGTCATAAAGCCTGTCATA", "TATGACAGGCTTTATGACAG"),  # Same Pho box
    "PompF":   ("TGAAACTTTTTTTATGTTCA", "TGAACATAAAAAAGTTTTCA"),  # OmpR box F1
    "PompC":   ("TGAAACTTTTTCTATGTTCA", "TGAACATAGAAAAGTTTCA"),   # OmpR box F1 variant
    "PnarK":   ("TACCCATTTACCCATTTACC", "GGTAAAATGGGTAAATGGGT"),  # NarL box consensus
    "PnarG":   ("TACCCATTTACCCATTTACC", "GGTAAAATGGGTAAATGGGT"),  # NarL box consensus
    "PglnAp2": ("TGCACCATATTTGCACCAT",  "ATGGTGCAAATATGGTGCA"),   # NtrC UAS at glnAp2
    "PttrB":   ("TTTACATAAATGTATCAATA", "TATTGATACATTTATGTAAA"),  # Predicted NarL-like box (Daeffler 2017)
    "PthsA":   ("TTTACATAAATGTATCAATA", "TATTGATACATTTATGTAAA"),  # Predicted OmpR-like box (Daeffler 2017)
}


def extract_dbd(sequence: str, family: str) -> str:
    """Return C-terminal DBD subsequence based on family heuristic."""
    length = DBD_CTERMINAL_LENGTH.get(family, 80)
    return sequence[-length:]


def make_af3_json(pair_id: str, protein_seq: str,
                  dna_sense: str, dna_antisense: str) -> dict:
    """Build AF3 local inference JSON for a protein-dsDNA complex.

    Format: AF3 local inference JSON (Abramson et al. 2024).
    Compatible with both local run_alphafold.py and the web server upload.
    """
    return {
        "name": pair_id,
        "modelSeeds": [42],
        "sequences": [
            {"protein": {"id": "A", "sequence": protein_seq}},
            {"dna": {"id": "B", "sequence": dna_sense}},
            {"dna": {"id": "C", "sequence": dna_antisense}},
        ]
    }


def parse_af3_output(output_dir: Path) -> dict | None:
    """Parse AF3 output directory for ipTM and mean pLDDT.

    AF3 writes: {name}_summary_confidences.json and {name}_confidences.json
    Returns dict with iptm, ptm, mean_plddt, ranking_score. Returns None if missing.
    """
    summary_files = list(output_dir.glob("*_summary_confidences*.json"))
    if not summary_files:
        return None
    with open(summary_files[0]) as f:
        summary = json.load(f)

    mean_plddt = None
    conf_files = list(output_dir.glob("*_confidences*.json"))
    if conf_files:
        with open(conf_files[0]) as f:
            conf = json.load(f)
        plddt_vals = conf.get("plddt", [])
        if plddt_vals:
            mean_plddt = round(sum(plddt_vals) / len(plddt_vals), 2)

    return {
        "iptm":          summary.get("iptm"),
        "ptm":           summary.get("ptm"),
        "mean_plddt":    mean_plddt,
        "ranking_score": summary.get("ranking_score"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates",  required=True)
    parser.add_argument("--promoters",   required=True)
    parser.add_argument("--rr_fasta",    required=True)
    parser.add_argument("--outdir",      required=True)
    parser.add_argument("--top_n",       type=int, default=10)
    parser.add_argument("--run_af3",     action="store_true",
                        help="Run AF3 locally after preparing inputs")
    parser.add_argument("--af3_dir",     default="~/alphafold3",
                        help="Path to local AlphaFold3 installation")
    args = parser.parse_args()

    outdir   = Path(args.outdir)
    indir    = outdir / "inputs"
    outdir_a = outdir / "outputs"
    indir.mkdir(parents=True, exist_ok=True)
    outdir_a.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.candidates, sep="\t")
    sequences  = {r.id: str(r.seq) for r in SeqIO.parse(args.rr_fasta, "fasta")}

    # Select top RR DBD-swap candidates that have characterized promoters
    rr_cands = candidates[
        candidates["chimera_type"] == "RR_DBD_swap"
    ]
    if "has_characterized_promoter" in rr_cands.columns:
        rr_cands = rr_cands[
            rr_cands["has_characterized_promoter"].astype(str).str.lower() == "true"
        ]
    rr_cands = rr_cands.head(args.top_n)

    if rr_cands.empty:
        print("No RR DBD-swap candidates with characterized promoters found.")
        pd.DataFrame().to_csv(outdir / "af3_dbd_promoter_scores.tsv", sep="\t", index=False)
        return

    score_rows = []

    for _, row in rr_cands.iterrows():
        pid           = row["protein_id"]
        family        = row.get("dbd_family", "")
        rec_promoters = str(row.get("recommended_promoters", "")).split(",")

        if pid not in sequences:
            print(f"  [skip] {pid} not found in rr_fasta")
            continue

        dbd_seq = extract_dbd(sequences[pid], family)

        for pname in rec_promoters:
            pname = pname.strip()
            if not pname or pname not in PROMOTER_BINDING_SITES:
                continue
            sense, antisense = PROMOTER_BINDING_SITES[pname]
            pair_id   = f"{pid}__{pname}".replace(".", "_").replace(" ", "_")
            json_path = indir / f"{pair_id}.json"

            job = make_af3_json(pair_id, dbd_seq, sense, antisense)
            with open(json_path, "w") as fh:
                json.dump(job, fh, indent=2)
            print(f"  Written: {json_path}")

            # Parse existing AF3 output if present
            af3_out_dir = outdir_a / pair_id
            scores = parse_af3_output(af3_out_dir) if af3_out_dir.exists() else None

            score_rows.append({
                "protein_id":       pid,
                "dbd_family":       family,
                "promoter":         pname,
                "dbd_length":       len(dbd_seq),
                "dna_length":       len(sense),
                "af3_input_json":   str(json_path),
                "iptm":             scores["iptm"]          if scores else None,
                "ptm":              scores["ptm"]           if scores else None,
                "mean_plddt":       scores["mean_plddt"]    if scores else None,
                "ranking_score":    scores["ranking_score"] if scores else None,
                "confident_binding": (
                    (scores["iptm"] > 0.5)
                    if (scores and scores.get("iptm") is not None)
                    else None
                ),
            })

            if args.run_af3:
                af3_dir = Path(args.af3_dir).expanduser()
                af3_out_dir.mkdir(exist_ok=True)
                cmd = [
                    "python", str(af3_dir / "run_alphafold.py"),
                    f"--json_path={json_path}",
                    f"--output_dir={outdir_a}",
                    f"--model_dir={af3_dir / 'models'}",
                ]
                print(f"  Running AF3: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)

    results = pd.DataFrame(score_rows)
    out_tsv = outdir / "af3_dbd_promoter_scores.tsv"
    results.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nSaved: {out_tsv} ({len(results)} DBD-promoter pairs)")

    if not results.empty and "confident_binding" in results.columns:
        n_eval = results["iptm"].notna().sum()
        n_conf = results["confident_binding"].eq(True).sum()
        if n_eval > 0:
            print(f"Confident binding (ipTM > 0.5): {n_conf}/{n_eval} evaluated pairs")
        else:
            print("AF3 outputs not yet available. Upload JSONs from results/af3_screening/inputs/ to:")
            print("  https://alphafoldserver.com  or run locally with --run_af3")


if __name__ == "__main__":
    main()
