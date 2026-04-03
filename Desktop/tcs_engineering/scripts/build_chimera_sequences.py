#!/usr/bin/env python3
"""Build chimeric protein sequences from validated HAMP sensor-swap candidates.

For each acceptor/donor pair in the chimera screen, constructs the fusion
protein by concatenating:

    acceptor[1 : junction_pos_a]  +  donor[junction_pos_b : end]

The junction positions define where the N-x-[ML] HAMP AS-1 motif begins.
Acceptor contributes its sensor domain + TM1 + periplasmic loop + TM2 up to
the junction; donor contributes its HAMP domain + DHp + CA kinase core from
the junction onward. The resulting chimera has:

  - NEW input specificity (from the acceptor sensor)
  - ORIGINAL output specificity (donor HAMP + kinase routes signal to donor RR)

Filtering applied before sequence construction:
  - esm_structural_ok (HAMP + kinase pLDDT >= threshold in BOTH parents)
  - seam_ok           (Hatstat 2025 HAMP a/d character compatibility)
  - Optionally: cross_sensor (acceptor and donor sense different inputs)

Output files:
  chimera_sequences.faa    — FASTA of all chimeric sequences
  chimera_design_manifest.tsv — one row per chimera with:
      chimera_id, acceptor, donor, junction_pos_a, junction_pos_b,
      acceptor_len, donor_len, chimera_len, junction_seq (10 aa window),
      esm_structural_ok, seam_ok, phase_match, motif_score,
      acceptor_gene, donor_gene, cross_sensor
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def load_fasta(path: str) -> dict[str, str]:
    """Return {protein_id: sequence} from a FASTA file."""
    seqs: dict[str, str] = {}
    pid: str | None = None
    buf: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if pid:
                    seqs[pid] = "".join(buf)
                pid = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
    if pid:
        seqs[pid] = "".join(buf)
    return seqs


def chimera_id(acceptor: str, donor: str, j_a: int, j_b: int) -> str:
    """Generate a stable, readable ID for a chimera pair."""
    # Shorten WP_XXXXXXXXX.1 → WPXXX for compact IDs
    def shorten(pid: str) -> str:
        m = re.match(r"[A-Z]+_(\d+)\.", pid)
        return m.group(1)[-6:] if m else pid[:8]
    return f"CHI_{shorten(acceptor)}_{shorten(donor)}_j{j_a}_{j_b}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--screen_tsv",   required=True,
                        help="results/chimera_screen/hamp_chimera_screen_esm.tsv")
    parser.add_argument("--hk_reps",      required=True,
                        help="results/representatives/hk_reps.faa")
    parser.add_argument("--outdir",       required=True)
    parser.add_argument("--require_esm",  action="store_true", default=True,
                        help="Only include pairs where esm_structural_ok=True (default: on)")
    parser.add_argument("--require_seam", action="store_true", default=True,
                        help="Only include pairs where seam_ok=True (default: on)")
    parser.add_argument("--cross_sensor_only", action="store_true", default=False,
                        help="Only include pairs where acceptor and donor have "
                             "different sensor types (cross_sensor=True)")
    parser.add_argument("--plddt_min",    type=float, default=70.0)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    screen = pd.read_csv(args.screen_tsv, sep="\t")
    seqs   = load_fasta(args.hk_reps)
    print(f"Loaded {len(screen)} candidate pairs, {len(seqs)} HK sequences")

    # ------------------------------------------------------------------
    # 2. Filter to validated pairs
    # ------------------------------------------------------------------
    mask = pd.Series([True] * len(screen), index=screen.index)

    if args.require_esm and "esm_structural_ok" in screen.columns:
        mask &= screen["esm_structural_ok"].fillna(False)
    elif args.require_esm and "structural_ok" in screen.columns:
        mask &= screen["structural_ok"].fillna(False)

    if args.require_seam and "seam_ok" in screen.columns:
        mask &= screen["seam_ok"].fillna(False)

    if args.cross_sensor_only and "cross_sensor" in screen.columns:
        mask &= screen["cross_sensor"].fillna(False)

    candidates = screen[mask].copy()
    print(f"After filtering: {len(candidates)} pairs pass")
    if len(candidates) == 0:
        print("  No candidates pass filters. Check --require_esm / --require_seam flags.")
        return

    # ------------------------------------------------------------------
    # 3. Build chimeric sequences
    # ------------------------------------------------------------------
    rows = []
    faa_lines: list[str] = []
    skipped = 0

    for _, row in candidates.iterrows():
        a, b  = row["acceptor"], row["donor"]
        j_a   = int(row["junction_pos_a"])   # 1-indexed, start of acceptor HAMP
        j_b   = int(row["junction_pos_b"])   # 1-indexed, start of donor HAMP

        seq_a = seqs.get(a)
        seq_b = seqs.get(b)
        if seq_a is None or seq_b is None:
            print(f"  [skip] {a}/{b}: sequence not found in hk_reps")
            skipped += 1
            continue

        # Chimera: acceptor sensor+TM (0-indexed: [0:j_a-1]) + donor HAMP+kinase ([j_b-1:])
        # junction_pos is 1-indexed; Python slicing is 0-indexed
        sensor_part = seq_a[:j_a - 1]          # up to but NOT including junction residue
        hamp_part   = seq_b[j_b - 1:]          # from junction residue onward (includes N-x-[ML])

        chimera_seq  = sensor_part + hamp_part
        cid          = chimera_id(a, b, j_a, j_b)

        # 10-residue window spanning the junction for inspection
        junc_start = max(0, len(sensor_part) - 5)
        junc_seq   = chimera_seq[junc_start : junc_start + 10]

        faa_lines.append(
            f">{cid} acceptor={a} j_a={j_a} donor={b} j_b={j_b} "
            f"len={len(chimera_seq)}"
        )
        faa_lines.append(chimera_seq)

        rec: dict = {
            "chimera_id":     cid,
            "acceptor":       a,
            "donor":          b,
            "junction_pos_a": j_a,
            "junction_pos_b": j_b,
            "acceptor_len":   len(seq_a),
            "donor_len":      len(seq_b),
            "chimera_len":    len(chimera_seq),
            "junction_seq":   junc_seq,
        }
        # Carry forward key columns from the screen
        for col in ["junction_identity", "sensor_identity", "phase_match",
                    "motif_score", "motif5_a", "motif5_b",
                    "cross_sensor", "acceptor_gene", "donor_gene",
                    "seam_fraction", "seam_ok",
                    "esm_structural_ok",
                    "esm_acceptor_plddt_hamp", "esm_acceptor_plddt_kinase",
                    "esm_donor_plddt_hamp",    "esm_donor_plddt_kinase",
                    "structural_ok",
                    "acceptor_plddt_hamp", "acceptor_plddt_kinase",
                    "donor_plddt_hamp",    "donor_plddt_kinase"]:
            if col in row.index:
                rec[col] = row[col]

        rows.append(rec)

    # ------------------------------------------------------------------
    # 4. Write outputs
    # ------------------------------------------------------------------
    faa_path = outdir / "chimera_sequences.faa"
    faa_path.write_text("\n".join(faa_lines) + "\n")
    print(f"\nWrote {len(rows)} chimeric sequences → {faa_path}")

    manifest_path = outdir / "chimera_design_manifest.tsv"
    pd.DataFrame(rows).to_csv(manifest_path, sep="\t", index=False)
    print(f"Wrote design manifest → {manifest_path}")

    # ------------------------------------------------------------------
    # 5. Write ESMFold compatibility files for fold_chimeras rule
    #
    #    run_esmfold.py expects:
    #      - screen_tsv with "acceptor" and "donor" columns (both = chimera_id)
    #      - hamp_fasta with per-protein HAMP start positions
    #        (header format: ">chimera_id_HAMP_1 hamp_start=N")
    #
    #    The HAMP start in a chimera = junction_pos_a - 1 (0-indexed length of
    #    the acceptor sensor fragment, which is where HAMP begins in the chimera).
    # ------------------------------------------------------------------
    screen_tsv_rows = [{"acceptor": r["chimera_id"], "donor": r["chimera_id"],
                        "junction_pos_a": r["junction_pos_a"] - 1,
                        "junction_pos_b": r["junction_pos_a"] - 1}
                       for r in rows]
    screen_tsv_path = outdir / "chimera_fold_screen.tsv"
    pd.DataFrame(screen_tsv_rows).to_csv(screen_tsv_path, sep="\t", index=False)

    hamp_faa_lines: list[str] = []
    for r in rows:
        cid        = r["chimera_id"]
        hamp_start = r["junction_pos_a"] - 1   # 0-indexed → but hamp_fasta uses 1-indexed
        hamp_start_1idx = hamp_start            # junction_pos_a is already 1-indexed
        hamp_faa_lines.append(f">{cid}_HAMP_{hamp_start_1idx} HAMP_start={hamp_start_1idx}")
        hamp_faa_lines.append("X")              # placeholder sequence (only header is parsed)
    hamp_faa_path = outdir / "chimera_hamp_positions.faa"
    hamp_faa_path.write_text("\n".join(hamp_faa_lines) + "\n")
    print(f"Wrote ESMFold compatibility files → {screen_tsv_path}, {hamp_faa_path}")
    if skipped:
        print(f"  Skipped {skipped} pairs (sequence not in hk_reps)")

    # Summary
    print("\n=== Chimera design manifest ===")
    show = ["chimera_id", "acceptor_gene", "donor_gene", "chimera_len",
            "junction_seq", "phase_match", "motif_score", "seam_fraction",
            "esm_acceptor_plddt_hamp", "esm_donor_plddt_hamp"]
    show = [c for c in show if c in pd.DataFrame(rows).columns]
    print(pd.DataFrame(rows)[show].to_string(index=False))


if __name__ == "__main__":
    main()
