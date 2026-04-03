#!/usr/bin/env python3
"""Build chimeric RR sequences to complete orthogonal TCS phosphorelay pairs.

The Peruzzi 2023 design uses chimeric RRs of the form:
    RR_A[receiver] + RR_B[DBD]

where RR_A is the cognate RR for the chimeric SHK (so it gets phosphorylated
correctly) and RR_B contributes its DNA-binding domain (DBD) to drive a
desired reporter promoter.

The result is a fully rewired TCS:
    New sensor (acceptor SHK) → chimeric SHK kinase → chimeric RR
                               → chimeric RR DBD activates target promoter

This is distinct from the SHK chimera pipeline, which only rewires INPUT
specificity. The RR chimera rewires OUTPUT specificity (which promoter is
activated). Together, they form a complete orthogonal phosphorelay.

Receiver-DBD boundary positions (literature-curated):
  OmpR/PhoB family  — split at residue ~125 (alpha4-beta5-alpha5 linker,
                       Miyatake et al. 2000; Kottur et al. 2023)
  NarL/FixJ family  — split at residue ~130 (helix-turn-helix linker,
                       Maris et al. 2002; Peruzzi et al. 2023 uses NarL)
  LuxR-solo family  — not recommended for swapping (linker poorly defined)

Input: results/chimera_targets/chimera_candidates.tsv (from identify_chimera_targets)
       which already contains RR_DBD_swap candidates from identify_chimera_targets.py.

Output: results/chimera_design/rr_chimera_sequences.faa
        results/chimera_design/rr_chimera_design_manifest.tsv
"""

import argparse
import re
from pathlib import Path

import pandas as pd


# Literature-curated receiver-DBD boundary per DBD family (1-indexed residue
# that starts the DBD). These are conservative midpoints; the actual cut site
# can be shifted ±5 residues based on sequence alignment to the structural
# template without changing function (Schmidl et al. 2019).
DBD_BOUNDARY: dict[str, int] = {
    "OmpR_PhoB": 125,
    "NarL_FixJ": 130,
}


def load_fasta(path: str) -> dict[str, str]:
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


def rr_chimera_id(receiver_rr: str, dbd_rr: str) -> str:
    def shorten(pid: str) -> str:
        m = re.match(r"[A-Z]+_(\d+)\.", pid)
        return m.group(1)[-6:] if m else pid[:8]
    return f"RRC_{shorten(receiver_rr)}_{shorten(dbd_rr)}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--candidates",   required=True,
                        help="results/chimera_targets/chimera_candidates.tsv "
                             "(contains RR_DBD_swap rows)")
    parser.add_argument("--rr_reps",      required=True,
                        help="results/representatives/rr_reps.faa")
    parser.add_argument("--shk_manifest", default="",
                        help="results/chimera_design/chimera_design_manifest.tsv "
                             "(optional: pair RR chimeras with matching SHK chimeras)")
    parser.add_argument("--outdir",       required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load RR DBD-swap candidates
    # ------------------------------------------------------------------
    df = pd.read_csv(args.candidates, sep="\t")
    rr_cands = df[df["chimera_type"] == "RR_DBD_swap"].copy()
    print(f"Loaded {len(rr_cands)} RR_DBD_swap candidates from {args.candidates}")

    if rr_cands.empty:
        print("No RR_DBD_swap candidates found. "
              "Ensure identify_chimera_targets has been run.")
        return

    rr_seqs = load_fasta(args.rr_reps)
    print(f"Loaded {len(rr_seqs)} RR sequences")

    # ------------------------------------------------------------------
    # 2. Optional: load SHK chimera manifest to pair RR and SHK chimeras
    # ------------------------------------------------------------------
    shk_pairs: dict[str, str] = {}   # kinase_protein_id → chimera_id
    if args.shk_manifest and Path(args.shk_manifest).exists():
        shk_df = pd.read_csv(args.shk_manifest, sep="\t")
        for _, row in shk_df.iterrows():
            # donor of SHK chimera = the kinase core donor; pair by donor protein
            shk_pairs[row["donor"]] = row["chimera_id"]
        print(f"Loaded {len(shk_pairs)} SHK chimera pairings from {args.shk_manifest}")

    # ------------------------------------------------------------------
    # 3. Build chimeric RR sequences
    # ------------------------------------------------------------------
    rows: list[dict] = []
    faa_lines: list[str] = []
    skipped = 0

    for _, row in rr_cands.iterrows():
        # receiver_rr: RR whose receiver domain accepts phosphorylation from
        #              the cognate SHK (or the chimeric SHK's kinase donor)
        # dbd_rr: RR whose DBD targets the desired output promoter
        # In the candidates TSV, protein_id is the cluster representative.
        receiver_rr = row["protein_id"]
        dbd_family  = row.get("dbd_family", "")
        boundary    = DBD_BOUNDARY.get(dbd_family, 125)

        seq_receiver = rr_seqs.get(receiver_rr)
        if seq_receiver is None:
            skipped += 1
            continue

        if len(seq_receiver) < boundary + 10:
            print(f"  [skip] {receiver_rr}: sequence too short "
                  f"({len(seq_receiver)} aa) for DBD boundary at {boundary}")
            skipped += 1
            continue

        # For each receiver RR, pair with every other RR of the same DBD family
        # that has a different DBD (different gene name / product). In practice
        # this is a small set; the key target is NarL-YdfI or OmpR-PhoB swaps
        # from characterized systems.
        same_family = rr_cands[
            (rr_cands["dbd_family"] == dbd_family) &
            (rr_cands["protein_id"] != receiver_rr)
        ]

        for _, dbd_row in same_family.iterrows():
            dbd_rr   = dbd_row["protein_id"]
            seq_dbd  = rr_seqs.get(dbd_rr)
            if seq_dbd is None:
                continue

            if len(seq_dbd) < boundary + 10:
                continue

            # Construct chimera: receiver_rr[0:boundary] + dbd_rr[boundary:]
            receiver_part = seq_receiver[:boundary]
            dbd_part      = seq_dbd[boundary:]
            chimera_seq   = receiver_part + dbd_part
            cid           = rr_chimera_id(receiver_rr, dbd_rr)

            # 10-residue junction window
            junc_start = max(0, boundary - 5)
            junc_seq   = chimera_seq[junc_start : junc_start + 10]

            faa_lines.append(
                f">{cid} receiver_rr={receiver_rr} dbd_rr={dbd_rr} "
                f"dbd_family={dbd_family} boundary={boundary} "
                f"len={len(chimera_seq)}"
            )
            faa_lines.append(chimera_seq)

            rec: dict = {
                "rr_chimera_id":  cid,
                "receiver_rr":    receiver_rr,
                "dbd_rr":         dbd_rr,
                "dbd_family":     dbd_family,
                "boundary":       boundary,
                "receiver_len":   len(seq_receiver),
                "dbd_rr_len":     len(seq_dbd),
                "chimera_len":    len(chimera_seq),
                "junction_seq":   junc_seq,
                "receiver_gene":  row.get("gene", ""),
                "dbd_gene":       dbd_row.get("gene", ""),
                "receiver_product": row.get("product", ""),
                "dbd_product":    dbd_row.get("product", ""),
            }
            # Link to corresponding SHK chimera if available
            rec["paired_shk_chimera"] = shk_pairs.get(receiver_rr, "")
            rows.append(rec)

    # ------------------------------------------------------------------
    # 4. Write outputs
    # ------------------------------------------------------------------
    faa_path = outdir / "rr_chimera_sequences.faa"
    faa_path.write_text("\n".join(faa_lines) + "\n")
    print(f"\nWrote {len(rows)} RR chimeric sequences → {faa_path}")

    manifest_path = outdir / "rr_chimera_design_manifest.tsv"
    pd.DataFrame(rows).to_csv(manifest_path, sep="\t", index=False)
    print(f"Wrote RR design manifest → {manifest_path}")
    if skipped:
        print(f"  Skipped {skipped} candidates (sequence not found / too short)")

    print("\n=== RR chimera design summary ===")
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        print(df_out[["rr_chimera_id", "receiver_gene", "dbd_gene",
                      "dbd_family", "chimera_len", "junction_seq",
                      "paired_shk_chimera"]].to_string(index=False))

        print("\n=== Paired SHK+RR chimera systems ===")
        paired = df_out[df_out["paired_shk_chimera"] != ""]
        if paired.empty:
            print("  No SHK pairings found. Run with --shk_manifest to link systems.")
        else:
            print(paired[["paired_shk_chimera", "rr_chimera_id",
                           "receiver_gene", "dbd_gene"]].to_string(index=False))


if __name__ == "__main__":
    main()
