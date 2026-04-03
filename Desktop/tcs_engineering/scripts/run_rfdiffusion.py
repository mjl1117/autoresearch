#!/usr/bin/env python3
"""Run RFDiffusion partial diffusion for TCS chimera linker design.

For each prepared chimera candidate (candidates_for_design.tsv):
  1. Build the RFDiffusion command with contig map and partial_T
  2. Filter designed outputs to keep only designs where linker length ≡ 0 (mod 7)
     relative to the original linker (Hatstat 2025: heptad register critical)
  3. Write per-candidate output directories

RFDiffusion installation:
  git clone https://github.com/RosettaCommons/RFdiffusion
  cd RFdiffusion && pip install -e .
  # Download weights to RFdiffusion/models/

Key parameters (from Hatstat 2025 and RFDiffusion paper):
  partial_T: noise level (0.10–0.20 for conservative linker redesign)
  contig: specifies which residues are fixed vs diffused

Note: RFDiffusion requires a GPU for practical runtimes. CPU mode available
but slow (~30 min/design). M-series Apple Silicon is supported via MPS but
may require the nightly PyTorch build with MPS support.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_rfdiffusion_job(
    rfdiffusion_path, pdb, contig, output_prefix, n_designs, partial_T
):
    """Invoke run_inference.py for one chimera candidate."""
    inference_script = Path(rfdiffusion_path).expanduser() / "run_inference.py"
    if not inference_script.exists():
        raise FileNotFoundError(
            f"RFDiffusion not found at {inference_script}. "
            "Install: https://github.com/RosettaCommons/RFdiffusion"
        )

    cmd = [
        sys.executable,
        str(inference_script),
        f"inference.input_pdb={pdb}",
        f"'contigmap.contigs=[{contig}]'",
        f"inference.output_prefix={output_prefix}",
        f"inference.num_designs={n_designs}",
        f"denoising.partial_T={partial_T}",
        "inference.schedule_directory_path=null",  # use default
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(
        " ".join(cmd),
        shell=True,
        cwd=str(Path(rfdiffusion_path).expanduser()),
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  WARNING: RFDiffusion exited with code {result.returncode}")
    return result.returncode == 0


def filter_designs_by_phase(output_dir, original_linker_length):
    """Keep only designs where new linker length ≡ original mod 7.

    RFDiffusion output PDBs have REMARK lines with design metadata.
    We count residues between the two fixed regions to get designed linker length.
    Returns list of kept PDB paths.
    """
    kept = []
    for pdb in sorted(Path(output_dir).glob("*.pdb")):
        # RFDiffusion PDBs: residue numbering is reset; count all residues
        # in chain A between the two fixed segments (B-factor = 0 for designed)
        designed_residues = []
        with open(pdb) as f:
            for line in f:
                if line.startswith("ATOM") and line[21] == "A":
                    try:
                        bfactor = float(line[60:66].strip())
                        resnum = int(line[22:26].strip())
                        if bfactor < 0.01:  # Designed (not fixed) residues have B=0
                            designed_residues.append(resnum)
                    except ValueError:
                        pass
        if not designed_residues:
            continue
        designed_length = len(set(designed_residues))
        delta = abs(designed_length - original_linker_length)
        if delta % 7 == 0:
            kept.append(pdb)
            print(f"    KEPT {pdb.name}: linker={designed_length} "
                  f"(Δ={delta}, {delta//7} heptads from original)")
        else:
            print(f"    FILTERED {pdb.name}: linker={designed_length} "
                  f"(Δ={delta} — not in heptad register)")
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True,
                        help="candidates_for_design.tsv from prepare_rfdiffusion.py")
    parser.add_argument("--rfdiffusion_path", required=True,
                        help="Path to RFdiffusion repository (contains run_inference.py)")
    parser.add_argument("--n_designs", type=int, default=10)
    parser.add_argument("--partial_T", type=float, default=0.15)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.candidates, sep="\t")
    if candidates.empty:
        print("No candidates to design.")
        return

    summary = []
    for _, row in candidates.iterrows():
        pid = row["protein_id"]
        candidate_dir = Path(args.outdir) / pid
        candidate_dir.mkdir(exist_ok=True)
        output_prefix = str(candidate_dir / "design")

        print(f"\n── {pid} ──")
        print(f"  AF2 PDB:  {row['af2_pdb']}")
        print(f"  Contig:   {row['rfdiffusion_contig']}")
        print(f"  Linker:   {row['linker_length_original']} aa (phase {row['hamp_phase']} mod 7)")

        success = run_rfdiffusion_job(
            rfdiffusion_path=args.rfdiffusion_path,
            pdb=row["af2_pdb"],
            contig=row["rfdiffusion_contig"],
            output_prefix=output_prefix,
            n_designs=args.n_designs,
            partial_T=args.partial_T,
        )

        if success:
            kept = filter_designs_by_phase(str(candidate_dir), int(row["linker_length_original"]))
            summary.append({
                "protein_id": pid,
                "total_designs": args.n_designs,
                "phase_compatible_designs": len(kept),
                "outdir": str(candidate_dir),
            })

    if summary:
        summary_df = pd.DataFrame(summary)
        summary_path = Path(args.outdir) / "rfdiffusion_summary.tsv"
        summary_df.to_csv(summary_path, sep="\t", index=False)
        print(f"\nRFDiffusion summary: {summary_path}")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
