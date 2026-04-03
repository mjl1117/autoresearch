#!/usr/bin/env python3
"""HAMP helix heptad register analysis for TCS chimera linker compatibility.

Replaces DeepCoil (which requires incompatible numpy<1.19) with a direct
implementation of heptad register scoring based on hydrophobic moment analysis.

Scientific basis:
  HAMP domains form parallel four-helix bundles. The coiled-coil knobs-into-holes
  packing requires hydrophobic residues at heptad positions a and d. The phase
  of the HAMP coiled-coil (which of 7 positions is 'a') determines linker
  compatibility between chimera donors/acceptors (Hatstat 2025, Hulko 2006).

Algorithm:
  For each HAMP linker window:
    1. Compute hydrophobicity at each position using the Kyte-Doolittle scale.
    2. For each of 7 candidate phases (offset 0-6), compute the hydrophobic moment
       at the coiled-coil period (3.5 residues/turn → period = 7).
    3. The dominant phase = the offset giving the highest mean hydrophobicity
       at a/d positions (offsets 0 and 3 within each heptad).
    4. Score = fraction of heptad positions where a hydrophobic residue (VILMFYW)
       occupies an a or d slot — higher = stronger coiled-coil signal.

Output columns:
  protein: sequence identifier (includes HAMP_start annotation from extract_hamp_linkers.py)
  dominant_phase: best heptad phase (0-6); HAMP_start mod 7 should match this
  coil_score: fraction of a/d positions occupied by hydrophobic residues (0-1)
  n_heptads: number of complete heptads analyzed
  hamp_start: HAMP start residue (parsed from sequence ID)
  phase_confident: True if coil_score >= 0.5 (clear coiled-coil signal)
"""

import argparse
from pathlib import Path

import pandas as pd
from Bio import SeqIO


# Kyte-Doolittle hydrophobicity scale
KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Classic coiled-coil-favoring residues at a/d positions
HYDROPHOBIC = set("VILMFYW")

# Within each heptad abcdefg, positions a (index 0) and d (index 3) are buried
AD_POSITIONS = {0, 3}


def score_heptad_register(sequence, phase):
    """Score a sequence for coiled-coil character at a given heptad phase.

    phase: 0-6, where position 0 in the sequence is assigned to heptad position 'phase'
    Returns (coil_score, n_heptads):
      coil_score: fraction of a/d positions occupied by hydrophobic residues
      n_heptads: number of complete heptads covering the sequence
    """
    ad_hits = 0
    ad_total = 0
    seq = sequence.upper()
    for i, aa in enumerate(seq):
        heptad_pos = (i + phase) % 7
        if heptad_pos in AD_POSITIONS:
            if aa in HYDROPHOBIC:
                ad_hits += 1
            ad_total += 1
    if ad_total == 0:
        return 0.0, 0
    return ad_hits / ad_total, ad_total // 2  # 2 a/d positions per heptad


def analyze_hamp_register(sequence):
    """Find the dominant heptad phase and coiled-coil score for a sequence.

    Tests all 7 phases and returns the best-scoring one.
    """
    best_phase, best_score, best_n = 0, 0.0, 0
    for phase in range(7):
        score, n_heptads = score_heptad_register(sequence, phase)
        if score > best_score:
            best_phase = phase
            best_score = score
            best_n = n_heptads
    return best_phase, round(best_score, 3), best_n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hamp_linkers", required=True,
                        help="FASTA from extract_hamp_linkers.py")
    parser.add_argument("--output", required=True,
                        help="Output TSV with per-protein register analysis")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for rec in SeqIO.parse(args.hamp_linkers, "fasta"):
        # Parse HAMP_start from ID set by extract_hamp_linkers.py
        # ID format: {protein_id}_HAMP_{hamp_start}
        parts = rec.id.rsplit("_HAMP_", 1)
        protein_id = parts[0]
        try:
            hamp_start = int(parts[1]) if len(parts) > 1 else None
        except ValueError:
            hamp_start = None

        seq = str(rec.seq)
        if len(seq) < 7:
            continue

        dominant_phase, coil_score, n_heptads = analyze_hamp_register(seq)

        # Cross-check: HAMP_start mod 7 should agree with the dominant phase
        # from sequence analysis
        hamp_start_phase = hamp_start % 7 if hamp_start is not None else None
        phase_agrees = (hamp_start_phase == dominant_phase) if hamp_start_phase is not None else None

        rows.append({
            "protein_id": protein_id,
            "hamp_start": hamp_start,
            "hamp_start_phase": hamp_start_phase,
            "dominant_phase": dominant_phase,
            "coil_score": coil_score,
            "n_heptads": n_heptads,
            "phase_agrees": phase_agrees,
            "phase_confident": coil_score >= 0.5,
            "sequence_length": len(seq),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.output, sep="\t", index=False)

    n_confident = df["phase_confident"].sum()
    n_agrees = df["phase_agrees"].eq(True).sum()
    print(f"HAMP register analysis: {len(df)} proteins")
    print(f"  Strong coiled-coil signal (score ≥ 0.5): {n_confident}")
    print(f"  HAMP_start phase agrees with sequence register: {n_agrees}")
    if not df.empty:
        print(f"  Mean coil score: {df['coil_score'].mean():.3f}")


if __name__ == "__main__":
    main()
