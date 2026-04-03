#!/usr/bin/env python3
"""HAMP-centric sensor-swap candidate finder.

Core insight (Peruzzi 2023, Fig. 4B):
  The chimera crossover point is the N-x-[ML] motif that begins HAMP-AS1.
  This is NOT the same as HMMER's ali_from for the HAMP Pfam domain — the Pfam
  HAMP profile (PF00672) starts from TM2, placing ali_from ~30-40 residues
  upstream of the actual AS-1 junction. Identity computed on the full Pfam HAMP
  hit conflates TM2 diversity with AS-1 conservation, producing spuriously low
  similarity between known-compatible chimera partners (NarX/CusS: 27% on full
  HAMP hit, but ~50% on the junction-centered window).

Selection logic:
  1. For each HAMP-containing protein, find the N-x-[ML] motif that marks the
     AS-1 helix start in a search window around HMMER HAMP_start.
  2. Extract a ±JUNCTION_WINDOW residue stretch centered on that N.
  3. Use Biopython PairwiseAligner (BLOSUM62) for all-vs-all junction identity —
     correct for short 50-aa sequences where MMseqs2 k-mers miss twilight-zone hits.
  4. Retain pairs: junction identity >= HAMP_ID_THRESHOLD (default 35%).
  5. Extract sensor domains (residues 1..N_junction-1) and compare.
  6. Keep pairs where sensor_identity < junction_identity (sensor is the variable
     element, HAMP junction is the conserved element).

Phase matching:
  N_junction mod 7 gives the heptad register of the sensor→HAMP junction.
  Phase-matched pairs are spliced at equivalent coiled-coil positions and
  require no heptad correction (Hatstat 2025).

Ground truth validation:
  NarX/CusS, NarX/RssA, NarX/VanS (Peruzzi 2023) should appear at ~35-55%
  junction identity. NarX/NrsS requires NrsS (cyanobacterial, absent from
  our genome set) and will not appear.
"""

import argparse
import os
import re
import tempfile
from pathlib import Path

import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HAMP_ID_THRESHOLD = 35.0    # % junction identity — permissive to recover ground truths
JUNCTION_WINDOW   = 25      # residues on each side of N for comparison (50 aa total)
MOTIF_SEARCH_PAD  = 60      # search for N-x-[ML] up to this many residues past HAMP_start
SENSOR_ID_MAX     = None    # no upper cap on sensor identity (any divergence qualifies)


# ---------------------------------------------------------------------------
# FASTA helpers
# ---------------------------------------------------------------------------

def parse_hamp_fasta(hamp_fasta: str) -> dict[str, dict]:
    """Return {protein_id: {hamp_start, ext_start, seq}} from hamp_linker_regions.faa.

    Header format: >WP_xxx.1_HAMP_253 HAMP_start=253 extracted_residues=223-302
    """
    records = {}
    current_id = None
    current_seq = []
    with open(hamp_fasta) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id:
                    records[current_id]["seq"] = "".join(current_seq)
                parts = line[1:].split()
                protein_id = parts[0].rsplit("_HAMP_", 1)[0]
                hamp_start = int(parts[1].split("=")[1])
                m = re.search(r"extracted_residues=(\d+)", line)
                ext_start = int(m.group(1)) if m else max(1, hamp_start - 30)
                current_id = protein_id
                records[protein_id] = {
                    "hamp_start": hamp_start,
                    "ext_start":  ext_start,
                }
                current_seq = []
            else:
                current_seq.append(line)
    if current_id:
        records[current_id]["seq"] = "".join(current_seq)
    return records


def parse_fasta_index(fasta_path: str) -> dict[str, str]:
    """Return {header_id: sequence}."""
    seqs = {}
    current_id = None
    current_seq = []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current_id:
                    seqs[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id:
        seqs[current_id] = "".join(current_seq)
    return seqs


def write_fasta(records: dict[str, str], path: str) -> None:
    with open(path, "w") as fh:
        for pid, seq in records.items():
            fh.write(f">{pid}\n{seq}\n")


# ---------------------------------------------------------------------------
# Junction detection: find the N-x-[ML] motif that begins HAMP-AS1
# ---------------------------------------------------------------------------

# Full Peruzzi Fig. 4B crossover motif — 5 positions:
#   pos 1: N  — fully conserved
#   pos 2: x  — variable; NOT D/E (acidic); Peruzzi shows N,T,K,H,T across 5 donors
#               Acidic residues at pos2 indicate spurious match (e.g. NEM in NarX TM2)
#   pos 3: M or L — strongly conserved hydrophobic anchor of HAMP-AS1
#   pos 4: L, V, or I — strongly conserved (first heptad hydrophobic core)
#   pos 5: L, I, V, or F — partially conserved (second heptad position)
#
# Detection: use the 3-aa anchor (N[^DE][ML]) to find the junction.
# After finding the junction, score how many of the extended positions match.
_MOTIF_RE = re.compile(r"N[^DE]([ML])")

# Extended pattern for scoring positions 4 and 5 downstream of the N-x-[ML] anchor
_MOTIF_POS4 = frozenset("LVI")
_MOTIF_POS5 = frozenset("LIVF")


def score_motif(seq: str, junction_pos: int) -> int:
    """Score the Peruzzi 5-residue AS-1 crossover motif (0–5).

    junction_pos is 0-indexed position of N in the full sequence.
    Returns count of positions matching the Peruzzi consensus (max 5):
      pos1=N, pos2=not D/E, pos3=M/L, pos4=L/V/I, pos5=L/I/V/F
    """
    score = 0
    for i, (aa, check) in enumerate([
        (seq[junction_pos]     if junction_pos     < len(seq) else "", lambda a: a == "N"),
        (seq[junction_pos + 1] if junction_pos + 1 < len(seq) else "", lambda a: a not in "DE"),
        (seq[junction_pos + 2] if junction_pos + 2 < len(seq) else "", lambda a: a in "ML"),
        (seq[junction_pos + 3] if junction_pos + 3 < len(seq) else "", lambda a: a in _MOTIF_POS4),
        (seq[junction_pos + 4] if junction_pos + 4 < len(seq) else "", lambda a: a in _MOTIF_POS5),
    ]):
        if aa and check(aa):
            score += 1
    return score


def find_junction(protein_id: str,
                  hamp_info: dict,
                  full_seqs: dict) -> int | None:
    """Return 0-indexed position of N in N-x-[ML] within the full protein.

    Searches the extracted HAMP region (from ext_start to ext_start+len(seq))
    for the Peruzzi crossover motif. Returns the FIRST match at or after
    HAMP_start-5 (allows for small HMMER boundary jitter).

    Returns None if motif not found in the search window.
    """
    info = hamp_info.get(protein_id)
    if info is None or protein_id not in full_seqs:
        return None

    full_seq   = full_seqs[protein_id]
    hamp_start = info["hamp_start"]      # 1-indexed in full protein
    # Convert to 0-indexed and define search window
    search_from = max(0, hamp_start - 6)                    # a few residues before
    search_to   = min(len(full_seq), hamp_start + MOTIF_SEARCH_PAD)
    window      = full_seq[search_from:search_to]

    m = _MOTIF_RE.search(window)
    if m is None:
        return None
    # Return 0-indexed position in full_seq
    return search_from + m.start()


# ---------------------------------------------------------------------------
# Pairwise alignment for short junction sequences
# ---------------------------------------------------------------------------

def _make_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score    = -10.0
    aligner.extend_gap_score  = -0.5
    aligner.mode              = "global"
    return aligner


def pct_identity(s1: str, s2: str, aligner: PairwiseAligner) -> float:
    """Global BLOSUM62 alignment identity over aligned length.

    Parses Biopython PairwiseAligner output format:
      target  0 SEQUENCE 9
                0 |||.||.. 9
      query   0 SEQUENCE 9
    """
    if not s1 or not s2:
        return 0.0
    aln = aligner.align(s1, s2)[0]
    lines = str(aln).strip().split("\n")
    # Lines: target row, match row, query row — each group of 3
    target_chars, match_chars, query_chars = [], [], []
    i = 0
    while i < len(lines):
        if lines[i].startswith("target"):
            t_seq = lines[i].split()[2] if len(lines[i].split()) >= 3 else ""
            m_seq = lines[i + 1].strip() if i + 1 < len(lines) else ""
            q_seq = lines[i + 2].split()[2] if (i + 2 < len(lines)
                                                 and len(lines[i + 2].split()) >= 3) else ""
            target_chars.append(t_seq)
            # match row may have leading spaces — strip to content length
            match_chars.append(m_seq[-len(t_seq):] if len(m_seq) >= len(t_seq) else m_seq)
            query_chars.append(q_seq)
            i += 3
        else:
            i += 1
    t = "".join(target_chars)
    q = "".join(query_chars)
    matches = sum(a == b for a, b in zip(t, q) if a not in ("-", "X") and b not in ("-", "X"))
    aln_len = sum(1 for a, b in zip(t, q) if not (a == "-" and b == "-"))
    return 100.0 * matches / aln_len if aln_len > 0 else 0.0


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

def filter_swap_candidates(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Keep pairs with high junction identity and any sensor divergence.

    Strict HAMP junction conservation (>= HAMP_ID_THRESHOLD) with any sensor
    divergence (sensor_identity < junction_identity). No minimum floor on
    sensor divergence — structure prediction (AlphaFold + Hatstat features)
    performs the final functional filter.

    Phase-matched pairs (same heptad register at N_junction) are sorted first
    as they require no heptad correction at the chimera junction.
    """
    return pairs_df[
        (pairs_df["junction_identity"] >= HAMP_ID_THRESHOLD) &
        (pairs_df["sensor_identity"]   <  pairs_df["junction_identity"])
    ].copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hamp_fasta",   required=True)
    parser.add_argument("--hk_reps",      required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--hamp_threshold", type=float, default=HAMP_ID_THRESHOLD)
    parser.add_argument("--junction_window", type=int, default=JUNCTION_WINDOW)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    aligner = _make_aligner()

    # ------------------------------------------------------------------
    # 1. Parse inputs
    # ------------------------------------------------------------------
    print(f"Parsing HAMP metadata from {args.hamp_fasta} ...")
    hamp_info = parse_hamp_fasta(args.hamp_fasta)
    print(f"  {len(hamp_info)} proteins with HAMP domain")

    print(f"Indexing full HK sequences from {args.hk_reps} ...")
    full_seqs = parse_fasta_index(args.hk_reps)

    # ------------------------------------------------------------------
    # 2. Find N-x-[ML] junction position for each protein
    # ------------------------------------------------------------------
    print("Locating N-x-[ML] junction motif (Peruzzi 2023 crossover point) ...")
    junctions = {}       # protein_id -> 0-indexed junction position in full_seq
    junction_seqs = {}   # protein_id -> junction-centered ~50aa sequence

    W = args.junction_window
    motif_scores = {}   # protein_id -> Peruzzi 5-position motif score (0-5)
    for pid in hamp_info:
        j = find_junction(pid, hamp_info, full_seqs)
        if j is None:
            continue
        full = full_seqs[pid]
        # Extract ±W window; pad with X if near protein termini
        # +5 captures the full 5-residue Peruzzi motif (N-x-[ML]-[LVI]-[LIVF])
        left  = full[max(0, j - W) : j]
        right = full[j : j + W + 5]
        if len(left) < W:
            left = "X" * (W - len(left)) + left
        junctions[pid]     = j
        junction_seqs[pid] = left + right
        motif_scores[pid]  = score_motif(full, j)

    n_with_motif = len(junctions)
    n_without    = len(hamp_info) - n_with_motif
    print(f"  {n_with_motif} proteins with N-x-[ML] motif found "
          f"({n_without} skipped — motif absent in search window)")

    # ------------------------------------------------------------------
    # 3. All-vs-all pairwise junction alignment
    # ------------------------------------------------------------------
    print(f"Computing all-vs-all junction identity "
          f"({n_with_motif}x{n_with_motif} = {n_with_motif**2:,} pairs) ...")
    pids = list(junctions)

    rows = []
    for i, a in enumerate(pids):
        if i % 50 == 0:
            print(f"  {i}/{len(pids)} proteins processed ...", end="\r")
        for b in pids[i + 1:]:
            jid = pct_identity(junction_seqs[a], junction_seqs[b], aligner)
            if jid < args.hamp_threshold:
                continue
            phase_a = junctions[a] % 7
            phase_b = junctions[b] % 7
            rows.append({
                "acceptor":          a,
                "donor":             b,
                "junction_identity": round(jid, 1),
                "junction_pos_a":    junctions[a] + 1,   # 1-indexed
                "junction_pos_b":    junctions[b] + 1,
                "phase_a":           phase_a,
                "phase_b":           phase_b,
                "phase_match":       phase_a == phase_b,
                # 3-aa anchor (N-x-[ML]) for quick inspection
                "motif_a":           junction_seqs[a][W:W+3],
                "motif_b":           junction_seqs[b][W:W+3],
                # Full 5-aa Peruzzi crossover signature
                "motif5_a":          junction_seqs[a][W:W+5],
                "motif5_b":          junction_seqs[b][W:W+5],
                # Motif score (0-5): how many Peruzzi consensus positions match
                "motif_score_a":     motif_scores.get(a, 0),
                "motif_score_b":     motif_scores.get(b, 0),
            })
    print(f"\n  {len(rows)} pairs pass junction identity >= {args.hamp_threshold}%")

    if not rows:
        print("No pairs found. Try lowering --hamp_threshold.")
        pd.DataFrame().to_csv(args.output, sep="\t", index=False)
        return

    pairs = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 4. Sensor domain comparison for retained pairs
    # ------------------------------------------------------------------
    proteins_needed = set(pairs["acceptor"]) | set(pairs["donor"])
    sensor_seqs = {
        pid: full_seqs[pid][: junctions[pid]]          # 0..junction-1
        for pid in proteins_needed
        if pid in full_seqs and junctions[pid] > 20    # skip implausibly short sensors
    }
    print(f"Computing sensor identity for {len(pairs)} pairs ...")

    sensor_identity = []
    for _, row in pairs.iterrows():
        a, b = row["acceptor"], row["donor"]
        sa = sensor_seqs.get(a, "")
        sb = sensor_seqs.get(b, "")
        if sa and sb:
            # Align only last 100 aa of sensor (near-HAMP region most informative)
            sid = pct_identity(sa[-100:], sb[-100:], aligner)
        else:
            sid = float("nan")
        sensor_identity.append(round(sid, 1))
    pairs["sensor_identity"] = sensor_identity

    # ------------------------------------------------------------------
    # 5. Filter and rank
    # ------------------------------------------------------------------
    candidates = filter_swap_candidates(pairs)
    # Composite motif score: sum of both partners (max 10)
    candidates["motif_score"] = candidates["motif_score_a"] + candidates["motif_score_b"]

    print(f"\nSwap candidates after filter: {len(candidates)}")
    print(f"  Phase-matched:          {candidates['phase_match'].sum()}")
    print(f"  Full motif (score=10):  {(candidates['motif_score'] == 10).sum()}")
    print(f"  Extended motif (≥8):    {(candidates['motif_score'] >= 8).sum()}")

    # Sort: phase-matched > full/extended motif score > junction identity
    candidates = candidates.sort_values(
        ["phase_match", "motif_score", "junction_identity"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    candidates.to_csv(args.output, sep="\t", index=False)
    print(f"\nWritten: {args.output}")
    cols = ["acceptor", "donor", "junction_identity", "sensor_identity",
            "phase_match", "motif5_a", "motif5_b", "motif_score",
            "junction_pos_a", "junction_pos_b"]
    print(candidates[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
