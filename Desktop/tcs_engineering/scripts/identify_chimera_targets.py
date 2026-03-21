#!/usr/bin/env python3
"""Identify candidate TCS chimera targets from cluster, annotation, operon, and homology data.

Design criteria derived from:
  - Skerker et al. 2008 (Cell): DHp helix-1 specificity residues; 7-residue segment
    swap sufficient to fully rewire HK→RR specificity.
  - Schmidl et al. 2019 (Nat Chem Biol): RR DBD swap at REC-DBD linker boundary
    (OmpR family: ~position 122-137; NarL family: equivalent linker); 1,300-fold
    activation; works across gram+/gram-.
  - Peruzzi et al. 2023 (PNAS): HK sensor domain swap retaining second-half HAMP +
    DHp + CA; donor selected by HAMP alignment; 3/4 chimeras functional.
  - Hatstat et al. 2025 (JACS): Linker phase and length are critical for transmembrane
    signal fidelity; sensor domain stability is a prerequisite.

Two chimera strategies implemented:
  A) HK SENSOR SWAP: identify HK pairs where kinase cores (DHp+CA) cluster together
     but sensor domains differ → same core, different input = natural chimera template.
  B) RR DBD SWAP: identify RRs by DBD family (OmpR vs NarL/FixJ) using annotation
     → known modular junction available for rewiring output promoter specificity.

Bioinformatics linker phase validation (Hatstat 2025):
  HAMP domain start position (from HMMER domtblout) mod 7 gives the heptad register.
  Within each conserved-core cluster we compute the dominant phase and within-cluster
  coherence (fraction of members sharing the dominant register). Candidates where the
  individual protein's HAMP phase matches the cluster dominant phase are flagged
  linker_phase_compatible=True. Outliers or proteins without HAMP boundaries are
  flagged linker_validation_required=True.

Output: results/chimera_targets/chimera_candidates.tsv
"""

import argparse
import pandas as pd
from pathlib import Path


# ─── Annotation keywords ────────────────────────────────────────────────────

OMP_R_KEYWORDS = [
    "ompr", "phob", "phop", "pmra", "cpxr", "rsca", "bvga", "kppe", "baer",
    "kdpe", "rcsb", "rsta", "rstb", "phob", "phoq", "envz"
]
NARL_KEYWORDS = [
    "narl", "narP", "fixj", "uvry", "gacA", "chea", "regA", "agmr",
    "flgr", "nara", "narq"
]
CHEY_KEYWORDS = ["chey", "chemotaxis", "chev"]


def classify_rr_family(stitle):
    """Classify RR into DBD family based on best-hit annotation."""
    t = stitle.lower()
    if any(k in t for k in CHEY_KEYWORDS):
        return "CheY_standalone"
    if any(k in t for k in OMP_R_KEYWORDS):
        return "OmpR_PhoB"
    if any(k in t for k in NARL_KEYWORDS):
        return "NarL_FixJ"
    return "Unknown"


def load_clusters(tsv):
    """Return dict: rep_id -> list[member_ids] and member -> rep."""
    df = pd.read_csv(tsv, sep="\t", header=None, names=["rep", "member"])
    rep_to_members = df.groupby("rep")["member"].apply(list).to_dict()
    member_to_rep = df.set_index("member")["rep"].to_dict()
    return rep_to_members, member_to_rep


def load_operons(operon_dir):
    """Concatenate per-genome operon TSVs."""
    frames = []
    for f in Path(operon_dir).glob("*.tsv"):
        try:
            df = pd.read_csv(f, sep="\t")
            if not df.empty and "hk" in df.columns:
                df["genome"] = f.stem
                frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ─── Bioinformatics linker phase analysis ────────────────────────────────────

def load_hamp_boundaries(domtbl_path):
    """Parse HMMER --domtblout → per-protein HAMP ali_from position.

    ali_from (column 16 in 1-indexed domtblout) is the sequence residue number
    where the HAMP domain alignment starts. This is our proxy for linker length:
    the heptad register of the sensor→HAMP junction is determined by HAMP_start mod 7.
    """
    hamp_starts = {}
    with open(domtbl_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 23:
                continue
            protein  = parts[0]
            domain   = parts[3]          # query/domain name — corrected (Bug #14)
            score    = float(parts[13])  # domain score — corrected
            ali_from = int(parts[17])    # ali_from on target sequence — corrected
            if domain in ("HAMP", "HAMP_2", "CovS-like_HAMP"):
                # Keep only the highest-scoring HAMP hit per protein
                if protein not in hamp_starts or score > hamp_starts[protein][1]:
                    hamp_starts[protein] = (ali_from, score)
    return {p: v[0] for p, v in hamp_starts.items()}


def compute_linker_phase(hamp_starts, hk_rep_to_members, conserved_cores_df):
    """Add per-candidate linker phase compatibility columns.

    Algorithm:
      1. For each conserved-core cluster, collect HAMP_start values from all members
         with HAMP domain data.
      2. Compute HAMP_start mod 7 (heptad phase) for each member.
      3. Dominant phase = mode of those values within the cluster.
      4. Cluster coherence = fraction of members matching the dominant phase.
         High coherence (≥0.8) → the cluster has consistent linker architecture.
      5. For each candidate: flag linker_phase_compatible if the candidate's own
         HAMP phase matches the cluster dominant phase AND coherence ≥ 0.8.
      6. Flag linker_validation_required when phase is uncertain or incompatible.

    This directly tests whether swapping sensor domains within a conserved-core
    cluster will preserve HAMP heptad register — the key criterion from Hatstat 2025.
    """
    # Build cluster → list of HAMP start positions across all members
    cluster_hamp_starts = {}
    for rep, members in hk_rep_to_members.items():
        starts = [hamp_starts[m] for m in members if m in hamp_starts]
        if starts:
            cluster_hamp_starts[rep] = starts

    # Compute dominant heptad phase and within-cluster coherence per cluster
    cluster_phase = {}
    cluster_phase_coherence = {}
    cluster_hamp_n = {}
    for rep, starts in cluster_hamp_starts.items():
        phases = [s % 7 for s in starts]
        dominant = max(set(phases), key=phases.count)
        coherence = phases.count(dominant) / len(phases)
        cluster_phase[rep] = dominant
        cluster_phase_coherence[rep] = round(coherence, 3)
        cluster_hamp_n[rep] = len(starts)

    df = conserved_cores_df.copy()
    df["hamp_start"] = df["qseqid"].map(hamp_starts)
    df["hamp_phase"] = df["hamp_start"].apply(
        lambda x: int(x) % 7 if pd.notna(x) else None
    )
    df["cluster_dominant_phase"] = df["cluster_rep"].map(cluster_phase)
    df["cluster_phase_coherence"] = df["cluster_rep"].map(cluster_phase_coherence)
    df["cluster_hamp_n"] = df["cluster_rep"].map(cluster_hamp_n).fillna(0).astype(int)

    def _is_compatible(row):
        if pd.isna(row.get("hamp_phase")) or pd.isna(row.get("cluster_dominant_phase")):
            return None
        if row["cluster_phase_coherence"] < 0.8:
            return None  # Cluster has inconsistent linker architecture — unknown
        return row["hamp_phase"] == row["cluster_dominant_phase"]

    df["linker_phase_compatible"] = df.apply(_is_compatible, axis=1)
    df["linker_validation_required"] = df["linker_phase_compatible"].apply(
        lambda x: True if x is None or x is False else False
    )
    return df


# ─── Reference TCS matching ──────────────────────────────────────────────────

def load_reference_tcs(reference_tcs_path):
    """Load well-characterized TCS reference table."""
    return pd.read_csv(reference_tcs_path, sep="\t")


def match_to_reference_tcs(ann_df, ref_df, protein_type):
    """Add known_tcs_system and working_in_user_system columns to annotation rows.

    Matches DIAMOND Swiss-Prot hit descriptions against reference TCS keyword lists.
    Returns the annotation DataFrame with added columns.

    Prioritises user's working systems (NarXL, PhoRB) in the match ranking.
    """
    if ann_df.empty:
        ann_df["known_tcs_system"] = None
        ann_df["working_in_user_system"] = False
        return ann_df

    keyword_col = "hk_swiss_prot_keywords" if protein_type == "HK" else "rr_swiss_prot_keywords"

    known_system = []
    working_flag = []

    stitle_lower = ann_df["stitle"].str.lower()

    for _, row in ann_df.iterrows():
        title = row["stitle"].lower() if pd.notna(row["stitle"]) else ""
        best_match = None
        best_working = False
        for _, ref_row in ref_df.iterrows():
            keywords = str(ref_row[keyword_col]).lower().split(";")
            if keywords == ["null"] or keywords == ["nan"]:
                continue
            if any(k.strip() in title for k in keywords):
                # Prefer working systems first
                is_working = str(ref_row["working_in_user_system"]).lower() == "yes"
                if best_match is None or (is_working and not best_working):
                    best_match = ref_row["system_name"]
                    best_working = is_working
        known_system.append(best_match)
        working_flag.append(best_working)

    ann_df = ann_df.copy()
    ann_df["known_tcs_system"] = known_system
    ann_df["working_in_user_system"] = working_flag
    return ann_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hk_clusters", required=True)
    parser.add_argument("--rr_clusters", required=True)
    parser.add_argument("--hk_annotation", required=True)
    parser.add_argument("--rr_annotation", required=True)
    parser.add_argument("--hk_cross_homology", required=True,
                        help="MMseqs2 .m8 HK-vs-RR cross hits")
    parser.add_argument("--operon_dir", required=True)
    parser.add_argument("--hk_domtbl", required=True,
                        help="HMMER --domtblout for HK reps (provides HAMP coordinates)")
    parser.add_argument("--reference_tcs", required=True,
                        help="data/reference/well_characterized_tcs.tsv")
    parser.add_argument("--characterized_promoters", default=None,
                        help="data/reference/characterized_promoters.tsv — if provided, "
                             "adds has_characterized_promoter and recommended_promoters columns")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    hk_rep_to_members, hk_member_to_rep = load_clusters(args.hk_clusters)
    rr_rep_to_members, rr_member_to_rep = load_clusters(args.rr_clusters)

    hk_ann = pd.read_csv(args.hk_annotation, sep="\t", header=None,
                         names=["qseqid", "sseqid", "pident", "length",
                                "qcovhsp", "evalue", "bitscore", "stitle"])
    rr_ann = pd.read_csv(args.rr_annotation, sep="\t", header=None,
                         names=["qseqid", "sseqid", "pident", "length",
                                "qcovhsp", "evalue", "bitscore", "stitle"])

    cross = pd.read_csv(args.hk_cross_homology, sep="\t", header=None,
                        names=["hk", "rr", "pident", "alnlen",
                               "mismatch", "gapopen", "qstart", "qend",
                               "tstart", "tend", "evalue", "bitscore"])
    cross = cross[cross["hk"] != cross["rr"]]

    operons = load_operons(args.operon_dir)

    hamp_starts = load_hamp_boundaries(args.hk_domtbl)
    print(f"HAMP domain boundaries loaded: {len(hamp_starts)} proteins")

    ref_tcs = load_reference_tcs(args.reference_tcs)

    # Match annotations to reference TCS before filtering
    hk_ann = match_to_reference_tcs(hk_ann, ref_tcs, "HK")
    rr_ann = match_to_reference_tcs(rr_ann, ref_tcs, "RR")

    # ── Strategy B: RR DBD-swap candidates ────────────────────────────────
    rr_ann["dbd_family"] = rr_ann["stitle"].apply(classify_rr_family)
    rr_ann["cluster_rep"] = rr_ann["qseqid"].map(rr_member_to_rep).fillna(rr_ann["qseqid"])
    rr_ann["cluster_size"] = rr_ann["cluster_rep"].map(
        lambda r: len(rr_rep_to_members.get(r, [r])))

    dbd_candidates = rr_ann[rr_ann["dbd_family"].isin(["OmpR_PhoB", "NarL_FixJ"])].copy()
    dbd_candidates["chimera_type"] = "RR_DBD_swap"
    dbd_candidates["rationale"] = dbd_candidates["dbd_family"].map({
        "OmpR_PhoB": "OmpR/PhoB family: swap DBD at ~OmpR-122/137 linker boundary (Schmidl 2019)",
        "NarL_FixJ": "NarL/FixJ family: swap DBD at equivalent linker boundary (Schmidl 2019)"
    })
    # RR DBD swaps don't involve HAMP — phase analysis N/A
    dbd_candidates["linker_phase_compatible"] = None
    dbd_candidates["linker_validation_required"] = False
    dbd_candidates["cluster_dominant_phase"] = None
    dbd_candidates["cluster_phase_coherence"] = None
    dbd_candidates["hamp_start"] = None

    # ── Strategy A: HK sensor-swap candidates ─────────────────────────────
    hk_ann["cluster_rep"] = hk_ann["qseqid"].map(hk_member_to_rep).fillna(hk_ann["qseqid"])
    hk_ann["cluster_size"] = hk_ann["cluster_rep"].map(
        lambda r: len(hk_rep_to_members.get(r, [r])))

    # Large clusters (≥50 members) → conserved kinase core
    conserved_cores = hk_ann[hk_ann["cluster_size"] >= 50].copy()
    conserved_cores["chimera_type"] = "HK_sensor_swap"
    conserved_cores["rationale"] = (
        "Large HK cluster → conserved kinase core (DHp+CA). "
        "Swap sensor domain retaining HAMP second-half through CA. "
        "Select donor by HAMP alignment (Peruzzi 2023). "
        "Verify linker phase (Hatstat 2025)."
    )

    # ── Bioinformatics linker phase analysis ──────────────────────────────
    conserved_cores = compute_linker_phase(hamp_starts, hk_rep_to_members, conserved_cores)

    # ── Operon-linked pairs ───────────────────────────────────────────────
    if not operons.empty and "hk" in operons.columns and "rr" in operons.columns:
        operon_map = operons.set_index("hk")["rr"].to_dict()
        conserved_cores["cognate_rr"] = conserved_cores["qseqid"].map(operon_map)
        conserved_cores["cognate_rr_cluster"] = conserved_cores["cognate_rr"].map(
            rr_member_to_rep)

    # ── Cross-homology: hybrid/bifunctional candidates ────────────────────
    hybrid_candidates = cross[cross["pident"] > 25].copy()
    hybrid_candidates["chimera_type"] = "Hybrid_TCS"
    hybrid_candidates["rationale"] = (
        "HK with RR homology (>25% identity) → candidate hybrid kinase. "
        "May contain fused REC domain. Verify domain architecture before design."
    )

    # ── Compile output ────────────────────────────────────────────────────
    out_rows = []

    for _, row in dbd_candidates.iterrows():
        out_rows.append({
            "protein_id": row["qseqid"],
            "best_hit": row["sseqid"],
            "hit_description": row["stitle"],
            "pident": row["pident"],
            "cluster_size": row["cluster_size"],
            "chimera_type": row["chimera_type"],
            "dbd_family": row["dbd_family"],
            "known_tcs_system": row.get("known_tcs_system"),
            "working_in_user_system": row.get("working_in_user_system", False),
            "rationale": row["rationale"],
            "evalue": row["evalue"],
            "hamp_start": row["hamp_start"],
            "cluster_dominant_phase": row["cluster_dominant_phase"],
            "cluster_phase_coherence": row["cluster_phase_coherence"],
            "linker_phase_compatible": row["linker_phase_compatible"],
            "linker_validation_required": row["linker_validation_required"],
        })

    for _, row in conserved_cores.iterrows():
        out_rows.append({
            "protein_id": row["qseqid"],
            "best_hit": row.get("sseqid", ""),
            "hit_description": row.get("stitle", ""),
            "pident": row.get("pident", ""),
            "cluster_size": row["cluster_size"],
            "chimera_type": row["chimera_type"],
            "dbd_family": "N/A",
            "known_tcs_system": row.get("known_tcs_system"),
            "working_in_user_system": row.get("working_in_user_system", False),
            "rationale": row["rationale"],
            "evalue": row.get("evalue", ""),
            "hamp_start": row.get("hamp_start"),
            "cluster_dominant_phase": row.get("cluster_dominant_phase"),
            "cluster_phase_coherence": row.get("cluster_phase_coherence"),
            "linker_phase_compatible": row.get("linker_phase_compatible"),
            "linker_validation_required": row.get("linker_validation_required", True),
        })

    result = pd.DataFrame(out_rows)
    # Sort: user working systems first, then by type + cluster size
    result["_sort_priority"] = result["working_in_user_system"].apply(lambda x: 0 if x else 1)
    result = result.sort_values(
        ["_sort_priority", "chimera_type", "cluster_size"],
        ascending=[True, True, False]
    ).drop(columns=["_sort_priority"])
    # ── Promoter mapping (if table provided) ──────────────────────────────
    if args.characterized_promoters and Path(args.characterized_promoters).exists():
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from map_promoters_to_candidates import map_promoters
        promoters_df = pd.read_csv(args.characterized_promoters, sep="\t")
        result = map_promoters(result, promoters_df)
        n_with = result["has_characterized_promoter"].sum()
        print(f"\nPromoter mapping: {n_with}/{len(result)} candidates have characterized promoters")

    result.to_csv(args.output, sep="\t", index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\nChimera target summary:")
    print(result["chimera_type"].value_counts().to_string())

    hk_sub = result[result["chimera_type"] == "HK_sensor_swap"]
    n_phase_ok = hk_sub["linker_phase_compatible"].eq(True).sum()
    n_phase_unknown = hk_sub["linker_phase_compatible"].isna().sum()
    n_phase_bad = hk_sub["linker_phase_compatible"].eq(False).sum()
    print(f"\nHK_sensor_swap linker phase status:")
    print(f"  Phase compatible:  {n_phase_ok}")
    print(f"  Phase unknown:     {n_phase_unknown} (manual validation required)")
    print(f"  Phase incompatible:{n_phase_bad} (manual validation required)")

    print(f"\nTop RR DBD-swap candidates:")
    print(result[result["chimera_type"] == "RR_DBD_swap"]
          .groupby("dbd_family")["cluster_size"].describe().to_string())
    print(f"\nTop HK sensor-swap candidates (largest clusters):")
    print(result[result["chimera_type"] == "HK_sensor_swap"]
          .nlargest(10, "cluster_size")[
              ["protein_id", "cluster_size", "linker_phase_compatible",
               "cluster_phase_coherence", "hit_description"]
          ].to_string(index=False))
    working = result[result["working_in_user_system"] == True]
    print(f"\nCandidates similar to user's working systems (NarXL, PhoRB): {len(working)}")
    if not working.empty:
        print(working[["protein_id", "known_tcs_system", "chimera_type",
                        "cluster_size", "hit_description"]].head(10).to_string(index=False))
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
