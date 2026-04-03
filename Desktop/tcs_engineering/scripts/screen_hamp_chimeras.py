#!/usr/bin/env python3
"""Screen HAMP sensor-swap chimera candidates using AF2 structural confidence.

Domain-aware pLDDT strategy for transmembrane histidine kinases:
  - TM + sensor region (residues 1..hamp_start): EXCLUDED from confidence
    calculations. AF2 always gives low pLDDT (~30-50) for membrane-embedded
    segments — this is expected and is NOT a chimera design concern. Including
    TM pLDDT in a global score would penalise every valid TM-HK candidate.

  - HAMP domain (hamp_start..junction+30): TARGET >70. This is the chimera
    junction region. Low pLDDT here indicates structural ambiguity in the
    linker that would compromise junction design.

  - Kinase core DHp + CA (junction+30..end): TARGET >70. Low pLDDT in the
    kinase core indicates autophosphorylation domain instability.

Hatstat 2025 hydrophobic seam check:
  The HAMP-AS1 helix begins at the junction N-x-[ML]. In a canonical 4-helix
  bundle coiled-coil, heptad positions a and d are hydrophobic. Starting from
  the M/L at junction+2:
    a-positions: junction+2, junction+9, junction+16 ...
    d-positions: junction+5, junction+12, junction+19 ...
  If polar or charged residues appear at a/d positions, the hydrophobic seam
  is broken — this chimera pair requires structural correction.

UniProt mapping:
  RefSeq WP_/NP_ accessions are mapped to UniProt via the UniProt REST
  ID mapping API (batch, 500 IDs at a time). AF2 structures are then
  downloaded from EBI AlphaFold (https://alphafoldserver.com).
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOP_N           = 50          # screen top N candidates by phase/motif/identity
PLDDT_THRESHOLD = 70.0        # minimum mean pLDDT for HAMP and kinase core
HYDROPHOBIC     = frozenset("LIVAMF")

# Pfam accessions for domain boundary parsing
HAMP_PFAM       = "PF00672"
DHp_PFAMS       = {"PF00512", "PF07730", "PF08447", "PF13589", "PF13188"}
CA_PFAMS        = {"PF02518", "PF13581", "PF13426", "PF13589", "PF08448"}

EBI_AF2_URL     = "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v{ver}.pdb"
EBI_AF2_VERSIONS = [6, 5, 4]   # try newest first
UNIPROT_MAP_RUN = "https://rest.uniprot.org/idmapping/run"
UNIPROT_MAP_STATUS = "https://rest.uniprot.org/idmapping/status/{job}"
UNIPROT_MAP_RESULTS = "https://rest.uniprot.org/idmapping/uniprotkb/results/stream/{job}"


# ---------------------------------------------------------------------------
# UniProt ID mapping
# ---------------------------------------------------------------------------

def map_refseq_to_uniprot(refseq_ids: list[str]) -> dict[str, str]:
    """Batch-map RefSeq protein accessions → UniProt accessions.

    Uses the UniProt REST ID mapping API.  Returns {refseq_id: uniprot_ac}.
    IDs with no UniProt match are absent from the returned dict.
    """
    if not refseq_ids:
        return {}

    mapping: dict[str, str] = {}
    # API limit: 500 IDs per request
    batch_size = 500
    for i in range(0, len(refseq_ids), batch_size):
        batch = refseq_ids[i : i + batch_size]
        try:
            resp = requests.post(
                UNIPROT_MAP_RUN,
                data={"from": "RefSeq_Protein", "to": "UniProtKB", "ids": ",".join(batch)},
                timeout=30,
            )
            resp.raise_for_status()
            job_id = resp.json()["jobId"]

            # Poll until finished (max 120 s)
            for _ in range(40):
                st = requests.get(
                    UNIPROT_MAP_STATUS.format(job=job_id), timeout=15
                ).json()
                if "results" in st or st.get("jobStatus") == "FINISHED":
                    break
                time.sleep(3)

            # Stream TSV results
            res = requests.get(
                UNIPROT_MAP_RESULTS.format(job=job_id) + "?format=tsv",
                timeout=60,
            )
            for line in res.text.strip().split("\n")[1:]:   # skip header
                parts = line.split("\t")
                if len(parts) >= 2:
                    # TSV cols: From  Entry  Entry Name  ...
                    mapping[parts[0]] = parts[1]
        except Exception as e:
            print(f"  [warn] UniProt mapping batch {i//batch_size} failed: {e}")

    return mapping


# ---------------------------------------------------------------------------
# AF2 structure download
# ---------------------------------------------------------------------------

def download_af2(uniprot_id: str, outdir: Path) -> Path | None:
    """Download AF2 PDB from EBI (tries v6, v5, v4). Returns path or None."""
    outfile = outdir / f"{uniprot_id}.pdb"
    if outfile.exists() and outfile.stat().st_size > 100:
        return outfile
    for ver in EBI_AF2_VERSIONS:
        url = EBI_AF2_URL.format(uid=uniprot_id, ver=ver)
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                outfile.write_text(r.text)
                return outfile
        except requests.RequestException:
            continue
    return None


# ---------------------------------------------------------------------------
# PDB parsing — pLDDT from B-factor
# ---------------------------------------------------------------------------

def parse_plddt(pdb_path: Path) -> dict[int, float]:
    """Return {residue_number: pLDDT} from an AF2 PDB file.

    AF2 stores per-residue pLDDT in the B-factor column of ATOM records.
    Only CA atoms are used (one value per residue).
    """
    plddt: dict[int, float] = {}
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    resnum = int(line[22:26])
                    bfac   = float(line[60:66])
                    plddt[resnum] = bfac
                except ValueError:
                    continue
    return plddt


# ---------------------------------------------------------------------------
# Domain boundary extraction from domtblout
# ---------------------------------------------------------------------------

def load_domain_boundaries(domtbl_path: str) -> dict[str, dict]:
    """Parse HMMER domtblout for HAMP, DHp, and CA domain boundaries.

    Returns {protein_id: {hamp: (start,end), dhp: (start,end), ca: (start,end)}}
    Uses envelope coordinates (env_from, env_to) for maximum coverage.
    """
    bounds: dict[str, dict] = {}
    with open(domtbl_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 22:
                continue
            protein_id = parts[0]
            pfam_acc   = parts[4].split(".")[0]   # strip version
            env_from   = int(parts[19])
            env_to     = int(parts[20])

            if protein_id not in bounds:
                bounds[protein_id] = {}

            if pfam_acc == HAMP_PFAM:
                # Keep the best (lowest e-value) HAMP hit if multiple
                if "hamp" not in bounds[protein_id]:
                    bounds[protein_id]["hamp"] = (env_from, env_to)
            elif pfam_acc in DHp_PFAMS:
                prev = bounds[protein_id].get("dhp")
                if prev is None or env_to - env_from > prev[1] - prev[0]:
                    bounds[protein_id]["dhp"] = (env_from, env_to)
            elif pfam_acc in CA_PFAMS:
                prev = bounds[protein_id].get("ca")
                if prev is None or env_to - env_from > prev[1] - prev[0]:
                    bounds[protein_id]["ca"] = (env_from, env_to)

    return bounds


# ---------------------------------------------------------------------------
# Domain-aware pLDDT analysis
# ---------------------------------------------------------------------------

def domain_plddt(plddt_map: dict[int, float],
                 hamp_start: int,
                 junction_pos: int,    # 1-indexed
                 domain_bounds: dict) -> dict:
    """Compute per-domain pLDDT statistics, excluding the TM region.

    TM region definition: residues 1..(hamp_start - 1)
      This covers TM1 + periplasmic sensor + TM2. AF2 always gives low
      pLDDT here for membrane-embedded α-helices — excluded from all
      confidence calculations.

    HAMP region: hamp_start..(junction_pos + 30)
    Kinase core: (junction_pos + 30)..end
    """
    def region_stats(residues: list[float]) -> dict:
        if not residues:
            return {"mean": float("nan"), "frac_high": float("nan"), "n": 0}
        mean = sum(residues) / len(residues)
        frac = sum(1 for v in residues if v >= PLDDT_THRESHOLD) / len(residues)
        return {"mean": round(mean, 1), "frac_high": round(frac, 2), "n": len(residues)}

    # Use domain bounds if available; fall back to junction-based estimates
    if "hamp" in domain_bounds:
        h_start, h_end = domain_bounds["hamp"]
    else:
        h_start, h_end = hamp_start, junction_pos + 30

    kinase_start = junction_pos + 30
    if "dhp" in domain_bounds:
        kinase_start = min(kinase_start, domain_bounds["dhp"][0])
    if "ca" in domain_bounds:
        kinase_end = domain_bounds["ca"][1]
    else:
        kinase_end = max(plddt_map.keys(), default=kinase_start)

    # TM: everything before HAMP (informational only)
    tm_vals    = [v for r, v in plddt_map.items() if r < h_start]
    hamp_vals  = [v for r, v in plddt_map.items() if h_start <= r <= h_end]
    kin_vals   = [v for r, v in plddt_map.items()
                  if kinase_start <= r <= kinase_end]

    return {
        "plddt_tm":     region_stats(tm_vals),
        "plddt_hamp":   region_stats(hamp_vals),
        "plddt_kinase": region_stats(kin_vals),
    }


# ---------------------------------------------------------------------------
# Hatstat hydrophobic seam check
# ---------------------------------------------------------------------------

def _seam_pattern(seq: str, junction_0idx: int) -> list[str]:
    """Return character class at HAMP-AS1 a/d heptad positions.

    Starting from M/L at junction+2, the AS-1 coiled-coil core positions are:
      a-positions: junction+2, +9, +16, +23
      d-positions: junction+5, +12, +19, +26
    Returns a list of 'H' (hydrophobic: LIVAMF), 'P' (polar/charged), or '?' (missing).
    """
    j = junction_0idx
    core_positions = sorted(
        [j + 2 + 7 * k for k in range(4)] +
        [j + 5 + 7 * k for k in range(4)]
    )
    pattern = []
    for pos in core_positions:
        if 0 <= pos < len(seq):
            pattern.append("H" if seq[pos] in HYDROPHOBIC else "P")
        else:
            pattern.append("?")
    return pattern


def check_seam_compatibility(seq_a: str, seq_b: str,
                              junction_a: int, junction_b: int) -> dict:
    """Check HAMP-AS1 seam compatibility between acceptor and donor (Hatstat 2025).

    HAMP uses a dynamic asymmetric 4-helix bundle — a/d positions are NOT
    required to be purely hydrophobic. What matters for chimera compatibility
    is that acceptor and donor have the SAME residue character class (both
    hydrophobic or both polar) at equivalent heptad positions. This ensures
    the junction environment is compatible without requiring a classical
    leucine-zipper seam.

    seam_compatible = fraction of matched positions >= 0.6
    """
    pa = _seam_pattern(seq_a, junction_a)
    pb = _seam_pattern(seq_b, junction_b)
    total = sum(1 for a, b in zip(pa, pb) if "?" not in (a, b))
    match = sum(1 for a, b in zip(pa, pb) if a == b and "?" not in (a, b))
    fraction = round(match / total, 2) if total > 0 else float("nan")
    return {
        "seam_fraction":    fraction,
        "seam_pattern_a":   "".join(pa),
        "seam_pattern_b":   "".join(pb),
        "seam_compatible":  fraction >= 0.6,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_annotation_gene_names(ann_path: str) -> dict[str, str]:
    """Parse {protein_id: gene_name} from a DIAMOND Swiss-Prot annotation TSV.

    The Swiss-Prot stitle contains 'GN=<gene>' which is more reliable than the
    sseqid accession for cross-sensor comparison — e.g. 'GN=narX' vs 'GN=cusS'.
    Returns empty string for proteins with no named Swiss-Prot hit.
    """
    if not Path(ann_path).exists():
        print(f"  [warn] annotation file not found: {ann_path}")
        return {}
    df = pd.read_csv(ann_path, sep="\t", header=None,
                     names=["qseqid", "sseqid", "pident", "length",
                            "qcovhsp", "evalue", "bitscore", "stitle"])
    gene_map: dict[str, str] = {}
    for _, row in df.iterrows():
        gn = re.search(r"\bGN=(\S+)", str(row["stitle"]))
        gene_map[row["qseqid"]] = gn.group(1).lower() if gn else ""
    return gene_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates",    required=True,
                        help="results/chimera_targets/hamp_swap_candidates.tsv")
    parser.add_argument("--hamp_fasta",    required=True,
                        help="results/deepcoil/hamp_linker_regions.faa")
    parser.add_argument("--hk_reps",       required=True,
                        help="results/representatives/hk_reps.faa")
    parser.add_argument("--domtbl",        required=True,
                        help="results/domains/hk_reps_domtbl.txt")
    parser.add_argument("--outdir",        required=True)
    parser.add_argument("--hk_annotation", default="",
                        help="results/annotation/hk_annotation.tsv — enables cross-sensor prioritization")
    parser.add_argument("--top_n",         type=int, default=TOP_N)
    parser.add_argument("--motif_min",     type=int, default=5,
                        help="Minimum combined motif score (0-10). "
                             "Default 5: requires at least one partner to have a "
                             "strong N-x-[ML]-[LVI]-[LIVF] motif. Lower values "
                             "admit more candidates but include weaker junctions.")
    parser.add_argument("--plddt_min",     type=float, default=PLDDT_THRESHOLD)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    pdb_dir = outdir / "pdb"
    outdir.mkdir(parents=True, exist_ok=True)
    pdb_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and filter candidates
    # ------------------------------------------------------------------
    df = pd.read_csv(args.candidates, sep="\t")
    total = len(df)

    # Optional: load Swiss-Prot gene names for cross-sensor prioritization.
    # cross_sensor=True when acceptor and donor have DIFFERENT named Swiss-Prot
    # hits — these are the chimeras that actually combine different sensing inputs.
    gene_names: dict[str, str] = {}
    if args.hk_annotation:
        print(f"Loading annotation for cross-sensor prioritization: {args.hk_annotation}")
        gene_names = load_annotation_gene_names(args.hk_annotation)
        print(f"  {sum(1 for v in gene_names.values() if v)} proteins with named hits")

    def is_cross_sensor(row) -> bool:
        ga = gene_names.get(row["acceptor"], "")
        gb = gene_names.get(row["donor"], "")
        return bool(ga and gb and ga != gb)

    if gene_names:
        df["cross_sensor"] = df.apply(is_cross_sensor, axis=1)
        n_cross = df["cross_sensor"].sum()
        print(f"  Cross-sensor pairs (different Swiss-Prot gene): {n_cross}/{total}")
    else:
        df["cross_sensor"] = False

    # Priority tiers (highest first):
    #   1. cross_sensor — different sensing inputs (key engineering requirement)
    #   2. phase_match  — no heptad correction needed
    #   3. motif_score  — Peruzzi AS-1 signature quality
    #   4. junction_identity — HAMP sequence conservation
    sort_cols = ["cross_sensor", "phase_match", "motif_score", "junction_identity"]
    df = df.sort_values(sort_cols, ascending=[False, False, False, False])

    df_top = df[(df["motif_score"] >= args.motif_min)].head(args.top_n).copy()
    if len(df_top) < args.top_n:
        df_rest = df[~df.index.isin(df_top.index)]
        df_top = pd.concat([df_top, df_rest]).head(args.top_n)

    print(f"Loaded {total} HAMP swap candidates")
    n_cross_top = df_top["cross_sensor"].sum() if "cross_sensor" in df_top.columns else 0
    print(f"Screening top {len(df_top)} — {n_cross_top} cross-sensor pairs"
          f" (motif≥{args.motif_min})")

    # ------------------------------------------------------------------
    # 2. Parse HAMP metadata (hamp_start per protein)
    # ------------------------------------------------------------------
    hamp_info = {}
    for line in open(args.hamp_fasta):
        if line.startswith(">"):
            parts = line[1:].split()
            pid = parts[0].rsplit("_HAMP_", 1)[0]
            hs = int(parts[1].split("=")[1])
            hamp_info[pid] = hs   # 1-indexed

    full_seqs: dict[str, str] = {}
    pid = None
    buf: list[str] = []
    for line in open(args.hk_reps):
        line = line.rstrip()
        if line.startswith(">"):
            if pid:
                full_seqs[pid] = "".join(buf)
            pid = line[1:].split()[0]
            buf = []
        else:
            buf.append(line)
    if pid:
        full_seqs[pid] = "".join(buf)

    # ------------------------------------------------------------------
    # 3. Load domain boundaries
    # ------------------------------------------------------------------
    print(f"Loading domain boundaries from {args.domtbl} ...")
    dom_bounds = load_domain_boundaries(args.domtbl)

    # ------------------------------------------------------------------
    # 4. Map RefSeq → UniProt
    # ------------------------------------------------------------------
    unique_pids = list(set(df_top["acceptor"]) | set(df_top["donor"]))
    print(f"Mapping {len(unique_pids)} protein IDs to UniProt ...")
    refseq_to_uniprot = map_refseq_to_uniprot(unique_pids)
    print(f"  {len(refseq_to_uniprot)} IDs mapped to UniProt")

    # ------------------------------------------------------------------
    # 5. Download AF2 structures
    # ------------------------------------------------------------------
    print("Downloading AF2 structures from EBI ...")
    pdb_paths: dict[str, Path] = {}
    for pid, uid in refseq_to_uniprot.items():
        path = download_af2(uid, pdb_dir)
        if path:
            pdb_paths[pid] = path
        else:
            print(f"  [miss] {pid} → {uid}: no AF2 structure")
    print(f"  {len(pdb_paths)} structures downloaded")

    # ------------------------------------------------------------------
    # 6. Structural analysis per candidate pair
    # ------------------------------------------------------------------
    print(f"Analysing {len(df_top)} candidate pairs ...")
    results = []

    for _, row in df_top.iterrows():
        a, b = row["acceptor"], row["donor"]
        j_a = int(row["junction_pos_a"])   # 1-indexed
        j_b = int(row["junction_pos_b"])

        rec = {
            "acceptor":          a,
            "donor":             b,
            "junction_identity": row["junction_identity"],
            "sensor_identity":   row["sensor_identity"],
            "phase_match":       row["phase_match"],
            "motif_score":       row["motif_score"],
            "motif5_a":          row.get("motif5_a", ""),
            "motif5_b":          row.get("motif5_b", ""),
            "junction_pos_a":    j_a,
            "junction_pos_b":    j_b,
            "cross_sensor":      row.get("cross_sensor", False),
            "acceptor_gene":     gene_names.get(a, ""),
            "donor_gene":        gene_names.get(b, ""),
        }

        for role, pid, j_pos in [("acceptor", a, j_a), ("donor", b, j_b)]:
            pdb = pdb_paths.get(pid)
            uid = refseq_to_uniprot.get(pid, "")
            hamp_start = hamp_info.get(pid, j_pos - 30)
            bounds = dom_bounds.get(pid, {})

            rec[f"{role}_uniprot"] = uid if uid else None

            if pdb:
                plddt_map = parse_plddt(pdb)
                stats = domain_plddt(plddt_map, hamp_start, j_pos, bounds)
                rec[f"{role}_plddt_tm"]       = stats["plddt_tm"]["mean"]
                rec[f"{role}_plddt_hamp"]     = stats["plddt_hamp"]["mean"]
                rec[f"{role}_plddt_hamp_n"]   = stats["plddt_hamp"]["n"]
                rec[f"{role}_plddt_kinase"]   = stats["plddt_kinase"]["mean"]
                rec[f"{role}_plddt_kinase_n"] = stats["plddt_kinase"]["n"]
                rec[f"{role}_hamp_highconf"]  = (
                    stats["plddt_hamp"]["mean"] >= args.plddt_min
                    if stats["plddt_hamp"]["n"] > 0 else False
                )
                rec[f"{role}_kinase_highconf"] = (
                    stats["plddt_kinase"]["mean"] >= args.plddt_min
                    if stats["plddt_kinase"]["n"] > 0 else False
                )
            else:
                for col in ["plddt_tm", "plddt_hamp", "plddt_hamp_n",
                            "plddt_kinase", "plddt_kinase_n",
                            "hamp_highconf", "kinase_highconf"]:
                    rec[f"{role}_{col}"] = None

        # Pairwise seam compatibility (Hatstat 2025): compare acceptor vs donor
        # HAMP a/d character must be CONSISTENT between partners, not purely hydrophobic
        seq_a = full_seqs.get(a, "")
        seq_b = full_seqs.get(b, "")
        if seq_a and seq_b:
            seam = check_seam_compatibility(seq_a, seq_b, j_a - 1, j_b - 1)
            rec["seam_fraction"]   = seam["seam_fraction"]
            rec["seam_pattern_a"]  = seam["seam_pattern_a"]
            rec["seam_pattern_b"]  = seam["seam_pattern_b"]
            rec["seam_compatible"] = seam["seam_compatible"]
        else:
            rec["seam_fraction"]   = float("nan")
            rec["seam_pattern_a"]  = ""
            rec["seam_pattern_b"]  = ""
            rec["seam_compatible"] = False

        results.append(rec)

    out = pd.DataFrame(results)

    # ------------------------------------------------------------------
    # 7. Composite structural confidence flag
    # ------------------------------------------------------------------
    # structural_ok: BOTH proteins have high-confidence HAMP AND kinase pLDDT
    # TM pLDDT is INTENTIONALLY excluded — low TM pLDDT is expected and OK
    out["structural_ok"] = (
        out["acceptor_hamp_highconf"].fillna(False)
        & out["acceptor_kinase_highconf"].fillna(False)
        & out["donor_hamp_highconf"].fillna(False)
        & out["donor_kinase_highconf"].fillna(False)
    )
    # seam_compatible already computed pairwise (not per-protein)
    out["seam_ok"] = out["seam_compatible"].fillna(False)
    out["af2_available"] = (
        out["acceptor_uniprot"].notna()
        & out["donor_uniprot"].notna()
    )

    # ------------------------------------------------------------------
    # 8. Rank and write outputs
    # ------------------------------------------------------------------
    out = out.sort_values(
        ["cross_sensor", "structural_ok", "seam_ok", "phase_match", "motif_score", "junction_identity"],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)

    tsv_out = outdir / "hamp_chimera_screen.tsv"
    out.to_csv(tsv_out, sep="\t", index=False)
    print(f"\nWritten: {tsv_out}")

    # Summary
    n_struct = out["structural_ok"].sum()
    n_seam   = out["seam_ok"].sum()
    n_af2    = out["af2_available"].sum()
    print(f"\n=== Screening summary (top {len(out)} candidates) ===")
    print(f"  AF2 structure available (both proteins):         {n_af2}")
    print(f"  Structural OK (HAMP + kinase pLDDT ≥{args.plddt_min}): {n_struct}")
    print(f"  Seam compatible (Hatstat a/d character match):   {n_seam}")
    print(f"  Structural OK + seam compatible:                 {(out['structural_ok'] & out['seam_ok']).sum()}")
    print(f"\n  Note: TM pLDDT intentionally excluded — low TM")
    print(f"        pLDDT is expected for membrane HKs and is")
    print(f"        NOT a structural confidence concern.")
    print()

    display_cols = [
        "acceptor", "donor", "junction_identity", "sensor_identity",
        "cross_sensor", "acceptor_gene", "donor_gene",
        "phase_match", "motif5_a", "motif5_b", "motif_score",
        "acceptor_plddt_hamp", "acceptor_plddt_kinase",
        "donor_plddt_hamp",    "donor_plddt_kinase",
        "seam_fraction", "seam_pattern_a", "seam_pattern_b",
        "structural_ok", "seam_ok",
    ]
    # Only show columns that exist
    display_cols = [c for c in display_cols if c in out.columns]
    print(out[display_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
