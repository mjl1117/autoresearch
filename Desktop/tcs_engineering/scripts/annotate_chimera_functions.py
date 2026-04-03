#!/usr/bin/env python3
"""Functional annotation of HAMP chimera candidate proteins.

For each acceptor/donor protein in the chimera screen, retrieves:
  1. Gene name, product description, organism from NCBI protein records
     (Entrez e-utils, batch fetch via Biopython)
  2. Swiss-Prot DIAMOND best-hit annotation from hk_annotation.tsv
  3. Domain architecture from HMMER domtblout (sensor domain type inference)

Sensor type classification:
  Uses product description keywords and domain architecture to infer what
  environmental signal the histidine kinase senses. Classified into:
    nitrate/nitrite, phosphate, pH/acid, osmolarity, metal (Cu/Fe/Ni/Zn/Co),
    vancomycin/cell-wall, redox/oxygen, temperature, quorum, carbon/energy,
    uncharacterized.

The RR (response regulator) pairing is looked up from the operon TSV files
to identify the downstream gene target.

Output: chimera_annotations.tsv with one row per unique protein ID.
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd
from Bio import Entrez, SeqIO


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Map keywords (word-boundary matched) to sensor type categories.
# Tuples: (sensor_type, [whole-word patterns])
# Each pattern is matched as a word boundary (\b...\b) to prevent
# substring false positives (e.g., "kinA" matching "kinase").
SENSOR_KEYWORDS: list[tuple[str, list[str]]] = [
    ("nitrate/nitrite",      ["nitrate", "nitrite", r"\bnar[xsq]\b"]),
    ("phosphate",            ["phosphate", r"\bpho[rpq]\b", r"\bphob\b"]),
    ("osmolarity",           ["osmolar", "osmosens", r"\benvz\b", r"\bompr\b"]),
    ("pH/acid",              [r"\bacid\s+stress\b", r"\blow.?ph\b"]),
    ("metal_copper",         ["copper", r"\bcus[sr]\b", r"\bcop[aybr]\b"]),
    ("metal_iron",           [r"\biron\b", r"\bferri", r"\brss[a-z]\b"]),
    ("metal_nickel",         [r"\bnickel\b", r"\bnrs[a-z]\b"]),
    ("metal_zinc",           [r"\bzinc\b", r"\bzn\b"]),
    ("metal_heavy",          ["heavy metal"]),
    ("vancomycin/cell-wall", [r"\bvancomycin\b", r"\bcell\s+wall\b", r"\bvan[rs]\b", r"\bbce[ab]\b"]),
    ("redox/oxygen",         [r"\bredox\b", r"\boxygen\b", r"\baerob", r"\banaerob", r"\barcb\b"]),
    ("quorum",               [r"\bquorum\b", r"\bautoinducer\b", r"\bluxn\b"]),
    ("carbon/energy",        [r"\bcarbon\b", r"\bglycerol\b", r"\bglucose\b", r"\bgacs\b"]),
    ("temperature",          [r"\btemperature\b", r"\bheat\s+shock\b", r"\bcold\s+shock\b"]),
    ("chemotaxis",           [r"\bchemotax", r"\bmethyl-accepting\b", r"\bche[acy]\b"]),
    ("cell-division",        [r"\bcell\s+division\b", r"\bsporul", r"\bkin[ab]\b"]),
    ("phosphorelay",         [r"\bpars\b", r"\bparb\b"]),
    ("arsenic",              [r"\barsenic\b", r"\bars[rbs]\b"]),
    ("two-component",        [r"\btwo-component\b"]),    # last-resort generic
]

# Known TCS gene name → sensor type (Swiss-Prot curated)
# Used to classify via DIAMOND best-hit gene name when description is generic.
GENE_SENSOR_MAP: dict[str, str] = {
    # Nitrate/nitrite
    "narx": "nitrate/nitrite", "narq": "nitrate/nitrite", "nars": "nitrate/nitrite",
    # Phosphate
    "phoq": "Mg2+/antimicrobial_peptide", "phor": "phosphate", "phob": "phosphate",
    # Osmolarity
    "envz": "osmolarity",
    # Acid/outer membrane
    "rstb": "acid/OM_stress", "cpxa": "envelope_stress", "basr": "envelope_stress",
    # Metal
    "cuss": "metal_copper", "cusa": "metal_copper", "czcs": "metal_Zn/Co/Cd",
    "rssa": "metal_iron", "feur": "metal_iron",
    "nrss": "metal_nickel",
    "vars": "vancomycin/cell-wall",
    # Arsenic
    "arss": "arsenic",
    # Carbon/energy
    "gacs": "carbon_energy", "bara": "carbon_energy", "uvrb": "carbon_energy",
    # Redox/oxygen
    "arcb": "redox/oxygen", "aera": "redox/oxygen",
    # Quorum/AI
    "luxn": "quorum", "lqss": "quorum",
    # Chemotaxis
    "chea": "chemotaxis",
    # Cell division / chromosome partition
    "pars": "chromosome_partition",
    # Temperature
    "rhts": "temperature",
}

# Pfam sensor domain accessions → sensor class
SENSOR_PFAM: dict[str, str] = {
    "PF02743": "HAMP-related",
    "PF00512": "DHp",
    "PF02518": "HATPase_c",
    "PF00672": "HAMP",
    "PF01261": "PAS",         # PAS domain — common O2/redox/light sensor
    "PF00989": "PAS",
    "PF13426": "PAC",
    "PF08447": "HAMP_like",
    "PF02362": "MHYT",        # membrane-integrated sensor (C1/CO/O2)
    "PF01NaN": "",
}

ENTREZ_BATCH = 100            # IDs per e-utils request
ENTREZ_DELAY = 0.35           # seconds between requests (NCBI limit: 3/s)


# ---------------------------------------------------------------------------
# NCBI fetch
# ---------------------------------------------------------------------------

def fetch_ncbi_summaries(protein_ids: list[str], email: str) -> dict[str, dict]:
    """Batch-fetch protein records from NCBI via efetch (GenPept format).

    Returns {protein_id: {gene, product, organism, length}} for each ID.
    IDs with no record are absent from the result.
    """
    Entrez.email = email
    results: dict[str, dict] = {}

    for i in range(0, len(protein_ids), ENTREZ_BATCH):
        batch = protein_ids[i : i + ENTREZ_BATCH]
        try:
            handle = Entrez.efetch(
                db="protein",
                id=",".join(batch),
                rettype="gb",
                retmode="text",
            )
            records = list(SeqIO.parse(handle, "genbank"))
            handle.close()

            for rec in records:
                pid = rec.id.split(".")[0] + "." + rec.id.split(".")[-1] if "." in rec.id else rec.id
                # Try to match back to input IDs
                matched = next((b for b in batch if rec.id.startswith(b.split(".")[0])), rec.id)

                gene = ""
                product = rec.description
                for feat in rec.features:
                    if feat.type == "Protein":
                        product = feat.qualifiers.get("product", [product])[0]
                    if feat.type == "CDS":
                        gene = feat.qualifiers.get("gene", [""])[0]

                results[matched] = {
                    "gene":     gene,
                    "product":  product,
                    "organism": rec.annotations.get("organism", ""),
                    "length":   len(rec.seq),
                }
        except Exception as e:
            print(f"  [warn] NCBI batch {i//ENTREZ_BATCH}: {e}")

        time.sleep(ENTREZ_DELAY)

    return results


# ---------------------------------------------------------------------------
# Sensor type inference
# ---------------------------------------------------------------------------

def infer_sensor_type(product: str, gene: str, pfam_hits: set[str],
                      diamond_stitle: str = "") -> str:
    """Classify HK sensor type from product description, gene name, and Pfam domains.

    Uses word-boundary regex matching to avoid substring false positives
    (e.g., 'kinA' matching inside 'histidine kinase'). Also uses the
    DIAMOND Swiss-Prot best-hit title (much more informative than NCBI
    generic descriptions) and a curated gene-name lookup table.
    """
    # 1. Gene-name lookup from DIAMOND title (e.g. "GN=phoQ" → phosphate)
    gn_match = re.search(r'\bGN=(\w+)\b', diamond_stitle)
    if gn_match:
        gn = gn_match.group(1).lower()
        if gn in GENE_SENSOR_MAP:
            return GENE_SENSOR_MAP[gn]

    # 2. Known gene names from NCBI gene field or product name
    gene_lower = gene.lower()
    if gene_lower in GENE_SENSOR_MAP:
        return GENE_SENSOR_MAP[gene_lower]

    # 3. Keyword regex on combined text (NCBI product + DIAMOND title)
    text = (product + " " + gene + " " + diamond_stitle).lower()

    for sensor_type, patterns in SENSOR_KEYWORDS:
        for pat in patterns:
            # Patterns starting with \b are already regex; others need word boundaries
            if re.search(pat, text, re.IGNORECASE):
                return sensor_type

    # Fall back to Pfam-based inference for sensor domains
    for pfam, label in SENSOR_PFAM.items():
        if pfam in pfam_hits and label not in ("DHp", "HATPase_c", "HAMP"):
            return f"Pfam:{pfam}({label})"

    return "uncharacterized"


# ---------------------------------------------------------------------------
# Load supporting data
# ---------------------------------------------------------------------------

def load_diamond_annotation(ann_path: str) -> dict[str, dict]:
    """Load DIAMOND Swiss-Prot annotation. Returns {protein_id: {hit, pident, stitle}}."""
    if not Path(ann_path).exists():
        return {}
    df = pd.read_csv(ann_path, sep="\t", header=None,
                     names=["qseqid", "sseqid", "pident", "length",
                            "qcovhsp", "evalue", "bitscore", "stitle"])
    out = {}
    for _, row in df.iterrows():
        out[row["qseqid"]] = {
            "diamond_hit":    row["sseqid"],
            "diamond_pident": row["pident"],
            "diamond_stitle": row["stitle"],
        }
    return out


def load_pfam_hits(domtbl_path: str, protein_ids: set[str]) -> dict[str, set[str]]:
    """Return {protein_id: set_of_pfam_accessions} for the given protein IDs."""
    hits: dict[str, set[str]] = {pid: set() for pid in protein_ids}
    with open(domtbl_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            pid = parts[0]
            if pid in hits:
                pfam = parts[4].split(".")[0]
                hits[pid].add(pfam)
    return hits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--screen_tsv",   required=True,
                        help="results/chimera_screen/hamp_chimera_screen.tsv")
    parser.add_argument("--hk_annotation", required=True,
                        help="results/annotation/hk_annotation.tsv")
    parser.add_argument("--rr_annotation", required=True,
                        help="results/annotation/rr_annotation.tsv (for RR function lookup)")
    parser.add_argument("--domtbl",        required=True,
                        help="results/domains/hk_reps_domtbl.txt")
    parser.add_argument("--output",        required=True)
    parser.add_argument("--email",         default="researcher@example.com",
                        help="Email for NCBI Entrez (required by NCBI)")
    parser.add_argument("--all_candidates", action="store_true",
                        help="Annotate all 2471 HAMP candidates, not just screened top 50")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Collect all unique protein IDs to annotate
    # ------------------------------------------------------------------
    screen_df = pd.read_csv(args.screen_tsv, sep="\t")
    protein_ids = sorted(set(screen_df["acceptor"]) | set(screen_df["donor"]))
    print(f"Annotating {len(protein_ids)} unique proteins from chimera screen")

    # ------------------------------------------------------------------
    # 2. Load existing DIAMOND annotations
    # ------------------------------------------------------------------
    diamond_hk = load_diamond_annotation(args.hk_annotation)
    diamond_rr = load_diamond_annotation(args.rr_annotation)
    print(f"Loaded {len(diamond_hk)} HK + {len(diamond_rr)} RR DIAMOND annotations")

    # ------------------------------------------------------------------
    # 3. Load Pfam domain hits for sensor type inference
    # ------------------------------------------------------------------
    pfam_hits = load_pfam_hits(args.domtbl, set(protein_ids))
    print(f"Loaded Pfam hits for {sum(1 for v in pfam_hits.values() if v)} proteins")

    # ------------------------------------------------------------------
    # 4. Fetch NCBI protein records
    # ------------------------------------------------------------------
    print(f"Fetching NCBI protein records (batch size {ENTREZ_BATCH}) ...")
    ncbi_data = fetch_ncbi_summaries(protein_ids, args.email)
    print(f"  Retrieved {len(ncbi_data)} records from NCBI")

    # ------------------------------------------------------------------
    # 5. Build annotation table
    # ------------------------------------------------------------------
    rows = []
    for pid in protein_ids:
        ncbi = ncbi_data.get(pid, {})
        diamond = diamond_hk.get(pid, diamond_rr.get(pid, {}))
        pfam = pfam_hits.get(pid, set())

        gene    = ncbi.get("gene", "")
        product = ncbi.get("product", diamond.get("diamond_stitle", ""))
        org     = ncbi.get("organism", "")

        # Infer sensor type using NCBI description + DIAMOND Swiss-Prot title + Pfam
        sensor_type = infer_sensor_type(
            product, gene, pfam,
            diamond_stitle=diamond.get("diamond_stitle", ""),
        )

        # Pfam domain summary (comma-separated, sorted)
        pfam_str = ",".join(sorted(pfam)) if pfam else ""

        rows.append({
            "protein_id":     pid,
            "gene":           gene,
            "product":        product,
            "organism":       org,
            "length":         ncbi.get("length", ""),
            "sensor_type":    sensor_type,
            "pfam_domains":   pfam_str,
            "diamond_hit":    diamond.get("diamond_hit", ""),
            "diamond_pident": diamond.get("diamond_pident", ""),
            "diamond_stitle": diamond.get("diamond_stitle", ""),
        })

    ann_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 6. Merge annotations into screen TSV
    # ------------------------------------------------------------------
    ann_a = ann_df.rename(columns={c: f"acceptor_{c}" for c in ann_df.columns
                                   if c != "protein_id"}).rename(
                                   columns={"protein_id": "acceptor"})
    ann_b = ann_df.rename(columns={c: f"donor_{c}" for c in ann_df.columns
                                   if c != "protein_id"}).rename(
                                   columns={"protein_id": "donor"})

    merged = screen_df.merge(ann_a, on="acceptor", how="left") \
                      .merge(ann_b, on="donor", how="left")

    merged.to_csv(args.output, sep="\t", index=False)
    print(f"\nWritten: {args.output}")

    # ------------------------------------------------------------------
    # 7. Summary of sensor types
    # ------------------------------------------------------------------
    print("\n=== Acceptor sensor type distribution ===")
    print(ann_df["sensor_type"].value_counts().to_string())
    print("\n=== Top annotated chimera pairs ===")
    display_cols = [
        "acceptor", "donor", "junction_identity",
        "acceptor_gene", "acceptor_product",
        "donor_gene",    "donor_product",
        "acceptor_sensor_type",
    ]
    display_cols = [c for c in display_cols if c in merged.columns]
    print(merged[display_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
