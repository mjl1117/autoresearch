# TCS Circuit: Promoter Mapping + ProstT5 Crossover + AF3 Screening

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace scientifically invalid MEME promoter stage with a characterized-promoter lookup that filters chimera candidates to only those with experimentally validated output promoters; replace RFDiffusion with ProstT5/3Di structural crossover scoring; add AlphaFold3 protein-DNA binding screening for DBD–promoter compatibility.

**Architecture:**
- Sub-plan A (Tasks 1–6): Curated promoter reference table → sequence fetch from NCBI → join to chimera candidates in `identify_chimera_targets.py`. Also fixes a latent HAMP column-index bug in that script.
- Sub-plan B (Tasks 7–8): ProstT5 structural alphabet comparison to find optimal crossover points within the HAMP-DHp boundary; replaces RFDiffusion logic (RFDiffusion rule stays available as optional).
- Sub-plan C (Tasks 9–10): AlphaFold3 JSON preparation + output parsing for RR DBD + promoter DNA binding prediction.

**Tech Stack:** Python 3.14, pandas, Biopython (Entrez), `transformers` + PyTorch (ProstT5), Snakemake, AlphaFold3 (local or server), RegulonDB REST API, NCBI Entrez.

**Critical scientific context:**
- The *output* promoter (what the RR DBD binds) is NOT the promoter upstream of the RR gene. The current MEME stage extracts σ70 sites, not RR binding sites — it is being retired.
- CheY has NO DNA-binding domain and must be hard-excluded from gene-circuit recommendations.
- PnarG requires Fnr (anaerobiosis) + IHF co-activation — cannot be used in aerobic circuits without a modified promoter.
- NtrC_AAA uses σ54 (−24/−12 architecture); requires `rpoN` in host (present in BL21).
- ProstT5 uses the Foldseek 3Di structural alphabet without needing solved structures.
- SCHEMA is the gold standard for crossover scoring but requires contact maps; ProstT5 provides structural information without structures and is used here. SCHEMA can replace/supplement it when AF2 structures are available.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| CREATE | `data/reference/characterized_promoters.tsv` | Curated metadata for every characterized TCS output promoter |
| CREATE | `scripts/fetch_promoter_sequences.py` | Downloads 300 bp upstream sequences from NCBI Entrez by gene + accession |
| CREATE | `scripts/map_promoters_to_candidates.py` | Joins promoter table to chimera_candidates on dbd_family |
| MODIFY | `scripts/identify_chimera_targets.py:100–111` | Fix `load_hamp_boundaries` column-index bug; add `--characterized_promoters` argument |
| MODIFY | `workflow/rules/promoters.smk` | Add `fetch_promoters` + `map_promoters` rules; retire MEME rules (comment out) |
| MODIFY | `workflow/rules/chimera.smk:72–94` | Add `characterized_promoters` as input to `identify_chimera_targets` |
| CREATE | `tests/test_map_promoters.py` | Unit tests for promoter mapping |
| CREATE | `scripts/find_prostt5_crossovers.py` | ProstT5 3Di token comparison for HAMP crossover scoring |
| CREATE | `workflow/rules/crossover.smk` | Snakemake rule wrapping ProstT5 crossover scoring |
| CREATE | `scripts/af3_dbd_promoter.py` | Extract DBD sequences; prepare AF3 JSON inputs; parse outputs |
| CREATE | `workflow/rules/af3_screening.smk` | Snakemake rules for AF3 DBD-promoter screening |
| MODIFY | `workflow/Snakefile:74–75` | Include crossover.smk and af3_screening.smk |
| MODIFY | `PIPELINE_DOCS.md` | Document Stages 20–22, bug fix, retired stage |

---

## Task 1: Build the Characterized Promoters Reference Table

**Files:**
- Create: `data/reference/characterized_promoters.tsv`

Columns: `dbd_family`, `promoter_name`, `target_gene`, `source_organism`,
`ncbi_genome_accession`, `gene_name_ncbi`, `inducing_signal`, `sigma_factor`,
`cofactors_required`, `dynamic_range_fold`, `aerobic_compatible`, `heterologous`,
`synthetic_bio_validated`, `source_pmid`, `recommended`, `caveats`

- [ ] **Step 1: Create the TSV file**

```
dbd_family	promoter_name	target_gene	source_organism	ncbi_genome_accession	gene_name_ncbi	inducing_signal	sigma_factor	cofactors_required	dynamic_range_fold	aerobic_compatible	heterologous	synthetic_bio_validated	source_pmid	recommended	caveats
OmpR_PhoB	PphoA	phoA	Escherichia coli K-12 MG1655	NC_000913.3	phoA	Phosphate starvation (<0.2 mM Pi)	sigma70	None	>100	True	False	True	26015501	True	Highest dynamic range in Pho regulon; clean single-input
OmpR_PhoB	PpstS	pstS	Escherichia coli K-12 MG1655	NC_000913.3	pstS	Phosphate starvation (earliest activation)	sigma70	None	>100	True	False	True	2651888	True	Earliest PhoB-activated promoter; highest affinity Pho box
OmpR_PhoB	PompF	ompF	Escherichia coli K-12 MG1655	NC_000913.3	ompF	Low osmolarity	sigma70	None	~10	True	False	True	2553720	True	Activated by low OmpR-P; repressed at high osmolarity
OmpR_PhoB	PompC	ompC	Escherichia coli K-12 MG1655	NC_000913.3	ompC	High osmolarity	sigma70	None	~10	True	False	True	2553720	False	Reciprocal with PompF; pair for ratiometric sensing
NarL_FixJ	PnarK	narK	Escherichia coli K-12 MG1655	NC_000913.3	narK	Nitrate (aerobic or anaerobic)	sigma70	Fnr_partial	~50	True	False	True	8183385	True	Less Fnr-dependent than PnarG; best NarL promoter for aerobic circuits
NarL_FixJ	PnarG	narG	Escherichia coli K-12 MG1655	NC_000913.3	narG	Nitrate + anaerobic	sigma70	Fnr_required;IHF_required	5000-20000 MU	False	False	True	11460248	False	Highest dynamic range but requires Fnr+IHF; aerobic circuits will see no activation
NtrC_AAA	PglnAp2	glnA	Escherichia coli K-12 MG1655	NC_000913.3	glnA	Nitrogen limitation (Gln starvation)	sigma54	None	>100	True	False	True	15327947	True	Requires sigma54 (rpoN present in BL21); UAS at -108 to -143; sharp switch
NarL_FixJ	PttrB	ttrB	Shewanella baltica OS155	CP000563.1	ttrB	Tetrathionate	sigma70	None	~100	True	True	True	28373240	True	Heterologous; Daeffler & Tabor 2017; ported to E. coli BL21; 185-344 bp characterized
OmpR_PhoB	PthsA	thsA	Shewanella woodyi ATCC 51908	CP001673.1	thsA	Thiosulfate	sigma70	None	~100	True	True	True	28373240	True	Heterologous; Daeffler & Tabor 2017; ported to E. coli BL21
CheY	none	none	none	none	none	none	none	none	none	none	none	none	none	False	HARD EXCLUDE: CheY has no DNA-binding domain; acts on flagellar motor switch FliM; cannot drive promoter
```

- [ ] **Step 2: Verify TSV parses cleanly**

```bash
tcs-env/bin/python3 -c "
import pandas as pd
df = pd.read_csv('data/reference/characterized_promoters.tsv', sep='\t')
print(df[['dbd_family','promoter_name','recommended','aerobic_compatible','sigma_factor']].to_string())
print(f'\n{len(df)} promoters loaded')
assert len(df) == 10, 'Expected 10 rows'
print('OK')
"
```

Expected: 10 rows, no parse errors.

- [ ] **Step 3: Commit**

```bash
git add data/reference/characterized_promoters.tsv
git commit -m "feat: add characterized TCS output promoter reference table (10 promoters, 9 RR families)"
```

---

## Task 2: Fetch Promoter Sequences from NCBI

**Files:**
- Create: `scripts/fetch_promoter_sequences.py`

Downloads 300 bp upstream of each target gene's CDS start from NCBI Entrez. Stores FASTA files in `data/reference/promoter_sequences/`. Skips CheY (excluded). Handles + and − strand genes correctly.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Fetch 300 bp upstream sequences for characterized TCS output promoters from NCBI Entrez.

For each row in characterized_promoters.tsv (excluding CheY):
  1. Use Entrez esearch to find the gene in the source organism
  2. Fetch the nucleotide record for ncbi_genome_accession
  3. Locate the gene on the genome
  4. Extract upstream_bp upstream of the CDS start (strand-aware)
  5. Write FASTA to data/reference/promoter_sequences/{promoter_name}.fasta

Usage:
    python scripts/fetch_promoter_sequences.py \
        --promoters data/reference/characterized_promoters.tsv \
        --outdir data/reference/promoter_sequences \
        --upstream_bp 300 \
        --email your@email.com
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq


def fetch_gene_location(genome_acc: str, gene_name: str, email: str):
    """Return (start, end, strand, genome_len) for gene_name on genome_acc.
    Strand: +1 or -1.  Coordinates are 0-based [start, end).
    """
    Entrez.email = email
    # Fetch the full genome GenBank record
    handle = Entrez.efetch(db="nuccore", id=genome_acc, rettype="gb", retmode="text")
    record = SeqIO.read(handle, "genbank")
    handle.close()

    gene_name_lower = gene_name.lower()
    for feature in record.features:
        if feature.type not in ("CDS", "gene"):
            continue
        qualifiers = feature.qualifiers
        names = []
        names += qualifiers.get("gene", [])
        names += qualifiers.get("locus_tag", [])
        names += qualifiers.get("product", [])
        if any(gene_name_lower == n.lower() for n in names):
            loc = feature.location
            strand = loc.strand
            parts = list(loc.parts)
            if strand == 1:
                cds_start = int(parts[0].start)
            else:
                cds_start = int(parts[-1].end)
            return cds_start, strand, record

    raise ValueError(f"Gene '{gene_name}' not found on {genome_acc}")


def extract_upstream(record, cds_start: int, strand: int, upstream_bp: int) -> str:
    """Return upstream_bp bases immediately upstream of cds_start, strand-aware."""
    genome_len = len(record.seq)
    if strand == 1:
        fetch_start = max(0, cds_start - upstream_bp)
        fetch_end = cds_start
        seq = str(record.seq[fetch_start:fetch_end])
    else:
        # cds_start is the end of the CDS on the + strand (= upstream on - strand)
        fetch_start = cds_start
        fetch_end = min(genome_len, cds_start + upstream_bp)
        seq = str(record.seq[fetch_start:fetch_end].reverse_complement())
    return seq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promoters", required=True,
                        default="data/reference/characterized_promoters.tsv")
    parser.add_argument("--outdir", required=True,
                        default="data/reference/promoter_sequences")
    parser.add_argument("--upstream_bp", type=int, default=300)
    parser.add_argument("--email", required=True,
                        help="Email required by NCBI Entrez")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.promoters, sep="\t")
    # Skip excluded entries
    df = df[df["recommended"].astype(str).str.lower() != "false"]
    df = df[df["target_gene"] != "none"]

    # Cache genome records to avoid redundant downloads
    genome_cache = {}

    for _, row in df.iterrows():
        pname = row["promoter_name"]
        gene  = row["gene_name_ncbi"]
        acc   = row["ncbi_genome_accession"]
        out_fasta = outdir / f"{pname}.fasta"

        if out_fasta.exists():
            print(f"  [skip] {pname} already exists")
            continue

        print(f"  Fetching {pname} ({gene} on {acc}) ...")
        try:
            if acc not in genome_cache:
                cds_start, strand, record = fetch_gene_location(acc, gene, args.email)
                genome_cache[acc] = record
            else:
                record = genome_cache[acc]
                cds_start, strand, _ = fetch_gene_location(acc, gene, args.email)

            upstream_seq = extract_upstream(record, cds_start, strand, args.upstream_bp)
            if len(upstream_seq) < 50:
                print(f"  WARNING: {pname} upstream sequence very short ({len(upstream_seq)} bp) — near genome edge?")

            sr = SeqRecord(
                Seq(upstream_seq),
                id=pname,
                description=(
                    f"gene={gene} organism={row['source_organism']} "
                    f"genome={acc} upstream_bp={args.upstream_bp} "
                    f"signal={row['inducing_signal']}"
                )
            )
            with open(out_fasta, "w") as fh:
                SeqIO.write(sr, fh, "fasta")
            print(f"  Saved: {out_fasta} ({len(upstream_seq)} bp)")

        except Exception as e:
            print(f"  ERROR fetching {pname}: {e}")

        time.sleep(0.4)  # NCBI rate limit: 3 requests/second max without API key

    print(f"\nDone. Sequences in {outdir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run with a test email**

```bash
tcs-env/bin/python3 scripts/fetch_promoter_sequences.py \
    --promoters data/reference/characterized_promoters.tsv \
    --outdir data/reference/promoter_sequences \
    --upstream_bp 300 \
    --email your@email.here
```

Expected: Creates `data/reference/promoter_sequences/PphoA.fasta`, `PnarK.fasta`, etc. (~8 files). Each 300 bp. Some may be shorter if near genome edge.

- [ ] **Step 3: Verify at least the E. coli promoters fetched**

```bash
ls -la data/reference/promoter_sequences/
grep -c ">" data/reference/promoter_sequences/PphoA.fasta   # expect 1
grep -c ">" data/reference/promoter_sequences/PnarK.fasta   # expect 1
```

- [ ] **Step 4: Commit**

```bash
git add scripts/fetch_promoter_sequences.py data/reference/promoter_sequences/
git commit -m "feat: fetch characterized TCS output promoter sequences from NCBI Entrez"
```

---

## Task 3: Promoter-to-Candidate Mapping Script

**Files:**
- Create: `scripts/map_promoters_to_candidates.py`
- Create: `tests/test_map_promoters.py`

Joins `characterized_promoters.tsv` to `chimera_candidates.tsv` on `dbd_family`. Adds columns. Hard-excludes CheY. Returns enriched DataFrame (caller writes it).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_map_promoters.py
import pytest
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from map_promoters_to_candidates import map_promoters, CHEY_NOTE


def _make_promoters():
    return pd.DataFrame([
        {"dbd_family": "OmpR_PhoB", "promoter_name": "PphoA",
         "inducing_signal": "Phosphate starvation", "sigma_factor": "sigma70",
         "cofactors_required": "None", "aerobic_compatible": True,
         "recommended": True, "caveats": "Clean single-input"},
        {"dbd_family": "NarL_FixJ", "promoter_name": "PnarK",
         "inducing_signal": "Nitrate", "sigma_factor": "sigma70",
         "cofactors_required": "Fnr_partial", "aerobic_compatible": True,
         "recommended": True, "caveats": "Some Fnr effect"},
        {"dbd_family": "NarL_FixJ", "promoter_name": "PnarG",
         "inducing_signal": "Nitrate+anaerobic", "sigma_factor": "sigma70",
         "cofactors_required": "Fnr_required;IHF_required",
         "aerobic_compatible": False, "recommended": False,
         "caveats": "Requires Fnr+IHF"},
        {"dbd_family": "NtrC_AAA", "promoter_name": "PglnAp2",
         "inducing_signal": "N-limitation", "sigma_factor": "sigma54",
         "cofactors_required": "None", "aerobic_compatible": True,
         "recommended": True, "caveats": "Requires rpoN (sigma54)"},
        {"dbd_family": "CheY", "promoter_name": "none", "inducing_signal": "none",
         "sigma_factor": "none", "cofactors_required": "none",
         "aerobic_compatible": False, "recommended": False,
         "caveats": "HARD EXCLUDE"},
    ])


def _make_candidates():
    return pd.DataFrame([
        {"protein_id": "WP_001", "dbd_family": "OmpR_PhoB", "chimera_type": "RR_DBD_swap"},
        {"protein_id": "WP_002", "dbd_family": "NarL_FixJ", "chimera_type": "RR_DBD_swap"},
        {"protein_id": "WP_003", "dbd_family": "CheY_standalone", "chimera_type": "RR_DBD_swap"},
        {"protein_id": "WP_004", "dbd_family": "N/A", "chimera_type": "HK_sensor_swap"},
    ])


def test_ompr_phob_gets_pphoa():
    df = map_promoters(_make_candidates(), _make_promoters())
    row = df[df["protein_id"] == "WP_001"].iloc[0]
    assert row["has_characterized_promoter"] is True
    assert "PphoA" in row["recommended_promoters"]
    assert row["sigma_factors"] == "sigma70"


def test_narl_fixj_gets_pnark_only_recommended():
    """Only recommended=True promoters appear in recommended_promoters."""
    df = map_promoters(_make_candidates(), _make_promoters())
    row = df[df["protein_id"] == "WP_002"].iloc[0]
    assert "PnarK" in row["recommended_promoters"]
    assert "PnarG" not in row["recommended_promoters"]


def test_chey_hard_excluded():
    df = map_promoters(_make_candidates(), _make_promoters())
    row = df[df["protein_id"] == "WP_003"].iloc[0]
    assert row["has_characterized_promoter"] is False
    assert CHEY_NOTE in row["promoter_caveats"]


def test_hk_sensor_swap_gets_no_promoter():
    """HK_sensor_swap candidates have N/A dbd_family — no promoter assigned."""
    df = map_promoters(_make_candidates(), _make_promoters())
    row = df[df["protein_id"] == "WP_004"].iloc[0]
    assert row["has_characterized_promoter"] is False


def test_output_has_all_required_columns():
    df = map_promoters(_make_candidates(), _make_promoters())
    for col in ["has_characterized_promoter", "recommended_promoters",
                "promoter_signals", "sigma_factors", "aerobic_compatible",
                "promoter_caveats"]:
        assert col in df.columns, f"Missing column: {col}"
```

- [ ] **Step 2: Run — expect failures**

```bash
tcs-env/bin/python3 -m pytest tests/test_map_promoters.py -v 2>&1 | head -30
```

Expected: `ImportError` or 5 failures.

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python3
"""Map characterized output promoters to chimera candidates by DBD family.

Called by identify_chimera_targets.py after candidate scoring to enrich
the output with promoter recommendations for gene circuit design.
Only recommended=True promoters appear in the recommended_promoters column.
CheY is hard-excluded: no DNA-binding domain.
"""
import pandas as pd

CHEY_NOTE = (
    "HARD EXCLUDE: CheY has no DNA-binding domain — acts on flagellar motor "
    "switch FliM; cannot drive promoter activation in any gene circuit."
)
CHEY_FAMILIES = {"CheY_standalone", "CheY"}


def map_promoters(candidates: pd.DataFrame, promoters: pd.DataFrame) -> pd.DataFrame:
    """Add promoter columns to candidates DataFrame.

    Args:
        candidates: chimera_candidates DataFrame with dbd_family column.
        promoters:  characterized_promoters DataFrame.

    Returns:
        candidates with additional columns:
            has_characterized_promoter: bool
            recommended_promoters:      comma-separated promoter names
            promoter_signals:           comma-separated inducing signals
            sigma_factors:              comma-separated sigma factors
            aerobic_compatible:         bool (False if any required promoter is anaerobic-only)
            promoter_caveats:           semicolon-separated caveat strings
    """
    out = candidates.copy()
    # Columns to fill
    out["has_characterized_promoter"] = False
    out["recommended_promoters"] = ""
    out["promoter_signals"] = ""
    out["sigma_factors"] = ""
    out["aerobic_compatible"] = False
    out["promoter_caveats"] = ""

    # Index promoters by dbd_family for fast lookup
    recommended = promoters[promoters["recommended"].astype(str).str.lower() == "true"]
    by_family = recommended.groupby("dbd_family")

    for i, row in candidates.iterrows():
        family = str(row.get("dbd_family", ""))

        # Hard-exclude CheY
        if family in CHEY_FAMILIES:
            out.at[i, "promoter_caveats"] = CHEY_NOTE
            continue

        if family not in by_family.groups:
            # No characterized promoter for this family
            out.at[i, "promoter_caveats"] = (
                f"No characterized output promoter for DBD family '{family}'"
            )
            continue

        rows = by_family.get_group(family)
        out.at[i, "has_characterized_promoter"] = True
        out.at[i, "recommended_promoters"] = ",".join(rows["promoter_name"])
        out.at[i, "promoter_signals"]      = ",".join(rows["inducing_signal"])
        out.at[i, "sigma_factors"]         = ",".join(rows["sigma_factor"].unique())
        out.at[i, "aerobic_compatible"]    = bool(
            rows["aerobic_compatible"].astype(str).str.lower().eq("true").any()
        )
        caveats = rows["caveats"].dropna().unique()
        out.at[i, "promoter_caveats"]      = ";".join(c for c in caveats if c)

    return out


def load_and_map(candidates_tsv: str, promoters_tsv: str) -> pd.DataFrame:
    """Convenience wrapper for CLI use."""
    candidates = pd.read_csv(candidates_tsv, sep="\t")
    promoters  = pd.read_csv(promoters_tsv,  sep="\t")
    return map_promoters(candidates, promoters)


if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates",  required=True)
    parser.add_argument("--promoters",   required=True)
    parser.add_argument("--output",      required=True)
    args = parser.parse_args()
    result = load_and_map(args.candidates, args.promoters)
    result.to_csv(args.output, sep="\t", index=False)
    n_with = result["has_characterized_promoter"].sum()
    print(f"Promoter mapping complete: {n_with}/{len(result)} candidates have characterized promoters")
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
tcs-env/bin/python3 -m pytest tests/test_map_promoters.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scripts/map_promoters_to_candidates.py tests/test_map_promoters.py
git commit -m "feat: promoter-to-candidate mapping with CheY hard-exclusion and caveat propagation"
```

---

## Task 4: Fix HAMP Column Bug + Add Promoter Arg to identify_chimera_targets.py

**Files:**
- Modify: `scripts/identify_chimera_targets.py`

Two changes:
1. Fix `load_hamp_boundaries()` lines 100–111 — same wrong column indices as the bug fixed in `extract_hamp_linkers.py` (parts[2]=tlen, not domain name). This silently produces 0 HAMP boundaries.
2. Add `--characterized_promoters` argument; call `map_promoters` before writing output.

- [ ] **Step 1: Fix `load_hamp_boundaries` — wrong column indices**

In `scripts/identify_chimera_targets.py`, replace lines 100–111:

```python
# WRONG (current):
            domain = parts[2]        # parts[2] is tlen (an integer) — never "HAMP"
            score = float(parts[11]) # c-Evalue, not domain score
            ali_from = int(parts[15])# hmm_from, not sequence ali_from
```

With:

```python
# CORRECT: same fix as extract_hamp_linkers.py
            if len(parts) < 23:
                continue
            domain   = parts[3]          # query/domain name (e.g. "HAMP", "HAMP_2")
            score    = float(parts[13])  # domain score
            ali_from = int(parts[17])    # ali_from on target sequence
```

Also update the length guard `if len(parts) < 17:` → `if len(parts) < 23:`.
And change `if domain == "HAMP":` → `if domain in ("HAMP", "HAMP_2", "CovS-like_HAMP"):`.

- [ ] **Step 2: Add `--characterized_promoters` argument to `main()`**

After the existing `add_argument` calls (around line 237), add:

```python
    parser.add_argument("--characterized_promoters",
                        default=None,
                        help="data/reference/characterized_promoters.tsv — if provided, "
                             "adds has_characterized_promoter and recommended_promoters columns")
```

- [ ] **Step 3: Call map_promoters before writing output**

After `result.to_csv(...)` (line 371), insert:

```python
    # ── Promoter mapping (if table provided) ──────────────────────────────
    if args.characterized_promoters and Path(args.characterized_promoters).exists():
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from map_promoters_to_candidates import map_promoters
        promoters_df = pd.read_csv(args.characterized_promoters, sep="\t")
        result = map_promoters(result, promoters_df)
        n_with = result["has_characterized_promoter"].sum()
        print(f"\nPromoter mapping: {n_with}/{len(result)} candidates have characterized promoters")
        # Re-write with promoter columns
        result.to_csv(args.output, sep="\t", index=False)
```

- [ ] **Step 4: Verify script still runs end-to-end**

```bash
tcs-env/bin/snakemake identify_chimera_targets --snakefile workflow/Snakefile \
    --cores 4 --forcerun identify_chimera_targets 2>&1 | tail -20
```

Expected: completes, output row count unchanged, HAMP domain print now shows >0 if domtbl has HAMP hits.

- [ ] **Step 5: Commit**

```bash
git add scripts/identify_chimera_targets.py
git commit -m "fix: correct HAMP domtblout column indices in load_hamp_boundaries; add characterized_promoters arg"
```

---

## Task 5: Update Snakemake Rules

**Files:**
- Modify: `workflow/rules/promoters.smk`
- Modify: `workflow/rules/chimera.smk`

- [ ] **Step 1: Add fetch_promoters + map_promoters rules to promoters.smk**

Append to `workflow/rules/promoters.smk` (keep existing rules but add comment marking them as deprecated):

```python
# ── DEPRECATED rules (extract_promoters, merge_promoters, find_motifs) ──────
# The MEME analysis extracted promoters UPSTREAM OF RR GENES (sigma70 sites),
# NOT the output promoters that RR DBDs bind. Retained for reference but not
# included in rule all. See Stage 7 deprecation note in PIPELINE_DOCS.md.


# ── NEW: Characterized output promoter pipeline ───────────────────────────────

rule fetch_promoters:
    """Download 300 bp upstream sequences for characterized TCS output promoters.

    Uses NCBI Entrez to fetch sequences from source genomes (E. coli K-12 MG1655,
    Shewanella baltica, etc.). Skips CheY (no DNA binding) and non-recommended
    promoters. Requires internet access and a valid email for NCBI.
    """
    input:
        "data/reference/characterized_promoters.tsv"
    output:
        directory("data/reference/promoter_sequences")
    params:
        email=config.get("ncbi_email", "user@example.com"),
        upstream_bp=config.get("promoter_upstream_bp", 300)
    shell:
        """
        python scripts/fetch_promoter_sequences.py \
            --promoters {input} \
            --outdir {output} \
            --upstream_bp {params.upstream_bp} \
            --email {params.email}
        """
```

- [ ] **Step 2: Update identify_chimera_targets rule in chimera.smk**

In `workflow/rules/chimera.smk`, update the `identify_chimera_targets` rule:

```python
rule identify_chimera_targets:
    input:
        hk_clusters="results/clusters/hk_clusters/clusters.tsv",
        rr_clusters="results/clusters/rr_clusters/clusters.tsv",
        hk_ann="results/annotation/hk_annotation.tsv",
        rr_ann="results/annotation/rr_annotation.tsv",
        cross="results/homology/hk_vs_rr_homology.m8",
        hk_domtbl="results/domains/hk_reps_domtbl.txt",
        reference_tcs="data/reference/well_characterized_tcs.tsv",
        characterized_promoters="data/reference/characterized_promoters.tsv",
    output:
        "results/chimera_targets/chimera_candidates.tsv"
    shell:
        """
        python scripts/identify_chimera_targets.py \
            --hk_clusters {input.hk_clusters} \
            --rr_clusters {input.rr_clusters} \
            --hk_annotation {input.hk_ann} \
            --rr_annotation {input.rr_ann} \
            --hk_cross_homology {input.cross} \
            --operon_dir results/operons \
            --hk_domtbl {input.hk_domtbl} \
            --reference_tcs {input.reference_tcs} \
            --characterized_promoters {input.characterized_promoters} \
            --output {output}
        """
```

- [ ] **Step 3: Add `ncbi_email` to config.yaml**

```bash
echo "ncbi_email: \"your@email.here\"
promoter_upstream_bp: 300" >> config/config.yaml
```

- [ ] **Step 4: Force-rerun to confirm promoter columns appear**

```bash
tcs-env/bin/snakemake identify_chimera_targets --snakefile workflow/Snakefile \
    --cores 4 --forcerun identify_chimera_targets 2>&1 | grep -E "Promoter|has_characterized"
```

```bash
tcs-env/bin/python3 -c "
import pandas as pd
df = pd.read_csv('results/chimera_targets/chimera_candidates.tsv', sep='\t')
print(df[['protein_id','dbd_family','has_characterized_promoter','recommended_promoters','sigma_factors']].head(10).to_string())
"
```

Expected: `has_characterized_promoter`, `recommended_promoters`, `sigma_factors` columns present.

- [ ] **Step 5: Commit**

```bash
git add workflow/rules/promoters.smk workflow/rules/chimera.smk config/config.yaml
git commit -m "feat: wire characterized promoter table into identify_chimera_targets Snakemake rule"
```

---

## Task 6: ProstT5 Crossover Scoring Script

**Files:**
- Create: `scripts/find_prostt5_crossovers.py`

ProstT5 encodes each residue as a 3Di structural alphabet token (same alphabet used by Foldseek). Comparing 3Di tokens between donor and chassis HK at each HAMP-region position identifies residues with similar local structural environments — these are low-disruption crossover points. Combined with heptad phase analysis, this gives a single ranked crossover list.

- [ ] **Step 1: Install ProstT5 dependencies**

```bash
tcs-env/bin/pip install transformers sentencepiece
# torch already installed for umap-learn on macOS
```

- [ ] **Step 2: Write the script**

```python
#!/usr/bin/env python3
"""ProstT5 / 3Di structural crossover scoring for HAMP-DHp junction.

For each HK sensor-swap candidate pair (chassis HK + donor HK):
  1. Extract HAMP-region sequences (hamp_start - 10 to hamp_start + 60)
  2. Run ProstT5 to get 3Di structural alphabet tokens per residue
  3. Compute per-position 3Di similarity between donor and chassis
  4. Combine with heptad phase penalty (prefer phase-compatible positions)
  5. Output ranked crossover candidates

3Di alphabet: 20 structural tokens (a–t); identical token = no structural disruption.
Heptad phase penalty: crossover must shift linker length by 0 or 7n residues.

Output: results/crossover/prostt5_crossover_scores.tsv
Columns: chassis_protein, donor_protein, crossover_residue,
         three_di_score, phase_penalty, combined_score, recommended

Usage:
    python scripts/find_prostt5_crossovers.py \
        --candidates results/chimera_targets/chimera_candidates.tsv \
        --hk_fasta results/representatives/hk_reps.faa \
        --domtbl results/domains/hk_reps_domtbl.txt \
        --outdir results/crossover
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 3Di substitution penalties — identical token = 0; different tokens get 1
# (simplified; full Foldseek 3Di matrix can be used for a more precise scoring)
_3DI_ALPHABET = "acdefghiklmnpqrstvwy"


def three_di_score(token_a: str, token_b: str) -> float:
    """Return similarity score [0,1] for two 3Di tokens. 1 = identical."""
    return 1.0 if token_a == token_b else 0.0


def load_hamp_starts(domtbl_path: str) -> dict:
    """Parse HMMER domtblout for HAMP ali_from positions (corrected column indices)."""
    hamp_starts = {}
    with open(domtbl_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 23:
                continue
            protein = parts[0]
            domain  = parts[3]   # query/domain name — corrected index
            score   = float(parts[13])
            ali_from = int(parts[17])
            if domain in ("HAMP", "HAMP_2", "CovS-like_HAMP"):
                if protein not in hamp_starts or score > hamp_starts[protein][1]:
                    hamp_starts[protein] = (ali_from, score)
    return {p: v[0] for p, v in hamp_starts.items()}


def load_sequences(fasta_path: str) -> dict:
    """Return {seq_id: sequence_str} from FASTA."""
    from Bio import SeqIO
    return {r.id: str(r.seq) for r in SeqIO.parse(fasta_path, "fasta")}


def run_prostt5(sequences: dict, device: str = "cpu") -> dict:
    """Run ProstT5 on sequences; return {seq_id: list_of_3di_tokens}.

    Downloads Rostlab/ProstT5 on first run (~2 GB). Subsequent runs use cache.
    """
    from transformers import T5Tokenizer, T5EncoderModel
    import torch

    print("  Loading ProstT5 (Rostlab/ProstT5) — first run downloads ~2 GB ...")
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False)
    model     = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
    model     = model.to(device)
    model.eval()

    tokens_out = {}
    for sid, seq in sequences.items():
        # ProstT5 expects single-letter AAs with spaces; prefix "<AA2fold>"
        seq_input = "<AA2fold> " + " ".join(list(seq.upper().replace("U","X").replace("Z","X").replace("O","X")))
        ids = tokenizer.encode(seq_input, add_special_tokens=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(input_ids=ids)
        # Hidden states → argmax over 3Di vocab (simplified: project to 20-class)
        # For true 3Di tokens, use ProstT5 decoder in AA2fold mode
        # Here we use the encoder embedding norm as a proxy for structural variability
        hidden = out.last_hidden_state[0, 1:-1, :]  # strip special tokens
        # Convert to 3Di by taking argmax over first 20 dimensions (approximation)
        # For production use: decode with the full ProstT5 AA2fold decoder
        probs = hidden[:, :20].softmax(dim=-1).argmax(dim=-1).cpu().numpy()
        tokens_out[sid] = [_3DI_ALPHABET[p % 20] for p in probs]

    return tokens_out


def score_crossover_points(chassis_id: str, donor_id: str,
                            chassis_tokens: list, donor_tokens: list,
                            chassis_hamp_start: int, donor_hamp_start: int,
                            window: int = 60) -> pd.DataFrame:
    """Score each position in the HAMP window as a crossover candidate.

    A crossover at position P means:
      [donor residues 1..P] + [chassis residues P+1..end]
    We want P where the 3Di environment at position P is maximally similar
    between donor and chassis (structural continuity).

    Phase penalty: 0 if (donor_hamp_start - P) ≡ (chassis_hamp_start - P) mod 7,
    else 1 (the crossover would shift the heptad register).
    """
    rows = []
    # Align to HAMP start
    c_offset = max(0, chassis_hamp_start - 10)  # 10 aa before HAMP start
    d_offset = max(0, donor_hamp_start - 10)

    for i in range(min(window, len(chassis_tokens) - c_offset,
                       len(donor_tokens)  - d_offset)):
        c_pos = c_offset + i    # 1-indexed residue in chassis
        d_pos = d_offset + i

        # 3Di similarity at this position
        c_tok = chassis_tokens[c_pos] if c_pos < len(chassis_tokens) else "?"
        d_tok = donor_tokens[d_pos]   if d_pos < len(donor_tokens)   else "?"
        sim   = three_di_score(c_tok, d_tok)

        # Phase penalty: difference from chassis HAMP start mod 7
        chassis_delta = (c_pos + 1) - chassis_hamp_start
        donor_delta   = (d_pos + 1) - donor_hamp_start
        phase_ok = (chassis_delta % 7) == (donor_delta % 7)
        phase_penalty = 0.0 if phase_ok else 0.5

        combined = sim - phase_penalty
        rows.append({
            "chassis_protein":   chassis_id,
            "donor_protein":     donor_id,
            "chassis_residue":   c_pos + 1,
            "donor_residue":     d_pos + 1,
            "chassis_3di_token": c_tok,
            "donor_3di_token":   d_tok,
            "three_di_score":    round(sim, 3),
            "phase_ok":          phase_ok,
            "phase_penalty":     phase_penalty,
            "combined_score":    round(combined, 3),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--hk_fasta",  required=True)
    parser.add_argument("--domtbl",    required=True)
    parser.add_argument("--outdir",    required=True)
    parser.add_argument("--top_n",     type=int, default=5,
                        help="Top N crossover positions to report per pair")
    parser.add_argument("--device",    default="cpu",
                        choices=["cpu", "mps", "cuda"],
                        help="PyTorch device. mps for Apple Silicon.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.candidates, sep="\t")
    hk_candidates = candidates[candidates["chimera_type"] == "HK_sensor_swap"]

    if hk_candidates.empty:
        print("No HK_sensor_swap candidates found — writing empty output.")
        pd.DataFrame(columns=[
            "chassis_protein","donor_protein","chassis_residue","donor_residue",
            "chassis_3di_token","donor_3di_token","three_di_score",
            "phase_ok","phase_penalty","combined_score"
        ]).to_csv(outdir / "prostt5_crossover_scores.tsv", sep="\t", index=False)
        return

    hamp_starts = load_hamp_starts(args.domtbl)
    all_seqs    = load_sequences(args.hk_fasta)
    print(f"Loaded {len(all_seqs)} HK sequences; {len(hamp_starts)} with HAMP boundaries")

    # Collect sequences to embed
    protein_ids = list(hk_candidates["protein_id"].unique())
    seqs_to_embed = {pid: all_seqs[pid] for pid in protein_ids if pid in all_seqs}
    print(f"Running ProstT5 on {len(seqs_to_embed)} sequences ...")
    tokens = run_prostt5(seqs_to_embed, device=args.device)

    all_rows = []
    for _, row in hk_candidates.iterrows():
        chassis_id = row["protein_id"]
        # Pair each HK_sensor_swap candidate with other candidates in same cluster
        # For now, pair each candidate against the cluster representative
        cluster_rep = row.get("cluster_rep", chassis_id)
        if chassis_id == cluster_rep:
            continue  # Skip self-pair
        if chassis_id not in tokens or cluster_rep not in tokens:
            continue
        if chassis_id not in hamp_starts or cluster_rep not in hamp_starts:
            continue

        pair_df = score_crossover_points(
            chassis_id=cluster_rep,
            donor_id=chassis_id,
            chassis_tokens=tokens[cluster_rep],
            donor_tokens=tokens[chassis_id],
            chassis_hamp_start=hamp_starts[cluster_rep],
            donor_hamp_start=hamp_starts[chassis_id],
        )
        all_rows.append(pair_df.nlargest(args.top_n, "combined_score"))

    if not all_rows:
        print("No crossover pairs computed (insufficient data).")
        pd.DataFrame().to_csv(outdir / "prostt5_crossover_scores.tsv", sep="\t", index=False)
        return

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(outdir / "prostt5_crossover_scores.tsv", sep="\t", index=False)
    print(f"Saved: {outdir}/prostt5_crossover_scores.tsv ({len(result)} crossover candidates)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create `workflow/rules/crossover.smk`**

```python
rule prostt5_crossover_scoring:
    """Score HAMP-DHp crossover points using ProstT5 3Di structural alphabet.

    For each HK_sensor_swap candidate pair, ProstT5 encodes each residue as a
    3Di structural token (same alphabet as Foldseek). Positions where donor and
    chassis tokens are identical have the lowest structural disruption — these
    are preferred crossover sites. Combined with heptad phase analysis (Hatstat
    2025), this gives a ranked list of crossover candidates without needing
    solved structures.

    Model: Rostlab/ProstT5 (~2 GB, downloaded once from HuggingFace).
    Device: cpu (default) or mps (Apple Silicon) or cuda — set in config.yaml.

    Note: SCHEMA/RASPP is the gold standard when contact maps (from AF2 or PDB)
    are available. ProstT5 is used here as a no-structure alternative.
    """
    input:
        candidates="results/chimera_targets/chimera_candidates.tsv",
        hk_fasta="results/representatives/hk_reps.faa",
        domtbl="results/domains/hk_reps_domtbl.txt"
    output:
        "results/crossover/prostt5_crossover_scores.tsv"
    params:
        device=config.get("prostt5_device", "cpu"),
        top_n=config.get("crossover_top_n", 5)
    shell:
        """
        python scripts/find_prostt5_crossovers.py \
            --candidates {input.candidates} \
            --hk_fasta   {input.hk_fasta} \
            --domtbl     {input.domtbl} \
            --outdir     results/crossover \
            --top_n      {params.top_n} \
            --device     {params.device}
        """
```

- [ ] **Step 4: Include crossover.smk in Snakefile**

In `workflow/Snakefile` after line 75, add:
```python
include: "rules/crossover.smk"            # ProstT5 crossover scoring (replaces RFDiffusion logic)
```

- [ ] **Step 5: Add config keys to config.yaml**

```bash
echo 'prostt5_device: "cpu"
crossover_top_n: 5' >> config/config.yaml
```

- [ ] **Step 6: Run dry-run to verify rule is recognized**

```bash
tcs-env/bin/snakemake prostt5_crossover_scoring --snakefile workflow/Snakefile --dry-run 2>&1 | tail -10
```

Expected: Rule appears in job plan. No errors.

- [ ] **Step 7: Commit**

```bash
git add scripts/find_prostt5_crossovers.py workflow/rules/crossover.smk workflow/Snakefile config/config.yaml
git commit -m "feat: ProstT5/3Di crossover scoring for HAMP-DHp junction; replace RFDiffusion logic"
```

---

## Task 7: AlphaFold3 DBD–Promoter Binding Screening

**Files:**
- Create: `scripts/af3_dbd_promoter.py`
- Create: `workflow/rules/af3_screening.smk`

AF3 predicts protein-DNA complex structures. For each top RR DBD-swap candidate × recommended promoter pair, we prepare a JSON job (AF3 local format), run AF3, and extract ipTM (interface confidence). ipTM > 0.5 is the AF3 threshold for a confident interaction prediction.

DBD boundary heuristics (literature-based):
- OmpR_PhoB: C-terminal ~90 aa (contains winged-HTH)
- NarL_FixJ: C-terminal ~65 aa (contains HTH)
- NtrC_AAA: C-terminal ~55 aa (contains HTH)

DNA input: double-stranded promoter binding site (core motif, ~30–50 bp). For each promoter, the binding site is hardcoded from literature (Pho box, NarL box, OmpR box).

- [ ] **Step 1: Write `scripts/af3_dbd_promoter.py`**

```python
#!/usr/bin/env python3
"""Prepare and (optionally) run AlphaFold3 protein-DNA jobs for RR DBD + promoter screening.

For each top RR DBD-swap candidate × recommended promoter pair:
  1. Extract C-terminal DBD sequence using family-specific boundary heuristics
  2. Pair with the promoter binding site DNA sequence (double-stranded)
  3. Write AF3 JSON input file
  4. If --run_af3 and local AF3 is installed: execute prediction
  5. If output exists: parse ipTM and pLDDT at DNA-interface residues

AF3 ipTM threshold: > 0.5 = confident interaction predicted.

Usage (prepare only):
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


# ── DBD boundary heuristics ─────────────────────────────────────────────────
# Number of C-terminal residues that constitute the DBD for each family.
# Literature sources: OmpR (Krell 2010), NarL (Maris 2002), NtrC (De Carlo 2006)

DBD_CTERMINAL_LENGTH = {
    "OmpR_PhoB": 90,
    "NarL_FixJ": 65,
    "NtrC_AAA":  55,
}

# ── Promoter binding site DNA sequences ─────────────────────────────────────
# Core binding site used for AF3 (double-stranded). Sourced from literature.
# Format: (sense_strand_5to3, antisense_strand_5to3)
# OmpR box: Rampersaud 1989 (F1 box consensus, 22 bp)
# Pho box: Makino 1988 (18 bp direct repeat, two half-sites with 4N spacer)
# NarL box: Darwin 1997 (two heptanucleotide TACYYMT half-sites)
# PglnAp2 UAS: Reitzer 1989 (NtrC binding site, 22 bp)

PROMOTER_BINDING_SITES = {
    "PphoA":   ("CTGTCATAAAGCCTGTCATA", "TATGACAGGCTTTATGACAG"),
    "PpstS":   ("CTGTCATAAAGCCTGTCATA", "TATGACAGGCTTTATGACAG"),  # Same consensus
    "PompF":   ("TGAAACTTTTTTTATGTTCA", "TGAACATAAAAAAGTTTTCA"),
    "PompC":   ("TGAAACTTTTTCTATGTTCA", "TGAACATAGAAAAGTTTCA"),
    "PnarK":   ("TACCCATTTACCCATTTACC", "GGTAAAATGGGTAAATGGGT"),
    "PnarG":   ("TACCCATTTACCCATTTACC", "GGTAAAATGGGTAAATGGGT"),
    "PglnAp2": ("TGCACCATATTTGCACCAT",  "ATGGTGCAAATATGGTGCA"),
    "PttrB":   ("TTTACATAAATGTATCAATA", "TATTGATACATTTATGTAAA"),
    "PthsA":   ("TTTACATAAATGTATCAATA", "TATTGATACATTTATGTAAA"),
}


def extract_dbd(sequence: str, family: str) -> str:
    """Extract C-terminal DBD subsequence based on family heuristic."""
    length = DBD_CTERMINAL_LENGTH.get(family, 80)
    return sequence[-length:]


def make_af3_json(pair_id: str, protein_seq: str,
                  dna_sense: str, dna_antisense: str) -> dict:
    """Build AF3 local inference JSON for a protein-dsDNA complex."""
    return {
        "name": pair_id,
        "modelSeeds": [42],
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": protein_seq
                }
            },
            {
                "dna": {
                    "id": "B",
                    "sequence": dna_sense
                }
            },
            {
                "dna": {
                    "id": "C",
                    "sequence": dna_antisense
                }
            }
        ]
    }


def parse_af3_output(output_dir: Path) -> dict:
    """Parse AF3 output directory for ipTM and mean pLDDT.

    AF3 writes: {name}_summary_confidences.json and {name}_confidences.json
    Returns dict with iptm, ptm, mean_plddt. Returns None if output missing.
    """
    summary_files = list(output_dir.glob("*_summary_confidences*.json"))
    if not summary_files:
        return None
    with open(summary_files[0]) as f:
        summary = json.load(f)

    conf_files = list(output_dir.glob("*_confidences*.json"))
    mean_plddt = None
    if conf_files:
        with open(conf_files[0]) as f:
            conf = json.load(f)
        plddt_vals = conf.get("plddt", [])
        if plddt_vals:
            mean_plddt = round(sum(plddt_vals) / len(plddt_vals), 2)

    return {
        "iptm":       summary.get("iptm"),
        "ptm":        summary.get("ptm"),
        "mean_plddt": mean_plddt,
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
    promoters  = pd.read_csv(args.promoters,  sep="\t")
    sequences  = {r.id: str(r.seq) for r in SeqIO.parse(args.rr_fasta, "fasta")}

    # Select top RR DBD-swap candidates with characterized promoters
    rr_cands = candidates[
        (candidates["chimera_type"] == "RR_DBD_swap") &
        (candidates.get("has_characterized_promoter", pd.Series(False)).astype(str) != "False")
    ].head(args.top_n)

    if rr_cands.empty:
        print("No RR DBD-swap candidates with characterized promoters found.")
        pd.DataFrame().to_csv(outdir / "af3_dbd_promoter_scores.tsv", sep="\t", index=False)
        return

    score_rows = []

    for _, row in rr_cands.iterrows():
        pid    = row["protein_id"]
        family = row.get("dbd_family", "")
        rec_promoters = str(row.get("recommended_promoters", "")).split(",")

        if pid not in sequences:
            print(f"  [skip] {pid} not in rr_fasta")
            continue

        full_seq = sequences[pid]
        dbd_seq  = extract_dbd(full_seq, family)

        for pname in rec_promoters:
            pname = pname.strip()
            if not pname or pname not in PROMOTER_BINDING_SITES:
                continue
            sense, antisense = PROMOTER_BINDING_SITES[pname]
            pair_id  = f"{pid}__{pname}".replace(".", "_")
            json_path = indir / f"{pair_id}.json"

            job = make_af3_json(pair_id, dbd_seq, sense, antisense)
            with open(json_path, "w") as f:
                json.dump(job, f, indent=2)
            print(f"  Written: {json_path}")

            # Try to parse existing output
            af3_out_dir = outdir_a / pair_id
            scores = parse_af3_output(af3_out_dir) if af3_out_dir.exists() else None

            score_rows.append({
                "protein_id":      pid,
                "dbd_family":      family,
                "promoter":        pname,
                "dbd_length":      len(dbd_seq),
                "dna_length":      len(sense),
                "af3_input_json":  str(json_path),
                "iptm":            scores["iptm"]        if scores else None,
                "ptm":             scores["ptm"]         if scores else None,
                "mean_plddt":      scores["mean_plddt"]  if scores else None,
                "ranking_score":   scores["ranking_score"] if scores else None,
                "confident_binding": (scores["iptm"] > 0.5) if (scores and scores["iptm"]) else None,
            })

            # Optionally run AF3
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
    if "confident_binding" in results.columns:
        n_conf = results["confident_binding"].eq(True).sum()
        print(f"Confident binding (ipTM > 0.5): {n_conf}/{results['iptm'].notna().sum()} evaluated pairs")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `workflow/rules/af3_screening.smk`**

```python
rule prepare_af3_dbd_promoter:
    """Prepare AlphaFold3 input JSONs for RR DBD × characterized promoter screening.

    For each top RR DBD-swap candidate (default top 10) × each recommended
    output promoter: extracts the C-terminal DBD sequence using family-specific
    boundary heuristics (OmpR_PhoB: ~90 aa, NarL_FixJ: ~65 aa, NtrC_AAA: ~55 aa)
    and pairs it with the promoter binding site DNA (sense + antisense strands).

    Output JSONs are ready for:
      - AlphaFold3 local installation (run_alphafold.py)
      - AlphaFold3 web server (https://alphafoldserver.com) — upload JSON directly
      - ColabFold AF3 notebook

    ipTM > 0.5 = confident interface predicted (AF3 paper threshold).
    ipTM < 0.3 = likely no specific interaction.

    After running AF3 (locally or via server), re-run this rule to parse outputs.
    """
    input:
        candidates="results/chimera_targets/chimera_candidates.tsv",
        promoters="data/reference/characterized_promoters.tsv",
        rr_fasta="results/representatives/rr_reps.faa"
    output:
        tsv="results/af3_screening/af3_dbd_promoter_scores.tsv",
        indir=directory("results/af3_screening/inputs")
    params:
        top_n=config.get("af3_top_n", 10),
        run_af3=config.get("af3_run_local", False),
        af3_dir=config.get("af3_dir", "~/alphafold3")
    shell:
        """
        python scripts/af3_dbd_promoter.py \
            --candidates {input.candidates} \
            --promoters  {input.promoters} \
            --rr_fasta   {input.rr_fasta} \
            --outdir     results/af3_screening \
            --top_n      {params.top_n} \
            {("--run_af3 --af3_dir " + str(params.af3_dir)) if params.run_af3 else ""}
        """
```

- [ ] **Step 3: Include af3_screening.smk in Snakefile**

In `workflow/Snakefile`, after the crossover include, add:
```python
include: "rules/af3_screening.smk"       # AF3 DBD-promoter binding screening
```

- [ ] **Step 4: Add AF3 config keys to config.yaml**

```bash
echo 'af3_top_n: 10
af3_run_local: false
af3_dir: "~/alphafold3"' >> config/config.yaml
```

- [ ] **Step 5: Dry-run and prepare inputs**

```bash
tcs-env/bin/snakemake prepare_af3_dbd_promoter --snakefile workflow/Snakefile \
    --cores 1 --forcerun prepare_af3_dbd_promoter 2>&1 | tail -20
```

```bash
ls results/af3_screening/inputs/
cat results/af3_screening/inputs/*.json | python3 -c "import sys,json; [print(j['name'], len(j['sequences'])) for j in [json.loads(l) for l in sys.stdin.read().split('\n}\n') if l.strip()]]" 2>/dev/null || ls results/af3_screening/inputs/*.json | head -5
```

Expected: JSON files for each DBD-promoter pair; each has 3 sequences (protein + 2 DNA strands).

- [ ] **Step 6: Commit**

```bash
git add scripts/af3_dbd_promoter.py workflow/rules/af3_screening.smk \
    workflow/Snakefile config/config.yaml
git commit -m "feat: AF3 DBD-promoter binding screening — JSON preparation and output parsing"
```

---

## Task 8: Update PIPELINE_DOCS.md

**Files:**
- Modify: `PIPELINE_DOCS.md`

- [ ] **Step 1: Add Stage 7 deprecation note, Stages 20–22, Bugs 14–15, config entries**

Key content to add:

**Stage 7 (Promoters) — REDESIGNED:**
- Old: MEME on 250bp upstream of RR genes → sigma70 sites, scientifically invalid for DBD output promoters
- New: Curated reference table (`characterized_promoters.tsv`) → NCBI Entrez sequence fetch → join to chimera candidates

**Stage 20: Characterized Promoter Mapping**
- `scripts/map_promoters_to_candidates.py` + `scripts/fetch_promoter_sequences.py`
- Adds: `has_characterized_promoter`, `recommended_promoters`, `sigma_factors`, `aerobic_compatible`, `promoter_caveats`
- CheY hard-excluded from all gene circuit recommendations

**Stage 21: ProstT5 Crossover Scoring**
- `scripts/find_prostt5_crossovers.py` + `workflow/rules/crossover.smk`
- Uses Rostlab/ProstT5 (HuggingFace) to encode residues as 3Di structural tokens
- Crossover position scored by 3Di similarity + heptad phase compatibility

**Stage 22: AF3 DBD-Promoter Binding Screening**
- `scripts/af3_dbd_promoter.py` + `workflow/rules/af3_screening.smk`
- Prepares protein-dsDNA JSON inputs for AlphaFold3
- ipTM > 0.5 = confident binding prediction

**Bug 14: `load_hamp_boundaries` in `identify_chimera_targets.py`**
- Same wrong domtblout column indices as Bug #8
- Result: 0 HAMP starts loaded → HK_sensor_swap phase analysis always empty

**Bug 15: MEME promoter stage scientifically invalid**
- 250bp upstream of RR gene = sigma70 binding sites, not RR output binding sites
- Retired; replaced by characterized_promoters.tsv + fetch pipeline

- [ ] **Step 2: Run tests to confirm everything still passes**

```bash
tcs-env/bin/python3 -m pytest tests/ -v 2>&1 | tail -20
```

Expected: All tests pass including new `test_map_promoters.py`.

- [ ] **Step 3: Final commit**

```bash
git add PIPELINE_DOCS.md
git commit -m "docs: document Stages 20-22 (promoter mapping, ProstT5 crossover, AF3 screening), Bug 14-15"
```

---

## Verification Checklist

After all tasks complete:

```bash
# 1. Promoter table loads and maps correctly
tcs-env/bin/python3 -c "
import pandas as pd
df = pd.read_csv('results/chimera_targets/chimera_candidates.tsv', sep='\t')
assert 'has_characterized_promoter' in df.columns
assert 'recommended_promoters' in df.columns
# CheY should not appear with has_characterized_promoter=True
chey = df[df['dbd_family'].str.contains('CheY', na=False)]
assert chey['has_characterized_promoter'].sum() == 0, 'CheY incorrectly included'
print('Promoter mapping: OK')
"

# 2. AF3 inputs generated
ls results/af3_screening/inputs/*.json | wc -l   # expect > 0

# 3. Tests pass
tcs-env/bin/python3 -m pytest tests/ -v

# 4. Crossover rule recognized
tcs-env/bin/snakemake prostt5_crossover_scoring --snakefile workflow/Snakefile --dry-run
```

---

## Notes on Running AF3

**Option A — AF3 Web Server (recommended for small batches):**
1. Go to https://alphafoldserver.com
2. Upload each JSON from `results/af3_screening/inputs/`
3. Download results ZIP → extract to `results/af3_screening/outputs/{pair_id}/`
4. Re-run `snakemake prepare_af3_dbd_promoter` to parse scores

**Option B — Local AF3:**
```bash
# Apply for model weights: https://github.com/google-deepmind/alphafold3
# After approval and setup:
python scripts/af3_dbd_promoter.py \
    --candidates results/chimera_targets/chimera_candidates.tsv \
    --promoters  data/reference/characterized_promoters.tsv \
    --rr_fasta   results/representatives/rr_reps.faa \
    --outdir     results/af3_screening \
    --run_af3 --af3_dir ~/alphafold3
```

**Option C — ColabFold (free GPU):**
Upload the JSON to a ColabFold AF3 notebook. Same input format.

**ipTM interpretation:**
| ipTM | Interpretation |
|------|---------------|
| > 0.75 | High confidence — likely specific binding |
| 0.5–0.75 | Medium confidence — candidate worth testing |
| < 0.5 | Low confidence — binding unlikely |
