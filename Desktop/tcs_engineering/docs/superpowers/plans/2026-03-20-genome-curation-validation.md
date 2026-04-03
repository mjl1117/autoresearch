# Genome Curation & Ground Truth Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pre-pipeline genome curation (genus capping + reference organism inclusion) and systematic ground truth validation against well_characterized_tcs.tsv, then update all documentation and regenerate the pipeline DAG.

**Architecture:** Option A — genome curation is a one-shot pre-pipeline script that writes `data/reference/curated_genome_list.txt`; the Snakefile reads this file at startup if it exists, falling back to `os.listdir` if not. Ground truth validation is a Snakemake rule at the end of the DAG that asserts pipeline outputs against known biology, producing a pass/fail report. This keeps curation decisions explicit, version-controlled, and traceable.

**Tech Stack:** Python 3.14, pandas, Snakemake, pytest, graphviz (dot command, for DAG PNG)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| **Create** | `scripts/curate_genome_set.py` | Genus-capped genome selection + mandatory reference organism inclusion |
| **Create** | `scripts/validate_ground_truth.py` | Assert pipeline outputs against well_characterized_tcs.tsv; emit PASS/FAIL/WARN per check |
| **Create** | `workflow/rules/validation.smk` | Snakemake rule wrapping validate_ground_truth.py |
| **Create** | `tests/test_curate_genome_set.py` | Unit tests for curation logic |
| **Create** | `tests/test_validate_ground_truth.py` | Unit tests for validation logic |
| **Modify** | `workflow/Snakefile` | Read curated_genome_list.txt at startup; include validation.smk; add validation output to rule all |
| **Modify** | `config/config.yaml` | Add curation parameters: max_per_genus, assembly_level_priority |
| **Overwrite** | `PIPELINE_DOCS.md` | Full rewrite: add Stage 0 (curation), Stage 17 (validation), update directory tree, update config table |
| **Regenerate** | `dag.png` | Snakemake --dag \| dot -Tpng |

---

## Task 1: Write curate_genome_set.py

**Files:**
- Create: `scripts/curate_genome_set.py`

### Design

`curate_genome_set.py` groups downloaded genomes by genus, applies a per-genus cap while always preserving RECOMMENDED_GENOMES accessions, and writes a curated accession list. Selection within an over-represented genus prioritises assembly completeness (Complete Genome > Chromosome > Scaffold > Contig), then sorts by accession for determinism.

The `RECOMMENDED_GENOMES` dict is the canonical reference list shared with `audit_species_diversity.py`. It must live in one place — define it in a new `scripts/tcs_constants.py` module that both scripts import.

**Files for this task:**
- Create: `scripts/tcs_constants.py`
- Create: `scripts/curate_genome_set.py`

- [ ] **Step 1: Create scripts/tcs_constants.py**

```python
# scripts/tcs_constants.py
"""Shared constants for TCS pipeline scripts."""

# NCBI RefSeq assembly accessions for organisms that should always be
# present in a broad TCS survey. Maintained here so audit_species_diversity.py
# and curate_genome_set.py reference the same list.
RECOMMENDED_GENOMES = {
    "Escherichia coli K-12 MG1655":       "GCF_000005845.2",
    "Bacillus subtilis 168":              "GCF_000009045.1",
    "Caulobacter crescentus NA1000":      "GCF_000022005.1",
    "Myxococcus xanthus DK1622":          "GCF_000012685.1",
    "Streptomyces coelicolor A3(2)":      "GCF_000203835.1",
    "Synechocystis sp. PCC 6803":         "GCF_000009725.1",
    "Rhodobacter capsulatus SB1003":      "GCF_000009485.1",
    "Vibrio harveyi BB120":               "GCF_000021505.1",
    "Thermotoga maritima MSB8":           "GCF_000008545.1",
    "Staphylococcus aureus MRSA252":      "GCF_000011505.1",
    "Rhodopseudomonas palustris CGA009":  "GCF_000195775.1",
    "Anabaena sp. PCC 7120":              "GCF_000009705.1",
}

ASSEMBLY_LEVEL_PRIORITY = {
    "Complete Genome": 0,
    "Chromosome":      1,
    "Scaffold":        2,
    "Contig":          3,
}

MAX_GENUS_FRACTION = 0.10  # flag threshold (audit only — not enforced in curation)
MIN_SPECIES_COUNT  = 5
```

- [ ] **Step 2: Update audit_species_diversity.py to import from tcs_constants**

In `scripts/audit_species_diversity.py`, replace the inline `RECOMMENDED_GENOMES` dict at line 38–63 with:

```python
from tcs_constants import RECOMMENDED_GENOMES, MAX_GENUS_FRACTION, MIN_SPECIES_COUNT
```

Remove the duplicate constant definitions from that file (lines 30–63).

- [ ] **Step 3: Write scripts/curate_genome_set.py**

```python
#!/usr/bin/env python3
"""Curate the genome set for the TCS pipeline.

Pre-pipeline step (Option A): produces data/reference/curated_genome_list.txt
which the Snakefile reads at startup. The curated list:
  - Caps over-represented genera at --max_per_genus (default 5)
  - Always includes all RECOMMENDED_GENOMES accessions regardless of cap
  - Within a capped genus, prefers: Complete Genome > Chromosome > Scaffold > Contig,
    then alphabetical accession for determinism
  - Writes a human-readable curation_report.txt alongside the list

Usage (run once before pipeline execution):
  python scripts/curate_genome_set.py \\
      --assembly_summary data/metadata/assembly_summary.txt \\
      --genome_dir data/genomes \\
      --output data/reference/curated_genome_list.txt \\
      --report data/reference/curation_report.txt \\
      --max_per_genus 5
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from tcs_constants import ASSEMBLY_LEVEL_PRIORITY, RECOMMENDED_GENOMES


def load_assembly_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", skiprows=2, header=None,
        usecols=[0, 7, 11, 15, 5],
        names=["accession", "organism_name", "assembly_level", "asm_name", "taxid"],
        low_memory=False,
    )
    df["full_id"] = df["accession"] + "_" + df["asm_name"]
    df["genus"] = df["organism_name"].str.split().str[0]
    df["level_rank"] = df["assembly_level"].map(ASSEMBLY_LEVEL_PRIORITY).fillna(99)
    return df


def get_downloaded_genomes(genome_dir: str) -> set:
    return {d for d in os.listdir(genome_dir) if d.startswith("GCF_")}


def match_to_assembly(downloaded: set, asm_df: pd.DataFrame) -> pd.DataFrame:
    return asm_df[
        asm_df["accession"].isin(downloaded) | asm_df["full_id"].isin(downloaded)
    ].copy()


def curate(matched: pd.DataFrame, max_per_genus: int) -> tuple[list, dict]:
    """Return (curated_accessions, report_dict).

    Always preserves RECOMMENDED_GENOMES. Caps remaining genomes per genus.
    Selection within a capped genus: best assembly level, then alphabetical accession.
    """
    protected = set(RECOMMENDED_GENOMES.values())

    # Sort: protected first, then by assembly level rank, then accession
    matched = matched.copy()
    matched["is_protected"] = matched["accession"].isin(protected)
    matched = matched.sort_values(
        ["genus", "is_protected", "level_rank", "accession"],
        ascending=[True, False, True, True],
    )

    selected = []
    genus_counts: dict[str, dict] = {}

    for genus, grp in matched.groupby("genus"):
        protected_rows = grp[grp["is_protected"]]
        other_rows = grp[~grp["is_protected"]]

        n_available = len(grp)
        n_protected = len(protected_rows)
        slots_remaining = max(0, max_per_genus - n_protected)

        # Write full_id (accession_asmname) so the Snakefile can match against
        # existing_folders directly with `g in curated_set`.
        kept = list(protected_rows["full_id"])
        kept += list(other_rows["full_id"].iloc[:slots_remaining])
        dropped = list(other_rows["full_id"].iloc[slots_remaining:])

        selected.extend(kept)
        genus_counts[genus] = {
            "available": n_available,
            "kept": len(kept),
            "dropped": len(dropped),
            "protected": n_protected,
        }

    return selected, genus_counts


def write_report(
    original: int,
    curated: list,
    genus_counts: dict,
    report_path: str,
    max_per_genus: int,
) -> None:
    lines = [
        "# Genome Set Curation Report",
        f"# max_per_genus = {max_per_genus}",
        "",
        f"Original genome count : {original}",
        f"Curated genome count  : {len(curated)}",
        f"Genomes removed       : {original - len(curated)}",
        "",
        f"{'Genus':<30} {'Available':>10} {'Kept':>6} {'Dropped':>8} {'Protected':>10}",
        "-" * 68,
    ]
    for genus, c in sorted(genus_counts.items(), key=lambda x: -x[1]["available"]):
        lines.append(
            f"{genus:<30} {c['available']:>10} {c['kept']:>6} {c['dropped']:>8} {c['protected']:>10}"
        )
    Path(report_path).write_text("\n".join(lines) + "\n")
    print(f"Curation report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly_summary", required=True)
    parser.add_argument("--genome_dir", required=True)
    parser.add_argument("--output", required=True,
                        help="Path to write curated_genome_list.txt")
    parser.add_argument("--report", required=True,
                        help="Path to write curation_report.txt")
    parser.add_argument("--max_per_genus", type=int, default=5,
                        help="Max genomes per genus (recommended reference organisms always kept)")
    args = parser.parse_args()

    asm = load_assembly_summary(args.assembly_summary)
    downloaded = get_downloaded_genomes(args.genome_dir)
    matched = match_to_assembly(downloaded, asm)

    if matched.empty:
        print("ERROR: no genomes matched assembly_summary.txt", file=sys.stderr)
        sys.exit(1)

    curated, genus_counts = curate(matched, args.max_per_genus)
    write_report(len(matched), curated, genus_counts, args.report, args.max_per_genus)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Write full_ids (e.g. GCF_000005845.2_ASM584v2) so the Snakefile can
    # match against existing_folders with a direct set lookup.
    out.write_text(
        "# Curated genome list — generated by curate_genome_set.py\n"
        f"# max_per_genus={args.max_per_genus}\n"
        f"# Format: full_id (accession_asmname) — matches data/genomes/ folder names\n"
        + "\n".join(curated)
        + "\n"
    )
    print(f"Curated list ({len(curated)} genomes): {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
cd /Users/matthew/Desktop/tcs_engineering
git add scripts/tcs_constants.py scripts/curate_genome_set.py scripts/audit_species_diversity.py
git commit -m "feat: add genome curation script and shared TCS constants module"
```

---

## Task 2: Write tests for curate_genome_set.py

**Files:**
- Create: `tests/test_curate_genome_set.py`

- [ ] **Step 1: Create tests/test_curate_genome_set.py**

```python
"""Tests for curate_genome_set.py curation logic."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from curate_genome_set import curate, match_to_assembly
from tcs_constants import RECOMMENDED_GENOMES


def _make_asm(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["full_id"] = df["accession"] + "_" + df["asm_name"]
    df["genus"] = df["organism_name"].str.split().str[0]
    from tcs_constants import ASSEMBLY_LEVEL_PRIORITY
    df["level_rank"] = df["assembly_level"].map(ASSEMBLY_LEVEL_PRIORITY).fillna(99)
    df["is_protected"] = df["accession"].isin(set(RECOMMENDED_GENOMES.values()))
    return df


def test_curate_caps_genus():
    """Genera with more than max_per_genus genomes are capped."""
    rows = [
        {"accession": f"GCF_00000{i}.1", "organism_name": "Pseudomonas aeruginosa",
         "assembly_level": "Complete Genome", "asm_name": f"asm{i}", "taxid": "287"}
        for i in range(10)
    ]
    matched = _make_asm(rows)
    curated, counts = curate(matched, max_per_genus=3)
    assert len(curated) == 3
    assert counts["Pseudomonas"]["kept"] == 3
    assert counts["Pseudomonas"]["dropped"] == 7


def test_curate_always_keeps_recommended():
    """RECOMMENDED_GENOMES accessions are never dropped regardless of cap."""
    protected_acc = list(RECOMMENDED_GENOMES.values())[0]  # e.g. GCF_000005845.2
    rows = [
        {"accession": protected_acc, "organism_name": "Escherichia coli",
         "assembly_level": "Complete Genome", "asm_name": "asm0", "taxid": "562"},
    ] + [
        {"accession": f"GCF_99990{i}.1", "organism_name": "Escherichia coli",
         "assembly_level": "Scaffold", "asm_name": f"asm{i}", "taxid": "562"}
        for i in range(1, 10)
    ]
    matched = _make_asm(rows)
    curated, _ = curate(matched, max_per_genus=1)
    # cap=1 but protected always kept; total should be 1 (the protected one only,
    # since cap=1 and protected counts as 1)
    assert protected_acc in curated


def test_curate_prefers_complete_genome():
    """Within a capped genus, Complete Genome is preferred over Scaffold."""
    rows = [
        {"accession": "GCF_000001.1", "organism_name": "Campylobacter jejuni",
         "assembly_level": "Scaffold", "asm_name": "asm1", "taxid": "197"},
        {"accession": "GCF_000002.1", "organism_name": "Campylobacter jejuni",
         "assembly_level": "Complete Genome", "asm_name": "asm2", "taxid": "197"},
        {"accession": "GCF_000003.1", "organism_name": "Campylobacter jejuni",
         "assembly_level": "Scaffold", "asm_name": "asm3", "taxid": "197"},
    ]
    matched = _make_asm(rows)
    curated, _ = curate(matched, max_per_genus=1)
    assert curated == ["GCF_000002.1"]


def test_curate_deterministic():
    """Same input always produces same output (sort is stable and deterministic)."""
    rows = [
        {"accession": f"GCF_00000{i}.1", "organism_name": "Bacillus cereus",
         "assembly_level": "Scaffold", "asm_name": f"asm{i}", "taxid": "1396"}
        for i in range(5)
    ]
    matched = _make_asm(rows)
    c1, _ = curate(matched, max_per_genus=2)
    c2, _ = curate(matched, max_per_genus=2)
    assert c1 == c2


def test_match_to_assembly_finds_full_id():
    """Matches downloaded full_id (accession_asmname) folders."""
    asm = pd.DataFrame([{
        "accession": "GCF_000005845.2", "organism_name": "Escherichia coli",
        "assembly_level": "Complete Genome", "asm_name": "ASM584v2", "taxid": "562"
    }])
    asm["full_id"] = asm["accession"] + "_" + asm["asm_name"]
    asm["genus"] = asm["organism_name"].str.split().str[0]
    from tcs_constants import ASSEMBLY_LEVEL_PRIORITY
    asm["level_rank"] = asm["assembly_level"].map(ASSEMBLY_LEVEL_PRIORITY).fillna(99)

    downloaded = {"GCF_000005845.2_ASM584v2"}
    result = match_to_assembly(downloaded, asm)
    assert len(result) == 1
```

- [ ] **Step 2: Run tests**

```bash
cd /Users/matthew/Desktop/tcs_engineering
tcs-env/bin/pytest tests/test_curate_genome_set.py -v
```

Expected: 5 PASSED (tests written before full implementation — some may fail until curate() handles edge cases; fix curate_genome_set.py to pass all).

- [ ] **Step 3: Commit**

```bash
git add tests/test_curate_genome_set.py
git commit -m "test: add unit tests for genome curation logic"
```

---

## Task 3: Write validate_ground_truth.py

**Files:**
- Create: `scripts/validate_ground_truth.py`

### Ground Truth Checks

The validator runs five independent checks against `well_characterized_tcs.tsv`:

| Check | Description | Pass condition |
|-------|-------------|----------------|
| GT-1 | HK annotation coverage | ≥ 1 HK rep hits each `working_in_user_system=yes` system by keyword |
| GT-2 | RR annotation coverage | Same for RR partners |
| GT-3 | Chimera candidate coverage | Each `working_in_user_system=yes` system appears in `chimera_candidates.tsv` as `known_tcs_system` |
| GT-4 | Classifier specificity | No known non-HK contaminants (FtsZ, HtpG, GyrB) in `hk_annotation.tsv` best hits |
| GT-5 | Phase coherence for working systems | NarXL and PhoRB candidates have `linker_phase_compatible=True` or `linker_validation_required=True` (not absent) |

Outputs:
- `results/validation/ground_truth_validation.tsv` — one row per check (check_id, status, message, n_pass, n_fail)
- `results/validation/validation_summary.txt` — human-readable

- [ ] **Step 1: Write scripts/validate_ground_truth.py**

```python
#!/usr/bin/env python3
"""Ground truth validation for TCS pipeline outputs.

Checks pipeline results against well_characterized_tcs.tsv.
Each check produces a PASS / WARN / FAIL result with a message.

Outputs:
  results/validation/ground_truth_validation.tsv
  results/validation/validation_summary.txt

Exit code: 0 if all checks PASS or WARN; 1 if any FAIL.
The pipeline should not be blocked by WARN — only FAIL halts downstream use.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


# ─── Contaminants that must NOT appear as top HK hits ──────────────────────────
KNOWN_CONTAMINANTS = ["ftsz", "htpg", "gyrb", "mutl", "hsph", "dnak"]


def load_reference(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def load_annotation(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["qseqid", "sseqid", "pident", "length",
                             "qcovhsp", "evalue", "bitscore", "stitle"])
    df["stitle_lower"] = df["stitle"].str.lower()
    return df


def load_candidates(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    return pd.read_csv(path, sep="\t")


def check_annotation_coverage(
    ref: pd.DataFrame,
    ann: pd.DataFrame | None,
    protein_type: str,  # "HK" or "RR"
    check_id: str,
) -> dict:
    """GT-1 / GT-2: each working system keyword should appear in annotation."""
    if ann is None:
        return {
            "check_id": check_id,
            "status": "WARN",
            "message": f"{protein_type} annotation not yet available",
            "n_pass": 0,
            "n_fail": 0,
        }

    kw_col = "hk_swiss_prot_keywords" if protein_type == "HK" else "rr_swiss_prot_keywords"
    working = ref[ref["working_in_user_system"] == "yes"]

    passed, failed = [], []
    for _, row in working.iterrows():
        keywords = [k.strip() for k in str(row[kw_col]).lower().split(";")
                    if k.strip() not in ("nan", "null", "")]
        if not keywords:
            continue
        hits = ann[ann["stitle_lower"].apply(
            lambda t: any(k in t for k in keywords)
        )]
        if not hits.empty:
            passed.append(row["system_name"])
        else:
            failed.append(row["system_name"])

    if failed:
        status = "WARN"  # WARN not FAIL — organism may not be in genome set
        msg = f"Not detected in {protein_type} annotation: {', '.join(failed)}"
    else:
        status = "PASS"
        msg = f"All working {protein_type} systems detected ({len(passed)})"

    return {"check_id": check_id, "status": status, "message": msg,
            "n_pass": len(passed), "n_fail": len(failed)}


def check_chimera_coverage(
    ref: pd.DataFrame,
    candidates: pd.DataFrame | None,
) -> dict:
    """GT-3: working systems should appear as known_tcs_system in candidates."""
    if candidates is None:
        return {"check_id": "GT-3", "status": "WARN",
                "message": "chimera_candidates.tsv not yet available",
                "n_pass": 0, "n_fail": 0}

    if "known_tcs_system" not in candidates.columns:
        return {"check_id": "GT-3", "status": "FAIL",
                "message": "known_tcs_system column missing from chimera_candidates.tsv — "
                           "run identify_chimera_targets.py with --reference_tcs",
                "n_pass": 0, "n_fail": 1}

    working_names = set(ref[ref["working_in_user_system"] == "yes"]["system_name"])
    found = set(candidates["known_tcs_system"].dropna().unique())
    present = working_names & found
    missing = working_names - found

    if missing:
        status = "WARN"
        msg = f"Working systems absent from chimera candidates: {', '.join(sorted(missing))}"
    else:
        status = "PASS"
        msg = f"All {len(present)} working systems present in chimera candidates"

    return {"check_id": "GT-3", "status": status, "message": msg,
            "n_pass": len(present), "n_fail": len(missing)}


def check_classifier_specificity(
    hk_ann: pd.DataFrame | None,
) -> dict:
    """GT-4: known non-HK contaminants must not appear in HK annotation."""
    if hk_ann is None:
        return {"check_id": "GT-4", "status": "WARN",
                "message": "HK annotation not yet available",
                "n_pass": 0, "n_fail": 0}

    contaminated = hk_ann[hk_ann["stitle_lower"].apply(
        lambda t: any(c in t for c in KNOWN_CONTAMINANTS)
    )]

    if not contaminated.empty:
        examples = contaminated["stitle"].head(3).tolist()
        return {"check_id": "GT-4", "status": "FAIL",
                "message": f"Contaminants in HK annotation ({len(contaminated)} hits): "
                           f"{examples}",
                "n_pass": 0, "n_fail": len(contaminated)}

    return {"check_id": "GT-4", "status": "PASS",
            "message": "No known non-HK contaminants in HK annotation",
            "n_pass": len(hk_ann), "n_fail": 0}


def check_phase_coherence(
    ref: pd.DataFrame,
    candidates: pd.DataFrame | None,
) -> dict:
    """GT-5: working system candidates must have phase information (not absent)."""
    if candidates is None:
        return {"check_id": "GT-5", "status": "WARN",
                "message": "chimera_candidates.tsv not yet available",
                "n_pass": 0, "n_fail": 0}

    required_cols = {"linker_phase_compatible", "linker_validation_required",
                     "known_tcs_system"}
    if not required_cols.issubset(candidates.columns):
        return {"check_id": "GT-5", "status": "FAIL",
                "message": f"Missing columns in chimera_candidates.tsv: "
                           f"{required_cols - set(candidates.columns)}",
                "n_pass": 0, "n_fail": 1}

    working_names = set(ref[ref["working_in_user_system"] == "yes"]["system_name"])
    working_cands = candidates[candidates["known_tcs_system"].isin(working_names)]

    if working_cands.empty:
        return {"check_id": "GT-5", "status": "WARN",
                "message": "No working-system candidates found to check phase coherence",
                "n_pass": 0, "n_fail": 0}

    no_phase = working_cands[
        working_cands["linker_phase_compatible"].isna() &
        working_cands["linker_validation_required"].isna()
    ]

    if not no_phase.empty:
        return {"check_id": "GT-5", "status": "WARN",
                "message": f"{len(no_phase)} working-system candidates have no phase data",
                "n_pass": len(working_cands) - len(no_phase),
                "n_fail": len(no_phase)}

    return {"check_id": "GT-5", "status": "PASS",
            "message": f"All {len(working_cands)} working-system candidates have phase data",
            "n_pass": len(working_cands), "n_fail": 0}


def write_outputs(results: list[dict], outdir: Path) -> bool:
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(outdir / "ground_truth_validation.tsv", sep="\t", index=False)

    lines = ["# Ground Truth Validation Report", ""]
    any_fail = False
    for r in results:
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(r["status"], "?")
        lines.append(f"[{r['status']:4s}] {r['check_id']}  {icon}  {r['message']}")
        if r["status"] == "FAIL":
            any_fail = True

    (outdir / "validation_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return any_fail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_tcs", required=True)
    parser.add_argument("--hk_annotation", required=True)
    parser.add_argument("--rr_annotation", required=True)
    parser.add_argument("--chimera_candidates", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    ref = load_reference(args.reference_tcs)
    hk_ann = load_annotation(args.hk_annotation)
    rr_ann = load_annotation(args.rr_annotation)
    candidates = load_candidates(args.chimera_candidates)

    results = [
        check_annotation_coverage(ref, hk_ann, "HK", "GT-1"),
        check_annotation_coverage(ref, rr_ann, "RR", "GT-2"),
        check_chimera_coverage(ref, candidates),
        check_classifier_specificity(hk_ann),
        check_phase_coherence(ref, candidates),
    ]

    any_fail = write_outputs(results, Path(args.outdir))
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/validate_ground_truth.py
git commit -m "feat: add ground truth validation script (GT-1 through GT-5 checks)"
```

---

## Task 4: Write tests for validate_ground_truth.py

**Files:**
- Create: `tests/test_validate_ground_truth.py`

- [ ] **Step 1: Create tests/test_validate_ground_truth.py**

```python
"""Tests for validate_ground_truth.py check functions."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from validate_ground_truth import (
    check_annotation_coverage,
    check_chimera_coverage,
    check_classifier_specificity,
    check_phase_coherence,
)


REF = pd.DataFrame([
    {"system_name": "NarXL", "working_in_user_system": "yes",
     "hk_swiss_prot_keywords": "narx", "rr_swiss_prot_keywords": "narl"},
    {"system_name": "PhoRB", "working_in_user_system": "yes",
     "hk_swiss_prot_keywords": "phor", "rr_swiss_prot_keywords": "phob"},
    {"system_name": "EnvZOmpR", "working_in_user_system": "no",
     "hk_swiss_prot_keywords": "envz", "rr_swiss_prot_keywords": "ompr"},
])


def _make_ann(titles: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({
        "qseqid": [f"p{i}" for i in range(len(titles))],
        "sseqid": [f"sp|X{i}|Y" for i in range(len(titles))],
        "pident": [80.0] * len(titles),
        "length": [300] * len(titles),
        "qcovhsp": [90.0] * len(titles),
        "evalue": [1e-10] * len(titles),
        "bitscore": [300.0] * len(titles),
        "stitle": titles,
    })
    df["stitle_lower"] = df["stitle"].str.lower()
    return df


def test_annotation_coverage_pass():
    ann = _make_ann(["Nitrate sensor NarX", "Phosphate sensor PhoR"])
    result = check_annotation_coverage(REF, ann, "HK", "GT-1")
    assert result["status"] == "PASS"
    assert result["n_pass"] == 2


def test_annotation_coverage_warn_missing():
    ann = _make_ann(["Nitrate sensor NarX"])  # PhoR missing
    result = check_annotation_coverage(REF, ann, "HK", "GT-1")
    assert result["status"] == "WARN"
    assert result["n_fail"] == 1


def test_annotation_coverage_none():
    result = check_annotation_coverage(REF, None, "HK", "GT-1")
    assert result["status"] == "WARN"


def test_chimera_coverage_pass():
    cands = pd.DataFrame({
        "known_tcs_system": ["NarXL", "PhoRB", None],
        "linker_phase_compatible": [True, True, None],
        "linker_validation_required": [False, False, None],
    })
    result = check_chimera_coverage(REF, cands)
    assert result["status"] == "PASS"


def test_chimera_coverage_missing_column():
    cands = pd.DataFrame({"protein_id": ["p1"]})
    result = check_chimera_coverage(REF, cands)
    assert result["status"] == "FAIL"


def test_classifier_specificity_pass():
    ann = _make_ann(["Histidine kinase NarX ECOLI", "Histidine kinase PhoR"])
    result = check_classifier_specificity(ann)
    assert result["status"] == "PASS"


def test_classifier_specificity_fail():
    ann = _make_ann(["Cell division protein FtsZ", "Histidine kinase NarX"])
    result = check_classifier_specificity(ann)
    assert result["status"] == "FAIL"
    assert result["n_fail"] == 1


def test_phase_coherence_pass():
    cands = pd.DataFrame({
        "known_tcs_system": ["NarXL", "PhoRB"],
        "linker_phase_compatible": [True, False],
        "linker_validation_required": [False, True],
    })
    result = check_phase_coherence(REF, cands)
    assert result["status"] == "PASS"


def test_phase_coherence_missing_phase_data():
    cands = pd.DataFrame({
        "known_tcs_system": ["NarXL"],
        "linker_phase_compatible": [None],
        "linker_validation_required": [None],
    })
    result = check_phase_coherence(REF, cands)
    assert result["status"] == "WARN"
```

- [ ] **Step 2: Run tests**

```bash
tcs-env/bin/pytest tests/test_validate_ground_truth.py -v
```

Expected: all PASSED.

- [ ] **Step 3: Commit**

```bash
git add tests/test_validate_ground_truth.py
git commit -m "test: add unit tests for ground truth validation checks"
```

---

## Task 5: Add validation.smk and update Snakefile

**Files:**
- Create: `workflow/rules/validation.smk`
- Modify: `workflow/Snakefile`
- Modify: `config/config.yaml`

- [ ] **Step 1: Create workflow/rules/validation.smk**

```python
rule validate_ground_truth:
    """Systematically check pipeline outputs against well_characterized_tcs.tsv.

    Runs five checks (GT-1 through GT-5) covering:
      - HK annotation detection of working systems (GT-1)
      - RR annotation detection of working systems (GT-2)
      - Chimera candidate coverage of working systems (GT-3)
      - HK classifier specificity — no known contaminants (GT-4)
      - Phase coherence data present for all working-system candidates (GT-5)

    Outputs PASS / WARN / FAIL per check. Exit 1 only on FAIL (not WARN).
    WARN indicates the organism may not be in the genome set.
    """
    input:
        reference_tcs="data/reference/well_characterized_tcs.tsv",
        hk_ann="results/annotation/hk_annotation.tsv",
        rr_ann="results/annotation/rr_annotation.tsv",
        candidates="results/chimera_targets/chimera_candidates.tsv"
    output:
        tsv="results/validation/ground_truth_validation.tsv",
        summary="results/validation/validation_summary.txt"
    shell:
        """
        python scripts/validate_ground_truth.py \
            --reference_tcs {input.reference_tcs} \
            --hk_annotation {input.hk_ann} \
            --rr_annotation {input.rr_ann} \
            --chimera_candidates {input.candidates} \
            --outdir results/validation
        """
```

- [ ] **Step 2: Update workflow/Snakefile**

Replace the genome list construction block (lines 32–34) with:

```python
# 3. Determine genome list
# If a curated list exists (run scripts/curate_genome_set.py first), use it.
# Curated list contains full_ids (e.g. GCF_000005845.2_ASM584v2) matching
# data/genomes/ folder names — so set intersection is a direct lookup.
# Otherwise fall back to all GCF_ folders in data/genomes/.
curated_list_path = "data/reference/curated_genome_list.txt"
if os.path.exists(curated_list_path):
    with open(curated_list_path) as _f:
        _curated = set(l.strip() for l in _f if l.strip() and not l.startswith("#"))
    genomes = [g for g in FTP_LOOKUP.keys() if g in set(existing_folders) and g in _curated]
    print(f"[Snakefile] Using curated genome list: {len(genomes)} genomes")
else:
    genomes = [g for g in FTP_LOOKUP.keys() if g in existing_folders]
    print(f"[Snakefile] No curated list found — using all {len(genomes)} genomes in data/genomes/")
    print("[Snakefile] Run: python scripts/curate_genome_set.py --assembly_summary "
          "data/metadata/assembly_summary.txt --genome_dir data/genomes "
          "--output data/reference/curated_genome_list.txt "
          "--report data/reference/curation_report.txt")
```

Add to includes section (after audit.smk line):
```python
include: "rules/validation.smk"           # Ground truth validation
```

Add to rule all inputs:
```python
        # 10. Ground truth validation
        "results/validation/validation_summary.txt",
```

- [ ] **Step 3: Update config/config.yaml**

Add curation parameters block:

```yaml
# Genome set curation (run curate_genome_set.py before the pipeline)
# Recommended reference organisms (see scripts/tcs_constants.py) are always kept.
max_per_genus: 5          # Cap per genus; 5 allows species-level diversity within genus
                          # Reduce to 3 for maximum phylogenetic breadth
```

Remove the unused `genome_limit: 500` parameter (it is not read by the Snakefile).

- [ ] **Step 4: Run Snakemake dry run to verify DAG**

```bash
cd /Users/matthew/Desktop/tcs_engineering
tcs-env/bin/snakemake --snakefile workflow/Snakefile --dry-run 2>&1 | tail -20
```

Expected: dry run lists jobs including `validate_ground_truth`. No errors.

- [ ] **Step 5: Commit**

```bash
git add workflow/rules/validation.smk workflow/Snakefile config/config.yaml
git commit -m "feat: add validation rule, curated genome list integration, update config"
```

---

## Task 6: Generate new dag.png

**Files:**
- Regenerate: `dag.png`

- [ ] **Step 1: Generate DAG**

```bash
cd /Users/matthew/Desktop/tcs_engineering
tcs-env/bin/snakemake --snakefile workflow/Snakefile --dag --dry-run 2>/dev/null \
    | dot -Tpng -Gdpi=150 -o dag.png
```

Expected: `dag.png` updated with new nodes (validate_ground_truth, and the curated-genome fallback path).

If `dot` is not on PATH:
```bash
/opt/homebrew/bin/dot -Tpng -Gdpi=150 -o dag.png <(tcs-env/bin/snakemake --snakefile workflow/Snakefile --dag --dry-run 2>/dev/null)
```

- [ ] **Step 2: Verify file**

```bash
ls -lh dag.png
file dag.png
```

Expected: `dag.png: PNG image data, ...`

- [ ] **Step 3: Commit**

```bash
git add dag.png
git commit -m "docs: regenerate pipeline DAG with curation and validation nodes"
```

---

## Task 7: Rewrite PIPELINE_DOCS.md

**Files:**
- Overwrite: `PIPELINE_DOCS.md`

This is a full rewrite. Key additions vs current version:

1. **Stage 0: Genome Set Curation** (new) — documents curate_genome_set.py, Option A rationale, how to run
2. **Stage 17: Ground Truth Validation** (new) — documents all 5 GT checks, pass/fail/warn semantics
3. **Updated Directory Structure** — add validation.smk, validation.smk, tcs_constants.py, curation outputs
4. **Updated Configuration table** — remove genome_limit, add max_per_genus
5. **Updated Bug Fixes** — add Bug Fix #7: pseudo-replication bias from uncapped genera
6. **Genome Selection Rationale** — explain phylogenetic stratification philosophy, why 5-per-genus

- [ ] **Step 1: Overwrite PIPELINE_DOCS.md**

Write the complete file. Required sections in order:

**Section 1 — Overview** (update genome count from 1,475 → curated N; update core outputs table to add validation row)

**Section 2 — Pre-run: Genome Curation (NEW)**
```
## Stage 0: Genome Set Curation (Pre-Pipeline Step)

Run once before the pipeline. Caps each genus at `max_per_genus` (default 5).
Recommended reference organisms (see `scripts/tcs_constants.py`) are always kept.

    python scripts/curate_genome_set.py \
        --assembly_summary data/metadata/assembly_summary.txt \
        --genome_dir data/genomes \
        --output data/reference/curated_genome_list.txt \
        --report data/reference/curation_report.txt \
        --max_per_genus 5

Outputs:
  data/reference/curated_genome_list.txt   — full_ids for pipeline (one per line)
  data/reference/curation_report.txt       — before/after counts per genus

If curated_genome_list.txt exists, the Snakefile reads it at startup. Otherwise
falls back to all GCF_ folders in data/genomes/ with a warning. Rationale: 83%
of the 1,475-genome set came from 3 genera (Pseudomonas 32%, Campylobacter 32%),
inflating cluster sizes and phase coherence scores. A 5-per-genus cap preserves
taxonomic signal while removing pseudo-replication (Laub 2007, Capra & Laub 2012).
```

**Section 3 — Running the Pipeline** (same as before; add curation step at top)

**Section 4 — Directory Structure** (add validation.smk, validation.smk, tcs_constants.py, data/reference/curated_genome_list.txt, data/reference/curation_report.txt, results/validation/)

**Section 5 — Stages 1–16** (copy from existing PIPELINE_DOCS.md unchanged)

**Section 6 — Stage 17: Ground Truth Validation (NEW)**
```
## Stage 17: Ground Truth Validation (`validation.smk`, `scripts/validate_ground_truth.py`)

**Input:** hk_annotation.tsv, rr_annotation.tsv, chimera_candidates.tsv, well_characterized_tcs.tsv
**Output:** results/validation/ground_truth_validation.tsv, results/validation/validation_summary.txt

Five automated checks against well_characterized_tcs.tsv:

| Check | Description | Pass condition |
|-------|-------------|----------------|
| GT-1 | HK annotation coverage | ≥1 HK rep hits each working_in_user_system=yes system |
| GT-2 | RR annotation coverage | Same for RR partners |
| GT-3 | Chimera candidate coverage | All working systems present as known_tcs_system |
| GT-4 | Classifier specificity | No FtsZ/HtpG/GyrB in HK annotation |
| GT-5 | Phase coherence data present | NarXL, PhoRB candidates have phase columns populated |

Status semantics:
  PASS — criterion met
  WARN — criterion unmet but not fatal (e.g. organism not in genome set)
  FAIL — criterion unmet and indicates a pipeline bug; exit code 1

Current working systems: NarXL (E. coli nitrate sensor) and PhoRB (E. coli phosphate sensor).
Both confirmed functional in user's expression system. GT-3 PASS validates that the chimera
scoring correctly identifies and prioritises these systems.
```

**Section 7 — Bug Fix #7: Pseudo-Replication (NEW)**
```
### 7. Pseudo-Replication from Uncapped Genera

Problem: 83% of original 1,475 genomes from 3 genera. Pseudomonas (32%) and
Campylobacter (32%) each contributed >10× the recommended maximum per genus.
Effect: cluster sizes for these genera are inflated, making their phase coherence
scores appear statistically robust when they merely reflect within-genus conservation.

Fix: curate_genome_set.py caps at max_per_genus=5. This does not affect biological
conclusions for under-represented genera. All RECOMMENDED_GENOMES are exempt from
the cap regardless.
```

**Section 8 — Configuration** (update table: remove genome_limit, add max_per_genus; document all existing params)

**Section 9 — Interpreting chimera_candidates.tsv** (copy from existing, add known_tcs_system and working_in_user_system columns)

**Section 10 — Ground Truth Validation Interpretation (NEW)**
```
## Interpreting Ground Truth Validation

After each pipeline run, check results/validation/validation_summary.txt.

A clean run looks like:
  [PASS] GT-1  ✓  All working HK systems detected (2)
  [PASS] GT-2  ✓  All working RR systems detected (2)
  [PASS] GT-3  ✓  All 2 working systems present in chimera candidates
  [PASS] GT-4  ✓  No known non-HK contaminants in HK annotation
  [PASS] GT-5  ✓  All 2 working-system candidates have phase data

WARN on GT-1/GT-2: the organism is not in the genome set. Add it and re-run.
FAIL on GT-4: the HK classifier is contaminated again. Inspect hk_annotation.tsv
              for FtsZ/HtpG hits. Check 01_detect_domains.py HK_DOMAINS_REQUIRED.
FAIL on GT-3 + known_tcs_system column missing: re-run identify_chimera_targets.py
              with --reference_tcs flag.
```

**Section 11 — Literature** (copy from existing unchanged)

**Section 12 — Dependencies** (copy from existing, add pytest row)

**Section 13 — Expected Outputs** (copy from existing, add results/validation/ subtree)

- [ ] **Step 2: Commit**

```bash
git add PIPELINE_DOCS.md
git commit -m "docs: full rewrite of PIPELINE_DOCS.md with curation and validation stages"
```

---

## Task 8: Run all tests

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/matthew/Desktop/tcs_engineering
tcs-env/bin/pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 2: Final dry run**

```bash
tcs-env/bin/snakemake --snakefile workflow/Snakefile --dry-run 2>&1 | grep -E "^[0-9]|error|Error"
```

Expected: job count printed, no errors.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: genome curation, ground truth validation, updated docs and DAG"
```

---

## Post-Implementation Checklist

- [ ] `curate_genome_set.py --help` works without error
- [ ] `validate_ground_truth.py --help` works without error
- [ ] All 5 pytest tests in `test_curate_genome_set.py` pass
- [ ] All 8 pytest tests in `test_validate_ground_truth.py` pass
- [ ] `dag.png` reflects the new validation and curation nodes
- [ ] `PIPELINE_DOCS.md` has Stage 0 and Stage 17
- [ ] `config/config.yaml` has `max_per_genus` and no stale `genome_limit`
- [ ] `audit_species_diversity.py` imports from `tcs_constants.py` (no duplicate dict)
- [ ] Snakemake dry-run reports `validate_ground_truth` in job list
