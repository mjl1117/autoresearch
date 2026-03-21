# Genome Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the TCS pipeline from 10 to ~200 genomes by building a selection + download script, enabling HK_sensor_swap chimera candidates via cluster sizes ≥ 50.

**Architecture:** A new `select_expansion_genomes.py` reads the 481K-line local `assembly_summary.txt`, seeds from `RECOMMENDED_GENOMES`, guarantees Pseudomonas representation, then diversity-fills to 200 with a 5/genus cap. It writes a download manifest (new genomes only) and the updated `curated_genome_list.txt`. `00_download_genomes.py` gains a `--manifest` flag that downloads entries from that manifest using existing FTP logic.

**Tech Stack:** Python 3, pandas, pathlib, concurrent.futures (already used), tcs_constants (existing shared module), pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `scripts/select_expansion_genomes.py` | CREATE | Selection logic: seed RECOMMENDED_GENOMES → priority genus fill → diversity fill → write manifest + curated list |
| `scripts/00_download_genomes.py` | MODIFY | Add `--manifest` flag for manifest-driven download |
| `tests/test_select_expansion_genomes.py` | CREATE | Unit tests for selection logic |

---

## Codebase Orientation

**`scripts/curate_genome_set.py`** — existing post-download curation script. Follow its patterns exactly:
- `pd.read_csv(..., skiprows=2, header=None, usecols=[0,5,7,11,15], names=[...])` — **use this pattern**, NOT `skiprows=1`
- `full_id = accession + "_" + asm_name` — folder naming convention
- `genus = organism_name.str.split().str[0]`
- Imports `ASSEMBLY_LEVEL_PRIORITY, RECOMMENDED_GENOMES` from `tcs_constants`

**`scripts/tcs_constants.py`** — shared constants:
- `RECOMMENDED_GENOMES`: dict mapping organism name → accession (12 entries). Already covers Myxococcus, Streptomyces, Anabaena, Caulobacter, Rhodobacter. **Pseudomonas is absent** — this is the only priority genus that needs explicit selection.
- `ASSEMBLY_LEVEL_PRIORITY`: `{"Complete Genome": 0, "Chromosome": 1, "Scaffold": 2, "Contig": 3}`

**`scripts/00_download_genomes.py`** — existing download script. Key function:
```python
def download_genome(row, genome_dir):
    ftp = row["ftp_path"].rstrip("/")
    assembly = ftp.split("/")[-1]        # this IS the full_id
    outdir = genome_dir / assembly
    # downloads .fna, .faa, .gff into outdir
```
The `row` dict needs a `"ftp_path"` key. The manifest rows provide exactly this.

**Assembly summary column indices** (0-based, after `skiprows=2, header=None`):
```
0:  accession       5:  taxid          7:  organism_name
11: assembly_level  15: asm_name       19: ftp_path
```

**`tests/test_curate_genome_set.py`** — reference for test style. Uses `_make_matched()` helper to build minimal DataFrames. Follow this pattern.

---

## Task 1: `select_expansion_genomes.py` — core selection logic

**Files:**
- Create: `scripts/select_expansion_genomes.py`
- Test: `tests/test_select_expansion_genomes.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_select_expansion_genomes.py
"""Tests for select_expansion_genomes.py selection logic."""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from select_expansion_genomes import (
    parse_assembly_summary,
    select_priority,
    select_diversity,
)
from tcs_constants import ASSEMBLY_LEVEL_PRIORITY, RECOMMENDED_GENOMES


def _make_asm_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal assembly summary DataFrame matching parse_assembly_summary output."""
    df = pd.DataFrame(rows)
    df["full_id"] = df["accession"] + "_" + df["asm_name"]
    df["genus"] = df["organism_name"].str.split().str[0]
    df["level_rank"] = df["assembly_level"].map(ASSEMBLY_LEVEL_PRIORITY).fillna(99)
    df["already_downloaded"] = False
    return df


def test_select_priority_takes_best_quality():
    """Priority pass picks Complete Genome over Scaffold from same genus."""
    df = _make_asm_df([
        {"accession": "GCF_000001.1", "asm_name": "asm1",
         "organism_name": "Pseudomonas aeruginosa", "assembly_level": "Scaffold",
         "ftp_path": "ftp://ftp.ncbi.nlm.nih.gov/path/GCF_000001.1_asm1"},
        {"accession": "GCF_000002.1", "asm_name": "asm2",
         "organism_name": "Pseudomonas fluorescens", "assembly_level": "Complete Genome",
         "ftp_path": "ftp://ftp.ncbi.nlm.nih.gov/path/GCF_000002.1_asm2"},
    ])
    result = select_priority(df, priority_genera=["Pseudomonas"])
    assert len(result) == 1
    assert result.iloc[0]["accession"] == "GCF_000002.1"


def test_select_priority_prefers_already_downloaded():
    """Priority pass prefers already-downloaded genome over better-assembly-level one."""
    df = _make_asm_df([
        {"accession": "GCF_000001.1", "asm_name": "asm1",
         "organism_name": "Pseudomonas aeruginosa", "assembly_level": "Complete Genome",
         "ftp_path": "ftp://example/GCF_000001.1_asm1"},
        {"accession": "GCF_000002.1", "asm_name": "asm2",
         "organism_name": "Pseudomonas fluorescens", "assembly_level": "Scaffold",
         "ftp_path": "ftp://example/GCF_000002.1_asm2"},
    ])
    df.loc[df["accession"] == "GCF_000002.1", "already_downloaded"] = True
    result = select_priority(df, priority_genera=["Pseudomonas"])
    assert len(result) == 1
    assert result.iloc[0]["accession"] == "GCF_000002.1"


def test_select_priority_skips_absent_genus(capsys):
    """Priority pass logs warning and skips genus with no catalog entries."""
    df = _make_asm_df([
        {"accession": "GCF_000001.1", "asm_name": "asm1",
         "organism_name": "Pseudomonas aeruginosa", "assembly_level": "Complete Genome",
         "ftp_path": "ftp://example/GCF_000001.1_asm1"},
    ])
    result = select_priority(df, priority_genera=["Pseudomonas", "Myxococcus"])
    assert len(result) == 1  # Myxococcus absent, no crash
    captured = capsys.readouterr()
    assert "Myxococcus" in captured.out


def test_select_diversity_caps_genus():
    """Diversity fill caps each genus at max_per_genus."""
    rows = [
        {"accession": f"GCF_0000{i:02d}.1", "asm_name": f"asm{i}",
         "organism_name": "Vibrio harveyi", "assembly_level": "Complete Genome",
         "ftp_path": f"ftp://example/GCF_0000{i:02d}.1_asm{i}"}
        for i in range(10)
    ]
    df = _make_asm_df(rows)
    result = select_diversity(df, exclude_genera=set(), exclude_full_ids=set(),
                              target_n=10, max_per_genus=3)
    assert len(result) == 3
    assert all(result["genus"] == "Vibrio")


def test_select_diversity_excludes_priority_genera():
    """Diversity fill skips genera in exclude_genera."""
    rows = [
        {"accession": "GCF_000001.1", "asm_name": "asm1",
         "organism_name": "Pseudomonas aeruginosa", "assembly_level": "Complete Genome",
         "ftp_path": "ftp://example/GCF_000001.1_asm1"},
        {"accession": "GCF_000002.1", "asm_name": "asm2",
         "organism_name": "Vibrio harveyi", "assembly_level": "Complete Genome",
         "ftp_path": "ftp://example/GCF_000002.1_asm2"},
    ]
    df = _make_asm_df(rows)
    result = select_diversity(df, exclude_genera={"Pseudomonas"},
                              exclude_full_ids=set(), target_n=10, max_per_genus=5)
    assert len(result) == 1
    assert result.iloc[0]["genus"] == "Vibrio"


def test_select_diversity_excludes_full_ids():
    """Diversity fill skips accessions already in exclude_full_ids."""
    rows = [
        {"accession": "GCF_000001.1", "asm_name": "asm1",
         "organism_name": "Vibrio harveyi", "assembly_level": "Complete Genome",
         "ftp_path": "ftp://example/GCF_000001.1_asm1"},
        {"accession": "GCF_000002.1", "asm_name": "asm2",
         "organism_name": "Vibrio cholerae", "assembly_level": "Complete Genome",
         "ftp_path": "ftp://example/GCF_000002.1_asm2"},
    ]
    df = _make_asm_df(rows)
    result = select_diversity(df, exclude_genera=set(),
                              exclude_full_ids={"GCF_000001.1_asm1"},
                              target_n=10, max_per_genus=5)
    assert len(result) == 1
    assert result.iloc[0]["accession"] == "GCF_000002.1"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/matthew/Desktop/tcs_engineering
source tcs-env/bin/activate
pytest tests/test_select_expansion_genomes.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'select_expansion_genomes'`

- [ ] **Step 3: Implement `scripts/select_expansion_genomes.py`**

```python
#!/usr/bin/env python3
"""Select ~200 genomes from NCBI assembly_summary.txt for TCS pipeline expansion.

Selection strategy:
  1. Seed: always include RECOMMENDED_GENOMES accessions (from tcs_constants).
  2. Priority fill: guarantee 1 genome per priority genus not already in seed
     (currently only Pseudomonas — others are covered by RECOMMENDED_GENOMES).
  3. Diversity fill: 5/genus cap, Complete Genome preferred, until target_n reached.
     Priority genera are excluded from the diversity fill.

Outputs:
  expansion_download_manifest.txt  — new genomes only (tab-sep: full_id, ftp_path)
  curated_genome_list.txt          — all selected full_ids (one per line)

Usage:
  python scripts/select_expansion_genomes.py \\
      --assembly_summary data/metadata/assembly_summary.txt \\
      --genome_dir       data/genomes \\
      --output_manifest  data/reference/expansion_download_manifest.txt \\
      --output_list      data/reference/curated_genome_list.txt \\
      --target_n         200
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from tcs_constants import ASSEMBLY_LEVEL_PRIORITY, RECOMMENDED_GENOMES

# Priority genera: must have at least 1 representative in the final set.
# Myxococcus, Streptomyces, Anabaena, Caulobacter, Rhodobacter are already
# in RECOMMENDED_GENOMES; only Pseudomonas needs explicit priority selection.
PRIORITY_GENERA = [
    "Myxococcus", "Streptomyces", "Pseudomonas",
    "Anabaena", "Caulobacter", "Rhodobacter",
]


def parse_assembly_summary(path: str, genome_dir: str) -> pd.DataFrame:
    """Load assembly_summary.txt; add derived columns used for selection.

    Follows curate_genome_set.py conventions exactly:
      skiprows=2, header=None, usecols=[0,5,7,11,15] + col 19 (ftp_path).
    """
    existing_dirs = {d for d in os.listdir(genome_dir) if d.startswith("GCF_")}

    df = pd.read_csv(
        path, sep="\t", skiprows=2, header=None,
        usecols=[0, 5, 7, 11, 15, 19],
        names=["accession", "taxid", "organism_name",
               "assembly_level", "asm_name", "ftp_path"],
        low_memory=False,
        dtype=str,
    )
    df = df[df["ftp_path"] != "na"].copy()
    df["full_id"] = df["accession"] + "_" + df["asm_name"]
    df["genus"] = df["organism_name"].str.split().str[0]
    df["level_rank"] = df["assembly_level"].map(ASSEMBLY_LEVEL_PRIORITY).fillna(99)
    df["already_downloaded"] = df["full_id"].isin(existing_dirs)
    return df


def select_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows matching RECOMMENDED_GENOMES accessions."""
    protected = set(RECOMMENDED_GENOMES.values())
    return df[df["accession"].isin(protected)].copy()


def select_priority(df: pd.DataFrame, priority_genera: list[str]) -> pd.DataFrame:
    """Return best 1 genome per priority genus not already in seed.

    Sort key: already_downloaded DESC, level_rank ASC (Complete Genome first).
    Logs a warning for any genus absent from the catalog.
    """
    rows = []
    for genus in priority_genera:
        subset = df[df["genus"] == genus]
        if subset.empty:
            print(f"  WARNING: priority genus '{genus}' not found in assembly_summary")
            continue
        best = subset.sort_values(
            ["already_downloaded", "level_rank"],
            ascending=[False, True],
        ).iloc[[0]]
        rows.append(best)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=df.columns)


def select_diversity(
    df: pd.DataFrame,
    exclude_genera: set[str],
    exclude_full_ids: set[str],
    target_n: int,
    max_per_genus: int = 5,
) -> pd.DataFrame:
    """Return up to target_n genomes with genus cap, excluding specified genera/ids.

    Logs a warning if catalog exhausted before target_n.
    """
    pool = df[
        ~df["genus"].isin(exclude_genera) &
        ~df["full_id"].isin(exclude_full_ids)
    ].sort_values(["genus", "level_rank", "full_id"])

    selected = (
        pool.groupby("genus", group_keys=False)
        .apply(lambda g: g.head(max_per_genus))
        .reset_index(drop=True)
    )

    if len(selected) < target_n:
        print(f"  WARNING: only {len(selected)} diversity genomes available "
              f"(target was {target_n})")
    return selected.head(target_n)


def write_outputs(
    selected: pd.DataFrame,
    existing_dirs: set[str],
    manifest_path: str,
    curated_list_path: str,
) -> None:
    """Write manifest (new genomes only) and curated list (all selected)."""
    new_genomes = selected[~selected["full_id"].isin(existing_dirs)]

    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    new_genomes[["full_id", "ftp_path"]].to_csv(
        manifest_path, sep="\t", index=False, header=False,
    )
    print(f"Manifest: {len(new_genomes)} new genomes → {manifest_path}")

    Path(curated_list_path).parent.mkdir(parents=True, exist_ok=True)
    Path(curated_list_path).write_text(
        "# Curated genome list — generated by select_expansion_genomes.py\n"
        f"# target_n=200  priority_genera={','.join(PRIORITY_GENERA)}\n"
        "# Format: full_id (accession_asmname) — matches data/genomes/ folder names\n"
        + "\n".join(selected["full_id"].tolist())
        + "\n"
    )
    print(f"Curated list: {len(selected)} genomes → {curated_list_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--assembly_summary", required=True)
    parser.add_argument("--genome_dir",       required=True)
    parser.add_argument("--output_manifest",  required=True)
    parser.add_argument("--output_list",      required=True)
    parser.add_argument("--target_n",         type=int, default=200)
    args = parser.parse_args()

    print("Loading assembly summary...")
    df = parse_assembly_summary(args.assembly_summary, args.genome_dir)
    print(f"  {len(df):,} assemblies with valid FTP paths")

    # Step 1: seed from RECOMMENDED_GENOMES
    seed = select_seed(df)
    print(f"  Seed (RECOMMENDED_GENOMES): {len(seed)} genomes")

    # Step 2: priority fill — 1/genus for genera not already in seed
    seed_accessions = set(seed["accession"])
    seed_genera = set(seed["genus"])
    priority_missing = [g for g in PRIORITY_GENERA if g not in seed_genera]
    priority_pool = df[~df["accession"].isin(seed_accessions)]
    priority = select_priority(priority_pool, priority_missing)
    print(f"  Priority fill ({', '.join(priority_missing)}): {len(priority)} genome(s)")

    # Step 3: diversity fill
    already_selected = pd.concat([seed, priority], ignore_index=True)
    already_ids = set(already_selected["full_id"])
    diversity_target = args.target_n - len(already_selected)
    diversity = select_diversity(
        df,
        exclude_genera=set(PRIORITY_GENERA),
        exclude_full_ids=already_ids,
        target_n=diversity_target,
        max_per_genus=5,
    )
    print(f"  Diversity fill: {len(diversity)} genomes")

    all_selected = pd.concat([already_selected, diversity], ignore_index=True)
    print(f"  Total selected: {len(all_selected)} genomes")

    existing_dirs = {d for d in os.listdir(args.genome_dir) if d.startswith("GCF_")}
    write_outputs(all_selected, existing_dirs, args.output_manifest, args.output_list)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/matthew/Desktop/tcs_engineering
source tcs-env/bin/activate
pytest tests/test_select_expansion_genomes.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/select_expansion_genomes.py tests/test_select_expansion_genomes.py
git commit -m "feat: add select_expansion_genomes.py for 10→200 genome expansion"
```

---

## Task 2: Add `--manifest` flag to `scripts/00_download_genomes.py`

**Files:**
- Modify: `scripts/00_download_genomes.py`

The existing `download_genome(row, genome_dir)` reads `row["ftp_path"]` and derives the folder name as `ftp_path.split("/")[-1]`. The manifest rows (full_id, ftp_path) provide exactly what it needs.

- [ ] **Step 1: Add manifest reading + dispatch in `main()`**

In `scripts/00_download_genomes.py`, modify `main()` to add a `--manifest` branch. Insert after the `parser = argparse.ArgumentParser()` block:

```python
# In main(), after parser creation:
parser.add_argument("--manifest", default=None,
                    help="TSV manifest from select_expansion_genomes.py "
                         "(columns: full_id, ftp_path). When provided, "
                         "skips assembly_summary download and TCS detection.")
```

Then add a manifest branch before the existing `summary = download_assembly_summary(...)` line:

```python
    if args.manifest:
        _download_from_manifest(args.manifest, genome_dir)
        return
```

- [ ] **Step 2: Implement `_download_from_manifest()`**

No new imports needed — `ThreadPoolExecutor`, `as_completed`, and `THREADS` are already defined at the top of `00_download_genomes.py`.

Add this function to `scripts/00_download_genomes.py` before `main()`:

```python
def _download_from_manifest(manifest_path: str, genome_dir: Path) -> None:
    """Download genomes listed in a two-column TSV (full_id, ftp_path).

    Skips entries whose folder already exists in genome_dir (idempotent).
    Uses the same download_genome() logic as the standard path.
    Does NOT run TCS detection — manifested genomes are pre-selected.
    """
    rows = []
    with open(manifest_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                print(f"  Skipping malformed manifest line: {line!r}")
                continue
            full_id, ftp_path = parts[0], parts[1]
            if (genome_dir / full_id).exists():
                print(f"  Already present, skipping: {full_id}")
                continue
            rows.append({"full_id": full_id, "ftp_path": ftp_path})

    print(f"Manifest: {len(rows)} genomes to download")

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = {
            executor.submit(download_genome, row, genome_dir): row
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                future.result()
                print(f"  Downloaded: {row['full_id']}")
            except Exception as e:
                print(f"  FAILED: {row['full_id']}: {e}")
```

- [ ] **Step 3: Verify the modification manually**

```bash
cd /Users/matthew/Desktop/tcs_engineering
source tcs-env/bin/activate
python scripts/00_download_genomes.py --help
```

Expected: `--manifest` appears in the help output.

- [ ] **Step 4: Commit**

```bash
git add scripts/00_download_genomes.py
git commit -m "feat: add --manifest flag to 00_download_genomes.py for bulk expansion"
```

---

## Task 3: Run the genome selection

- [ ] **Step 1: Run `select_expansion_genomes.py` against the full catalog**

```bash
cd /Users/matthew/Desktop/tcs_engineering
source tcs-env/bin/activate
python scripts/select_expansion_genomes.py \
    --assembly_summary data/metadata/assembly_summary.txt \
    --genome_dir       data/genomes \
    --output_manifest  data/reference/expansion_download_manifest.txt \
    --output_list      data/reference/curated_genome_list.txt \
    --target_n         200
```

Expected output (approximate):
```
Loading assembly summary...
  481,101 assemblies with valid FTP paths
  Seed (RECOMMENDED_GENOMES): 12 genomes
  Priority fill (Pseudomonas): 1 genome(s)
  Diversity fill: 187 genomes
  Total selected: 200 genomes
Manifest: ~190 new genomes → data/reference/expansion_download_manifest.txt
Curated list: 200 genomes → data/reference/curated_genome_list.txt
```

- [ ] **Step 2: Sanity-check the manifest**

```bash
wc -l data/reference/expansion_download_manifest.txt
head -5 data/reference/expansion_download_manifest.txt
grep "Pseudomonas\|GCF_000012685\|GCF_000005845" data/reference/curated_genome_list.txt | head -10
```

Expected: ~188–192 lines in manifest; Myxococcus (`GCF_000012685`) and E. coli (`GCF_000005845`) present in curated list.

- [ ] **Step 3: Commit the updated curated list and manifest**

```bash
git add data/reference/curated_genome_list.txt data/reference/expansion_download_manifest.txt
git commit -m "data: select 200 genomes for TCS pipeline expansion (TCS-rich priority + diversity)"
```

---

## Task 4: Download the new genomes

This step takes significant wall-clock time (~190 genomes × 3 files, 10 threads). Run in the background.

- [ ] **Step 1: Start the download**

```bash
cd /Users/matthew/Desktop/tcs_engineering
source tcs-env/bin/activate
mkdir -p logs
nohup python scripts/00_download_genomes.py \
    --manifest  data/reference/expansion_download_manifest.txt \
    --data_dir  data \
    > logs/genome_expansion_download.log 2>&1 &
echo "Download PID: $!"
```

Note: `--data_dir data` is the default but is stated explicitly to be unambiguous. `genome_dir` resolves to `data/genomes/`.


- [ ] **Step 2: Monitor progress**

```bash
# Count downloaded genome folders
ls data/genomes/ | grep "^GCF_" | wc -l

# Check for failures in log
grep "FAILED" logs/genome_expansion_download.log | wc -l
tail -20 logs/genome_expansion_download.log
```

Expected: folder count increases toward 200; FAILED count stays low (occasional network errors are normal — re-run is idempotent).

- [ ] **Step 3: Verify completion**

```bash
ls data/genomes/ | grep "^GCF_" | wc -l
# Expected: ~190-200 (some may fail due to FTP issues; 180+ is fine to proceed)
```

---

## Task 5: Re-run the pipeline on 200 genomes

- [ ] **Step 1: Dry-run to confirm Snakemake sees all 200 genomes**

```bash
cd /Users/matthew/Desktop/tcs_engineering
source tcs-env/bin/activate
snakemake --dry-run --cores 1 2>&1 | grep "genomes loaded\|Curated genome\|job\|rule" | head -30
```

Expected: Snakemake reports ~200 genomes loaded and plans jobs for all new genomes.

- [ ] **Step 2: Run the full pipeline**

```bash
snakemake --cores 8 --rerun-incomplete 2>&1 | tee logs/pipeline_200genomes.log
```

This will process all stages for new genomes. The clustering and chimera steps will re-run on the combined dataset.

- [ ] **Step 3: Check for HK_sensor_swap candidates**

```bash
grep "HK_sensor_swap" results/chimera_targets/chimera_candidates.tsv | wc -l
head -3 results/crossover/prostt5_crossover_scores.tsv
```

Expected: >0 HK_sensor_swap candidates; `prostt5_crossover_scores.tsv` is no longer empty.

- [ ] **Step 4: Final commit**

```bash
git add logs/pipeline_200genomes.log results/chimera_targets/ results/crossover/ \
    results/clusters/ results/representatives/
git commit -m "results: TCS pipeline run on 200-genome expanded dataset"
```
