# Genome Expansion Design: 10 → 200 Genomes

**Date:** 2026-03-21
**Goal:** Expand the TCS pipeline genome repertoire from 10 to ~200 genomes to unlock HK_sensor_swap candidates (requires cluster_size ≥ 50).

---

## Motivation

The current 10-genome dataset produces a maximum HK cluster size of 13. The `identify_chimera_targets.py` threshold for `HK_sensor_swap` candidates is `cluster_size ≥ 50`. No HK_sensor_swap candidates are produced. The ProstT5 crossover scoring rule outputs an empty TSV as a result. Expanding to ~200 genomes is expected to push HK cluster sizes above the threshold.

---

## Organism Selection Strategy

**TCS-rich taxa first, then phylogenetic diversity.**

Priority taxa (1 genome each — best assembly quality): Myxococcus, Streptomyces, Pseudomonas, Anabaena, Caulobacter, Rhodobacter. These genera have among the highest known TCS gene counts and provide key reference systems. They are represented but not over-sampled (1/genus = 6 priority genomes, leaving 194 slots for diversity fill).

Remaining 194 slots filled with genus-capped diversity (max 5/genus) from the full assembly summary catalog, sorted by assembly quality. Priority genera are excluded from the diversity fill entirely — they already have their representative.

---

## Architecture

Three-step process. **Note:** `00_download_genomes.py` is modified (see below) to accept a manifest. All other pipeline files are unchanged.

```
Step 1:  python scripts/select_expansion_genomes.py \
             --assembly_summary data/metadata/assembly_summary.txt \
             --genome_dir       data/genomes \
             --output_manifest  data/reference/expansion_download_manifest.txt \
             --output_list      data/reference/curated_genome_list.txt \
             --target_n         200

         → writes curated_genome_list.txt (all 200 full_ids, including not-yet-downloaded)
         → writes expansion_download_manifest.txt (only genomes not yet in data/genomes/)
         Snakefile reads curated_genome_list.txt but intersects with existing_folders,
         so writing the list before download is safe — Snakemake will process only
         the genomes that have been downloaded.

Step 2:  python scripts/00_download_genomes.py \
             --manifest data/reference/expansion_download_manifest.txt
         → downloads ~190 new genomes into data/genomes/

Step 3:  snakemake
         → unchanged; reads curated_genome_list.txt, now intersects all 200 with
           data/genomes/ (fully populated after Step 2), processes all 200
```

Re-running Step 1 after partial download is idempotent: manifest only includes genomes not in `data/genomes/`.

---

## New Script: `scripts/select_expansion_genomes.py`

### Inputs
- `data/metadata/assembly_summary.txt` — 481,101-line NCBI assembly summary (already local, 204 MB)
- `data/genomes/` — existing genome folders (currently 10 GCF_* directories)

### Outputs

**`data/reference/expansion_download_manifest.txt`**
Two-column tab-separated file, no header:
```
full_id\tftp_path
GCF_000012345.1_ASM1234v1\thttps://ftp.ncbi.nlm.nih.gov/genomes/...
```
Only genomes not already present in `data/genomes/`. The `--manifest` path in `00_download_genomes.py` reads `ftp_path` directly — no need to re-parse assembly_summary.

**`data/reference/curated_genome_list.txt`**
One `full_id` per line (all 200 selected, including already-downloaded). Replaces current 10-genome list. Same format as the existing file.

### Assembly Summary Parsing

Use `skiprows=2, header=None, usecols=[0,5,7,11,15,19], names=["accession","taxid","organism_name","assembly_level","asm_name","ftp_path"]` — identical pattern to `curate_genome_set.py` (which uses `usecols=[0,5,7,11,15]`), extended with col 19 (`ftp_path`). This is the established codebase pattern and must NOT use `skiprows=1` or auto column names.

**full_id construction:** `accession + "_" + asm_name` — matches `curate_genome_set.py` and the Snakefile's `FTP_LOOKUP` key format exactly.

### Derived Columns (added in `parse_assembly_summary`)

- `genus`: `organism_name.split()[0]`
- `assembly_level_rank`: integer sort key derived from `assembly_level`:
  ```
  "Complete Genome" → 0
  "Chromosome"      → 1
  "Scaffold"        → 2
  "Contig"          → 3
  anything else     → 4
  ```
- `already_downloaded`: boolean — `full_id in existing_dirs`

Where `existing_dirs: set[str]` is constructed as:
```python
existing_dirs = set(
    d for d in os.listdir(genome_dir) if d.startswith("GCF_")
)
```

### Selection Logic

**`main()` runs three passes (seed → priority fill → diversity fill):**

**Seed pass** — always included, exempt from all caps:
```python
seed = select_seed(df)   # RECOMMENDED_GENOMES accessions found in assembly_summary (~12 rows)
```
`RECOMMENDED_GENOMES` covers Myxococcus, Streptomyces, Anabaena, Caulobacter, Rhodobacter but NOT Pseudomonas.

**Priority fill** — called after seed, on the non-seed remainder:
```python
select_priority(df, priority_genera) -> pd.DataFrame
```
`priority_genera` is the subset of `PRIORITY_GENERA` whose genus is not already in `seed`. In normal operation this is only `["Pseudomonas"]` → 1 additional genome.

Sort key per genus: `already_downloaded DESC` (primary), `level_rank ASC` (secondary) — an already-downloaded genome is preferred regardless of assembly quality, since it avoids a download.

**Diversity fill** — fills remaining `target_n - len(seed) - len(priority)` slots:
```python
select_diversity(df, exclude_genera, exclude_full_ids, target_n, max_per_genus=5) -> pd.DataFrame
```
- `exclude_genera`: all of `PRIORITY_GENERA` (excluded entirely)
- `exclude_full_ids`: full_ids already in seed or priority
- Sort within each genus by `level_rank ASC` then `full_id ASC` for determinism
- Log warning if catalog exhausted before `target_n`

**`main()` slot arithmetic (explicit):**
```python
seed              = select_seed(df)                          # ~12 rows
priority_missing  = [g for g in PRIORITY_GENERA if g not in seed_genera]
priority          = select_priority(pool, priority_missing)  # ~1 row (Pseudomonas)
diversity_target  = args.target_n - len(seed) - len(priority)  # ~187
diversity         = select_diversity(..., target_n=diversity_target)
all_selected      = pd.concat([seed, priority, diversity])   # ~200 rows
```

### Functions
```
parse_assembly_summary(path, genome_dir) -> pd.DataFrame
    skiprows=1, adds genus, assembly_level_rank, already_downloaded, full_id columns
    filters ftp_path != "na"

select_priority(df, priority_genera, max_per_genus=1) -> pd.DataFrame
    returns best-quality genome per priority genus; warns if genus absent

select_diversity(df, exclude_genera, exclude_full_ids, target_n, max_per_genus=5) -> pd.DataFrame
    genus-capped quality-sorted fill; warns if catalog exhausted before target_n

write_outputs(selected_df, existing_dirs, manifest_path, curated_list_path)
    manifest: rows where full_id NOT in existing_dirs, writes full_id + ftp_path TSV (no header)
    curated_list: all full_ids, one per line

main()
    argparse: --assembly_summary, --genome_dir, --output_manifest, --output_list, --target_n (default 200)
    calls parse → select_priority → select_diversity → write_outputs
    prints summary: N priority, N diversity, N new to download
```

### Error Handling
- Priority genus with zero catalog entries: log warning, skip (do not fail).
- Total selected < `target_n`: emit all available with warning showing actual count.
- `ftp_path == "na"`: excluded in `parse_assembly_summary` before selection.

---

## Modification: `scripts/00_download_genomes.py`

Add `--manifest` flag (optional). When provided:
- Read the two-column TSV (full_id, ftp_path) produced by `select_expansion_genomes.py`
- For each row, call the existing `download_genome(row, genome_dir)` function — it reads `ftp_path` from the row dict, so the manifest row format is compatible
- Skip rows where the full_id folder already exists in `data/genomes/` (idempotent)
- Skip the TCS detection step (`detect_tcs`) — all genomes from the curated list are already expected to have TCS genes
- Do NOT re-download or re-parse `assembly_summary.txt`

When `--manifest` is not provided, existing behavior is completely unchanged.

---

## Integration with Existing Pipeline

No changes to `workflow/Snakefile`, `workflow/rules/`, or `config/config.yaml`.

The Snakefile already reads `data/reference/curated_genome_list.txt` and intersects with `data/genomes/` folder names. Once all 200 genomes are downloaded, `snakemake` processes all 200 automatically.

---

## Expected Outcomes

- ~200 genomes in `data/genomes/`
- HK cluster sizes expected to exceed 50 → HK_sensor_swap candidates produced
- `results/crossover/prostt5_crossover_scores.tsv` populated (currently empty)
- Priority taxa (Myxococcus, Streptomyces, Pseudomonas, Anabaena, Caulobacter, Rhodobacter) all represented with one high-quality genome each

---

## Out of Scope

- Modifying `curate_genome_set.py`
- Changing pipeline rules or config
- Downloading more than 200 genomes
- Automated re-run of the full pipeline (user runs `snakemake` manually after download)
- TCS detection filtering during manifest-driven download (all selected genomes are from TCS-rich organisms)
