# TCS Engineering Pipeline — Documentation

## Overview

This Snakemake pipeline identifies, clusters, annotates, and scores bacterial
two-component signaling (TCS) proteins across a phylogenetically curated set of
bacterial genomes for the purpose of rational chimera engineering. It integrates
comparative genomics, structural bioinformatics, and protein design tools into a
single reproducible workflow.

The pipeline produces ranked chimera engineering candidates validated against a
curated set of well-characterised TCS systems, with NarXL and PhoRB from
*E. coli* K-12 serving as confirmed positive controls (user-validated in expression
system).

**Core outputs:**

| File | Description |
|---|---|
| `results/chimera_targets/chimera_candidates.tsv` | Ranked TCS chimera design candidates with linker phase scoring |
| `results/validation/validation_summary.txt` | Ground truth validation (GT-1–GT-5 pass/fail report) |
| `results/alphafold/af2_manifest.tsv` | AlphaFold2 structures for top candidates |
| `results/deepcoil/hamp_register_predictions.tsv` | HAMP heptad register analysis |
| `results/deepcoil/af2_plddt_analysis.tsv` | AF2 structural confidence for HAMP regions |
| `results/rfdiffusion/candidates_for_design.tsv` | RFDiffusion-ready contig specifications |
| `results/phylogeny/tcs_tree.treefile` | Maximum likelihood TCS phylogeny |
| `results/promoters/meme.html` | MEME promoter motif discovery |
| `results/diversity_audit/diversity_audit_summary.txt` | Taxonomic composition + reference TCS coverage |

---

## Pre-Run: Genome Set Curation (Stage 0 — Run Once)

Before the pipeline, curate the genome set to remove pseudo-replication from
over-represented genera. This is an explicit, version-controlled decision — the
curated list is saved alongside results.

```bash
python scripts/curate_genome_set.py \
    --assembly_summary data/metadata/assembly_summary.txt \
    --genome_dir data/genomes \
    --output data/reference/curated_genome_list.txt \
    --report data/reference/curation_report.txt \
    --max_per_genus 5
```

**Outputs:**

| File | Description |
|---|---|
| `data/reference/curated_genome_list.txt` | Full IDs for pipeline (one per line) |
| `data/reference/curation_report.txt` | Before/after counts per genus |

If `curated_genome_list.txt` exists, the Snakefile reads it at startup. Otherwise
it falls back to all GCF_ folders in `data/genomes/` with a warning.

**Rationale:** The original 1,475-genome dataset had 83% of genomes from 3 genera
(Pseudomonas 32%, Campylobacter 32%), inflating cluster sizes and phase coherence
scores. A 5-per-genus cap preserves taxonomic signal while eliminating within-genus
pseudo-replication (Laub 2007, Capra & Laub 2012).

**Recommended reference organisms are always kept** regardless of the cap.
See `scripts/tcs_constants.py:RECOMMENDED_GENOMES` for the list.

---

## Running the Pipeline

```bash
cd /Users/matthew/Desktop/tcs_engineering

# Step 0: Curate genome set (run once, re-run after adding genomes)
python scripts/curate_genome_set.py \
    --assembly_summary data/metadata/assembly_summary.txt \
    --genome_dir data/genomes \
    --output data/reference/curated_genome_list.txt \
    --report data/reference/curation_report.txt

# Step 1: Run pipeline
tcs-env/bin/snakemake --snakefile workflow/Snakefile \
    --cores 8 --rerun-incomplete --latency-wait 15
```

After RFDiffusion is installed (see below):

```bash
tcs-env/bin/snakemake run_rfdiffusion --snakefile workflow/Snakefile --cores 1
```

Dry run to inspect jobs:

```bash
tcs-env/bin/snakemake --snakefile workflow/Snakefile --dry-run
```

Regenerate rule-level DAG diagram:

```bash
tcs-env/bin/snakemake --snakefile workflow/Snakefile --rulegraph --dry-run 2>/dev/null \
    | grep -v "^Building" | dot -Tpng -Gdpi=150 -o dag.png
```

Run tests:

```bash
tcs-env/bin/python -m pytest tests/ -v
```

---

## Directory Structure

```
tcs_engineering/
├── dag.png                        # Rule-level pipeline DAG (regenerate with --rulegraph)
├── PIPELINE_DOCS.md               # This file
├── workflow/
│   ├── Snakefile                  # Master DAG: reads curated list, includes all rules
│   └── rules/
│       ├── genomes.smk            # Genome download
│       ├── domains.smk            # Pfam HMM search (per genome)
│       ├── extract_hk_rr.smk      # HK/RR classification + FASTA extraction
│       ├── merge_tcs_sequences.smk# Aggregate all sequences
│       ├── split_sequences.smk    # Split merged FAA into HK / RR
│       ├── clustering.smk         # MMseqs2 clustering (separate HK, RR)
│       ├── representatives.smk    # One seq per cluster
│       ├── operons.smk            # Gene-pair / operon detection
│       ├── promoters.smk          # Promoter extraction + MEME motifs
│       ├── phylogeny.smk          # MAFFT alignment + FastTree
│       ├── homology.smk           # All-vs-all MMseqs2 homology
│       ├── chimera.smk            # DIAMOND annotation + chimera scoring
│       ├── audit.smk              # Species diversity audit + reference TCS coverage
│       ├── alphafold.smk          # AlphaFold2 structure download (EBI API)
│       ├── deepcoil.smk           # HAMP linker extraction + heptad register
│       ├── rfdiffusion.smk        # RFDiffusion linker design
│       └── validation.smk         # Ground truth validation (GT-1–GT-5)
├── scripts/
│   ├── tcs_constants.py           # Shared constants: RECOMMENDED_GENOMES, thresholds
│   ├── curate_genome_set.py       # Genus-capped genome curation (pre-pipeline)
│   ├── 00_download_genomes.py     # Genome download utility
│   ├── 01_detect_domains.py       # HMMER domain parsing + HK/RR classification
│   ├── 02_pair_adjacent_genes.py  # Operon / adjacent gene pair detection
│   ├── 03_cluster_sequences.py    # MMseqs2 clustering wrapper
│   ├── 04_build_phylogeny.py      # (legacy; replaced by phylogeny.smk)
│   ├── 05_extract_promoters.py    # Promoter sequence extraction from GFF
│   ├── 06_discover_motifs.py      # MEME wrapper
│   ├── split_by_type.py           # Partition merged FAA into HK / RR
│   ├── extract_cluster_reps.py    # One representative FASTA per cluster
│   ├── subsample_promoters.py     # Subsample to ≤600 seqs for MEME
│   ├── identify_chimera_targets.py# Main chimera scoring script
│   ├── extract_hamp_linkers.py    # HAMP linker window extraction
│   ├── hamp_register_analysis.py  # Heptad register analysis (replaces DeepCoil)
│   ├── analyze_af2_hamp_plddt.py  # AF2 pLDDT extraction for HAMP regions
│   ├── download_alphafold.py      # EBI AlphaFold2 API download
│   ├── prepare_rfdiffusion.py     # RFDiffusion input preparation
│   ├── run_rfdiffusion.py         # RFDiffusion execution + phase filtering
│   ├── audit_species_diversity.py # Taxonomic diversity audit
│   └── validate_ground_truth.py   # GT-1–GT-5 pipeline validation checks
├── tests/
│   ├── test_curate_genome_set.py  # Unit tests for genome curation (6 tests)
│   └── test_validate_ground_truth.py  # Unit tests for validation checks (14 tests)
├── data/
│   ├── genomes/                   # Per-genome: proteins.faa, genome.fna, genome.gff
│   ├── metadata/assembly_summary.txt
│   ├── reference/
│   │   ├── well_characterized_tcs.tsv   # 30 curated TCS systems + user working systems
│   │   ├── curated_genome_list.txt      # Generated by curate_genome_set.py
│   │   └── curation_report.txt          # Before/after genus counts
│   ├── pfam_tcs.hmm               # TCS-relevant Pfam HMM profiles
│   └── databases/
│       ├── uniprot_sprot.fasta    # Swiss-Prot (574,627 reviewed proteins)
│       └── swissprot.dmnd         # DIAMOND database built from Swiss-Prot
├── results/                       # All pipeline outputs (generated)
├── config/config.yaml             # Pipeline parameters
└── tcs-env/                       # Python virtual environment
```

---

## Pipeline Stages

### Stage 0: Genome Set Curation (`scripts/curate_genome_set.py`) — Pre-pipeline

*See "Pre-Run" section above.*

Selection logic within a capped genus:
1. Protected (RECOMMENDED_GENOMES) — always kept regardless of count
2. Complete Genome > Chromosome > Scaffold > Contig (assembly quality)
3. Alphabetical full_id for determinism when quality is tied

### Stage 1: Genome Processing (`genomes.smk`, `domains.smk`)

**Input:** `data/genomes/{genome}/proteins.faa`
**Output:** `results/domains/{genome}.tbl`

- Runs `hmmsearch` against `data/pfam_tcs.hmm` for each genome
- Uses `--tblout` format (per-sequence hits, no coordinates)
- 4 threads per genome; genomes processed in parallel up to `--cores`

### Stage 2: HK/RR Classification (`extract_hk_rr.smk`, `scripts/01_detect_domains.py`)

**Input:** `results/domains/{genome}.tbl`, `data/genomes/{genome}/proteins.faa`
**Output:** `results/sequences/{genome}.faa`, `results/classifications/{genome}_proteins.csv`

Classifies proteins as HK or RR using strict domain logic:

```
HK: must carry ≥1 of {HisKA, HisKA_3, HisK_5, HA1_5, HisK_3KD}
RR: must carry ≥1 of {Response_reg, REC}
```

**Critical fix (Bug #1):** The original code used `{HisKA, HATPase_c}` with OR logic.
`HATPase_c` (GHKL ATPase fold) is also present in FtsZ (cell division), HtpG
(Hsp90 chaperone), GyrB (DNA gyrase), and MutL (mismatch repair). This caused
~29.4% contamination in the HK set. Fixed by requiring any HisKA-family domain.

### Stage 3: Sequence Aggregation (`merge_tcs_sequences.smk`, `split_sequences.smk`)

**Output:** `results/tcs_sequences.faa`, `results/hk_sequences.faa`, `results/rr_sequences.faa`

Merges per-genome FASTAs and then partitions by protein type using the
per-genome classification CSVs.

### Stage 4: Clustering (`clustering.smk`, `scripts/03_cluster_sequences.py`)

**Output:** `results/clusters/hk_clusters/clusters.tsv`, `results/clusters/rr_clusters/clusters.tsv`

- MMseqs2 `cluster` at 40% identity, coverage 0.8
- HK and RR clustered separately (avoids cross-type merging)
- Output format: `rep_id<TAB>member_id` (TSV, no header)

### Stage 5: Representatives (`representatives.smk`, `scripts/extract_cluster_reps.py`)

**Output:** `results/representatives/hk_reps.faa`, `results/representatives/rr_reps.faa`, `results/representatives/tcs_reps.faa`

One sequence per cluster. Deduplication by `written` set required because
RefSeq WP_ accessions are shared across strains — the same sequence appears
in multiple genome FASTAs (Bug #4).

### Stage 6: Operon Detection (`operons.smk`, `scripts/02_pair_adjacent_genes.py`)

**Input:** `data/genomes/{genome}/genome.gff`, `results/classifications/{genome}_proteins.csv`
**Output:** `results/operons/{genome}.tsv`

Detects HK–RR gene pairs within 500 bp on the same strand. Provides cognate
partner information used in chimera target scoring.

### Stage 7: Promoter Extraction + MEME (`promoters.smk`)

**Output:** `results/promoters/meme.html`

- Extracts 250 bp upstream of each RR gene from GFF + genome FASTA
- Bug fix (Bug #2): NCBI GFF attributes contain `ID=cds-WP_xxxxx.1` before
  `protein_id=WP_xxxxx.1`. Fixed by building a dict from all attribute fields
  and prioritising `protein_id=` over `ID=`.
- Subsamples to ≤600 sequences (MEME practical limit) prioritising RR cluster
  representatives via `scripts/subsample_promoters.py` (Bug #3).
- MEME flags: `-nmotifs 5 -minw 6 -maxw 20`

### Stage 8: Phylogeny (`phylogeny.smk`)

**Output:** `results/phylogeny/tcs_tree.treefile`

- MAFFT `--auto --thread 8` alignment on TCS cluster representatives
- FastTree (BLOSUM45, JTT+CAT model, SH-like support) — used instead of IQ-TREE
  because IQ-TREE symlinks in `/opt/homebrew/bin/` point to deleted paths (Bug #5)
- Runs on cluster representatives only — aligning all sequences produced a 3.1 GB
  alignment and a ~42-minute tree

### Stage 9: Homology (`homology.smk`)

**Output:** `results/homology/hk_homology.m8`, `results/homology/rr_homology.m8`, `results/homology/hk_vs_rr_homology.m8`

MMseqs2 `easy-search` all-vs-all:

| Search | Min identity | Purpose |
|---|---|---|
| HK vs HK | 30% | Identify paralogous kinase families |
| RR vs RR | 30% | Identify response regulator families |
| HK vs RR | 20% | Detect hybrid kinases (HK+REC fusions) |

### Stage 10: DIAMOND Annotation (`chimera.smk`)

**Database:** `data/databases/swissprot.dmnd` (Swiss-Prot, 574,627 reviewed proteins)
**Output:** `results/annotation/hk_annotation.tsv`, `results/annotation/rr_annotation.tsv`

DIAMOND BLASTP against Swiss-Prot:
- `--sensitive` mode
- 1 hit per query (`--max-target-seqs 1`)
- E-value cutoff 1e-5
- Output format 6: `qseqid sseqid pident length qcovhsp evalue bitscore stitle`

Swiss-Prot (reviewed only) chosen over TrEMBL to avoid the ~200M unreviewed,
largely unannotated sequences that degrade specificity for functional inference.

### Stage 11: HAMP Domain Coordinates (`deepcoil.smk`)

**Output:** `results/domains/hk_reps_domtbl.txt`

Runs `hmmsearch --domtblout` on HK cluster representatives to get per-domain
sequence coordinates (ali_from, ali_to). Separate from the per-genome `--tblout`
output because (1) `--tblout` has no residue coordinates and (2) we only need
coordinates for the representatives, not all genomes.

### Stage 12: Chimera Target Scoring (`chimera.smk`, `scripts/identify_chimera_targets.py`)

**Output:** `results/chimera_targets/chimera_candidates.tsv`

Implements three chimera design strategies from primary literature:

#### Strategy A — HK Sensor Swap
*Literature basis: Peruzzi et al. 2023 (PNAS), Hatstat et al. 2025 (JACS)*

Selection: HK cluster size ≥50 → conserved DHp+CA core (large clusters indicate
the kinase domain is structurally conserved across organisms).

Design: Swap sensor domain retaining second-half HAMP + DHp + CA. Select donor
by HAMP alignment (Peruzzi 2023).

**Bioinformatics linker phase validation (Hatstat 2025):**

1. Parse HAMP domain start positions from `--domtblout` for all HK reps
2. For each cluster: compute `HAMP_start mod 7` for all members
3. `cluster_dominant_phase` = mode of those values
4. `cluster_phase_coherence` = fraction of members in dominant phase
5. `linker_phase_compatible = True` when candidate's HAMP phase matches cluster
   dominant phase AND coherence ≥ 0.8

#### Strategy B — RR DBD Swap
*Literature basis: Schmidl et al. 2019 (Nat Chem Biol)*

Classifies RR proteins by DBD family from DIAMOND annotation keywords:
- `OmpR_PhoB`: swap DBD at ~OmpR residues 122–137 (1,300-fold activation demonstrated)
- `NarL_FixJ`: swap DBD at equivalent linker junction

**Priority columns for known TCS systems:**

| Column | Description |
|---|---|
| `known_tcs_system` | Matched system name from well_characterized_tcs.tsv |
| `working_in_user_system` | True if user-confirmed working chimera (NarXL, PhoRB) |

Candidates are sorted: `working_in_user_system=True` first, then by chimera_type,
then cluster_size descending.

#### Strategy C — Hybrid TCS
HK proteins with >25% identity to RR proteins → candidate hybrid kinases with
fused REC domains. Require manual domain architecture verification.

### Stage 13: Species Diversity Audit (`audit.smk`, `scripts/audit_species_diversity.py`)

**Output:** `results/diversity_audit/` (5 files)

Documents taxonomic composition and well-characterised TCS coverage:

| Output file | Content |
|---|---|
| `diversity_genus_counts.tsv` | Per-genus genome counts + fractions |
| `diversity_species_counts.tsv` | Per-species counts |
| `diversity_assembly_levels.tsv` | Complete/Scaffold/Contig breakdown |
| `recommended_genomes_status.tsv` | Which reference organisms are present |
| `reference_tcs_coverage.tsv` | Which known TCS are in DIAMOND annotation |
| `diversity_audit_summary.txt` | Human-readable summary |

Does not block the pipeline even when problems are found.

### Stage 14: AlphaFold2 Structures (`alphafold.smk`, `scripts/download_alphafold.py`)

**Output:** `results/alphafold/*.pdb`, `results/alphafold/af2_manifest.tsv`

Downloads AF2 predicted structures from EBI for top chimera candidate UniProt hits:

```
https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb
```

UniProt IDs extracted from DIAMOND Swiss-Prot best-hit `sseqid` format:
`sp|P23837|PHOQ_ECOLI` → `P23837`

Configurable: `alphafold_max_structures` in config.yaml (default 150).

### Stage 15: HAMP Heptad Register Analysis (`deepcoil.smk`, `scripts/hamp_register_analysis.py`)

**Output:** `results/deepcoil/hamp_register_predictions.tsv`

Sequence-based coiled-coil register analysis using Kyte-Doolittle hydrophobicity:

1. Extract HAMP linker windows (30 aa upstream + 50 aa into HAMP) per representative
2. For each of 7 candidate heptad phases (offset 0–6), score hydrophobic occupancy
   at a/d positions (indices 0 and 3 within each heptad)
3. `dominant_phase` = offset with highest hydrophobic moment
4. `coil_score` = fraction of a/d positions with hydrophobic residues (VILMFYW)
5. `phase_confident = True` if `coil_score ≥ 0.5`

**Why not DeepCoil?** DeepCoil (Ludwiczak et al. 2019) requires `numpy<1.19`,
incompatible with Python 3.8+ on modern macOS. This implementation gives equivalent
per-protein dominant heptad phase and coiled-coil score with full reproducibility
and no external dependencies.

Cross-check: `hamp_start_phase` (from HMMER) should agree with `dominant_phase`
(from sequence hydrophobicity). Concordance validates the bioinformatics linker
phase assignments in `chimera_candidates.tsv`.

### Stage 16: AF2 pLDDT Analysis (`deepcoil.smk`, `scripts/analyze_af2_hamp_plddt.py`)

**Output:** `results/deepcoil/af2_plddt_analysis.tsv`

Extracts per-residue pLDDT from the B-factor column of AF2 PDBs:

| pLDDT range | Interpretation for chimera design |
|---|---|
| > 90 | Very high confidence — use directly |
| 70–90 | High confidence — suitable for RFDiffusion input |
| 50–70 | Low confidence — linker may be flexible; extra validation needed |
| < 50 | Very low — likely disordered; manual inspection required |

Summary columns: `plddt_hamp_mean`, `plddt_linker_mean`, `hamp_high_confidence`.

### Stage 17: Ground Truth Validation (`validation.smk`, `scripts/validate_ground_truth.py`)

**Input:** `hk_annotation.tsv`, `rr_annotation.tsv`, `chimera_candidates.tsv`, `well_characterized_tcs.tsv`
**Output:** `results/validation/ground_truth_validation.tsv`, `results/validation/validation_summary.txt`

Five automated checks against `well_characterized_tcs.tsv`:

| Check | Description | Pass condition |
|---|---|---|
| GT-1 | HK annotation coverage | ≥1 HK rep hits each `working_in_user_system=yes` system |
| GT-2 | RR annotation coverage | Same for RR partners |
| GT-3 | Chimera candidate coverage | All working systems present as `known_tcs_system` |
| GT-4 | HK classifier specificity | No FtsZ/HtpG/GyrB/MutL in HK annotation |
| GT-5 | Phase coherence data present | Working-system candidates have phase columns populated |

**Status semantics:**
- `PASS` — criterion met
- `WARN` — unmet but not fatal (e.g. organism not in genome set)
- `FAIL` — unmet and indicates a pipeline bug; script exits with code 1

**Currently confirmed working systems (`working_in_user_system=yes`):**
- **NarXL** — E. coli K-12 nitrate/nitrite sensor; user-confirmed chimera
- **PhoRB** — E. coli K-12 phosphate limitation sensor; user-confirmed chimera

### Stage 18: RFDiffusion Linker Design (`rfdiffusion.smk`)

**Preparation output:** `results/rfdiffusion/candidates_for_design.tsv`
**Design output:** `results/rfdiffusion/designs/{protein_id}/design_*.pdb`

Selects top `rfdiffusion_n_candidates` HK_sensor_swap candidates with AF2
structures and prepares RFDiffusion partial diffusion inputs.

**Contig specification:**
```
A1-{sensor_end}/{linker_min}-{linker_max}/A{hamp_start}-{chain_length}
```

**Post-diffusion phase filter (Hatstat 2025):**
Keeps only designs where `|designed_linker_length - original_linker_length| mod 7 == 0`.

**RFDiffusion installation:**
```bash
git clone https://github.com/RosettaCommons/RFdiffusion
cd RFdiffusion && pip install -e .
# Download weights to RFdiffusion/models/
```
Set `rfdiffusion_path` in `config/config.yaml`.

---

## Key Bug Fixes and Design Decisions

### Bug 1: HK Classifier Contamination (Critical)

**Problem:** Original: `HK_DOMAINS = {"HisKA", "HATPase_c"}` with OR logic.
`HATPase_c` (GHKL ATPase fold) is present in FtsZ, HtpG, GyrB, and MutL —
none of which are histidine kinases. This caused ~29.4% contamination (FtsZ
and HtpG were the top two "HK" clusters).

**Fix:** Require any HisKA-family domain (the phosphorylatable histidine helix):
```python
HK_DOMAINS_REQUIRED = {"HisKA", "HisKA_3", "HisK_5", "HA1_5", "HisK_3KD"}
```

**Ground truth check:** GT-4 verifies this regression cannot silently recur.

### Bug 2: Promoter Extraction — GFF Attribute Parsing

**Problem:** `extract_protein_id()` checked `ID=` first. In NCBI GFF format,
`ID` is `cds-WP_xxxxx.1` (with `cds-` prefix), not the protein accession.
This caused 100% of promoters to be empty (0 bytes).

**Fix:** Build a dict from all GFF attribute fields; prioritise `protein_id=`:
```python
def extract_protein_id(attr):
    fields = {f.split("=")[0]: f.split("=")[1] for f in attr.split(";") if "=" in f}
    return fields.get("protein_id") or fields.get("ID")
```

### Bug 3: MEME Scale Limit

**Problem:** ~104K promoter sequences; MEME practical limit is ~600 sequences.

**Fix:** `scripts/subsample_promoters.py` subsamples to ≤600, prioritising
sequences that match RR cluster representatives.

### Bug 4: Duplicate WP_ Accessions

**Problem:** RefSeq WP_ accessions are shared across strains. The same protein
appeared in hundreds of genome FASTAs, causing duplicates in representative FASTAs.

**Fix:** `extract_cluster_reps.py` tracks a `written` set and skips already-written IDs.

### Bug 5: IQ-TREE Broken Symlinks

**Problem:** IQ-TREE symlinks in `/opt/homebrew/bin/` point to a deleted installation.
Additionally, IQ-TREE writes `{input}.treefile` by default, not the declared output.

**Fix:** Replaced with FastTree (`~/miniforge3/bin/FastTree`), which writes
directly to the `-out` path and is available via conda bioconda.

### Bug 6: Early-Return CSV Missing in 01_detect_domains.py

**Problem:** Two early return paths (empty domain table, no HK/RR found) created
the FAA output but NOT the CSV output. Snakemake's `MissingOutputException` blocked
any genome with zero TCS proteins.

**Fix:** Compute `csv_path` before early returns; write an empty-column DataFrame
in both exit paths.

### Bug 7: Pseudo-Replication from Uncapped Genera

**Problem:** 83% of the 1,475-genome set came from 3 genera (Pseudomonas 32%,
Campylobacter 32%). This inflated cluster sizes and phase coherence scores
artificially — large clusters for these genera appeared statistically robust
when they merely reflected within-genus conservation.

**Fix:** `scripts/curate_genome_set.py` applies a per-genus cap (default 5)
while always preserving recommended reference organisms. The curated list is
written to `data/reference/curated_genome_list.txt` and committed alongside
results for reproducibility.

---

## Configuration (`config/config.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `threads` | 16 | Threads for HMMER, MMseqs2, MAFFT, DIAMOND |
| `max_per_genus` | 5 | Per-genus cap for genome curation; pass to `curate_genome_set.py --max_per_genus` |
| `alphafold_max_structures` | 150 | Top N chimera candidates to download AF2 structures for |
| `alphafold_version` | 4 | EBI AlphaFold2 model version |
| `deepcoil_upstream_residues` | 30 | Residues before HAMP start in linker window |
| `deepcoil_downstream_residues` | 50 | Residues into HAMP domain in linker window |
| `rfdiffusion_path` | `~/RFdiffusion` | Path to RFdiffusion repo (`run_inference.py` lives here) |
| `rfdiffusion_n_candidates` | 10 | Top HK_sensor_swap candidates to design |
| `rfdiffusion_n_designs` | 10 | Designs per candidate |
| `rfdiffusion_partial_T` | 0.15 | Noise level for partial diffusion (0.1–0.2 = conservative redesign) |

---

## Interpreting `chimera_candidates.tsv`

| Column | Description | How to use |
|---|---|---|
| `protein_id` | Query protein (WP_ accession) | NCBI lookup |
| `best_hit` | Swiss-Prot best match | Confirms identity |
| `hit_description` | Swiss-Prot description | Functional annotation |
| `pident` | % identity to best hit | Quality indicator |
| `cluster_size` | Members in MMseqs2 cluster | Larger = more conserved core |
| `chimera_type` | HK_sensor_swap / RR_DBD_swap / Hybrid_TCS | Strategy |
| `dbd_family` | OmpR_PhoB / NarL_FixJ (RR only) | Swap junction |
| `known_tcs_system` | Matched name from well_characterized_tcs.tsv | System identity |
| `working_in_user_system` | True if user-confirmed functional chimera | Top priority filter |
| `hamp_start` | HAMP domain start residue | Linker boundary |
| `hamp_phase` | `hamp_start mod 7` | Individual register |
| `cluster_dominant_phase` | Mode phase across cluster | Reference register |
| `cluster_phase_coherence` | Fraction of cluster in dominant phase | Confidence (0–1) |
| `linker_phase_compatible` | True/False/None | Phase compatibility flag |
| `linker_validation_required` | True = manual check needed | Caution flag |

**Prioritisation for chimera design:**
1. `working_in_user_system == True` — confirmed in your system; most reliable starting point
2. `chimera_type == HK_sensor_swap` AND `linker_phase_compatible == True` AND `cluster_size` large
3. `chimera_type == RR_DBD_swap` AND `cluster_size` large AND `dbd_family == OmpR_PhoB`
4. Cross-validate against `af2_plddt_analysis.tsv`: require `hamp_high_confidence == True`

---

## Interpreting Ground Truth Validation

After each pipeline run, check `results/validation/validation_summary.txt`.

**A clean run looks like:**
```
[PASS] GT-1  ✓  All 2 working HK systems detected in annotation
[PASS] GT-2  ✓  All 2 working RR systems detected in annotation
[PASS] GT-3  ✓  All 2 working systems present in chimera candidates
[PASS] GT-4  ✓  No known non-HK contaminants found in HK annotation (2847 entries checked)
[PASS] GT-5  ✓  All 4 working-system candidates have phase data
```

**How to respond to failures:**

| Status | Check | Action |
|---|---|---|
| WARN | GT-1/GT-2 | The organism is not in the genome set. Add it and re-run. |
| FAIL | GT-4 | HK classifier is contaminated. Inspect `hk_annotation.tsv` for FtsZ/HtpG hits. Check `HK_DOMAINS_REQUIRED` in `01_detect_domains.py`. |
| FAIL | GT-3 + missing column | Re-run `identify_chimera_targets.py` with `--reference_tcs` flag. |
| WARN | GT-5 | `hmmsearch_hk_reps_domtbl` may not have run. Check `results/domains/hk_reps_domtbl.txt`. |

---

## Literature Basis for Chimera Design

### Laub & Goulian 2007 (Annu Rev Genet) / Capra & Laub 2012 (Cell)
Specificity co-evolution between HK and cognate RR. Framework for understanding
why non-cognate phosphotransfer is disfavoured and how rewiring is possible.

### Skerker et al. 2008 (Cell)
DHp helix-1 + helix-1/2 loop = specificity interface between HK and cognate RR.
Swapping 7 residues in this region is sufficient to fully rewire HK→RR specificity.

### Schmidl et al. 2019 (Nat Chem Biol)
RR DBD swap at the REC-DBD linker boundary (OmpR family: ~positions 122–137).
1,300-fold activation of target promoter demonstrated. Works across gram+ and gram−
organisms. Identified OmpR and NarL/FixJ as the two modular DBD families.

### Peruzzi et al. 2023 (PNAS)
HK sensor domain swap retaining second-half HAMP + DHp + CA. Donor selected by
HAMP sequence alignment (not sensor domain similarity). 3 of 4 chimeras functional.
Key insight: the HAMP domain, not the sensor, determines kinase core compatibility.

### Hatstat et al. 2025 (JACS)
Linker phase (heptad register) and length are critical for transmembrane signal
fidelity. Changing linker length by non-multiples of 7 (disrupting heptad register)
abolishes signalling. Provides the quantitative framework for our bioinformatics
linker phase validation.

---

## Dependencies

| Tool | Installation | Used for |
|---|---|---|
| Python 3.14 | tcs-env | All scripts |
| Snakemake | tcs-env | Pipeline orchestration |
| pandas | tcs-env | Data processing |
| requests | tcs-env | AlphaFold2 API downloads |
| Biopython | tcs-env | FASTA I/O |
| pytest | tcs-env | Unit tests (`python -m pytest tests/ -v`) |
| HMMER (hmmsearch) | `/opt/homebrew/bin` | Domain detection |
| MMseqs2 | tcs-env | Clustering, homology |
| MAFFT | tcs-env | Multiple sequence alignment |
| FastTree | `~/miniforge3/bin` | Phylogenetic tree |
| DIAMOND | tcs-env | Swiss-Prot annotation |
| MEME | tcs-env | Promoter motif discovery |
| graphviz (dot) | `/opt/homebrew/bin` | DAG diagram generation |
| RFDiffusion | manual install | Linker design (optional) |

---

## Expected Outputs After Full Run

```
data/reference/
├── curated_genome_list.txt         # Genome set used in this run
└── curation_report.txt             # Before/after genus counts

results/
├── classifications/                # Per-genome protein type CSVs
├── sequences/                      # Per-genome HK+RR FASTAs
├── operons/                        # Per-genome HK–RR gene pair tables
├── tcs_sequences.faa               # All TCS sequences merged
├── hk_sequences.faa                # HK sequences only
├── rr_sequences.faa                # RR sequences only
├── clusters/
│   ├── hk_clusters/clusters.tsv   # MMseqs2 HK cluster assignments
│   └── rr_clusters/clusters.tsv   # MMseqs2 RR cluster assignments
├── representatives/
│   ├── hk_reps.faa                 # One HK per cluster
│   ├── rr_reps.faa                 # One RR per cluster
│   └── tcs_reps.faa                # Combined for phylogeny
├── domains/
│   ├── {genome}.tbl                # Per-genome hmmsearch --tblout
│   └── hk_reps_domtbl.txt          # HK rep hmmsearch --domtblout (with residue coordinates)
├── homology/
│   ├── hk_homology.m8
│   ├── rr_homology.m8
│   └── hk_vs_rr_homology.m8
├── annotation/
│   ├── hk_annotation.tsv           # DIAMOND Swiss-Prot hits for HK reps
│   └── rr_annotation.tsv           # DIAMOND Swiss-Prot hits for RR reps
├── promoters/
│   ├── promoters_for_meme.fasta    # Subsampled ≤600 promoters
│   └── meme.html                   # MEME motif output
├── alignment/
│   └── tcs_alignment.faa           # MAFFT alignment of TCS reps
├── phylogeny/
│   └── tcs_tree.treefile           # FastTree ML tree
├── chimera_targets/
│   └── chimera_candidates.tsv      # Scored chimera candidates (NarXL/PhoRB prioritised)
├── diversity_audit/
│   ├── diversity_genus_counts.tsv
│   ├── diversity_species_counts.tsv
│   ├── diversity_assembly_levels.tsv
│   ├── recommended_genomes_status.tsv
│   ├── reference_tcs_coverage.tsv
│   └── diversity_audit_summary.txt
├── validation/
│   ├── ground_truth_validation.tsv # GT-1–GT-5 results (one row per check)
│   └── validation_summary.txt      # Human-readable pass/fail report
├── alphafold/
│   ├── {UniProtID}.pdb             # AF2 predicted structures
│   └── af2_manifest.tsv            # Manifest: candidates → PDB paths
├── deepcoil/
│   ├── hamp_linker_regions.faa     # HAMP window sequences
│   ├── hamp_register_predictions.tsv  # Heptad register scores
│   └── af2_plddt_analysis.tsv      # AF2 structural confidence per candidate
└── rfdiffusion/
    ├── candidates_for_design.tsv   # RFDiffusion contig specifications
    └── designs/{protein_id}/       # RFDiffusion output PDBs (after install)
```
