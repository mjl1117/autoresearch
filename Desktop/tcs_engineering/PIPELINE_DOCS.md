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
| `results/promoters/meme.html` | MEME promoter motif discovery (**DEPRECATED** — see Bug #15) |
| `data/reference/characterized_promoters.tsv` | Curated TCS output promoters (Stage 20) |
| `data/reference/promoter_sequences/` | 300 bp upstream FASTA files per promoter (Stage 20) |
| `results/crossover/prostt5_crossover_scores.tsv` | ProstT5 3Di crossover scoring (Stage 21) |
| `results/af3_screening/af3_dbd_promoter_scores.tsv` | AF3 DBD-promoter ipTM scores (Stage 22) |
| `results/af3_screening/inputs/` | AF3 JSON inputs ready for web server or local run |
| `results/diversity_audit/diversity_audit_summary.txt` | Taxonomic composition + reference TCS coverage |
| `results/visualization/tcs_umap.png` | Dual-panel UMAP: HK sensor architecture + RR DBD family |
| `results/visualization/tcs_phylogeny.png` | ML phylogenetic tree coloured by type + DBD family |
| `results/visualization/chimera_candidates_heatmap.png` | Top-15 chimera candidates × key metrics heatmap |
| `results/visualization/tcs_regulatory_network.png` | Bipartite HK→RR cognate pair network |
| `results/visualization/phase_coherence_heatmap.png` | Heptad register phase coherence heatmap |
| `results/visualization/cluster_size_distribution.png` | HK cluster size distribution histogram |

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

Run all visualizations (after the main pipeline completes):

```bash
tcs-env/bin/snakemake visualize_all --snakefile workflow/Snakefile --cores 4
```

Force-regenerate specific visualizations:

```bash
tcs-env/bin/snakemake viz_umap viz_phylogeny --snakefile workflow/Snakefile --cores 4 --forcerun viz_umap viz_phylogeny
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
│       ├── validation.smk         # Ground truth validation (GT-1–GT-5)
│       └── visualize.smk          # Publication-quality figures (run after pipeline)
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
│   ├── validate_ground_truth.py   # GT-1–GT-5 pipeline validation checks
│   └── visualize/
│       ├── plot_tcs_phylogeny.py  # ML tree: HK/RR colour, DBD shape, highlighted systems
│       ├── plot_tcs_umap.py       # Dual-panel UMAP: HK sensor arch + RR DBD family
│       ├── plot_chimera_structures.py  # Domain cartoon + pLDDT track + phase wheel
│       ├── plot_regulatory_network.py  # Bipartite HK→RR cognate pair network
│       └── plot_phase_coherence.py     # Phase heatmap + cluster size histogram
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

### Stage 19: Publication-Quality Visualization (`workflow/rules/visualize.smk`)

**Run separately after the main pipeline completes:**
```bash
tcs-env/bin/snakemake visualize_all --snakefile workflow/Snakefile --cores 4
```

**Output directory:** `results/visualization/`

#### `viz_umap` → `plot_tcs_umap.py`

Two separate UMAP embeddings built from MMseqs2 pairwise percent-identity
(converted to distance as `1 - pident/100`). Running HK and RR separately
produces biologically meaningful class-level separation.

- **Left panel (HK):** Coloured by sensor domain architecture inferred from
  DIAMOND stitle keywords: HAMP, PAS, GAF/PHY, other. Point size scales with
  cluster size. Gold stars mark chimera candidates with crossover annotation
  (e.g., `NarXL [sensor↔HAMP@89]`).
- **Right panel (RR):** Coloured by DBD family: OmpR_PhoB, NarL_FixJ, NtrC_AAA,
  CheY, Spo0A, other. Same star/label treatment for known system representatives.
- DBD classification uses `KNOWN_SYSTEM_LABELS` dict for labeling 30 curated
  TCS systems, not only the 2 working-system controls.

#### `viz_phylogeny` → `plot_tcs_phylogeny.py`

FastTree ML tree of TCS cluster representatives with tip annotations:

- **Colour:** HK (blue `#2166AC`) vs RR (red-orange `#D6604D`)
- **Marker shape:** DBD family — circle (OmpR_PhoB), square (NarL_FixJ),
  triangle (NtrC_AAA), diamond (CheY), plus (Spo0A), dot (other)
- **Bold labels:** NarXL, PhoRB, NarX, NarL, PhoR, PhoB, EnvZ, OmpR
- DBD classification runs on the full RR annotation (`rr_annotation.tsv`)
  via `_classify_dbd()` keyword matching — ensures NtrC/CheY/Spo0A tips
  are correctly shaped even when absent from `chimera_candidates.tsv`

#### `viz_chimera_structures` → `plot_chimera_structures.py`

Per-candidate panel (top N, default 8):
1. Linear domain architecture cartoon: Sensor (grey) → HAMP (yellow) → DHp (blue) → CA (green)
2. Continuous pLDDT track beneath the cartoon, coloured RdYlGn (red=low, green=high)
3. Polar phase wheel showing heptad register coherence for the candidate's cluster

Summary heatmap: top-15 candidates × {cluster_size, pident, phase_coherence, HAMP pLDDT}
with normalised viridis colours and boolean flags panel (phase_compatible, working_in_user_system).

#### `viz_regulatory_network` → `plot_regulatory_network.py`

Bipartite layout: HK nodes (left, blue) → RR nodes (right, orange).
Edges = cognate pairs detected within 500 bp on the same strand by operon detection.
Node size scales with cluster size; nodes matching known systems are labeled.

#### `viz_phase_coherence` → `plot_phase_coherence.py`

- **Heatmap:** Top-50 candidates sorted by `cluster_phase_coherence`;
  rows coloured by phase; working-system rows gold-bordered
- **Histogram:** HK cluster size distribution with phase-compatible overlay

If no phase data is present (e.g., no HAMP domains detected), both scripts write
placeholder "No phase data available" figures to satisfy Snakemake output declarations.

---

### Stage 20: Characterized Promoter Mapping (`scripts/map_promoters_to_candidates.py`)

**Scientific motivation:** The output promoter that an RR DBD binds (e.g., Pho box, NarL box,
OmpR box) is entirely different from the promoter upstream of the RR gene. The original MEME
stage extracted sigma70 sites controlling TCS expression — not the regulatory outputs.
Only RR candidates with a characterized output promoter can be used to design gene circuits.

**Curated reference table:** `data/reference/characterized_promoters.tsv` (10 rows)

| Promoter | DBD family | Signal | Sigma | Aerobic | Source |
|----------|-----------|--------|-------|---------|--------|
| PphoA | OmpR_PhoB | Phosphate starvation | σ70 | Yes | PMID 26015501 |
| PpstS | OmpR_PhoB | Phosphate starvation | σ70 | Yes | PMID 2651888 |
| PompF | OmpR_PhoB | Low osmolarity | σ70 | Yes | PMID 2553720 |
| PnarK | NarL_FixJ | Nitrate (aerobic) | σ70 | Yes | PMID 8183385 |
| PnarG | NarL_FixJ | Nitrate + anaerobic | σ70 | **No** | PMID 11460248 |
| PglnAp2 | NtrC_AAA | N-limitation | **σ54** | Yes | PMID 15327947 |
| PttrB | NarL_FixJ | Tetrathionate | σ70 | Yes | PMID 28373240 |
| PthsA | OmpR_PhoB | Thiosulfate | σ70 | Yes | PMID 28373240 |
| CheY | (hard exclude) | — | — | — | CheY has no DBD |

**Critical notes:**
- `PnarG` requires Fnr (anaerobiosis) + IHF. **Cannot activate in aerobic BL21 circuits.**
  `PnarK` is the correct NarL promoter for aerobic conditions.
- `PglnAp2` requires σ54 (rpoN). BL21 has rpoN — this promoter is viable.
- `CheY_standalone` proteins are hard-excluded with an explicit `HARD EXCLUDE` caveat.
  CheY acts on flagellar motor switch FliM, not on DNA.
- `PthsA` sequences must be retrieved from Daeffler & Tabor 2017 (PMID 28373240)
  supplementary materials — not annotated in genome.

**Columns added to `chimera_candidates.tsv`:**
`has_characterized_promoter`, `recommended_promoters`, `promoter_signals`,
`sigma_factors`, `aerobic_compatible`, `promoter_caveats`

**Promoter sequences fetched:** `scripts/fetch_promoter_sequences.py` downloads 300 bp upstream
of each target gene CDS from NCBI Entrez (`gbwithparts` format required for chromosomal records).
Sequences stored in `data/reference/promoter_sequences/`.

---

### Stage 21: ProstT5 Crossover Scoring (`workflow/rules/crossover.smk`)

**Purpose:** Find optimal crossover positions within the HAMP-DHp junction for HK sensor swap
chimeras, without requiring solved structures.

**Method:** ProstT5 (Rostlab/ProstT5, Heinzinger 2023) encodes each amino acid residue as a
3Di structural alphabet token (the same 20-token alphabet used by Foldseek). Positions where
the 3Di token is identical between the donor and chassis HK represent structurally similar
local environments — low-disruption crossover points.

**Scoring:**
- `three_di_score`: 1.0 if tokens identical, 0.0 otherwise (simplified; full Foldseek 3Di
  substitution matrix can be substituted for finer resolution)
- `phase_penalty`: 0.5 if the crossover shifts heptad register by non-multiple of 7
- `combined_score` = `three_di_score` − `phase_penalty`

**Note on SCHEMA/RASPP:** SCHEMA (Voigt 2002) is the gold standard crossover scoring method
and uses residue-residue contact maps from solved or AF2-predicted structures. ProstT5 is used
here as a no-structure alternative. When AF2 structures become available, replace with SCHEMA.

**Model:** Rostlab/ProstT5 (~2 GB, downloaded once from HuggingFace). Set `prostt5_device: mps`
in config.yaml for Apple Silicon acceleration.

**Output:** `results/crossover/prostt5_crossover_scores.tsv`
Empty if no `HK_sensor_swap` candidates exist (expected with small genome sets).

---

### Stage 22: AF3 DBD-Promoter Binding Screening (`workflow/rules/af3_screening.smk`)

**Purpose:** Predict whether each RR DBD will specifically bind its target promoter using
AlphaFold3 protein-DNA complex prediction.

**DBD boundary heuristics (literature-based):**
- OmpR_PhoB: C-terminal ~90 aa (winged-HTH, Krell 2010)
- NarL_FixJ: C-terminal ~65 aa (HTH, Maris 2002)
- NtrC_AAA:  C-terminal ~55 aa (HTH, De Carlo 2006)

**DNA binding site sequences:**
- Pho box (OmpR/PhoB): `CTGTCATAAAGCCTGTCATA` — Makino 1988
- OmpR F1 box: `TGAAACTTTTTTTATGTTCA` — Rampersaud 1989
- NarL box: `TACCCATTTACCCATTTACC` — Darwin 1997
- NtrC UAS (glnAp2): `TGCACCATATTTGCACCAT` — Reitzer 1989

**AF3 JSON format:** protein chain A + sense DNA chain B + antisense DNA chain C.
28 pairs prepared (top 10 RR candidates × 1–4 recommended promoters each).

**ipTM threshold (Abramson 2024):**
| ipTM | Interpretation |
|------|---------------|
| > 0.75 | High confidence — likely specific binding |
| 0.5–0.75 | Medium confidence — worth testing |
| < 0.5 | Low confidence — binding unlikely |

**Running AF3:**
- **Web server (recommended for small batches):** Upload JSONs from
  `results/af3_screening/inputs/` to https://alphafoldserver.com. Download results ZIP →
  extract to `results/af3_screening/outputs/{pair_id}/`. Re-run
  `snakemake prepare_af3_dbd_promoter` to parse scores.
- **Local:** Apply for model weights (https://github.com/google-deepmind/alphafold3),
  then set `af3_run_local: true` and `af3_dir: ~/alphafold3` in config.yaml.

**Output:** `results/af3_screening/af3_dbd_promoter_scores.tsv` (ipTM, pTM, pLDDT per pair)

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

### Bug 8: HMMER domtblout Column Indices (Critical — 0 HAMP Domains Found)

**Problem:** `extract_hamp_linkers.py:parse_domtbl()` read the wrong columns from
`--domtblout` output. The HMMER domtblout format (22+ whitespace-delimited fields)
was being parsed with these wrong indices:

```python
domain   = parts[2]        # Actually tlen (integer!) — not the domain name
score    = float(parts[11])# c-Evalue — not domain score
ali_from = int(parts[15])  # hmm_from — not sequence ali_from
ali_to   = int(parts[16])  # hmm_to — not sequence ali_to
```

This caused `domain` to always be a number string like `"477"`, which never
matched `"HAMP"`, `"HAMP_2"`, or `"CovS-like_HAMP"` — resulting in exactly
0 HAMP boundaries found out of 477 sequences.

**Correct domtblout column layout (0-indexed):**

| Index | Field |
|---|---|
| 0 | target name (protein ID) |
| 1 | target accession |
| 2 | tlen (target length — an **integer**) |
| 3 | query name (Pfam profile, e.g. `HAMP`) |
| 13 | domain score |
| 17 | ali_from (alignment start on sequence) |
| 18 | ali_to (alignment end on sequence) |

**Fix:**
```python
domain   = parts[3]        # query/domain name
score    = float(parts[13])# domain score
ali_from = int(parts[17])  # ali_from on target sequence
ali_to   = int(parts[18])  # ali_to on target sequence
```
Also changed length guard from `len(parts) < 17` → `len(parts) < 23`.
Result: 188 HAMP domains found (was 0).

**HAMP Pfam profile names in Pfam-A.hmm:** `HAMP`, `HAMP_2`, `CovS-like_HAMP`.

### Bug 9: DBD Classification Gap — NtrC/CheY/Spo0A Invisible

**Problem:** Both `plot_tcs_umap.py` and `plot_tcs_phylogeny.py` built their DBD
family classification map (`dbd_map`) only from the 11 rows of `chimera_candidates.tsv`.
Any RR protein not in that table (including all NtrC_AAA, CheY, and Spo0A proteins)
appeared as "other" (grey dot, `"."` marker). A grep of `rr_annotation.tsv` confirmed
19 NtrC/CheY/Spo0A proteins existed but were being silently mislabelled.

**Fix (both scripts):** Load `rr_annotation.tsv` independently and classify DBD family
for **all** RR proteins using `_classify_dbd()` keyword matching against the DIAMOND
`stitle` field. Chimera candidates then **override** these auto-classifications with
their curated `dbd_family` column:

```python
# Step 1: classify all RR proteins from stitle
if Path(rr_ann).exists():
    df = pd.read_csv(rr_ann, sep="\t", header=None, names=ann_cols)
    for _, row in df.iterrows():
        dbd_map[row["qseqid"]] = _classify_dbd(row["stitle"])

# Step 2: chimera curated labels override auto-classification
if Path(chimera_tsv).exists():
    df = pd.read_csv(chimera_tsv, sep="\t")
    for pid, fam in df.set_index("protein_id")["dbd_family"].items():
        if pd.notna(fam) and fam != "N/A":
            dbd_map[pid] = fam
```

**Keyword dictionaries** (`_RR_DBD_KEYWORDS` in phylogeny, `RR_DBD_KEYWORDS` in UMAP):
- `Spo0A`: "spo0a", "sporulation", "stage 0"
- `CheY`: "chey ", "chemotaxis response regulator"
- `NtrC_AAA`: "ntrc", "nifa", "luxo", "sigma-54", "aaa+", "enhancer-binding"
- `NarL_FixJ`: "narl", "narp", "fixj", "gera", "coma", "flhd"
- `OmpR_PhoB`: "ompr", "phob", "phop", "rsca", "cpxr", "baer", "kdpe", etc.

### Bug 10: TCS System Naming Convention (Normalisation)

**Problem:** `well_characterized_tcs.tsv` used inconsistent system names — some full
gene concatenations (`PhoQPhoP`, `KdpDKdpE`) and some abbreviated (`NarXL`, `PhoRB`).
The full-form names caused UMAP label clutter and mismatches in chimera scoring
(which was matching against these names for `known_tcs_system`).

**Convention established:** Short concatenated abbreviations:
- Sensor gene abbreviation + regulator abbreviation
- Examples: `NarXL`, `PhoRB`, `PhoQP`, `KdpDE`, `CheAY`, `TorSR`, `DcuSR`
- Dashes for non-standard pairs: `EnvZ-OmpR`, `KinA-Spo0A`, `Cph1-Rcp1`

**All 26 system_name values normalised** (e.g., `PhoQPhoP` → `PhoQP`,
`KdpDKdpE` → `KdpDE`, `ArcBArcA` → `ArcBA`, `CpxACpxR` → `CpxAR`).

**Action required after this fix:** Force-rerun `identify_chimera_targets` to
regenerate `chimera_candidates.tsv` with correct `known_tcs_system` values:
```bash
tcs-env/bin/snakemake identify_chimera_targets --snakefile workflow/Snakefile \
    --cores 4 --forcerun identify_chimera_targets
```

### Bug 11: pandas `usecols` Column Ordering

**Problem:** `curate_genome_set.py` and `audit_species_diversity.py` both called:
```python
pd.read_csv(..., usecols=[0, 7, 11, 15, 5], names=["accession", "taxid", ...])
```
pandas silently **sorts** integer `usecols` before assigning `names` sequentially.
So `[0, 7, 11, 15, 5]` becomes sorted `[0, 5, 7, 11, 15]`, making column 5
(an integer taxid) be named `"organism_name"`. Subsequent `.str` calls on an
integer column raised `AttributeError`.

**Fix:** Reorder `usecols` to match the intended name order:
```python
pd.read_csv(..., usecols=[0, 5, 7, 11, 15],
            names=["accession", "taxid", "organism_name", "assembly_level", "asm_name"])
```

### Bug 12: Missing `analyze_af2_hamp_plddt.py` Script

**Problem:** `deepcoil.smk` declared `analyze_af2_hamp_plddt` as a rule invoking
`scripts/analyze_af2_hamp_plddt.py`, but that script did not exist in the repository.

**Fix:** Created `scripts/analyze_af2_hamp_plddt.py` from scratch. The script:
1. Reads `af2_manifest.tsv` to find which PDB files were downloaded
2. Reads `hk_reps_domtbl.txt` to get HAMP coordinates per protein
3. Extracts per-residue pLDDT from the B-factor column of each PDB (`CA` atoms only)
4. Computes `plddt_hamp_mean`, `plddt_linker_mean`, `hamp_high_confidence` (mean > 70)
5. **Handles empty manifests gracefully**: writes a valid header-only TSV when no
   PDB files are present (avoids `MissingOutputException`)

**EBI AlphaFold outage note (March 2026):** The EBI AlphaFold API
(`https://alphafold.ebi.ac.uk/files/`) was returning HTTP 500 (server error) and
HTTP 404 for some accessions during pipeline development. The script and manifest
handle this gracefully — the pipeline continues without AF2 structural data.
Re-run Stage 14 once the API recovers:
```bash
tcs-env/bin/snakemake download_alphafold --snakefile workflow/Snakefile --cores 1 --forcerun download_alphafold
```

### Bug 13: FastTree Hardcoded Path

**Problem:** `phylogeny.smk` contained `{workflow.basedir}/../miniforge3/bin/FastTree`
(a hardcoded absolute path specific to one machine). The rule failed on any system
where FastTree was not installed at that exact location.

**Fix:** Changed to bare `FastTree` (relies on `PATH`). Install via:
```bash
brew install fasttree
```

### Bug 14: `load_hamp_boundaries` in `identify_chimera_targets.py` (Critical — Silent)

**Problem:** `identify_chimera_targets.py:load_hamp_boundaries()` (lines 100–111) had the
same wrong column indices as Bug #8. The function was silently loading 0 HAMP start positions
for all HK proteins, making `linker_phase_compatible` and `cluster_phase_coherence` always NaN
and preventing any HK_sensor_swap candidate from passing phase validation.

**Wrong indices used (same as Bug #8):**
```python
domain   = parts[2]        # tlen integer — not domain name
score    = float(parts[11])# c-Evalue — not domain score
ali_from = int(parts[15])  # hmm_from — not sequence ali_from
```

**Fix:**
```python
domain   = parts[3]        # query/domain name (e.g. "HAMP", "HAMP_2")
score    = float(parts[13])# domain score
ali_from = int(parts[17])  # ali_from on target sequence
if len(parts) < 23: continue   # guard before access
if domain in ("HAMP", "HAMP_2", "CovS-like_HAMP"):
```

**Result:** 115 HAMP starts now loaded (was 0). Phase analysis is now functional.

### Bug 15: MEME Promoter Stage Scientifically Invalid (Stage 7 Redesigned)

**Problem:** Stage 7 (`promoters.smk`) extracted 200–250 bp upstream of RR genes and
ran MEME on those sequences. These regions are the sigma70 sites that control TCS gene
expression (transcription of the RR itself) — they are NOT the output promoters that the RR
DBD binds after phosphorylation.

Using these sequences to identify promoters for gene circuit design is scientifically invalid.
A MEME motif found upstream of NarL would be the NarL gene promoter, not the PnarK/PnarG
binding sites that NarL-P activates.

**Impact:** Any circuit designed using MEME-detected "promoters" from this stage would
likely fail to activate, as the sensor would phosphorylate the RR, but the RR would have
no characterized binding sites provided.

**Fix:** Stage 7 rules renamed `*_DEPRECATED`. Replaced with:
1. **`data/reference/characterized_promoters.tsv`** — curated from RegulonDB/literature
2. **`scripts/fetch_promoter_sequences.py`** — NCBI Entrez fetch of characterized binding regions
3. **`scripts/map_promoters_to_candidates.py`** — joins promoter table to chimera candidates

The MEME rules are retained in `promoters.smk` with `_DEPRECATED` suffix for reference
but are excluded from `rule all` and from the Snakemake DAG.

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
| `ncbi_email` | `your@email.here` | Required by NCBI Entrez policy; set before running `fetch_promoters` |
| `promoter_upstream_bp` | 300 | bp upstream of CDS start to extract per promoter |
| `prostt5_device` | `cpu` | PyTorch device: `cpu` / `mps` (Apple Silicon) / `cuda` |
| `crossover_top_n` | 5 | Top N crossover positions to report per candidate pair |
| `af3_top_n` | 10 | Top N RR candidates to screen with AF3 |
| `af3_run_local` | `false` | Set `true` to run AF3 locally after JSON preparation |
| `af3_dir` | `~/alphafold3` | Path to local AF3 installation |

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
| `has_characterized_promoter` | True/False | Whether a characterized RR output promoter exists |
| `recommended_promoters` | Comma-separated list | Promoters suitable for aerobic E. coli circuits |
| `promoter_signals` | Comma-separated list | Inducing signals for each recommended promoter |
| `sigma_factors` | sigma70 / sigma54 | Sigma factor required (sigma54 needs rpoN in host) |
| `aerobic_compatible` | True/False | False = promoter requires anaerobic conditions (e.g. PnarG) |
| `promoter_caveats` | Text | Warnings: CheY hard-exclude, anaerobic requirement, cofactors |

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
| matplotlib | tcs-env | All visualization scripts |
| umap-learn | tcs-env | UMAP dimensionality reduction (`pip install umap-learn`) |
| scikit-learn | tcs-env | Distance matrix preprocessing for UMAP |

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
├── rfdiffusion/
│   ├── candidates_for_design.tsv   # RFDiffusion contig specifications
│   └── designs/{protein_id}/       # RFDiffusion output PDBs (after install)
└── visualization/
    ├── tcs_umap.png/.pdf           # Dual-panel HK/RR UMAP
    ├── tcs_phylogeny.png/.pdf      # ML phylogenetic tree
    ├── chimera_candidates_heatmap.png/.pdf  # Top-15 candidates heatmap
    ├── chimera_{system}.png/.pdf   # Per-candidate domain + phase panels
    ├── tcs_regulatory_network.png/.pdf     # HK→RR bipartite network
    ├── phase_coherence_heatmap.png/.pdf    # Heptad register heatmap
    └── cluster_size_distribution.png/.pdf  # Cluster size histogram
```
