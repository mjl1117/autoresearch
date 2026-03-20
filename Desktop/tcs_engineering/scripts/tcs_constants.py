"""Shared constants for TCS pipeline scripts.

Centralised here so audit_species_diversity.py and curate_genome_set.py
reference the same RECOMMENDED_GENOMES dict and threshold values.
"""

# NCBI RefSeq assembly accessions for organisms that should always be present
# in a broad TCS survey. These are excluded from the per-genus cap in
# curate_genome_set.py regardless of how many other genomes from the same
# genus are in the dataset.
RECOMMENDED_GENOMES = {
    # E. coli K-12 MG1655 — primary organism for TCS characterisation
    "Escherichia coli K-12 MG1655":       "GCF_000005845.2",
    # B. subtilis 168 — gram+ model; Schmidl 2019 chimera work
    "Bacillus subtilis 168":              "GCF_000009045.1",
    # Caulobacter crescentus NA1000 — CckA/CtrA paradigm cell-cycle TCS
    "Caulobacter crescentus NA1000":      "GCF_000022005.1",
    # Myxococcus xanthus DK1622 — ~245 HKs; most TCS-rich genome known
    "Myxococcus xanthus DK1622":          "GCF_000012685.1",
    # Streptomyces coelicolor A3(2) — >65 TCS; industrial relevance
    "Streptomyces coelicolor A3(2)":      "GCF_000203835.1",
    # Synechocystis sp. PCC 6803 — 46 TCS; light-sensing phytochromes (Cph1)
    "Synechocystis sp. PCC 6803":         "GCF_000009725.1",
    # Rhodobacter capsulatus SB1003 — RegB/RegA redox sensing; photosynthesis
    "Rhodobacter capsulatus SB1003":      "GCF_000009485.1",
    # Vibrio harveyi BB120 — LuxN/LuxO quorum sensing paradigm
    "Vibrio harveyi BB120":               "GCF_000021505.1",
    # Thermotoga maritima MSB8 — archaeal-like TCS; structural studies
    "Thermotoga maritima MSB8":           "GCF_000008545.1",
    # Staphylococcus aureus MRSA252 — NreB/NreC GAF oxygen sensor
    "Staphylococcus aureus MRSA252":      "GCF_000011505.1",
    # Rhodopseudomonas palustris CGA009 — BphP1 bacteriophytochrome
    "Rhodopseudomonas palustris CGA009":  "GCF_000195775.1",
    # Anabaena PCC 7120 — >100 TCS; most TCS-dense cyanobacterium
    "Anabaena sp. PCC 7120":              "GCF_000009705.1",
}

ASSEMBLY_LEVEL_PRIORITY = {
    "Complete Genome": 0,
    "Chromosome":      1,
    "Scaffold":        2,
    "Contig":          3,
}

# Audit thresholds (used by audit_species_diversity.py)
MAX_GENUS_FRACTION = 0.10   # Flag genera that are >10% of the dataset
MIN_SPECIES_COUNT  = 5      # Warn if fewer than this many distinct species total
