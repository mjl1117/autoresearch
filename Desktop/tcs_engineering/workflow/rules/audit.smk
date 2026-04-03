rule audit_species_diversity:
    """Document taxonomic composition and well-characterized TCS coverage.

    This rule is the reproducibility checkpoint for the genome sampling design.
    It runs automatically and flags:
      - Genera that are over-represented (>10% of dataset) — pseudo-replication risk
      - Missing recommended reference organisms (E. coli K-12, B. subtilis 168,
        Caulobacter, Myxococcus, Synechocystis, etc.)
      - Well-characterized TCS systems not detected in the annotation

    Outputs (in results/diversity_audit/):
      diversity_genus_counts.tsv        — per-genus genome counts + fractions
      diversity_species_counts.tsv      — per-species counts
      diversity_assembly_levels.tsv     — Complete/Scaffold/Contig breakdown
      recommended_genomes_status.tsv    — which reference organisms are present
      reference_tcs_coverage.tsv        — which known TCS are in DIAMOND annotation
      diversity_audit_summary.txt       — human-readable summary

    The audit does NOT block the pipeline even when problems are found — it
    produces warnings so the user can decide whether to add genomes before
    interpreting results.
    """
    input:
        hk_ann="results/annotation/hk_annotation.tsv",
        rr_ann="results/annotation/rr_annotation.tsv",
        reference_tcs="data/reference/well_characterized_tcs.tsv"
    output:
        summary="results/diversity_audit/diversity_audit_summary.txt",
        genus_counts="results/diversity_audit/diversity_genus_counts.tsv",
        rec_genomes="results/diversity_audit/recommended_genomes_status.tsv",
        tcs_coverage="results/diversity_audit/reference_tcs_coverage.tsv"
    shell:
        """
        python scripts/audit_species_diversity.py \
            --genome_dir data/genomes \
            --assembly_summary data/metadata/assembly_summary.txt \
            --reference_tcs {input.reference_tcs} \
            --outdir results/diversity_audit \
            --hk_annotation {input.hk_ann} \
            --rr_annotation {input.rr_ann}
        """
