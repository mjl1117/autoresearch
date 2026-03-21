rule build_diamond_db:
    """Build DIAMOND database from Swiss-Prot.
    Swiss-Prot (reviewed only) is the correct choice here — TrEMBL adds ~200M
    unreviewed sequences, most unannotated, which degrades annotation specificity
    for functional inference."""
    input:
        "data/databases/uniprot_sprot.fasta"
    output:
        "data/databases/swissprot.dmnd"
    shell:
        "diamond makedb --in {input} --db data/databases/swissprot --threads 8"


rule annotate_hk:
    """DIAMOND Swiss-Prot annotation of HK cluster representatives.
    Uses sensitive mode; 1 hit per query; e-value 1e-5.
    CRITICAL: only true HKs reach this step post-classifier fix."""
    input:
        faa="results/representatives/hk_reps.faa",
        db="data/databases/swissprot.dmnd"
    output:
        "results/annotation/hk_annotation.tsv"
    shell:
        """
        mkdir -p results/annotation
        diamond blastp \
            --query {input.faa} \
            --db {input.db} \
            --out {output} \
            --outfmt 6 qseqid sseqid pident length qcovhsp evalue bitscore stitle \
            --max-target-seqs 1 \
            --evalue 1e-5 \
            --threads 8 \
            --sensitive
        """


rule annotate_rr:
    """DIAMOND Swiss-Prot annotation of RR cluster representatives."""
    input:
        faa="results/representatives/rr_reps.faa",
        db="data/databases/swissprot.dmnd"
    output:
        "results/annotation/rr_annotation.tsv"
    shell:
        """
        mkdir -p results/annotation
        diamond blastp \
            --query {input.faa} \
            --db {input.db} \
            --out {output} \
            --outfmt 6 qseqid sseqid pident length qcovhsp evalue bitscore stitle \
            --max-target-seqs 1 \
            --evalue 1e-5 \
            --threads 8 \
            --sensitive
        """


rule identify_chimera_targets:
    """Score and rank TCS proteins as chimera engineering candidates.

    Two strategies (see scripts/identify_chimera_targets.py for criteria):
      A) HK sensor swap  — conserved DHp+CA core, swappable sensor
      B) RR DBD swap     — OmpR/PhoB or NarL/FixJ family, modular linker junction

    Bioinformatics linker phase validation:
      HAMP domain boundaries from domtblout provide HAMP_start positions.
      Per-cluster heptad register coherence (HAMP_start mod 7) flags whether
      candidates are phase-compatible for sensor swap (Hatstat 2025).
    """
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
