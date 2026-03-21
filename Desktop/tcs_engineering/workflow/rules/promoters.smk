# ── DEPRECATED rules ────────────────────────────────────────────────────────
# extract_promoters, merge_promoters, find_motifs were Stage 7 of the pipeline.
# They extracted 200-250 bp upstream of RR GENES (sigma70 sites controlling
# TCS expression), which are NOT the output promoters that RR DBDs bind.
# This is scientifically invalid for gene circuit design — see Bug #15 in
# PIPELINE_DOCS.md. These rules are retained for reference but excluded from
# rule all and replaced by the fetch_promoters rule below.

rule extract_promoters_DEPRECATED:
    input:
        sequences = "results/sequences/{genome}.faa",
        gff = "data/genomes/{genome}/genome.gff",
        fna = "data/genomes/{genome}/genome.fna"
    output:
        "results/sequences/{genome}_promoters.fasta"
    params:
        upstream = 200
    shell:
        """
        python scripts/05_extract_promoters.py \
            --gff {input.gff} \
            --fna {input.fna} \
            --ids {input.sequences} \
            --upstream {params.upstream} \
            --output {output}
        """


rule merge_promoters_DEPRECATED:
    """DEPRECATED — see Bug #15 in PIPELINE_DOCS.md."""
    input:
        promoters=expand("results/sequences/{genome}_promoters.fasta", genome=genomes),
        rr_reps="results/representatives/rr_reps.faa"
    output:
        "results/promoters/promoters_for_meme.fasta"
    shell:
        """
        mkdir -p results/promoters
        python scripts/subsample_promoters.py \
            --promoters {input.promoters} \
            --rr_reps {input.rr_reps} \
            --output {output} \
            --max_seqs 600
        """


rule find_motifs_DEPRECATED:
    """DEPRECATED — see Bug #15 in PIPELINE_DOCS.md."""
    input:
        "results/promoters/promoters_for_meme.fasta"
    output:
        "results/promoters/meme.html"
    shell:
        "meme {input} -oc results/promoters -dna -nmotifs 5 -minw 6 -maxw 20"


# ── NEW: Characterized output promoter pipeline (Stage 20) ──────────────────

rule fetch_promoters:
    """Download 300 bp upstream sequences for characterized TCS output promoters.

    Uses NCBI Entrez (gbwithparts) to fetch sequences from source genomes.
    Skips CheY (no DNA binding domain), non-recommended, and entries with
    no annotated gene name.

    Requires internet access and ncbi_email set in config.yaml.
    PthsA (Shewanella woodyi thiosulfate promoter) must be retrieved manually
    from Daeffler & Tabor 2017 (PMID 28373240) supplementary materials.
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
