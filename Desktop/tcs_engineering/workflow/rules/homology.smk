rule homology_search_hk:
    """All-vs-all sequence homology for histidine kinases.

    Uses MMseqs2 easy-search to compare all HK proteins against themselves.
    Output is BLAST-tabular format (m8): query, target, pident, alnlen,
    mismatches, gaps, qstart, qend, tstart, tend, evalue, bitscore.
    """
    input:
        "results/hk_sequences.faa"
    output:
        "results/homology/hk_homology.m8"
    params:
        min_seq_id=0.3,
        sensitivity=7.5  # mmseqs -s; 7.5 = high sensitivity
    shell:
        """
        mkdir -p results/homology/tmp_hk
        mmseqs easy-search \
            {input} {input} \
            {output} \
            results/homology/tmp_hk \
            --min-seq-id {params.min_seq_id} \
            -s {params.sensitivity} \
            --format-mode 0
        rm -rf results/homology/tmp_hk
        """


rule homology_search_rr:
    """All-vs-all sequence homology for response regulators."""
    input:
        "results/rr_sequences.faa"
    output:
        "results/homology/rr_homology.m8"
    params:
        min_seq_id=0.3,
        sensitivity=7.5
    shell:
        """
        mkdir -p results/homology/tmp_rr
        mmseqs easy-search \
            {input} {input} \
            {output} \
            results/homology/tmp_rr \
            --min-seq-id {params.min_seq_id} \
            -s {params.sensitivity} \
            --format-mode 0
        rm -rf results/homology/tmp_rr
        """


rule homology_search_cross:
    """Cross-type homology: HK vs RR.

    Detects proteins with homology across signal transduction families —
    useful for finding hybrid or bifunctional TCS proteins.
    """
    input:
        hk="results/hk_sequences.faa",
        rr="results/rr_sequences.faa"
    output:
        "results/homology/hk_vs_rr_homology.m8"
    params:
        min_seq_id=0.2,
        sensitivity=7.5
    shell:
        """
        mkdir -p results/homology/tmp_cross
        mmseqs easy-search \
            {input.hk} {input.rr} \
            {output} \
            results/homology/tmp_cross \
            --min-seq-id {params.min_seq_id} \
            -s {params.sensitivity} \
            --format-mode 0
        rm -rf results/homology/tmp_cross
        """
