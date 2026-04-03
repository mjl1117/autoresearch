rule cluster_tcs:
    input:
        hk="results/hk_sequences.faa",
        rr="results/rr_sequences.faa"
    output:
        hk="results/clusters/hk_clusters/clusters.tsv",
        rr="results/clusters/rr_clusters/clusters.tsv"
    shell:
        """
        python scripts/03_cluster_sequences.py \
            --hk_fasta {input.hk} \
            --rr_fasta {input.rr} \
            --outdir results/clusters
        """
