rule hmmsearch_domains:
    input:
        "data/genomes/{genome}/proteins.faa"

    output:
        "results/domains/{genome}.tbl"

    threads: 4

    shell:
        """
        hmmsearch \
            --tblout {output} \
            data/pfam_tcs.hmm \
            {input}
        """