rule extract_hk_rr_sequences:
    input:
        hmm="results/domains/{genome}.tbl",
        proteins="data/genomes/{genome}/proteins.faa"
    output:
        faa="results/sequences/{genome}.faa",
        csv="results/classifications/{genome}_proteins.csv"
    shell:
        """
        python scripts/01_detect_domains.py \
            --proteins {input.proteins} \
            --hmm_table {input.hmm} \
            --output {output.faa} \
            --csv_output {output.csv}
        """
