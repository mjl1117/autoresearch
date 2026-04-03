rule split_hk_rr:
    """Split the merged TCS FASTA into separate HK and RR files using
    the per-genome classification CSVs so MMseqs2 can cluster each type
    independently."""
    input:
        faa="results/tcs_sequences.faa",
        csvs=expand("results/classifications/{genome}_proteins.csv", genome=genomes)
    output:
        hk="results/hk_sequences.faa",
        rr="results/rr_sequences.faa"
    shell:
        """
        python scripts/split_by_type.py \
            --faa {input.faa} \
            --csvs {input.csvs} \
            --hk_out {output.hk} \
            --rr_out {output.rr}
        """
