rule merge_sequences:
    input:
        expand("results/sequences/{genome}.faa", genome=genomes)

    output:
        "results/tcs_sequences.faa"

    shell:
        "cat {input} > {output}"