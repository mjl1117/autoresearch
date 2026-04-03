rule merge_operons:
    input:
        expand("results/operons/{genome}.tsv", genome=genomes)
    output:
        "results/operons/tcs_operons.tsv"
    shell:
        "cat {input} > {output}"

rule pair_adjacent_genes:
    input:
        proteins="results/classifications/{genome}_proteins.csv",
        gff="data/genomes/{genome}/genome.gff"  # Updated to match genomes.smk output
    output:
        "results/operons/{genome}.tsv"
    shell:
        "python scripts/02_pair_adjacent_genes.py --proteins {input.proteins} --gff {input.gff} --out {output}"
