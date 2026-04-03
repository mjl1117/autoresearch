rule align_sequences:
    """Align cluster representatives (not all ~104K sequences)."""
    input:
        "results/representatives/tcs_reps.faa"
    output:
        "results/alignment/tcs_alignment.faa"
    shell:
        "mafft --auto --thread 8 {input} > {output}"


rule build_tree:
    input:
        "results/alignment/tcs_alignment.faa"
    output:
        "results/phylogeny/tcs_tree.treefile"
    shell:
        """
        mkdir -p results/phylogeny
        FastTree \
            -out {output} \
            {input}
        """
