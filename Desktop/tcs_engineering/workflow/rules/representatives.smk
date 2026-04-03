rule extract_hk_reps:
    """One representative HK sequence per MMseqs2 cluster (2,048 seqs)."""
    input:
        clusters="results/clusters/hk_clusters/clusters.tsv",
        fasta="results/hk_sequences.faa"
    output:
        "results/representatives/hk_reps.faa"
    shell:
        "python scripts/extract_cluster_reps.py --clusters {input.clusters} --fasta {input.fasta} --output {output}"


rule extract_rr_reps:
    """One representative RR sequence per MMseqs2 cluster (871 seqs)."""
    input:
        clusters="results/clusters/rr_clusters/clusters.tsv",
        fasta="results/rr_sequences.faa"
    output:
        "results/representatives/rr_reps.faa"
    shell:
        "python scripts/extract_cluster_reps.py --clusters {input.clusters} --fasta {input.fasta} --output {output}"


rule merge_reps:
    """Combine HK and RR representatives for joint phylogenetic analysis."""
    input:
        hk="results/representatives/hk_reps.faa",
        rr="results/representatives/rr_reps.faa"
    output:
        "results/representatives/tcs_reps.faa"
    shell:
        "cat {input.hk} {input.rr} > {output}"
