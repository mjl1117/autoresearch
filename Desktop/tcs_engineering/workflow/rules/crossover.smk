rule prostt5_crossover_scoring:
    """Score HAMP-DHp crossover points using ProstT5 3Di structural alphabet.

    For each HK_sensor_swap candidate pair, ProstT5 encodes each residue as a
    3Di structural token (same alphabet as Foldseek). Positions where donor and
    chassis tokens are identical have the lowest structural disruption — these
    are preferred crossover sites. Combined with heptad phase analysis (Hatstat
    2025), this gives a ranked list of crossover candidates without requiring
    solved structures.

    Model: Rostlab/ProstT5 (~2 GB, downloaded once from HuggingFace on first run).
    Device: cpu (default) | mps (Apple Silicon) | cuda — set prostt5_device in config.yaml.

    Outputs an empty TSV if no HK_sensor_swap candidates are present (expected
    with small genome sets; expand genome set to ≥50 clusters to see candidates).

    Note: SCHEMA/RASPP is the gold standard crossover scoring method when contact
    maps are available from AF2 or PDB structures. ProstT5 is used here as a
    no-structure alternative. Replace with SCHEMA when AF2 structures are available.
    """
    input:
        candidates="results/chimera_targets/chimera_candidates.tsv",
        hk_fasta="results/representatives/hk_reps.faa",
        domtbl="results/domains/hk_reps_domtbl.txt"
    output:
        "results/crossover/prostt5_crossover_scores.tsv"
    params:
        device=config.get("prostt5_device", "cpu"),
        top_n=config.get("crossover_top_n", 5)
    shell:
        """
        python scripts/find_prostt5_crossovers.py \
            --candidates {input.candidates} \
            --hk_fasta   {input.hk_fasta} \
            --domtbl     {input.domtbl} \
            --outdir     results/crossover \
            --top_n      {params.top_n} \
            --device     {params.device}
        """
