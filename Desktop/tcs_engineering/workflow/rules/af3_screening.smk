rule prepare_af3_dbd_promoter:
    """Prepare AlphaFold3 input JSONs for RR DBD × characterized promoter screening.

    For each top RR DBD-swap candidate (default top 10) × each recommended
    output promoter: extracts the C-terminal DBD sequence using family-specific
    boundary heuristics (OmpR_PhoB: ~90 aa, NarL_FixJ: ~65 aa, NtrC_AAA: ~55 aa)
    and pairs it with the promoter binding site DNA (sense + antisense strands).

    Binding sites used (from literature):
      OmpR box (F1 consensus, 20 bp) — Rampersaud 1989
      Pho box (18 bp, two half-sites)  — Makino 1988
      NarL box (two heptanucleotide TACYYMT half-sites) — Darwin 1997
      NtrC UAS at glnAp2 (22 bp)       — Reitzer 1989

    Output JSONs are ready for:
      - AlphaFold3 web server (https://alphafoldserver.com) — upload JSON directly
      - AlphaFold3 local installation (run_alphafold.py)
      - ColabFold AF3 notebook

    ipTM > 0.5 = confident interface predicted (AF3 threshold, Abramson 2024).
    ipTM < 0.3 = likely no specific interaction.

    Re-run this rule after depositing AF3 output directories in
    results/af3_screening/outputs/{pair_id}/ to parse scores automatically.
    """
    input:
        candidates="results/chimera_targets/chimera_candidates.tsv",
        promoters="data/reference/characterized_promoters.tsv",
        rr_fasta="results/representatives/rr_reps.faa"
    output:
        tsv=   "results/af3_screening/af3_dbd_promoter_scores.tsv",
        indir= directory("results/af3_screening/inputs")
    params:
        top_n=      config.get("af3_top_n",     10),
        af3_flags=  ("--run_af3 --af3_dir " + str(config.get("af3_dir", "~/alphafold3")))
                    if config.get("af3_run_local", False) else ""
    shell:
        """
        python scripts/af3_dbd_promoter.py \
            --candidates {input.candidates} \
            --promoters  {input.promoters} \
            --rr_fasta   {input.rr_fasta} \
            --outdir     results/af3_screening \
            --top_n      {params.top_n} \
            {params.af3_flags}
        """
