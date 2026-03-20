rule validate_ground_truth:
    """Systematically check pipeline outputs against well_characterized_tcs.tsv.

    Runs five checks (GT-1 through GT-5) covering:
      GT-1  HK annotation coverage — working systems in HK DIAMOND hits
      GT-2  RR annotation coverage — working systems in RR DIAMOND hits
      GT-3  Chimera candidate coverage — working systems as known_tcs_system
      GT-4  HK classifier specificity — no FtsZ/HtpG/GyrB/MutL contaminants
      GT-5  Phase coherence present — working-system candidates have phase columns

    Status semantics:
      PASS — criterion met
      WARN — criterion unmet but not fatal (organism may not be in genome set)
      FAIL — indicates a pipeline bug; exits with code 1

    Currently confirmed working systems (working_in_user_system=yes):
      NarXL  (E. coli nitrate sensor; user confirmed)
      PhoRB  (E. coli phosphate sensor; user confirmed)
    """
    input:
        reference_tcs="data/reference/well_characterized_tcs.tsv",
        hk_ann="results/annotation/hk_annotation.tsv",
        rr_ann="results/annotation/rr_annotation.tsv",
        candidates="results/chimera_targets/chimera_candidates.tsv"
    output:
        tsv="results/validation/ground_truth_validation.tsv",
        summary="results/validation/validation_summary.txt"
    shell:
        """
        python scripts/validate_ground_truth.py \
            --reference_tcs {input.reference_tcs} \
            --hk_annotation {input.hk_ann} \
            --rr_annotation {input.rr_ann} \
            --chimera_candidates {input.candidates} \
            --outdir results/validation
        """
