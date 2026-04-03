rule prepare_rfdiffusion_inputs:
    """Select top chimera candidates for RFDiffusion linker redesign.

    For each HK_sensor_swap candidate with an AF2 structure:
      - Retrieve HAMP domain start from domtblout
      - Estimate sensor domain end from last detected pre-HAMP domain
      - Generate RFDiffusion contig string:
          A1-{sensor_end}/5-20/A{hamp_start}-{chain_length}
          (fix sensor → design 5-20 linker residues → fix HAMP onward)

    Contig design rationale (Hatstat 2025):
      The linker range 5-20 encompasses natural TCS linker lengths.
      Post-diffusion filtering keeps only designs where:
        designed_linker_length ≡ original_linker_length (mod 7)
      This enforces heptad register preservation without constraining
      the exact linker length — allowing RFDiffusion to explore phase-compatible
      linker geometries.
    """
    input:
        candidates="results/chimera_targets/chimera_candidates.tsv",
        af2_manifest="results/alphafold/af2_manifest.tsv"
    output:
        "results/rfdiffusion/candidates_for_design.tsv"
    params:
        n_candidates=config.get("rfdiffusion_n_candidates", 10)
    shell:
        """
        python scripts/prepare_rfdiffusion.py \
            --candidates {input.candidates} \
            --af2_manifest {input.af2_manifest} \
            --domtbl results/domains/hk_reps_domtbl.txt \
            --outdir results/rfdiffusion \
            --n_candidates {params.n_candidates}
        """


rule run_rfdiffusion:
    """Run RFDiffusion partial diffusion for chimera linker design.

    Partial diffusion mode: fixes flanking structured domains (sensor + kinase core),
    diffuses only the inter-domain linker region. This is the correct mode for
    linker redesign when the domain structures are known from AF2.

    Key parameters:
      partial_T: noise level for partial diffusion. Lower = more conservative
                 redesign, closer to the original linker geometry. Start at 0.15.
      n_designs: designs per candidate; 10 gives reasonable diversity for
                 phase-compatible sampling while keeping compute tractable.

    IMPORTANT — requires installation:
      git clone https://github.com/RosettaCommons/RFdiffusion
      cd RFdiffusion && pip install -e .
      Download weights to RFdiffusion/models/
    Set rfdiffusion_path in config/config.yaml.

    GPU strongly recommended. Apple MPS supported with recent PyTorch nightly.
    """
    input:
        "results/rfdiffusion/candidates_for_design.tsv"
    output:
        directory("results/rfdiffusion/designs/")
    params:
        rfdiffusion=config.get("rfdiffusion_path", "~/RFdiffusion"),
        n_designs=config.get("rfdiffusion_n_designs", 10),
        partial_T=config.get("rfdiffusion_partial_T", 0.15)
    shell:
        """
        python scripts/run_rfdiffusion.py \
            --candidates {input} \
            --rfdiffusion_path {params.rfdiffusion} \
            --n_designs {params.n_designs} \
            --partial_T {params.partial_T} \
            --outdir results/rfdiffusion/designs
        """
