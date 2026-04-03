rule hmmsearch_hk_reps_domtbl:
    """HMMER domain boundary search on HK cluster representatives.

    Uses --domtblout (per-domain hits with sequence coordinates) rather than
    --tblout (per-sequence). This provides ali_from / ali_to per domain instance,
    required for:
      1. HAMP domain boundary extraction → linker phase analysis in chimera scoring
      2. Sensor domain end estimation → RFDiffusion contig specification
      3. HAMP linker window extraction → heptad register analysis

    Only runs on the cluster representative subset (~3K sequences), not all genomes.
    """
    input:
        faa="results/representatives/hk_reps.faa",
        hmm="data/pfam_tcs.hmm"
    output:
        "results/domains/hk_reps_domtbl.txt"
    threads: 8
    shell:
        "hmmsearch --domtblout {output} --cpu {threads} {input.hmm} {input.faa} > /dev/null"


rule extract_hamp_linkers:
    """Extract HAMP domain + flanking linker windows from HK representatives.

    Window: [HAMP_start - {upstream} : HAMP_start + {downstream}] residues.
    This captures the sensor→HAMP linker (register-critical) plus the HAMP
    domain N-terminal helix (coiled-coil structured).

    Proteins without a detected HAMP domain are excluded — they lack the
    sensor-kinase coupling interface required for sensor swap chimeras.
    """
    input:
        faa="results/representatives/hk_reps.faa",
        domtbl="results/domains/hk_reps_domtbl.txt"
    output:
        "results/deepcoil/hamp_linker_regions.faa"
    params:
        upstream=config.get("deepcoil_upstream_residues", 30),
        downstream=config.get("deepcoil_downstream_residues", 50)
    shell:
        """
        mkdir -p results/deepcoil
        python scripts/extract_hamp_linkers.py \
            --hk_reps {input.faa} \
            --domtbl {input.domtbl} \
            --output {output} \
            --upstream {params.upstream} \
            --downstream {params.downstream}
        """


rule hamp_register_analysis:
    """Heptad register analysis of HAMP linker regions.

    Implements coiled-coil heptad scoring from first principles (Kyte-Doolittle
    hydrophobicity at a/d positions) rather than using DeepCoil, which requires
    numpy<1.19 and cannot be installed on modern Python environments.

    For each HAMP linker window:
      - Tests all 7 heptad phases
      - Returns the dominant phase (highest hydrophobic moment at a/d positions)
      - coil_score: fraction of a/d positions with hydrophobic residues (0–1)
      - phase_confident: True if coil_score >= 0.5

    Cross-check: HAMP_start mod 7 (from HMMER boundaries) should agree with the
    dominant phase from sequence analysis — concordance validates the bioinformatics
    linker phase assignments in chimera_candidates.tsv.
    """
    input:
        "results/deepcoil/hamp_linker_regions.faa"
    output:
        "results/deepcoil/hamp_register_predictions.tsv"
    shell:
        """
        python scripts/hamp_register_analysis.py \
            --hamp_linkers {input} \
            --output {output}
        """


rule analyze_af2_hamp_plddt:
    """Extract per-residue pLDDT scores for HAMP regions from AlphaFold2 PDBs.

    pLDDT (B-factor column in AF2 PDB) is the per-residue confidence metric.
    For chimera design:
      pLDDT > 70 in HAMP helix → reliable structure for register inference
      pLDDT < 50 in linker     → disordered connection (design risk flag)

    Outputs a summary table with:
      plddt_hamp_mean:    mean pLDDT over HAMP domain residues
      plddt_linker_mean:  mean pLDDT over pre-HAMP linker (30 residues)
      hamp_high_confidence: True if plddt_hamp_mean > 70
    """
    input:
        af2_manifest="results/alphafold/af2_manifest.tsv",
        domtbl="results/domains/hk_reps_domtbl.txt"
    output:
        "results/deepcoil/af2_plddt_analysis.tsv"
    shell:
        """
        python scripts/analyze_af2_hamp_plddt.py \
            --af2_manifest {input.af2_manifest} \
            --domtbl {input.domtbl} \
            --output {output}
        """
