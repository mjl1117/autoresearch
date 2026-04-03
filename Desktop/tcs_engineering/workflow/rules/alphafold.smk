rule download_alphafold_structures:
    """Download AlphaFold2 predicted structures from EBI for top chimera candidates.

    Maps DIAMOND Swiss-Prot best-hit accessions (sp|UniProtID|...) to UniProt IDs
    and fetches PDB files from:
      https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb

    B-factor column in AF2 PDBs = per-residue pLDDT (0–100).
    pLDDT > 70 in the HAMP helix → reliable structural model for linker analysis.
    pLDDT < 50 in the linker region → intrinsically disordered → chimera design risk.

    Outputs af2_manifest.tsv linking protein_id → uniprot_id → pdb_path.
    """
    input:
        "results/chimera_targets/chimera_candidates.tsv"
    output:
        "results/alphafold/af2_manifest.tsv"
    params:
        max_structures=config.get("alphafold_max_structures", 150),
        af2_version=config.get("alphafold_version", 4)
    shell:
        """
        python scripts/download_alphafold.py \
            --chimera_candidates {input} \
            --outdir results/alphafold \
            --max_structures {params.max_structures} \
            --af2_version {params.af2_version}
        """
