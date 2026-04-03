rule build_diamond_db:
    """Build DIAMOND database from Swiss-Prot.
    Swiss-Prot (reviewed only) is the correct choice here — TrEMBL adds ~200M
    unreviewed sequences, most unannotated, which degrades annotation specificity
    for functional inference."""
    input:
        "data/databases/uniprot_sprot.fasta"
    output:
        "data/databases/swissprot.dmnd"
    shell:
        "diamond makedb --in {input} --db data/databases/swissprot --threads 8"


rule annotate_hk:
    """DIAMOND Swiss-Prot annotation of HK cluster representatives.
    Uses sensitive mode; 1 hit per query; e-value 1e-5.
    CRITICAL: only true HKs reach this step post-classifier fix."""
    input:
        faa="results/representatives/hk_reps.faa",
        db="data/databases/swissprot.dmnd"
    output:
        "results/annotation/hk_annotation.tsv"
    shell:
        """
        mkdir -p results/annotation
        diamond blastp \
            --query {input.faa} \
            --db {input.db} \
            --out {output} \
            --outfmt 6 qseqid sseqid pident length qcovhsp evalue bitscore stitle \
            --max-target-seqs 1 \
            --evalue 1e-5 \
            --threads 8 \
            --sensitive
        """


rule annotate_rr:
    """DIAMOND Swiss-Prot annotation of RR cluster representatives."""
    input:
        faa="results/representatives/rr_reps.faa",
        db="data/databases/swissprot.dmnd"
    output:
        "results/annotation/rr_annotation.tsv"
    shell:
        """
        mkdir -p results/annotation
        diamond blastp \
            --query {input.faa} \
            --db {input.db} \
            --out {output} \
            --outfmt 6 qseqid sseqid pident length qcovhsp evalue bitscore stitle \
            --max-target-seqs 1 \
            --evalue 1e-5 \
            --threads 8 \
            --sensitive
        """


rule identify_chimera_targets:
    """Score and rank TCS proteins as chimera engineering candidates.

    Two strategies (see scripts/identify_chimera_targets.py for criteria):
      A) HK sensor swap  — conserved DHp+CA core, swappable sensor
      B) RR DBD swap     — OmpR/PhoB or NarL/FixJ family, modular linker junction

    Bioinformatics linker phase validation:
      HAMP domain boundaries from domtblout provide HAMP_start positions.
      Per-cluster heptad register coherence (HAMP_start mod 7) flags whether
      candidates are phase-compatible for sensor swap (Hatstat 2025).
    """
    input:
        hk_clusters="results/clusters/hk_clusters/clusters.tsv",
        rr_clusters="results/clusters/rr_clusters/clusters.tsv",
        hk_ann="results/annotation/hk_annotation.tsv",
        rr_ann="results/annotation/rr_annotation.tsv",
        cross="results/homology/hk_vs_rr_homology.m8",
        hk_domtbl="results/domains/hk_reps_domtbl.txt",
        reference_tcs="data/reference/well_characterized_tcs.tsv",
        characterized_promoters="data/reference/characterized_promoters.tsv",
    output:
        "results/chimera_targets/chimera_candidates.tsv"
    shell:
        """
        python scripts/identify_chimera_targets.py \
            --hk_clusters {input.hk_clusters} \
            --rr_clusters {input.rr_clusters} \
            --hk_annotation {input.hk_ann} \
            --rr_annotation {input.rr_ann} \
            --hk_cross_homology {input.cross} \
            --operon_dir results/operons \
            --hk_domtbl {input.hk_domtbl} \
            --reference_tcs {input.reference_tcs} \
            --characterized_promoters {input.characterized_promoters} \
            --output {output}
        """


rule screen_hamp_chimeras:
    """AF2 structural screening of HAMP sensor-swap chimera candidates.

    Domain-aware pLDDT analysis for transmembrane histidine kinases:
      - TM region (residues 1..hamp_start): EXCLUDED — AF2 always gives low
        pLDDT (~30-50) for membrane-embedded helices. This is expected and is
        NOT a design concern. Including TM pLDDT would penalise every valid
        TM-HK candidate unfairly.
      - HAMP domain (hamp_start..junction+30): target pLDDT > 70.
      - Kinase core DHp + CA (junction+30..end): target pLDDT > 70.

    Hatstat 2025 hydrophobic seam check: verifies that HAMP-AS1 heptad a/d
    positions (starting at junction+2) are occupied by hydrophobic residues
    (LIVAMF). Broken seam = structural correction required before synthesis.

    RefSeq WP_/NP_ accessions are mapped to UniProt via the UniProt REST API
    to enable EBI AlphaFold structure downloads.

    Output: hamp_chimera_screen.tsv with per-pair pLDDT stats and seam flags,
    ranked by structural confidence then junction identity.
    """
    input:
        candidates  = "results/chimera_targets/hamp_swap_candidates.tsv",
        hamp_fasta  = "results/deepcoil/hamp_linker_regions.faa",
        hk_reps     = "results/representatives/hk_reps.faa",
        domtbl      = "results/domains/hk_reps_domtbl.txt",
        hk_ann      = "results/annotation/hk_annotation.tsv",
    output:
        tsv = "results/chimera_screen/hamp_chimera_screen.tsv",
        pdb = directory("results/chimera_screen/pdb"),
    params:
        top_n    = config.get("chimera_screen_top_n", 50),
        motif_min= config.get("chimera_screen_motif_min", 5),
        plddt_min= config.get("chimera_screen_plddt_min", 70),
    shell:
        """
        mkdir -p results/chimera_screen
        python scripts/screen_hamp_chimeras.py \
            --candidates    {input.candidates} \
            --hamp_fasta    {input.hamp_fasta} \
            --hk_reps       {input.hk_reps} \
            --domtbl        {input.domtbl} \
            --hk_annotation {input.hk_ann} \
            --outdir        results/chimera_screen \
            --top_n         {params.top_n} \
            --motif_min     {params.motif_min} \
            --plddt_min     {params.plddt_min}
        """


rule hamp_centric_swap_candidates:
    """HAMP-centric sensor-swap candidate finder (decoupled from full-protein clustering).

    Selects HK pairs by HAMP sequence conservation (>= 70%) rather than
    full-protein identity, then confirms sensor divergence. This finds chimera
    partners that share a compatible signal-transduction linker but respond to
    different stimuli — the key criterion for sensor swapping per Peruzzi 2023.

    Phase matching (HAMP_start mod 7) is flagged so that structure prediction
    (AlphaFold/RFDiffusion) can prioritize phase-compatible pairs first.
    Pairs that pass HAMP identity but fail phase match are retained — they are
    candidates for heptad-corrected junction design per Hatstat 2025.
    """
    input:
        hamp_fasta = "results/deepcoil/hamp_linker_regions.faa",
        hk_reps    = "results/representatives/hk_reps.faa",
    output:
        "results/chimera_targets/hamp_swap_candidates.tsv"
    threads: 8
    shell:
        """
        python scripts/hamp_centric_swap_candidates.py \
            --hamp_fasta {input.hamp_fasta} \
            --hk_reps    {input.hk_reps} \
            --output     {output} \
            --threads    {threads}
        """


rule run_esmfold:
    """Run ESMFold on HAMP chimera candidates for domain-aware pLDDT.

    ESMFold (Lin et al. 2023) is run locally on each unique acceptor and donor
    protein in the chimera screen, producing per-residue pLDDT confidence scores
    and PDB structure files without requiring UniProt accessions or external APIs.

    Domain-aware analysis (mirrors AF2 screening in screen_hamp_chimeras.py):
      - TM region (1..hamp_start): EXCLUDED — ESMFold gives low pLDDT for
        membrane-embedded helices; this is expected and NOT a design concern.
      - HAMP domain (hamp_start..junction+30): target pLDDT > 70.
      - Kinase core DHp + CA (junction+30..end): target pLDDT > 70.

    Proteins longer than esmfold_max_length residues are skipped to avoid
    running out of memory (ESMFold memory scales quadratically with length).
    Set esmfold_device: "auto" in config to use MPS on Apple Silicon.

    Outputs:
      esmfold_plddt.tsv      — per-protein mean/median pLDDT + per-residue values
      hamp_chimera_screen_esm.tsv — screen TSV annotated with esm_* pLDDT columns
    """
    input:
        screen     = "results/chimera_screen/hamp_chimera_screen.tsv",
        hk_reps    = "results/representatives/hk_reps.faa",
        hamp_fasta = "results/deepcoil/hamp_linker_regions.faa",
    output:
        plddt  = "results/chimera_screen/esmfold/esmfold_plddt.tsv",
        screen = "results/chimera_screen/hamp_chimera_screen_esm.tsv",
    params:
        device     = config.get("esmfold_device", "auto"),
        max_length = config.get("esmfold_max_length", 800),
    shell:
        """
        python scripts/run_esmfold.py \
            --screen_tsv    {input.screen} \
            --hk_reps       {input.hk_reps} \
            --hamp_fasta    {input.hamp_fasta} \
            --outdir        results/chimera_screen/esmfold \
            --device        {params.device} \
            --max_length    {params.max_length} \
            --output_screen {output.screen}
        """


rule annotate_chimera_functions:
    """Functional annotation of HAMP chimera candidate proteins.

    Fetches NCBI protein records (gene name, product, organism) for each
    unique acceptor/donor protein in the chimera screen, cross-references
    with Swiss-Prot DIAMOND annotations, and infers HK sensor type from
    product description keywords and Pfam domain architecture.

    Output: chimera_functions.tsv — chimera screen TSV enriched with gene
    names, product descriptions, organism, and inferred sensor type for both
    acceptor and donor proteins in each pair.
    """
    input:
        screen_tsv    = "results/chimera_screen/hamp_chimera_screen_esm.tsv",
        hk_annotation = "results/annotation/hk_annotation.tsv",
        rr_annotation = "results/annotation/rr_annotation.tsv",
        domtbl        = "results/domains/hk_reps_domtbl.txt",
    output:
        "results/chimera_screen/chimera_functions_esm.tsv"
    params:
        email = config.get("entrez_email", "researcher@example.com")
    shell:
        """
        python scripts/annotate_chimera_functions.py \
            --screen_tsv    {input.screen_tsv} \
            --hk_annotation {input.hk_annotation} \
            --rr_annotation {input.rr_annotation} \
            --domtbl        {input.domtbl} \
            --output        {output} \
            --email         {params.email}
        """


rule search_reference_analogs:
    """Cross-species functional analog search against model organism proteomes.

    Maps each chimera candidate to its closest characterized relative in
    E. coli K-12 MG1655, B. subtilis 168, and P. aeruginosa PAO1 using
    DIAMOND BLASTp (sensitive mode, min 25% identity). Gene names are
    extracted from GFF3 files to provide named analogs (e.g., cusS, envZ).

    This is complementary to Swiss-Prot DIAMOND annotation:
      - Swiss-Prot: best reviewed hit, may be distant or absent
      - Model organism search: closest characterized relative, reveals
        the signaling system even at 25-35% identity

    Outputs chimera_annotated.tsv with acceptor_ref_gene, donor_ref_gene,
    ref_organism, and analog_pident columns for all 50 screened candidates.
    """
    input:
        screen_tsv = "results/chimera_screen/chimera_functions_esm.tsv",
        hk_reps    = "results/representatives/hk_reps.faa",
    output:
        tsv      = "results/chimera_screen/chimera_annotated.tsv",
        hits     = "results/chimera_screen/reference_analogs/reference_analog_hits.m8",
    params:
        min_pident = config.get("analog_min_pident", 25),
    threads: 4
    shell:
        """
        python scripts/search_reference_analogs.py \
            --screen_tsv  {input.screen_tsv} \
            --hk_reps     {input.hk_reps} \
            --outdir      results/chimera_screen/reference_analogs \
            --output      {output.tsv} \
            --min_pident  {params.min_pident} \
            --threads     {threads}
        """


rule setup_eggnogmapper:
    """Download eggNOG-mapper databases for bacteria (taxid 2).

    Downloads the annotation SQLite database (~8 GB) and bacteria-specific
    HMMER profiles (~1-2 GB) into data/eggnog_db/. Run this once before
    the annotate_eggnogmapper rule.

    Uses HMMER mode (not DIAMOND) to avoid downloading the full 24 GB
    eggnog_proteins.dmnd database. Bacteria-specific HMM profiles are
    sufficient for TCS sensor and kinase domain annotation.
    """
    output:
        db = "data/eggnog_db/eggnog.db",
    shell:
        """
        mkdir -p data/eggnog_db
        conda run -n esmfold-env -- download_eggnog_data.py \
            -y \
            -D \
            -H -d 2 --dbname Bacteria \
            --data_dir data/eggnog_db
        """


rule annotate_eggnogmapper:
    """eggNOG-mapper annotation of HK representative sequences.

    Maps TCS proteins to eggNOG Orthologous Groups (OGs) using HMMER against
    bacteria-specific profiles. Provides:
      - COG functional category (e.g., T=signal transduction, K=transcription)
      - OG description (OG name from eggNOG 5.0)
      - GO terms
      - KEGG pathway / reaction IDs
      - Best eggNOG OG at species, genus, family, and bacteria level

    Requires setup_eggnogmapper to have been run first.

    Note: eggNOG-mapper must be run in the esmfold-env (Python 3.11) because
    eggnog-mapper is not compatible with Python 3.14 used by tcs-env.
    """
    input:
        db      = "data/eggnog_db/eggnog.db",
        hk_reps = "results/representatives/hk_reps.faa",
    output:
        "results/annotation/eggnogmapper/hk_reps.emapper.annotations"
    threads: 8
    shell:
        """
        mkdir -p results/annotation/eggnogmapper
        conda run -n esmfold-env -- emapper.py \
            -i     {input.hk_reps} \
            --data_dir data/eggnog_db \
            -m     hmmer \
            -d     Bacteria \
            --tax_scope bacteria \
            --cpu  {threads} \
            --output results/annotation/eggnogmapper/hk_reps \
            --override
        """


rule build_rr_chimera_sequences:
    """Build chimeric RR sequences to complete orthogonal TCS phosphorelay pairs.

    Constructs RR chimeras of the form:
        RR_A[receiver domain] + RR_B[DBD]

    RR_A is the cognate RR for the chimeric SHK's kinase core — it accepts
    phosphorylation correctly. RR_B contributes its DNA-binding domain to
    drive a desired reporter promoter.

    Together with SHK chimeras, these form complete orthogonal phosphorelays:
        New sensor → chimeric SHK kinase → chimeric RR receiver
                   → chimeric RR DBD → target promoter → reporter

    Receiver-DBD split positions (from literature):
      OmpR/PhoB family: residue 125  (Schmidl 2019, Kottur 2023)
      NarL/FixJ family: residue 130  (Maris 2002, Peruzzi 2023)
    """
    input:
        candidates  = "results/chimera_targets/chimera_candidates.tsv",
        rr_reps     = "results/representatives/rr_reps.faa",
        shk_manifest= "results/chimera_design/chimera_design_manifest.tsv",
    output:
        faa      = "results/chimera_design/rr_chimera_sequences.faa",
        manifest = "results/chimera_design/rr_chimera_design_manifest.tsv",
    shell:
        """
        python scripts/build_rr_chimera_sequences.py \
            --candidates   {input.candidates} \
            --rr_reps      {input.rr_reps} \
            --shk_manifest {input.shk_manifest} \
            --outdir       results/chimera_design
        """


rule build_chimera_sequences:
    """Construct chimeric protein sequences from validated HAMP swap pairs.

    For each pair passing ESMFold + seam filters, builds the fusion protein:
        acceptor[1 : junction_pos_a] + donor[junction_pos_b : end]

    The junction positions mark the N-x-[ML] HAMP AS-1 start — the natural
    crossover point where sensor domain ends and signal transduction begins.
    Acceptor contributes its input-sensing domain; donor contributes its
    HAMP + DHp + CA kinase core (which routes signal to the donor RR).

    Output:
      chimera_sequences.faa       — FASTA for ESMFold validation of the fusions
      chimera_design_manifest.tsv — metadata including junction_seq window,
                                    parent pLDDT scores, and sensor gene names
    """
    input:
        screen_tsv = "results/chimera_screen/hamp_chimera_screen_esm.tsv",
        hk_reps    = "results/representatives/hk_reps.faa",
    output:
        faa      = "results/chimera_design/chimera_sequences.faa",
        manifest = "results/chimera_design/chimera_design_manifest.tsv",
    params:
        cross_sensor_flag = "--cross_sensor_only" if config.get("chimera_cross_sensor_only", False) else "",
    shell:
        """
        mkdir -p results/chimera_design
        python scripts/build_chimera_sequences.py \
            --screen_tsv  {input.screen_tsv} \
            --hk_reps     {input.hk_reps} \
            --outdir      results/chimera_design \
            {params.cross_sensor_flag}
        """


rule fold_chimeras:
    """Fold chimeric sequences with ESMFold to validate the fusion junction.

    Runs the same domain-aware ESMFold pipeline as run_esmfold, but on the
    chimeric sequences themselves (not the parent proteins). A chimera with
    high HAMP pLDDT (>70) is likely to maintain the 4-helix bundle geometry
    required for signal transduction, even if parents individually fold well.

    This is the critical computational validation step before synthesis:
    it tests whether the junction is compatible at the structural level,
    not just at the sequence motif level.
    """
    input:
        screen_tsv = "results/chimera_design/chimera_fold_screen.tsv",
        faa        = "results/chimera_design/chimera_sequences.faa",
        hamp_faa   = "results/chimera_design/chimera_hamp_positions.faa",
    output:
        plddt      = "results/chimera_design/esmfold/esmfold_plddt.tsv",
        screen     = "results/chimera_design/chimera_validated.tsv",
    params:
        device     = config.get("esmfold_device", "cpu"),
        max_length = config.get("esmfold_max_length", 1000),
    shell:
        """
        mkdir -p results/chimera_design/esmfold
        conda run -n esmfold-env -- python scripts/run_esmfold.py \
            --screen_tsv    {input.screen_tsv} \
            --hk_reps       {input.faa} \
            --hamp_fasta    {input.hamp_faa} \
            --outdir        results/chimera_design/esmfold \
            --device        {params.device} \
            --max_length    {params.max_length} \
            --output_screen {output.screen}
        """
