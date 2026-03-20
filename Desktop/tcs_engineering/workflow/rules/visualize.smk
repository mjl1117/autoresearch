"""Visualization rules — run after pipeline completes.

All outputs land in results/visualization/.
Invoke individually or add to rule all after analysis rules complete.

Quick run (all viz):
  snakemake visualize_all --snakefile workflow/Snakefile --cores 4
"""


rule visualize_all:
    """Meta-rule: run all visualization steps."""
    input:
        "results/visualization/tcs_phylogeny.png",
        "results/visualization/tcs_umap.png",
        "results/visualization/chimera_candidates_heatmap.png",
        "results/visualization/tcs_regulatory_network.png",
        "results/visualization/phase_coherence_heatmap.png",
        "results/visualization/cluster_size_distribution.png",


rule viz_phylogeny:
    """Publication-quality TCS phylogenetic tree.

    Colours tips by HK (blue) vs RR (orange), marks DBD family by shape,
    and labels user-confirmed working-system representatives.
    """
    input:
        tree    = "results/phylogeny/tcs_tree.treefile",
        hk_ann  = "results/annotation/hk_annotation.tsv",
        rr_ann  = "results/annotation/rr_annotation.tsv",
        chimera = "results/chimera_targets/chimera_candidates.tsv",
    output:
        png = "results/visualization/tcs_phylogeny.png",
        pdf = "results/visualization/tcs_phylogeny.pdf",
    shell:
        """
        python scripts/visualize/plot_tcs_phylogeny.py \
            --tree    {input.tree} \
            --hk_ann  {input.hk_ann} \
            --rr_ann  {input.rr_ann} \
            --chimera {input.chimera} \
            --outdir  results/visualization \
            --max_tips 500
        """


rule viz_umap:
    """UMAP of TCS sequence space from pairwise MMseqs2 identity.

    Primary colour: protein type (HK/RR).
    Secondary colour: DBD family (OmpR_PhoB, NarL_FixJ, NtrC, CheY).
    Point size: cluster size.  Gold star: user working systems.
    """
    input:
        hk_hom  = "results/homology/hk_homology.m8",
        rr_hom  = "results/homology/rr_homology.m8",
        chimera = "results/chimera_targets/chimera_candidates.tsv",
        hk_ann  = "results/annotation/hk_annotation.tsv",
    output:
        png = "results/visualization/tcs_umap.png",
        pdf = "results/visualization/tcs_umap.pdf",
    shell:
        """
        python scripts/visualize/plot_tcs_umap.py \
            --hk_homology {input.hk_hom} \
            --rr_homology {input.rr_hom} \
            --chimera     {input.chimera} \
            --hk_ann      {input.hk_ann} \
            --outdir      results/visualization
        """


rule viz_chimera_structures:
    """Chimera structural feasibility panels.

    Per-candidate: domain architecture cartoon + pLDDT track + heptad phase wheel.
    Summary heatmap: top-15 candidates × key metrics.
    """
    input:
        chimera  = "results/chimera_targets/chimera_candidates.tsv",
        af2_dir  = "results/alphafold",
        plddt    = "results/deepcoil/af2_plddt_analysis.tsv",
    output:
        png = "results/visualization/chimera_candidates_heatmap.png",
        pdf = "results/visualization/chimera_candidates_heatmap.pdf",
    shell:
        """
        python scripts/visualize/plot_chimera_structures.py \
            --chimera   {input.chimera} \
            --af2_dir   {input.af2_dir} \
            --plddt_tsv {input.plddt} \
            --outdir    results/visualization \
            --top_n     8
        """


rule viz_regulatory_network:
    """TCS regulatory network: HK → RR cognate pairs from operon detection.

    Bipartite layout; node colour = signal type; node size = cluster size.
    Edges = cognate pair detected within 500 bp on same strand.
    """
    input:
        operon_dir    = "results/operons",
        hk_ann        = "results/annotation/hk_annotation.tsv",
        rr_ann        = "results/annotation/rr_annotation.tsv",
        chimera       = "results/chimera_targets/chimera_candidates.tsv",
        reference_tcs = "data/reference/well_characterized_tcs.tsv",
    output:
        png = "results/visualization/tcs_regulatory_network.png",
        pdf = "results/visualization/tcs_regulatory_network.pdf",
    shell:
        """
        python scripts/visualize/plot_regulatory_network.py \
            --operon_dir    {input.operon_dir} \
            --hk_ann        {input.hk_ann} \
            --rr_ann        {input.rr_ann} \
            --chimera       {input.chimera} \
            --reference_tcs {input.reference_tcs} \
            --outdir        results/visualization
        """


rule viz_phase_coherence:
    """Heptad register phase coherence heatmap + cluster size distribution.

    Heatmap: clusters sorted by phase coherence; working systems gold-bordered.
    Histogram: HK cluster size distribution with phase-compatible overlay.
    """
    input:
        chimera = "results/chimera_targets/chimera_candidates.tsv",
    output:
        heatmap  = "results/visualization/phase_coherence_heatmap.png",
        heatpdf  = "results/visualization/phase_coherence_heatmap.pdf",
        hist_png = "results/visualization/cluster_size_distribution.png",
        hist_pdf = "results/visualization/cluster_size_distribution.pdf",
    shell:
        """
        python scripts/visualize/plot_phase_coherence.py \
            --chimera {input.chimera} \
            --outdir  results/visualization \
            --top_n   50
        """
