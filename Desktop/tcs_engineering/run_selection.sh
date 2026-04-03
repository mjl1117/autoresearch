#!/bin/bash
source tcs-env/bin/activate
python scripts/select_expansion_genomes.py \
    --assembly_summary data/metadata/assembly_summary.txt \
    --genome_dir data/genomes \
    --output_manifest data/reference/expansion_download_manifest.txt \
    --output_list data/reference/curated_genome_list.txt \
    --target_n 200
