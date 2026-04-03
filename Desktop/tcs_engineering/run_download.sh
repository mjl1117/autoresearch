#!/bin/bash
source tcs-env/bin/activate
mkdir -p logs
python scripts/00_download_genomes.py \
    --manifest data/reference/expansion_download_manifest.txt \
    --data_dir data
