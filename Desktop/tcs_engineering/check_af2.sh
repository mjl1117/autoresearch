#!/bin/bash
cd /Users/matthew/Desktop/tcs_engineering
echo "=== AF2 manifest ==="
wc -l results/alphafold/af2_manifest.tsv 2>/dev/null || echo "manifest missing"
head -3 results/alphafold/af2_manifest.tsv 2>/dev/null
echo "=== AF2 plddt error ==="
tcs-env/bin/python scripts/analyze_af2_hamp_plddt.py \
    --af2_manifest results/alphafold/af2_manifest.tsv \
    --domtbl results/domains/hk_reps_domtbl.txt \
    --output /tmp/af2_test.tsv 2>&1 | head -30
