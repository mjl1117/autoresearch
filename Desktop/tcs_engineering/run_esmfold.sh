#!/usr/bin/env bash
# Run local ESMFold structural screening for HAMP chimera candidates.
# Uses the fixed -- separator to prevent conda from consuming script arguments.
# Run from the tcs_engineering project root.

set -euo pipefail

conda run -n esmfold-env -- python scripts/run_esmfold.py \
    --screen_tsv    results/chimera_screen/hamp_chimera_screen.tsv \
    --hk_reps       results/representatives/hk_reps.faa \
    --hamp_fasta    results/deepcoil/hamp_linker_regions.faa \
    --outdir        results/chimera_screen/esmfold \
    --output_screen results/chimera_screen/hamp_chimera_screen_esm.tsv \
    --device        cpu \
    --batch

echo "ESMFold run complete."
