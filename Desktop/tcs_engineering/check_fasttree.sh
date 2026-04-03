#!/bin/bash
cd /Users/matthew/Desktop/tcs_engineering
echo "=== FastTree locations ==="
ls miniforge3/bin/FastTree 2>/dev/null || echo "NOT in miniforge3/bin"
ls tcs-env/bin/FastTree 2>/dev/null || echo "NOT in tcs-env/bin"
which FastTree 2>/dev/null || echo "FastTree not on PATH"
echo "=== alignment file ==="
ls -lh results/alignment/tcs_alignment.faa 2>/dev/null || echo "alignment file missing"
echo "=== first 2 lines of alignment ==="
head -2 results/alignment/tcs_alignment.faa 2>/dev/null
