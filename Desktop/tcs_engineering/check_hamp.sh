#!/bin/bash
cd /Users/matthew/Desktop/tcs_engineering
DOMTBL="results/domains/hk_reps_domtbl.txt"
echo "=== Total non-comment lines ==="
grep -v "^#" "$DOMTBL" | wc -l
echo "=== Unique domain names (col 3) ==="
grep -v "^#" "$DOMTBL" | awk '{print $3}' | sort | uniq -c | sort -rn | head -30
echo "=== Lines matching HAMP ==="
grep -i "hamp" "$DOMTBL" | head -5
