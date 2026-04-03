#!/bin/bash
cd /Users/matthew/Desktop/tcs_engineering
echo "=== HAMP profile in Pfam-A.hmm ==="
grep -m3 "^NAME.*HAMP" data/pfam/Pfam-A.hmm
echo "=== Domain names in domtbl (col 4 = parts[3]) ==="
grep -v "^#" results/domains/hk_reps_domtbl.txt | awk '{print $4}' | sort | uniq -c | sort -rn | head -20
echo "=== HAMP hits in domtbl ==="
grep -v "^#" results/domains/hk_reps_domtbl.txt | awk '$4=="HAMP"' | wc -l
