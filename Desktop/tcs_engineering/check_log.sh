#!/bin/bash
cd /Users/matthew/Desktop/tcs_engineering
LOG=$(ls -t .snakemake/log/*.snakemake.log | head -1)
echo "Checking: $LOG"
grep -iE "error|diamond|illegal|cannot|failed|killed|signal" "$LOG" | tail -40
