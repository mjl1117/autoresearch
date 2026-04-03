#!/bin/bash
cd /Users/matthew/Desktop/tcs_engineering
mkdir -p logs
ln -sf /Users/matthew/Desktop/tcs_engineering/data/pfam/Pfam-A.hmm /Users/matthew/Desktop/tcs_engineering/data/pfam_tcs.hmm
nohup tcs-env/bin/snakemake --snakefile workflow/Snakefile --cores 8 --rerun-incomplete --latency-wait 15 > logs/snakemake_run.log 2>&1 &
echo "Pipeline started with PID $!"
tail -f logs/snakemake_run.log
