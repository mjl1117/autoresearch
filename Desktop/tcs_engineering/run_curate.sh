#!/bin/bash
cd /Users/matthew/Desktop/tcs_engineering
tcs-env/bin/python scripts/curate_genome_set.py --assembly_summary data/metadata/assembly_summary.txt --genome_dir data/genomes --output data/reference/curated_genome_list.txt --report data/reference/curation_report.txt --max_per_genus 5
