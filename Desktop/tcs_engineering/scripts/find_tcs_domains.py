import subprocess

proteome="genomes/ecoli.faa"

subprocess.run([
"hmmsearch",
"--tblout",
"results/domain_hits.txt",
"data/pfam/Pfam-A.hmm",
proteome
])