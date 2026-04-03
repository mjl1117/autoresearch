import pandas as pd

summary = "data/metadata/assembly_summary.txt"

df = pd.read_csv(summary, sep="\t", skiprows=1)

df = df[df["assembly_level"] == "Complete Genome"]
df = df[df["ftp_path"] != "na"]

df["assembly"] = df["ftp_path"].str.split("/").str[-1]

df[["assembly", "ftp_path"]].to_csv(
    "data/metadata/genome_table.tsv",
    sep="\t",
    index=False
)
