# tests/test_select_expansion_genomes.py
"""Tests for select_expansion_genomes.py selection logic."""
import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from select_expansion_genomes import (
    parse_assembly_summary,
    select_seed,
    select_priority,
    select_diversity,
)
from tcs_constants import ASSEMBLY_LEVEL_PRIORITY, RECOMMENDED_GENOMES


def _make_asm_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal assembly summary DataFrame matching parse_assembly_summary output."""
    df = pd.DataFrame(rows)
    df["full_id"] = df["accession"] + "_" + df["asm_name"]
    df["genus"] = df["organism_name"].str.split().str[0]
    df["level_rank"] = df["assembly_level"].map(ASSEMBLY_LEVEL_PRIORITY).fillna(99)
    df["already_downloaded"] = False
    return df


def test_select_priority_takes_best_quality():
    """Priority pass picks Complete Genome over Scaffold from same genus."""
    df = _make_asm_df([
        {"accession": "GCF_000001.1", "asm_name": "asm1",
         "organism_name": "Pseudomonas aeruginosa", "assembly_level": "Scaffold",
         "ftp_path": "ftp://ftp.ncbi.nlm.nih.gov/path/GCF_000001.1_asm1"},
        {"accession": "GCF_000002.1", "asm_name": "asm2",
         "organism_name": "Pseudomonas fluorescens", "assembly_level": "Complete Genome",
         "ftp_path": "ftp://ftp.ncbi.nlm.nih.gov/path/GCF_000002.1_asm2"},
    ])
    result = select_priority(df, priority_genera=["Pseudomonas"])
    assert len(result) == 1
    assert result.iloc[0]["accession"] == "GCF_000002.1"


def test_select_priority_prefers_already_downloaded():
    """Priority pass prefers already-downloaded genome over better-assembly-level one."""
    df = _make_asm_df([
        {"accession": "GCF_000001.1", "asm_name": "asm1",
         "organism_name": "Pseudomonas aeruginosa", "assembly_level": "Complete Genome",
         "ftp_path": "ftp://example/GCF_000001.1_asm1"},
        {"accession": "GCF_000002.1", "asm_name": "asm2",
         "organism_name": "Pseudomonas fluorescens", "assembly_level": "Scaffold",
         "ftp_path": "ftp://example/GCF_000002.1_asm2"},
    ])
    df.loc[df["accession"] == "GCF_000002.1", "already_downloaded"] = True
    result = select_priority(df, priority_genera=["Pseudomonas"])
    assert len(result) == 1
    assert result.iloc[0]["accession"] == "GCF_000002.1"


def test_select_priority_skips_absent_genus(capsys):
    """Priority pass logs warning and skips genus with no catalog entries."""
    df = _make_asm_df([
        {"accession": "GCF_000001.1", "asm_name": "asm1",
         "organism_name": "Pseudomonas aeruginosa", "assembly_level": "Complete Genome",
         "ftp_path": "ftp://example/GCF_000001.1_asm1"},
    ])
    result = select_priority(df, priority_genera=["Pseudomonas", "Myxococcus"])
    assert len(result) == 1  # Myxococcus absent, no crash
    captured = capsys.readouterr()
    assert "Myxococcus" in captured.out


def test_select_diversity_caps_genus():
    """Diversity fill caps each genus at max_per_genus."""
    rows = [
        {"accession": f"GCF_0000{i:02d}.1", "asm_name": f"asm{i}",
         "organism_name": "Vibrio harveyi", "assembly_level": "Complete Genome",
         "ftp_path": f"ftp://example/GCF_0000{i:02d}.1_asm{i}"}
        for i in range(10)
    ]
    df = _make_asm_df(rows)
    result = select_diversity(df, exclude_genera=set(), exclude_full_ids=set(),
                              target_n=10, max_per_genus=3)
    assert len(result) == 3
    assert all(result["genus"] == "Vibrio")


def test_select_diversity_excludes_priority_genera():
    """Diversity fill skips genera in exclude_genera."""
    rows = [
        {"accession": "GCF_000001.1", "asm_name": "asm1",
         "organism_name": "Pseudomonas aeruginosa", "assembly_level": "Complete Genome",
         "ftp_path": "ftp://example/GCF_000001.1_asm1"},
        {"accession": "GCF_000002.1", "asm_name": "asm2",
         "organism_name": "Vibrio harveyi", "assembly_level": "Complete Genome",
         "ftp_path": "ftp://example/GCF_000002.1_asm2"},
    ]
    df = _make_asm_df(rows)
    result = select_diversity(df, exclude_genera={"Pseudomonas"},
                              exclude_full_ids=set(), target_n=10, max_per_genus=5)
    assert len(result) == 1
    assert result.iloc[0]["genus"] == "Vibrio"


def test_select_diversity_excludes_full_ids():
    """Diversity fill skips accessions already in exclude_full_ids."""
    rows = [
        {"accession": "GCF_000001.1", "asm_name": "asm1",
         "organism_name": "Vibrio harveyi", "assembly_level": "Complete Genome",
         "ftp_path": "ftp://example/GCF_000001.1_asm1"},
        {"accession": "GCF_000002.1", "asm_name": "asm2",
         "organism_name": "Vibrio cholerae", "assembly_level": "Complete Genome",
         "ftp_path": "ftp://example/GCF_000002.1_asm2"},
    ]
    df = _make_asm_df(rows)
    result = select_diversity(df, exclude_genera=set(),
                              exclude_full_ids={"GCF_000001.1_asm1"},
                              target_n=10, max_per_genus=5)
    assert len(result) == 1
    assert result.iloc[0]["accession"] == "GCF_000002.1"


def test_select_diversity_returns_empty_for_zero_target():
    """select_diversity with target_n=0 returns empty DataFrame without error."""
    rows = [
        {"accession": "GCF_000001.1", "asm_name": "asm1",
         "organism_name": "Vibrio harveyi", "assembly_level": "Complete Genome",
         "ftp_path": "ftp://example/GCF_000001.1_asm1"},
    ]
    df = _make_asm_df(rows)
    result = select_diversity(df, exclude_genera=set(), exclude_full_ids=set(),
                              target_n=0, max_per_genus=5)
    assert len(result) == 0


def test_select_seed_returns_recommended_genomes():
    """select_seed returns rows whose accession is in RECOMMENDED_GENOMES."""
    real_acc = list(RECOMMENDED_GENOMES.values())[0]
    rows = [
        {"accession": real_acc, "asm_name": "asm1",
         "organism_name": "Escherichia coli", "assembly_level": "Complete Genome",
         "ftp_path": f"ftp://example/{real_acc}_asm1"},
        {"accession": "GCF_NOTREAL.1", "asm_name": "asm2",
         "organism_name": "Fakebacterium sp.", "assembly_level": "Complete Genome",
         "ftp_path": "ftp://example/GCF_NOTREAL.1_asm2"},
    ]
    df = _make_asm_df(rows)
    result = select_seed(df)
    assert len(result) == 1
    assert result.iloc[0]["accession"] == real_acc
