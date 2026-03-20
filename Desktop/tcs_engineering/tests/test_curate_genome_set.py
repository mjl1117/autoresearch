"""Tests for curate_genome_set.py curation logic."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from curate_genome_set import curate, match_to_assembly
from tcs_constants import ASSEMBLY_LEVEL_PRIORITY, RECOMMENDED_GENOMES


def _make_matched(rows: list[dict]) -> pd.DataFrame:
    """Build a matched DataFrame with all columns curate() expects."""
    df = pd.DataFrame(rows)
    df["full_id"] = df["accession"] + "_" + df["asm_name"]
    df["genus"] = df["organism_name"].str.split().str[0]
    df["level_rank"] = df["assembly_level"].map(ASSEMBLY_LEVEL_PRIORITY).fillna(99)
    df["is_protected"] = df["accession"].isin(set(RECOMMENDED_GENOMES.values()))
    return df


def test_curate_caps_genus():
    """Genera with more than max_per_genus genomes are capped."""
    rows = [
        {
            "accession": f"GCF_00000{i}.1",
            "organism_name": "Pseudomonas aeruginosa",
            "assembly_level": "Complete Genome",
            "asm_name": f"asm{i}",
            "taxid": "287",
        }
        for i in range(10)
    ]
    matched = _make_matched(rows)
    curated, counts = curate(matched, max_per_genus=3)
    assert len(curated) == 3
    assert counts["Pseudomonas"]["kept"] == 3
    assert counts["Pseudomonas"]["dropped"] == 7


def test_curate_always_keeps_recommended():
    """RECOMMENDED_GENOMES accessions are never dropped regardless of cap."""
    protected_acc = "GCF_000005845.2"  # E. coli K-12 MG1655
    protected_asm = "ASM584v2"
    rows = [
        {
            "accession": protected_acc,
            "organism_name": "Escherichia coli K-12",
            "assembly_level": "Complete Genome",
            "asm_name": protected_asm,
            "taxid": "511145",
        },
    ] + [
        {
            "accession": f"GCF_99990{i}.1",
            "organism_name": "Escherichia coli",
            "assembly_level": "Scaffold",
            "asm_name": f"asm{i}",
            "taxid": "562",
        }
        for i in range(1, 8)
    ]
    matched = _make_matched(rows)
    curated, _ = curate(matched, max_per_genus=1)
    # Protected genome must always be in curated list
    assert f"{protected_acc}_{protected_asm}" in curated


def test_curate_prefers_complete_genome():
    """Within a capped genus, Complete Genome is preferred over Scaffold."""
    rows = [
        {
            "accession": "GCF_000001.1",
            "organism_name": "Campylobacter jejuni",
            "assembly_level": "Scaffold",
            "asm_name": "asm1",
            "taxid": "197",
        },
        {
            "accession": "GCF_000002.1",
            "organism_name": "Campylobacter jejuni",
            "assembly_level": "Complete Genome",
            "asm_name": "asm2",
            "taxid": "197",
        },
        {
            "accession": "GCF_000003.1",
            "organism_name": "Campylobacter jejuni",
            "assembly_level": "Scaffold",
            "asm_name": "asm3",
            "taxid": "197",
        },
    ]
    matched = _make_matched(rows)
    curated, _ = curate(matched, max_per_genus=1)
    assert curated == ["GCF_000002.1_asm2"]


def test_curate_deterministic():
    """Same input always produces same output (stable sort)."""
    rows = [
        {
            "accession": f"GCF_00000{i}.1",
            "organism_name": "Bacillus cereus",
            "assembly_level": "Scaffold",
            "asm_name": f"asm{i}",
            "taxid": "1396",
        }
        for i in range(5)
    ]
    matched = _make_matched(rows)
    c1, _ = curate(matched, max_per_genus=2)
    c2, _ = curate(matched, max_per_genus=2)
    assert c1 == c2


def test_curate_multi_genus():
    """Cap applies independently per genus."""
    rows = (
        [
            {
                "accession": f"GCF_1000{i}.1",
                "organism_name": "Pseudomonas aeruginosa",
                "assembly_level": "Scaffold",
                "asm_name": f"pa{i}",
                "taxid": "287",
            }
            for i in range(6)
        ]
        + [
            {
                "accession": f"GCF_2000{i}.1",
                "organism_name": "Bacillus subtilis",
                "assembly_level": "Complete Genome",
                "asm_name": f"bs{i}",
                "taxid": "1423",
            }
            for i in range(2)
        ]
    )
    matched = _make_matched(rows)
    curated, counts = curate(matched, max_per_genus=3)
    assert counts["Pseudomonas"]["kept"] == 3
    assert counts["Bacillus"]["kept"] == 2  # only 2 available, both kept


def test_match_to_assembly_finds_full_id():
    """Matches downloaded full_id (accession_asmname) folder names."""
    asm = pd.DataFrame(
        [
            {
                "accession": "GCF_000005845.2",
                "organism_name": "Escherichia coli",
                "assembly_level": "Complete Genome",
                "asm_name": "ASM584v2",
                "taxid": "562",
            }
        ]
    )
    asm["full_id"] = asm["accession"] + "_" + asm["asm_name"]
    asm["genus"] = asm["organism_name"].str.split().str[0]
    asm["level_rank"] = asm["assembly_level"].map(ASSEMBLY_LEVEL_PRIORITY).fillna(99)

    downloaded = {"GCF_000005845.2_ASM584v2"}
    result = match_to_assembly(downloaded, asm)
    assert len(result) == 1
    assert result.iloc[0]["full_id"] == "GCF_000005845.2_ASM584v2"
