"""Tests for validate_ground_truth.py check functions."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from validate_ground_truth import (
    check_annotation_coverage,
    check_chimera_coverage,
    check_classifier_specificity,
    check_phase_coherence,
)

# Minimal reference table: NarXL and PhoRB are user-confirmed working systems
REF = pd.DataFrame([
    {
        "system_name": "NarXL",
        "working_in_user_system": "yes",
        "hk_swiss_prot_keywords": "narx",
        "rr_swiss_prot_keywords": "narl",
    },
    {
        "system_name": "PhoRB",
        "working_in_user_system": "yes",
        "hk_swiss_prot_keywords": "phor",
        "rr_swiss_prot_keywords": "phob",
    },
    {
        "system_name": "EnvZOmpR",
        "working_in_user_system": "no",
        "hk_swiss_prot_keywords": "envz",
        "rr_swiss_prot_keywords": "ompr",
    },
])


def _make_ann(titles: list) -> pd.DataFrame:
    df = pd.DataFrame({
        "qseqid": [f"p{i}" for i in range(len(titles))],
        "sseqid": [f"sp|X{i}|Y" for i in range(len(titles))],
        "pident": [80.0] * len(titles),
        "length": [300] * len(titles),
        "qcovhsp": [90.0] * len(titles),
        "evalue": [1e-10] * len(titles),
        "bitscore": [300.0] * len(titles),
        "stitle": titles,
    })
    df["stitle_lower"] = df["stitle"].str.lower()
    return df


# ─── GT-1 / GT-2 ─────────────────────────────────────────────────────────────

def test_annotation_coverage_pass():
    ann = _make_ann(["Nitrate sensor NarX ECOLI", "Phosphate sensor PhoR ECOLI"])
    result = check_annotation_coverage(REF, ann, "HK", "GT-1")
    assert result["status"] == "PASS"
    assert result["n_pass"] == 2
    assert result["n_fail"] == 0


def test_annotation_coverage_warn_missing():
    ann = _make_ann(["Nitrate sensor NarX ECOLI"])  # PhoR missing
    result = check_annotation_coverage(REF, ann, "HK", "GT-1")
    assert result["status"] == "WARN"
    assert result["n_fail"] == 1
    assert "PhoRB" in result["message"]


def test_annotation_coverage_none():
    result = check_annotation_coverage(REF, None, "HK", "GT-1")
    assert result["status"] == "WARN"
    assert "not yet available" in result["message"]


def test_annotation_coverage_non_working_not_required():
    """EnvZOmpR (working_in_user_system=no) is not checked."""
    ann = _make_ann(["Nitrate sensor NarX ECOLI", "Phosphate sensor PhoR ECOLI"])
    result = check_annotation_coverage(REF, ann, "HK", "GT-1")
    assert result["status"] == "PASS"
    assert result["n_pass"] == 2  # only 2 working systems checked


# ─── GT-3 ────────────────────────────────────────────────────────────────────

def test_chimera_coverage_pass():
    cands = pd.DataFrame({
        "known_tcs_system": ["NarXL", "PhoRB", None],
        "linker_phase_compatible": [True, True, None],
        "linker_validation_required": [False, False, None],
    })
    result = check_chimera_coverage(REF, cands)
    assert result["status"] == "PASS"
    assert result["n_pass"] == 2


def test_chimera_coverage_missing_column():
    """FAIL when known_tcs_system column absent."""
    cands = pd.DataFrame({"protein_id": ["p1"]})
    result = check_chimera_coverage(REF, cands)
    assert result["status"] == "FAIL"


def test_chimera_coverage_partial_miss():
    """WARN when only one working system present."""
    cands = pd.DataFrame({
        "known_tcs_system": ["NarXL"],
        "linker_phase_compatible": [True],
        "linker_validation_required": [False],
    })
    result = check_chimera_coverage(REF, cands)
    assert result["status"] == "WARN"
    assert "PhoRB" in result["message"]


# ─── GT-4 ────────────────────────────────────────────────────────────────────

def test_classifier_specificity_pass():
    ann = _make_ann(["Histidine kinase NarX ECOLI", "Histidine kinase PhoR ECOLI"])
    result = check_classifier_specificity(ann)
    assert result["status"] == "PASS"


def test_classifier_specificity_fail_ftsz():
    ann = _make_ann(["Cell division protein FtsZ ECOLI", "Histidine kinase NarX"])
    result = check_classifier_specificity(ann)
    assert result["status"] == "FAIL"
    assert result["n_fail"] == 1


def test_classifier_specificity_fail_htpg():
    ann = _make_ann(["Chaperone protein HtpG ECOLI"])
    result = check_classifier_specificity(ann)
    assert result["status"] == "FAIL"


def test_classifier_specificity_none():
    result = check_classifier_specificity(None)
    assert result["status"] == "WARN"


# ─── GT-5 ────────────────────────────────────────────────────────────────────

def test_phase_coherence_pass():
    cands = pd.DataFrame({
        "known_tcs_system": ["NarXL", "PhoRB"],
        "linker_phase_compatible": [True, False],
        "linker_validation_required": [False, True],
    })
    result = check_phase_coherence(REF, cands)
    assert result["status"] == "PASS"
    assert result["n_pass"] == 2


def test_phase_coherence_warn_both_null():
    cands = pd.DataFrame({
        "known_tcs_system": ["NarXL"],
        "linker_phase_compatible": [None],
        "linker_validation_required": [None],
    })
    result = check_phase_coherence(REF, cands)
    assert result["status"] == "WARN"
    assert result["n_fail"] == 1


def test_phase_coherence_fail_missing_columns():
    cands = pd.DataFrame({"protein_id": ["p1"]})
    result = check_phase_coherence(REF, cands)
    assert result["status"] == "FAIL"
