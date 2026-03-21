import pytest
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from map_promoters_to_candidates import map_promoters, CHEY_NOTE


def _make_promoters():
    return pd.DataFrame([
        {"dbd_family": "OmpR_PhoB", "promoter_name": "PphoA",
         "inducing_signal": "Phosphate starvation", "sigma_factor": "sigma70",
         "cofactors_required": "None", "aerobic_compatible": True,
         "recommended": True, "caveats": "Clean single-input"},
        {"dbd_family": "NarL_FixJ", "promoter_name": "PnarK",
         "inducing_signal": "Nitrate", "sigma_factor": "sigma70",
         "cofactors_required": "Fnr_partial", "aerobic_compatible": True,
         "recommended": True, "caveats": "Some Fnr effect"},
        {"dbd_family": "NarL_FixJ", "promoter_name": "PnarG",
         "inducing_signal": "Nitrate+anaerobic", "sigma_factor": "sigma70",
         "cofactors_required": "Fnr_required;IHF_required",
         "aerobic_compatible": False, "recommended": False,
         "caveats": "Requires Fnr+IHF"},
        {"dbd_family": "NtrC_AAA", "promoter_name": "PglnAp2",
         "inducing_signal": "N-limitation", "sigma_factor": "sigma54",
         "cofactors_required": "None", "aerobic_compatible": True,
         "recommended": True, "caveats": "Requires rpoN (sigma54)"},
        {"dbd_family": "CheY", "promoter_name": "none", "inducing_signal": "none",
         "sigma_factor": "none", "cofactors_required": "none",
         "aerobic_compatible": False, "recommended": False,
         "caveats": "HARD EXCLUDE"},
    ])


def _make_candidates():
    return pd.DataFrame([
        {"protein_id": "WP_001", "dbd_family": "OmpR_PhoB", "chimera_type": "RR_DBD_swap"},
        {"protein_id": "WP_002", "dbd_family": "NarL_FixJ", "chimera_type": "RR_DBD_swap"},
        {"protein_id": "WP_003", "dbd_family": "CheY_standalone", "chimera_type": "RR_DBD_swap"},
        {"protein_id": "WP_004", "dbd_family": "N/A", "chimera_type": "HK_sensor_swap"},
    ])


def test_ompr_phob_gets_pphoa():
    df = map_promoters(_make_candidates(), _make_promoters())
    row = df[df["protein_id"] == "WP_001"].iloc[0]
    assert row["has_characterized_promoter"] == True
    assert "PphoA" in row["recommended_promoters"]
    assert row["sigma_factors"] == "sigma70"


def test_narl_fixj_gets_pnark_only_recommended():
    """Only recommended=True promoters appear in recommended_promoters."""
    df = map_promoters(_make_candidates(), _make_promoters())
    row = df[df["protein_id"] == "WP_002"].iloc[0]
    assert "PnarK" in row["recommended_promoters"]
    assert "PnarG" not in row["recommended_promoters"]


def test_chey_hard_excluded():
    df = map_promoters(_make_candidates(), _make_promoters())
    row = df[df["protein_id"] == "WP_003"].iloc[0]
    assert row["has_characterized_promoter"] == False
    assert CHEY_NOTE in row["promoter_caveats"]


def test_hk_sensor_swap_gets_no_promoter():
    """HK_sensor_swap candidates have N/A dbd_family — no promoter assigned."""
    df = map_promoters(_make_candidates(), _make_promoters())
    row = df[df["protein_id"] == "WP_004"].iloc[0]
    assert row["has_characterized_promoter"] == False


def test_output_has_all_required_columns():
    df = map_promoters(_make_candidates(), _make_promoters())
    for col in ["has_characterized_promoter", "recommended_promoters",
                "promoter_signals", "sigma_factors", "aerobic_compatible",
                "promoter_caveats"]:
        assert col in df.columns, f"Missing column: {col}"
