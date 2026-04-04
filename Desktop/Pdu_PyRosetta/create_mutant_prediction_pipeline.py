"""Script to create mutant_prediction_pipeline.ipynb using nbformat."""
import nbformat

cell1_source = '''# Cell 1: Imports and config
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import ipywidgets as widgets
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from IPython.display import display

os.chdir("/Users/matthew/Desktop/Pdu_PyRosetta")

# Positions: (protein, PDB_resnum, wt_aa)
POSITIONS = {
    "PduA_K37": ("PduA", 37, "K"),
    "PduA_S40": ("PduA", 40, "S"),
    "PduJ_K36": ("PduJ", 36, "K"),
    "PduJ_S39": ("PduJ", 39, "S"),
}

# H-bond donor amino acids
HBOND_DONORS = set("STNQHKRYW")

# Charge at physiological pH
AA_CHARGE = {
    "K": +1, "R": +1, "H": +0.1,
    "D": -1, "E": -1,
    "A": 0, "C": 0, "F": 0, "G": 0,
    "I": 0, "L": 0, "M": 0, "N": 0,
    "P": 0, "Q": 0, "S": 0, "T": 0,
    "V": 0, "W": 0, "Y": 0,
}

# Variants already tested experimentally by the user
ALREADY_VALIDATED = {"PduA_S40": {"S40A", "S40H", "S40Q", "S40L"}}

print("✓ Imports and config loaded")
'''

cell2_source = '''# Cell 2: Load all data sources
print("Loading docking summaries...")
docking = {}
DOCKING_PATHS = {
    "PduA_K37": "PduA_docking_mutants/K37_site_saturation_summary.csv",
    "PduA_S40": "PduA_docking_mutants/S40_site_saturation_summary.csv",
    "PduJ_K36": "PduJ_docking_mutants/K36_site_saturation_summary.csv",
    "PduJ_S39": "PduJ_docking_mutants/S39_site_saturation_summary.csv",
}
for pos_key, csv_path in DOCKING_PATHS.items():
    df = pd.read_csv(csv_path)
    df["position_key"] = pos_key
    docking[pos_key] = df
    print(f"  {pos_key}: {len(df)} mutants, ddg_mean [{df[\'ddg_mean\'].min():.2f}, {df[\'ddg_mean\'].max():.2f}]")

print("\\nLoading SSM Rosetta \\u0394\\u0394G...")
ssm_a = pd.read_csv("PduA_SSM_results/PduA_SSM_ddg_rigorous.csv")
ssm_a = ssm_a[ssm_a["baseline_type"] == "local_relaxed"]
ssm_j = pd.read_csv("PduJ_SSM_results/PduJ_SSM_ddg_rigorous.csv")
ssm_j = ssm_j[ssm_j["baseline_type"] == "local_relaxed"]
print(f"  PduA SSM: {len(ssm_a)}, PduJ SSM: {len(ssm_j)}")

print("\\nLoading ThermoMPNN stability predictions...")
thermo_a = pd.read_csv("PduA_ThermoMPNN_results/PduA_ThermoMPNN_combined.csv")
thermo_j = pd.read_csv("PduJ_SSM_results/PduJ_ThermoMPNN_zeroshot.csv")
thermo_j = thermo_j.rename(columns={"ddG_thermo": "ddG_ft"})

print("\\nLoading conservation scores...")
cons = pd.read_csv("PduA_conservation_analysis/PduA_conservation_scores.csv")

print("\\nLoading coevolution APC scores...")
coev = pd.read_csv("PduA_coevolution/top_coevolving_pairs.csv")
k37_s40_pair = coev[
    ((coev["res_i"] == 37) & (coev["res_j"] == 40)) |
    ((coev["res_i"] == 40) & (coev["res_j"] == 37))
]
APC_K37_S40 = float(k37_s40_pair["APC_score"].iloc[0]) if len(k37_s40_pair) > 0 else 0.1894
APC_K36_S39 = APC_K37_S40  # PduJ proxy (no coevolution file; same pore sequence)
print(f"  K37-S40 APC = {APC_K37_S40:.4f}  (K36-S39 proxy = {APC_K36_S39:.4f})")

print("\\nLoading BindingHead predictions...")
binding_preds_path = "PduA_ThermoMPNN_results/binding_head_all_predictions.csv"
if os.path.exists(binding_preds_path):
    binding_preds = pd.read_csv(binding_preds_path)
    print(f"  {len(binding_preds)} predictions loaded")
else:
    print("  \\u26a0 binding_head_all_predictions.csv not found yet \\u2014 run BMC_ThermoMPNN_binding.py first")
    binding_preds = pd.DataFrame(columns=["position_key","mut_aa","ddg_binding_predicted"])

print("\\n\\u2713 All data loaded.")
'''

cell3_source = '''# Cell 3: Build unified feature dataframe per position

def build_feature_df(pos_key: str) -> pd.DataFrame:
    protein, pdb_resnum, wt_aa = POSITIONS[pos_key]

    # 1. Docking scores
    doc = docking[pos_key].copy()
    doc["mut_aa"] = doc["mutant"].str[-1]
    doc["wt_aa"]  = wt_aa
    doc = doc.rename(columns={
        "ddg_mean":      "ddg_binding",
        "ddg_best":      "ddg_binding_best",
        "ddg_top10":     "ddg_binding_top10",
        "std_interface": "ddg_binding_std",
    })

    # 2. SSM structural \\u0394\\u0394G
    ssm = ssm_a if protein == "PduA" else ssm_j
    ssm_pos = ssm[ssm["residue_num"] == pdb_resnum][["mut_aa", "ddG_rigorous_REU"]].copy()
    ssm_pos = ssm_pos.rename(columns={"ddG_rigorous_REU": "ddg_structural"})

    # 3. ThermoMPNN stability
    thermo = thermo_a if protein == "PduA" else thermo_j
    thermo_pos = thermo[thermo["residue_num"] == pdb_resnum][["mut_aa", "ddG_ft"]].copy()

    # 4. BindingHead predictions
    bp = binding_preds[binding_preds["position_key"] == pos_key][["mut_aa", "ddg_binding_predicted"]].copy()

    # 5. Conservation: homolog frequency of mut_aa
    cons_pos = cons[cons["residue_num"] == pdb_resnum]
    def get_freq(mut_aa):
        col = f"freq_{mut_aa}"
        if not cons_pos.empty and col in cons_pos.columns:
            return float(cons_pos[col].iloc[0])
        return 0.0

    # Merge
    keep_cols = ["mutant", "mut_aa", "wt_aa", "ddg_binding", "ddg_binding_best",
                 "ddg_binding_top10", "ddg_binding_std", "mean_pore_dist"]
    keep_cols = [c for c in keep_cols if c in doc.columns]
    df = doc[keep_cols].copy()
    df = df.merge(ssm_pos,    on="mut_aa", how="left")
    df = df.merge(thermo_pos, on="mut_aa", how="left")
    df = df.merge(bp,         on="mut_aa", how="left")

    df["homolog_freq"]    = df["mut_aa"].apply(get_freq)
    df["charge_mut"]      = df["mut_aa"].map(AA_CHARGE).fillna(0)
    df["hbond_donor_mut"] = df["mut_aa"].apply(lambda a: 1 if a in HBOND_DONORS else 0)
    df["charge_wt"]       = AA_CHARGE.get(wt_aa, 0)
    df["hbond_donor_wt"]  = 1 if wt_aa in HBOND_DONORS else 0
    df["delta_charge"]    = df["charge_mut"] - df["charge_wt"]
    df["delta_hbond"]     = df["hbond_donor_mut"] - df["hbond_donor_wt"]
    df["position_key"]    = pos_key
    df["protein"]         = protein
    df["pdb_resnum"]      = pdb_resnum

    # Bootstrap 95% CI approx: mean \\u00b1 1.96 * std / sqrt(100)
    df["bootstrap_ci_lo"] = df["ddg_binding"] - 1.96 * df["ddg_binding_std"] / 10
    df["bootstrap_ci_hi"] = df["ddg_binding"] + 1.96 * df["ddg_binding_std"] / 10

    df["already_validated"] = df["mutant"].isin(ALREADY_VALIDATED.get(pos_key, set()))
    return df.reset_index(drop=True)

feature_dfs = {pk: build_feature_df(pk) for pk in POSITIONS}
for pk, df in feature_dfs.items():
    print(f"{pk}: {len(df)} variants, ddg_binding range [{df[\'ddg_binding\'].min():.2f}, {df[\'ddg_binding\'].max():.2f}]")
print("\\u2713 Feature dataframes built")
'''

cell4_source = '''# ============================================================
# Cell 4: Composite Score — USER CONTRIBUTION POINT
# ============================================================
# WHY THIS MATTERS:
# The relative weights encode your scientific prior about what predicts
# experimental success.
#   - High ddg_binding weight → picks strong binders that may be unstable
#   - High ddg_structural weight → picks stable variants that may not change binding
#   - High homolog_freq weight → conservative; prioritizes natural variants
#
# CONSTRAINTS:
#   - All weights must be >= 0 and sum to 1.0
#   - Z-scoring is done within each position (across the 19 variants)
#   - Lower composite score = better candidate
#   - homolog_freq is INVERTED before Z-scoring (higher freq = lower z = better)
# ============================================================

# Adjust these weights to reflect your scientific priorities:
WEIGHTS = {
    "ddg_binding":           0.40,   # direct binding evidence (docking \\u0394\\u0394G)
    "ddg_binding_predicted": 0.20,   # BindingHead model prediction
    "ddg_structural":        0.25,   # apo hexamer stability (SSM Rosetta \\u0394\\u0394G)
    "ddg_ft":                0.10,   # ThermoMPNN stability score
    "homolog_freq":          0.05,   # evolutionary support
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"


def z_normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each feature within the position\'s variant set.
    homolog_freq is inverted so lower z = better for all features.
    """
    df = df.copy()
    for col in ["ddg_binding", "ddg_binding_predicted", "ddg_structural", "ddg_ft"]:
        if col in df.columns:
            mu, sigma = df[col].mean(), df[col].std()
            df[f"z_{col}"] = (df[col] - mu) / (sigma + 1e-9)
    if "homolog_freq" in df.columns:
        mu, sigma = df["homolog_freq"].mean(), df["homolog_freq"].std()
        df["z_homolog_freq"] = -1.0 * (df["homolog_freq"] - mu) / (sigma + 1e-9)
    return df


def compute_composite_score(row: pd.Series, weights: dict) -> float:
    """Weighted composite score. Lower = better candidate.

    Handles missing features by renormalizing available weights.
    Caps Z-scores at \\u00b13 to prevent extreme values from dominating.
    """
    score = 0.0
    available_weight = 0.0
    for feature, w in weights.items():
        z_col = f"z_{feature}"
        if z_col in row.index and not pd.isna(row[z_col]):
            z_val = float(np.clip(row[z_col], -3, 3))
            score += w * z_val
            available_weight += w
    if available_weight > 0:
        score = score / available_weight
    return score


# Verify
test_df = z_normalize_features(feature_dfs["PduA_K37"])
test_score = compute_composite_score(test_df.iloc[0], WEIGHTS)
print(f"Test composite score (PduA_K37 row 0): {test_score:.3f}")
print(f"Z-columns present: {[c for c in test_df.columns if c.startswith(\'z_\')]}")
print("\\u2713 compute_composite_score ready \\u2014 adjust WEIGHTS above to tune rankings")
'''

cell5_source = '''# Cell 5: Single mutant rankings for all 4 positions

def rank_single_mutants(pos_key: str, weights: dict) -> pd.DataFrame:
    df = z_normalize_features(feature_dfs[pos_key])
    df["composite_score"] = df.apply(compute_composite_score, axis=1, weights=weights)
    df = df.sort_values("composite_score").reset_index(drop=True)
    df["rank"] = df.index + 1
    df["top5"]  = df["rank"] <= 5
    return df

all_single_rankings = {}
for pos_key in POSITIONS:
    ranked = rank_single_mutants(pos_key, WEIGHTS)
    all_single_rankings[pos_key] = ranked
    ranked.to_csv(f"mutant_candidates/{pos_key}_single_mutant_ranking.csv", index=False)

    display_cols = ["rank", "mutant", "ddg_binding", "ddg_structural", "ddG_ft",
                    "homolog_freq", "composite_score", "already_validated", "top5"]
    show_cols = [c for c in display_cols if c in ranked.columns]
    print(f"\\n{\'=\'*60}")
    print(f"  {pos_key} \\u2014 Single Mutant Rankings")
    print(f"{\'=\'*60}")
    print(ranked[show_cols].to_string(index=False))

print("\\n\\u2713 Single mutant rankings saved to mutant_candidates/")

# Quick check: validated PduA S40 variants should appear
validated = {"S40A", "S40H", "S40Q", "S40L"}
s40_df = all_single_rankings["PduA_S40"]
found = set(s40_df["mutant"]) & validated
print(f"\\nPduA S40 validated variants found: {found}")
assert found == validated, f"Missing: {validated - found}"
print("\\u2713 All validated variants present in rankings")
'''

nb = nbformat.v4.new_notebook()
nb.cells = [
    nbformat.v4.new_code_cell(source=cell1_source),
    nbformat.v4.new_code_cell(source=cell2_source),
    nbformat.v4.new_code_cell(source=cell3_source),
    nbformat.v4.new_code_cell(source=cell4_source),
    nbformat.v4.new_code_cell(source=cell5_source),
]

output_path = "/Users/matthew/Desktop/Pdu_PyRosetta/mutant_prediction_pipeline.ipynb"
with open(output_path, "w") as f:
    nbformat.write(nb, f)

print(f"Notebook written to: {output_path}")
print(f"Number of cells: {len(nb.cells)}")
