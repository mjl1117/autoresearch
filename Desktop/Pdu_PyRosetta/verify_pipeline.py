#!/usr/bin/env python3
"""verify_pipeline.py — End-to-end checks for mutant_prediction_pipeline outputs.

Run after executing mutant_prediction_pipeline.ipynb end-to-end.
Usage: conda run -n pyrosetta_env python3 verify_pipeline.py
"""
import os, sys
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

os.chdir("/Users/matthew/Desktop/Pdu_PyRosetta")
errors = []
passed = 0

def check(name, condition, detail=""):
    global passed
    if condition:
        print(f"  ✓ {name}")
        passed += 1
    else:
        print(f"  ✗ FAIL: {name}" + (f" — {detail}" if detail else ""))
        errors.append(name)

print("=" * 60)
print("VERIFICATION: mutant_prediction_pipeline outputs")
print("=" * 60)

# ── 1. LOO-CV results ────────────────────────────────────────
print("\n[1] LOO-CV results")
if Path("PduA_ThermoMPNN_results/binding_head_loo_cv.csv").exists():
    loo = pd.read_csv("PduA_ThermoMPNN_results/binding_head_loo_cv.csv")
    check("LOO-CV has 76 rows", len(loo) == 76, f"got {len(loo)}")
    check("LOO-CV has required columns",
          {"position_key","mutant","ddg_true","ddg_pred"}.issubset(loo.columns))
    overall_rho = spearmanr(loo["ddg_true"], loo["ddg_pred"]).statistic
    check("LOO-CV overall Spearman rho > 0.3", overall_rho > 0.3,
          f"got {overall_rho:.3f}")
    for pk, grp in loo.groupby("position_key"):
        rho = spearmanr(grp["ddg_true"], grp["ddg_pred"]).statistic
        print(f"    {pk}: rho = {rho:.3f}  (n={len(grp)})")
else:
    errors.append("LOO-CV file missing")
    print("  ✗ SKIP: binding_head_loo_cv.csv not found")

# ── 2. Single mutant rankings ─────────────────────────────────
print("\n[2] Single mutant rankings")
for pos_key in ["PduA_K37", "PduA_S40", "PduJ_K36", "PduJ_S39"]:
    f = f"mutant_candidates/{pos_key}_single_mutant_ranking.csv"
    if Path(f).exists():
        df = pd.read_csv(f)
        check(f"{pos_key}: 19 variants", len(df) == 19, f"got {len(df)}")
        check(f"{pos_key}: has composite_score column", "composite_score" in df.columns)
    else:
        errors.append(f"Missing {f}")
        print(f"  ✗ SKIP: {f} not found")

# Validated variants present in PduA S40
s40_f = "mutant_candidates/PduA_S40_single_mutant_ranking.csv"
if Path(s40_f).exists():
    s40 = pd.read_csv(s40_f)
    validated = {"S40A", "S40H", "S40Q", "S40L"}
    found = set(s40["mutant"]) & validated
    check("PduA S40: all 4 validated variants present",
          found == validated, f"missing: {validated - found}")

# Top K37 candidate should be K37F, K37I, or K37V (known best binders)
k37_f = "mutant_candidates/PduA_K37_single_mutant_ranking.csv"
if Path(k37_f).exists():
    k37 = pd.read_csv(k37_f)
    top1 = k37.iloc[0]["mutant"]
    check("PduA K37: top candidate is K37F, K37I, or K37V",
          top1 in {"K37F", "K37I", "K37V"}, f"got {top1}")

# ── 3. Double mutant predictions ──────────────────────────────
print("\n[3] Double mutant predictions")
for protein in ["PduA", "PduJ"]:
    full_f = f"mutant_candidates/{protein}_double_mutant_predictions.csv"
    top20_f = f"mutant_candidates/{protein}_double_mutant_top20.csv"
    if Path(full_f).exists():
        df = pd.read_csv(full_f)
        check(f"{protein} full double mutants: 361 rows", len(df) == 361, f"got {len(df)}")
        check(f"{protein} has ddg_epistatic_score column", "ddg_epistatic_score" in df.columns)
    else:
        errors.append(f"Missing {full_f}")
    if Path(top20_f).exists():
        top20 = pd.read_csv(top20_f)
        check(f"{protein} top20: ≤20 rows", len(top20) <= 20, f"got {len(top20)}")
        if len(top20) > 0:
            check(f"{protein} top20: all pass filters",
                  bool(top20["passes_filters"].all()))
    else:
        errors.append(f"Missing {top20_f}")

# ── 4. Selectivity predictions ────────────────────────────────
print("\n[4] Selectivity predictions")
for protein in ["PduA", "PduJ"]:
    for pos in ["K37", "S40", "K36", "S39"]:
        # Only check positions that belong to each protein
        if (protein == "PduA" and pos in ["K36","S39"]) or \
           (protein == "PduJ" and pos in ["K37","S40"]):
            continue
        for ligand in ["propionaldehyde", "propionate"]:
            f = f"mutant_candidates/{protein}_{pos}_{ligand}_selectivity.csv"
            if Path(f).exists():
                df = pd.read_csv(f)
                check(f"{protein} {pos} {ligand}: has selectivity index",
                      f"{'PA' if ligand=='propionaldehyde' else 'PR'}_selectivity_index" in df.columns)
            else:
                errors.append(f"Missing {f}")
                print(f"  ✗ SKIP: {f} not found")

# Propionate direction check
pr_f = "mutant_candidates/PduA_K37_propionate_selectivity.csv"
if Path(pr_f).exists():
    pr = pd.read_csv(pr_f)
    if "K37R" in pr["mutant"].values and "K37A" in pr["mutant"].values:
        k37r_pr = float(pr[pr["mutant"]=="K37R"]["PR_selectivity_index"].iloc[0])
        k37a_pr = float(pr[pr["mutant"]=="K37A"]["PR_selectivity_index"].iloc[0])
        check("K37A has higher PR index than K37R (K37A allows more propionate)",
              k37a_pr > k37r_pr, f"K37A={k37a_pr:.2f}, K37R={k37r_pr:.2f}")

# ── 5. Final summary ──────────────────────────────────────────
print("\n[5] Final summary")
summary_f = "mutant_candidates/final_summary.csv"
if Path(summary_f).exists():
    summary = pd.read_csv(summary_f)
    check("Summary has 10 rows (5 categories × 2 proteins)", len(summary) == 10, f"got {len(summary)}")
    check("Summary has Category, Protein, Top candidates columns",
          {"Category","Protein","Top candidates"}.issubset(summary.columns))
else:
    errors.append(f"Missing {summary_f}")

# ── 6. CSV count ──────────────────────────────────────────────
print("\n[6] Output file count")
csv_files = list(Path("mutant_candidates").glob("*.csv"))
check("17 CSV files in mutant_candidates/", len(csv_files) == 17, f"got {len(csv_files)}")

# ── Result ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  {passed} checks passed, {len(errors)} failed")
if errors:
    print(f"  FAILED: {errors}")
    sys.exit(1)
else:
    print("  ALL CHECKS PASSED ✓")
