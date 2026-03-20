#!/usr/bin/env python3
"""Ground truth validation for TCS pipeline outputs.

Checks pipeline results against well_characterized_tcs.tsv.
Each check produces a PASS / WARN / FAIL result with a message.

Five checks:
  GT-1  HK annotation coverage  — working systems detected in HK DIAMOND hits
  GT-2  RR annotation coverage  — working systems detected in RR DIAMOND hits
  GT-3  Chimera candidate coverage — working systems present as known_tcs_system
  GT-4  Classifier specificity  — no FtsZ/HtpG/GyrB/MutL in HK annotation
  GT-5  Phase coherence present — NarXL/PhoRB candidates have phase columns populated

Status semantics:
  PASS — criterion met
  WARN — criterion unmet but not fatal (e.g. organism absent from genome set)
  FAIL — criterion unmet and indicates a pipeline bug; script exits with code 1

Outputs:
  results/validation/ground_truth_validation.tsv
  results/validation/validation_summary.txt
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Known non-HK contaminants that pass the old (buggy) HATPase_c-only classifier.
# If these appear in HK annotation, the classifier has regressed.
KNOWN_CONTAMINANTS = ["ftsz", "htpg", "gyrb", "mutl", "hsph", "dnak", "grpe"]


def load_reference(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def load_annotation(path: str) -> "pd.DataFrame | None":
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=["qseqid", "sseqid", "pident", "length",
               "qcovhsp", "evalue", "bitscore", "stitle"],
    )
    df["stitle_lower"] = df["stitle"].str.lower()
    return df


def load_candidates(path: str) -> "pd.DataFrame | None":
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    return pd.read_csv(path, sep="\t")


def check_annotation_coverage(
    ref: pd.DataFrame,
    ann: "pd.DataFrame | None",
    protein_type: str,
    check_id: str,
) -> dict:
    """GT-1 / GT-2: each working_in_user_system=yes keyword appears in annotation."""
    if ann is None:
        return {
            "check_id": check_id,
            "status": "WARN",
            "message": f"{protein_type} annotation not yet available (pipeline incomplete)",
            "n_pass": 0,
            "n_fail": 0,
        }

    kw_col = "hk_swiss_prot_keywords" if protein_type == "HK" else "rr_swiss_prot_keywords"
    working = ref[ref["working_in_user_system"] == "yes"]

    passed, failed = [], []
    for _, row in working.iterrows():
        keywords = [
            k.strip()
            for k in str(row[kw_col]).lower().split(";")
            if k.strip() not in ("nan", "null", "")
        ]
        if not keywords:
            continue
        hits = ann[ann["stitle_lower"].apply(lambda t: any(k in t for k in keywords))]
        if not hits.empty:
            passed.append(row["system_name"])
        else:
            failed.append(row["system_name"])

    if failed:
        # WARN not FAIL: organism may not be in genome set — not a classifier bug
        status = "WARN"
        msg = (
            f"Not detected in {protein_type} annotation: {', '.join(failed)}. "
            "Add the organism to the genome set and re-run."
        )
    else:
        status = "PASS"
        msg = f"All {len(passed)} working {protein_type} systems detected in annotation"

    return {
        "check_id": check_id,
        "status": status,
        "message": msg,
        "n_pass": len(passed),
        "n_fail": len(failed),
    }


def check_chimera_coverage(
    ref: pd.DataFrame,
    candidates: "pd.DataFrame | None",
) -> dict:
    """GT-3: working systems appear as known_tcs_system in chimera_candidates."""
    if candidates is None:
        return {
            "check_id": "GT-3",
            "status": "WARN",
            "message": "chimera_candidates.tsv not yet available (pipeline incomplete)",
            "n_pass": 0,
            "n_fail": 0,
        }

    if "known_tcs_system" not in candidates.columns:
        return {
            "check_id": "GT-3",
            "status": "FAIL",
            "message": (
                "known_tcs_system column missing from chimera_candidates.tsv — "
                "re-run identify_chimera_targets.py with --reference_tcs flag"
            ),
            "n_pass": 0,
            "n_fail": 1,
        }

    working_names = set(ref[ref["working_in_user_system"] == "yes"]["system_name"])
    found = set(candidates["known_tcs_system"].dropna().unique())
    present = working_names & found
    missing = working_names - found

    if missing:
        status = "WARN"
        msg = (
            f"Working systems absent from chimera candidates: {', '.join(sorted(missing))}. "
            "Check that their organisms are in the genome set."
        )
    else:
        status = "PASS"
        msg = f"All {len(present)} working systems present in chimera candidates"

    return {
        "check_id": "GT-3",
        "status": status,
        "message": msg,
        "n_pass": len(present),
        "n_fail": len(missing),
    }


def check_classifier_specificity(hk_ann: "pd.DataFrame | None") -> dict:
    """GT-4: known non-HK contaminants must not appear in HK annotation top hits."""
    if hk_ann is None:
        return {
            "check_id": "GT-4",
            "status": "WARN",
            "message": "HK annotation not yet available (pipeline incomplete)",
            "n_pass": 0,
            "n_fail": 0,
        }

    contaminated = hk_ann[
        hk_ann["stitle_lower"].apply(
            lambda t: any(c in t for c in KNOWN_CONTAMINANTS)
        )
    ]

    if not contaminated.empty:
        examples = contaminated["stitle"].head(3).tolist()
        return {
            "check_id": "GT-4",
            "status": "FAIL",
            "message": (
                f"Contaminants in HK annotation ({len(contaminated)} hits). "
                f"Examples: {examples}. "
                "Check HK_DOMAINS_REQUIRED in 01_detect_domains.py — "
                "HATPase_c alone is not sufficient."
            ),
            "n_pass": 0,
            "n_fail": len(contaminated),
        }

    return {
        "check_id": "GT-4",
        "status": "PASS",
        "message": f"No known non-HK contaminants found in HK annotation ({len(hk_ann)} entries checked)",
        "n_pass": len(hk_ann),
        "n_fail": 0,
    }


def check_phase_coherence(
    ref: pd.DataFrame,
    candidates: "pd.DataFrame | None",
) -> dict:
    """GT-5: working-system candidates have phase columns populated (not both null)."""
    if candidates is None:
        return {
            "check_id": "GT-5",
            "status": "WARN",
            "message": "chimera_candidates.tsv not yet available (pipeline incomplete)",
            "n_pass": 0,
            "n_fail": 0,
        }

    required_cols = {"linker_phase_compatible", "linker_validation_required", "known_tcs_system"}
    missing_cols = required_cols - set(candidates.columns)
    if missing_cols:
        return {
            "check_id": "GT-5",
            "status": "FAIL",
            "message": f"Missing columns in chimera_candidates.tsv: {missing_cols}",
            "n_pass": 0,
            "n_fail": 1,
        }

    working_names = set(ref[ref["working_in_user_system"] == "yes"]["system_name"])
    working_cands = candidates[candidates["known_tcs_system"].isin(working_names)]

    if working_cands.empty:
        return {
            "check_id": "GT-5",
            "status": "WARN",
            "message": "No working-system candidates found — check GT-3 first",
            "n_pass": 0,
            "n_fail": 0,
        }

    no_phase = working_cands[
        working_cands["linker_phase_compatible"].isna()
        & working_cands["linker_validation_required"].isna()
    ]

    if not no_phase.empty:
        examples = no_phase["known_tcs_system"].head(3).tolist()
        return {
            "check_id": "GT-5",
            "status": "WARN",
            "message": (
                f"{len(no_phase)} working-system candidates have no phase data "
                f"(both linker_phase_compatible and linker_validation_required are null). "
                f"Examples: {examples}. Check that hmmsearch_hk_reps_domtbl ran."
            ),
            "n_pass": len(working_cands) - len(no_phase),
            "n_fail": len(no_phase),
        }

    return {
        "check_id": "GT-5",
        "status": "PASS",
        "message": (
            f"All {len(working_cands)} working-system candidates have phase data "
            "(linker_phase_compatible or linker_validation_required is set)"
        ),
        "n_pass": len(working_cands),
        "n_fail": 0,
    }


def write_outputs(results: list, outdir: Path) -> bool:
    """Write TSV + human-readable summary. Return True if any check FAILed."""
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(outdir / "ground_truth_validation.tsv", sep="\t", index=False)

    lines = [
        "# Ground Truth Validation Report",
        "# Generated by validate_ground_truth.py",
        "",
    ]
    any_fail = False
    for r in results:
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(r["status"], "?")
        lines.append(f"[{r['status']:4s}] {r['check_id']}  {icon}  {r['message']}")
        if r["status"] == "FAIL":
            any_fail = True

    if any_fail:
        lines.append("\nFAIL checks indicate pipeline bugs — see messages above.")
    else:
        lines.append("\nAll checks passed or produced only warnings.")

    summary_path = outdir / "validation_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return any_fail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference_tcs", required=True,
                        help="data/reference/well_characterized_tcs.tsv")
    parser.add_argument("--hk_annotation", required=True,
                        help="results/annotation/hk_annotation.tsv")
    parser.add_argument("--rr_annotation", required=True,
                        help="results/annotation/rr_annotation.tsv")
    parser.add_argument("--chimera_candidates", required=True,
                        help="results/chimera_targets/chimera_candidates.tsv")
    parser.add_argument("--outdir", required=True,
                        help="results/validation/")
    args = parser.parse_args()

    ref = load_reference(args.reference_tcs)
    hk_ann = load_annotation(args.hk_annotation)
    rr_ann = load_annotation(args.rr_annotation)
    candidates = load_candidates(args.chimera_candidates)

    results = [
        check_annotation_coverage(ref, hk_ann, "HK", "GT-1"),
        check_annotation_coverage(ref, rr_ann, "RR", "GT-2"),
        check_chimera_coverage(ref, candidates),
        check_classifier_specificity(hk_ann),
        check_phase_coherence(ref, candidates),
    ]

    any_fail = write_outputs(results, Path(args.outdir))
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
