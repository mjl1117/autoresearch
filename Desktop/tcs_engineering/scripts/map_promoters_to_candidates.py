#!/usr/bin/env python3
"""Map characterized output promoters to chimera candidates by DBD family.

Called by identify_chimera_targets.py after candidate scoring to enrich
the output with promoter recommendations for gene circuit design.
Only recommended=True promoters appear in the recommended_promoters column.
CheY is hard-excluded: no DNA-binding domain.
"""
import pandas as pd

CHEY_NOTE = (
    "HARD EXCLUDE: CheY has no DNA-binding domain — acts on flagellar motor "
    "switch FliM; cannot drive promoter activation in any gene circuit."
)
CHEY_FAMILIES = {"CheY_standalone", "CheY"}


def map_promoters(candidates: pd.DataFrame, promoters: pd.DataFrame) -> pd.DataFrame:
    """Add promoter columns to candidates DataFrame.

    Args:
        candidates: chimera_candidates DataFrame with dbd_family column.
        promoters:  characterized_promoters DataFrame.

    Returns:
        candidates with additional columns:
            has_characterized_promoter: bool
            recommended_promoters:      comma-separated promoter names
            promoter_signals:           comma-separated inducing signals
            sigma_factors:              comma-separated sigma factors
            aerobic_compatible:         bool (True if any recommended promoter is aerobic-compatible)
            promoter_caveats:           semicolon-separated caveat strings
    """
    out = candidates.copy()
    out["has_characterized_promoter"] = False
    out["recommended_promoters"] = ""
    out["promoter_signals"] = ""
    out["sigma_factors"] = ""
    out["aerobic_compatible"] = False
    out["promoter_caveats"] = ""

    # Index only recommended promoters by dbd_family
    recommended = promoters[promoters["recommended"].astype(str).str.lower() == "true"]
    by_family = recommended.groupby("dbd_family")

    for i, row in candidates.iterrows():
        family = str(row.get("dbd_family", ""))

        # Hard-exclude CheY
        if family in CHEY_FAMILIES:
            out.at[i, "promoter_caveats"] = CHEY_NOTE
            continue

        if family not in by_family.groups:
            out.at[i, "promoter_caveats"] = (
                f"No characterized output promoter for DBD family '{family}'"
            )
            continue

        rows = by_family.get_group(family)
        out.at[i, "has_characterized_promoter"] = True
        out.at[i, "recommended_promoters"] = ",".join(rows["promoter_name"])
        out.at[i, "promoter_signals"]      = ",".join(rows["inducing_signal"])
        out.at[i, "sigma_factors"]         = ",".join(rows["sigma_factor"].unique())
        out.at[i, "aerobic_compatible"]    = bool(
            rows["aerobic_compatible"].astype(str).str.lower().eq("true").any()
        )
        caveats = rows["caveats"].dropna().unique()
        out.at[i, "promoter_caveats"]      = ";".join(c for c in caveats if c)

    return out


def load_and_map(candidates_tsv: str, promoters_tsv: str) -> pd.DataFrame:
    """Convenience wrapper for CLI use."""
    candidates = pd.read_csv(candidates_tsv, sep="\t")
    promoters  = pd.read_csv(promoters_tsv,  sep="\t")
    return map_promoters(candidates, promoters)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates",  required=True)
    parser.add_argument("--promoters",   required=True)
    parser.add_argument("--output",      required=True)
    args = parser.parse_args()
    result = load_and_map(args.candidates, args.promoters)
    result.to_csv(args.output, sep="\t", index=False)
    n_with = result["has_characterized_promoter"].sum()
    print(f"Promoter mapping complete: {n_with}/{len(result)} candidates have characterized promoters")
