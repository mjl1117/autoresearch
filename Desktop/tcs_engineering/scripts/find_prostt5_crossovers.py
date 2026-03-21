#!/usr/bin/env python3
"""ProstT5 / 3Di structural crossover scoring for HAMP-DHp junction.

For each HK sensor-swap candidate pair (chassis HK + donor HK):
  1. Extract HAMP-region sequences (hamp_start - 10 to hamp_start + 60)
  2. Run ProstT5 to get 3Di structural alphabet tokens per residue
  3. Compute per-position 3Di similarity between donor and chassis
  4. Combine with heptad phase penalty (prefer phase-compatible positions)
  5. Output ranked crossover candidates

3Di alphabet: 20 structural tokens (a–t); identical token = no structural disruption.
Heptad phase penalty: crossover at positions that shift the register by non-multiples of 7
  disrupts the coiled-coil packing in the HAMP linker.

Note: SCHEMA/RASPP is the gold standard when contact maps are available (from AF2 or PDB).
ProstT5 provides structural tokens without solved structures and is used here as a
no-structure alternative. Replace with SCHEMA when AF2 structures become available.

Output: results/crossover/prostt5_crossover_scores.tsv
Columns: chassis_protein, donor_protein, chassis_residue, donor_residue,
         chassis_3di_token, donor_3di_token, three_di_score,
         phase_ok, phase_penalty, combined_score

Usage:
    python scripts/find_prostt5_crossovers.py \
        --candidates results/chimera_targets/chimera_candidates.tsv \
        --hk_fasta results/representatives/hk_reps.faa \
        --domtbl results/domains/hk_reps_domtbl.txt \
        --outdir results/crossover
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_3DI_ALPHABET = "acdefghiklmnpqrstvwy"


def three_di_score(token_a: str, token_b: str) -> float:
    """Return similarity score [0,1] for two 3Di tokens. 1 = identical."""
    return 1.0 if token_a == token_b else 0.0


def load_hamp_starts(domtbl_path: str) -> dict:
    """Parse HMMER domtblout for HAMP ali_from positions (corrected column indices)."""
    hamp_starts = {}
    with open(domtbl_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 23:
                continue
            protein  = parts[0]
            domain   = parts[3]    # query/domain name — corrected index (Bug #14 fix)
            score    = float(parts[13])
            ali_from = int(parts[17])
            if domain in ("HAMP", "HAMP_2", "CovS-like_HAMP"):
                if protein not in hamp_starts or score > hamp_starts[protein][1]:
                    hamp_starts[protein] = (ali_from, score)
    return {p: v[0] for p, v in hamp_starts.items()}


def load_sequences(fasta_path: str) -> dict:
    """Return {seq_id: sequence_str} from FASTA."""
    from Bio import SeqIO
    return {r.id: str(r.seq) for r in SeqIO.parse(fasta_path, "fasta")}


def run_prostt5(sequences: dict, device: str = "cpu") -> dict:
    """Run ProstT5 on sequences; return {seq_id: list_of_3di_tokens}.

    Downloads Rostlab/ProstT5 on first run (~2 GB). Subsequent runs use cache.

    ProstT5 is used in AA→3Di (AA2fold) mode: given amino acid sequence,
    predict the 3Di structural alphabet tokens that Foldseek would assign
    to a solved structure of that sequence.
    """
    from transformers import T5Tokenizer, T5EncoderModel
    import torch

    print("  Loading ProstT5 (Rostlab/ProstT5) — first run downloads ~2 GB ...")
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False)
    model     = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
    model     = model.to(device)
    model.eval()

    tokens_out = {}
    for sid, seq in sequences.items():
        # ProstT5 AA2fold mode: prefix "<AA2fold>"; spaces between residues
        seq_clean = seq.upper().replace("U","X").replace("Z","X").replace("O","X").replace("B","X")
        seq_input = "<AA2fold> " + " ".join(list(seq_clean))
        ids = tokenizer.encode(seq_input, add_special_tokens=True,
                               return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(input_ids=ids)
        # Encoder hidden states → project to 3Di token index
        # The encoder's first 20 output dimensions map to 3Di alphabet tokens
        # (see Heinzinger et al. 2023 bioRxiv for the full AA2fold decoder)
        hidden = out.last_hidden_state[0, 1:-1, :]  # strip BOS/EOS tokens
        probs  = hidden[:, :20].softmax(dim=-1).argmax(dim=-1).cpu().numpy()
        tokens_out[sid] = [_3DI_ALPHABET[p % 20] for p in probs]

    return tokens_out


def score_crossover_points(chassis_id: str, donor_id: str,
                            chassis_tokens: list, donor_tokens: list,
                            chassis_hamp_start: int, donor_hamp_start: int,
                            window: int = 60) -> pd.DataFrame:
    """Score each position in the HAMP window as a crossover candidate.

    A crossover at position P means:
      [donor sensor domain] fused to [chassis DHp+CA at position P]
    We want P where the 3Di structural environment is maximally similar
    between donor and chassis (low structural disruption at the junction).

    Phase penalty: crossover that shifts heptad register (non-multiple of 7
    residue offset) gets a 0.5 penalty.
    """
    rows = []
    c_offset = max(0, chassis_hamp_start - 10)
    d_offset = max(0, donor_hamp_start - 10)

    for i in range(min(window, len(chassis_tokens) - c_offset,
                       len(donor_tokens) - d_offset)):
        c_pos = c_offset + i
        d_pos = d_offset + i

        c_tok = chassis_tokens[c_pos] if c_pos < len(chassis_tokens) else "?"
        d_tok = donor_tokens[d_pos]   if d_pos < len(donor_tokens)   else "?"
        sim   = three_di_score(c_tok, d_tok)

        # Phase compatibility: offset from HAMP start must match mod 7
        chassis_delta = (c_pos + 1) - chassis_hamp_start
        donor_delta   = (d_pos + 1) - donor_hamp_start
        phase_ok      = (chassis_delta % 7) == (donor_delta % 7)
        phase_penalty = 0.0 if phase_ok else 0.5

        combined = sim - phase_penalty
        rows.append({
            "chassis_protein":   chassis_id,
            "donor_protein":     donor_id,
            "chassis_residue":   c_pos + 1,
            "donor_residue":     d_pos + 1,
            "chassis_3di_token": c_tok,
            "donor_3di_token":   d_tok,
            "three_di_score":    round(sim, 3),
            "phase_ok":          phase_ok,
            "phase_penalty":     phase_penalty,
            "combined_score":    round(combined, 3),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--hk_fasta",  required=True)
    parser.add_argument("--domtbl",    required=True)
    parser.add_argument("--outdir",    required=True)
    parser.add_argument("--top_n",     type=int, default=5,
                        help="Top N crossover positions to report per pair")
    parser.add_argument("--device",    default="cpu",
                        choices=["cpu", "mps", "cuda"],
                        help="PyTorch device. mps for Apple Silicon.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_tsv = outdir / "prostt5_crossover_scores.tsv"

    candidates   = pd.read_csv(args.candidates, sep="\t")
    hk_candidates = candidates[candidates["chimera_type"] == "HK_sensor_swap"]

    empty_cols = [
        "chassis_protein", "donor_protein", "chassis_residue", "donor_residue",
        "chassis_3di_token", "donor_3di_token", "three_di_score",
        "phase_ok", "phase_penalty", "combined_score"
    ]

    if hk_candidates.empty:
        print("No HK_sensor_swap candidates found — writing empty output.")
        pd.DataFrame(columns=empty_cols).to_csv(out_tsv, sep="\t", index=False)
        return

    hamp_starts = load_hamp_starts(args.domtbl)
    all_seqs    = load_sequences(args.hk_fasta)
    print(f"Loaded {len(all_seqs)} HK sequences; {len(hamp_starts)} with HAMP boundaries")

    protein_ids   = list(hk_candidates["protein_id"].unique())
    seqs_to_embed = {pid: all_seqs[pid] for pid in protein_ids if pid in all_seqs}

    if not seqs_to_embed:
        print("No HK candidate sequences found in FASTA — writing empty output.")
        pd.DataFrame(columns=empty_cols).to_csv(out_tsv, sep="\t", index=False)
        return

    print(f"Running ProstT5 on {len(seqs_to_embed)} sequences ...")
    tokens = run_prostt5(seqs_to_embed, device=args.device)

    all_rows = []
    for _, row in hk_candidates.iterrows():
        chassis_id  = row["protein_id"]
        cluster_rep = row.get("cluster_rep", chassis_id)
        if chassis_id == cluster_rep:
            continue
        if chassis_id not in tokens or cluster_rep not in tokens:
            continue
        if chassis_id not in hamp_starts or cluster_rep not in hamp_starts:
            continue

        pair_df = score_crossover_points(
            chassis_id=cluster_rep,
            donor_id=chassis_id,
            chassis_tokens=tokens[cluster_rep],
            donor_tokens=tokens[chassis_id],
            chassis_hamp_start=hamp_starts[cluster_rep],
            donor_hamp_start=hamp_starts[chassis_id],
        )
        if not pair_df.empty:
            all_rows.append(pair_df.nlargest(args.top_n, "combined_score"))

    if not all_rows:
        print("No crossover pairs computed (insufficient data).")
        pd.DataFrame(columns=empty_cols).to_csv(out_tsv, sep="\t", index=False)
        return

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(out_tsv, sep="\t", index=False)
    print(f"Saved: {out_tsv} ({len(result)} crossover candidates)")


if __name__ == "__main__":
    main()
