#!/usr/bin/env python3
"""BMC_ThermoMPNN_binding.py

Binding-affinity prediction head for BMC shell proteins.

Extends BMCTransferModel with a second MLP head trained on docking ΔΔG values
from saturation mutagenesis docking experiments (Rosetta/PyRosetta interface
ΔG scores). The stability head (both_out / ddg_out) is frozen after loading
a pre-trained stability checkpoint; only the new binding_out / binding_ddg_out
head is trainable.

Architecture (additions over BMCTransferModel):
    frozen BMCTransferModel embedding (IN_DIM = 512)
        → binding_out  : Sequential MLP → VOCAB_DIM   (mirrors both_out)
        → binding_ddg_out : Linear(1, 1)               (mirrors ddg_out)
        → ΔΔG_binding = binding_ddg_out(profile[mut]) - binding_ddg_out(profile[wt])

Docking summary CSVs (DOCKING_SUMMARIES) record per-mutant interface ΔG values
produced by the site-saturation docking notebooks for PduA K37, PduA S40,
PduJ K36, and PduJ S39.

Usage (future training script):
    from BMC_ThermoMPNN_binding import BMCBindingModel, DOCKING_SUMMARIES
"""

import sys
import copy
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

# ── ThermoMPNN repo ────────────────────────────────────────────────────────────
THERMOMPNN_DIR = Path("/Users/matthew/Desktop/ThermoMPNN")
sys.path.insert(0, str(THERMOMPNN_DIR))
sys.path.insert(0, str(THERMOMPNN_DIR / "analysis"))

from protein_mpnn_utils import tied_featurize
from transfer_model import VOCAB_DIM, ALPHABET

# ── Imports from existing BMC_ThermoMPNN ──────────────────────────────────────
from BMC_ThermoMPNN import (
    BMCTransferModel,
    parse_hexamer_pdb,
    make_cfg,
    Mutation,
    HIDDEN_DIM,
    NUM_FINAL_LAYERS,
    MLP_HIDDEN,
    ALL_CHAINS,
    CHAIN,
    device,
)

# ── Config constants ───────────────────────────────────────────────────────────
PDUA_PDB = "PduA_mutants/WT_3NGK.pdb"
PDUJ_PDB = "PduJ_mutants/WT_PduJ.pdb"

DOCKING_SUMMARIES = {
    "PduA_K37": ("PduA_docking_mutants/K37_site_saturation_summary.csv", "PduA", 37),
    "PduA_S40": ("PduA_docking_mutants/S40_site_saturation_summary.csv", "PduA", 40),
    "PduJ_K36": ("PduJ_docking_mutants/K36_site_saturation_summary.csv", "PduJ", 36),
    "PduJ_S39": ("PduJ_docking_mutants/S39_site_saturation_summary.csv", "PduJ", 39),
}

IN_DIM = HIDDEN_DIM * NUM_FINAL_LAYERS + HIDDEN_DIM  # 512

BINDING_LR = 1e-3
BINDING_EPOCHS = 100
BINDING_CKPT = Path("PduA_ThermoMPNN_results/bmc_binding_best.pt")


# ── BMC Binding Model ──────────────────────────────────────────────────────────

class BMCBindingModel(BMCTransferModel):
    """BMCTransferModel extended with a binding-affinity prediction head.

    The frozen stability head (both_out / ddg_out) provides pre-trained
    hexamer-aware embeddings.  The new binding_out / binding_ddg_out head is
    trained from scratch on docking ΔΔG labels.

    Parameters
    ----------
    cfg : OmegaConf DictConfig
        Same config used to construct BMCTransferModel (passed to super().__init__).
    **kwargs
        Forwarded to BMCTransferModel.__init__.
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Freeze all stability head parameters (both_out and ddg_out)
        for param in self.both_out.parameters():
            param.requires_grad = False
        for param in self.ddg_out.parameters():
            param.requires_grad = False

        # Binding head — mirrors both_out architecture
        layers = []
        prev = IN_DIM
        for h in MLP_HIDDEN:
            layers += [nn.ReLU(), nn.Linear(prev, h)]
            prev = h
        layers += [nn.ReLU(), nn.Linear(prev, VOCAB_DIM)]
        self.binding_out = nn.Sequential(*layers)

        # Binding scalar output — mirrors ddg_out
        self.binding_ddg_out = nn.Linear(1, 1)

    # ------------------------------------------------------------------
    @staticmethod
    def _chain_dicts_to_batch(chain_dicts):
        """Convert parse_hexamer_pdb output (list of per-chain dicts) into the
        single-structure dict format expected by tied_featurize.

        parse_hexamer_pdb returns a list of dicts like:
            {'name': 'prot_A', 'seq': '...', 'coords': {'N': [...], 'CA': [...], ...},
             'chain_id': 'A', 'num_of_chains': 6}

        tied_featurize expects a list containing one dict like:
            {'name': '...', 'seq': '...CONCAT...', 'num_of_chains': N,
             'seq_chain_A': '...', 'coords_chain_A': {'N_chain_A': [...], ...}, ...}
        """
        struct = {
            "name": chain_dicts[0]["name"].replace("prot_", "prot"),
            "num_of_chains": len(chain_dicts),
        }
        concat_seq = ""
        for cd in chain_dicts:
            letter = cd["chain_id"]
            chain_seq = cd["seq"]
            concat_seq += chain_seq
            struct[f"seq_chain_{letter}"] = chain_seq
            c = cd["coords"]
            struct[f"coords_chain_{letter}"] = {
                f"N_chain_{letter}":  c["N"].tolist() if hasattr(c["N"], "tolist") else c["N"],
                f"CA_chain_{letter}": c["CA"].tolist() if hasattr(c["CA"], "tolist") else c["CA"],
                f"C_chain_{letter}":  c["C"].tolist() if hasattr(c["C"], "tolist") else c["C"],
                f"O_chain_{letter}":  c["O"].tolist() if hasattr(c["O"], "tolist") else c["O"],
            }
        struct["seq"] = concat_seq
        return [struct]

    def extract_embedding(self, chain_dicts, pdb_to_seq_idx, pdb_residue_num) -> torch.Tensor:
        """Run frozen ProteinMPNN on the hexamer and return the symmetry-averaged
        embedding for a given PDB residue number.

        Parameters
        ----------
        chain_dicts : list of dict
            Chain dictionaries as returned by parse_hexamer_pdb.
        pdb_to_seq_idx : dict
            Mapping from PDB residue number (int) to 0-based sequential index
            within chain A.
        pdb_residue_num : int
            PDB residue number of the position of interest.

        Returns
        -------
        torch.Tensor, shape [IN_DIM]
            Symmetry-averaged embedding (detached from computation graph).
        """
        seq_pos = pdb_to_seq_idx[pdb_residue_num]

        # Convert from parse_hexamer_pdb's per-chain format to tied_featurize's
        # single-structure batch format
        batch = self._chain_dicts_to_batch(chain_dicts)

        with torch.no_grad():
            X, S, mask, lengths, chain_M, chain_encoding_all, \
                chain_list_list, visible_list_list, masked_list_list, \
                masked_chain_length_list_list, chain_M_pos, omit_AA_mask, \
                residue_idx, dihedral_mask, tied_pos_list_of_lists_list, \
                pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, \
                tied_beta = tied_featurize(
                    batch, device, None, None, None, None, None, None,
                    ca_only=False
                )

            all_mpnn_hid, mpnn_embed, _ = self.prot_mpnn(
                X, S, mask, chain_M, residue_idx, chain_encoding_all, None
            )

        # Concatenate hidden layers — shape [1, total_residues, num_final_layers * 128]
        mpnn_hid = torch.cat(all_mpnn_hid[:self.num_final_layers], dim=-1)

        # Build chain start offsets in the concatenated sequence
        chain_start = {}
        offset = 0
        for cd in chain_dicts:
            chain_start[cd["chain_id"]] = offset
            offset += len(cd["seq"])

        # Offsets for all 6 symmetric copies (C6 symmetry of BMC hexamer)
        sym_offsets = [chain_start[c] for c in ALL_CHAINS if c in chain_start]

        # Average hidden and embedding vectors across all symmetric copies
        hid_vecs = []
        embed_vecs = []
        for sym_start in sym_offsets:
            abs_pos = sym_start + seq_pos
            hid_vecs.append(mpnn_hid[0][abs_pos])
            embed_vecs.append(mpnn_embed[0][abs_pos])

        hid = torch.stack(hid_vecs).mean(dim=0)      # [NUM_FINAL_LAYERS * 128]
        embed = torch.stack(embed_vecs).mean(dim=0)  # [128]

        return torch.cat([hid, embed], dim=-1).detach()  # [IN_DIM]

    # ------------------------------------------------------------------
    def predict_binding_from_embedding(
        self, embedding: torch.Tensor, wt_aa: str, mut_aa: str
    ) -> torch.Tensor:
        """Predict ΔΔG_binding for a single amino-acid substitution.

        Parameters
        ----------
        embedding : torch.Tensor, shape [IN_DIM]
            Symmetry-averaged embedding from extract_embedding.
        wt_aa : str
            Single-letter wild-type amino acid at the position.
        mut_aa : str
            Single-letter mutant amino acid.

        Returns
        -------
        torch.Tensor, scalar
            Predicted ΔΔG_binding = binding_ddg_out(profile[mut]) - binding_ddg_out(profile[wt]).
        """
        wt_idx = ALPHABET.index(wt_aa)
        aa_idx = ALPHABET.index(mut_aa)

        # profile shape: [VOCAB_DIM]
        profile = self.binding_out(embedding)

        # binding_ddg_out expects shape [1] → returns [1]
        ddg_mut = self.binding_ddg_out(profile[aa_idx].unsqueeze(-1))
        ddg_wt  = self.binding_ddg_out(profile[wt_idx].unsqueeze(-1))

        return ddg_mut[0] - ddg_wt[0]


# ── Training dataset builder ───────────────────────────────────────────────────

def build_training_dataset(model: BMCBindingModel) -> tuple:
    """Extract embeddings for all 4 training positions and pair with docking ΔΔG.

    Returns:
        embeddings: [76, IN_DIM] tensor
        ddg_labels: [76] tensor (ddg_mean from docking summaries, REU)
        metadata:   list of 76 dicts with keys:
                    position_key, mutant, wt_aa, mut_aa, pdb_resnum
    """
    # Step 1: Parse PduA and PduJ structures once each
    structures = {}
    print("Parsing PduA hexamer...")
    structures["PduA"] = parse_hexamer_pdb(PDUA_PDB)
    print("Parsing PduJ hexamer...")
    structures["PduJ"] = parse_hexamer_pdb(PDUJ_PDB)

    embeddings = []
    ddg_labels = []
    metadata = []

    # Step 2: Iterate over all 4 positions
    for pos_key, (csv_path, protein, pdb_resnum) in DOCKING_SUMMARIES.items():
        chain_dicts, pdb_to_seq_idx = structures[protein]
        wt_seq = chain_dicts[0]["seq"]  # chain A sequence

        # Step 2b: Validate residue is in index
        if pdb_resnum not in pdb_to_seq_idx:
            raise KeyError(
                f"PDB residue {pdb_resnum} not found in {protein} pdb_to_seq_idx. "
                f"Available range: {min(pdb_to_seq_idx)}–{max(pdb_to_seq_idx)}"
            )

        # Step 2c: Get wild-type amino acid
        seq_pos = pdb_to_seq_idx[pdb_resnum]
        wt_aa = wt_seq[seq_pos]

        # Step 2d: Print diagnostic info
        print(f"  {pos_key}: PDB {pdb_resnum} → seq_pos {seq_pos}, WT = {wt_aa}")

        # Step 2e: Extract position embedding (shared across all mutants at this position)
        pos_embedding = model.extract_embedding(chain_dicts, pdb_to_seq_idx, pdb_resnum)

        # Step 2f: Load docking summary CSV
        df = pd.read_csv(csv_path)

        # Step 2g: Iterate rows, skip WT aa and the "WT" baseline row
        for _, row in df.iterrows():
            mutant_label = row["mutant"]
            # Skip the wildtype baseline row (labeled "WT")
            if mutant_label == "WT":
                continue
            mut_aa = mutant_label[-1]
            # Skip if mutant is same as wild-type
            if mut_aa == wt_aa:
                continue
            embeddings.append(pos_embedding)
            ddg_labels.append(float(row["ddg_mean"]))
            metadata.append({
                "position_key": pos_key,
                "mutant": mutant_label,
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "pdb_resnum": pdb_resnum,
            })

    # Step 3: Verify count before converting to tensors
    assert len(embeddings) == 76, (
        f"Expected 76 training examples, got {len(embeddings)}. "
        "Check that each position has exactly 19 non-WT mutants."
    )

    # Step 4: Return tensors and metadata
    return (
        torch.stack(embeddings).to(device),
        torch.tensor(ddg_labels, dtype=torch.float32).to(device),
        metadata,
    )


# ── Training helpers ──────────────────────────────────────────────────────────

def _one_epoch(model: BMCBindingModel, embeddings: torch.Tensor,
               labels: torch.Tensor, metadata: list,
               optimizer=None) -> tuple:
    """One pass over the 76-point dataset.  Returns (mse, spearman_rho).

    Embeddings are pre-computed, so no ProteinMPNN forward pass is needed here —
    this is fast (~1 s per epoch on MPS/CPU).
    """
    training = optimizer is not None
    model.train(training)

    losses, preds, targets = [], [], []
    indices = list(range(len(metadata)))
    if training:
        np.random.shuffle(indices)

    for i in indices:
        emb    = embeddings[i]
        label  = labels[i]
        wt_aa  = metadata[i]["wt_aa"]
        mut_aa = metadata[i]["mut_aa"]

        with torch.set_grad_enabled(training):
            ddg_pred = model.predict_binding_from_embedding(emb, wt_aa, mut_aa)
            loss = (ddg_pred - label) ** 2

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        preds.append(ddg_pred.detach().cpu().item())
        targets.append(label.cpu().item())

    mse = float(np.mean(losses))
    rho = spearmanr(preds, targets).statistic if len(preds) > 2 else float("nan")
    return mse, rho


def train_binding_head(model: BMCBindingModel,
                       embeddings: torch.Tensor,
                       labels: torch.Tensor,
                       metadata: list) -> Path:
    """Train binding_out + binding_ddg_out on all 76 points.  Save best by MSE."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=BINDING_LR)

    best_mse, best_epoch = float("inf"), 0
    history = []
    print(f"\n{'Epoch':>5}  {'MSE':>8}  {'rho':>6}")
    print("-" * 24)

    for epoch in range(1, BINDING_EPOCHS + 1):
        mse, rho = _one_epoch(model, embeddings, labels, metadata, optimizer)
        history.append({"epoch": epoch, "mse": mse, "rho": rho})
        if epoch % 10 == 0:
            print(f"{epoch:>5}  {mse:>8.4f}  {rho:>6.3f}")
        if mse < best_mse:
            best_mse, best_epoch = mse, epoch
            torch.save({
                "binding_out":     model.binding_out.state_dict(),
                "binding_ddg_out": model.binding_ddg_out.state_dict(),
            }, BINDING_CKPT)

    pd.DataFrame(history).to_csv(
        "PduA_ThermoMPNN_results/binding_head_training_history.csv", index=False)
    print(f"\nBest MSE = {best_mse:.4f} at epoch {best_epoch}  →  {BINDING_CKPT}")
    return BINDING_CKPT


def loo_cv(model: BMCBindingModel,
           embeddings: torch.Tensor,
           labels: torch.Tensor,
           metadata: list) -> "pd.DataFrame":
    """Leave-one-out cross-validation over all 76 training points.

    For each point i: re-initialises the binding head, trains on the other 75,
    predicts the held-out point.  Reports per-position and overall Spearman ρ.

    Returns a DataFrame with columns: position_key, mutant, ddg_true, ddg_pred
    """
    results = []
    print("\nRunning LOO-CV (76 folds × 100 epochs each) ...")

    for i in range(len(metadata)):
        train_idx = [j for j in range(len(metadata)) if j != i]
        emb_tr    = embeddings[train_idx]
        lbl_tr    = labels[train_idx]
        meta_tr   = [metadata[j] for j in train_idx]

        # Fresh binding head — deepcopy then reset parameters
        loo_model = copy.deepcopy(model)
        for layer in loo_model.binding_out:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        loo_model.binding_ddg_out.reset_parameters()

        trainable = [p for p in loo_model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(trainable, lr=BINDING_LR)
        for _ in range(BINDING_EPOCHS):
            _one_epoch(loo_model, emb_tr, lbl_tr, meta_tr, opt)

        loo_model.eval()
        with torch.no_grad():
            ddg_pred = loo_model.predict_binding_from_embedding(
                embeddings[i], metadata[i]["wt_aa"], metadata[i]["mut_aa"]
            ).cpu().item()

        results.append({
            "position_key": metadata[i]["position_key"],
            "mutant":       metadata[i]["mutant"],
            "ddg_true":     labels[i].cpu().item(),
            "ddg_pred":     ddg_pred,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/76 complete")

    df = pd.DataFrame(results)
    overall_rho = spearmanr(df["ddg_true"], df["ddg_pred"]).statistic
    print(f"\nLOO-CV overall Spearman ρ = {overall_rho:.3f}")
    for pk, grp in df.groupby("position_key"):
        rho = spearmanr(grp["ddg_true"], grp["ddg_pred"]).statistic
        print(f"  {pk}: ρ = {rho:.3f}  (n={len(grp)})")
    df.to_csv("PduA_ThermoMPNN_results/binding_head_loo_cv.csv", index=False)
    return df


def predict_all_positions(model: BMCBindingModel) -> "pd.DataFrame":
    """Run trained binding head over all 4 positions × 19 AAs."""
    ALL_AAS = list("ACDEFGHIKLMNPQRSTVWY")
    structure_cache = {}
    for protein, pdb_path in [("PduA", PDUA_PDB), ("PduJ", PDUJ_PDB)]:
        chain_dicts, pdb_to_seq_idx = parse_hexamer_pdb(pdb_path)
        wt_seq = chain_dicts[0]["seq"]
        structure_cache[protein] = (chain_dicts, pdb_to_seq_idx, wt_seq)

    model.eval()
    rows = []
    for pos_key, (csv_path, protein, pdb_resnum) in DOCKING_SUMMARIES.items():
        chain_dicts, pdb_to_seq_idx, wt_seq = structure_cache[protein]
        seq_pos = pdb_to_seq_idx[pdb_resnum]
        wt_aa   = wt_seq[seq_pos]
        emb     = model.extract_embedding(chain_dicts, pdb_to_seq_idx, pdb_resnum)

        for mut_aa in ALL_AAS:
            if mut_aa == wt_aa:
                continue
            with torch.no_grad():
                ddg = model.predict_binding_from_embedding(
                    emb, wt_aa, mut_aa
                ).cpu().item()
            rows.append({
                "position_key":          pos_key,
                "protein":               protein,
                "pdb_resnum":            pdb_resnum,
                "wt_aa":                 wt_aa,
                "mut_aa":                mut_aa,
                "mutant":                f"{wt_aa}{pdb_resnum}{mut_aa}",
                "ddg_binding_predicted": ddg,
            })

    df = pd.DataFrame(rows)
    df.to_csv("PduA_ThermoMPNN_results/binding_head_all_predictions.csv", index=False)
    print(f"Saved {len(df)} predictions → binding_head_all_predictions.csv")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main(infer_only: bool = False) -> None:
    cfg   = make_cfg()
    model = BMCBindingModel(cfg).to(device)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable binding head params: {trainable_count:,}")

    print("\nBuilding training dataset (extracts embeddings from WT structures)...")
    embeddings, labels, metadata = build_training_dataset(model)

    if not infer_only:
        print("\nRunning LOO-CV (76 folds × 100 epochs each)...")
        loo_cv(model, embeddings, labels, metadata)

        print("\nTraining final binding head on all 76 points...")
        ckpt = train_binding_head(model, embeddings, labels, metadata)
        ckpt_data = torch.load(ckpt, map_location=device)
        model.binding_out.load_state_dict(ckpt_data["binding_out"])
        model.binding_ddg_out.load_state_dict(ckpt_data["binding_ddg_out"])

    elif BINDING_CKPT.exists():
        ckpt_data = torch.load(BINDING_CKPT, map_location=device)
        model.binding_out.load_state_dict(ckpt_data["binding_out"])
        model.binding_ddg_out.load_state_dict(ckpt_data["binding_ddg_out"])
        print(f"Loaded checkpoint: {BINDING_CKPT}")
    else:
        raise FileNotFoundError(
            f"No checkpoint at {BINDING_CKPT}. Run without --infer first."
        )

    print("\nPredicting all positions × 19 AAs...")
    pred_df = predict_all_positions(model)
    print(pred_df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and run BMC binding prediction head"
    )
    parser.add_argument("--infer", action="store_true",
                        help="Skip training; load checkpoint and run predictions only.")
    args = parser.parse_args()
    main(infer_only=args.infer)
