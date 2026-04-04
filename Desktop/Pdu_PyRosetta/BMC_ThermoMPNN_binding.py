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

        with torch.no_grad():
            X, S, mask, lengths, chain_M, chain_encoding_all, \
                chain_list_list, visible_list_list, masked_list_list, \
                masked_chain_length_list_list, chain_M_pos, omit_AA_mask, \
                residue_idx, dihedral_mask, tied_pos_list_of_lists_list, \
                pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, \
                tied_beta = tied_featurize(
                    chain_dicts, device, None, None, None, None, None, None,
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
