# RMSD / EutM Consistency Fixes — Design Spec

**Date:** 2026-03-28
**Notebooks:** `rmsd_analysis.ipynb`, `EutM_Docking.ipynb`
**Goal:** Three targeted fixes for scientific rigor and cross-protein comparability.

---

## Fix 1 — Pore Position Consistency Audit (`rmsd_analysis.ipynb`)

### Problem

`batch_pore_geometry` (Rosetta Pose, `target_pdb_num`) and `batch_crystal_pore_geometry` (raw PDB,
`target_resseq`) must measure the **same residue** for results to be comparable. If they use
different position values, diameter outputs from two calls on the same structure set will differ.

### Audit Checklist

Verify these invariants hold for every existing call in the PduA section:

| Call | Parameter | Expected value | Note |
|------|-----------|---------------|------|
| `batch_pore_geometry` S40 variants | `target_pdb_num` | **40** | pore constriction |
| `batch_pore_geometry` K37 variants | `target_pdb_num` | **40** | deliberately measures S40, not K37 |
| `batch_crystal_pore_geometry` crystal structures | `target_resseq` | **40** | `use_absolute_resseq=True` (default) |
| `batch_crystal_pore_geometry` docked structures | `target_resseq` | **40** | `use_absolute_resseq=False` |
| Spearman heatmap unbound calls | `target_resseq` | **40** | `use_absolute_resseq=False` |

If any call deviates from the expected value, fix it in place.

### PduJ Position Mapping

PduJ pore constriction is **S39** (analogous to PduA S40). All new PduJ pore geometry calls must use:

| Call | Parameter | Value |
|------|-----------|-------|
| `batch_pore_geometry` S39 variants | `target_pdb_num` | **39** |
| `batch_pore_geometry` K36 variants | `target_pdb_num` | **39** (pore constriction, not K36) |
| `batch_crystal_pore_geometry` crystal structures | `target_resseq` | **39** |
| `batch_crystal_pore_geometry` docked structures | `target_resseq` | **39**, `use_absolute_resseq=False` |

---

## Fix 2 — PduJ S39 / K36 Full Analysis Block (`rmsd_analysis.ipynb`)

### Scope

Extend cell `cell-pduj` (the existing PduJ template cell) to include the full CALL 1–5 pipeline
that PduA S40/K37 receives. Currently the PduJ cell only has mutants-vs-WT RMSD and a pairwise
RMSD matrix.

### What to Add (after pairwise matrix)

Mirror the PduA analysis block exactly, with these substitutions:

| PduA | PduJ |
|------|------|
| `s40_pdbs` | `pduj_s39_pdbs` |
| `k37_pdbs` | `pduj_k36_pdbs` |
| `target_pdb_num=40` | `target_pdb_num=39` |
| `PORE_RANGE = (38, 42)` | `S39_PORE_RANGE = (37, 41)` |
| `K37_PORE_RANGE = (35, 39)` | `K36_PORE_RANGE = (34, 38)` |
| `target_resseq=40` | `target_resseq=39` |
| `OUT_DIR` | `PDUJ_OUT_DIR` |
| `'WT'` key → `WT_PDB` | `'WT'` key → `PDUJ_WT_PDB` |
| `CRYSTAL_PDB_MAP` | `PDUJ_CRYSTAL_PDB_MAP` (may be empty `{}`) |

### New Sub-variables to Define

```python
# Add after the existing pairwise matrix block in cell 31

# Docking root for PduJ (analogous to DOCKING_ROOT for PduA)
PDUJ_DOCKING_ROOT = os.path.join(HOME, 'PduJ_docking_mutants')
PDUJ_COMBINED_CSV = os.path.join(PDUJ_OUT_DIR, 'all_docking_scores.csv')

# Filter from pduj_mutant_pdbs (already populated by discover_mutant_pdbs)
pduj_s39_pdbs = {k: v for k, v in pduj_mutant_pdbs.items() if k.startswith('S39')}
pduj_s39_pdbs['WT'] = PDUJ_WT_PDB

pduj_k36_pdbs = {k: v for k, v in pduj_mutant_pdbs.items() if k.startswith('K36')}
pduj_k36_pdbs['WT'] = PDUJ_WT_PDB

S39_PORE_RANGE = (37, 41)   # S39-centred window (analogous to PORE_RANGE = (38, 42) for S40)
K36_PORE_RANGE = (34, 38)   # K36-centred window (analogous to K37_PORE_RANGE = (35, 39))

MUTANTS_FOR_BVU_S39 = sorted(k for k in pduj_s39_pdbs if k != 'WT')
MUTANTS_FOR_BVU_K36 = sorted(k for k in pduj_k36_pdbs if k != 'WT')
```

### Analysis Steps (in order)

**CALL 1 — S39 pore geometry (Rosetta models)**
```python
df_pore_geo_s39 = batch_pore_geometry(
    pdb_map        = pduj_s39_pdbs,
    target_pdb_num = 39,
    wt_key         = 'WT',
    fix_chains     = FIX_CHAINS,
)
```

**CALL 1b — S39 crystal pore geometry** (only if `PDUJ_CRYSTAL_PDB_MAP` is non-empty)
```python
if PDUJ_CRYSTAL_PDB_MAP:
    df_crystal_pore_s39 = batch_crystal_pore_geometry(
        crystal_pdb_map = PDUJ_CRYSTAL_PDB_MAP,
        target_resseq   = 39,
    )
```

**CALL 2 — K36 pore geometry (Rosetta models, measuring S39 constriction)**
```python
df_pore_geo_k36 = batch_pore_geometry(
    pdb_map        = pduj_k36_pdbs,
    target_pdb_num = 39,   # measure S39 pore (actual constriction), not K36 position
    wt_key         = 'WT',
    fix_chains     = FIX_CHAINS,
)
```

**CALL 3 — Bound vs. unbound (S39)**
```python
# Collect all docking scores into a combined CSV (mirrors PduA COMBINED_CSV pattern)
import glob
_score_files_j = sorted(glob.glob(os.path.join(PDUJ_DOCKING_ROOT, '*/docking_scores.csv')))
if _score_files_j:
    pd.concat([pd.read_csv(f) for f in _score_files_j], ignore_index=True).to_csv(
        PDUJ_COMBINED_CSV, index=False)
else:
    print(f'WARNING: no docking_scores.csv found under {PDUJ_DOCKING_ROOT}')

df_bvu_s39 = batch_bound_vs_unbound_fixed(
    unbound_dir   = PDUJ_MUTANT_DIR,
    docking_root  = PDUJ_DOCKING_ROOT,
    mutants       = MUTANTS_FOR_BVU_S39,
    chain         = 'A',
    pore_range    = S39_PORE_RANGE,
    atom_names    = ('CB',),
    fix_chains    = FIX_CHAINS,
    score_col     = 'interface_energy',
)
display(df_bvu_s39.round(3))
df_bvu_s39.to_csv(os.path.join(PDUJ_OUT_DIR, 'bound_vs_unbound_S39.csv'), index=False)

plot_bound_vs_unbound(
    df_bvu_s39,
    title     = 'PduJ S39X: CB RMSD Bound vs. Unbound',
    save_path = os.path.join(PDUJ_OUT_DIR, 'bound_vs_unbound_CB_S39.png'),
)
plt.show()
```

**CALL 4 — Bound vs. unbound (K36)**
```python
df_bvu_k36 = batch_bound_vs_unbound_fixed(
    unbound_dir   = PDUJ_MUTANT_DIR,
    docking_root  = PDUJ_DOCKING_ROOT,
    mutants       = MUTANTS_FOR_BVU_K36,
    chain         = 'A',
    pore_range    = K36_PORE_RANGE,
    atom_names    = ('CB',),
    fix_chains    = FIX_CHAINS,
    score_col     = 'interface_energy',
)
display(df_bvu_k36.round(3))
df_bvu_k36.to_csv(os.path.join(PDUJ_OUT_DIR, 'bound_vs_unbound_K36.csv'), index=False)

plot_bound_vs_unbound(
    df_bvu_k36,
    title     = 'PduJ K36X: CB RMSD Bound vs. Unbound',
    save_path = os.path.join(PDUJ_OUT_DIR, 'bound_vs_unbound_CB_K36.png'),
)
plt.show()
```

**CALL 5 — Pore geometry from best docked poses (bound + apo)**

```python
best_s39 = find_best_bound_poses(PDUJ_DOCKING_ROOT, MUTANTS_FOR_BVU_S39,
                                  score_col='interface_energy')
best_s39['WT'] = PDUJ_WT_PDB

best_k36_pore = find_best_bound_poses(PDUJ_DOCKING_ROOT, MUTANTS_FOR_BVU_K36,
                                       score_col='interface_energy')
best_k36_pore['WT'] = PDUJ_WT_PDB

df_pore_bound_s39 = batch_crystal_pore_geometry(
    crystal_pdb_map=best_s39, target_resseq=39, use_absolute_resseq=False)
df_pore_bound_k36 = batch_crystal_pore_geometry(
    crystal_pdb_map=best_k36_pore, target_resseq=39, use_absolute_resseq=False)

df_pore_apo_s39 = batch_crystal_pore_geometry(
    crystal_pdb_map=pduj_s39_pdbs, target_resseq=39, use_absolute_resseq=False)
df_pore_apo_k36 = batch_crystal_pore_geometry(
    crystal_pdb_map=pduj_k36_pdbs, target_resseq=39, use_absolute_resseq=False)
```

**CALL 6 — Spearman heatmap**

Uses `_build_spearman_matrix` and `_plot_spearman_heatmap` (private helpers already defined in
the PduA CALL 5 cell — they remain in scope). PduJ has no literature ΔΔG or OD600 data yet, so
those arguments are passed as empty dicts / `None`.

```python
# Unbound pore geometry for heatmap (rank-based, all PduJ S39/K36 models)
_df_apo_s39_hm = batch_crystal_pore_geometry(
    crystal_pdb_map=pduj_s39_pdbs, target_resseq=39,
    use_absolute_resseq=False, hexamer_chains=None)
_df_apo_k36_hm = batch_crystal_pore_geometry(
    crystal_pdb_map=pduj_k36_pdbs, target_resseq=39,
    use_absolute_resseq=False, hexamer_chains=None)

rho_s39, p_s39 = _build_spearman_matrix(
    site_prefix  = 'S39',
    combined_csv = PDUJ_COMBINED_CSV,
    df_pore_geo  = df_pore_geo_s39,
    df_apo_pore  = _df_apo_s39_hm,
    df_pr_delta  = None,        # pore radius delta not computed for PduJ
    lit_ddg      = {},          # no PduJ literature ΔΔG yet
    assembly_map = {},          # no PduJ crystal assembly map yet
)

rho_k36, p_k36 = _build_spearman_matrix(
    site_prefix  = 'K36',
    combined_csv = PDUJ_COMBINED_CSV,
    df_pore_geo  = df_pore_geo_k36,
    df_apo_pore  = _df_apo_k36_hm,
    df_pr_delta  = None,
    lit_ddg      = {},
    assembly_map = {},
)

_vabs = min(1.0, max(np.nanmax(np.abs(rho_s39.values)),
                     np.nanmax(np.abs(rho_k36.values))))
fig, (ax_s39, ax_k36) = plt.subplots(2, 1, figsize=(10, 12),
                                      gridspec_kw={'hspace': 0.55})
im_s39 = _plot_spearman_heatmap(rho_s39, p_s39,
    title='PduJ S39X — physicochemical property correlations', ax=ax_s39, vabs=_vabs)
im_k36 = _plot_spearman_heatmap(rho_k36, p_k36,
    title='PduJ K36X — physicochemical property correlations', ax=ax_k36, vabs=_vabs)
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.12, 0.025, 0.76])
fig.colorbar(im_s39, cax=cbar_ax).set_label('Spearman ρ', fontsize=10)
fig.savefig(os.path.join(PDUJ_OUT_DIR, 'spearman_property_heatmap_PduJ.png'),
            dpi=150, bbox_inches='tight')
rho_s39.to_csv(os.path.join(PDUJ_OUT_DIR, 'spearman_matrix_S39.csv'))
rho_k36.to_csv(os.path.join(PDUJ_OUT_DIR, 'spearman_matrix_K36.csv'))
plt.show()
```

### Crystal Structures

`PDUJ_CRYSTAL_PDB_MAP` is currently an empty dict. The crystal pore geometry call and crystal
comparison plot are wrapped in `if PDUJ_CRYSTAL_PDB_MAP:` guards so the cell runs cleanly even
without crystal structures. If PduJ crystal PDBs are added later, they will slot in automatically.

---

## Fix 3 — EutM Pore Sidechain Prep (`EutM_Docking.ipynb`)

### Problem

PduA_Mutant_Docking.ipynb repacks sidechains within 10 Å of the mutation site before docking,
relaxing crystal packing artifacts at the binding site. EutM_Docking.ipynb docks to WT EutM
without this step, making pore flexibility incomparable.

### Solution

Add a **pore sidechain prep cell** immediately before the docking loop (after the pose is loaded,
before the first `RosettaDock` call). Pack all 30 pore residues (6 chains × Ile36–Leu40, PDB
residues 36–40) using `PackRotamersMover` with `NeighborhoodResidueSelector`.

### Implementation

```python
from pyrosetta.rosetta.core.select.residue_selector import (
    ResidueIndexSelector,
    NeighborhoodResidueSelector,
    OrResidueSelector,
)
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import (
    RestrictToRepacking,
    OperateOnResidueSubset,
    PreventRepackingRLT,
)

PORE_RESIDUE_NUMS = [36, 37, 38, 39, 40]
HEXAMER_CHAINS   = ['A', 'B', 'C', 'D', 'E', 'F']

def prep_eutm_pore_sidechains(pose, scorefxn):
    """Pack pore sidechains within 10 Å before docking to relax crystal artifacts."""
    pi = pose.pdb_info()
    pore_pose_nums = []
    for i in range(1, pose.total_residue() + 1):
        if pi.chain(i) in HEXAMER_CHAINS and pi.number(i) in PORE_RESIDUE_NUMS:
            pore_pose_nums.append(i)

    pore_sel  = ResidueIndexSelector(','.join(str(r) for r in pore_pose_nums))
    nbr_sel   = NeighborhoodResidueSelector(pore_sel, distance=10.0, include_focus_in_subset=True)

    # Allow repacking only within neighborhood; freeze everything else
    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    freeze_op = OperateOnResidueSubset(PreventRepackingRLT(), nbr_sel, flip_subset=True)
    tf.push_back(freeze_op)

    packer = PackRotamersMover(scorefxn)
    packer.task_factory(tf)
    packer.apply(pose)
    print(f'  Packed {len(pore_pose_nums)} pore residues + 10 Å neighborhood.')
```

Call before the docking loop:
```python
prep_eutm_pore_sidechains(eutm_pose, scorefxn)
```

### Notes

- Backbone is **not** relaxed — backbone degrees of freedom remain frozen (only chi angles change),
  consistent with PduA's `relax_around_mutations()` approach.
- `relax.max_iter(200)` in the existing FastRelax step is kept as-is; it was a pre-existing
  difference and is out of scope for this fix.
- No mutation is introduced. This is purely a WT structure preparation step.

---

## Outputs

| Notebook | New/Changed | Description |
|----------|-------------|-------------|
| `rmsd_analysis.ipynb` | Existing calls audited | Confirm/fix position consistency |
| `rmsd_analysis.ipynb` | New cells | PduJ S39/K36 CALL 1–6 block |
| `EutM_Docking.ipynb` | New cell | `prep_eutm_pore_sidechains` before docking loop |

---

## Implementation Order

1. Pore consistency audit (read + fix existing PduA calls) → confirm no bugs
2. PduJ S39/K36 block (extend `cell-pduj`)
3. EutM pore prep cell
