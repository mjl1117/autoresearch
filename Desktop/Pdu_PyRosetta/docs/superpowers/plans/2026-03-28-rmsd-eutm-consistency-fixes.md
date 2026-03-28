# RMSD / EutM Consistency Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix pore position consistency in rmsd_analysis.ipynb, add full PduJ S39/K36 analysis block mirroring PduA, and add EutM pore sidechain prep before docking.

**Architecture:** Three sequential notebook edits. Task 1 audits existing PduA cells (no new code if clean). Tasks 2–6 extend `cell-pduj` and add new cells in rmsd_analysis.ipynb. Task 7 inserts a new cell in EutM_Docking.ipynb after the score-function cell. No new Python files are created.

**Tech Stack:** Jupyter notebooks, PyRosetta, pandas, scipy, matplotlib, seaborn

---

## File Structure

**Modified files:**
- `rmsd_analysis.ipynb` — extend cell `cell-pduj` (index 31) and insert new cells after it
- `EutM_Docking.ipynb` — insert new cell after cell `a1000009` (index 8, score functions)

**Cell IDs to know:**
- `rmsd_analysis.ipynb` `cell-pduj` (index 31): existing PduJ template — append config + CALL 1/2
- New cells inserted after `cell-pduj`: CALL 3/4, CALL 5, CALL 6 (each a separate cell)
- `EutM_Docking.ipynb` `a1000009` (index 8): CONSTRAINTS & SCORE FUNCTIONS — insert prep after this

---

## Task 1: Pore Position Consistency Audit

**Files:**
- Read: `rmsd_analysis.ipynb` cells 23, 25, 29 (PduA CALL 2, CALL 3, CALL 5)
- Modify: `rmsd_analysis.ipynb` (only if deviations found)

Verify every `batch_pore_geometry` and `batch_crystal_pore_geometry` call in the PduA section uses the correct position value. All PduA calls must measure at position 40 (S40 pore constriction).

- [ ] **Step 1: Check cell 23 (CALL 2 — pore geometry)**

Read the cell and verify these values:
```
batch_pore_geometry(target_pdb_num=40, ...)      # S40 variants ✓
batch_pore_geometry(target_pdb_num=40, ...)      # K37 variants ✓ (deliberately measures S40 constriction)
batch_crystal_pore_geometry(target_resseq=40, ...)  # crystal ✓
```

- [ ] **Step 2: Check cell 25 (CALL 3 — bound vs unbound)**

Verify:
```
PORE_RANGE = (38, 42)
K37_PORE_RANGE = (35, 39)
batch_crystal_pore_geometry(target_resseq=40, use_absolute_resseq=False, ...)  # docked structures ✓
```

- [ ] **Step 3: Check cell 29 (CALL 5 — Spearman heatmap)**

Verify:
```
batch_crystal_pore_geometry(target_resseq=40, use_absolute_resseq=False, hexamer_chains=None)  # ×2 ✓
```

- [ ] **Step 4: Fix any deviations**

If any call uses a value other than 40, correct it in place using NotebookEdit (replace mode on the relevant cell). If all values are correct, no edit needed.

- [ ] **Step 5: Commit**

```bash
git add rmsd_analysis.ipynb
git commit -m "fix: verify pore position consistency in PduA rmsd analysis cells"
```

---

## Task 2: Add PduJ Config + CALL 1 (S39) + CALL 2 (K36) Pore Geometry

**Files:**
- Modify: `rmsd_analysis.ipynb` cell `cell-pduj` (index 31) — append config and CALL 1/2

Extend the existing PduJ template cell with sub-variables and pore geometry calls for both S39 and K36.

- [ ] **Step 1: Append to cell `cell-pduj`**

Using NotebookEdit replace mode on `cell-pduj`, append the following to the END of the existing source (keep the existing mutants_vs_wt and pairwise matrix code, add this after `plt.show()`):

```python
# ─────────────────────────────────────────────────────────────────────────────
# PduJ-specific sub-variables
# S39 = pore constriction (analogous to PduA S40)
# K36 = lysine gate (analogous to PduA K37)
# ─────────────────────────────────────────────────────────────────────────────

PDUJ_DOCKING_ROOT = os.path.join(HOME, 'PduJ_docking_mutants')
PDUJ_COMBINED_CSV = os.path.join(PDUJ_OUT_DIR, 'all_docking_scores.csv')

pduj_s39_pdbs = {k: v for k, v in pduj_mutant_pdbs.items() if k.startswith('S39')}
pduj_s39_pdbs['WT'] = PDUJ_WT_PDB

pduj_k36_pdbs = {k: v for k, v in pduj_mutant_pdbs.items() if k.startswith('K36')}
pduj_k36_pdbs['WT'] = PDUJ_WT_PDB

S39_PORE_RANGE = (37, 41)   # S39-centred window  (cf. PORE_RANGE = (38, 42) for S40)
K36_PORE_RANGE = (34, 38)   # K36-centred window  (cf. K37_PORE_RANGE = (35, 39))

MUTANTS_FOR_BVU_S39 = sorted(k for k in pduj_s39_pdbs if k != 'WT')
MUTANTS_FOR_BVU_K36 = sorted(k for k in pduj_k36_pdbs if k != 'WT')

print(f'PduJ S39 variants: {len(pduj_s39_pdbs) - 1}')
print(f'PduJ K36 variants: {len(pduj_k36_pdbs) - 1}')

# ── CALL 1 — S39 pore geometry (Rosetta models) ───────────────────────────
print('=' * 60)
print('  PORE GEOMETRY — PduJ S39 VARIANTS (Rosetta models)')
print('=' * 60)

df_pore_geo_s39 = batch_pore_geometry(
    pdb_map        = pduj_s39_pdbs,
    target_pdb_num = 39,
    wt_key         = 'WT',
    fix_chains     = FIX_CHAINS,
)
display(df_pore_geo_s39.round(3))
df_pore_geo_s39.to_csv(os.path.join(PDUJ_OUT_DIR, 'pore_geometry_S39.csv'), index=False)

if PDUJ_CRYSTAL_PDB_MAP:
    print('=' * 60)
    print('  CRYSTAL PORE GEOMETRY — PduJ S39 (PDB resseq 39)')
    print('=' * 60)
    df_crystal_pore_s39 = batch_crystal_pore_geometry(
        crystal_pdb_map = PDUJ_CRYSTAL_PDB_MAP,
        target_resseq   = 39,
    )
    display(df_crystal_pore_s39.round(3))
    df_crystal_pore_s39.to_csv(os.path.join(PDUJ_OUT_DIR, 'crystal_pore_geometry_S39.csv'), index=False)
else:
    df_crystal_pore_s39 = None
    print('  (No PduJ crystal structures configured — skipping crystal pore geometry)')

# ── CALL 2 — K36 pore geometry (Rosetta models, measuring S39 constriction) ─
print('=' * 60)
print('  PORE GEOMETRY — PduJ K36 VARIANTS (measuring S39 constriction)')
print('=' * 60)

df_pore_geo_k36 = batch_pore_geometry(
    pdb_map        = pduj_k36_pdbs,
    target_pdb_num = 39,   # measure S39 pore (actual constriction), not K36 position
    wt_key         = 'WT',
    fix_chains     = FIX_CHAINS,
)
display(df_pore_geo_k36.round(3))
df_pore_geo_k36.to_csv(os.path.join(PDUJ_OUT_DIR, 'pore_geometry_K36.csv'), index=False)
```

- [ ] **Step 2: Run the cell and verify output**

Expected output (approximate):
```
PduJ S39 variants: <N>
PduJ K36 variants: <N>
============================
  PORE GEOMETRY — PduJ S39 VARIANTS (Rosetta models)
============================
<dataframe with pore_radius_A column, values in ~2–5 Å range>
  (No PduJ crystal structures configured — skipping crystal pore geometry)
============================
  PORE GEOMETRY — PduJ K36 VARIANTS (measuring S39 constriction)
============================
<dataframe with pore_radius_A column>
```

Confirm: no exceptions, S39 and K36 `pore_radius_A` values look plausible (0–6 Å for non-Gly residues; larger for open-pore mutants).

- [ ] **Step 3: Commit**

```bash
git add rmsd_analysis.ipynb
git commit -m "feat: add PduJ S39/K36 pore geometry (CALL 1+2)"
```

---

## Task 3: Add CALL 3 + 4 — Bound vs. Unbound

**Files:**
- Modify: `rmsd_analysis.ipynb` — insert new cell after `cell-pduj`

**Prerequisite:** `PDUJ_DOCKING_ROOT` must contain `{mutant}/docking_scores.csv` files from PduJ docking runs. If docking has not been run yet, the cell will print a warning and skip gracefully.

- [ ] **Step 1: Insert new cell after `cell-pduj`**

Use NotebookEdit insert mode with `cell_id='cell-pduj'`:

```python
# ─────────────────────────────────────────────────────────────────────────────
# PduJ CALL 3 — Bound vs. Unbound Conformational Shift (S39 and K36)
# ─────────────────────────────────────────────────────────────────────────────

# Collect all docking scores into combined CSV
_score_files_j = sorted(glob.glob(os.path.join(PDUJ_DOCKING_ROOT, '*/docking_scores.csv')))
if _score_files_j:
    pd.concat([pd.read_csv(f) for f in _score_files_j], ignore_index=True).to_csv(
        PDUJ_COMBINED_CSV, index=False)
    print(f'  Combined {len(_score_files_j)} docking score files → {PDUJ_COMBINED_CSV}')
else:
    print(f'  WARNING: no docking_scores.csv found under {PDUJ_DOCKING_ROOT}')
    print('  CALL 3-6 require PduJ docking data. Skipping.')

if _score_files_j:
    print('\nS39 variants — CB RMSD (bound vs. unbound):')
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

    # K36 is at PDB residue 36 — S39_PORE_RANGE (37,41) excludes it.
    # Use K36-centred window (34,38) so residue 36 is within the range.
    print('\nK36 variants — CB RMSD (pore_range centred on residue 36):')
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

- [ ] **Step 2: Run the cell and verify output**

If docking data exists: expect two tables + two bar charts (one per site).
If no docking data: expect the "WARNING" message and clean exit (no exception).

- [ ] **Step 3: Commit**

```bash
git add rmsd_analysis.ipynb
git commit -m "feat: add PduJ S39/K36 bound vs unbound analysis (CALL 3+4)"
```

---

## Task 4: Add CALL 5 — Pore Geometry from Best Docked Poses

**Files:**
- Modify: `rmsd_analysis.ipynb` — insert new cell after the CALL 3/4 cell

- [ ] **Step 1: Insert new cell after the CALL 3/4 cell**

The cell added in Task 3 has no explicit ID assigned. Use `NotebookEdit` insert mode with `cell_number` pointing to the index of that cell (which will be 32 after Task 3 insert), OR use the approach of reading the notebook, finding the new cell's auto-assigned ID, and inserting after it. The simplest approach: insert at `cell_number=33` with `edit_mode=insert`.

Alternatively, re-read the notebook with:
```python
import json
nb = json.load(open('/Users/matthew/Desktop/Pdu_PyRosetta/rmsd_analysis.ipynb'))
print(nb['cells'][32].get('id'))  # get ID of the CALL 3/4 cell just added
```
Then use that ID with `cell_id=<id>` + `edit_mode=insert`.

Cell source:
```python
# ─────────────────────────────────────────────────────────────────────────────
# PduJ CALL 5 — Pore Geometry from Best Docked Poses (bound + apo)
# ─────────────────────────────────────────────────────────────────────────────

if not _score_files_j:
    print('No docking data — skipping CALL 5.')
else:
    best_s39 = find_best_bound_poses(PDUJ_DOCKING_ROOT, MUTANTS_FOR_BVU_S39,
                                      score_col='interface_energy')
    best_s39['WT'] = PDUJ_WT_PDB

    best_k36_pore = find_best_bound_poses(PDUJ_DOCKING_ROOT, MUTANTS_FOR_BVU_K36,
                                           score_col='interface_energy')
    best_k36_pore['WT'] = PDUJ_WT_PDB

    print('Measuring pore geometry from S39 bound poses (pure PDB reader):')
    df_pore_bound_s39 = batch_crystal_pore_geometry(
        crystal_pdb_map     = best_s39,
        target_resseq       = 39,
        use_absolute_resseq = False,
    )
    display(df_pore_bound_s39.round(3))

    print('Measuring pore geometry from S39 unbound mutant PDBs:')
    df_pore_apo_s39 = batch_crystal_pore_geometry(
        crystal_pdb_map     = pduj_s39_pdbs,
        target_resseq       = 39,
        use_absolute_resseq = False,
    )
    display(df_pore_apo_s39.round(3))

    print('Measuring pore geometry from K36 bound poses (pure PDB reader):')
    df_pore_bound_k36 = batch_crystal_pore_geometry(
        crystal_pdb_map     = best_k36_pore,
        target_resseq       = 39,
        use_absolute_resseq = False,
    )
    display(df_pore_bound_k36.round(3))

    print('Measuring pore geometry from K36 unbound mutant PDBs:')
    df_pore_apo_k36 = batch_crystal_pore_geometry(
        crystal_pdb_map     = pduj_k36_pdbs,
        target_resseq       = 39,
        use_absolute_resseq = False,
    )
    display(df_pore_apo_k36.round(3))
```

- [ ] **Step 2: Run the cell and verify output**

If docking data exists: expect four dataframes, each with `mutant` and `pore_radius_A` columns.
If no docking data: expect "No docking data — skipping CALL 5." and clean exit.

- [ ] **Step 3: Commit**

```bash
git add rmsd_analysis.ipynb
git commit -m "feat: add PduJ pore geometry from best docked poses (CALL 5)"
```

---

## Task 5: Add CALL 6 — Spearman Heatmap

**Files:**
- Modify: `rmsd_analysis.ipynb` — insert new cell after the CALL 5 cell

`_build_spearman_matrix` and `_plot_spearman_heatmap` are private helpers defined inside the PduA CALL 5 cell (cell 29). They remain in scope after that cell runs. PduJ has no literature ΔΔG or OD600 data yet, so `lit_ddg={}`, `od600_map={}`, `assembly_map={}`. `df_pr_delta=None` is handled gracefully by `_build_spearman_matrix`.

- [ ] **Step 1: Insert new cell after the CALL 5 cell**

Read the notebook to get the ID of the CALL 5 cell just added (it will be the last non-empty cell before index 32+), then insert after it. Cell source:

```python
# ─────────────────────────────────────────────────────────────────────────────
# PduJ CALL 6 — Spearman ρ Heatmap: AA physicochemical properties vs. metrics
#
# Uses _build_spearman_matrix and _plot_spearman_heatmap defined in PduA CALL 5
# (cell 29). Requires that cell to have been run first.
#
# PduJ has no literature ΔΔG, OD600, or assembly enrichment data yet.
# Those columns will appear as NaN in the correlation matrices.
# ─────────────────────────────────────────────────────────────────────────────

if not os.path.exists(PDUJ_COMBINED_CSV):
    print(f'PDUJ_COMBINED_CSV not found at {PDUJ_COMBINED_CSV}')
    print('Run PduJ docking first, then re-run this cell.')
else:
    # Build unbound pore geometry for all PduJ S39 and K36 models (rank-based lookup)
    _df_apo_s39_hm = batch_crystal_pore_geometry(
        crystal_pdb_map     = pduj_s39_pdbs,
        target_resseq       = 39,
        use_absolute_resseq = False,
        hexamer_chains      = None,
    )
    _df_apo_k36_hm = batch_crystal_pore_geometry(
        crystal_pdb_map     = pduj_k36_pdbs,
        target_resseq       = 39,
        use_absolute_resseq = False,
        hexamer_chains      = None,
    )

    print('=' * 60)
    print('  BUILDING SPEARMAN ρ MATRICES — PduJ')
    print('=' * 60)

    rho_s39, p_s39 = _build_spearman_matrix(
        site_prefix  = 'S39',
        combined_csv = PDUJ_COMBINED_CSV,
        df_pore_geo  = df_pore_geo_s39,
        df_apo_pore  = _df_apo_s39_hm,
        df_pr_delta  = None,
        lit_ddg      = {},
        od600_map    = {},
        assembly_map = {},
    )

    rho_k36, p_k36 = _build_spearman_matrix(
        site_prefix  = 'K36',
        combined_csv = PDUJ_COMBINED_CSV,
        df_pore_geo  = df_pore_geo_k36,
        df_apo_pore  = _df_apo_k36_hm,
        df_pr_delta  = None,
        lit_ddg      = {},
        od600_map    = {},
        assembly_map = {},
    )

    print('\nPduJ S39 Spearman ρ matrix:')
    display(rho_s39.round(3))
    print('\nPduJ K36 Spearman ρ matrix:')
    display(rho_k36.round(3))

    _vabs = min(1.0, float(max(
        np.nanmax(np.abs(rho_s39.values)),
        np.nanmax(np.abs(rho_k36.values)),
    )))

    fig, (ax_s39, ax_k36) = plt.subplots(
        2, 1,
        figsize=(max(len(rho_s39.columns), len(rho_k36.columns)) * 1.55 + 1.5,
                 len(PHYS_PROPS) * 0.78 * 2 + 1.2),
        gridspec_kw={'hspace': 0.55},
    )
    im_s39 = _plot_spearman_heatmap(
        rho_s39, p_s39,
        title='PduJ S39X — physicochemical property correlations',
        ax=ax_s39, vabs=_vabs,
    )
    im_k36 = _plot_spearman_heatmap(
        rho_k36, p_k36,
        title='PduJ K36X — physicochemical property correlations',
        ax=ax_k36, vabs=_vabs,
    )
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.12, 0.025, 0.76])
    fig.colorbar(im_s39, cax=cbar_ax).set_label('Spearman ρ', fontsize=10)
    fig.suptitle(
        'PduJ: Spearman ρ — AA physicochemical properties vs. docking / structural metrics',
        fontsize=11, y=1.01,
    )
    _hm_path = os.path.join(PDUJ_OUT_DIR, 'spearman_property_heatmap_PduJ.png')
    fig.savefig(_hm_path, dpi=150, bbox_inches='tight')
    print(f'\n  Saved → {_hm_path}')
    plt.show()

    rho_s39.to_csv(os.path.join(PDUJ_OUT_DIR, 'spearman_matrix_S39.csv'))
    rho_k36.to_csv(os.path.join(PDUJ_OUT_DIR, 'spearman_matrix_K36.csv'))
    print(f'  Saved → {PDUJ_OUT_DIR}/spearman_matrix_S39.csv')
    print(f'  Saved → {PDUJ_OUT_DIR}/spearman_matrix_K36.csv')
```

- [ ] **Step 2: Run the cell and verify output**

If docking data and combined CSV exist: expect two ρ matrices (rows = physicochemical properties, columns = metrics) and a two-panel heatmap figure saved to `PDUJ_OUT_DIR`.
If no docking data: expect the "PDUJ_COMBINED_CSV not found" message and clean exit.

- [ ] **Step 3: Commit**

```bash
git add rmsd_analysis.ipynb
git commit -m "feat: add PduJ Spearman property heatmap (CALL 6)"
```

---

## Task 6: EutM Pore Sidechain Prep

**Files:**
- Modify: `EutM_Docking.ipynb` — insert new cell after cell `a1000009` (index 8, CONSTRAINTS & SCORE FUNCTIONS)

After `sf_hard` and `sf_soft` are defined (cell 8) and `pose` (protein + ligand complex) is built (cell 7), pack the 30 pore sidechains within a 10 Å neighborhood. Uses `pore_residues` (list of pose residue indices) already built by cell 6.

- [ ] **Step 1: Insert new cell after `a1000009`**

Use NotebookEdit insert mode with `cell_id='a1000009'`:

```python
# ── PORE SIDECHAIN PREP ────────────────────────────────────────────────────
# Pack pore sidechains + 10 Å neighborhood before docking.
# This relaxes crystal-packing artifacts at the binding site, matching the
# pre-docking repacking done in PduA_Mutant_Docking.ipynb.
# Only chi angles change — backbone is frozen (RestrictToRepacking).
from pyrosetta.rosetta.core.select.residue_selector import (
    ResidueIndexSelector,
    NeighborhoodResidueSelector,
)
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import (
    RestrictToRepacking,
    OperateOnResidueSubset,
    PreventRepackingRLT,
)

print("Pore sidechain prep (10 Å neighborhood repacking)...")
pore_sel = ResidueIndexSelector(','.join(str(r) for r in pore_residues))
nbr_sel  = NeighborhoodResidueSelector(pore_sel, distance=10.0,
                                        include_focus_in_subset=True)

tf = TaskFactory()
tf.push_back(RestrictToRepacking())
freeze_op = OperateOnResidueSubset(PreventRepackingRLT(), nbr_sel, flip_subset=True)
tf.push_back(freeze_op)

packer = PackRotamersMover(sf_hard)
packer.task_factory(tf)
packer.apply(pose)
print(f"  Packed {len(pore_residues)} pore residues + 10 Å neighborhood.")
```

- [ ] **Step 2: Run cells 0–9 in sequence and verify prep output**

After running all cells up through the new prep cell, expect:
```
Pore sidechain prep (10 Å neighborhood repacking)...
  Packed 30 pore residues + 10 Å neighborhood.
```
(30 = 6 chains × 5 residues Ile36–Leu40)

Confirm no exceptions. The docking loop cell (now index 12) can be run independently after this.

- [ ] **Step 3: Commit**

```bash
git add EutM_Docking.ipynb
git commit -m "fix: add pore sidechain prep before EutM docking for comparability with PduA/PduJ"
```

---

## Self-Review Notes

**Spec coverage:**
- Fix 1 (audit): Task 1 ✓
- Fix 2 (PduJ S39/K36 block): Tasks 2–5 ✓ (CALL 1+2 in Task 2, CALL 3+4 in Task 3, CALL 5 in Task 4, CALL 6 in Task 5)
- Fix 3 (EutM prep): Task 6 ✓

**Dependency note:** Task 5 (CALL 6 Spearman) requires PduA CALL 5 cell (cell 29) to have run first so `_build_spearman_matrix`, `_plot_spearman_heatmap`, and `PHYS_PROPS` are in scope. The implementer should run the full notebook top-to-bottom before testing CALL 6.

**Data availability note:** Tasks 3–5 for PduJ require PduJ docking data under `PduJ_docking_mutants/`. If that directory doesn't exist yet, those cells skip gracefully via the `_score_files_j` guard. Task 6 additionally requires `PDUJ_COMBINED_CSV` to exist.
