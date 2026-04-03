#!/usr/bin/env python3
"""Run ESMFold locally on HAMP chimera candidate proteins for domain-aware pLDDT analysis.

ESMFold (Lin et al. 2023, Science) provides AF2-comparable per-residue confidence
scores (pLDDT) from the model's output.plddt tensor. Unlike AF2, ESMFold runs
entirely locally with no API dependency, making it suitable for systematic screening
of chimera candidates that lack UniProt accessions.

Domain-aware pLDDT strategy (mirrors screen_hamp_chimeras.py):
  - TM + sensor region (residues 1..hamp_start - 1): EXCLUDED from all confidence
    calculations. Both AF2 and ESMFold give low pLDDT (~30-50) for membrane-embedded
    alpha-helices. This is a known limitation of structure predictors applied to
    TM-HKs, NOT a design concern. Including TM pLDDT in summary scores would
    penalise every valid TM-HK candidate unfairly.

  - HAMP domain (hamp_start..junction_pos + 30): TARGET pLDDT > 70. The chimera
    junction is embedded in this window. Low ESMFold confidence here indicates
    structural ambiguity in the signal-transduction linker that should be resolved
    before synthesis (register correction or junction redesign).

  - Kinase core DHp + CA (junction_pos + 30..end): TARGET pLDDT > 70. Instability
    in the autophosphorylation domain predicts loss of kinase function in the chimera.

ESMFold vs AF2 pLDDT correspondence:
  ESMFold output.plddt is on the 0-1 scale; multiply by 100 for standard pLDDT
  units. Empirically, ESMFold and AF2 pLDDT agree within ~5 units for well-folded
  globular domains. For HAMP coiled-coils, ESMFold tends to be slightly lower
  (~3-8 units) because it lacks AF2's iterative MSA refinement. The 70-unit
  threshold used for AF2 screening is applied directly to ESMFold pLDDT values.

Importable API:
  parse_esmfold_plddt()  — extract per-residue pLDDT from ESMFold output tensor
  domain_plddt_esm()     — compute TM/HAMP/kinase domain stats from per-residue array
  load_hamp_info()       — parse hamp_start per protein ID from hamp_linker_regions.faa
  load_fasta_seqs()      — load FASTA into {id: seq} dict

These functions can be imported by screen_hamp_chimeras.py or downstream analysis
scripts without re-running the ESMFold model.
"""

import argparse
import statistics
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLDDT_THRESHOLD = 70.0     # minimum mean pLDDT for HAMP and kinase core regions
HAMP_DOWNSTREAM = 30       # residues past junction_pos included in HAMP window


# ---------------------------------------------------------------------------
# Importable helpers: FASTA I/O
# ---------------------------------------------------------------------------

def load_fasta_seqs(fasta_path: str | Path) -> dict[str, str]:
    """Load a FASTA file into {sequence_id: sequence} dict.

    The sequence ID is the first whitespace-delimited token of the header line.
    Sequences spanning multiple lines are concatenated.
    """
    seqs: dict[str, str] = {}
    current_id: str | None = None
    buf: list[str] = []
    with open(fasta_path) as fh:
        for raw in fh:
            line = raw.rstrip()
            if line.startswith(">"):
                if current_id is not None:
                    seqs[current_id] = "".join(buf)
                current_id = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
    if current_id is not None:
        seqs[current_id] = "".join(buf)
    return seqs


def load_hamp_info(hamp_fasta: str | Path) -> dict[str, int]:
    """Parse HAMP start position per protein from hamp_linker_regions.faa.

    Header format (written by extract_hamp_linkers.py):
      >{protein_id}_HAMP_{hamp_start} HAMP_start={hamp_start} extracted_residues=...

    Returns {protein_id: hamp_start} with 1-indexed positions.
    """
    info: dict[str, int] = {}
    with open(hamp_fasta) as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            header = line[1:].split()
            seq_id = header[0]                          # e.g. WP_001234567.1_HAMP_98
            pid = seq_id.rsplit("_HAMP_", 1)[0]         # strip _HAMP_NNN suffix
            # Parse HAMP_start=NNN from description fields
            for field in header[1:]:
                if field.startswith("HAMP_start="):
                    try:
                        info[pid] = int(field.split("=")[1])
                    except ValueError:
                        pass
                    break
    return info


# ---------------------------------------------------------------------------
# Importable helpers: ESMFold pLDDT parsing
# ---------------------------------------------------------------------------

def parse_esmfold_plddt(output) -> list[float]:
    """Extract per-residue pLDDT (0–100 scale) from an ESMFold model output.

    ESMFold returns output.plddt as a torch.Tensor of shape (1, L) or (L,) on
    the 0–1 scale. This function converts to the standard 0–100 pLDDT scale
    and returns a plain Python list (1-indexed positions correspond to list
    index 0..L-1 → residue 1..L).

    Parameters
    ----------
    output : ESMFold model output object
        Must have a .plddt attribute (torch.Tensor).

    Returns
    -------
    list[float]
        Per-residue pLDDT values on the 0–100 scale, ordered from residue 1.
    """
    plddt_tensor = output.plddt
    # Remove batch dimension if present
    arr = plddt_tensor.squeeze().cpu().float().tolist()
    if isinstance(arr, float):
        arr = [arr]
    # ESMFold is 0–1; multiply by 100 for standard pLDDT units
    return [v * 100.0 for v in arr]


def domain_plddt_esm(
    plddt: list[float],
    hamp_start: int,
    junction_pos: int,
) -> dict:
    """Compute per-domain pLDDT statistics from an ESMFold per-residue array.

    Mirrors the domain_plddt() function in screen_hamp_chimeras.py so that ESMFold
    and AF2 results can be compared with identical region definitions.

    TM EXCLUSION RATIONALE:
      Residues 1..(hamp_start - 1) span TM1, the periplasmic sensor, and TM2.
      Structure predictors cannot resolve membrane-buried helices reliably in
      the absence of a membrane model — pLDDT in this region (~30-50) reflects
      prediction uncertainty, not structural disorder. Excluding TM pLDDT from
      chimera scoring prevents valid TM-HK candidates from being rejected on
      spurious grounds.

    Parameters
    ----------
    plddt : list[float]
        Per-residue pLDDT (0–100), 0-indexed (plddt[0] = residue 1).
    hamp_start : int
        1-indexed position of HAMP domain N-terminus (from hamp_linker_regions.faa).
    junction_pos : int
        1-indexed chimera junction position (from hamp_chimera_screen.tsv).

    Returns
    -------
    dict with keys:
        plddt_tm     — dict(mean, median, frac_high, n)  [informational only]
        plddt_hamp   — dict(mean, median, frac_high, n)  [target > 70]
        plddt_kinase — dict(mean, median, frac_high, n)  [target > 70]
    """
    L = len(plddt)

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"mean": float("nan"), "median": float("nan"),
                    "frac_high": float("nan"), "n": 0}
        mean   = sum(vals) / len(vals)
        med    = statistics.median(vals)
        frac   = sum(1 for v in vals if v >= PLDDT_THRESHOLD) / len(vals)
        return {
            "mean":      round(mean, 2),
            "median":    round(med, 2),
            "frac_high": round(frac, 3),
            "n":         len(vals),
        }

    # Convert 1-indexed domain boundaries to 0-indexed list slices
    tm_end        = hamp_start - 1          # residues 1..(hamp_start-1), 0-idx: 0..tm_end-1
    hamp_end_0idx = junction_pos + HAMP_DOWNSTREAM - 1   # 0-indexed inclusive

    tm_vals     = plddt[0 : tm_end - 1]                if tm_end > 1  else []
    hamp_vals   = plddt[hamp_start - 1 : hamp_end_0idx + 1]
    kinase_vals = plddt[hamp_end_0idx + 1 : L]

    return {
        "plddt_tm":     _stats(tm_vals),
        "plddt_hamp":   _stats(hamp_vals),
        "plddt_kinase": _stats(kinase_vals),
    }


# ---------------------------------------------------------------------------
# PDB writing via ESMFold / transformers
# ---------------------------------------------------------------------------

def write_pdb(output, tokenizer, pdb_path: Path) -> bool:
    """Write ESMFold structure output to a PDB file.

    Uses convert_outputs_to_pdb from transformers if available (transformers >=
    4.31). Falls back to a no-op (returns False) so the pLDDT TSV is still
    written even if the PDB conversion helper is not present.

    Returns True if a PDB file was written, False otherwise.
    """
    try:
        from transformers.models.esm.openfold_utils.protein import to_pdb, Protein
        from transformers import convert_outputs_to_pdb  # type: ignore[attr-defined]
        pdb_strings = convert_outputs_to_pdb([output])
        pdb_path.write_text(pdb_strings[0])
        return True
    except (ImportError, AttributeError, TypeError):
        pass

    # Second attempt: use the openfold protein utility directly from the output
    try:
        from transformers.models.esm.openfold_utils.protein import to_pdb
        import torch

        final_atom_positions = output.positions[-1].squeeze(0)   # (L, 37, 3)
        final_atom_mask      = output.atom37_atom_exists.squeeze(0)  # (L, 37)
        aatype               = output.aatype.squeeze(0)               # (L,)
        residue_index        = (
            torch.arange(final_atom_positions.shape[0]) + 1
        )
        b_factors = (
            output.plddt.squeeze(0).unsqueeze(-1)
            .expand(-1, 37)
            .mul(100.0)
            .cpu()
            .numpy()
        )

        from transformers.models.esm.openfold_utils.protein import Protein
        protein = Protein(
            atom_positions=final_atom_positions.cpu().numpy(),
            aatype=aatype.cpu().numpy(),
            atom_mask=final_atom_mask.cpu().numpy(),
            residue_index=residue_index.numpy(),
            b_factors=b_factors,
            chain_index=None,
        )
        pdb_path.write_text(to_pdb(protein))
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device(device_arg: str) -> "torch.device":
    """Return a torch.device based on --device argument.

    'auto' → MPS if Apple Silicon and torch.backends.mps.is_available(),
              else CUDA if available, else CPU.
    'mps'  → MPS (raises RuntimeError if not available).
    'cuda' → CUDA (raises RuntimeError if not available).
    'cpu'  → CPU (always available, slow for long sequences).
    """
    import torch

    if device_arg == "auto":
        # MPS skipped: ESMFold folding trunk has scatter_add/gather ops that are
        # not fully supported on MPS, causing "Placeholder storage has not been
        # allocated on MPS device!" for every sequence. CPU is reliable.
        if torch.cuda.is_available():
            print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
            return torch.device("cuda")
        print("Device: CPU (MPS skipped — ESMFold folding trunk incompatible with MPS)")
        return torch.device("cpu")

    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available on this system.")
        return torch.device("mps")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device found.")
        return torch.device("cuda")

    return torch.device("cpu")


# ---------------------------------------------------------------------------
# ESMFold inference
# ---------------------------------------------------------------------------

def _parse_plddt_from_pdb_string(pdb_string: str) -> list[float]:
    """Extract per-residue pLDDT from ESMFold PDB B-factor column.

    ESMFold writes pLDDT (0–100) into the B-factor column. Takes the CA atom
    for each residue as the representative value, indexed by residue number.
    """
    plddt_per_res: dict[int, float] = {}
    for line in pdb_string.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        try:
            res_num  = int(line[22:26].strip())
            b_factor = float(line[60:66].strip())
            plddt_per_res[res_num] = b_factor
        except ValueError:
            continue
    if not plddt_per_res:
        return []
    max_res = max(plddt_per_res.keys())
    return [plddt_per_res.get(i, float("nan")) for i in range(1, max_res + 1)]


def run_esmfold_on_sequence(
    sequence: str,
    model,
    tokenizer,  # kept for API compatibility; model.infer handles tokenization internally
    device,     # unused — model device is set before calling; kept for API compatibility
) -> tuple[list[float], str]:
    """Run ESMFold via model.infer() — gets pLDDT directly from tensor output.

    Uses model.infer([seq]) instead of the tokenizer → model(**inputs) path to
    avoid a transformers version incompatibility where the tokenizer's padding
    logic raises:
        TypeError: can't multiply sequence by non-int of type 'float'

    Also avoids parsing pLDDT from the PDB B-factor column, where some
    transformers versions write 0–1 fractions rather than 0–100 values.

    Returns (plddt_list, pdb_string) where plddt_list is per-residue pLDDT
    on the 0–100 scale from the output.plddt tensor.
    """
    import torch

    with torch.no_grad():
        output = model.infer([sequence])

    # output.plddt is nominally (batch=1, L) on 0–1 scale, but some transformers
    # versions add a trailing dimension (L, 1) or (L, 37). Flatten to 1D first.
    plddt_tensor = output.plddt[0].cpu().float() * 100.0
    if plddt_tensor.dim() > 1:
        plddt_tensor = plddt_tensor.mean(dim=-1)   # (L, D) → (L,) via mean over last dim
    plddt = plddt_tensor.tolist()

    # Generate PDB string via the model's own converter (same path as infer_pdb)
    pdb_strings = model.output_to_pdb(output)
    pdb_string = pdb_strings[0]

    return plddt, pdb_string


# ---------------------------------------------------------------------------
# Screen TSV annotation
# ---------------------------------------------------------------------------

def annotate_screen_with_esm(
    screen_df: pd.DataFrame,
    plddt_df: pd.DataFrame,
    hamp_info: dict[str, int],
    plddt_threshold: float = PLDDT_THRESHOLD,
) -> pd.DataFrame:
    """Merge ESMFold pLDDT stats into the chimera screen TSV.

    For each candidate pair (acceptor, donor), looks up ESMFold pLDDT values
    and computes domain-aware stats using junction_pos_a / junction_pos_b from
    the screen TSV. Adds esm_-prefixed columns and an esm_structural_ok flag.

    Mirrors the AF2-based column naming in screen_hamp_chimeras.py for
    side-by-side comparison.
    """
    # Build per-protein lookup: {protein_id: {mean_plddt, plddt_per_residue, ...}}
    plddt_lookup: dict[str, dict] = {}
    for _, row in plddt_df.iterrows():
        pid = row["protein_id"]
        per_res_str = row.get("plddt_per_residue", "")
        if pd.isna(per_res_str) or per_res_str == "":
            plddt_lookup[pid] = None
        else:
            try:
                per_res = [float(x) for x in str(per_res_str).split(",")]
            except ValueError:
                plddt_lookup[pid] = None
                continue
            plddt_lookup[pid] = {
                "mean_plddt":   row.get("mean_plddt"),
                "median_plddt": row.get("median_plddt"),
                "per_residue":  per_res,
            }

    esm_rows = []
    for _, row in screen_df.iterrows():
        rec: dict = {}
        for role in ("acceptor", "donor"):
            pid      = row[role]
            j_col    = "junction_pos_a" if role == "acceptor" else "junction_pos_b"
            j_pos    = int(row[j_col]) if j_col in row.index and pd.notna(row[j_col]) else None
            h_start  = hamp_info.get(pid)

            data = plddt_lookup.get(pid)
            if data is None or j_pos is None or h_start is None:
                rec[f"esm_{role}_mean_plddt"]      = float("nan")
                rec[f"esm_{role}_plddt_hamp"]      = float("nan")
                rec[f"esm_{role}_plddt_hamp_n"]    = 0
                rec[f"esm_{role}_plddt_kinase"]    = float("nan")
                rec[f"esm_{role}_plddt_kinase_n"]  = 0
                rec[f"esm_{role}_hamp_highconf"]   = False
                rec[f"esm_{role}_kinase_highconf"] = False
            else:
                per_res = data["per_residue"]
                rec[f"esm_{role}_mean_plddt"] = round(data["mean_plddt"], 2)
                stats = domain_plddt_esm(per_res, h_start, j_pos)
                rec[f"esm_{role}_plddt_hamp"]      = stats["plddt_hamp"]["mean"]
                rec[f"esm_{role}_plddt_hamp_n"]    = stats["plddt_hamp"]["n"]
                rec[f"esm_{role}_plddt_kinase"]    = stats["plddt_kinase"]["mean"]
                rec[f"esm_{role}_plddt_kinase_n"]  = stats["plddt_kinase"]["n"]
                rec[f"esm_{role}_hamp_highconf"] = (
                    stats["plddt_hamp"]["mean"] >= plddt_threshold
                    if stats["plddt_hamp"]["n"] > 0 else False
                )
                rec[f"esm_{role}_kinase_highconf"] = (
                    stats["plddt_kinase"]["mean"] >= plddt_threshold
                    if stats["plddt_kinase"]["n"] > 0 else False
                )
        esm_rows.append(rec)

    esm_df = pd.DataFrame(esm_rows, index=screen_df.index)
    out = pd.concat([screen_df, esm_df], axis=1)

    out["esm_structural_ok"] = (
        out["esm_acceptor_hamp_highconf"].fillna(False)
        & out["esm_acceptor_kinase_highconf"].fillna(False)
        & out["esm_donor_hamp_highconf"].fillna(False)
        & out["esm_donor_kinase_highconf"].fillna(False)
    )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--screen_tsv", required=True,
        help="results/chimera_screen/hamp_chimera_screen.tsv",
    )
    parser.add_argument(
        "--hk_reps", required=True,
        help="results/representatives/hk_reps.faa",
    )
    parser.add_argument(
        "--hamp_fasta", required=True,
        help="results/deepcoil/hamp_linker_regions.faa "
             "(provides hamp_start per protein for domain analysis)",
    )
    parser.add_argument(
        "--outdir", required=True,
        help="Output directory for PDB files and pLDDT TSV",
    )
    parser.add_argument(
        "--output_screen", required=True,
        help="Path for annotated screen TSV (hamp_chimera_screen_esm.tsv)",
    )
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Compute device. 'auto' selects MPS > CUDA > CPU (default: auto)",
    )
    parser.add_argument(
        "--max_length", type=int, default=800,
        help="Skip proteins longer than this many amino acids (default: 800). "
             "ESMFold memory scales quadratically with sequence length.",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Skip proteins whose PDB file already exists in --outdir "
             "(resume interrupted runs without recomputing).",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plddt_tsv = outdir / "esmfold_plddt.tsv"

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    print(f"Loading screen TSV: {args.screen_tsv}")
    screen_df = pd.read_csv(args.screen_tsv, sep="\t")

    print(f"Loading HK representative sequences: {args.hk_reps}")
    hk_seqs = load_fasta_seqs(args.hk_reps)

    print(f"Loading HAMP domain positions: {args.hamp_fasta}")
    hamp_info = load_hamp_info(args.hamp_fasta)
    print(f"  HAMP start positions for {len(hamp_info)} proteins")

    # Collect unique protein IDs from acceptor + donor columns
    unique_pids = sorted(
        set(screen_df["acceptor"].tolist()) | set(screen_df["donor"].tolist())
    )
    print(f"\n{len(unique_pids)} unique protein IDs across acceptor + donor columns")

    # ------------------------------------------------------------------
    # 2. Filter: length cap and batch-skip
    # ------------------------------------------------------------------
    to_run: list[tuple[str, str]] = []
    recover_from_pdb: list[tuple[str, str, int]] = []  # (pid, pdb_path, seq_len)
    skipped_len  = 0
    skipped_missing = 0
    skipped_batch = 0

    for pid in unique_pids:
        seq = hk_seqs.get(pid)
        if seq is None:
            print(f"  [miss] {pid}: not found in hk_reps — skipping")
            skipped_missing += 1
            continue
        if len(seq) > args.max_length:
            print(f"  [skip] {pid}: {len(seq)} aa > max_length {args.max_length}")
            skipped_len += 1
            continue
        pdb_out = outdir / f"{pid}.pdb"
        if args.batch and pdb_out.exists() and pdb_out.stat().st_size > 100:
            skipped_batch += 1
            recover_from_pdb.append((pid, str(pdb_out), len(seq)))
            continue
        to_run.append((pid, seq))

    print(f"\nProteins to run ESMFold:     {len(to_run)}")
    print(f"  Skipped (length > {args.max_length}): {skipped_len}")
    print(f"  Skipped (not in hk_reps):   {skipped_missing}")
    if args.batch:
        print(f"  Skipped (PDB exists):       {skipped_batch}")

    # ------------------------------------------------------------------
    # 3. Load ESMFold model (once, before the loop)
    # ------------------------------------------------------------------
    # Load existing results if we are in batch mode — they will be merged at end
    existing_rows: list[dict] = []
    already_done: set[str] = set()
    if args.batch and plddt_tsv.exists():
        existing_df = pd.read_csv(plddt_tsv, sep="\t")
        already_done = set(existing_df["protein_id"].tolist())
        existing_rows = existing_df.to_dict("records")
        existing_rows = [r for r in existing_rows
                         if r["protein_id"] not in {pid for pid, _ in to_run}]
        print(f"  Loaded {len(existing_rows)} existing results from {plddt_tsv}")

    # Recover pLDDT from existing PDB files for batch-skipped proteins.
    # ESMFold (via infer_pdb) writes pLDDT as 0–1 fractions in the B-factor
    # column; model.infer() gives the same scale. Detect and correct to 0–100.
    if recover_from_pdb:
        print(f"\nRecovering pLDDT from {len(recover_from_pdb)} existing PDB files ...")
        n_recovered = 0
        for pid, pdb_path, seq_len in recover_from_pdb:
            if pid in already_done:
                continue
            pdb_text = Path(pdb_path).read_text()
            plddt = _parse_plddt_from_pdb_string(pdb_text)
            if not plddt:
                print(f"  [warn] {pid}: no CA atoms found in PDB — skipping")
                continue
            # Scale to 0–100 if values are clearly on 0–1 scale
            if max(v for v in plddt if v == v) <= 1.0:  # nan-safe max check
                plddt = [v * 100.0 for v in plddt]
            mean_plddt   = round(sum(plddt) / len(plddt), 3)
            median_plddt = round(statistics.median(plddt), 3)
            per_res_str  = ",".join(f"{v:.3f}" for v in plddt)
            existing_rows.append({
                "protein_id":        pid,
                "sequence_length":   seq_len,
                "mean_plddt":        mean_plddt,
                "median_plddt":      median_plddt,
                "plddt_per_residue": per_res_str,
            })
            n_recovered += 1
        print(f"  Recovered pLDDT for {n_recovered} proteins from existing PDB files")

    new_rows: list[dict] = []

    if to_run:
        device = select_device(args.device)

        print("\nLoading ESMFold model (facebook/esmfold_v1) ...")
        print("  This downloads ~2.7 GB on first use and may take a minute.")
        try:
            from transformers import EsmForProteinFolding, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
            model = EsmForProteinFolding.from_pretrained(
                "facebook/esmfold_v1",
                low_cpu_mem_usage=True,
            )
        except ImportError as exc:
            raise SystemExit(
                "transformers >= 4.31 is required. "
                "Install with: pip install transformers[torch]"
            ) from exc

        # Move model to device. Note: ESMFold's folding trunk uses float32;
        # MPS does not support all operations needed, so fall back to CPU for
        # the folding trunk while keeping the LM on MPS.
        import torch

        str_device = str(device)
        if str_device == "mps":
            # ESMFold folding trunk has ops (scatter_add with float) that are
            # not yet fully supported on MPS (PyTorch < 2.3). Keep trunk on CPU
            # to avoid silent numerical errors; LM encoder runs on MPS.
            print("  Note: ESMFold folding trunk moved to CPU for MPS compatibility.")
            print("  The LM encoder will run on MPS; folding trunk on CPU.")
            model = model.to("cpu")
            model.esm = model.esm.to(device)
        else:
            model = model.to(device)

        model.eval()

        # ------------------------------------------------------------------
        # 4. Run ESMFold per protein
        # ------------------------------------------------------------------
        print(f"\nRunning ESMFold on {len(to_run)} proteins ...")
        for i, (pid, seq) in enumerate(to_run, start=1):
            print(f"  [{i}/{len(to_run)}] {pid} ({len(seq)} aa) ...", end=" ", flush=True)
            pdb_out = outdir / f"{pid}.pdb"

            # For MPS, run inference on CPU (trunk is on CPU, LM on MPS)
            infer_device = device if str_device != "mps" else torch.device("cpu")

            try:
                plddt_list, pdb_string = run_esmfold_on_sequence(
                    seq, model, tokenizer, infer_device
                )
            except Exception as exc:
                print(f"ERROR: {exc}")
                new_rows.append({
                    "protein_id":       pid,
                    "sequence_length":  len(seq),
                    "mean_plddt":       float("nan"),
                    "median_plddt":     float("nan"),
                    "plddt_per_residue": "",
                })
                continue

            mean_plddt   = round(sum(plddt_list) / len(plddt_list), 3)
            median_plddt = round(statistics.median(plddt_list), 3)
            per_res_str  = ",".join(f"{v:.3f}" for v in plddt_list)

            pdb_out.write_text(pdb_string)
            print(f"mean={mean_plddt:.1f} | PDB+pLDDT")

            new_rows.append({
                "protein_id":        pid,
                "sequence_length":   len(seq),
                "mean_plddt":        mean_plddt,
                "median_plddt":      median_plddt,
                "plddt_per_residue": per_res_str,
            })
    else:
        print("\nNothing to run (all proteins skipped or already processed).")

    # ------------------------------------------------------------------
    # 5. Write pLDDT TSV
    # ------------------------------------------------------------------
    all_rows = existing_rows + new_rows
    if all_rows:
        plddt_df = pd.DataFrame(all_rows)
        plddt_df.to_csv(plddt_tsv, sep="\t", index=False)
        print(f"\nWritten: {plddt_tsv} ({len(plddt_df)} proteins)")
    else:
        # Write empty TSV so Snakemake output is satisfied
        pd.DataFrame(
            columns=["protein_id", "sequence_length",
                     "mean_plddt", "median_plddt", "plddt_per_residue"]
        ).to_csv(plddt_tsv, sep="\t", index=False)
        print(f"\nWritten: {plddt_tsv} (empty — no proteins processed)")
        plddt_df = pd.read_csv(plddt_tsv, sep="\t")

    # ------------------------------------------------------------------
    # 6. Domain-aware annotation of screen TSV
    # ------------------------------------------------------------------
    print(f"\nAnnotating screen TSV with ESMFold pLDDT ...")
    annotated = annotate_screen_with_esm(screen_df, plddt_df, hamp_info)

    output_screen = Path(args.output_screen)
    output_screen.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(output_screen, sep="\t", index=False)
    print(f"Written: {output_screen}")

    # Summary
    n_esm_ok = int(annotated.get("esm_structural_ok", pd.Series([], dtype=bool)).sum())
    n_total  = len(annotated)
    print(f"\n=== ESMFold annotation summary ({n_total} candidate pairs) ===")
    print(f"  ESMFold structural OK (HAMP + kinase >= {PLDDT_THRESHOLD}): {n_esm_ok}")
    print(f"\n  Note: TM pLDDT intentionally excluded — low TM confidence")
    print(f"        is expected for membrane-embedded HKs and is NOT a")
    print(f"        chimera design concern (see module docstring).")


if __name__ == "__main__":
    main()
