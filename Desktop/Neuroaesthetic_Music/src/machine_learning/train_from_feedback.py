"""
train_from_feedback.py
Bootstrap valence/arousal RF models from collected human feedback ratings.

Uses the same 24-feature vector as _gesture_va_prediction() (6 spectral
statistics × mean/std/min/max computed from partial weight arrays), with
user_valence / user_arousal from data/feedback/ratings.jsonl as labels.

Repeated ratings for the same gesture are averaged before training.
Leave-One-Gesture-Out CV is reported as an honest performance estimate.

Saved models replace the audio-only models so the reveal panel uses
human-calibrated predictions.

Usage
─────
    python src/machine_learning/train_from_feedback.py
    python src/machine_learning/train_from_feedback.py --no-replace

MJL Neuroaesthetic Music Research — 2026
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gesture_features import extract_gesture_features, N_FEATURES

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
FEEDBACK_PATH   = ROOT / 'data' / 'feedback' / 'ratings.jsonl'
GESTURE_DIR     = ROOT / 'data' / 'gesture_library'
MODELS_DIR      = ROOT / 'models'
N_TREES         = 100
RIDGE_ALPHA     = 10.0  # best from LOO grid-search on initial 26-gesture dataset
RF_MIN_SAMPLES  = 80    # switch from Ridge to RF once enough data is collected


# ── feature extraction ────────────────────────────────────────────────────────

def _sanitise(name: str) -> str:
    s = re.sub(r'[^\w\s\-]', '', name).strip()
    s = re.sub(r'\s+', '_', s)
    return s[:64] or 'gesture'


def _gesture_file(name: str) -> Path | None:
    """Resolve gesture name to JSON path, handling synthetic_/gesture_ mismatch."""
    stem = _sanitise(name)
    p = GESTURE_DIR / f'{stem}.json'
    if p.exists():
        return p
    gen = GESTURE_DIR / 'generated_gestures'
    p2 = gen / f'{stem}.json'
    if p2.exists():
        return p2
    if stem.startswith('synthetic_'):
        p3 = gen / f'gesture_{stem[len("synthetic_"):]}.json'
        if p3.exists():
            return p3
    return None


def _extract_features(gesture_path: Path) -> np.ndarray | None:
    """Extract 37-element feature vector from a gesture JSON file.

    Delegates to gesture_features.extract_gesture_features, passing the
    gesture's own BPM so temporal features are correctly populated.
    """
    try:
        data = json.loads(gesture_path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return None
    if 'events' not in data:
        return None
    bpm = float(data.get('bpm', 80.0))
    return extract_gesture_features(data['events'], bpm=bpm)


# ── data loading ──────────────────────────────────────────────────────────────

def load_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and average ratings, then extract features.

    Returns (X, Y, gesture_names) where Y.shape == (n, 2) [valence, arousal].
    """
    if not FEEDBACK_PATH.exists():
        raise FileNotFoundError(f"Ratings file not found: {FEEDBACK_PATH}")

    # Accumulate per-gesture ratings
    accum: dict[str, dict] = {}   # item_id → {valence_sum, arousal_sum, count}
    with open(FEEDBACK_PATH, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get('item_type') != 'gesture':
                continue
            iid = r['item_id']
            if iid not in accum:
                accum[iid] = {'v': 0.0, 'a': 0.0, 'n': 0}
            accum[iid]['v'] += float(r['user_valence'])
            accum[iid]['a'] += float(r['user_arousal'])
            accum[iid]['n'] += 1

    X_list, Y_list, names = [], [], []
    skipped = 0
    for name, vals in accum.items():
        path = _gesture_file(name)
        if path is None:
            logger.warning("  skip '%s' — file not found", name)
            skipped += 1
            continue
        feat = _extract_features(path)
        if feat is None:
            logger.warning("  skip '%s' — no valid events", name)
            skipped += 1
            continue
        n = vals['n']
        X_list.append(feat)
        Y_list.append([vals['v'] / n, vals['a'] / n])
        names.append(name)

    if skipped:
        logger.info("Skipped %d gestures (file not found or no events).", skipped)

    return np.array(X_list), np.array(Y_list), names


# ── cross-validation ──────────────────────────────────────────────────────────

def _loo_cv(X: np.ndarray, y: np.ndarray, build_model) -> dict:
    """Leave-one-gesture-out cross-validation. Returns per-target R² summary."""
    n = len(X)
    preds = np.zeros(n)
    for i in range(n):
        idx_train = [j for j in range(n) if j != i]
        model = build_model()
        model.fit(X[idx_train], y[idx_train])
        preds[i] = model.predict(X[[i]])[0]

    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    mae = float(np.mean(np.abs(y - preds)))
    return {'r2': round(r2, 3), 'mae': round(mae, 2), 'n': n}


# ── training ──────────────────────────────────────────────────────────────────

def train(replace: bool = True) -> None:
    logger.info("Loading feedback dataset from %s …", FEEDBACK_PATH)
    X, Y, names = load_dataset()
    n = len(X)
    logger.info("Dataset: %d unique gestures with ratings", n)

    if n < 3:
        logger.error("Need at least 3 gestures to train. Collected %d so far.", n)
        sys.exit(1)

    use_rf = n >= RF_MIN_SAMPLES
    if not use_rf:
        logger.info(
            "%d gestures — using Ridge(α=%.0f). Switch to RF at %d gestures.",
            n, RIDGE_ALPHA, RF_MIN_SAMPLES,
        )

    def _make_ridge():
        return Pipeline([('scaler', StandardScaler()),
                         ('ridge', Ridge(alpha=RIDGE_ALPHA))])

    def _make_rf():
        return Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(
                n_estimators=N_TREES, max_depth=5, random_state=42, n_jobs=-1)),
        ])

    build = _make_rf if use_rf else _make_ridge
    model_label = f'RandomForest(depth=5, trees={N_TREES})' if use_rf else f'Ridge(α={RIDGE_ALPHA})'

    print(f"\n{'═'*58}")
    print(f"  Feedback-Trained V/A Model  ({model_label})")
    print(f"{'═'*58}")
    print(f"  Gestures: {n}   Features: {X.shape[1]}  ({N_FEATURES} expected)")
    print(f"  Valence  range: [{Y[:,0].min():.1f}, {Y[:,0].max():.1f}]  "
          f"mean={Y[:,0].mean():.1f}")
    print(f"  Arousal  range: [{Y[:,1].min():.1f}, {Y[:,1].max():.1f}]  "
          f"mean={Y[:,1].mean():.1f}")

    # LOO-CV
    print(f"\n  Leave-One-Gesture-Out CV:")
    for i, target in enumerate(['valence', 'arousal']):
        cv = _loo_cv(X, Y[:, i], build)
        print(f"    {target:8s}  R²={cv['r2']:+.3f}  MAE={cv['mae']:.1f}  "
              f"(n={cv['n']})")

    # Train final models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = {}
    for i, target in enumerate(['valence', 'arousal']):
        model = build()
        model.fit(X, Y[:, i])
        models[target] = model

    # Save feedback-specific copies always
    for target, model in models.items():
        path = MODELS_DIR / f'{target}_feedback.pkl'
        joblib.dump(model, path)
        logger.info("Saved %s", path)

    # Optionally replace the audio-only models used by the reveal panel
    if replace:
        for target, model in models.items():
            path = MODELS_DIR / f'{target}_audio_only.pkl'
            joblib.dump(model, path)
            logger.info("Replaced %s", path)
        print(f"\n  Models saved to {MODELS_DIR}/ (replaced audio_only + feedback copies)")
    else:
        print(f"\n  Models saved to {MODELS_DIR}/ (feedback copies only — audio_only unchanged)")

    print(f"{'═'*58}\n")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train V/A models from collected human feedback ratings.'
    )
    parser.add_argument(
        '--no-replace', action='store_true',
        help='Save as valence_feedback.pkl only; do not overwrite valence_audio_only.pkl'
    )
    args = parser.parse_args()
    train(replace=not args.no_replace)
