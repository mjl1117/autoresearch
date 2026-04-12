"""
train_generative.py
Transfer-learning V/A trainer for generative music.

Combines two feedback sources:
  1. Labeled gesture ratings   (item_type='gesture' in ratings.jsonl)
  2. Labeled chord gesture ratings (item_type='chord', matched to
     chord_gesture_log.jsonl by gesture name)

Chord gesture samples receive CHORD_WEIGHT (default 2.0) to emphasise the
generative music domain over the base gesture domain.  This implements a
lightweight transfer-learning strategy: broad coverage from gesture data,
fine-grained calibration from chord gesture data.

Features use the full 37-element schema from gesture_features.py, including
BPM, rhythm statistics, and harmonic-transition deltas — critical for arousal.

Saved models
────────────
  models/valence_generative.pkl
  models/arousal_generative.pkl
  (optionally replaces valence_audio_only.pkl / arousal_audio_only.pkl)

Usage
─────
    python src/machine_learning/train_generative.py
    python src/machine_learning/train_generative.py --no-replace
    python src/machine_learning/train_generative.py --chord-weight 3.0

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
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# gesture_features.py lives in the same directory
sys.path.insert(0, str(Path(__file__).parent))
from gesture_features import extract_gesture_features, N_FEATURES

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
logger = logging.getLogger(__name__)

ROOT            = Path(__file__).parent.parent.parent
FEEDBACK_PATH   = ROOT / 'data' / 'feedback' / 'ratings.jsonl'
CHORD_LOG_PATH  = ROOT / 'data' / 'feedback' / 'chord_gesture_log.jsonl'
GESTURE_DIR     = ROOT / 'data' / 'gesture_library'
MODELS_DIR      = ROOT / 'models'

N_TREES        = 100
RIDGE_ALPHA    = 10.0
RF_MIN_SAMPLES = 80


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sanitise(name: str) -> str:
    s = re.sub(r'[^\w\s\-]', '', name).strip()
    s = re.sub(r'\s+', '_', s)
    return s[:64] or 'gesture'


def _gesture_file(name: str) -> Path | None:
    """Resolve a gesture name to its JSON path."""
    stem = _sanitise(name)
    for candidate in [
        GESTURE_DIR / f'{stem}.json',
        GESTURE_DIR / 'generated_gestures' / f'{stem}.json',
        *(
            [GESTURE_DIR / 'generated_gestures' / f'gesture_{stem[len("synthetic_"):]}.json']
            if stem.startswith('synthetic_') else []
        ),
    ]:
        if candidate.exists():
            return candidate
    return None


def _load_chord_log() -> dict[str, dict]:
    """Return {gesture_name: gesture_dict} from chord_gesture_log.jsonl."""
    if not CHORD_LOG_PATH.exists():
        return {}
    index: dict[str, dict] = {}
    with open(CHORD_LOG_PATH, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                name = rec.get('name', '')
                if name:
                    index[name] = rec
            except json.JSONDecodeError:
                pass
    return index


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_combined_dataset(
    chord_weight: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Load gesture + chord gesture ratings and extract 37-feature vectors.

    Parameters
    ----------
    chord_weight : float
        Sample weight assigned to chord gesture ratings (gesture ratings = 1.0).

    Returns
    -------
    X            : (n, 37) feature matrix
    Y            : (n, 2) [valence, arousal]
    W            : (n,) sample weights
    names        : gesture/chord names (for LOO labelling)
    sources      : 'gesture' | 'chord' per sample (for per-source CV reporting)
    """
    if not FEEDBACK_PATH.exists():
        raise FileNotFoundError(f'Ratings file not found: {FEEDBACK_PATH}')

    chord_log = _load_chord_log()

    # ── Accumulate per-item average ratings ──────────────────────────────────
    accum: dict[str, dict] = {}
    with open(FEEDBACK_PATH, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            itype = r.get('item_type', '')
            if itype not in ('gesture', 'chord'):
                continue
            iid = r['item_id']
            if iid not in accum:
                accum[iid] = {'v': 0.0, 'a': 0.0, 'n': 0, 'type': itype}
            accum[iid]['v'] += float(r['user_valence'])
            accum[iid]['a'] += float(r['user_arousal'])
            accum[iid]['n'] += 1

    X_list, Y_list, W_list, names, sources = [], [], [], [], []
    skipped_gesture = skipped_chord = 0

    for iid, vals in accum.items():
        itype = vals['type']
        n     = vals['n']
        feat  = None

        if itype == 'gesture':
            path = _gesture_file(iid)
            if path is None:
                logger.warning("  skip gesture '%s' — file not found", iid)
                skipped_gesture += 1
                continue
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                feat = extract_gesture_features(data['events'],
                                                bpm=float(data.get('bpm', 80.0)))
            except Exception as e:
                logger.warning("  skip gesture '%s' — %s", iid, e)
                skipped_gesture += 1
                continue

        elif itype == 'chord':
            rec = chord_log.get(iid)
            if rec is None:
                logger.warning("  skip chord '%s' — not in chord gesture log", iid)
                skipped_chord += 1
                continue
            feat = extract_gesture_features(rec['events'],
                                            bpm=float(rec.get('bpm', 80.0)))

        if feat is None:
            logger.warning("  skip '%s' — no valid events", iid)
            if itype == 'gesture':
                skipped_gesture += 1
            else:
                skipped_chord += 1
            continue

        X_list.append(feat)
        Y_list.append([vals['v'] / n, vals['a'] / n])
        W_list.append(chord_weight if itype == 'chord' else 1.0)
        names.append(iid)
        sources.append(itype)

    if skipped_gesture:
        logger.info('Skipped %d gesture items.', skipped_gesture)
    if skipped_chord:
        logger.info('Skipped %d chord items (not in log — likely pre-generative ratings).', skipped_chord)

    return (np.array(X_list), np.array(Y_list),
            np.array(W_list), names, sources)


# ── Cross-validation ──────────────────────────────────────────────────────────

def _loo_cv(X: np.ndarray, y: np.ndarray, w: np.ndarray,
            build_model) -> dict:
    """Weighted leave-one-out CV. Returns R² and MAE."""
    n = len(X)
    preds = np.zeros(n)
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        m = build_model()
        m.fit(X[idx], y[idx], **_sw_kwargs(m, w[idx]))
        preds[i] = m.predict(X[[i]])[0]
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2  = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    mae = float(np.mean(np.abs(y - preds)))
    return {'r2': round(r2, 3), 'mae': round(mae, 2), 'n': n}


def _sw_kwargs(model, w: np.ndarray) -> dict:
    """Return sample_weight kwarg for Pipeline.fit() if the final estimator supports it."""
    try:
        final = model.steps[-1][1]
        final.fit.__doc__  # probe existence
        return {f'{model.steps[-1][0]}__sample_weight': w}
    except Exception:
        return {}


# ── Training ──────────────────────────────────────────────────────────────────

def train(replace: bool = True, chord_weight: float = 2.0) -> None:
    logger.info('Loading combined dataset (gestures + chord gestures) …')
    X, Y, W, names, sources = load_combined_dataset(chord_weight=chord_weight)
    n          = len(X)
    n_gesture  = sources.count('gesture')
    n_chord    = sources.count('chord')

    logger.info('Dataset: %d samples (%d gesture, %d chord gesture)',
                n, n_gesture, n_chord)

    if n < 3:
        logger.error('Need ≥3 samples to train.  Collected %d.', n)
        sys.exit(1)

    use_rf = n >= RF_MIN_SAMPLES
    if not use_rf:
        logger.info('%d samples — using Ridge(α=%.0f). Switch to RF at %d samples.',
                    n, RIDGE_ALPHA, RF_MIN_SAMPLES)

    def _make_ridge():
        return Pipeline([('scaler', StandardScaler()),
                         ('ridge', Ridge(alpha=RIDGE_ALPHA))])

    def _make_rf():
        return Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(
                n_estimators=N_TREES, max_depth=5, random_state=42, n_jobs=-1)),
        ])

    build       = _make_rf if use_rf else _make_ridge
    model_label = (f'RandomForest(depth=5, trees={N_TREES})'
                   if use_rf else f'Ridge(α={RIDGE_ALPHA})')

    print(f"\n{'═'*62}")
    print(f"  Generative V/A Model  ({model_label})")
    print(f"{'═'*62}")
    print(f"  Samples:  {n}  ({n_gesture} gesture, {n_chord} chord gesture)")
    print(f"  Features: {X.shape[1]}  |  chord_weight: {chord_weight}×")
    print(f"  Valence  [{Y[:,0].min():.1f}, {Y[:,0].max():.1f}]  "
          f"mean={Y[:,0].mean():.1f}")
    print(f"  Arousal  [{Y[:,1].min():.1f}, {Y[:,1].max():.1f}]  "
          f"mean={Y[:,1].mean():.1f}")

    # LOO-CV on the full combined set
    print(f"\n  Leave-One-Out CV (combined, weighted):")
    for i, target in enumerate(['valence', 'arousal']):
        cv = _loo_cv(X, Y[:, i], W, build)
        print(f"    {target:8s}  R²={cv['r2']:+.3f}  MAE={cv['mae']:.1f}  "
              f"(n={cv['n']})")

    # Per-source CV breakdowns (unweighted, for diagnostic purposes)
    for src in ('gesture', 'chord'):
        idx = [i for i, s in enumerate(sources) if s == src]
        if len(idx) < 3:
            continue
        Xs, Ys, Ws = X[idx], Y[idx], W[idx]
        print(f"\n  LOO CV — {src} subset only (n={len(idx)}):")
        for i, target in enumerate(['valence', 'arousal']):
            cv = _loo_cv(Xs, Ys[:, i], Ws, build)
            print(f"    {target:8s}  R²={cv['r2']:+.3f}  MAE={cv['mae']:.1f}")

    # Train final models on full dataset with sample weights
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = {}
    for i, target in enumerate(['valence', 'arousal']):
        m = build()
        m.fit(X, Y[:, i], **_sw_kwargs(m, W))
        models[target] = m

    for target, m in models.items():
        path = MODELS_DIR / f'{target}_generative.pkl'
        joblib.dump(m, path)
        logger.info('Saved %s', path)

    if replace:
        for target, m in models.items():
            path = MODELS_DIR / f'{target}_audio_only.pkl'
            joblib.dump(m, path)
            logger.info('Replaced %s', path)
        print(f"\n  Models saved to {MODELS_DIR}/ "
              f"(replaced audio_only + generative copies)")
    else:
        print(f"\n  Models saved to {MODELS_DIR}/ "
              f"(generative copies only — audio_only unchanged)")

    print(f"{'═'*62}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train generative V/A models from gesture + chord gesture feedback.'
    )
    parser.add_argument('--no-replace', action='store_true',
                        help='Save as *_generative.pkl only; do not overwrite *_audio_only.pkl')
    parser.add_argument('--chord-weight', type=float, default=2.0,
                        help='Sample weight for chord gesture ratings (default: 2.0)')
    args = parser.parse_args()
    train(replace=not args.no_replace, chord_weight=args.chord_weight)
