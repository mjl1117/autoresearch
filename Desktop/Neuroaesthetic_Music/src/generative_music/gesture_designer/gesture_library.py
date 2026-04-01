"""
gesture_library.py
JSON-backed library for saving and loading spectral gestures.

Each gesture is stored as an individual JSON file in the gestures/ directory.

MJL Neuroaesthetic Music Research — 2026
"""
from __future__ import annotations
import json, re
from pathlib import Path
from typing import List, Optional
from .gesture_model import Gesture

# Default library directory (relative to this file's location)
_DEFAULT_DIR = Path(__file__).parent.parent.parent.parent / 'data' / 'gesture_library'


def _sanitise(name: str) -> str:
    """Convert gesture name to a safe filename stem."""
    s = re.sub(r'[^\w\s\-]', '', name).strip()
    s = re.sub(r'\s+', '_', s)
    return s[:64] or 'gesture'


class GestureLibrary:
    """Manages a directory of JSON gesture files."""

    def __init__(self, directory: Optional[Path] = None):
        self.directory = Path(directory) if directory else _DEFAULT_DIR
        self.directory.mkdir(parents=True, exist_ok=True)

    # ── File path helpers ────────────────────────────────────────────────────

    def _path_for(self, name: str) -> Path:
        return self.directory / f'{_sanitise(name)}.json'

    def _unique_path_for(self, name: str) -> Path:
        """Return a path that doesn't already exist (appends _N if needed)."""
        base = _sanitise(name)
        p = self.directory / f'{base}.json'
        if not p.exists():
            return p
        i = 2
        while True:
            p = self.directory / f'{base}_{i}.json'
            if not p.exists():
                return p
            i += 1

    # ── CRUD ────────────────────────────────────────────────────────────────

    def save(self, gesture: Gesture, overwrite: bool = True) -> Path:
        """Save gesture to disk.  Returns the path written."""
        if overwrite:
            path = self._path_for(gesture.name)
        else:
            path = self._unique_path_for(gesture.name)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(gesture.to_dict(), f, indent=2)
        return path

    def load(self, name_or_path: str) -> Optional[Gesture]:
        """Load a gesture by name stem or full path string."""
        p = Path(name_or_path)
        if not p.suffix:
            p = self._path_for(name_or_path)
        if not p.exists():
            return None
        with open(p, 'r', encoding='utf-8') as f:
            return Gesture.from_dict(json.load(f))

    def delete(self, name: str) -> bool:
        """Delete a gesture file.  Returns True if deleted."""
        p = self._path_for(name)
        if p.exists():
            p.unlink()
            return True
        return False

    def rename(self, old_name: str, new_name: str) -> bool:
        old = self._path_for(old_name)
        if not old.exists():
            return False
        old.rename(self._path_for(new_name))
        return True

    # ── Listing ──────────────────────────────────────────────────────────────

    def list_gestures(self) -> List[dict]:
        """Return list of {name, path, bpm, event_count, tags, ratings} dicts, sorted by name."""
        items = []
        for p in sorted(self.directory.glob('*.json')):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                items.append({
                    'name':        d.get('name', p.stem),
                    'path':        str(p),
                    'bpm':         d.get('bpm', 80),
                    'event_count': len(d.get('events', [])),
                    'tags':        d.get('tags', []),
                    'ratings':     d.get('ratings', {}),
                })
            except (json.JSONDecodeError, KeyError):
                pass
        return items

    def load_all(self) -> List[Gesture]:
        return [Gesture.from_dict(json.loads(Path(item['path']).read_text()))
                for item in self.list_gestures()]

    def weighted_random(self, participant_id: str = '') -> Optional[dict]:
        """Draw one gesture dict at random, weighted toward under-explored items.

        Items with no rating for this participant have weight 1.0.
        Items rated r (1–5) have weight (6 - r) / 5, so 5-star items surface
        least often and unrated or low-rated items are explored first.

        Returns None if the library is empty.
        """
        import random as _random
        items = self.list_gestures()
        if not items:
            return None

        weights = []
        for item in items:
            rating = item.get('ratings', {}).get(participant_id)
            if rating is None:
                weights.append(1.0)
            else:
                r = min(5.0, max(1.0, float(rating)))   # clamp to valid range
                weights.append(max(0.1, (6.0 - r) / 5.0))

        return _random.choices(items, weights=weights, k=1)[0]

    # ── Export ───────────────────────────────────────────────────────────────

    def export_python(self, gesture: Gesture) -> str:
        """Generate a Python code snippet that plays the gesture via sc controller."""
        lines = [
            f"# Gesture: {gesture.name}  |  BPM: {gesture.bpm}",
            f"beat = 60.0 / {gesture.bpm}",
            "import time",
            "",
        ]
        for i, ev in enumerate(gesture.events):
            if ev.is_rest:
                lines.append(f"time.sleep({ev.beats} * beat)  # rest")
            else:
                lines.append(f"# Event {i+1}: {ev.pitch_label}")
                params_str = (
                    f"{{'root': {ev.frequency:.3f}, 'amp': {ev.amplitude}, "
                    f"'brightness': {ev.brightness}, "
                    + ', '.join(f"'w{j+1}': {ev.partials.get_index(j):.2f}"
                                for j in range(16))
                    + "}"
                )
                lines.append(f"e{i} = sc.Synth('harmonicSeries', {params_str})")
                lines.append(f"time.sleep({ev.beats} * beat - {ev.release})")
                lines.append(f"e{i}.set('gate', 0)")
                lines.append(f"time.sleep({ev.release})")
            lines.append("")
        return '\n'.join(lines)
