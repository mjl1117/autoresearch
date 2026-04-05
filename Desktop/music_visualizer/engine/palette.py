from __future__ import annotations
import colorsys
from enum import Enum, auto
import numpy as np


class KeyCharacter(Enum):
    MAJOR = auto()
    MINOR = auto()
    ATONAL = auto()


# Circle of Fifths: maps pitch class (C=0..B=11) → hue degrees
_COF_HUE: dict[int, float] = {
    0: 0.0,    # C
    7: 30.0,   # G
    2: 60.0,   # D
    9: 90.0,   # A
    4: 120.0,  # E
    11: 150.0, # B
    6: 180.0,  # F#
    1: 210.0,  # C#
    8: 240.0,  # Ab
    3: 270.0,  # Eb
    10: 300.0, # Bb
    5: 330.0,  # F
}

_MINOR_THIRD = 3
_MAJOR_THIRD = 4


def _classify_key(chroma: np.ndarray) -> tuple[int, KeyCharacter]:
    """Return (root_pitch_class, KeyCharacter)."""
    root = int(np.argmax(chroma))
    minor_strength = chroma[(root + _MINOR_THIRD) % 12]
    major_strength = chroma[(root + _MAJOR_THIRD) % 12]
    flatness = float(np.std(chroma))
    if flatness < 0.12:
        return root, KeyCharacter.ATONAL
    if major_strength > minor_strength:
        return root, KeyCharacter.MAJOR
    return root, KeyCharacter.MINOR


def _hue_to_rgb(h_deg: float, s: float, l: float) -> tuple[float, float, float]:
    r, g, b = colorsys.hls_to_rgb(h_deg / 360.0, l, s)
    return (r, g, b)


def _lerp_color(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


class Palette:
    def get_palette(
        self,
        chroma: np.ndarray,
        prev_color_a: tuple[float, float, float] | None = None,
        alpha: float = 1.0,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], KeyCharacter]:
        """
        Return (color_a, color_b, KeyCharacter) from a 12-bin chroma vector.
        alpha: blend weight toward new palette (0 = keep prev, 1 = full new).
        """
        root, char = _classify_key(chroma)
        hue = _COF_HUE[root]

        if char == KeyCharacter.MAJOR:
            # Complementary: primary hue + complement (+180)
            color_a = _hue_to_rgb(hue, 0.85, 0.55)
            color_b = _hue_to_rgb((hue + 180.0) % 360.0, 0.75, 0.60)
        elif char == KeyCharacter.MINOR:
            # Analogous cool: primary hue + adjacent (-30)
            color_a = _hue_to_rgb(hue, 0.70, 0.45)
            color_b = _hue_to_rgb((hue - 30.0) % 360.0, 0.65, 0.50)
        else:  # ATONAL
            # Neutral cool desaturated
            color_a = _hue_to_rgb(210.0, 0.25, 0.45)
            color_b = _hue_to_rgb(220.0, 0.20, 0.40)

        if prev_color_a is not None and alpha < 1.0:
            color_a = _lerp_color(prev_color_a, color_a, alpha)

        return color_a, color_b, char
