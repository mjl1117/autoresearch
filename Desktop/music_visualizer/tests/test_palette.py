import numpy as np
import pytest
from engine.palette import Palette, KeyCharacter


def test_major_key_returns_vibrant():
    p = Palette()
    chroma = np.zeros(12)
    chroma[0] = 1.0   # C
    chroma[4] = 0.7   # E (major third)
    chroma[7] = 0.6   # G (perfect fifth)
    color_a, color_b, char = p.get_palette(chroma)
    assert char == KeyCharacter.MAJOR
    assert len(color_a) == 3
    assert all(0.0 <= c <= 1.0 for c in color_a)
    assert all(0.0 <= c <= 1.0 for c in color_b)


def test_minor_key_returns_cool():
    p = Palette()
    chroma = np.zeros(12)
    chroma[9] = 1.0   # A
    chroma[0] = 0.7   # C (minor third of Am)
    chroma[4] = 0.5   # E (perfect fifth of Am)
    color_a, color_b, char = p.get_palette(chroma)
    assert char == KeyCharacter.MINOR


def test_atonal_returns_neutral():
    p = Palette()
    chroma = np.ones(12) / 12.0  # flat, ambiguous
    color_a, color_b, char = p.get_palette(chroma)
    assert char == KeyCharacter.ATONAL


def test_interpolated_palette_values_are_valid():
    p = Palette()
    chroma = np.zeros(12)
    chroma[0] = 1.0
    chroma[7] = 0.5
    prev_a = (0.31, 0.76, 0.63)
    color_a, color_b, _ = p.get_palette(chroma, prev_color_a=prev_a, alpha=0.3)
    assert all(0.0 <= c <= 1.0 for c in color_a)
