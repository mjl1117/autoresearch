"""Tests for prepare_training_images.py"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from prepare_training_images import find_tiff_files, select_frame_indices, compute_auto_params


def test_module_imports():
    """Script must import without errors."""
    mod = importlib.import_module("prepare_training_images")
    assert hasattr(mod, "CPSAM_MODEL_PATH")
    assert hasattr(mod, "FALLBACK_PYTHON")


def test_find_tiff_files_flat(tmp_path):
    """Finds .tif and .tiff files in the root directory."""
    (tmp_path / "a.tif").touch()
    (tmp_path / "b.tiff").touch()
    (tmp_path / "ignore.png").touch()
    result = find_tiff_files(tmp_path)
    names = {p.name for p in result}
    assert names == {"a.tif", "b.tiff"}


def test_find_tiff_files_recursive(tmp_path):
    """Finds TIFFs in nested subdirectories."""
    sub = tmp_path / "sub" / "deep"
    sub.mkdir(parents=True)
    (tmp_path / "root.tif").touch()
    (sub / "nested.tif").touch()
    result = find_tiff_files(tmp_path)
    names = {p.name for p in result}
    assert names == {"root.tif", "nested.tif"}


def test_find_tiff_files_empty(tmp_path):
    """Returns empty list when no TIFFs exist."""
    assert find_tiff_files(tmp_path) == []


def test_select_frame_indices_single():
    """Single-frame TIFF returns [0]."""
    assert select_frame_indices(1) == [0]


def test_select_frame_indices_two():
    """Two-frame TIFF returns [0, 1]."""
    assert select_frame_indices(2) == [0, 1]


def test_select_frame_indices_timelapse():
    """Multi-frame TIFF returns first and last only."""
    assert select_frame_indices(20) == [0, 19]


def test_select_frame_indices_three():
    """Three frames: first and last, not middle."""
    assert select_frame_indices(3) == [0, 2]


def test_compute_auto_params_with_pixel_size():
    """Computes radius and sigma from physical units when pixel size is known."""
    params = compute_auto_params(pixel_size_um=0.1, max_bead_diameter_um=50.0)
    assert params["rb_radius"] == pytest.approx(500.0)
    assert params["dog_sigma_high"] == pytest.approx(125.0)


def test_compute_auto_params_fallback():
    """Returns fallback values when pixel size is None."""
    params = compute_auto_params(pixel_size_um=None, max_bead_diameter_um=50.0)
    assert params["rb_radius"] == pytest.approx(50.0)
    assert params["dog_sigma_high"] == pytest.approx(10.0)


def test_compute_auto_params_zero_pixel_size():
    """pixel_size_um=0 is treated as unknown — falls back to defaults."""
    params = compute_auto_params(pixel_size_um=0, max_bead_diameter_um=50.0)
    assert params["rb_radius"] == pytest.approx(50.0)
    assert params["dog_sigma_high"] == pytest.approx(10.0)
