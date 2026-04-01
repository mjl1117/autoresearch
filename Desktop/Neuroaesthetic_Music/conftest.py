"""Pytest configuration for Neuroaesthetic Music project."""
import sys
from pathlib import Path

# Add src directory to Python path at the very beginning
project_root = Path(__file__).parent.absolute()
src_dir = project_root / 'src'
sys.path.insert(0, str(src_dir))
