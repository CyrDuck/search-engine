"""conftest.py - Pytest configuration for XJCO3011 Search Engine tests."""
import sys
from pathlib import Path

# Ensure src/ is importable from any test file
sys.path.insert(0, str(Path(__file__).parent / "src"))
