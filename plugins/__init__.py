"""Plugin package bootstrap logic."""

from pathlib import Path
import sys

_pkg_root = Path(__file__).resolve().parent
_repo_root = _pkg_root.parent
_project_parent = _repo_root.parent

# Ensure the repository parent is on sys.path so `cogbenchmark` imports succeed
# even when entry points execute as loose scripts from the repo root.
if str(_project_parent) not in sys.path:
	sys.path.insert(0, str(_project_parent))
