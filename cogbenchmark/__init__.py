"""
Lightweight package shim so code importing `cogbenchmark.<subpkg>` can resolve
to the repository's top-level subpackage directories (e.g. `user_simulator`,
`datasets`, `plugins`, etc.) without requiring the project to be installed.

This file inserts the repository root into the package `__path__`, making
`cogbenchmark` a namespace-like package whose subpackages can live at the
project root.
"""
from pathlib import Path
import os

# Ensure repo root (parent of this package dir) is included in the package path
try:
    here = Path(__file__).resolve().parent
    repo_root = str(here.parent)
    # Prepend repo_root so it takes precedence when resolving subpackages
    if repo_root not in __path__:
        __path__.insert(0, repo_root)
except Exception:
    # Best-effort only; if anything fails, downstream imports will raise as usual
    pass
