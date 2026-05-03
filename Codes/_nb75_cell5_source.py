"""Helpers notebook cell 5 — source of truth is ``7.5_comparison_check_log.ipynb``."""

from __future__ import annotations

import json
from pathlib import Path

_NB75 = Path(__file__).resolve().with_name("7.5_comparison_check_log.ipynb")


def load_cell5_source(nb_path: str | Path | None = None) -> str:
    """Return the concatenated source of the comparison-helper code cell (cell index 5)."""
    path = Path(nb_path) if nb_path is not None else _NB75
    with path.open(encoding="utf-8") as f:
        nb = json.load(f)
    return "".join(nb["cells"][5]["source"])


CELL5 = load_cell5_source()
