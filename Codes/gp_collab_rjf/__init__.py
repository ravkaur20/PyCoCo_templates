"""Vendored collaborator 2-D GP (rjf kernel / optimizer / diagnostics).

Import ``run_gp_from_bundle`` from ``gp_collab_rjf.run_inference`` or use the re-export below.
"""

from __future__ import annotations

from .run_inference import DEFAULT_KWARGS, run_gp_from_bundle

__all__ = ["DEFAULT_KWARGS", "run_gp_from_bundle"]
