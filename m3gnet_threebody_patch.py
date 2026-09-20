#!/usr/bin/env python
"""Fix the M3GNet three-body index-space defect in matgl.

The defect
----------
``M3GNet.forward`` builds its line graph with ``create_line_graph``, which first
drops every bond longer than ``threebody_cutoff`` and then numbers the surviving
bonds ``0..n_kept-1``. The triples it returns are expressed in that *pruned*
numbering. ``forward`` then hands those ids to ``ThreeBodyInteractions`` beside
two tensors built over **all** bonds::

    edge_dst_atom     = edge_index[1]                    # full bond space
    three_body_cutoff = polynomial_cutoff(bond_dist, rc) # full bond space
    line_edge_index   = l_g["line_edge_index"]           # PRUNED bond space

The two index spaces disagree, so every triple reads the wrong destination atom
and the wrong cutoff weight, and the angular update is scattered into the wrong
bond rows. No exception is raised: pruning only removes bonds, so a pruned-space
id is always <= its parent id and always in bounds.

The bundle already carries the translation table needed to reconcile them --
``kept_edge_ids`` -- and no released matgl has ever read it.

Precondition: the defect fires only when ``threebody_cutoff < cutoff``. When
they are equal nothing is pruned, ``kept_edge_ids == arange``, and the shipped
code is accidentally correct. So it is model by model: ``M3GNet-Eform-MP-2018.6.1``
(5.0/5.0) is immune; ``M3GNet-Eform-MP-2019.4.1`` and the MatPES PES models
(5.0/4.0) are affected.

Scale of the error, measured on ``M3GNet-PES-MatPES-r2SCAN-2025.2``: on fcc Al
(168 bonds, 48 within the 4.0 A three-body cutoff) 93.9% of triple weights
collapse to exactly zero and 35 of the 48 in-cutoff bonds receive no angular
update at all. Black box: permuting the atom order of a crystal -- same atoms,
same positions, same cell -- moves the predicted energy by up to 9.0 meV/atom.
With this patch installed that permutation test returns float noise.

What is changed
---------------
Two functions, both in ``matgl``, neither inside ``ThreeBodyInteractions``
(that layer's arithmetic is correct; it is fed a violated precondition):

1. ``matgl.graph._compute.create_line_graph`` -- remap the returned bundle into
   full parent-bond space via ``kept_edge_ids``. Every tensor ``forward``
   indexes then shares one space and matgl's existing code becomes
   self-consistent. No edit to ``forward`` is required.

2. ``matgl.utils.maths.get_segment_indices_from_n`` -- the remap makes
   ``n_triple_ij`` length ``num_bonds`` with zeros for bonds in no triple, and
   the stock implementation is wrong for any interior zero and raises
   ``IndexError`` on a trailing one::

       ns           stock                  correct
       [2, 3, 1]    [0,0,1,1,1,2]          same
       [2, 0, 3]    [0,0,1,1,1]  WRONG     [0,0,2,2,2]
       [2, 3, 0]    IndexError             [0,0,1,1,1]

   The replacement is provably identical to the original whenever no segment is
   empty, so it cannot change any result that was previously correct.

Usage
-----
Runtime, no files touched (recommended -- put this before you load a model)::

    import m3gnet_threebody_patch
    m3gnet_threebody_patch.install()

In place, editing the installed matgl so every process is fixed::

    python m3gnet_threebody_patch.py --check     # report status, change nothing
    python m3gnet_threebody_patch.py --apply     # patch, keeping .orig backups
    python m3gnet_threebody_patch.py --revert    # restore from those backups

``--apply`` appends a guarded block to each of the two files above; it never
rewrites existing lines, so it is safe to re-run and easy to read in a diff.

Where to put this file
----------------------
Anywhere on ``sys.path`` -- it patches matgl from outside and never needs to
live inside the matgl tree. Next to your run script is the usual choice. The
``--apply`` mode locates the installed matgl itself, so the file still does not
need to be moved.

Verify with ``--self-test``, which runs the permutation test on whichever
M3GNet you point it at and asserts the energy is invariant.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

__all__ = ["install", "is_installed", "segment_indices_from_n", "to_full_bond_space"]

__version__ = "1.0.0"

#: Marker written into patched files so --check and --revert can recognise them.
MARKER = "M3GNET_THREEBODY_INDEX_FIX"

_INSTALLED = False


# ---------------------------------------------------------------------------
# The fix itself. Both functions are pure and import nothing from matgl, so
# they can be copied into any codebase that needs them.
# ---------------------------------------------------------------------------


def segment_indices_from_n(ns):
    """Segment id of each element, given per-segment counts.

    ``[2, 0, 3] -> [0, 0, 2, 2, 2]``. Correct for empty segments, which the
    stock implementation is not.

    Returns int64: the result indexes ``scatter_add_``, which rejects int32.
    (The original returned int64 too, via ``cumsum``'s integer promotion, so
    this is not a dtype change.)
    """
    import torch

    return torch.repeat_interleave(
        torch.arange(ns.numel(), device=ns.device, dtype=torch.long), ns.long()
    )


def to_full_bond_space(lg, bond_dist, bond_vec, pbc_offset=None):
    """Re-express a line-graph bundle in full parent-bond indices.

    ``kept_edge_ids`` is ascending, so the remap preserves the
    sorted-by-source-bond ordering that the three-body scatter depends on, and
    ``n_triple_ij`` is rebuilt by ``bincount`` so it agrees with
    ``line_edge_index`` by construction.

    Downstream, ``ensure_line_graph_compatibility`` then takes its ``else``
    branch and slices ``bond_dist[:num_bonds]`` -- i.e. all bonds, which is now
    the correct thing to do.
    """
    import torch

    kept = lg["kept_edge_ids"].long()
    num_bonds = int(bond_dist.shape[0])
    lei = lg["line_edge_index"]

    lei_full = kept[lei.long()].to(lei.dtype)

    out = dict(lg)
    out["line_edge_index"] = lei_full
    out["n_triple_ij"] = torch.bincount(lei_full[0].long(), minlength=num_bonds).to(
        lg["n_triple_ij"].dtype
    )
    out["bond_dist"] = bond_dist
    out["bond_vec"] = bond_vec
    if pbc_offset is not None:
        out["pbc_offset"] = pbc_offset
    return out


# ---------------------------------------------------------------------------
# Runtime installation
# ---------------------------------------------------------------------------


def is_installed() -> bool:
    """True if this process has the runtime patch active."""
    return _INSTALLED


def install(verbose: bool = False) -> bool:
    """Patch matgl in this process. Idempotent; True if this call did the work.

    Call before constructing or loading any model. Patching after a model is
    built still works -- both functions are looked up at call time -- but
    patching before keeps the ordering obvious.
    """
    global _INSTALLED
    if _INSTALLED:
        return False

    import matgl.graph._compute as _compute
    import matgl.layers._three_body as _three_body
    import matgl.models._m3gnet as _m3gnet
    import matgl.utils.maths as _maths

    _stock_create = _compute.create_line_graph

    def create_line_graph(
        edge_index, bond_dist, bond_vec, pbc_offset, num_nodes, threebody_cutoff
    ):
        lg = _stock_create(
            edge_index, bond_dist, bond_vec, pbc_offset, num_nodes, threebody_cutoff
        )
        return to_full_bond_space(lg, bond_dist, bond_vec, pbc_offset)

    create_line_graph.__doc__ = _stock_create.__doc__
    create_line_graph.__wrapped__ = _stock_create
    segment_indices_from_n.__wrapped__ = _maths.get_segment_indices_from_n

    # Each name is bound into the namespace of every module that imported it,
    # so patching the defining module alone would not take effect.
    _compute.create_line_graph = create_line_graph
    _m3gnet.create_line_graph = create_line_graph

    _maths.get_segment_indices_from_n = segment_indices_from_n
    _three_body.get_segment_indices_from_n = segment_indices_from_n

    _INSTALLED = True
    if verbose:
        print(f"[{MARKER}] runtime patch installed (matgl {_matgl_version()})")
    return True


def _matgl_version() -> str:
    try:
        import matgl

        return getattr(matgl, "__version__", "unknown")
    except Exception:
        return "not importable"


# ---------------------------------------------------------------------------
# In-place source patching
# ---------------------------------------------------------------------------

_MATHS_PATCH = f'''

# --- {MARKER} -------------------------------------------------------------
# Stock get_segment_indices_from_n is wrong for empty segments: it writes a 1 at
# each boundary and cumsums, so a zero-length segment is skipped rather than
# counted ([2, 0, 3] -> [0,0,1,1,1] instead of [0,0,2,2,2]), and a trailing zero
# indexes out of bounds. Identical to the original whenever no segment is empty.
def get_segment_indices_from_n(ns):  # noqa: F811
    """Get segment indices from a number array (empty-segment safe)."""
    return torch.repeat_interleave(
        torch.arange(ns.numel(), device=ns.device, dtype=torch.long), ns.long()
    )
# --- end {MARKER} ---------------------------------------------------------
'''

_COMPUTE_PATCH = f'''

# --- {MARKER} -------------------------------------------------------------
# create_line_graph numbers its triples over the bonds that survived the
# three-body cutoff, but M3GNet.forward indexes edge_index[1] and
# polynomial_cutoff(bond_dist), which are built over ALL bonds. Remap the bundle
# into full parent-bond space with kept_edge_ids -- the translation table the
# bundle already carries and no released matgl has ever read -- so that every
# tensor forward touches shares one index space.
_stock_create_line_graph = create_line_graph  # noqa: F811


def create_line_graph(  # noqa: F811
    edge_index, bond_dist, bond_vec, pbc_offset, num_nodes, threebody_cutoff
):
    """Build the M3GNet 3-body line graph in full parent-bond index space."""
    lg = _stock_create_line_graph(
        edge_index, bond_dist, bond_vec, pbc_offset, num_nodes, threebody_cutoff
    )
    kept = lg["kept_edge_ids"].long()
    lei = lg["line_edge_index"]
    lei_full = kept[lei.long()].to(lei.dtype)

    lg["line_edge_index"] = lei_full
    lg["n_triple_ij"] = torch.bincount(
        lei_full[0].long(), minlength=int(bond_dist.shape[0])
    ).to(lg["n_triple_ij"].dtype)
    lg["bond_dist"] = bond_dist
    lg["bond_vec"] = bond_vec
    if pbc_offset is not None:
        lg["pbc_offset"] = pbc_offset
    return lg
# --- end {MARKER} ---------------------------------------------------------
'''


def matgl_root() -> Path:
    """Directory of the installed matgl package."""
    import matgl

    return Path(matgl.__file__).resolve().parent


def _targets() -> list[tuple[Path, str]]:
    root = matgl_root()
    return [
        (root / "utils" / "maths.py", _MATHS_PATCH),
        (root / "graph" / "_compute.py", _COMPUTE_PATCH),
    ]


def check() -> int:
    """Report which files are patched. Returns 0 if all are."""
    print(f"matgl {_matgl_version()} at {matgl_root()}")
    n_patched = 0
    for path, _ in _targets():
        if not path.exists():
            print(f"  MISSING   {path}")
            continue
        patched = MARKER in path.read_text()
        n_patched += patched
        print(f"  {'PATCHED  ' if patched else 'unpatched'} {path}")
    total = len(_targets())
    print(f"\n{n_patched}/{total} patched", end="")
    print(" -- run --apply to patch" if n_patched < total else " -- fix is active")
    return 0 if n_patched == total else 1


def apply() -> int:
    """Append the guarded fix to each target file, backing it up first."""
    for path, patch in _targets():
        if not path.exists():
            print(f"MISSING {path} -- matgl layout not recognised, aborting")
            return 2
        text = path.read_text()
        if MARKER in text:
            print(f"already patched  {path}")
            continue
        backup = path.with_suffix(path.suffix + ".orig")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(text + patch)
        print(f"patched          {path}  (backup {backup.name})")
    print("\nDone. Verify with --check, then --self-test.")
    return 0


def revert() -> int:
    """Restore both files from their .orig backups."""
    for path, _ in _targets():
        backup = path.with_suffix(path.suffix + ".orig")
        if backup.exists():
            shutil.copy2(backup, path)
            backup.unlink()
            print(f"reverted  {path}")
        elif path.exists() and MARKER in path.read_text():
            print(f"NO BACKUP for {path} -- reinstall matgl to restore it")
        else:
            print(f"unpatched {path}")
    return 0


# ---------------------------------------------------------------------------
# Self-test: permutation invariance
# ---------------------------------------------------------------------------


def self_test(model: str = "M3GNet-PES-MatPES-r2SCAN-2025.2", tol: float = 1e-4) -> int:
    """Assert relabelling invariance on a real model.

    Permuting the atom order of a structure changes nothing physical, so the
    energy must not move. Under the defect it moves by meV/atom, because the
    pruned-space ids select a different set of bonds once the order changes.
    This is the test to put in front of a skeptic: it needs no notion of index
    spaces, only that atom order is not physics.

    Runs whichever fix is active -- the runtime patch if ``install()`` was
    called, or the in-place edit if ``--apply`` was used.
    """
    import numpy as np
    import torch
    from ase.build import bulk

    import matgl
    from matgl.ext.ase import PESCalculator

    print(f"matgl {_matgl_version()}   runtime patch: {is_installed()}")
    print(f"model {model}\n")

    pot = matgl.load_model(model).eval()
    if not hasattr(pot, "calc_stresses"):
        print(
            f"{model} is a property model, not a Potential, so PESCalculator "
            "cannot wrap it.\nThe permutation test needs a PES model -- try "
            "M3GNet-PES-MatPES-r2SCAN-2025.2. The patch itself applies to both."
        )
        return 2
    tbc = float(pot.model.threebody_cutoff)
    cut = float(pot.model.cutoff)
    print(f"cutoff {cut}  threebody_cutoff {tbc}", end="")
    if tbc >= cut:
        print("  -- equal, so this model is immune to the defect by construction")
    else:
        print("  -- threebody_cutoff < cutoff, so the defect applies")

    rng = np.random.default_rng(0)
    worst = 0.0
    ok = True
    for name, atoms in [
        ("fcc Al", bulk("Al", "fcc", a=4.05, cubic=True) * (2, 2, 2)),
        ("NaCl", bulk("NaCl", "rocksalt", a=5.64, cubic=True) * (2, 2, 2)),
        ("Si rattled", _rattled(bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2))),
    ]:
        calc = PESCalculator(pot)
        atoms.calc = calc
        e0 = atoms.get_potential_energy()

        order = rng.permutation(len(atoms))
        shuffled = atoms[order]
        shuffled.calc = PESCalculator(pot)
        e1 = shuffled.get_potential_energy()

        d_meV = abs(e1 - e0) * 1000.0
        per_atom = d_meV / len(atoms)
        worst = max(worst, per_atom)
        good = per_atom < tol * 1000.0
        ok &= good
        print(
            f"  {name:12s} {len(atoms):3d} atoms   dE = {d_meV:9.4f} meV "
            f"({per_atom:8.4f} meV/atom)  {'ok' if good else 'FAIL'}"
        )

    print(f"\nworst {worst:.4f} meV/atom", end="")
    if ok:
        print(" -- invariant. The fix is active.")
        return 0
    print(" -- NOT invariant. The defect is present; install the fix.")
    return 1


def _rattled(atoms, amplitude: float = 0.1, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    atoms.positions += rng.normal(scale=amplitude, size=atoms.positions.shape)
    return atoms


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with no arguments for --check.",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="report status, change nothing")
    g.add_argument("--apply", action="store_true", help="patch installed matgl in place")
    g.add_argument("--revert", action="store_true", help="restore from .orig backups")
    g.add_argument("--self-test", action="store_true", help="permutation-invariance test")
    p.add_argument("--runtime", action="store_true", help="with --self-test: install() first")
    p.add_argument("--model", default="M3GNet-PES-MatPES-r2SCAN-2025.2", help="model for --self-test")
    args = p.parse_args(argv)

    if args.apply:
        return apply()
    if args.revert:
        return revert()
    if args.self_test:
        if args.runtime:
            install(verbose=True)
        return self_test(args.model)
    return check()


if __name__ == "__main__":
    sys.exit(main())
