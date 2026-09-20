"""Correctness tests for the shipped package.

    pytest tests/ -v            # everything (loads the models; a few minutes)
    pytest tests/ -v -m fast    # only the tests that need no model weights

These are the claims the package rests on. Each one has failed at least once
during development, which is why it is here rather than in a docstring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from m3gnet_fixed.pruning import (  # noqa: E402
    assert_lg_invariants,
    build_empty_bundle,
    build_fixed_bundle,
    segment_indices_from_n,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# ---------------------------------------------------------------------------
# The segment-index fix, in isolation. No weights needed.
# ---------------------------------------------------------------------------


@pytest.mark.fast
@pytest.mark.parametrize(
    "counts,expected",
    [
        ([2, 3, 1], [0, 0, 1, 1, 1, 2]),   # no empty segment: stock is right too
        ([2, 0, 3], [0, 0, 2, 2, 2]),      # interior zero: stock is silently WRONG
        ([2, 3, 0], [0, 0, 1, 1, 1]),      # trailing zero: stock raises IndexError
        ([0, 0, 0], []),                   # all empty: stock raises IndexError
        ([0, 2], [1, 1]),                  # leading zero
    ],
)
def test_segment_indices(counts, expected):
    got = segment_indices_from_n(torch.tensor(counts))
    assert got.tolist() == expected
    assert got.dtype == torch.long, "must be int64: it indexes scatter_add_"


@pytest.mark.fast
def test_segment_indices_matches_stock_when_no_empty_segments():
    """The replacement may only differ where stock was already wrong."""
    import matgl

    stock = matgl.utils.maths.get_segment_indices_from_n
    if getattr(stock, "__name__", "") == "segment_indices_from_n":
        pytest.skip("matgl already patched in this process")
    rng = np.random.default_rng(0)
    for _ in range(20):
        counts = torch.tensor(rng.integers(1, 6, size=12).tolist())  # all non-empty
        assert segment_indices_from_n(counts).tolist() == stock(counts).long().tolist()


# ---------------------------------------------------------------------------
# Bundle invariants
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def potential():
    from m3gnet_fixed import load_potential

    pot, _ = load_potential("fixed")
    return pot


@pytest.fixture(scope="module")
def graph(potential):
    """A rattled NaCl 2x2x2 as (graph, lattice, state)."""
    import matgl
    from ase.build import bulk
    from matgl.ext.pymatgen import Structure2Graph
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True) * (2, 2, 2)
    rng = np.random.default_rng(0)
    atoms.positions += rng.normal(scale=0.1, size=atoms.positions.shape)

    converter = Structure2Graph(
        element_types=potential.model.element_types, cutoff=float(potential.model.cutoff)
    )
    g, lattice, state = converter.get_graph(AseAtomsAdaptor.get_structure(atoms))
    # build_fixed_bundle computes its own positions and offsets from the
    # lattice, so the graph needs neither `pos` nor `pbc_offshift` here.
    return g, lattice, torch.as_tensor(np.asarray(state), dtype=matgl.float_th)


def test_fixed_bundle_is_in_full_bond_space(potential, graph):
    """The whole fix in one assertion: one row per bond, not per kept bond."""
    g, lattice, _ = graph
    bundle = build_fixed_bundle(g, lattice, float(potential.model.threebody_cutoff))

    n_bonds = g.edge_index.shape[1]
    assert bundle["n_triple_ij"].shape[0] == n_bonds
    assert bundle["bond_dist"].shape[0] == n_bonds
    # If nothing were pruned the fix would be a no-op; assert it is a real test.
    assert bundle["kept_edge_ids"].numel() < n_bonds, "cutoffs equal: defect cannot fire here"
    assert_lg_invariants(bundle)


def test_empty_bundle_invariants(potential, graph):
    g, lattice, _ = graph
    bundle = build_empty_bundle(g, lattice)

    n_bonds = g.edge_index.shape[1]
    assert bundle["line_edge_index"].shape == (2, 0)
    assert bundle["n_triple_ij"].shape[0] == n_bonds
    assert int(bundle["n_triple_ij"].sum()) == 0
    assert_lg_invariants(bundle)


def test_assert_lg_invariants_catches_corruption(potential, graph):
    """The guard must actually fire -- a guard that never fails is decoration."""
    g, lattice, _ = graph
    bundle = build_fixed_bundle(g, lattice, float(potential.model.threebody_cutoff))

    bad = dict(bundle)
    bad["n_triple_ij"] = bundle["n_triple_ij"].clone()
    bad["n_triple_ij"][0] += 1
    with pytest.raises(AssertionError):
        assert_lg_invariants(bad)


# ---------------------------------------------------------------------------
# The claims about the model
# ---------------------------------------------------------------------------


def test_relabelling_invariance(potential):
    """Permuting atom order is not physics, so the energy must not move.

    This is the black-box signature of the defect: under stock wiring the same
    structure gives a different energy depending on the order its atoms are
    listed in, by up to 9 meV/atom.
    """
    from ase.build import bulk

    from m3gnet_fixed import M3GNetCalculator

    atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
    rng = np.random.default_rng(0)
    atoms.positions += rng.normal(scale=0.1, size=atoms.positions.shape)

    atoms.calc = M3GNetCalculator(potential, wiring="fixed")
    e0, f0 = atoms.get_potential_energy(), atoms.get_forces()

    order = rng.permutation(len(atoms))
    shuffled = atoms[order]
    shuffled.calc = M3GNetCalculator(potential, wiring="fixed")
    e1, f1 = shuffled.get_potential_energy(), shuffled.get_forces()

    assert abs(e1 - e0) / len(atoms) < 1e-5, "energy depends on atom ordering"
    assert np.abs(f0[order] - f1).max() < 1e-4, "forces do not permute with the atoms"


def test_triplet_removal_changes_the_prediction(potential):
    """The corrected model must actually use its angular term.

    If this ever came out near zero it would mean the retrain had learned to
    ignore the three-body term the way the published model does, and the
    triplet-removal experiment would be measuring nothing.
    """
    from ase.build import bulk

    from m3gnet_fixed import M3GNetCalculator

    atoms = bulk("Si", "diamond", a=5.43, cubic=True) * (2, 2, 2)
    rng = np.random.default_rng(0)
    atoms.positions += rng.normal(scale=0.1, size=atoms.positions.shape)

    forces = {}
    for wiring in ("fixed", "none"):
        a = atoms.copy()
        a.calc = M3GNetCalculator(potential, wiring=wiring)
        forces[wiring] = a.get_forces()

    rel = np.sqrt(((forces["fixed"] - forces["none"]) ** 2).mean()) / np.sqrt(
        (forces["fixed"] ** 2).mean()
    )
    assert rel > 0.05, f"angular term contributes only {rel:.1%} of the force rms"


def test_regression_model_predicts_sensible_values():
    """Real rocksalts against experiment, on conventional cells.

    Loose tolerances -- this is a smoke test for the *wiring*, not an accuracy
    benchmark. It is written against absolute values on purpose: the periodic
    offsets (`pbc_offshift`) were once left unset, which measured every
    boundary-crossing bond as if it sat in the home cell. That produced NaN on
    a dense cell and merely-wrong numbers on a sparse one, and any test phrased
    only as "gap of A exceeds gap of B" would have passed straight through it.
    """
    from pymatgen.core import Lattice, Structure

    from m3gnet_fixed.regression import load_regression, predict

    model, norm = load_regression()
    sites = [[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [0, 0, .5],
             [.5, 0, .5], [0, .5, 0], [0, .5, .5], [.5, 0, 0]]
    nacl = Structure(Lattice.cubic(5.64), ["Na", "Cl"] * 4, sites)
    mgo = Structure(Lattice.cubic(4.21), ["Mg", "O"] * 4, sites)

    out = predict(model, norm, [nacl, mgo], linear=True)
    assert set(out) >= {"eform", "bandgap", "log10_G_VRH", "log10_K_VRH"}
    assert np.isfinite(np.concatenate([out[k] for k in out])).all(), "non-finite prediction"

    # NaCl: eform -2.05 eV/atom, gap 5.2 eV, K ~25 GPa.
    assert -2.6 < out["eform"][0] < -1.6
    assert 3.5 < out["bandgap"][0] < 6.5
    assert 12 < out["K_VRH"][0] < 45
    # MgO: eform -3.0 eV/atom, gap 4.7 eV, K ~160 GPa. The dense cell is the one
    # that went NaN, and its bulk modulus is 6x NaCl's -- a wiring error cannot
    # reproduce that spread by accident.
    assert -3.8 < out["eform"][1] < -2.6
    assert 3.0 < out["bandgap"][1] < 6.0
    assert 110 < out["K_VRH"][1] < 230


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunking_reproduces_unchunked_forces(potential):
    """At a converged halo the chunked forces must match the direct evaluation.

    Uses a cell large enough that the boxes are strict subsets -- on a small
    cell every box wraps to the whole cell and the test passes vacuously.
    """
    import matgl
    from matgl.ext.pymatgen import Structure2Graph
    from pymatgen.core import Lattice, Structure

    from m3gnet_fixed.chunking import chunked_forces, reference_forces

    s = Structure(
        Lattice.cubic(5.64),
        ["Na", "Cl", "Na", "Cl", "Na", "Cl", "Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0, 0, 0.5],
         [0.5, 0, 0.5], [0, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0]],
    )
    s.make_supercell([6, 6, 6])   # 33.8 A edge: a 12 A halo is a strict subset
    s.perturb(distance=0.15, seed=0)

    converter = Structure2Graph(
        element_types=potential.model.element_types, cutoff=float(potential.model.cutoff)
    )
    g, lattice, state = converter.get_graph(s)
    # chunking.py sets `pos` and `pbc_offshift` itself from the lattice, so
    # neither is needed here; setting `pos` alone would be actively misleading,
    # since a graph carrying `pos` without `pbc_offshift` measures every
    # periodic-image bond as though it were in the home cell.
    state = torch.as_tensor(np.asarray(state), dtype=matgl.float_th)

    ref = reference_forces(potential, g, lattice, state, wiring="fixed")
    got = chunked_forces(
        potential, g, lattice, state, num_chunks=1, chunk_padding=12.0, wiring="fixed"
    )

    rms = float((got.forces - ref.forces).pow(2).sum(1).mean().sqrt())
    assert rms < 1e-4, f"chunked forces differ by {rms*1000:.4f} meV/A at a 12 A halo"
    # accumulate conserves momentum exactly, at any halo.
    assert float(got.forces.sum(0).abs().max()) < 1e-3
    assert abs(float(got.energy) - float(ref.energy)) / len(s) < 1e-4


def test_chunking_error_grows_as_the_halo_shrinks(potential):
    """A halo sweep must be monotone enough to justify the 12 A operating point."""
    import matgl
    from matgl.ext.pymatgen import Structure2Graph
    from pymatgen.core import Lattice, Structure

    from m3gnet_fixed.chunking import chunked_forces, reference_forces

    s = Structure(
        Lattice.cubic(5.64),
        ["Na", "Cl", "Na", "Cl", "Na", "Cl", "Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0, 0, 0.5],
         [0.5, 0, 0.5], [0, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0]],
    )
    s.make_supercell([6, 6, 6])
    s.perturb(distance=0.15, seed=0)

    converter = Structure2Graph(
        element_types=potential.model.element_types, cutoff=float(potential.model.cutoff)
    )
    g, lattice, state = converter.get_graph(s)
    # chunking.py sets `pos` and `pbc_offshift` itself from the lattice, so
    # neither is needed here; setting `pos` alone would be actively misleading,
    # since a graph carrying `pos` without `pbc_offshift` measures every
    # periodic-image bond as though it were in the home cell.
    state = torch.as_tensor(np.asarray(state), dtype=matgl.float_th)
    ref = reference_forces(potential, g, lattice, state, wiring="fixed")

    err = {}
    for pad in (5.0, 8.0, 12.0):
        got = chunked_forces(
            potential, g, lattice, state, num_chunks=1, chunk_padding=pad, wiring="fixed"
        )
        err[pad] = float((got.forces - ref.forces).pow(2).sum(1).mean().sqrt())

    assert err[5.0] > err[8.0] > err[12.0]
    assert err[12.0] < 1e-5 < err[8.0]
