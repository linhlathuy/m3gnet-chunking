"""Spatial chunking of an M3GNet force evaluation, in matgl 4.0.2 (PyG).

This is a port of ``pes.py`` from https://github.com/linhlathuy/m3gnet-chunking
onto the graph API the corrected model needs, plus the one change that makes
the test meaningful for that model: **three-body terms stay on**.

What chunking is
----------------
Split the cell into ``(num_chunks + 1)**3`` boxes on the fractional axes. For
each box, take the *core* atoms inside it plus every atom within ``pad`` A of
it, evaluate the potential on that padded node-induced subgraph, and keep only
the forces on the core atoms. Every atom is a core atom of exactly one box, so
the boxes tile the force array with no double counting.

The approximation is bounded by ``pad``: a force on a core atom is exact once
the subgraph contains everything the model can propagate information from,
which for ``n_blocks`` rounds of message passing is ``n_blocks * cutoff``.
Anything less truncates the receptive field, and the error shows up as a force
residual against the unchunked evaluation.

How this differs from the upstream repo
---------------------------------------
Upstream sets ``use_edges=False`` and always passes ``l_g=None``, so the model
runs with **no three-body term at all**. That is a defensible thing to do to a
model whose three-body wiring is broken anyway, but it makes the chunking test
blind to the angular part of the PES — which is the part chunking is most
likely to damage, because triples reach further than bonds.

Here the line graph is rebuilt per chunk, in the corrected full-parent-bond
index space (:func:`m3gnet_fixed.pruning.to_full_bond_space`), so the chunked
arm and the reference arm are the same physics and the residual measures
chunking alone. ``wiring="stock"`` reproduces the upstream ``l_g=None`` path
for the released weights, which is the space those weights were fitted in.

Energy
------
Chunked *forces* are exact-up-to-``pad``; a chunked *total energy* is not
available this way, because the padded subgraphs overlap and matgl's
``Potential`` returns a per-graph total rather than per-atom energies. As
upstream does, the total energy is taken from one unchunked no-grad pass. That
pass is cheap relative to the backward it avoids, but it does mean the peak
memory of the full graph is still touched once per step — see the README.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import torch
from torch_geometric.data import Data

from .pruning import (
    assert_lg_invariants,
    build_empty_bundle,
    build_fixed_bundle,
    install_segment_index_fix,
)

__all__ = [
    "atomic_energies",
    "build_line_graph",
    "plan_chunking",
    "tiling_cost",
    "ChunkStats",
    "ChunkedResult",
    "chunk_masks",
    "default_padding",
    "node_subgraph",
    "chunked_forces",
    "reference_forces",
]


# --------------------------------------------------------------------------
# fractional-interval masks
# --------------------------------------------------------------------------


def _wrap(frac: torch.Tensor) -> torch.Tensor:
    """Fold fractional coordinates into ``[0, 1)``, strictly.

    ``frac - floor(frac)`` is not enough. A coordinate a hair below zero -- which
    is routine once MD has been running, since ASE does not wrap positions --
    folds to ``1.0 - eps``, and in float32 that rounds to exactly ``1.0``. The
    box intervals are half-open, so such an atom is the core of no box and its
    force is silently left at zero. Push those back to 0.0.
    """
    out = frac - torch.floor(frac)
    return torch.where(out >= 1.0, torch.zeros_like(out), out)


def _interval_mask(frac: torch.Tensor, start: float, end: float) -> torch.Tensor:
    """Atoms in the half-open fractional interval ``[start, end)``, wrapping."""
    if start <= end:
        return (frac >= start) & (frac < end)
    return (frac >= start) | (frac < end)


def _interval_mask_with_pad(
    frac: torch.Tensor, start: float, end: float, pad: float
) -> torch.Tensor:
    """``[start - pad, end + pad)`` with periodic wrap on a single axis.

    Upstream's version silently mis-selects when the padded interval is wider
    than the cell (both ends wrap at once), which happens for a small cell or a
    generous pad — exactly the regime a chunking test explores. Guarded here:
    once the padding covers the axis, every atom is in range.
    """
    start_pad = start - pad
    end_pad = end + pad
    if end_pad - start_pad >= 1.0:
        return torch.ones_like(frac, dtype=torch.bool)
    if start_pad < 0.0:
        return (frac >= start_pad + 1.0) | (frac < end_pad)
    if end_pad > 1.0:
        return (frac >= start_pad) | (frac < end_pad - 1.0)
    return (frac >= start_pad) & (frac < end_pad)


def energy_radius(model) -> float:
    """How far an atom's *energy* reaches: ``n_blocks * cutoff``.

    ``n_blocks`` rounds of message passing, each over ``cutoff``, so
    ``e_i`` is a function of exactly the atoms within this radius of ``i``.
    """
    return float(model.cutoff) * int(getattr(model, "n_blocks", 1))


def default_padding(model, wiring: str, scheme: str = "accumulate") -> float:
    """The padding that makes chunked forces exact, for the given ``scheme``.

    Let ``R`` be the energy radius: ``e_i`` is a function of exactly the atoms
    within ``R`` of ``i``. The two schemes need different multiples of it.

    ``scheme="extract"`` evaluates the whole padded box and keeps the core
    atoms' forces. That force must be *complete inside its own box*, and
    ``F_i = -dE/dx_i`` with ``E = sum_j e_j`` collects ``de_j/dx_i`` for every
    ``j`` within ``R`` of ``i`` -- each of which needs its own ``R``. So the
    dependency reaches ``2R``. This is what upstream does, and the factor of
    two is the reason chunking looked unaffordable.

    ``scheme="accumulate"`` backpropagates only ``sum_{i in core} e_i`` and
    adds the resulting gradient onto *every* atom of the box, halo included.
    Each atom is core in exactly one box, so summing over boxes gives atom
    ``j`` the full ``sum_i de_i/dx_j``. Now only the core *energies* must be
    exact, so ``R`` suffices -- half the pad, and on top of that the total
    energy comes out of the same pass instead of needing a full-cell one.

    Measured on LPGS (sc=8, 25600 atoms, exact sphere around 6 atoms, relative
    force error against a full unchunked pass) for the ``extract`` scheme::

        radius   8 A    12 A     16 A     20 A     24 A     28 A
        error   1.4    9.1e-2   8.2e-4   5.3e-6   9.3e-6   1.2e-5

    Both bounds are loose in practice -- measured on LPGS 6x6x6 against a full
    unchunked pass, ``accumulate`` reaches the float32 floor at 12 A and
    ``extract`` at 16-20 A, below the 15/30 the formulas give. Message passing
    decays much faster than its formal radius, so the operating pad is
    calibrated rather than derived; these values are the safe ceiling.

    The angular term does *not* widen ``R``, which is worth stating because the
    opposite is easy to assume. matgl's triples share their **source** atom
    (``_compute.py``), the three-body layer reads the partner bond's
    destination (``_three_body.py``), and the convolution scatters into source
    nodes (``_graph_convolution.py``) -- so a block still carries information
    exactly ``cutoff``, not ``cutoff + threebody_cutoff``. If triples chained
    through a shared *bond* instead, this function would be wrong.
    """
    r = energy_radius(model)
    pad = (r if scheme == "accumulate" else 2.0 * r) + 1.0
    if wiring not in ("stock", "none"):
        pad = max(pad, float(getattr(model, "threebody_cutoff", 0.0)))
    return pad


def _pad_to_fractional(lattice: torch.Tensor, pad: float) -> list[float]:
    """Cartesian ``pad`` as a fractional width on each axis.

    Not ``pad / |a_i|``. With row-vector convention ``r = f @ A``, a Cartesian
    displacement ``d`` shifts ``f_i`` by ``d . col_i(A^-1)``, so covering every
    atom within ``pad`` A needs ``pad * |col_i(A^-1)|`` -- the reciprocal of the
    *interplanar spacing*, which for a skewed cell is strictly smaller than
    ``|a_i|``. Using the edge length there under-pads the halo and silently
    drops atoms that are inside the requested radius.

    The two agree exactly for an orthogonal cell, which is why LPGS never saw
    it. Found by review, not by a failing run.
    """
    lat = lattice.reshape(3, 3)
    inv = torch.linalg.inv(lat.double())
    return [pad * float(torch.linalg.norm(inv[:, i])) for i in range(3)]


def chunk_masks(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    num_chunks: int,
    pad: float,
):
    """Yield ``(index, core_mask, pad_mask)`` for every non-empty chunk.

    ``index`` is the ``(ix, iy, iz)`` box index. ``core_mask`` selects the
    atoms this box owns; ``pad_mask`` selects those plus the padding shell.
    """
    pad_frac = _pad_to_fractional(lattice, pad)

    frac = _wrap(frac_coords)
    divisions = num_chunks + 1
    # Explicit edges rather than i * width: the boxes must tile [0, 1) exactly,
    # and float rounding in `divisions * width` can leave the last box ending
    # just short of 1.0, orphaning any atom in the gap.
    edges = [i / divisions for i in range(divisions)] + [1.0]

    for ix in range(divisions):
        mx = _interval_mask(frac[:, 0], edges[ix], edges[ix + 1])
        if not torch.any(mx):
            continue
        px = _interval_mask_with_pad(frac[:, 0], edges[ix], edges[ix + 1], pad_frac[0])
        for iy in range(divisions):
            my = _interval_mask(frac[:, 1], edges[iy], edges[iy + 1])
            if not torch.any(my):
                continue
            py = _interval_mask_with_pad(frac[:, 1], edges[iy], edges[iy + 1], pad_frac[1])
            for iz in range(divisions):
                mz = _interval_mask(frac[:, 2], edges[iz], edges[iz + 1])
                core = mx & my & mz
                if not torch.any(core):
                    continue
                pz = _interval_mask_with_pad(
                    frac[:, 2], edges[iz], edges[iz + 1], pad_frac[2]
                )
                yield (ix, iy, iz), core, (px & py & pz)


def max_sub_nodes_for(
    frac_coords: torch.Tensor,
    lattice: torch.Tensor,
    num_chunks: int,
    pad: float,
) -> int:
    """Largest padded subgraph a given ``(num_chunks, pad)`` will produce.

    Exact -- it runs the real mask generator -- but touches nothing beyond
    boolean masks over the atoms: no subgraph, no line graph, no forward pass.
    That lets a driver refuse a configuration *before* it exhausts memory
    rather than discovering the limit as an OOM two hours into a sweep.
    """
    return max(
        int(pad_mask.sum())
        for _, _, pad_mask in chunk_masks(frac_coords, lattice, num_chunks, pad)
    )


# --------------------------------------------------------------------------
# subgraph
# --------------------------------------------------------------------------


def node_subgraph(g: Data, node_mask: torch.Tensor) -> tuple[Data, torch.Tensor]:
    """Node-induced subgraph of a matgl PyG graph, with relabelled node ids.

    Returns ``(subgraph, orig_ids)`` where ``orig_ids[i]`` is the parent-graph
    id of subgraph node ``i``. Edges survive only when both endpoints do; the
    dropped ones are precisely the ties to atoms outside the padded region,
    which is what makes the chunk cheaper than the whole cell.
    """
    orig_ids = node_mask.nonzero(as_tuple=False).view(-1)
    relabel = torch.full((g.num_nodes,), -1, dtype=torch.long, device=orig_ids.device)
    relabel[orig_ids] = torch.arange(orig_ids.numel(), device=orig_ids.device)

    src = g.edge_index[0].long()
    dst = g.edge_index[1].long()
    edge_mask = node_mask[src] & node_mask[dst]

    edge_index = torch.stack([relabel[src[edge_mask]], relabel[dst[edge_mask]]], dim=0)

    sub = Data(num_nodes=int(orig_ids.numel()), edge_index=edge_index.to(g.edge_index.dtype))
    sub.pbc_offset = g.pbc_offset[edge_mask]
    sub.node_type = g.node_type[orig_ids]
    sub.frac_coords = g.frac_coords[orig_ids]
    # Parent bond id of each subgraph bond, in subgraph bond order. The audit in
    # threebody_audit.py needs this to compare a chunk's triples against the
    # full graph's; nothing in the force path uses it.
    sub.parent_edge_ids = edge_mask.nonzero(as_tuple=False).view(-1)
    return sub, orig_ids


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass
class ChunkStats:
    """Per-chunk bookkeeping, for the log and for sanity checks."""

    index: tuple[int, int, int]
    n_core: int
    n_pad: int
    n_edges: int
    n_triples: int
    energy: float


@dataclass
class ChunkedResult:
    energy: torch.Tensor
    forces: torch.Tensor
    n_chunks: int
    pad: float
    wall_s: float
    peak_bytes: int
    max_sub_nodes: int
    chunks: list[ChunkStats] = field(default_factory=list)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def build_line_graph(
    g: Data, lattice: torch.Tensor, wiring: str, threebody_cutoff: float, check: bool = False
):
    """The line graph for one graph, chunk or whole cell, in the arm's wiring.

    Public because every path that evaluates the potential must go through it.
    The bond-space assertion is not decoration: without it
    ``ensure_line_graph_compatibility`` can take its degenerate branch and
    silently reinstate pruned-space geometry, which is the exact defect this
    project exists to correct. A second code path that builds the bundle inline
    is a second place for that guard to go missing.
    """
    if wiring == "stock":
        return None
    # Mandatory for both bundle-building wirings, and easy to lose: in "fixed"
    # every bond in no triple gets an n_triple_ij of zero, and under "none"
    # every segment is empty. Stock get_segment_indices_from_n mis-assigns
    # interior zeros silently and raises IndexError on a trailing one. Chunking
    # meets both cases far more often than a whole-cell pass does, because a box
    # truncates coordination shells at its halo edge.
    install_segment_index_fix()
    if wiring == "none":
        # Three-body inference off. With line_edge_index empty the three-body
        # layer scatters an empty basis into num_bonds segments, gets an exact
        # zero, and update_network_bond (a GatedMLP with use_bias=False) maps
        # 0 -> SiLU(0)*sigmoid(0) = 0, so every block passes edge_feat through
        # untouched. Exactly a no-op, not approximately one.
        return build_empty_bundle(g, lattice)
    bundle = build_fixed_bundle(g, lattice, threebody_cutoff)
    if bundle["bond_dist"].size(0) != g.edge_index.size(1):
        raise AssertionError("line-graph bundle is not in full bond space")
    if check:
        assert_lg_invariants(bundle)
    return bundle


_line_graph = build_line_graph


# --------------------------------------------------------------------------
# cost model
# --------------------------------------------------------------------------


def tiling_cost(lengths, num_chunks: int, pad: float) -> dict[str, float]:
    """Peak-memory and compute cost of one ``(num_chunks, pad)`` tiling.

    Pure geometry -- no model, no GPU, milliseconds. The two numbers that
    decide whether chunking is worth doing at all:

    ``peak_fraction``
        sub-box volume (core + halo) over cell volume, i.e. the factor by which
        peak memory falls. Chunking is only useful when this is well below 1.

    ``overhead``
        total atom-evaluations over cell atoms, ``prod_d (1 + 2*pad/L_d)``.
        Note what this does *not* contain: cell size. Fix the box edge and the
        overhead is a constant, whatever the cell -- which is the whole point.
        Peak memory stays flat while the cell grows, so the trade is a fixed
        multiple of compute for unbounded cells.

    The halo is why pad is expensive rather than merely inconvenient. At a
    budget-limited 82 A sub-box on LPGS, ``pad=20`` leaves a 42 A core and
    costs 7.4x; ``pad=14`` leaves 54 A and costs 3.5x. Halving the pad is
    worth roughly a factor of two in compute, which is the entire argument for
    :func:`chunked_forces` with ``scheme="accumulate"``.
    """
    k = int(num_chunks) + 1
    edges = [float(x) / k for x in lengths]
    boxes = [e + 2.0 * pad for e in edges]
    # A box wider than the cell wraps onto itself: the tiling has collapsed to
    # the unchunked evaluation and neither number means anything else.
    collapsed = any(b >= float(x) for b, x in zip(boxes, lengths))
    cell_vol = float(lengths[0]) * float(lengths[1]) * float(lengths[2])
    box_vol = boxes[0] * boxes[1] * boxes[2]
    if collapsed:
        return {
            "num_chunks": float(num_chunks),
            "n_boxes": float(k**3),
            "peak_fraction": 1.0,
            "overhead": float(k**3),
            "collapsed": 1.0,
        }
    return {
        "num_chunks": float(num_chunks),
        "n_boxes": float(k**3),
        "peak_fraction": box_vol / cell_vol,
        "overhead": (k**3) * box_vol / cell_vol,
        "collapsed": 0.0,
    }


def plan_chunking(lengths, pad: float, budget_atoms: int, n_atoms: int, ceiling: int = 31):
    """Every tiling that fits ``budget_atoms``, cheapest first.

    Returns a list of :func:`tiling_cost` dicts with ``sub_atoms`` added,
    ordered by ``overhead``. Empty means no tiling fits at this pad and budget
    -- worth getting from arithmetic rather than from an out-of-memory error
    twenty minutes in.

    **An estimate, not a feasibility proof.** ``sub_atoms`` is
    ``peak_fraction * n_atoms``, i.e. it assumes uniform density. A clustered
    or void-containing cell can put far more atoms in one box than its volume
    share suggests (a constructed case: predicted 12.5, actual 100). Use it to
    choose a tiling; :func:`max_sub_nodes_for` counts the real masks and is
    what the force driver gates rows on.
    """
    out = []
    for nc in range(0, ceiling + 1):
        cost = tiling_cost(lengths, nc, pad)
        cost["sub_atoms"] = cost["peak_fraction"] * float(n_atoms)
        if cost["sub_atoms"] <= budget_atoms:
            out.append(cost)
    return sorted(out, key=lambda c: c["overhead"])


def atomic_energies(potential, g: Data, lattice: torch.Tensor, state_attr, l_g):
    """Per-atom energies of ``g``, with autograd live back to Cartesian positions.

    matgl's ``Potential`` returns a per-graph total, but the extensive head
    builds it by summing a per-atom vector, and that vector is left on
    ``model.feature_dict["readout"]`` still attached to the graph. Reading it
    there is what makes a *masked* backward possible -- ``sum_{i in core} e_i``
    rather than the whole box's energy -- which is the whole trick.

    The position construction mirrors ``Potential.forward`` exactly; the
    per-graph terms it adds afterwards (``data_mean``, ``element_refs``) are
    constants in position and so contribute nothing to a force, but they are
    needed for the energy and are added once by the caller.

    Returns ``(atomic, pos)`` with ``atomic`` already scaled by ``data_std``.
    """
    if getattr(potential, "calc_repuls", False):
        raise NotImplementedError(
            "calc_repuls adds a pairwise term outside the per-atom head, so it "
            "cannot be split into core and halo contributions this way"
        )
    model = potential.model
    lat = lattice.reshape(-1, 3, 3)[:1]

    g = copy.copy(g)
    n_edges = int(g.edge_index.size(1))
    g.lattice = lat.expand(n_edges, 3, 3)
    g.pbc_offshift = (g.pbc_offset.unsqueeze(-1) * g.lattice).sum(dim=1)
    pos = (g.frac_coords.unsqueeze(-1) * lat.expand(int(g.num_nodes), 3, 3)).sum(dim=1)
    pos.requires_grad_(True)
    g.pos = pos

    model(g=g, state_attr=state_attr, l_g=l_g)
    atomic = model.feature_dict["readout"].reshape(-1)
    return potential.data_std * atomic, pos


def _constant_energy_terms(potential, g: Data) -> float:
    """``data_mean`` plus element reference offsets: per-graph, position-independent.

    ``atomic_energies`` deliberately omits these because they cannot be
    attributed to a box, and they are invisible to forces. Added once to the
    accumulated chunk energies to land on the same total the unchunked path
    reports.
    """
    total = float(potential.data_mean.reshape(-1)[0])
    if getattr(potential, "element_refs", None) is not None:
        with torch.no_grad():
            total += float(torch.squeeze(potential.element_refs(g)).reshape(-1)[0])
    return total


def reference_forces(
    potential,
    g: Data,
    lattice: torch.Tensor,
    state_attr: torch.Tensor,
    *,
    wiring: str = "fixed",
    threebody_cutoff: float | None = None,
    check_invariants: bool = False,
) -> ChunkedResult:
    """Unchunked evaluation, in the same wiring — the thing chunks are scored against."""
    tbc = float(threebody_cutoff or potential.model.threebody_cutoff)
    device = g.frac_coords.device
    _reset_peak(device)
    t0 = time.perf_counter()
    l_g = _line_graph(g, lattice, wiring, tbc, check_invariants)
    out = potential(g=g, lat=lattice, state_attr=state_attr, l_g=l_g)
    energy, forces = out[0].detach(), out[1].detach()
    _sync(device)
    return ChunkedResult(
        energy=energy,
        forces=forces,
        n_chunks=0,
        pad=float("nan"),
        wall_s=time.perf_counter() - t0,
        peak_bytes=_peak(device),
        max_sub_nodes=int(g.num_nodes),
    )


def chunked_forces(
    potential,
    g: Data,
    lattice: torch.Tensor,
    state_attr: torch.Tensor,
    *,
    num_chunks: int,
    chunk_padding: float | None = None,
    wiring: str = "fixed",
    threebody_cutoff: float | None = None,
    check_invariants: bool = False,
    collect_stats: bool = True,
    total_energy: bool = True,
    scheme: str = "accumulate",
) -> ChunkedResult:
    """Forces from spatial chunks, by one of two schemes.

    ``scheme="accumulate"`` (default) backpropagates each box's *core energy*
    and adds the gradient onto every atom of the box, halo included. Exact at
    ``pad = R``, and three properties come free that the other scheme has to
    be argued into:

    * the total energy is ``sum_boxes E_core``, so no full-cell pass is needed
      -- which matters, because that pass was the one O(N) allocation left in
      a method whose entire purpose is to avoid O(N) allocations;
    * ``sum_j F_j = 0`` to float precision at *any* pad, because each box's
      core energy is invariant under rigid translation of that box;
    * the forces are the exact gradient of ``sum_boxes E_core``, a genuine
      scalar, so they are conservative at any pad -- *piecewise*. The scalar is
      defined by the current masks, and MD changes them: an atom crossing a box
      face changes which box owns it, and an atom crossing a *halo* boundary
      changes which subgraph edges and triples exist, which moves ``E_core``
      even when no core assignment changed. Both jumps vanish at a converged
      pad, where ``e_i`` does not depend on which box sees it. So NVE drift is
      the measurement of convergence, not an assumption about it.

    ``scheme="extract"`` is the upstream method: evaluate the padded box, keep
    the core atoms' forces, discard the halo's. Needs ``pad = 2R``, needs a
    separate pass for the energy, and conserves neither momentum nor energy
    below convergence. Kept because it is what the port started from and what
    the comparison is against.

    ``num_chunks=0`` degenerates to a single chunk covering the cell, i.e. the
    unchunked answer plus the subgraph machinery -- a useful null test that the
    chunking path itself introduces nothing.

    ``total_energy`` is consulted only by ``extract``; under ``accumulate`` the
    energy is a by-product and is always returned.
    """
    if scheme not in ("accumulate", "extract"):
        raise ValueError(f"scheme must be 'accumulate' or 'extract', got {scheme!r}")
    tbc = float(threebody_cutoff or potential.model.threebody_cutoff)
    pad = (
        default_padding(potential.model, wiring, scheme)
        if chunk_padding is None
        else float(chunk_padding)
    )
    device = g.frac_coords.device

    _reset_peak(device)
    t0 = time.perf_counter()

    forces = torch.zeros_like(g.frac_coords)
    covered = torch.zeros(g.num_nodes, dtype=torch.bool, device=device)
    stats: list[ChunkStats] = []
    max_sub_nodes = 0
    core_energy = 0.0

    for index, core_mask, pad_mask in chunk_masks(g.frac_coords, lattice, num_chunks, pad):
        sub, orig_ids = node_subgraph(g, pad_mask)
        max_sub_nodes = max(max_sub_nodes, int(sub.num_nodes))
        core_in_sub = core_mask[orig_ids]
        core_ids = orig_ids[core_in_sub]

        l_g = _line_graph(sub, lattice, wiring, tbc, check_invariants)

        if scheme == "accumulate":
            atomic, pos = atomic_energies(potential, sub, lattice, state_attr, l_g)
            e_core = atomic[core_in_sub].sum()
            grad = torch.autograd.grad(e_core, pos)[0]
            # index_add_, not assignment: an atom is halo to many boxes and core
            # to one, and every one of those visits owes it a contribution.
            forces.index_add_(0, orig_ids, -grad.detach())
            chunk_energy = float(e_core.detach())
            core_energy += chunk_energy
        else:
            out = potential(g=sub, lat=lattice, state_attr=state_attr, l_g=l_g)
            chunk_forces = out[1].detach()
            forces[core_ids] = chunk_forces[core_in_sub]
            chunk_energy = float(out[0].detach().reshape(-1)[0])

        covered[core_ids] = True

        if collect_stats:
            stats.append(
                ChunkStats(
                    index=index,
                    n_core=int(core_in_sub.sum()),
                    n_pad=int(sub.num_nodes) - int(core_in_sub.sum()),
                    n_edges=int(sub.edge_index.size(1)),
                    n_triples=0 if l_g is None else int(l_g["line_edge_index"].size(1)),
                    energy=chunk_energy,
                )
            )

    if not bool(covered.all()):
        raise RuntimeError(
            f"{int((~covered).sum())} of {g.num_nodes} atoms were not the core of any "
            "chunk; the box tiling is broken"
        )

    if scheme == "accumulate":
        energy = torch.tensor(
            [core_energy + _constant_energy_terms(potential, g)],
            dtype=forces.dtype,
            device=device,
        )
    elif total_energy:
        with torch.no_grad():
            l_g = _line_graph(g, lattice, wiring, tbc, check_invariants)
            saved = potential.calc_forces, potential.calc_stresses
            potential.calc_forces, potential.calc_stresses = False, False
            try:
                energy = potential(g=g, lat=lattice, state_attr=state_attr, l_g=l_g)[0].detach()
            finally:
                potential.calc_forces, potential.calc_stresses = saved
    else:
        energy = torch.full((1,), float("nan"), device=device)

    _sync(device)
    return ChunkedResult(
        energy=energy,
        forces=forces,
        n_chunks=num_chunks,
        pad=pad,
        wall_s=time.perf_counter() - t0,
        peak_bytes=_peak(device),
        max_sub_nodes=max_sub_nodes,
        chunks=stats,
    )


# --------------------------------------------------------------------------
# device helpers
# --------------------------------------------------------------------------


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _peak(device: torch.device) -> int:
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
