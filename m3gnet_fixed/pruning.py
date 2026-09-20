"""The three-body index fix: line-graph remapping and the segment-index repair.

Standalone copy of the two corrections this model requires at inference. Both
are transcriptions of the code the model was *trained* with -- if you evaluate
these weights without them, you are evaluating the model off-distribution.

Background
----------
matgl (every release v0.1.0-v4.0.3) does not persist line graphs.
``M3GNet.forward`` builds one on the fly via
``matgl.graph._compute.create_line_graph``, which returns a ``dict[str, Tensor]``
bundle rather than a graph object.

``matgl/layers/_three_body.py`` computes
``segment_ids = get_segment_indices_from_n(n_triple_ij)`` and scatters the
angular basis into bonds using those segment ids. That is only correct when:

1. ``line_edge_index[0]`` is sorted ascending (triples grouped by source bond);
2. ``n_triple_ij == bincount(line_edge_index[0], minlength=n_lg_nodes)``;
3. ``n_triple_ij.sum() == line_edge_index.shape[1]``.

Break any of these and angular terms are aggregated into the **wrong bonds**
with no error raised. :func:`assert_lg_invariants` exists to make that failure
loud instead of silent.
"""

from __future__ import annotations

import torch

import matgl
from matgl.graph._compute import compute_pair_vector_and_distance, create_line_graph

__all__ = [
    "install_segment_index_fix",
    "segment_indices_from_n",
    "compute_positions",
    "to_full_bond_space",
    "build_fixed_bundle",
    "build_empty_bundle",
    "assert_lg_invariants",
]

_FIX_INSTALLED = False


def segment_indices_from_n(ns: torch.Tensor) -> torch.Tensor:
    """Correct replacement for ``matgl.utils.maths.get_segment_indices_from_n``.

    Given per-segment counts, return the segment id of each element:
    ``[2, 0, 3] -> [0, 0, 2, 2, 2]``.

    Returns int64: the result indexes ``scatter_add_``, which rejects int32.
    """
    return torch.repeat_interleave(
        torch.arange(ns.numel(), device=ns.device, dtype=torch.long), ns.long()
    )


def install_segment_index_fix() -> bool:
    """Repair ``get_segment_indices_from_n`` for line graphs with empty segments.

    Stock matgl computes segment ids as::

        segments = torch.zeros(ns.sum(), dtype=matgl.int_th)
        segments[ns.cumsum(0)[:-1]] = 1
        return segments.cumsum(0)

    which is only correct when **every** segment is non-empty:

    ===================  ==========================  ====================
    ``ns``               stock matgl                 correct
    ===================  ==========================  ====================
    ``[2, 3, 1]``        ``[0,0,1,1,1,2]``           same
    ``[2, 0, 3]``        ``[0,0,1,1,1]`` (**wrong**) ``[0,0,2,2,2]``
    ``[2, 3, 0]``        ``IndexError``              ``[0,0,1,1,1]``
    ``[0, 0, 0]``        ``IndexError``              ``[]``
    ===================  ==========================  ====================

    Interior zeros are the dangerous case: no exception is raised, and the
    angular basis is scattered into the **wrong bonds**.

    After :func:`to_full_bond_space`, ``n_triple_ij`` is zero for every bond that
    is in no triple, so empty segments are routine. This fix is therefore a
    prerequisite for the corrected wiring, not a nicety.

    :func:`segment_indices_from_n` is provably identical to the original
    whenever no segment is empty, so nothing else changes.

    **This monkeypatches matgl globally.** Never run a stock-wiring model in the
    same process as a fixed-wiring one.

    Idempotent. Returns True if this call installed the fix.
    """
    global _FIX_INSTALLED
    if _FIX_INSTALLED:
        return False

    import matgl.layers._three_body as _three_body
    import matgl.utils.maths as _maths

    segment_indices_from_n.__wrapped__ = _maths.get_segment_indices_from_n  # type: ignore[attr-defined]

    _maths.get_segment_indices_from_n = segment_indices_from_n
    # The name is imported into the layer module's namespace, so patching
    # matgl.utils.maths alone would not take effect.
    _three_body.get_segment_indices_from_n = segment_indices_from_n

    _FIX_INSTALLED = True
    return True


def compute_positions(
    frac_coords: torch.Tensor,
    pbc_offset: torch.Tensor,
    lattice: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cartesian positions and PBC offset shifts for a single structure.

    Mirrors ``ModelLightningModule.forward``, which does
    ``(frac_coords.unsqueeze(-1) * node_lat).sum(dim=1)`` -- i.e. ``frac @ lat``.
    """
    lat = lattice.reshape(3, 3).to(frac_coords.dtype)
    pos = frac_coords @ lat
    offshift = pbc_offset.to(frac_coords.dtype) @ lat
    return pos, offshift


def to_full_bond_space(
    lg: dict[str, torch.Tensor],
    bond_dist: torch.Tensor,
    bond_vec: torch.Tensor,
    offshift: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Re-express the bundle in **full** parent-bond indices. This is the fix.

    Why this exists
    ---------------
    ``create_line_graph`` numbers line-graph nodes over the *within-cutoff*
    bonds only (ids ``0..n_kept-1``). ``M3GNet.forward`` then indexes two
    tensors built over **all** bonds with those ids::

        edge_dst_atom     = edge_index[1]                       # all bonds
        three_body_cutoff = polynomial_cutoff(bond_dist, rc)    # all bonds

    The two index spaces do not agree whenever ``threebody_cutoff < cutoff``,
    and the consequences are severe. On fcc-Al (168 bonds, 48 within a 4.0 A
    cutoff) the ids select bonds spanning 2.86-4.96 A, 35 of which lie beyond
    the cutoff, so their polynomial cutoff is ~0: **93.9% of triple weights
    collapse to zero** and the angular basis is suppressed ~16x. 418 of 528
    triples also read the wrong destination atom, and the scatter writes updates
    into wrong bond rows.

    The fix
    -------
    Map the line graph into full bond space via ``kept_edge_ids``. Every tensor
    matgl indexes then shares one space and its existing code becomes
    self-consistent -- no patch to ``M3GNet.forward`` required.

    * ``kept_edge_ids`` is ascending, so remapping preserves the
      sorted-by-source-bond invariant that the three-body scatter depends on.
    * ``n_triple_ij`` becomes length ``num_bonds``, zero for bonds in no triple.
      This makes empty segments routine, so :func:`install_segment_index_fix`
      is mandatory.
    * ``ensure_line_graph_compatibility`` takes its ``else`` branch and slices
      ``bond_dist[:num_bonds]`` -- i.e. all bonds, which is now correct.

    **This deliberately breaks equivalence with stock matgl.** A bundle from
    this function does not reproduce ``l_g=None``; that is the point, since
    ``l_g=None`` is the broken path.
    """
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
    if offshift is not None:
        out["pbc_offset"] = offshift
    return out


def build_fixed_bundle(graph, lattice, threebody_cutoff: float) -> dict[str, torch.Tensor]:
    """The corrected line-graph bundle, built exactly as training built it.

    Keeping this a literal transcription of the training-time code is the point:
    the MD path must present the model the same wiring its weights were fitted
    against.

    The bundle supplies **topology only**. ``M3GNet.forward`` recomputes
    ``bond_vec``/``bond_dist`` from live positions and
    ``ensure_line_graph_compatibility`` re-attaches them, so forces stay
    autograd-correct.
    """
    pos, offshift = compute_positions(graph.frac_coords, graph.pbc_offset, lattice)
    bond_vec, bond_dist = compute_pair_vector_and_distance(pos, graph.edge_index, offshift)
    lg = create_line_graph(
        graph.edge_index, bond_dist, bond_vec, offshift, pos.size(0), threebody_cutoff
    )
    return to_full_bond_space(lg, bond_dist, bond_vec, offshift)


def build_empty_bundle(graph, lattice) -> dict[str, torch.Tensor]:
    """A line-graph bundle holding **no** triples: three-body inference, off.

    This is the triplet-removal arm. It is the same thing as pruning 100% of the
    triples, with the line graph never built in the first place --
    ``_compute_3body_indices`` is O(sum_i d_i^2) in the coordination numbers and
    materialises a triple list that would only be thrown away. On a 6000-atom
    cell that list is the memory ceiling of the step, so building it to delete
    it would defeat the point. ``tests/test_release.py`` asserts the two routes
    give bit-identical energies and forces.

    Why the three-body layer really is a no-op on this bundle, rather than
    approximately one. With ``line_edge_index`` empty, ``ThreeBodyInteractions``
    scatters an empty basis into ``num_bonds`` segments and gets an exact zero
    matrix, then returns ``edge_feat + update_network_bond(0)``. That last term
    vanishes because ``update_network_bond`` is a single-layer ``GatedMLP`` built
    with ``use_bias=False`` (``matgl/models/_m3gnet.py``), so it computes
    ``SiLU(W x) * sigmoid(V x)`` -- and ``SiLU(0) = 0``. Every block therefore
    passes ``edge_feat`` through untouched. Restore a bias to that MLP and this
    stops being exact.

    Two invariants of the parent bundle are preserved, both load-bearing:
    ``n_triple_ij`` is length ``num_bonds`` (full parent-bond space, not the
    pruned space), and it is all zeros -- which makes every segment empty and so
    requires :func:`install_segment_index_fix`, exactly as the fixed path does.
    """
    pos, offshift = compute_positions(graph.frac_coords, graph.pbc_offset, lattice)
    bond_vec, bond_dist = compute_pair_vector_and_distance(pos, graph.edge_index, offshift)
    num_bonds = int(bond_dist.shape[0])
    device = bond_dist.device

    # Geometry is carried only for shape: M3GNet.forward hands the bundle to
    # ensure_line_graph_compatibility, which overwrites bond_dist / bond_vec /
    # pbc_offset from its own positions before anything reads them. What
    # survives the round trip is line_edge_index and n_triple_ij.
    return {
        "edge_index_pruned": graph.edge_index,
        "kept_edge_ids": torch.arange(num_bonds, dtype=matgl.int_th, device=device),
        "bond_dist": bond_dist,
        "bond_vec": bond_vec,
        "pbc_offset": offshift,
        "line_edge_index": torch.zeros((2, 0), dtype=matgl.int_th, device=device),
        "n_triple_ij": torch.zeros(num_bonds, dtype=matgl.int_th, device=device),
    }


def assert_lg_invariants(lg: dict[str, torch.Tensor]) -> None:
    """Raise if the bundle would silently mis-scatter in the three-body layer."""
    lei = lg["line_edge_index"]
    n_triple = lg["n_triple_ij"]
    n_nodes = int(n_triple.shape[0])

    if lei.shape[1] > 0:
        src = lei[0]
        if not bool(torch.all(src[1:] >= src[:-1])):
            raise AssertionError("line_edge_index[0] is not sorted ascending")
        if int(src.max()) >= n_nodes:
            raise AssertionError(
                f"line_edge_index references bond {int(src.max())} but only "
                f"{n_nodes} line-graph nodes exist"
            )

    expected = torch.bincount(lei[0].long(), minlength=n_nodes).to(n_triple.dtype)
    if not torch.equal(expected, n_triple):
        raise AssertionError("n_triple_ij != bincount(line_edge_index[0])")

    if int(n_triple.sum()) != int(lei.shape[1]):
        raise AssertionError(
            f"n_triple_ij.sum()={int(n_triple.sum())} != n_triples={int(lei.shape[1])}"
        )

    if lg["bond_dist"].shape[0] != n_nodes:
        raise AssertionError(
            f"bond_dist has {lg['bond_dist'].shape[0]} rows but n_triple_ij has {n_nodes}"
        )
