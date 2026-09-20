#!/usr/bin/env python
"""Spatial chunking: evaluate a cell too large to fit, one box at a time.

    python chunk_forces.py --supercell 4 4 4                    # accuracy vs pad
    python chunk_forces.py --supercell 6 6 6 --pad 12 --chunks 2 --device cuda
    python chunk_forces.py --structure my.cif --no-reference    # just run it

Split the cell into ``(chunks + 1)**3`` boxes on the fractional axes. For each
box take the *core* atoms inside it plus every atom within ``pad`` A of it,
evaluate the potential on that padded subgraph, and keep the core atoms'
contribution. Every atom is a core atom of exactly one box, so the boxes tile
the force array with no double counting, and peak memory is set by the largest
box rather than by the cell.

The default scheme is ``accumulate``: backpropagate each box's *core* energy and
add the gradient onto every atom of the box, halo included. That gives the total
energy as ``sum_boxes E_core`` with no full-cell pass, exact momentum
conservation (``sum_j F_j = 0``) at any pad, and forces that are the exact
gradient of a genuine scalar. The alternative, ``extract`` -- evaluate the box,
keep the core forces, discard the halo's -- needs twice the halo, needs a
separate full-cell pass for the energy, and its forces are not the gradient of
anything, so they violate Newton's third law below convergence.

Choosing the halo
-----------------
12 A is the operating point. Measured on FeNi 21x21x15, an Au/La/Mn/Na/Pt alloy
3x3x3 and LPGS 6x6x5, the force residual against an unchunked pass decays
60 -> 15 -> 0.7 -> 0.03 -> 0.0005 meV/A at pad 3/5/8/10/12, and at 12 A the
partition test (rigid-translating the cell and measuring the force spread across
translations -- the discontinuity MD would step over) sits exactly at the
float32 floor. Below that the run is not partition-invisible; above it there is
nothing left to gain and 16 A simply costs 30% more. ``--pad`` defaults to the
model's own receptive field, which is more conservative still.

Raising ``--chunks`` is the cheap way to afford a large halo: a box is a strict
subset of the cell only where ``L_d > 2 * pad * k / (k - 1)`` with ``k = chunks + 1``,
so at k=2 a 12 A halo needs a 48 A edge, while at k=4 a 20 A halo needs only 53 A.

Overhead in production is 2.3-3.7x, not the 30x this script will show on a small
cell: the cost depends only on the *core* edge, and a demo cell small enough to
also fit an unchunked reference has a core of ~10 A against a 12 A halo. Sized to
fill a GPU (a 45-75 A core) the halo is a thin shell.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from m3gnet_fixed import load_potential  # noqa: E402
from m3gnet_fixed.chunking import (  # noqa: E402
    chunked_forces,
    default_padding,
    reference_forces,
)


def build_graph(structure, potential, device):
    """Structure -> (graph, lattice, state) on ``device``, as the model wants it."""
    from matgl.ext.pymatgen import Structure2Graph

    import matgl

    converter = Structure2Graph(
        element_types=potential.model.element_types, cutoff=float(potential.model.cutoff)
    )
    g, lattice, state_default = converter.get_graph(structure)
    g = g.to(device)
    lattice = lattice.to(device)
    # chunking.py sets `pos` and `pbc_offshift` itself from the lattice, so
    # neither is needed here; setting `pos` alone would be actively misleading,
    # since a graph carrying `pos` without `pbc_offshift` measures every
    # periodic-image bond as though it were in the home cell.
    state = torch.as_tensor(np.asarray(state_default), dtype=matgl.float_th, device=device)
    return g, lattice, state


def load_structure(args):
    from pymatgen.core import Lattice, Structure

    if args.structure:
        s = Structure.from_file(args.structure)
    else:
        # NaCl rocksalt: a real cell with real forces once rattled.
        s = Structure(
            Lattice.cubic(5.64),
            ["Na", "Cl", "Na", "Cl", "Na", "Cl", "Na", "Cl"],
            [
                [0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0, 0, 0.5],
                [0.5, 0, 0.5], [0, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0],
            ],
        )
    s.make_supercell(args.supercell)
    if args.rattle:
        # A perfect crystal sits at a force-free point where |F|rms is ~1e-6 and
        # every *relative* error metric becomes a division by noise. Rattling
        # gives the cell real forces to measure the chunking residual against.
        s.perturb(distance=args.rattle, seed=args.seed)
    return s


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--structure", type=Path, default=None, help="CIF/POSCAR; default NaCl")
    p.add_argument("--supercell", type=int, nargs=3, default=[3, 3, 3])
    p.add_argument("--rattle", type=float, default=0.15, help="A of random displacement")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunks", type=int, default=1, help="boxes per axis = chunks + 1")
    p.add_argument("--pad", type=float, default=None, help="halo in A; default = receptive field")
    p.add_argument("--pad-sweep", type=float, nargs="*", default=None,
                   help="compare several halos, e.g. --pad-sweep 5 8 10 12")
    p.add_argument("--scheme", default="accumulate", choices=["accumulate", "extract"])
    p.add_argument("--arm", default="fixed", choices=["fixed", "published"])
    p.add_argument("--wiring", default=None, help="override the arm's native wiring")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-reference", action="store_true",
                   help="skip the unchunked pass (which is what chunking exists to avoid)")
    a = p.parse_args(argv)

    potential, native = load_potential(a.arm)
    wiring = a.wiring or native
    potential = potential.to(a.device)

    structure = load_structure(a)
    g, lattice, state = build_graph(structure, potential, a.device)
    n_atoms = len(structure)
    lengths = structure.lattice.abc
    print(f"{n_atoms} atoms   cell {lengths[0]:.1f} x {lengths[1]:.1f} x {lengths[2]:.1f} A")
    print(f"arm {a.arm!r}  wiring {wiring!r}  scheme {a.scheme!r}  device {a.device}\n")

    pads = a.pad_sweep if a.pad_sweep else [a.pad or default_padding(potential.model, wiring, a.scheme)]

    ref = None
    if not a.no_reference:
        ref = reference_forces(potential, g, lattice, state, wiring=wiring)
        f_ref = ref.forces
        print(f"reference (unchunked): E = {float(ref.energy):.6f} eV   "
              f"|F|rms = {float(f_ref.pow(2).sum(1).mean().sqrt())*1000:.3f} meV/A   "
              f"{ref.wall_s*1000:.0f} ms   peak {ref.peak_bytes/2**30:.2f} GiB\n")

    header = f"{'pad':>6}  {'boxes':>5}  {'E (eV)':>14}  {'wall':>8}  {'peak':>8}"
    if ref is not None:
        header += f"  {'rms err':>11}  {'max err':>11}  {'|sum F|':>10}"
    print(header)
    print("-" * len(header))

    for pad in pads:
        res = chunked_forces(
            potential, g, lattice, state,
            num_chunks=a.chunks, chunk_padding=pad, wiring=wiring, scheme=a.scheme,
        )
        row = (f"{pad:6.1f}  {(a.chunks+1)**3:5d}  {float(res.energy):14.6f}  "
               f"{res.wall_s*1000:6.0f}ms  {res.peak_bytes/2**30:6.2f}G")
        if ref is not None:
            d = (res.forces - ref.forces)
            rms = float(d.pow(2).sum(1).mean().sqrt()) * 1000
            mx = float(d.abs().max()) * 1000
            net = float(res.forces.sum(0).abs().max()) * 1000
            row += f"  {rms:9.4f}mA  {mx:9.4f}mA  {net:8.2e}"
        print(row)

    if ref is not None:
        print("\nerrors in meV/A. |sum F| is the net force -- it must stay at float noise;")
        print("a growing value means the scheme is not conserving momentum.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
