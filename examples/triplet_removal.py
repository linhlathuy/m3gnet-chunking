#!/usr/bin/env python
"""How much does the three-body term actually contribute?

    python triplet_removal.py                          # shipped corrected model
    python triplet_removal.py --arm published          # the published model
    python triplet_removal.py --both                   # both, side by side
    python triplet_removal.py --structure my.cif --supercell 2 2 2

Switch the angular term off at inference -- keep the weights, hand the model a
line graph with no triples -- and measure how far the energy and forces move.
One force call per arm: no dynamics, no GPU hours.

Why this is worth running before any MD
---------------------------------------
An MD comparison between "with triples" and "without" is expensive and its
result can be null. A null result is only interesting if it was not already
visible for free, and it is: if the angular term contributes 0.2% of the force
RMS at t=0, no trajectory is going to show a structural difference, and an RDF
comparison is then measuring the thermostat. Running this first turns "the RDFs
came out the same" into "the RDFs came out the same, as this predicted".

It is a static probe and nothing more. It cannot see amplification -- a small
force error integrated over thousands of steps can still take a trajectory
somewhere else -- so a small delta here is a reason to *expect* a small MD
difference, not a reason to skip the run.

What the numbers mean
---------------------
``dE``      (E_control - E_no_triplets) per atom, in meV. Signed: the angular
            term is not a repulsive correction and its sign varies by chemistry.
``dF rms``  root-mean-square force change over every atom and component.
``%``       that, as a percentage of the control's own force RMS. This is the
            scale-free number, and the only one worth comparing across
            chemistries and models.

Each arm's control is its own training wiring -- ``"fixed"`` for the corrected
model, ``"stock"`` for the published one -- so each model is compared against
itself, not against the other.

The expected result, and why it is the finding
----------------------------------------------
The published model barely uses its three-body term: removing every triple moves
its forces by 0.1-2.4% of its own force RMS, and under 1 meV/atom in energy. The
corrected retrain moves by 13-43% on the same cells -- 18x to 119x more. That is
what a working angular term looks like, and it is consistent with the defect: an
angular term fed misindexed inputs throughout training carries little signal, so
training learns to down-weight it. So a near-null result for the published model
is the measurement, not a failed run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from m3gnet_fixed import ARMS, M3GNetCalculator, load_potential  # noqa: E402


def build_cells(args):
    """The structures to probe. Rattled, because a perfect crystal can sit at a
    force-free point where every relative force metric is a division by noise."""
    from ase.build import bulk
    from ase.io import read

    if args.structure:
        cells = [(Path(args.structure).name, read(args.structure))]
    else:
        cells = [
            ("Si diamond", bulk("Si", "diamond", a=5.43, cubic=True)),
            ("NaCl rocksalt", bulk("NaCl", "rocksalt", a=5.64, cubic=True)),
            ("Al fcc", bulk("Al", "fcc", a=4.05, cubic=True)),
            ("TiO2 rutile", _rutile()),
        ]

    out = []
    for name, atoms in cells:
        atoms = atoms * tuple(args.supercell)
        rng = np.random.default_rng(args.seed)
        atoms.positions += rng.normal(scale=args.rattle, size=atoms.positions.shape)
        out.append((name, atoms))
    return out


def _rutile():
    from ase import Atoms

    a, c, u = 4.594, 2.959, 0.305
    return Atoms(
        "Ti2O4",
        scaled_positions=[
            [0, 0, 0], [0.5, 0.5, 0.5],
            [u, u, 0], [1 - u, 1 - u, 0],
            [0.5 + u, 0.5 - u, 0.5], [0.5 - u, 0.5 + u, 0.5],
        ],
        cell=[a, a, c],
        pbc=True,
    )


def probe(arm: str, cells, device: str) -> list[dict]:
    """One control call and one triplet-free call per cell, same weights."""
    potential, control_wiring = load_potential(arm)
    rows = []
    for name, atoms in cells:
        results = {}
        for label, wiring in (("control", control_wiring), ("no_triplets", "none")):
            a = atoms.copy()
            a.calc = M3GNetCalculator(potential, wiring=wiring, device=device)
            results[label] = (a.get_potential_energy(), a.get_forces())

        e_c, f_c = results["control"]
        e_n, f_n = results["no_triplets"]
        f_rms = float(np.sqrt((f_c**2).mean()))
        df_rms = float(np.sqrt(((f_c - f_n) ** 2).mean()))
        rows.append(
            {
                "cell": name,
                "arm": arm,
                "control_wiring": control_wiring,
                "n_atoms": len(atoms),
                "dE_meV_per_atom": (e_c - e_n) * 1000 / len(atoms),
                "dF_rms_eV_A": df_rms,
                "F_rms_eV_A": f_rms,
                "dF_percent": 100.0 * df_rms / f_rms if f_rms > 0 else float("nan"),
            }
        )
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arm", default="fixed", choices=sorted(ARMS))
    p.add_argument("--both", action="store_true", help="run both arms")
    p.add_argument("--structure", default=None, help="CIF/POSCAR instead of the built-ins")
    p.add_argument("--supercell", type=int, nargs=3, default=[2, 2, 2])
    p.add_argument("--rattle", type=float, default=0.1, help="A of random displacement")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--json", type=Path, default=None, help="also write the rows here")
    a = p.parse_args(argv)

    cells = build_cells(a)
    arms = ["fixed", "published"] if a.both else [a.arm]

    rows: list[dict] = []
    for arm in arms:
        # One arm per process would be cleaner -- install_segment_index_fix is a
        # global monkeypatch. It is safe in this direction because the patch is
        # provably identical to stock whenever no segment is empty, and a stock
        # bundle never has one; but do not build a --both loop that relies on
        # undoing it.
        rows += probe(arm, cells, a.device)

    width = max(len(r["cell"]) for r in rows)
    header = (f"{'cell':<{width}}  {'arm':>9}  {'atoms':>5}  {'dE meV/at':>10}  "
              f"{'dF rms eV/A':>11}  {'F rms eV/A':>10}  {'dF %':>7}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['cell']:<{width}}  {r['arm']:>9}  {r['n_atoms']:5d}  "
              f"{r['dE_meV_per_atom']:10.3f}  {r['dF_rms_eV_A']:11.4f}  "
              f"{r['F_rms_eV_A']:10.4f}  {r['dF_percent']:6.2f}%")

    if a.both:
        print()
        for cell in dict.fromkeys(r["cell"] for r in rows):
            per = {r["arm"]: r["dF_percent"] for r in rows if r["cell"] == cell}
            if per.get("published", 0) > 0:
                ratio = per["fixed"] / per["published"]
                print(f"  {cell:<{width}}  corrected model uses the angular term "
                      f"{ratio:.0f}x more strongly than the published one")

    if a.json:
        a.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
