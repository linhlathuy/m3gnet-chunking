#!/usr/bin/env python
"""Molecular dynamics with the corrected M3GNet potential.

    python run_md.py --check                                  # ~1 min, do this first
    python run_md.py --structure my.cif --supercell 3 3 3 --temperature 1000 --steps 4000
    python run_md.py --ensemble npt --temperature 300 --pressure 1.0 --device cuda
    python run_md.py --arm published --triplets off           # an ablation arm

Ensembles: ``nvt`` (Berendsen, the default), ``nve``, ``npt`` (Berendsen with
isotropic cell scaling). Berendsen is not a canonical thermostat -- it does not
sample the canonical ensemble, and nothing here should be read as a converged
canonical average. It is used because it is what the comparison runs used, and
changing it would make trajectories incomparable.

One process, one arm
--------------------
The corrected wiring installs a global monkeypatch (``install_segment_index_fix``)
that must not be active while a stock-wiring model runs. This script therefore
runs exactly one arm per process, by construction. To compare arms, run it twice.

Outputs, under ``--out`` (default ``runs/<tag>/``):

    trajectory.traj   ASE trajectory, sampled every --sample steps
    log.csv           step, time_ps, T, E_pot, E_kin, E_tot, and volume under npt
    run.json          every argument, the resolved model provenance, and timings

``log.csv``'s ``E_tot`` is the conserved quantity under ``nve``: plot it to see
whether the timestep is small enough. Drift of more than a few meV/atom over the
run means the timestep is too long for the chemistry, not that the model is wrong.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from m3gnet_fixed import ARMS, WIRINGS, M3GNetCalculator, load_potential  # noqa: E402


def build_atoms(args):
    from ase.build import bulk
    from ase.io import read

    if args.structure:
        atoms = read(args.structure)
    else:
        atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms = atoms * tuple(args.supercell)
    if args.rattle:
        rng = np.random.default_rng(args.seed)
        atoms.positions += rng.normal(scale=args.rattle, size=atoms.positions.shape)
    return atoms


def make_dynamics(atoms, args):
    """The integrator. Constructed directly rather than through
    ``matgl.ext.ase.MolecularDynamics``, which hardwires a stock CPU-only
    ``PESCalculator`` and so cannot express a chosen wiring at all."""
    from ase import units
    from ase.md.nptberendsen import NPTBerendsen
    from ase.md.nvtberendsen import NVTBerendsen
    from ase.md.verlet import VelocityVerlet

    dt = args.timestep * units.fs
    if args.ensemble == "nve":
        return VelocityVerlet(atoms, timestep=dt)
    if args.ensemble == "nvt":
        return NVTBerendsen(
            atoms, timestep=dt, temperature_K=args.temperature, taut=args.taut * units.fs
        )
    return NPTBerendsen(
        atoms,
        timestep=dt,
        temperature_K=args.temperature,
        taut=args.taut * units.fs,
        pressure_au=args.pressure * units.bar,
        taup=args.taup * units.fs,
        compressibility_au=args.compressibility / units.bar,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--structure", default=None, help="CIF/POSCAR; default bulk Si")
    p.add_argument("--supercell", type=int, nargs=3, default=[2, 2, 2])
    p.add_argument("--rattle", type=float, default=0.0, help="A of initial displacement")
    p.add_argument("--arm", default="fixed", choices=sorted(ARMS))
    p.add_argument("--triplets", default="on", choices=["on", "off"],
                   help="'off' switches the three-body term off at inference")
    p.add_argument("--wiring", default=None, choices=[None, *WIRINGS],
                   help="override the arm's native wiring; normally leave unset")
    p.add_argument("--ensemble", default="nvt", choices=["nvt", "nve", "npt"])
    p.add_argument("--temperature", type=float, default=300.0, help="K")
    p.add_argument("--pressure", type=float, default=1.0, help="bar, npt only")
    p.add_argument("--compressibility", type=float, default=4.57e-5, help="1/bar, npt only")
    p.add_argument("--timestep", type=float, default=1.0, help="fs")
    p.add_argument("--taut", type=float, default=100.0, help="fs, thermostat time constant")
    p.add_argument("--taup", type=float, default=1000.0, help="fs, barostat time constant")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--sample", type=int, default=10, help="write every N steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--check", action="store_true",
                   help="20 steps on a small cell, to prove the install works")
    a = p.parse_args(argv)

    if a.check:
        a.steps, a.sample, a.supercell, a.tag = 20, 5, [2, 2, 2], a.tag or "check"

    from ase import units
    from ase.io import Trajectory
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    atoms = build_atoms(a)
    potential, native = load_potential(a.arm)
    wiring = a.wiring or ("none" if a.triplets == "off" else native)

    atoms.calc = M3GNetCalculator(potential, wiring=wiring, device=a.device)

    # Seeded, and the seed goes into run.json: the whole point of a second arm
    # is to start it from the same velocities.
    rng = np.random.RandomState(a.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=a.temperature, rng=rng)

    tag = a.tag or f"{a.arm}_{wiring}_{a.ensemble}_{a.temperature:.0f}K"
    out = a.out or Path(__file__).resolve().parent / "runs" / tag
    out.mkdir(parents=True, exist_ok=True)

    print(f"{len(atoms)} atoms   arm {a.arm!r}   wiring {wiring!r}   "
          f"{a.ensemble} {a.temperature:g} K   {a.steps} x {a.timestep:g} fs   {a.device}")
    print(f"-> {out}\n")

    dyn = make_dynamics(atoms, a)
    traj = Trajectory(str(out / "trajectory.traj"), "w", atoms)
    log = open(out / "log.csv", "w", newline="")
    writer = csv.writer(log)
    columns = ["step", "time_ps", "T_K", "E_pot_eV", "E_kin_eV", "E_tot_eV", "volume_A3"]
    writer.writerow(columns)

    t0 = time.perf_counter()

    def record():
        step = dyn.get_number_of_steps()
        e_pot = atoms.get_potential_energy()
        e_kin = atoms.get_kinetic_energy()
        writer.writerow([
            step,
            step * a.timestep / 1000.0,
            e_kin / (1.5 * units.kB * len(atoms)),
            e_pot, e_kin, e_pot + e_kin,
            atoms.get_volume(),
        ])
        log.flush()

    # attach() alone already fires once at step 0 when run() starts, so there
    # is no separate priming call -- adding one duplicates the first row.
    dyn.attach(traj.write, interval=a.sample)
    dyn.attach(record, interval=a.sample)
    dyn.run(a.steps)

    wall = time.perf_counter() - t0
    traj.close()
    log.close()

    meta = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "n_atoms": len(atoms),
        "arm": a.arm,
        "wiring": wiring,
        "native_wiring_for_arm": native,
        "formula": atoms.get_chemical_formula(),
        "wall_s": wall,
        "ms_per_step": 1000 * wall / max(a.steps, 1),
        "final_energy_eV": float(atoms.get_potential_energy()),
        "final_volume_A3": float(atoms.get_volume()),
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\n{a.steps} steps in {wall:.1f} s ({meta['ms_per_step']:.1f} ms/step)")
    print(f"final E = {meta['final_energy_eV']:.4f} eV "
          f"({meta['final_energy_eV']/len(atoms):.4f} eV/atom)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
