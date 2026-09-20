"""One ASE calculator, three line-graph wirings, on the GPU.

Why a single class for every arm
--------------------------------
If the published model ran through stock ``matgl.ext.ase.PESCalculator`` and the
corrected model through a subclass, any difference in the result would be
confounded by that code difference. Routing every arm through this one class
reduces the difference between arms to two things: the weights, and the ``l_g``
argument.

It is also forced rather than preferred. Stock ``PESCalculator`` has no device
handling -- ``Atoms2Graph`` builds CPU tensors and the potential stays on CPU.
Measured at ~1200 ms/step on 192 atoms, that is far too slow for any real
trajectory. This class moves the graph, lattice and state onto the device and
keeps them there.

The three wirings
-----------------
``"fixed"``   the corrected index space. Required by the model in ``models/pes``.
``"stock"``   plain matgl (``l_g=None``), i.e. the defective space. This is the
              space the *published* weights were fitted in, so it is the right
              way to run them -- and the arm you compare against.
``"none"``    no triples at all, so the three-body term is switched off at
              inference regardless of how the weights were trained. This is the
              triplet-removal test; see :func:`~m3gnet_fixed.pruning.build_empty_bundle`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

import matgl
from matgl.ext.ase import PESCalculator

from .pruning import (
    assert_lg_invariants,
    build_empty_bundle,
    build_fixed_bundle,
    install_segment_index_fix,
)

__all__ = [
    "WIRINGS",
    "ARMS",
    "PES_MODEL_DIR",
    "PUBLISHED_MODEL",
    "load_potential",
    "make_calculator",
    "M3GNetCalculator",
]

WIRINGS = ("stock", "fixed", "none")

#: The corrected PES retrain shipped with this package.
PES_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "pes"

#: The published potential, fetched from matgl's model hub on first use.
PUBLISHED_MODEL = "M3GNet-PES-MatPES-r2SCAN-2025.2"

#: arm name -> the line-graph wiring that arm's weights were trained against.
#: Evaluating an arm in any other wiring is off-distribution and is only
#: meaningful as a deliberate ablation.
ARMS = {"fixed": "fixed", "published": "stock"}


def load_potential(arm: str = "fixed", model_dir: str | Path | None = None):
    """Return ``(potential, wiring)`` for ``arm``, in eval mode.

    ``"fixed"``     -- the corrected retrain in ``models/pes`` (shipped here).
    ``"published"`` -- ``M3GNet-PES-MatPES-r2SCAN-2025.2``, downloaded by matgl
                       on first use. Needs network access once.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {sorted(ARMS)}, got {arm!r}")
    if model_dir is not None:
        src = str(model_dir)
    else:
        src = str(PES_MODEL_DIR) if arm == "fixed" else PUBLISHED_MODEL
    if arm == "fixed" and model_dir is None and not PES_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"{PES_MODEL_DIR} is missing. The corrected model ships in models/pes; "
            "re-download the release if it is not there."
        )
    return matgl.load_model(src).eval(), ARMS[arm]


def make_calculator(
    arm: str = "fixed",
    device: str | torch.device = "cpu",
    wiring: str | None = None,
    model_dir: str | Path | None = None,
    **kwargs,
) -> "M3GNetCalculator":
    """Load an arm and wrap it in its correct wiring.

    Args:
        arm: ``"fixed"`` or ``"published"``.
        wiring: override the arm's native wiring. Use ``"none"`` for the
            triplet-removal test; anything else is an off-distribution ablation
            and should be labelled as one.
    """
    potential, native = load_potential(arm, model_dir=model_dir)
    return M3GNetCalculator(potential, wiring=wiring or native, device=device, **kwargs)


class M3GNetCalculator(PESCalculator):
    """``PESCalculator`` with a selectable index space and a real device.

    Args:
        potential: a ``matgl.apps.pes.Potential`` in eval mode.
        wiring: one of :data:`WIRINGS`. See the module docstring.
        device: torch device the potential and every per-step tensor live on.
        threebody_cutoff: must match the model's. Read off the model when None.
        check_invariants: assert the three-body scatter preconditions on every
            step. Costs an extra bincount; leave off for production.
    """

    def __init__(
        self,
        potential,
        wiring: str = "fixed",
        device: str | torch.device = "cpu",
        threebody_cutoff: float | None = None,
        stress_unit: str = "eV/A3",
        check_invariants: bool = False,
        **kwargs,
    ) -> None:
        if wiring not in WIRINGS:
            raise ValueError(f"wiring must be one of {WIRINGS}, got {wiring!r}")
        super().__init__(potential, stress_unit=stress_unit, **kwargs)

        self.wiring = wiring
        self.device = torch.device(device)
        self.check_invariants = bool(check_invariants)
        self.potential = self.potential.to(self.device)

        if threebody_cutoff is None:
            threebody_cutoff = float(potential.model.threebody_cutoff)
        self.threebody_cutoff = float(threebody_cutoff)

        if wiring in ("fixed", "none"):
            # Mandatory, not optional. to_full_bond_space makes n_triple_ij
            # length num_bonds with zeros for bonds in no triple, and stock
            # get_segment_indices_from_n is wrong for any interior-zero segment.
            # Under "none" every segment is empty, which stock raises
            # IndexError on outright.
            install_segment_index_fix()

    def calculate(self, atoms=None, properties=None, system_changes=None):
        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        # Deliberately skips PESCalculator.calculate -- that IS the stock path,
        # which is CPU-pinned and cannot pass l_g. Go straight to the ASE base.
        Calculator.calculate(
            self, atoms=atoms, properties=properties, system_changes=system_changes
        )

        # Wrap into the primary cell before building the graph. Nothing in an MD
        # loop wraps, so over a long run atoms diffuse many cells out and
        # pymatgen's find_points_in_spheres -- which sizes its image search from
        # the bounding box of the input points -- does cubically more work for an
        # identical answer. Measured on a combustion trajectory: 0.23 ms at frac
        # range [0, 1] (step 0) against 45.7 ms at [-4.3, 6.1] (step 62k), both
        # yielding ~3.3k edges. Unwrapped, the run asymptotically stalls.
        #
        # The wrap goes on a copy: the caller's positions feed the integrator's
        # momentum update and the centre-of-mass drift diagnostic, and wrapping
        # those in place would corrupt both. Wrapping is exact under PBC -- the
        # graph carries pbc_offset beside frac_coords, so bond vectors are
        # unchanged up to float32 rounding (~1e-3 eV/A).
        source = atoms if atoms is not None else self.atoms
        wrapped = source.copy()
        wrapped.wrap()

        graph, lattice, state_default = self._atoms2graph.get_graph(wrapped)
        graph = graph.to(self.device)
        lattice = lattice.to(self.device)
        state_attr = self.state_attr if self.state_attr is not None else state_default
        state_attr = torch.as_tensor(
            np.asarray(state_attr), dtype=matgl.float_th, device=self.device
        )

        line_graph = None
        if self.wiring == "fixed":
            line_graph = build_fixed_bundle(graph, lattice, self.threebody_cutoff)
        elif self.wiring == "none":
            line_graph = build_empty_bundle(graph, lattice)
        if line_graph is not None and self.check_invariants:
            assert_lg_invariants(line_graph)

        out = self.potential(g=graph, lat=lattice, state_attr=state_attr, l_g=line_graph)

        energy = out[0].detach().cpu().numpy().item()
        self.results.update(
            energy=energy,
            free_energy=energy,
            forces=out[1].detach().cpu().numpy(),
        )
        if self.compute_stress:
            stress = out[2][0] if out[2].dim() == 3 else out[2]
            stress = stress.detach().cpu().numpy()
            if self.use_voigt:
                stress = full_3x3_to_voigt_6_stress(stress)
            # conversion_factor is GPa -> eV/A^3 given stress_unit="eV/A3".
            self.results.update(stress=stress * self.conversion_factor)
