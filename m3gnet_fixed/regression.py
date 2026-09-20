"""The multi-target property-regression model: loading and prediction.

Separate from :mod:`~m3gnet_fixed.calculator` because it is a different kind of
object. The PES model is a ``matgl.apps.pes.Potential`` -- energy, forces and
stress of one structure, wrapped in an ASE calculator. This is a bare
``M3GNet`` with ``is_intensive=True, ntargets=4`` that maps a structure to four
scalar properties and has no forces at all, so ASE has nothing to attach to.

Targets, in the model's output order:

======================  ============  ==================================
name                    unit          what
======================  ============  ==================================
``eform``               eV/atom       formation energy
``bandgap``             eV            band gap
``log10_G_VRH``         log10(GPa)    shear modulus, Voigt-Reuss-Hill
``log10_K_VRH``         log10(GPa)    bulk modulus, Voigt-Reuss-Hill
======================  ============  ==================================

The model is trained on standardized targets, so predictions come out of the
network in normalized space and :func:`predict` un-standardizes them with the
``normalizer.json`` shipped beside the weights. Predicting without that step
returns z-scores, which look plausible and are wrong.

The moduli are trained and predicted as ``log10(GPa)``. ``predict`` returns them
that way, as the model's own units; pass ``linear=True`` to also get
``G_VRH``/``K_VRH`` in GPa.

Like the PES model, these weights were fitted in the **corrected** three-body
index space, so the fix must be active. :func:`load_regression` installs it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import matgl

from .pruning import build_fixed_bundle, install_segment_index_fix

__all__ = ["REGRESSION_MODEL_DIR", "TARGETS", "load_regression", "predict"]

#: The corrected regression model shipped with this package.
REGRESSION_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "regression"

TARGETS = ("eform", "bandgap", "log10_G_VRH", "log10_K_VRH")


def load_regression(model_dir: str | Path | None = None, device: str | torch.device = "cpu"):
    """Return ``(model, normalizer)``, the model in eval mode on ``device``.

    ``normalizer`` is the dict from ``normalizer.json``: ``targets``, ``units``,
    ``mean`` and ``std``, each a list of four. Hand both to :func:`predict`.
    """
    from matgl.models import M3GNet

    path = Path(model_dir) if model_dir is not None else REGRESSION_MODEL_DIR
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. The regression model ships in models/regression; "
            "re-download the release if it is not there."
        )

    # These weights were fitted in the corrected index space. Without this the
    # angular terms scatter into the wrong bonds and the predictions are wrong
    # in a way that raises no error.
    install_segment_index_fix()

    config = json.loads((path / "model.json").read_text())
    model = M3GNet(**config)
    model.load_state_dict(torch.load(path / "state.pt", map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    normalizer = json.loads((path / "normalizer.json").read_text())
    return model, normalizer


@torch.no_grad()
def predict(
    model,
    normalizer: dict,
    structures,
    device: str | torch.device = "cpu",
    linear: bool = False,
) -> dict[str, np.ndarray]:
    """Predict the four properties for one or more pymatgen ``Structure``s.

    Args:
        structures: a ``Structure`` or a sequence of them.
        linear: also return ``G_VRH`` and ``K_VRH`` in GPa, as ``10 ** log10_*``.

    Returns a dict of target name -> ``(n_structures,)`` array, in real units.

    Structures are evaluated one at a time. The model is intensive and the
    per-structure cost is dominated by graph construction, so batching buys
    little and would need the line-graph bundle offset per structure -- the one
    place the index remap is easy to get subtly wrong.
    """
    from pymatgen.core import Structure

    from matgl.ext.pymatgen import Structure2Graph

    if isinstance(structures, Structure):
        structures = [structures]
    structures = list(structures)

    element_types = model.element_types
    cutoff = float(model.cutoff)
    threebody_cutoff = float(model.threebody_cutoff)
    converter = Structure2Graph(element_types=element_types, cutoff=cutoff)
    device = torch.device(device)

    rows = []
    for structure in structures:
        graph, lattice, state_default = converter.get_graph(structure)
        graph = graph.to(device)
        lattice = lattice.to(device)
        # Both are required, and only `pos` is obvious. M3GNet.forward reads
        # `g.pbc_offshift` and passes it to compute_pair_vector_and_distance;
        # left unset it defaults to None, every periodic-image bond is measured
        # as if its neighbour sat in the home cell, and the ~87% of bonds that
        # cross a boundary collapse to length 0. The SphericalBessel basis then
        # divides by r and the prediction is NaN -- or, on a cell where fewer
        # bonds wrap, merely wrong with no warning at all.
        cart = graph.frac_coords @ lattice.reshape(3, 3)
        graph.pos = cart
        graph.pbc_offshift = graph.pbc_offset.to(cart.dtype) @ lattice.reshape(3, 3)
        state_attr = torch.as_tensor(
            np.asarray(state_default), dtype=matgl.float_th, device=device
        )
        # The same corrected wiring the weights were trained with.
        l_g = build_fixed_bundle(graph, lattice, threebody_cutoff)
        out = model(g=graph, state_attr=state_attr, l_g=l_g)
        rows.append(out.detach().cpu().numpy().reshape(-1))

    z = np.stack(rows)  # (n, 4), normalized
    mean = np.asarray(normalizer["mean"], dtype=float)
    std = np.asarray(normalizer["std"], dtype=float)
    values = z * std + mean

    names = list(normalizer["targets"])
    result = {name: values[:, i] for i, name in enumerate(names)}
    if linear:
        for name in names:
            if name.startswith("log10_"):
                result[name[len("log10_") :]] = 10.0 ** values[:, names.index(name)]
    return result
