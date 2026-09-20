"""M3GNet retrained in the corrected three-body index space.

Potential (energy, forces, stress; drop-in ASE calculator)::

    from m3gnet_fixed import make_calculator
    atoms.calc = make_calculator("fixed", device="cuda")

Property regression (formation energy, band gap, shear and bulk moduli)::

    from m3gnet_fixed.regression import load_regression, predict
    model, norm = load_regression()
    predict(model, norm, structure)

Both shipped models were fitted in the corrected index space and must be
evaluated in it -- the loaders above see to that. To fix a *stock* matgl install
instead, so that the published models and your own training runs are corrected
too, use the standalone ``m3gnet_threebody_patch.py`` at the top of this
package.

``models/*/PROVENANCE.json`` says what the weights are; ``m3gnet_fixed/pruning.py``
says what "corrected index space" means and why it matters.
"""

from .calculator import (
    ARMS,
    PES_MODEL_DIR,
    PUBLISHED_MODEL,
    WIRINGS,
    M3GNetCalculator,
    load_potential,
    make_calculator,
)
from .pruning import (
    assert_lg_invariants,
    build_empty_bundle,
    build_fixed_bundle,
    install_segment_index_fix,
    segment_indices_from_n,
    to_full_bond_space,
)
from .regression import REGRESSION_MODEL_DIR, TARGETS, load_regression, predict

__version__ = "1.0.0"

__all__ = [
    # potential
    "ARMS", "PES_MODEL_DIR", "PUBLISHED_MODEL", "WIRINGS",
    "M3GNetCalculator", "load_potential", "make_calculator",
    # regression
    "REGRESSION_MODEL_DIR", "TARGETS", "load_regression", "predict",
    # the fix
    "assert_lg_invariants", "build_empty_bundle", "build_fixed_bundle",
    "install_segment_index_fix", "segment_indices_from_n", "to_full_bond_space",
    "__version__",
]
