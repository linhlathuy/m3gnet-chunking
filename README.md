# M3GNet with a corrected three-body index space

Two retrained M3GNet models, the code to run them, and a one-file patch that
fixes the underlying defect in any matgl install.

Every released version of [matgl](https://github.com/materialsvirtuallab/matgl)
— v0.1.0 through v4.0.3 — hands M3GNet's three-body layer indices from one
numbering scheme while the tensors it indexes are built in another. The angular
terms are then scattered into the wrong bonds and weighted by the wrong cutoffs,
silently: no exception is raised, and the model trains and runs to convergence
around the error.


## What is here

```
m3gnet_threebody_patch.py   the fix for a stock matgl install, standalone
m3gnet_fixed/               the library: calculator, regression, chunking
models/pes/                 potential — energy, forces, stress
models/regression/          properties — formation energy, gap, moduli
examples/run_md.py          molecular dynamics
examples/chunk_forces.py    spatial chunking for cells too large to fit
examples/triplet_removal.py the three-body ablation
tests/                      20 tests; every claim below is one of them
tools/export_regression.py  how models/regression was written, for the record
environment.yml             the environment, pinned (requirements.txt for pip)
```

## Install

```bash
conda env create -f environment.yml
conda activate m3gnet-fixed
pytest tests/ -q                        # ~4 min
python examples/run_md.py --check       # ~1 min, proves the models load
```

`matgl` is pinned to 4.0.2 on purpose. There is no later release to upgrade to
that fixes this — the defect is present in all 64 of them.

---

## The defect

`M3GNet.forward` builds its line graph with `create_line_graph`, which drops
every bond longer than `threebody_cutoff` and numbers the survivors
`0..n_kept-1`. The triples come back in that **pruned** numbering. `forward`
then passes them to `ThreeBodyInteractions` alongside two tensors built over
**all** bonds:

```python
edge_dst_atom     = edge_index[1]                     # full bond space
three_body_cutoff = polynomial_cutoff(bond_dist, rc)  # full bond space
line_edge_index   = l_g["line_edge_index"]            # PRUNED bond space
```

The bundle already carries the translation table that reconciles the two spaces
— `kept_edge_ids`. No released matgl has ever read it.

## The fix

Two changes, neither of them to the three-body layer:

1. **Remap the bundle into full parent-bond space** through `kept_edge_ids`.
   Every tensor `forward` indexes then shares one space and matgl's existing
   code becomes self-consistent — `forward` itself needs no edit.

2. **Repair `get_segment_indices_from_n`.** The remap makes `n_triple_ij` length
   `num_bonds`, with zeros for bonds in no triple, and the stock implementation
   is wrong for any empty segment:

   | `ns` | stock matgl | correct |
   |---|---|---|
   | `[2, 3, 1]` | `[0,0,1,1,1,2]` | same |
   | `[2, 0, 3]` | `[0,0,1,1,1]` **wrong** | `[0,0,2,2,2]` |
   | `[2, 3, 0]` | `IndexError` | `[0,0,1,1,1]` |

   Interior zeros are the dangerous case — no exception, angular basis scattered
   into the wrong bonds. The replacement is provably identical to the original
   whenever no segment is empty, so it cannot change a result that was already
   correct.

### Patching your own matgl

`m3gnet_threebody_patch.py` is self-contained and depends on nothing in this
repo. **Put it anywhere on your `PYTHONPATH`** — next to your run script is
usual. It patches matgl from the outside and never needs to live inside the
matgl tree.

Runtime, no files touched — call it before you load a model:

```python
import m3gnet_threebody_patch
m3gnet_threebody_patch.install()
```

Or edit the installed matgl once, so every process is fixed:

```bash
python m3gnet_threebody_patch.py --check      # report status, change nothing
python m3gnet_threebody_patch.py --apply      # patch, keeping .orig backups
python m3gnet_threebody_patch.py --revert     # restore from those backups
```

`--apply` finds matgl itself and appends a guarded block to exactly two files:

| file | what changes |
|---|---|
| `matgl/graph/_compute.py` | `create_line_graph` returns a full-bond-space bundle |
| `matgl/utils/maths.py` | `get_segment_indices_from_n` handles empty segments |

It never rewrites existing lines, so it re-runs safely and reads clearly in a
diff. `--revert` restores both files byte-identically (a test asserts this).

**A patched matgl changes what your existing weights mean.** Any model trained
under the defect was fitted in the broken index space; correcting the space at
inference moves it off-distribution. Patch matgl to *train* a correct model, or
to run the models shipped here — not to "improve" a published checkpoint you
then evaluate as though nothing changed. That is what the two arms below are for.

---

## The models

Both were trained with the corrected wiring and must be evaluated with it. The
loaders in `m3gnet_fixed` handle that; `models/*/PROVENANCE.json` records the
architecture, the source checkpoint's md5, and the held-out metrics.

### Potential — `models/pes`

M3GNet PES retrained on MatPES-r2SCAN (349,107 training structures, 89 elements,
285,469 parameters, cutoff 5.0 Å / three-body 4.0 Å).

| held-out test | this model | published `M3GNet-PES-MatPES-r2SCAN-2025.2` |
|---|---|---|
| energy MAE | **0.0325** eV/atom | 0.0435 eV/atom |
| force MAE | **0.1513** eV/Å | 0.2184 eV/Å |
| stress MAE | **0.742** GPa | 0.976 GPa |

The published figures are that model's own reported card values, on its own
split — this model's split is not identical, so read the comparison as
indicative rather than as a controlled head-to-head.

```python
from m3gnet_fixed import make_calculator
atoms.calc = make_calculator("fixed", device="cuda")
atoms.get_potential_energy(), atoms.get_forces()
```

`make_calculator("published")` loads the released model instead, in stock
wiring — the space *its* weights were fitted in, and so the right way to run it.
That is the comparison arm.

### Regression — `models/regression`

M3GNet trained on 70,552 Materials Project entries for four properties at once
(391,967 parameters, same cutoffs).

| target | unit | test MAE | RMSE |
|---|---|---|---|
| formation energy | eV/atom | 0.0750 | 0.1280 |
| band gap | eV | 0.2902 | 0.5419 |
| shear modulus `log10_G_VRH` | log₁₀(GPa) | 0.0897 | 0.1375 |
| bulk modulus `log10_K_VRH` | log₁₀(GPa) | 0.0696 | 0.1227 |

```python
from m3gnet_fixed.regression import load_regression, predict
model, norm = load_regression()
predict(model, norm, structure, linear=True)
```

On conventional rocksalt cells, against experiment:

| | formation energy | gap | K |
|---|---|---|---|
| NaCl | −2.12 (exp. −2.05) eV/atom | 5.02 (5.2) eV | 25.0 (~25) GPa |
| MgO | −3.23 (−3.0) eV/atom | 4.47 (4.7) eV | 165 (~160) GPa |

Moduli are trained and returned as log₁₀(GPa); `linear=True` adds `G_VRH` and
`K_VRH` in GPa. The model outputs standardized targets, and `predict`
un-standardizes them using `normalizer.json` — skip that step and you get
z-scores, which look plausible and are wrong.

Give it the structure you mean. The moduli heads saw only 8,787 labelled
entries against 55,347 for energy and gap, so they are the least well
constrained of the four; and a primitive cell that is not actually the phase you
have in mind (two atoms in a cubic box is not rocksalt) will be predicted
faithfully as the thing you passed.

---

## Molecular dynamics

```bash
python examples/run_md.py --check
python examples/run_md.py --structure my.cif --supercell 3 3 3 \
    --ensemble nvt --temperature 1000 --steps 4000 --device cuda
python examples/run_md.py --ensemble npt --temperature 300 --pressure 1.0
```

NVT (Berendsen), NVE and NPT. Writes `trajectory.traj`, a `log.csv` carrying
step/temperature/energies/volume, and a `run.json` recording every argument and
the resolved model provenance. Under NVE, `E_tot` in the log is the conserved
quantity — plot it to check the timestep.

Berendsen is not a canonical thermostat and does not sample the canonical
ensemble; it is here because it is what the comparison runs used. Do not read a
single trajectory as a converged average.

**One arm per process.** The corrected wiring installs a global monkeypatch that
must not be live while a stock-wiring model runs. To compare arms, run the
script twice.

## Chunking

Evaluate a cell too large to fit in memory, one spatial box at a time. Split the
cell into `(chunks+1)³` boxes; for each, take the atoms inside plus every atom
within `pad` Å, and evaluate that padded subgraph. Peak memory is set by the
largest box, not by the cell.

```bash
python examples/chunk_forces.py --supercell 6 6 6 --pad-sweep 3 5 8 12
```

Reproduced by that command on rattled NaCl (1728 atoms, 33.8 Å, 8 boxes):

| halo | force rms error vs unchunked | net force |
|---|---|---|
| 3 Å | 33.24 meV/Å | 4.8e-04 |
| 5 Å | 33.24 meV/Å | 1.2e-03 |
| 8 Å | 3.99 meV/Å | 7.2e-04 |
| **12 Å** | **0.0002 meV/Å** | 1.9e-03 |

## Triplet removal

Keep the weights, hand the model a line graph with no triples, and measure how
far the prediction moves. One force call per arm.

```bash
python examples/triplet_removal.py --both
```




