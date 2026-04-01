# M3GNet Chunking

An extension of [matgl](https://github.com/materialsvirtuallab/matgl) that enables **spatial chunking** for large-scale potential energy calculations using M3GNet and other [MatPES](https://matpes.ai/) models.

## Overview

This project modifies `matgl/apps/pes.py` to support chunked evaluation of interatomic potentials — useful for large supercells where full-system evaluation is memory-prohibitive.

Key changes:
- **Triplet interactions disabled** via `use_edges=False`
- **Spatial chunking** via `apply_chunking=True`, which partitions the simulation cell into sub-regions evaluated independently

## Chunking Parameters

| Parameter | Description |
|---|---|
| `apply_chunking` | Enable/disable chunking (`True`/`False`) |
| `num_chunks` | Subdivisions per axis; total chunks = `(num_chunks + 1)³` |
| `chunk_padding` | Padding around each chunk in Angstroms |

## Usage

### Loading a model with chunking

```python
import matgl

pot = matgl.load_model("trained_model")

pot.apply_chunking = True
pot.num_chunks = 1        # total chunks = (num_chunks + 1)^3
pot.chunk_padding = 5     # extension in Angstroms
```

### Running molecular dynamics

See [run_md.py](run_md.py) for a full example using ASE's `MolecularDynamics` interface.

```python
pot = matgl.load_model(MODEL_DIR)
pot.apply_chunking = True
pot.num_chunks = 1
pot.chunk_padding = 5
pot.write_chunk_logs = True
```
