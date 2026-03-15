from pymatgen.core import Structure
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.core import Structure
import matgl
from matgl.ext.ase import MolecularDynamics
from ase.io.trajectory import TrajectoryReader

import matgl
import torch

# Configure backend before importing backend-specific ASE wrappers.
# matgl.set_backend("DGL")
# from matgl.ext.ase import MolecularDynamics

torch.multiprocessing.set_start_method('spawn')
torch.set_default_device("cuda")
trajectories = './trajectories_ver_LPGS'
MODEL_DIR = 'trained_model'

pot = matgl.load_model(MODEL_DIR)
pot.write_chunk_logs = True
pot.debug_mode = False
pot.apply_chunking = False
pot.num_chunks = 0
pot.chunk_padding = 5
print(f"num_chunks = {pot.num_chunks}")
print(f"chunk_padding = {pot.chunk_padding}")
print(f"write_chunk_logs: {pot.write_chunk_logs}")
print(f"debug_mode: {pot.debug_mode}")

print(pot, flush=True)
import timeit
folders = [f"{trajectories}/logs",f"{trajectories}/final_cifs", f"{trajectories}/trajs"]
import os
for f in folders:
    if not os.path.exists(f):
        os.makedirs(f)
        print(f"Created: {f}")
    else:
        print(f"Exists: {f}")

i = 5
s = Structure.from_file('structures/LPGS.cif')
print(f"supercell size = {i} x {i} x {i}")
s = s.make_supercell(i,i,i)
s = s.to_ase_atoms()

MaxwellBoltzmannDistribution(s, temperature_K=1000)
import time

md = MolecularDynamics(
    atoms=s,
    potential=pot,
    ensemble="nvt",
    temperature=1000,
    trajectory=f"{trajectories}/trajs/traj-{i}-{pot.chunk_padding}-{pot.num_chunks}.traj",
    logfile=f"{trajectories}/logs/log-{i}-{pot.chunk_padding}-{pot.num_chunks}.log",
    loginterval=10,
)

t0 = time.perf_counter()
t_prev = [time.perf_counter()]  # mutable container

def log_step_time():
    now = time.perf_counter()
    step_time = now - t_prev[0]
    t_prev[0] = now
    print(f"step={md.dyn.nsteps} step_wall={step_time:.3f}s", flush=True)

md.dyn.attach(log_step_time, interval=1)  # call every MD step
md.run(4000)
