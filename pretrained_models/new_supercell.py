from pymatgen.core import Structure
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import pandas as pd
from pymatgen.core import Structure
import matgl
from matgl.ext.ase import MolecularDynamics
from ase.io.trajectory import TrajectoryReader
import sys
import logging

import torch
torch.multiprocessing.set_start_method('spawn')
torch.set_default_device("cuda")
trajectories = './trajectories_ver1'
MODEL_DIR = 'trained_model'

pot = matgl.load_model(MODEL_DIR)
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

import argparse

parser = argparse.ArgumentParser(description='run md')
parser.add_argument('--index', type=int)
args = parser.parse_args()

def get_dgl_graph_size(g):
    size = 0
    # Node and edge data
    for key, value in g.ndata.items():
        size += value.element_size() * value.nelement()
    for key, value in g.edata.items():
        size += value.element_size() * value.nelement()
    # Graph structure (edges)
    size += g.edges()[0].element_size() * g.edges()[0].nelement()
    size += g.edges()[1].element_size() * g.edges()[1].nelement()
    return size

df = pd.read_json('mp_2025_1k.json')
i = args.index
start = timeit.default_timer()
s = Structure.from_str(df['structures'][i], fmt='cif')
s = s.make_supercell(5,5,5)
s = s.to_ase_atoms()
MaxwellBoltzmannDistribution(s, temperature_K=300)
out_md = MolecularDynamics(atoms = s, potential=pot, trajectory=f"{trajectories}/trajs/traj-{i}.traj", logfile=f"{trajectories}/logs/log-{i}.log", loginterval=100).run(10000)
traj = TrajectoryReader(f"{trajectories}/trajs/traj-{i}.traj")
a  = traj[-1]
a.write(filename=f"{trajectories}/final_cifs/{i}.cif")
stop = timeit.default_timer()
dt = stop - start
print(f"Time {i}-th: {dt} s", flush=True) 
print(f"done {i}-th", flush=True)
