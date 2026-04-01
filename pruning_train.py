from __future__ import annotations
import json
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from dgl.data.utils import split_dataset
from matgl.ext.pymatgen import Structure2Graph
import warnings
from functools import partial
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping
import numpy as np
from dgl.data.utils import split_dataset
from pytorch_lightning.loggers import CSVLogger
from matgl.config import DEFAULT_ELEMENTS
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_pes
from matgl.models import M3GNet
from matgl.utils.training import PotentialLightningModule
import torch
import random
import torch
from ase.stress import voigt_6_to_full_3x3_stress
warnings.simplefilter("ignore")
import collections
from tqdm import tqdm
from matgl.utils.training import PotentialLightningModule, xavier_init
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from lightning.pytorch.callbacks import ModelCheckpoint
import dgl

PRUNE_RATIO = 0.6
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

with open("../MatPES-R2SCAN-atoms.json", "r") as f:
    isolated_energies_pbe = json.load(f)
isolated_energies_pbe = {d["elements"][0]: d["energy"] for d in isolated_energies_pbe}

with open("../MatPES-R2SCAN-2025.1.json", "r") as f:
    data = json.load(f)

structures = []
labels = collections.defaultdict(list)
for d in tqdm(data):
    structures.append(Structure.from_dict(d["structure"]))
    labels["energies"].append(d["energy"])
    labels["forces"].append(d["forces"])
    labels["stresses"].append((voigt_6_to_full_3x3_stress(np.array(d["stress"])) * -0.1).tolist())

element_types = DEFAULT_ELEMENTS
converter = Structure2Graph(element_types=element_types, cutoff=5.0)
dataset = MGLDataset(structures=structures,include_line_graph=True,threebody_cutoff=4.0, converter=converter, labels=labels, save_cache=False)

def prune_line_graph(dataset, prune_ratio, seed=42):
    if prune_ratio == 0.0:
        return dataset
    
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    total_edges = 0
    pruned_edges = 0

    for idx, line_graph in enumerate(dataset.line_graphs):
        num_edges = line_graph.num_edges()
        total_edges += num_edges

        if num_edges == 0:
            continue

        num_to_prune = int(num_edges * prune_ratio)
        if num_to_prune == 0:
            continue

        edge_perm = torch.randperm(num_edges, generator=generator)
        
        mask_keep = torch.zeros(num_edges, dtype=torch.bool)
        mask_keep[edge_perm[num_to_prune:]] = True

        src, dst = line_graph.edges(order="eid")

        src_total = torch.bincount(src, minlength=line_graph.num_nodes())
        src_kept = torch.bincount(src[mask_keep], minlength=line_graph.num_nodes()) if mask_keep.any() else torch.zeros_like(src_total)
        depleted_nodes = torch.nonzero((src_total > 0) & (src_kept == 0), as_tuple=False).flatten().tolist()

        if depleted_nodes:
            for eid in edge_perm.tolist():
                src_node = int(src[eid])
                if src_node in depleted_nodes and not mask_keep[eid]:
                    mask_keep[eid] = True
                    depleted_nodes.remove(src_node)
                    if not depleted_nodes:
                        break

        keep_idx = mask_keep.nonzero(as_tuple=True)[0]
        keep_idx, _ = torch.sort(keep_idx)
        src_keep = src[keep_idx]
        dst_keep = dst[keep_idx]

        device = line_graph.device
        new_line_graph = dgl.graph((src_keep.to(device), dst_keep.to(device)), num_nodes=line_graph.num_nodes(), device=device)

        for key, value in line_graph.ndata.items():
            if key == "n_triple_ij":
                continue
            new_line_graph.ndata[key] = value.clone()

        if "n_triple_ij" in line_graph.ndata:
            if src_keep.numel() == 0:
                new_counts = torch.zeros(line_graph.num_nodes(), dtype=line_graph.ndata["n_triple_ij"].dtype, device=device)
            else:
                new_counts = torch.bincount(src_keep, minlength=line_graph.num_nodes()).to(device=device)
                new_counts = new_counts.to(line_graph.ndata["n_triple_ij"].dtype)
            new_line_graph.ndata["n_triple_ij"] = new_counts

        if len(line_graph.edata) > 0:
            for key, value in line_graph.edata.items():
                new_line_graph.edata[key] = value[keep_idx].clone()

        dataset.line_graphs[idx] = new_line_graph
        pruned_edges += num_edges - keep_idx.numel()

    return dataset

edges_before = sum(lg.num_edges() for lg in dataset.line_graphs)
print(f"Total line graph edges before pruning: {edges_before}")
dataset = prune_line_graph(dataset, prune_ratio=PRUNE_RATIO, seed=42)
edges_after = sum(lg.num_edges() for lg in dataset.line_graphs)
print(f"Total line graph edges after pruning:  {edges_after}")
print(f"Edges removed: {edges_before - edges_after}")
print(f"Actual prune ratio: {(edges_before - edges_after) / edges_before:.2%}")

training_set, validation_set, test_set = split_dataset(dataset, frac_list=[0.9, 0.05, 0.05], random_state=42, shuffle=True)
collate_fn = partial(collate_fn_pes, include_line_graph=True, include_stress=True)
train_loader, val_loader, test_loader = MGLDataLoader(train_data=training_set, val_data=validation_set, test_data=test_set, collate_fn=collate_fn, batch_size=32,num_workers=0,)
model = M3GNet(element_types=element_types,is_intensive=False,use_smooth=True,units=128)

train_graphs = []
energies = []
forces = []

for _g, _lat, _line_graph, _attrs, lbs in training_set:
    forces.append(lbs["forces"])
forces = torch.concatenate(forces)
rms_forces = torch.sqrt(torch.mean(torch.sum(forces**2, dim=1)))

xavier_init(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5, amsgrad=True)
scheduler = CosineAnnealingLR(optimizer, T_max=1000 * 10, eta_min=1.0e-2 * 1.0e-3)
energies_offsets = np.array([isolated_energies_pbe[element] for element in DEFAULT_ELEMENTS])
lit_model = PotentialLightningModule(
    model=model,
    element_refs=energies_offsets,
    data_std=rms_forces,
    optimizer=optimizer,
    scheduler=scheduler,
    loss="l1_loss",
    stress_weight=0.1,
    include_line_graph=True,
)
path = os.getcwd()
DIR = f"m3gnet-{int(PRUNE_RATIO * 100)}"
logger = CSVLogger(save_dir=path, name=DIR)

checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_Total_Loss",
    mode="min",
    filename="{epoch:04d}-best_model",
)

trainer = pl.Trainer(
    logger=logger,
    callbacks=[EarlyStopping(monitor="val_Total_Loss", mode="min", patience=200), checkpoint_callback],
    max_epochs=200,
    accelerator="gpu",  
    gradient_clip_val=2.0,
    accumulate_grad_batches=4,
    devices=2,
    inference_mode=False    
)
trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
model_export_path = os.path.join(path, DIR, "trained_model")
os.makedirs(model_export_path, exist_ok=True)
lit_model.model.save(model_export_path)

trainer.test(dataloaders=test_loader)




