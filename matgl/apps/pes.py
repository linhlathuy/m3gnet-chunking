"""Implementation of Interatomic Potentials."""
 
from __future__ import annotations
 
from typing import TYPE_CHECKING
import os
import secrets
 
import torch
import dgl
from torch import nn
from torch.autograd import grad
 
import matgl
from matgl.layers import AtomRef, NuclearRepulsion
from matgl.utils.io import IOMixIn
 
if TYPE_CHECKING:
    import dgl
    import numpy as np
 
 
class Potential(nn.Module, IOMixIn):
    """A class representing an interatomic potential."""
 
    __version__ = 3
 
    def __init__(
        self,
        model: nn.Module,
        data_mean: torch.Tensor | float = 0.0,
        data_std: torch.Tensor | float = 1.0,
        element_refs: np.ndarray | None = None,
        calc_forces: bool = True,
        calc_stresses: bool = True,
        calc_hessian: bool = False,
        calc_magmom: bool = False,
        calc_repuls: bool = False,
        zbl_trainable: bool = False,
        debug_mode: bool = False,
        use_edges: bool | None = None,
        calc_stresses_override: bool | None = None,
        calc_hessian_override: bool | None = None,
        num_chunks=1,
        chunk_padding: float | None = None,
        apply_chunking: bool = False,
        write_chunk_logs: bool = True,
    ):
        """Initialize Potential from a model and elemental references.
 
        Args:
            model: Model for predicting energies.
            data_mean: Mean of target.
            data_std: Std dev of target.
            element_refs: Element reference values for each element.
            calc_forces: Enable force calculations.
            calc_stresses: Enable stress calculations.
            calc_hessian: Enable hessian calculations.
            calc_magmom: Enable site-wise property calculation.
            calc_repuls: Whether the ZBL repulsion is included
            zbl_trainable: Whether zbl repulsion is trainable
            debug_mode: Return gradient of total energy with respect to atomic positions and lattices for checking
            use_edges: If set, override model edge usage (triplets/line graph).
            calc_stresses_override: If set, override stress calculation.
            calc_hessian_override: If set, override hessian calculation.
            num_chunks: 0 disables chunking. Otherwise split each axis into (num_chunks + 1)
                parts, yielding (num_chunks + 1)^3 chunks.
            chunk_padding: Padding size around each chunk (at least cutoff).
            apply_chunking: If True, compute energies/forces from chunks only (no full-structure evaluation).
            write_chunk_logs: If False, do not write chunk XYZ files.
        """
        super().__init__()
        self.save_args(locals())
        self.model = model
        self.calc_forces = calc_forces
        self.calc_stresses = (
            calc_stresses_override
            if calc_stresses_override is not None
            else calc_stresses
        )
        self.calc_hessian = (
            calc_hessian_override if calc_hessian_override is not None else calc_hessian
        )
        self.calc_magmom = calc_magmom
        self.element_refs: AtomRef | None
        self.debug_mode = debug_mode
        self.calc_repuls = calc_repuls
        self.use_edges = use_edges
        self.num_chunks = num_chunks
        self.chunk_padding = chunk_padding
        self.apply_chunking = apply_chunking
        self.write_chunk_logs = write_chunk_logs
 
        if use_edges is not None and hasattr(self.model, "use_edges"):
            self.model.use_edges = use_edges
 
        if calc_repuls:
            self.repuls = NuclearRepulsion(self.model.cutoff, trainable=zbl_trainable)
 
        if element_refs is not None:
            self.element_refs = AtomRef(
                property_offset=torch.tensor(element_refs, dtype=matgl.float_th)
            )
        else:
            self.element_refs = None
        # for backward compatibility
        if data_mean is None:
            data_mean = 0.0
        self.register_buffer("data_mean", torch.tensor(data_mean, dtype=matgl.float_th))
        self.register_buffer("data_std", torch.tensor(data_std, dtype=matgl.float_th))
 
    def _interval_mask(
        self, frac_z: torch.Tensor, start: float, end: float
    ) -> torch.Tensor:
        if start <= end:
            return (frac_z >= start) & (frac_z < end)
        return (frac_z >= start) | (frac_z < end)
 
    def _interval_mask_with_pad(
        self, frac_z: torch.Tensor, start: float, end: float, pad: float
    ) -> torch.Tensor:
        start_pad = start - pad
        end_pad = end + pad
        if start_pad < 0.0:
            return (frac_z >= start_pad + 1.0) | (frac_z < end_pad)
        if end_pad > 1.0:
            return (frac_z >= start_pad) | (frac_z < end_pad - 1.0)
        return (frac_z >= start_pad) & (frac_z < end_pad)
 
    def _compute_energy_forces(
        self,
        g: dgl.DGLGraph,
        lattice: torch.Tensor,
        state_attr: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g.edata["lattice"] = torch.repeat_interleave(
            lattice, g.batch_num_edges(), dim=0
        )
        g.edata["pbc_offshift"] = (
            g.edata["pbc_offset"].unsqueeze(dim=-1) * g.edata["lattice"]
        ).sum(dim=1)
        g.ndata["pos"] = (
            g.ndata["frac_coords"].unsqueeze(dim=-1)
            * torch.repeat_interleave(lattice, g.batch_num_nodes(), dim=0)
        ).sum(dim=1)
        g.ndata["pos"].requires_grad_(True)
        # l_g = create_line_graph(g, 4)
        l_g = None if self.use_edges is False else None
        total_energies = self.model(g=g, state_attr=state_attr, l_g=l_g)
        total_energies = self.data_std * total_energies + self.data_mean
        if self.calc_repuls:
            total_energies += self.repuls(self.model.element_types, g)
        if self.element_refs is not None:
            property_offset = torch.squeeze(self.element_refs(g))
            total_energies += property_offset
        grads = grad(
            total_energies,
            [g.ndata["pos"]],
            grad_outputs=torch.ones_like(total_energies),
            create_graph=False,
            retain_graph=False,
        )
        forces = -grads[0]
        return total_energies, forces
 
    def _write_chunk_xyz(
        self,
        path: str,
        symbols: list[str],
        positions: torch.Tensor,
        forces_chunk: torch.Tensor,
        forces_full: torch.Tensor,
        energy: torch.Tensor,
    ) -> None:
        positions = positions.detach().cpu().numpy()
        forces_chunk = forces_chunk.detach().cpu().numpy()
        forces_full = forces_full.detach().cpu().numpy()
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"{len(symbols)}\n")
            handle.write(f"energy={energy.item():.8f}\n")
            for sym, pos, frc_chunk, frc_full in zip(
                symbols, positions, forces_chunk, forces_full
            ):
                handle.write(
                    f"{sym} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f} "
                    f"{frc_chunk[0]:.8f} {frc_chunk[1]:.8f} {frc_chunk[2]:.8f} "
                    f"{frc_full[0]:.8f} {frc_full[1]:.8f} {frc_full[2]:.8f}\n"
                )
 
    def forward(
        self,
        g: dgl.DGLGraph,
        lat: torch.Tensor,
        state_attr: torch.Tensor | None = None,
        l_g: dgl.DGLGraph | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Args:
            g: DGL graph
            lat: lattice
            state_attr: State attrs
            l_g: Line graph.
 
        Returns:
            (energies, forces, stresses, hessian) or (energies, forces, stresses, hessian, site-wise properties)
        """
        # st (strain) for stress calculations
        st = lat.new_zeros([g.batch_size, 3, 3])
        if self.calc_stresses:
            st.requires_grad_(True)
        lattice = lat @ (torch.eye(3, device=lat.device) + st)
        if self.apply_chunking:
            if self.num_chunks is None:
                raise RuntimeError("apply_chunking=True requires num_chunks to be set.")
            if self.num_chunks == 0:
                self.apply_chunking = False
                return self.forward(g=g, lat=lat, state_attr=state_attr, l_g=l_g)
            if g.batch_size != 1:
                raise RuntimeError(
                    "Chunking only supports a single structure (batch_size=1)."
                )
            device = lat.device
            g_cpu = g if g.device.type == "cpu" else g.to("cpu")
            a_len = torch.linalg.norm(lattice[0][0]).item()
            b_len = torch.linalg.norm(lattice[0][1]).item()
            c_len = torch.linalg.norm(lattice[0][2]).item()
            if self.chunk_padding is None:
                nblocks = int(getattr(self.model, "n_blocks", 1))
                # nblocks = 1
                pad = float(self.model.cutoff) * nblocks + 1
                if self.use_edges is not False:
                    pad = max(pad, float(getattr(self.model, "threebody_cutoff", 0.0)))
            else:
                pad = self.chunk_padding
            # pad = max(pad, float(self.model.cutoff))
            pad_frac_a = pad / a_len if a_len > 0 else 0.0
            pad_frac_b = pad / b_len if b_len > 0 else 0.0
            pad_frac_c = pad / c_len if c_len > 0 else 0.0
            frac_coords = g_cpu.ndata["frac_coords"]
            frac_coords = frac_coords - torch.floor(frac_coords)
            divisions = self.num_chunks + 1
            chunk_width = 1.0 / float(divisions)
            run_id = None
            if self.write_chunk_logs:
                os.makedirs("chunking_log", exist_ok=True)
                run_id = os.path.join(
                    "chunking_log",
                    f"chunks_apply{int(self.apply_chunking)}_chunks{self.num_chunks}_pad{self.chunk_padding}_"
                    + secrets.token_hex(4),
                )
                os.makedirs(run_id, exist_ok=True)
            full_forces = torch.zeros_like(g_cpu.ndata["frac_coords"]).to(device)
            total_energy = None
            for ix in range(divisions):
                start_x = ix * chunk_width
                end_x = (ix + 1) * chunk_width
                mask_x = self._interval_mask(frac_coords[:, 0], start_x, end_x)
                if not torch.any(mask_x):
                    continue
                pad_x = self._interval_mask_with_pad(
                    frac_coords[:, 0], start_x, end_x, pad_frac_a
                )
                for iy in range(divisions):
                    start_y = iy * chunk_width
                    end_y = (iy + 1) * chunk_width
                    mask_y = self._interval_mask(frac_coords[:, 1], start_y, end_y)
                    if not torch.any(mask_y):
                        continue
                    pad_y = self._interval_mask_with_pad(
                        frac_coords[:, 1], start_y, end_y, pad_frac_b
                    )
                    for iz in range(divisions):
                        start_z = iz * chunk_width
                        end_z = (iz + 1) * chunk_width
                        mask_z = self._interval_mask(
                            frac_coords[:, 2], start_z, end_z
                        )
                        core_mask = mask_x & mask_y & mask_z
                        if not torch.any(core_mask):
                            continue
                        pad_z = self._interval_mask_with_pad(
                            frac_coords[:, 2], start_z, end_z, pad_frac_c
                        )
                        pad_mask = pad_x & pad_y & pad_z

                        core_count = int(core_mask.sum().item())
                        pad_only = pad_mask & (~core_mask)
                        pad_count = int(pad_only.sum().item())
                        src, dst = g_cpu.edges()
                        edge_core_pad = (
                            (core_mask[src] & pad_only[dst])
                            | (core_mask[dst] & pad_only[src])
                        )
                        edge_core_pad_count = int(edge_core_pad.sum().item())
                        print(f"{ix}       {iy}       {iz}       {core_count}       {pad_count}    {edge_core_pad_count}")

                        subg_cpu = dgl.node_subgraph(g_cpu, pad_mask)
                        orig_ids = subg_cpu.ndata[dgl.NID]
                        core_subg_mask = core_mask[orig_ids]
                        if not torch.any(core_subg_mask):
                            continue
                        subg = subg_cpu.to(device)
                        core_subg_mask_dev = core_subg_mask.to(device)
                        chunk_energy, chunk_forces = self._compute_energy_forces(
                            subg, lattice, state_attr
                        )
                        core_frac = subg.ndata["frac_coords"][core_subg_mask_dev]
                        core_frac = core_frac - torch.floor(core_frac)
                        core_positions = core_frac @ lattice[0]
                        core_forces = chunk_forces[core_subg_mask_dev]
                        core_orig_ids = orig_ids[core_subg_mask].to(device)
                        full_forces[core_orig_ids] = core_forces
                        node_types = subg.ndata["node_type"][core_subg_mask_dev]
                        symbols = [
                            self.model.element_types[int(i)]
                            for i in node_types.detach().cpu().tolist()
                        ]
                        if self.write_chunk_logs and run_id is not None:
                            chunk_id = ix * divisions * divisions + iy * divisions + iz
                            fname = os.path.join(run_id, f"{chunk_id}.xyz")
                            self._write_chunk_xyz(
                                fname,
                                symbols,
                                core_positions,
                                core_forces,
                                full_forces[core_orig_ids],
                                chunk_energy,
                            )
            with g.local_scope(), torch.no_grad():
                g.edata["lattice"] = torch.repeat_interleave(
                    lattice, g.batch_num_edges(), dim=0
                )
                g.edata["pbc_offshift"] = (
                    g.edata["pbc_offset"].unsqueeze(dim=-1) * g.edata["lattice"]
                ).sum(dim=1)
                g.ndata["pos"] = (
                    g.ndata["frac_coords"].unsqueeze(dim=-1)
                    * torch.repeat_interleave(lattice, g.batch_num_nodes(), dim=0)
                ).sum(dim=1)
                total_energy = self.model(g=g, state_attr=state_attr, l_g=None)
                total_energy = self.data_std * total_energy + self.data_mean
                if self.calc_repuls:
                    total_energy += self.repuls(self.model.element_types, g)
                if self.element_refs is not None:
                    property_offset = torch.squeeze(self.element_refs(g))
                    total_energy += property_offset
            if total_energy is None:
                total_energy = torch.zeros(1, device=lat.device)
            stresses = torch.zeros(1)
            hessian = torch.zeros(1)
            return total_energy, full_forces, stresses, hessian
 
 
        g.edata["lattice"] = torch.repeat_interleave(
            lattice, g.batch_num_edges(), dim=0
        )
        g.edata["pbc_offshift"] = (
            g.edata["pbc_offset"].unsqueeze(dim=-1) * g.edata["lattice"]
        ).sum(dim=1)
        g.ndata["pos"] = (
            g.ndata["frac_coords"].unsqueeze(dim=-1)
            * torch.repeat_interleave(lattice, g.batch_num_nodes(), dim=0)
        ).sum(dim=1)
        if self.calc_forces:
            g.ndata["pos"].requires_grad_(True)
 
        if self.use_edges is False:
            l_g = None
        total_energies = self.model(g=g, state_attr=state_attr, l_g=l_g)
 
        total_energies = self.data_std * total_energies + self.data_mean
 
        if self.calc_repuls:
            total_energies += self.repuls(self.model.element_types, g)
 
        if self.element_refs is not None:
            property_offset = torch.squeeze(self.element_refs(g))
            total_energies += property_offset
 
        forces = torch.zeros(1)
        stresses = torch.zeros(1)
        hessian = torch.zeros(1)
 
        grad_vars = [g.ndata["pos"], st] if self.calc_stresses else [g.ndata["pos"]]
 
        if self.calc_forces:
            grads = grad(
                total_energies,
                grad_vars,
                grad_outputs=torch.ones_like(total_energies),
                create_graph=True,
                retain_graph=True,
            )
            forces = -grads[0]
 
        if self.calc_hessian:
            r = -grads[0].view(-1)
            s = r.size(0)
            hessian = total_energies.new_zeros((s, s))
            for iatom in range(s):
                tmp = grad([r[iatom]], g.ndata["pos"], retain_graph=iatom < s)[0]
                if tmp is not None:
                    hessian[iatom] = tmp.view(-1)
 
        if self.calc_stresses:
            volume = (
                torch.abs(torch.det(lattice.float())).half()
                if matgl.float_th == torch.float16
                else torch.abs(torch.det(lattice))
            )
            sts = -grads[1]
            scale = 1.0 / volume * -160.21766208
            sts = (
                [i * j for i, j in zip(sts, scale)] if sts.dim() == 3 else [sts * scale]
            )
            stresses = torch.cat(sts)
 
        if self.debug_mode:
            return total_energies, -grads[0], grads[1]
 
        if self.calc_magmom:
            return total_energies, forces, stresses, hessian, g.ndata["magmom"]
 
        if self.num_chunks is not None and self.num_chunks != 0:
            if g.batch_size != 1:
                raise RuntimeError(
                    "Chunking only supports a single structure (batch_size=1)."
                )
            g_cpu = g if g.device.type == "cpu" else g.to("cpu")
            lattice = lat @ (torch.eye(3, device=lat.device) + st)
            a_len = torch.linalg.norm(lattice[0][0]).item()
            b_len = torch.linalg.norm(lattice[0][1]).item()
            c_len = torch.linalg.norm(lattice[0][2]).item()
            if self.chunk_padding is None:
                nblocks = int(getattr(self.model, "n_blocks", 1))
                pad = float(self.model.cutoff) * nblocks
                if self.use_edges is not False:
                    pad = max(pad, float(getattr(self.model, "threebody_cutoff", 0.0)))
            else:
                pad = self.chunk_padding
            # pad = max(pad, float(self.model.cutoff))
            pad_frac_a = pad / a_len if a_len > 0 else 0.0
            pad_frac_b = pad / b_len if b_len > 0 else 0.0
            pad_frac_c = pad / c_len if c_len > 0 else 0.0
            frac_coords = g.ndata["frac_coords"]
            frac_coords = frac_coords - torch.floor(frac_coords)
            divisions = self.num_chunks + 1
            chunk_width = 1.0 / float(divisions)
            run_id = None
            if self.write_chunk_logs:
                os.makedirs("chunking_log", exist_ok=True)
                run_id = os.path.join(
                    "chunking_log",
                    f"chunks_apply{int(self.apply_chunking)}_chunks{self.num_chunks}_pad{self.chunk_padding}_"
                    + secrets.token_hex(4),
                )
                os.makedirs(run_id, exist_ok=True)
            for ix in range(divisions):
                start_x = ix * chunk_width
                end_x = (ix + 1) * chunk_width
                mask_x = self._interval_mask(frac_coords[:, 0], start_x, end_x)
                if not torch.any(mask_x):
                    continue
                pad_x = self._interval_mask_with_pad(
                    frac_coords[:, 0], start_x, end_x, pad_frac_a
                )
                for iy in range(divisions):
                    start_y = iy * chunk_width
                    end_y = (iy + 1) * chunk_width
                    mask_y = self._interval_mask(frac_coords[:, 1], start_y, end_y)
                    if not torch.any(mask_y):
                        continue
                    pad_y = self._interval_mask_with_pad(
                        frac_coords[:, 1], start_y, end_y, pad_frac_b
                    )
                    for iz in range(divisions):
                        start_z = iz * chunk_width
                        end_z = (iz + 1) * chunk_width
                        mask_z = self._interval_mask(
                            frac_coords[:, 2], start_z, end_z
                        )
                        core_mask = mask_x & mask_y & mask_z
                        if not torch.any(core_mask):
                            continue
                        pad_z = self._interval_mask_with_pad(
                            frac_coords[:, 2], start_z, end_z, pad_frac_c
                        )
                        pad_mask = pad_x & pad_y & pad_z

                        core_count = int(core_mask.sum().item())
                        pad_only = pad_mask & (~core_mask)
                        pad_count = int(pad_only.sum().item())
                        src, dst = g_cpu.edges()
                        edge_core_pad = (
                            (core_mask[src] & pad_only[dst])
                            | (core_mask[dst] & pad_only[src])
                        )
                        edge_core_pad_count = int(edge_core_pad.sum().item())
                        print(f"{ix}   {iy}   {iz}       {core_count}       {pad_count}    {edge_core_pad_count}")

                        subg = dgl.node_subgraph(g, pad_mask)
                        orig_ids = subg.ndata[dgl.NID]
                        core_subg_mask = core_mask[orig_ids]
                        # print(f"core subg mask: {core_subg_mask}", flush = True)
                        if not torch.any(core_subg_mask):
                            continue
                        chunk_energy, chunk_forces = self._compute_energy_forces(
                            subg, lattice, state_attr
                        )
                        core_frac = subg.ndata["frac_coords"][core_subg_mask]
                        core_frac = core_frac - torch.floor(core_frac)
                        core_positions = core_frac @ lattice[0]
                        # print(f"len(core_positions)= {len(core_positions)}", flush = True)
                        core_forces = chunk_forces[core_subg_mask]
                        core_orig_ids = orig_ids[core_subg_mask]
                        full_forces = forces[core_orig_ids]
                        node_types = subg.ndata["node_type"][core_subg_mask]
                        symbols = [
                            self.model.element_types[int(i)]
                            for i in node_types.detach().cpu().tolist()
                        ]
                        if self.write_chunk_logs and run_id is not None:
                            chunk_id = ix * divisions * divisions + iy * divisions + iz
                            fname = os.path.join(run_id, f"{chunk_id}.xyz")
                            self._write_chunk_xyz(
                                fname,
                                symbols,
                                core_positions,
                                core_forces,
                                full_forces,
                                chunk_energy,
                            )
 
        return total_energies, forces, stresses, hessian
