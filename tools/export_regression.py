#!/usr/bin/env python
"""Export the multi-target regression checkpoint into a loadable model directory.

The PES model exports through ``matgl``'s own ``Potential.save``. The regression
model cannot: it is a bare ``M3GNet`` with ``is_intensive=True, ntargets=4``,
wrapped at training time in a Lightning module that also owns the per-target
normalizer. So this writes the pieces explicitly -- ``model.json`` (the
constructor arguments), ``state.pt`` (the weights) and ``normalizer.json`` (mean
and std per target) -- and ``m3gnet_fixed.regression.load_regression`` rebuilds
from exactly those three.

Every exported tensor is compared elementwise against the checkpoint after a
reload through the public path; a mismatch aborts the export.

    python export_regression.py --ckpt best-0196.ckpt --out ../models/regression
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import torch

TARGETS = ("eform", "bandgap", "log10_G_VRH", "log10_K_VRH")

#: Units of each target, for the record and for the README table.
UNITS = {
    "eform": "eV/atom",
    "bandgap": "eV",
    "log10_G_VRH": "log10(GPa)",
    "log10_K_VRH": "log10(GPa)",
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True, help="splits.json, for element_types")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--metrics", type=Path, default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--units", type=int, default=64)
    p.add_argument("--nblocks", type=int, default=3)
    p.add_argument("--readout", default="set2set")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)

    if a.out.exists() and any(a.out.iterdir()) and not a.force:
        raise SystemExit(f"{a.out} is not empty; pass --force to overwrite")

    meta = json.loads(a.splits.read_text()).get("meta", {})
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    state = {k[len("model.") :]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    hp = ckpt.get("hyper_parameters", {})

    config = {
        "element_types": list(meta["element_types"]),
        "is_intensive": True,
        "readout_type": a.readout,
        "ntargets": len(TARGETS),
        "units": a.units,
        "nblocks": a.nblocks,
        "cutoff": float(meta["cutoff"]),
        "threebody_cutoff": float(meta["threebody_cutoff"]),
    }

    from matgl.models import M3GNet

    model = M3GNet(**config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            "state_dict mismatch -- architecture flags likely wrong for this ckpt\n"
            f"  missing:    {missing}\n  unexpected: {unexpected}"
        )
    model.eval()

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "model.json").write_text(json.dumps(config, indent=2) + "\n")
    torch.save(state, a.out / "state.pt")

    normalizer = {
        "targets": list(TARGETS),
        "units": [UNITS[t] for t in TARGETS],
        "mean": [float(x) for x in hp["target_mean"]],
        "std": [float(x) for x in hp["target_std"]],
    }
    (a.out / "normalizer.json").write_text(json.dumps(normalizer, indent=2) + "\n")

    # Round trip through the public loading path and compare every tensor.
    reloaded = M3GNet(**json.loads((a.out / "model.json").read_text()))
    reloaded.load_state_dict(torch.load(a.out / "state.pt", map_location="cpu", weights_only=True))
    rs = reloaded.state_dict()
    bad = []
    for k, v in state.items():
        if k not in rs:
            bad.append((k, "absent after reload"))
        elif not torch.equal(rs[k].cpu(), v.cpu()):
            bad.append((k, f"max|d|={float((rs[k].cpu() - v.cpu()).abs().max()):.3e}"))
    if bad:
        raise SystemExit(
            "round-trip verification FAILED:\n"
            + "\n".join(f"  {k}: {why}" for k, why in bad)
        )

    prov = {
        "name": a.name or f"export of {a.ckpt.name}",
        "task": "multi-target property regression",
        "targets": [{"name": t, "unit": UNITS[t]} for t in TARGETS],
        "source_checkpoint": a.ckpt.name,
        "source_checkpoint_md5": md5(a.ckpt),
        "exported": date.today().isoformat(),
        "exported_by": "tools/export_regression.py",
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "architecture": {**config, "element_types": len(config["element_types"])},
        "n_parameters": sum(v.numel() for v in model.parameters()),
        "n_tensors_verified": len(state),
        "line_graph_index_space": "fixed (full parent-bond)",
        "REQUIRED_WIRING": (
            "MUST be evaluated with the three-body index fix active -- either "
            "m3gnet_threebody_patch.install() or the wiring='fixed' path in "
            "m3gnet_fixed. These weights were fitted in the corrected index "
            "space; evaluating them under stock matgl is off-distribution."
        ),
    }
    if a.metrics and a.metrics.exists():
        prov["run_metrics"] = json.loads(a.metrics.read_text())
    (a.out / "PROVENANCE.json").write_text(json.dumps(prov, indent=2) + "\n")

    print(f"wrote {a.out}")
    print(f"  {prov['n_parameters']} parameters, {len(state)} tensors verified equal")
    print(f"  epoch {prov['epoch']}, targets {', '.join(TARGETS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
