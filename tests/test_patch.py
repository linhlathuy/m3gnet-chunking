"""Tests for the standalone matgl patch.

The in-place ``--apply`` mode is exercised against a *copy* of matgl on a
throwaway ``sys.path``, never against the installed one, so running the suite
cannot mutate the environment.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import m3gnet_threebody_patch as patch  # noqa: E402


@pytest.mark.fast
def test_segment_indices_handles_empty_segments():
    import torch

    assert patch.segment_indices_from_n(torch.tensor([2, 0, 3])).tolist() == [0, 0, 2, 2, 2]
    assert patch.segment_indices_from_n(torch.tensor([2, 3, 0])).tolist() == [0, 0, 1, 1, 1]
    assert patch.segment_indices_from_n(torch.tensor([0, 0, 0])).tolist() == []


@pytest.mark.fast
def test_to_full_bond_space_remaps_into_parent_ids():
    """The remap must send pruned ids through kept_edge_ids and rebuild the counts."""
    import torch

    # 5 parent bonds; bonds 1 and 3 survived the three-body cutoff.
    kept = torch.tensor([1, 3])
    lg = {
        "kept_edge_ids": kept,
        # one triple, from pruned bond 0 to pruned bond 1 -- i.e. parent 1 -> 3
        "line_edge_index": torch.tensor([[0], [1]]),
        "n_triple_ij": torch.tensor([1, 0]),
    }
    bond_dist = torch.zeros(5)
    out = patch.to_full_bond_space(lg, bond_dist, torch.zeros(5, 3), None)

    assert out["line_edge_index"].tolist() == [[1], [3]]
    assert out["n_triple_ij"].tolist() == [0, 1, 0, 0, 0]
    assert int(out["n_triple_ij"].sum()) == out["line_edge_index"].shape[1]


@pytest.fixture
def matgl_copy(tmp_path):
    """A private copy of matgl, importable ahead of the installed one."""
    import matgl

    src = Path(matgl.__file__).resolve().parent
    dst = tmp_path / "matgl"
    shutil.copytree(src, dst)
    return tmp_path


def _run(args, path_root):
    env = {
        **{k: v for k, v in __import__("os").environ.items()},
        "PYTHONPATH": str(path_root),
    }
    return subprocess.run(
        [sys.executable, str(ROOT / "m3gnet_threebody_patch.py"), *args],
        capture_output=True, text=True, env=env,
    )


def test_apply_then_revert_restores_byte_identical_sources(matgl_copy):
    import matgl

    pristine = Path(matgl.__file__).resolve().parent
    targets = ["utils/maths.py", "graph/_compute.py"]

    assert _run(["--check"], matgl_copy).returncode == 1, "copy should start unpatched"

    applied = _run(["--apply"], matgl_copy)
    assert applied.returncode == 0, applied.stderr
    assert _run(["--check"], matgl_copy).returncode == 0

    for rel in targets:
        text = (matgl_copy / "matgl" / rel).read_text()
        assert patch.MARKER in text
        assert (matgl_copy / "matgl" / rel).with_suffix(".py.orig").exists()

    assert _run(["--revert"], matgl_copy).returncode == 0
    for rel in targets:
        assert (matgl_copy / "matgl" / rel).read_bytes() == (pristine / rel).read_bytes()
        assert not (matgl_copy / "matgl" / rel).with_suffix(".py.orig").exists()


def test_apply_is_idempotent(matgl_copy):
    assert _run(["--apply"], matgl_copy).returncode == 0
    second = _run(["--apply"], matgl_copy)
    assert second.returncode == 0
    assert "already patched" in second.stdout
    # A second apply must not append the block twice.
    text = (matgl_copy / "matgl" / "utils" / "maths.py").read_text()
    assert text.count(f"--- {patch.MARKER} ---") == 1


def test_runtime_patch_restores_relabelling_invariance():
    """The end-to-end claim, on the published weights, in a fresh process."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import m3gnet_threebody_patch as p\n"
        "p.install()\n"
        "raise SystemExit(p.self_test())\n" % str(ROOT)
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if "load_model" in out.stderr and "HTTP" in out.stderr:
        pytest.skip("published model not reachable")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "invariant" in out.stdout


def test_defect_is_present_without_the_patch():
    """The control: the same test must FAIL on unpatched matgl.

    Without this, a patch that did nothing would still pass the test above on a
    model whose cutoffs happen to be equal.
    """
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import m3gnet_threebody_patch as p\n"
        "raise SystemExit(p.self_test())\n" % str(ROOT)
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if "load_model" in out.stderr and "HTTP" in out.stderr:
        pytest.skip("published model not reachable")
    assert out.returncode == 1, "unpatched matgl should not be relabelling-invariant"
    assert "NOT invariant" in out.stdout
