# matgl/utils/memory_log.py
from __future__ import annotations

import os
import time
import threading
from typing import Any, Iterable

import numpy as np
import torch

_LOCK = threading.Lock()


def _fmt_bytes(n: int) -> str:
    # Human readable
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.3f} {u}"
        x /= 1024.0
    return f"{n} B"


def _tensor_nbytes(x: torch.Tensor) -> int:
    # True storage bytes (not accounting for allocator caching)
    return int(x.element_size() * x.numel())


def _numpy_nbytes(x: np.ndarray) -> int:
    return int(x.nbytes)


def _obj_nbytes(x: Any) -> int:
    # Conservative for common types
    if x is None:
        return 0
    if isinstance(x, torch.Tensor):
        return _tensor_nbytes(x)
    if isinstance(x, np.ndarray):
        return _numpy_nbytes(x)
    if isinstance(x, (bytes, bytearray, memoryview)):
        return len(x)
    if isinstance(x, str):
        return len(x.encode("utf-8"))
    if isinstance(x, (list, tuple)):
        # shallow: sum sizes of elements (recursive-ish)
        return sum(_obj_nbytes(v) for v in x)
    if isinstance(x, dict):
        return sum(_obj_nbytes(k) + _obj_nbytes(v) for k, v in x.items())
    # Fallback: unknown, return 0 rather than misleading
    return 0


def log_nbytes(
    filepath: str,
    scope: str,
    items: dict[str, Any],
    *,
    include_device: bool = True,
    include_shape: bool = True,
    flush: bool = True,
) -> None:
    """
    Append one log block to `filepath`, listing nbytes for each item.
    Designed for tensors/ndarrays and nested containers.

    Args:
        filepath: log file path
        scope: a label like "_compute_3body" or "M3GNet.forward"
        items: mapping of name -> object
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [f"[{ts}] scope={scope}"]

    total = 0
    for name, obj in items.items():
        nb = _obj_nbytes(obj)
        total += nb

        extra = ""
        if isinstance(obj, torch.Tensor):
            if include_shape:
                extra += f" shape={tuple(obj.shape)}"
            if include_device:
                extra += f" device={obj.device} dtype={obj.dtype}"
        elif isinstance(obj, np.ndarray):
            if include_shape:
                extra += f" shape={obj.shape}"
            extra += f" dtype={obj.dtype}"

        lines.append(f"  - {name}: {nb} ({_fmt_bytes(nb)}){extra}")

    lines.append(f"  = total_listed: {total} ({_fmt_bytes(total)})")
    lines.append("")

    with _LOCK:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if flush:
                f.flush()
