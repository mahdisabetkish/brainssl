"""Small helpers used across the package."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch. cuDNN autotuning stays on for speed, so
    runs are reproducible in distribution rather than bit for bit."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - third-party code still draws from the global state
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    torch.backends.cudnn.benchmark = True


def _plain(obj: Any) -> Any:
    """Python types for JSON, with NaN and infinity as null (browsers reject bare NaN)."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def write_json(path: Path | str, data: Any) -> None:
    """Write JSON atomically so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_plain(data), indent=2, allow_nan=False))
    tmp.replace(path)


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())
