from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_device(batch: dict[str, Any], device: str):
    import torch

    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def sinusoidal_embedding(timesteps, dim: int):
    import torch

    half = dim // 2
    device = timesteps.device
    scale = math.log(10000.0) / max(half - 1, 1)
    freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -scale)
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


def mlp(in_dim: int, hidden_dim: int, out_dim: int, layers: int = 2, dropout: float = 0.0):
    import torch.nn as nn

    if layers < 1:
        raise ValueError("layers must be >= 1")
    dims = [in_dim] + [hidden_dim] * (layers - 1) + [out_dim]
    blocks = []
    for i in range(len(dims) - 1):
        blocks.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            blocks.append(nn.GELU())
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
    return nn.Sequential(*blocks)


def normalize_vectors_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def pad_array(array: np.ndarray, length: int, value: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Pad or truncate the first dimension and return `(padded, mask)`."""
    array = np.asarray(array)
    out_shape = (length,) + array.shape[1:]
    out = np.full(out_shape, value, dtype=array.dtype)
    mask = np.zeros((length,), dtype=np.float32)
    n = min(length, array.shape[0])
    if n:
        out[:n] = array[:n]
        mask[:n] = 1.0
    return out, mask

