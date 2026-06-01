from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BaselinePrediction:
    selected_prompt: np.ndarray
    candidates: np.ndarray


def random_valid(
    object_points: np.ndarray,
    valid_mask: np.ndarray,
    num_candidates: int = 16,
    rng: np.random.Generator | None = None,
) -> BaselinePrediction:
    rng = np.random.default_rng() if rng is None else rng
    valid_idx = np.flatnonzero(valid_mask.astype(bool))
    if valid_idx.size == 0:
        valid_idx = np.arange(object_points.shape[0])
    choice = rng.choice(valid_idx, size=num_candidates, replace=True)
    candidates = object_points[choice]
    return BaselinePrediction(selected_prompt=candidates[0], candidates=candidates)


def affordance_center(
    object_points: np.ndarray,
    valid_mask: np.ndarray,
    num_candidates: int = 16,
) -> BaselinePrediction:
    valid_idx = np.flatnonzero(valid_mask.astype(bool))
    if valid_idx.size == 0:
        valid_idx = np.arange(object_points.shape[0])
    center = object_points[valid_idx].mean(axis=0)
    candidates = np.repeat(center[None, :], repeats=num_candidates, axis=0)
    return BaselinePrediction(selected_prompt=center, candidates=candidates)

