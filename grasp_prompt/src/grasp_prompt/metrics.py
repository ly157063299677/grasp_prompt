from __future__ import annotations

import numpy as np


def _masked_normalized_dist(
    points: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    dist = np.linalg.norm(points[..., None, :] - centers[:, None, :, :], axis=-1)
    norm = dist / np.maximum(scales[:, None, :], 1e-8)
    return np.where(mask[:, None, :].astype(bool), norm, 1e9)


def hit_at_1(
    selected: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
    mask: np.ndarray,
) -> float:
    norm = _masked_normalized_dist(selected[:, None, :], centers, scales, mask)[:, 0, :]
    has_label = mask.sum(axis=1) > 0
    hit = (norm.min(axis=1) <= 1.0) & has_label
    return float(hit.mean()) if hit.size else 0.0


def cover_at_k(
    candidates: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
    mask: np.ndarray,
) -> float:
    norm = _masked_normalized_dist(candidates, centers, scales, mask)
    has_label = mask.sum(axis=1) > 0
    cover = (norm.min(axis=(1, 2)) <= 1.0) & has_label
    return float(cover.mean()) if cover.size else 0.0


def mean_prompt_distance(
    selected: np.ndarray,
    centers: np.ndarray,
    mask: np.ndarray,
) -> float:
    dist = np.linalg.norm(selected[:, None, :] - centers, axis=-1)
    dist = np.where(mask.astype(bool), dist, 1e9)
    has_label = mask.sum(axis=1) > 0
    vals = dist.min(axis=1)[has_label]
    return float(vals.mean()) if vals.size else 0.0


def nearest_region_parts(
    selected: np.ndarray,
    centers: np.ndarray,
    part_labels: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    dist = np.linalg.norm(selected[:, None, :] - centers, axis=-1)
    dist = np.where(mask.astype(bool), dist, 1e9)
    idx = dist.argmin(axis=1)
    return part_labels[np.arange(part_labels.shape[0]), idx]


def candidate_binary_labels(
    candidates: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
    part_labels: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    norm = _masked_normalized_dist(candidates, centers, scales, mask)
    idx = norm.argmin(axis=-1)
    positive = (norm.min(axis=-1) <= 1.0) & (mask.sum(axis=1, keepdims=True) > 0)
    matched = np.take_along_axis(part_labels, idx, axis=1)
    return positive.astype(np.float32), matched.astype(np.int64)


def contact_part_accuracy(
    part_probs: np.ndarray,
    target_parts: np.ndarray,
    positive: np.ndarray,
) -> float:
    pred = part_probs.argmax(axis=-1)
    pos = positive.astype(bool)
    if not pos.any():
        return float("nan")
    return float((pred[pos] == target_parts[pos]).mean())


def joint_consistency_score(
    matched_part_labels: np.ndarray,
    target_joint_mask: np.ndarray,
    part_to_joint: np.ndarray,
) -> float:
    response = part_to_joint[matched_part_labels]
    numerator = 2.0 * (response * target_joint_mask).sum(axis=-1)
    denominator = np.abs(response).sum(axis=-1) + np.abs(target_joint_mask).sum(axis=-1)
    return float((numerator / np.maximum(denominator, 1e-8)).mean())


def compensation_penalty_score(
    matched_part_labels: np.ndarray,
    target_part: np.ndarray,
) -> float:
    one_hot = np.zeros_like(target_part)
    one_hot[np.arange(target_part.shape[0]), matched_part_labels] = 1.0
    penalty = (one_hot * (1.0 - target_part)).sum(axis=-1)
    return float(penalty.mean())


def roc_auc(binary_labels: np.ndarray, scores: np.ndarray) -> float:
    labels = binary_labels.reshape(-1).astype(bool)
    scores = scores.reshape(-1).astype(float)
    pos = labels.sum()
    neg = labels.size - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    rank_sum_pos = ranks[labels].sum()
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def pr_auc(binary_labels: np.ndarray, scores: np.ndarray) -> float:
    labels = binary_labels.reshape(-1).astype(bool)
    scores = scores.reshape(-1).astype(float)
    pos = labels.sum()
    if pos == 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(sorted_labels.size) + 1)
    return float((precision * sorted_labels).sum() / pos)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks

