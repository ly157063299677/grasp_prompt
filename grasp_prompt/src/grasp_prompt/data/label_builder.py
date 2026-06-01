from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from grasp_prompt.utils import normalize_vectors_np


@dataclass
class PromptRegions:
    centers: np.ndarray
    normals: np.ndarray
    part_labels: np.ndarray
    scales: np.ndarray


def normalize_point_cloud(points: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    points = np.asarray(points, dtype=np.float32)
    center = points.mean(axis=0, keepdims=True)
    shifted = points - center
    scale = np.linalg.norm(shifted, axis=1).max()
    scale = float(max(scale, eps))
    return shifted / scale, {"center": center.squeeze(0), "scale": np.array(scale, dtype=np.float32)}


def make_valid_region_mask(
    affordance_labels: np.ndarray,
    task_id: int,
    task_to_affordances: Mapping[int, Sequence[int]],
) -> np.ndarray:
    labels = np.asarray(affordance_labels)
    valid_labels = set(int(x) for x in task_to_affordances.get(int(task_id), []))
    if not valid_labels:
        return np.zeros(labels.shape[0], dtype=bool)
    return np.array([int(label) in valid_labels for label in labels], dtype=bool)


def extract_contact_points(
    object_points: np.ndarray,
    valid_mask: np.ndarray,
    hand_vertices: np.ndarray,
    threshold: float,
) -> np.ndarray:
    object_points = np.asarray(object_points, dtype=np.float32)
    valid_mask = np.asarray(valid_mask).astype(bool)
    hand_vertices = np.asarray(hand_vertices, dtype=np.float32)
    if object_points.ndim != 2 or object_points.shape[1] != 3:
        raise ValueError("object_points must have shape [N, 3]")
    if hand_vertices.ndim != 2 or hand_vertices.shape[1] != 3:
        raise ValueError("hand_vertices must have shape [M, 3]")
    if hand_vertices.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size == 0:
        return valid_idx
    candidates = object_points[valid_idx]
    dist = np.linalg.norm(candidates[:, None, :] - hand_vertices[None, :, :], axis=-1)
    keep = dist.min(axis=1) < threshold
    return valid_idx[keep].astype(np.int64)


def radius_connected_components(points: np.ndarray, radius: float, min_cluster_size: int = 3) -> list[np.ndarray]:
    """Simple deterministic radius graph clustering for contact points."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == 0:
        return []
    dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    adjacency = dist <= radius
    visited = np.zeros(points.shape[0], dtype=bool)
    clusters: list[np.ndarray] = []
    for start in range(points.shape[0]):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp = []
        while stack:
            node = stack.pop()
            comp.append(node)
            neigh = np.flatnonzero(adjacency[node])
            for nxt in neigh:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(int(nxt))
        comp_arr = np.asarray(comp, dtype=np.int64)
        if comp_arr.size >= min_cluster_size:
            clusters.append(comp_arr)
    return clusters


def dominant_contact_part(
    cluster_points: np.ndarray,
    hand_part_vertices: Mapping[int, np.ndarray],
) -> int:
    if not hand_part_vertices:
        raise ValueError("hand_part_vertices must not be empty")
    best_part = None
    best_dist = np.inf
    for part, vertices in hand_part_vertices.items():
        vertices = np.asarray(vertices, dtype=np.float32)
        if vertices.size == 0:
            continue
        dist = np.linalg.norm(cluster_points[:, None, :] - vertices[None, :, :], axis=-1)
        mean_min = float(dist.min(axis=1).mean())
        if mean_min < best_dist:
            best_dist = mean_min
            best_part = int(part)
    if best_part is None:
        raise ValueError("all hand_part_vertices entries are empty")
    return best_part


def build_prompt_regions(
    object_points: np.ndarray,
    object_normals: np.ndarray,
    valid_mask: np.ndarray,
    hand_vertices: np.ndarray,
    hand_part_vertices: Mapping[int, np.ndarray],
    contact_threshold: float = 0.025,
    cluster_radius: float = 0.04,
    min_cluster_size: int = 3,
) -> PromptRegions:
    contact_idx = extract_contact_points(object_points, valid_mask, hand_vertices, contact_threshold)
    if contact_idx.size == 0:
        return PromptRegions(
            centers=np.zeros((0, 3), dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
            part_labels=np.zeros((0,), dtype=np.int64),
            scales=np.zeros((0,), dtype=np.float32),
        )

    contact_points = object_points[contact_idx]
    components = radius_connected_components(contact_points, cluster_radius, min_cluster_size)
    centers = []
    normals = []
    part_labels = []
    scales = []
    for comp in components:
        object_idx = contact_idx[comp]
        pts = object_points[object_idx]
        nrm = object_normals[object_idx]
        center = pts.mean(axis=0)
        normal = normalize_vectors_np(nrm.sum(axis=0, keepdims=True)).squeeze(0)
        scale = float(np.sqrt(np.mean(np.sum((pts - center) ** 2, axis=1))))
        centers.append(center)
        normals.append(normal)
        scales.append(max(scale, 1e-4))
        part_labels.append(dominant_contact_part(pts, hand_part_vertices))

    if not centers:
        return PromptRegions(
            centers=np.zeros((0, 3), dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
            part_labels=np.zeros((0,), dtype=np.int64),
            scales=np.zeros((0,), dtype=np.float32),
        )
    return PromptRegions(
        centers=np.asarray(centers, dtype=np.float32),
        normals=np.asarray(normals, dtype=np.float32),
        part_labels=np.asarray(part_labels, dtype=np.int64),
        scales=np.asarray(scales, dtype=np.float32),
    )


def target_from_contact_part(part_label: int, part_to_joint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    part_to_joint = np.asarray(part_to_joint, dtype=np.float32)
    if part_label < 0 or part_label >= part_to_joint.shape[0]:
        raise ValueError("part_label is outside part_to_joint rows")
    target_part = np.zeros((part_to_joint.shape[0],), dtype=np.float32)
    target_part[int(part_label)] = 1.0
    target_joint = (part_to_joint[int(part_label)] > 0).astype(np.float32)
    return target_part, target_joint

