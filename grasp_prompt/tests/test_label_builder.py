from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grasp_prompt.data.label_builder import build_prompt_regions, make_valid_region_mask


class LabelBuilderTest(unittest.TestCase):
    def test_valid_region_mask_uses_task_mapping(self) -> None:
        labels = np.array([1, 2, 3, 2])
        mask = make_valid_region_mask(labels, 4, {4: [2, 3]})
        np.testing.assert_array_equal(mask, np.array([False, True, True, True]))

    def test_build_prompt_regions_assigns_dominant_part(self) -> None:
        object_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (4, 1))
        valid = np.array([True, True, True, False])
        hand_vertices = np.array([[0.0, 0.0, 0.01], [0.02, 0.0, 0.0]], dtype=np.float32)
        hand_parts = {
            0: np.array([[0.0, 0.0, 0.01]], dtype=np.float32),
            1: np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        }
        regions = build_prompt_regions(
            object_points,
            normals,
            valid,
            hand_vertices,
            hand_parts,
            contact_threshold=0.05,
            cluster_radius=0.05,
            min_cluster_size=2,
        )
        self.assertEqual(regions.centers.shape[0], 1)
        self.assertEqual(int(regions.part_labels[0]), 0)


if __name__ == "__main__":
    unittest.main()

