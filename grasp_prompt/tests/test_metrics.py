from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grasp_prompt.metrics import (
    candidate_binary_labels,
    compensation_penalty_score,
    cover_at_k,
    hit_at_1,
    joint_consistency_score,
    mean_prompt_distance,
    pr_auc,
    roc_auc,
)


class MetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.centers = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=np.float32)
        self.scales = np.array([[0.1, 0.2]], dtype=np.float32)
        self.mask = np.array([[1.0, 1.0]], dtype=np.float32)

    def test_prompt_metrics(self) -> None:
        selected = np.array([[0.05, 0.0, 0.0]], dtype=np.float32)
        candidates = np.array([[[0.5, 0.5, 0.5], [1.05, 1.0, 1.0]]], dtype=np.float32)
        self.assertEqual(hit_at_1(selected, self.centers, self.scales, self.mask), 1.0)
        self.assertEqual(cover_at_k(candidates, self.centers, self.scales, self.mask), 1.0)
        self.assertAlmostEqual(mean_prompt_distance(selected, self.centers, self.mask), 0.05, places=6)

    def test_candidate_labels_and_guidance_metrics(self) -> None:
        candidates = np.array([[[0.02, 0.0, 0.0], [0.5, 0.5, 0.5]]], dtype=np.float32)
        part_labels = np.array([[2, 1]], dtype=np.int64)
        positive, matched = candidate_binary_labels(
            candidates, self.centers, self.scales, part_labels, self.mask
        )
        np.testing.assert_array_equal(positive, np.array([[1.0, 0.0]], dtype=np.float32))
        self.assertEqual(int(matched[0, 0]), 2)

        part_to_joint = np.zeros((3, 4), dtype=np.float32)
        part_to_joint[2, [1, 2]] = 1.0
        target_joint = np.array([[0.0, 1.0, 1.0, 0.0]], dtype=np.float32)
        target_part = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        self.assertEqual(joint_consistency_score(np.array([2]), target_joint, part_to_joint), 1.0)
        self.assertEqual(compensation_penalty_score(np.array([2]), target_part), 0.0)

    def test_auc_helpers(self) -> None:
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertEqual(roc_auc(labels, scores), 1.0)
        self.assertEqual(pr_auc(labels, scores), 1.0)


if __name__ == "__main__":
    unittest.main()

