"""Synthetic provenance checks; no official dependency import is needed."""

import unittest

import numpy as np

from research.traffic_logic.e4_real_scenario_diagnostic import (
    child_source_evidence,
    final_source_evidence,
    sample_source_support,
)


class ResamplingSourceTests(unittest.TestCase):
    def test_contiguous_points_attribute_direct_and_boundary_samples(self):
        points = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
        support = sample_source_support(points, [True] * 4, 7)
        self.assertEqual(len(support), 7)
        self.assertTrue(all(not item[2] and not item[3] for item in support))
        first_child = child_source_evidence(support, (1, 3), (0, 4))
        self.assertGreater(first_child["direct_samples"], 0)
        self.assertGreater(first_child["mixed_boundary_samples"], 0)

    def test_discontinuous_fov_points_are_marked_as_cross_gap(self):
        points = np.array([[0., 0.], [1., 0.], [2., 0.]])
        support = sample_source_support(points, [True, False, True], 5)
        evidence = child_source_evidence(support, (2, 3), (0, 5))
        self.assertEqual(evidence["direct_samples"], 0)
        self.assertEqual(evidence["mixed_boundary_samples"], 5)
        self.assertEqual(evidence["cross_gap_samples"], 5)

    def test_single_point_propagates_with_degenerate_source_flag(self):
        points = np.array([[3., 4.]])
        support = sample_source_support(points, [True], 20)
        self.assertEqual(support, [(0, 0, False, True)] * 20)
        evidence = child_source_evidence(support, (0, 1), (0, 20))
        self.assertEqual(evidence["direct_samples"], 20)
        self.assertEqual(evidence["zero_length_samples"], 20)

    def test_no_fov_points_have_no_source_support(self):
        points = np.array([[0., 0.], [1., 0.]])
        self.assertEqual(sample_source_support(points, [False, False], 20), [])

    def test_final_resampling_composes_direct_and_mixed_brackets(self):
        parent = [(0, 1, False, False), (1, 2, False, False),
                  (2, 3, False, False)]
        final = [(1, 1, False, False), (0, 1, False, False)]
        self.assertEqual(final_source_evidence(parent, final, (1, 3)), {
            "direct_tensor_points": 1,
            "mixed_tensor_points": 1,
            "ambiguous_tensor_points": 0,
        })

    def test_final_resampling_flags_degenerate_parent_and_invalid_offset(self):
        parent = [(4, 4, False, True)]
        final = [(0, 0, False, True)]
        self.assertEqual(final_source_evidence(parent, final, (4, 5)), {
            "direct_tensor_points": 1,
            "mixed_tensor_points": 0,
            "ambiguous_tensor_points": 1,
        })
        with self.assertRaisesRegex(ValueError, "outside the parent"):
            final_source_evidence(parent, final, (4, 5), parent_start=1)


if __name__ == "__main__":
    unittest.main()
