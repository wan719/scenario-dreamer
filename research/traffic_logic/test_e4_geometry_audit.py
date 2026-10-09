"""Synthetic checks for point-level FoV and partition provenance."""

import unittest

import numpy as np

from research.traffic_logic.e4_geometry_audit import (
    audit_lane_fov,
    mask_to_runs,
    plan_partition_lineage,
)


class FovAuditTests(unittest.TestCase):
    def test_mask_keeps_order_and_half_open_disjoint_runs(self):
        self.assertEqual(
            mask_to_runs([False, True, True, False, True, False]),
            ((1, 2, 4), ((1, 3), (4, 5))),
        )
        self.assertEqual(mask_to_runs([False, False]), ((), ()))

    def test_group_retained_can_have_filtered_controlled_interval(self):
        points = np.array([[0, 0], [1, 0], [5, 0], [6, 0], [0, 1]])
        result = audit_lane_fov(points, [True, True, False, False, True],
                                (2, 4), source_point_count=3)
        self.assertEqual(result["status"], "filtered")
        self.assertEqual(result["point_status"], "filtered")
        self.assertEqual(result["kept_group_indices"], (0, 1, 4))
        self.assertEqual(result["kept_local_point_offsets"], ())

    def test_partial_local_points_keep_original_offsets_and_runs(self):
        points = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
        result = audit_lane_fov(points, [True, False, True, False, True],
                                (1, 5), source_point_count=5)
        self.assertEqual(result["status"], "partial")
        self.assertTrue(result["uncertain"])
        self.assertIn("discontinuous_fov_points", result["reasons"])
        self.assertEqual(result["kept_local_point_offsets"], (1, 3))
        self.assertEqual(result["kept_local_offset_runs"], ((1, 2), (3, 4)))

    def test_all_local_points_kept_but_group_gap_is_uncertain(self):
        points = np.array([[0, 0], [1, 0], [5, 0], [6, 0], [2, 0]])
        result = audit_lane_fov(points, [True, True, False, False, True],
                                (0, 2), source_point_count=2)
        self.assertEqual(result["point_status"], "retained")
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("discontinuous_fov_points", result["reasons"])

    def test_repeated_kept_points_make_resampling_origin_uncertain(self):
        points = np.array([[0, 0], [0, 0], [1, 0]])
        result = audit_lane_fov(points, [True, True, True],
                                (0, 3), source_point_count=3)
        self.assertEqual(result["point_status"], "retained")
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("repeated_kept_points", result["reasons"])

    def test_single_point_source_is_invalid_even_if_inside_fov(self):
        result = audit_lane_fov(np.array([[0, 0]]), [True], (0, 1),
                                source_point_count=1)
        self.assertEqual(result["status"], "invalid")
        self.assertIn("source_lane_has_fewer_than_two_points", result["reasons"])

    def test_empty_contribution_and_non_boolean_mask_are_rejected(self):
        points = np.array([[0, 0], [1, 0]])
        result = audit_lane_fov(points, [True, True], (1, 1),
                                source_point_count=2)
        self.assertEqual(result["status"], "invalid")
        with self.assertRaisesRegex(ValueError, "boolean"):
            mask_to_runs([1, 0])


class PartitionAuditTests(unittest.TestCase):
    def test_one_crossing_shares_endpoint_between_two_new_ids(self):
        lanes = {7: np.array([[0, -1], [1, -0.5], [2, 0.5], [3, 1]])}
        plan = plan_partition_lineage(lanes)[7]
        self.assertEqual(plan["crossings"], (2,))
        self.assertEqual(plan["children"], ((8, (0, 3)), (9, (2, 4))))
        self.assertEqual(plan["shared_endpoints"], (2,))
        self.assertEqual(plan["unused_ids"], ())

    def test_last_point_crossing_does_not_replace_parent_but_consumes_id(self):
        lanes = {
            7: np.array([[0, -1], [1, -1], [2, 1]]),
            8: np.array([[0, -1], [1, 1], [2, 1]]),
        }
        plan = plan_partition_lineage(lanes)
        self.assertEqual(plan[7]["crossings"], (2,))
        self.assertEqual(plan[7]["children"], ((7, (0, 3)),))
        self.assertEqual(plan[7]["unused_ids"], (9,))
        self.assertEqual(plan[8]["children"], ((10, (0, 2)), (11, (1, 3))))

    def test_multiple_crossings_preserve_all_source_intervals(self):
        lanes = {4: np.array([[0, -1], [1, 1], [2, -1], [3, 1]])}
        plan = plan_partition_lineage(lanes)[4]
        self.assertEqual(plan["crossings"], (1, 2, 3))
        self.assertEqual(plan["children"], (
            (5, (0, 2)), (6, (1, 3)), (7, (2, 4))))
        self.assertEqual(plan["shared_endpoints"], (1, 2))

    def test_no_crossing_and_single_point_keep_original_id(self):
        lanes = {0: np.array([[3, -2]]), 7: np.array([[0, 1], [1, 2]])}
        plan = plan_partition_lineage(lanes)
        self.assertEqual(plan[0]["children"], ((0, (0, 1)),))
        self.assertEqual(plan[0]["crossings"], ())
        self.assertEqual(plan[7]["children"], ((7, (0, 2)),))


if __name__ == "__main__":
    unittest.main()
