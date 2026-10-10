"""Synthetic evidence checks for Stop Point projection and provenance."""

import unittest

from research.traffic_logic.e4b_stop_geometry import (
    audit_stop_fov,
    audit_stop_resampled_support,
    project_stop_point,
)


class StopProjectionTests(unittest.TestCase):
    def test_interior_projection_reports_segment_distance_and_arclength(self):
        result = project_stop_point([(0, 0), (10, 0)], (4, 1))
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("stop_off_original_centerline", result["reasons"])
        self.assertEqual(result["nearest_segment_index"], 0)
        self.assertEqual(result["projected_xy"], (4.0, 0.0))
        self.assertAlmostEqual(result["distance_m"], 1.0)
        self.assertAlmostEqual(result["arc_length_m"], 4.0)
        self.assertAlmostEqual(result["position_ratio"], 0.4)
        self.assertAlmostEqual(result["segment_fraction"], 0.4)

    def test_nonunique_nearest_projection_is_uncertain(self):
        lane = [(0, 0), (10, 0), (10, 2), (0, 2)]
        result = project_stop_point(lane, (5, 1))
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["candidate_segment_indices"], (0, 2))
        self.assertIn("multiple_nearest_segments", result["reasons"])

    def test_zero_length_candidate_is_uncertain_and_all_zero_invalid(self):
        result = project_stop_point([(0, 0), (0, 0), (10, 0)], (0, 0))
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("zero_length_nearest_segment", result["reasons"])
        self.assertEqual(
            project_stop_point([(0, 0), (0, 0)], (0, 0))["status"],
            "invalid",
        )

    def test_stop_beyond_original_lane_endpoint_is_uncertain(self):
        result = project_stop_point([(0, 0), (10, 0)], (12, 0))
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("projection_clamped_beyond_segment", result["reasons"])
        self.assertAlmostEqual(result["distance_m"], 2.0)

    def test_distant_projection_keeps_metrics_but_cannot_prove_retention(self):
        projection = project_stop_point([(0, 0), (10, 0)], (4, 100))
        self.assertEqual(projection["status"], "uncertain")
        self.assertAlmostEqual(projection["distance_m"], 100.0)
        self.assertAlmostEqual(projection["arc_length_m"], 4.0)
        fov = audit_stop_fov(projection, (0, 2), (0, 2), (0, 1), True)
        parent = [(0, 0, False, False), (1, 1, False, False)]
        final = [(0, 0, False, False), (1, 1, False, False)]
        result = audit_stop_resampled_support(fov, parent, final, tensor_row=3)
        self.assertEqual(result["status"], "uncertain")
        self.assertNotEqual(result["status"], "attributable")

    def test_geometry_only_projection_can_disable_centerline_coincidence_gate(self):
        result = project_stop_point(
            [(0, 0), (10, 0)], (4, 100), on_centerline_tolerance_m=None)
        self.assertEqual(result["status"], "retained")
        self.assertAlmostEqual(result["distance_m"], 100.0)


class StopFovTests(unittest.TestCase):
    def test_stop_inside_fov_with_both_original_edge_points_kept(self):
        projection = project_stop_point([(0, 0), (10, 0)], (4, 0))
        result = audit_stop_fov(projection, (0, 2), (5, 7), (5, 6), True)
        self.assertEqual(result["status"], "retained")
        self.assertEqual(result["required_source_indices"], (0, 1))
        self.assertEqual(result["required_compact_indices"], (5, 6))
        self.assertAlmostEqual(result["compact_position"], 5.4)

    def test_stop_outside_fov_filtered_even_if_group_points_survive(self):
        projection = project_stop_point([(0, 0), (10, 0)], (4, 0))
        result = audit_stop_fov(projection, (0, 2), (5, 7), (5, 6), False)
        self.assertEqual(result["status"], "filtered")
        self.assertIn("stop_point_outside_fov", result["reasons"])

    def test_omitted_first_source_point_at_lane_boundary_is_uncertain(self):
        projection = project_stop_point([(0, 0), (10, 0), (20, 0)], (0, 0))
        result = audit_stop_fov(projection, (1, 3), (5, 7), (5, 6), True)
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["required_source_indices"], (0,))
        self.assertIn("source_point_omitted_by_compact", result["reasons"])

    def test_crop_break_at_original_edge_is_uncertain(self):
        projection = project_stop_point([(0, 0), (10, 0)], (4, 0))
        result = audit_stop_fov(projection, (0, 2), (5, 7), (5,), True)
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("source_edge_broken_by_fov_crop", result["reasons"])

    def test_ambiguous_projection_not_filtered_from_first_candidate_only(self):
        lane = [(0, 0), (10, 0), (10, 2), (0, 2)]
        projection = project_stop_point(lane, (5, 1))
        result = audit_stop_fov(
            projection, (0, 4), (0, 4), (2, 3), True)
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["candidate_compact_positions"], (0.5, 2.5))
        self.assertIn("multiple_nearest_segments", result["reasons"])

    def test_vertex_without_any_continuous_source_edge_is_uncertain(self):
        projection = project_stop_point([(0, 0), (10, 0)], (0, 0))
        result = audit_stop_fov(projection, (0, 2), (5, 7), (5,), True)
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("no_continuous_original_edge_at_stop", result["reasons"])


class StopTensorSupportTests(unittest.TestCase):
    def setUp(self):
        projection = project_stop_point(
            [(0, 0), (1, 0), (2, 0), (3, 0)], (1.5, 0))
        self.fov = audit_stop_fov(
            projection, (0, 4), (0, 4), (0, 1, 2, 3), True)
        self.exact_parent = [(i, i, False, False) for i in range(4)]
        self.exact_final = [(i, i, False, False) for i in range(4)]

    def test_adjacent_final_samples_bracket_stop_source_position(self):
        result = audit_stop_resampled_support(
            self.fov, self.exact_parent, self.exact_final, tensor_row=7)
        self.assertEqual(result["status"], "attributable")
        self.assertEqual(result["tensor_row"], 7)
        self.assertEqual(result["proven_adjacent_sample_pairs"], ((1, 2),))

    def test_row_alone_does_not_prove_stop_point_source_support(self):
        final = [(0, 0, False, False), (1, 1, False, False)]
        result = audit_stop_resampled_support(
            self.fov, self.exact_parent, final, tensor_row=7)
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["proven_adjacent_sample_pairs"], ())
        self.assertIn("no_proven_adjacent_final_samples", result["reasons"])

    def test_missing_row_is_filtered_even_when_fov_source_survives(self):
        result = audit_stop_resampled_support(
            self.fov, self.exact_parent, self.exact_final, tensor_row=None)
        self.assertEqual(result["status"], "filtered")

    def test_partition_shared_endpoint_is_uncertain(self):
        projection = project_stop_point([(0, 0), (3, 0)], (1.5, 0))
        fov = audit_stop_fov(projection, (0, 2), (0, 2), (0, 1), True)
        parent = [(0, 0, False, False), (0, 1, False, False),
                  (0, 1, False, False), (1, 1, False, False)]
        final = [(i, i, False, False) for i in range(4)]
        result = audit_stop_resampled_support(
            fov, parent, final, tensor_row=3,
            child_range=(0, 4), shared_endpoints=(2,))
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("partition_shared_endpoint", result["reasons"])
        self.assertEqual(result["shared_endpoint_candidates"], (2,))

    def test_gap_in_required_source_bracket_is_uncertain(self):
        parent = [(0, 0, False, False), (1, 1, False, False),
                  (1, 2, True, False), (3, 3, False, False)]
        result = audit_stop_resampled_support(
            self.fov, parent, self.exact_final, tensor_row=2)
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("fov_crop_gap_in_support", result["reasons"])

    def test_partition_child_with_row_but_no_stop_source_is_filtered(self):
        parent = [(i, i, False, False) for i in range(6)]
        child_final = [(i, i, False, False) for i in range(3)]
        result = audit_stop_resampled_support(
            self.fov, parent, child_final, tensor_row=4,
            child_range=(3, 6))
        self.assertEqual(result["status"], "filtered")
        self.assertIn("no_local_stop_support", result["reasons"])
        self.assertEqual(result["proven_adjacent_sample_pairs"], ())

    def test_alternative_projection_inside_child_prevents_false_filtering(self):
        lane = [(0, 0), (10, 0), (10, 2), (0, 2)]
        projection = project_stop_point(lane, (5, 1))
        fov = audit_stop_fov(projection, (0, 4), (0, 4),
                             (0, 1, 2, 3), True)
        parent = [(i, i, False, False) for i in range(4)]
        child_final = [(0, 0, False, False), (1, 1, False, False)]
        result = audit_stop_resampled_support(
            fov, parent, child_final, tensor_row=8,
            child_range=(2, 4))
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["candidate_compact_positions"], (0.5, 2.5))
        self.assertNotIn("no_local_stop_support", result["reasons"])

    def test_all_ambiguous_projection_candidates_outside_child_filter(self):
        lane = [(0, 0), (10, 0), (10, 2), (0, 2)]
        projection = project_stop_point(lane, (5, 1))
        fov = audit_stop_fov(projection, (0, 4), (0, 4),
                             (0, 1, 2, 3), True)
        parent = [(i, i, False, False) for i in range(8)]
        child_final = [(i, i, False, False) for i in range(4)]
        result = audit_stop_resampled_support(
            fov, parent, child_final, tensor_row=9,
            child_range=(4, 8))
        self.assertEqual(result["status"], "filtered")
        self.assertIn("no_local_stop_support", result["reasons"])

    def test_crop_gap_between_final_samples_prevents_proven_bracket(self):
        parent = [(0, 0, False, False), (1, 1, False, False),
                  (1, 2, True, False), (2, 2, False, False),
                  (3, 3, False, False)]
        final = [(1, 1, False, False), (4, 4, False, False)]
        result = audit_stop_resampled_support(
            self.fov, parent, final, tensor_row=5)
        self.assertEqual(result["status"], "uncertain")
        self.assertIn("fov_crop_gap_between_final_samples", result["reasons"])


if __name__ == "__main__":
    unittest.main()
