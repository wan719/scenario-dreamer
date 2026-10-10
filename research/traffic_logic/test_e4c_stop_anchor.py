"""Synthetic checks for an explicit Stop Anchor at a Compact join."""

import copy
import math
import unittest

from research.traffic_logic.e4b_stop_geometry import (
    audit_stop_fov,
    project_stop_point,
)
from research.traffic_logic.e4c_stop_anchor import audit_e4b_stop_anchor
from research.traffic_logic.lane_lineage import build_lane_lineage


def example_case(*, group_members=(101, 102), lane_102=None):
    """Build E4-B-shaped evidence from actual synthetic source polylines."""
    lanes = {
        100: [(0.0, 0.0), (0.5, 0.0)],
        101: [(0.5, 0.0), (1.0, 0.0)],
        102: lane_102 or [(1.0, 0.0), (2.0, 0.0)],
    }
    groups = {7: group_members}
    lineage = build_lane_lineage(groups, lanes)[102]
    stop = lanes[102][0]
    member_index = group_members.index(102)
    predecessor_id = group_members[member_index - 1]
    predecessor_end = lanes[predecessor_id][-1]
    boundary_index = lineage["compact_point_range"][0] - 1
    projection = project_stop_point(lanes[102], stop)
    fov = audit_stop_fov(
        projection,
        lineage["source_point_range"],
        lineage["compact_point_range"],
        (boundary_index, lineage["compact_point_range"][0]),
        True,
    )
    return {
        "raw_lane_id": 102,
        "signal_state": "STOP",
        "stop_point_global_xyz": [*stop, 0.0],
        "stop_point_ego_xy": list(stop),
        "raw_lane_point_count": len(lanes[102]),
        "stop_to_lane_first_xy_distance_m": 0.0,
        "original_lane_projection": projection,
        "group_id": lineage["group_id"],
        "group_members": group_members,
        "member_index": member_index,
        "source_point_range": lineage["source_point_range"],
        "compact_point_range": lineage["compact_point_range"],
        "shared_boundary": {
            "predecessor_lane_id": predecessor_id,
            "compact_vertex_index": boundary_index,
            "predecessor_end_distance_m": math.dist(stop, predecessor_end),
            "stop_to_compact_boundary_ego_distance_m": math.dist(
                stop, predecessor_end),
            "compact_vertex_kept_by_fov": True,
        },
        "stop_point_inside_fov": True,
        "stop_fov_source_audit": fov,
        "regular": {"tensor_row": 4, "stop_source_status": "uncertain"},
        "partitioned": [
            {"child_id": 8, "tensor_row": 5,
             "stop_source_status": "uncertain",
             "partition_shared_endpoint_candidates": []},
        ],
    }


class StopAnchorTests(unittest.TestCase):
    def test_adjacent_same_group_and_xy_join_verify_only_shared_vertex(self):
        anchor = audit_e4b_stop_anchor(example_case())
        self.assertEqual(anchor.raw_lane_id, 102)
        self.assertEqual(anchor.signal_state, "STOP")
        self.assertEqual(anchor.stop_point, (1.0, 0.0, 0.0))
        self.assertEqual(anchor.group_id, 7)
        self.assertEqual(anchor.predecessor_lane_id, 101)
        self.assertEqual(anchor.shared_compact_vertex_index, 1)
        self.assertTrue(anchor.topology_verified)
        self.assertTrue(anchor.geometry_verified)
        self.assertTrue(anchor.shared_vertex_verified)
        self.assertEqual(anchor.verification_status, "verified")
        self.assertIsNone(anchor.final_tensor_stop_provenance_verified)
        self.assertEqual(anchor.final_tensor_stop_provenance_status, "uncertain")
        self.assertEqual(anchor.regular_tensor_row, 4)

    def test_nonadjacent_predecessor_is_not_accepted(self):
        case = example_case(group_members=(100, 101, 102))
        case["shared_boundary"]["predecessor_lane_id"] = 100
        anchor = audit_e4b_stop_anchor(case)
        self.assertFalse(anchor.topology_verified)
        self.assertFalse(anchor.shared_vertex_verified)
        self.assertEqual(anchor.verification_status, "invalid")
        self.assertIn("predecessor_not_adjacent", anchor.reasons)

    def test_xy_join_mismatch_exceeds_explicit_tolerance(self):
        case = example_case(lane_102=[(1.2, 0.0), (2.0, 0.0)])
        anchor = audit_e4b_stop_anchor(case, xy_tolerance_m=1e-6)
        self.assertTrue(anchor.topology_verified)
        self.assertFalse(anchor.geometry_verified)
        self.assertFalse(anchor.shared_vertex_verified)
        self.assertGreater(anchor.connection_xy_upper_bound_m, 0.1)
        self.assertIn("predecessor_downstream_xy_mismatch", anchor.reasons)

    def test_multiple_geometric_connection_candidates_are_uncertain(self):
        anchor = audit_e4b_stop_anchor(
            example_case(), candidate_shared_vertex_indices=(1, 7))
        self.assertTrue(anchor.topology_verified)
        self.assertTrue(anchor.geometry_verified)
        self.assertFalse(anchor.shared_vertex_verified)
        self.assertEqual(anchor.verification_status, "uncertain")
        self.assertIn("multiple_shared_vertex_candidates", anchor.reasons)

    def test_full_compact_points_expose_repeated_join_coordinate(self):
        case = example_case()
        anchor = audit_e4b_stop_anchor(
            case, compact_group_points=[
                (0.5, 0.0), (1.0, 0.0), (2.0, 0.0), (1.0, 0.0)])
        self.assertTrue(anchor.geometric_candidate_scan_performed)
        self.assertEqual(anchor.verification_status, "uncertain")
        self.assertIn("multiple_shared_vertex_candidates", anchor.reasons)

    def test_full_compact_points_with_unique_join_verify_boundary(self):
        anchor = audit_e4b_stop_anchor(
            example_case(), compact_group_points=[
                (0.5, 0.0), (1.0, 0.0), (2.0, 0.0)])
        self.assertTrue(anchor.geometric_candidate_scan_performed)
        self.assertTrue(anchor.shared_vertex_verified)

    def test_full_compact_points_reject_recorded_distance_disagreement(self):
        anchor = audit_e4b_stop_anchor(
            example_case(), compact_group_points=[
                (0.5, 0.0), (1.1, 0.0), (2.0, 0.0)])
        self.assertFalse(anchor.geometry_verified)
        self.assertEqual(anchor.verification_status, "invalid")
        self.assertIn("recorded_compact_distance_mismatch", anchor.reasons)

    def test_wrong_single_candidate_cannot_be_substituted(self):
        anchor = audit_e4b_stop_anchor(
            example_case(), candidate_shared_vertex_indices=(7,))
        self.assertFalse(anchor.shared_vertex_verified)
        self.assertEqual(anchor.verification_status, "invalid")
        self.assertIn("shared_vertex_candidate_mismatch", anchor.reasons)

    def test_nonunique_original_projection_is_uncertain(self):
        case = example_case(lane_102=[(1.0, 0.0), (2.0, 0.0), (1.0, 0.0)])
        self.assertGreater(
            len(case["original_lane_projection"]["candidate_segment_indices"]), 1)
        anchor = audit_e4b_stop_anchor(case)
        self.assertEqual(anchor.verification_status, "uncertain")
        self.assertFalse(anchor.shared_vertex_verified)
        self.assertIn("nonunique_stop_projection", anchor.reasons)

    def test_stop_outside_fov_is_filtered_despite_shared_vertex(self):
        case = example_case()
        case["stop_point_inside_fov"] = False
        case["stop_fov_source_audit"]["status"] = "filtered"
        anchor = audit_e4b_stop_anchor(case)
        self.assertTrue(anchor.shared_vertex_verified)
        self.assertFalse(anchor.shared_vertex_fov_retained)
        self.assertEqual(anchor.verification_status, "filtered")
        self.assertEqual(anchor.final_tensor_stop_provenance_status, "filtered")

    def test_shared_vertex_removed_by_fov_is_filtered(self):
        case = example_case()
        case["shared_boundary"]["compact_vertex_kept_by_fov"] = False
        anchor = audit_e4b_stop_anchor(case)
        self.assertTrue(anchor.shared_vertex_verified)
        self.assertFalse(anchor.shared_vertex_fov_retained)
        self.assertEqual(anchor.verification_status, "filtered")
        self.assertIn("shared_vertex_filtered_by_fov", anchor.reasons)

    def test_partition_shared_endpoint_preserves_final_uncertainty(self):
        case = example_case()
        case["partitioned"][0]["partition_shared_endpoint_candidates"] = [500]
        anchor = audit_e4b_stop_anchor(case)
        self.assertTrue(anchor.shared_vertex_verified)
        self.assertEqual(anchor.verification_status, "verified")
        self.assertEqual(anchor.final_tensor_stop_provenance_status, "uncertain")
        self.assertIn("partition_shared_endpoint", anchor.reasons)

    def test_degenerate_single_point_lane_is_invalid(self):
        case = example_case()
        case["raw_lane_point_count"] = 1
        anchor = audit_e4b_stop_anchor(case)
        self.assertEqual(anchor.verification_status, "invalid")
        self.assertFalse(anchor.shared_vertex_verified)
        self.assertIn("degenerate_raw_lane", anchor.reasons)

    def test_all_zero_length_lane_is_invalid(self):
        case = example_case(lane_102=[(1.0, 0.0), (1.0, 0.0)])
        anchor = audit_e4b_stop_anchor(case)
        self.assertEqual(anchor.verification_status, "invalid")
        self.assertFalse(anchor.geometry_verified)
        self.assertIn("zero_total_raw_lane_arclength", anchor.reasons)

    def test_wrong_compact_contribution_range_is_invalid(self):
        case = example_case()
        case["source_point_range"] = (0, 2)
        anchor = audit_e4b_stop_anchor(case)
        self.assertFalse(anchor.topology_verified)
        self.assertEqual(anchor.verification_status, "invalid")
        self.assertIn("source_range_not_omitted_first_point", anchor.reasons)

    def test_shared_vertex_index_must_precede_contribution(self):
        case = example_case()
        case["shared_boundary"]["compact_vertex_index"] += 1
        anchor = audit_e4b_stop_anchor(case)
        self.assertFalse(anchor.topology_verified)
        self.assertFalse(anchor.shared_vertex_verified)
        self.assertIn("shared_vertex_index_mismatch", anchor.reasons)

    def test_row_and_attributable_e4b_status_do_not_promote_anchor(self):
        case = example_case()
        case["regular"]["stop_source_status"] = "attributable"
        case["partitioned"][0]["stop_source_status"] = "attributable"
        anchor = audit_e4b_stop_anchor(case)
        self.assertEqual(anchor.regular_tensor_row, 4)
        self.assertIsNone(anchor.final_tensor_stop_provenance_verified)
        self.assertEqual(anchor.final_tensor_stop_provenance_status, "uncertain")

    def test_unknown_partition_source_status_is_rejected(self):
        case = example_case()
        case["partitioned"][0]["stop_source_status"] = "unknown_status"
        with self.assertRaisesRegex(ValueError, "stop_source_status"):
            audit_e4b_stop_anchor(case)

    def test_nonfinite_projection_fraction_is_rejected(self):
        case = example_case()
        case["original_lane_projection"]["segment_fraction"] = float("nan")
        with self.assertRaisesRegex(ValueError, "segment_fraction"):
            audit_e4b_stop_anchor(case)

    def test_nonfinite_projection_position_ratio_is_rejected(self):
        case = example_case()
        case["original_lane_projection"]["position_ratio"] = float("nan")
        with self.assertRaisesRegex(ValueError, "position_ratio"):
            audit_e4b_stop_anchor(case)

    def test_duplicate_normalized_group_member_is_invalid(self):
        case = example_case()
        case["group_members"] = [101, "101", 102]
        case["member_index"] = 2
        anchor = audit_e4b_stop_anchor(case)
        self.assertFalse(anchor.topology_verified)
        self.assertEqual(anchor.verification_status, "invalid")
        self.assertIn("duplicate_group_member", anchor.reasons)

    def test_case_is_not_mutated(self):
        case = example_case()
        before = copy.deepcopy(case)
        audit_e4b_stop_anchor(case)
        self.assertEqual(case, before)


if __name__ == "__main__":
    unittest.main()
