"""Synthetic tests for independent global-coordinate Stop Anchor checks."""

import copy
import struct
import unittest

from research.traffic_logic.e4d_anchor_geometry import (
    audit_anchor_global,
    parse_stop_points,
)
from research.traffic_logic.real_scenario_helpers import build_graphs
from research.traffic_logic.test_real_scenario_helpers import field


def lane(points, *, entry=(), exit=()):
    return {
        "type": "TYPE_SURFACE_STREET",
        "polyline": [{"x": x, "y": y, "z": 0.0} for x, y in points],
        "entryLanes": [str(value) for value in entry],
        "exitLanes": [str(value) for value in exit],
    }


def example_graphs(*, downstream=None):
    raw = {
        101: lane([(100.0, 200.0), (101.0, 200.0)], exit=(102,)),
        102: lane(downstream or [(101.0, 200.0), (102.0, 200.0)], entry=(101,)),
    }
    return (raw, *build_graphs(raw))


class GlobalAnchorTests(unittest.TestCase):
    def audit(self, graph_tuple, stop=(101.0, 200.0, 0.0)):
        raw, lane_graph, compact, groups = graph_tuple
        return audit_anchor_global(
            raw, lane_graph, compact, groups, 102, stop,
            tolerance_m=1e-9)

    def test_unique_join_uses_raw_endpoints_and_global_compact_vertices(self):
        result = self.audit(example_graphs())
        self.assertEqual(result["group_members"], [101, 102])
        self.assertEqual(result["predecessor_lane_id"], 101)
        self.assertEqual(result["shared_compact_vertex_index"], 1)
        self.assertEqual(result["raw_downstream_first_xy"], [101.0, 200.0])
        self.assertEqual(result["raw_predecessor_last_xy"], [101.0, 200.0])
        self.assertEqual(result["compact_shared_vertex_global_xy"], [101.0, 200.0])
        self.assertEqual(result["distances_m"]["predecessor_to_downstream"], 0.0)
        self.assertTrue(result["selected_vertex_consistent"])
        self.assertTrue(result["topology_verified"])
        self.assertEqual(result["geometric_candidate_vertex_indices"], [1])
        self.assertTrue(result["geometric_candidate_scan_performed"])
        self.assertEqual(result["geometric_candidates_count"], 1)
        self.assertTrue(result["unique_anchor_verified"])

    def test_repeated_same_xy_vertex_keeps_selected_but_not_unique(self):
        result = self.audit(example_graphs(
            downstream=[(101.0, 200.0), (102.0, 200.0), (101.0, 200.0)]))
        self.assertTrue(result["selected_vertex_consistent"])
        self.assertTrue(result["topology_verified"])
        self.assertEqual(result["geometric_candidate_vertex_indices"], [1, 3])
        self.assertEqual(result["geometric_candidates_count"], 2)
        self.assertFalse(result["unique_anchor_verified"])
        self.assertIn("multiple_same_xy_vertices", result["reasons"])

    def test_missing_processed_reverse_edge_breaks_topology_only(self):
        graphs = example_graphs()
        graphs[1]["suc_pairs"][101] = []
        result = self.audit(graphs)
        self.assertTrue(result["selected_vertex_consistent"])
        self.assertFalse(result["topology_verified"])
        self.assertFalse(result["unique_anchor_verified"])
        self.assertIn("processed_successor_mismatch", result["reasons"])

    def test_missing_raw_entry_breaks_topology_only(self):
        graphs = example_graphs()
        graphs[0][102]["entryLanes"] = []
        result = self.audit(graphs)
        self.assertTrue(result["selected_vertex_consistent"])
        self.assertFalse(result["topology_verified"])
        self.assertIn("raw_entry_mismatch", result["reasons"])

    def test_wrong_selected_compact_coordinate_is_not_consistent(self):
        graphs = example_graphs()
        group_id = next(iter(graphs[3]))
        graphs[2]["lanes"][group_id][1] = (101.1, 200.0)
        result = self.audit(graphs)
        self.assertFalse(result["selected_vertex_consistent"])
        self.assertTrue(result["topology_verified"])
        self.assertFalse(result["unique_anchor_verified"])
        self.assertIn("selected_vertex_xy_mismatch", result["reasons"])

    def test_tolerance_is_inclusive_and_outside_stop_is_rejected(self):
        graphs = example_graphs()
        inside = self.audit(graphs, (101.0 + 0.5e-9, 200.0, 0.0))
        outside = self.audit(graphs, (101.0 + 2e-9, 200.0, 0.0))
        self.assertTrue(inside["unique_anchor_verified"])
        self.assertFalse(outside["selected_vertex_consistent"])
        self.assertEqual(outside["geometric_candidate_vertex_indices"], [])

    def test_ego_coordinate_cannot_be_mixed_with_global_group(self):
        result = self.audit(example_graphs(), (1.0, 0.0, 0.0))
        self.assertFalse(result["selected_vertex_consistent"])
        self.assertEqual(result["geometric_candidates_count"], 0)
        self.assertEqual(result["coordinate_frame"], "global_xy")

    def test_single_point_downstream_is_invalid(self):
        graphs = example_graphs()
        graphs[0][102]["polyline"] = graphs[0][102]["polyline"][:1]
        result = self.audit(graphs)
        self.assertFalse(result["selected_vertex_consistent"])
        self.assertFalse(result["unique_anchor_verified"])
        self.assertFalse(result["geometric_candidate_scan_performed"])
        self.assertIsNone(result["geometric_candidates_count"])
        self.assertIn("degenerate_raw_lane", result["reasons"])

    def test_lane_missing_from_group_is_invalid(self):
        graphs = example_graphs()
        graphs[3].clear()
        result = self.audit(graphs)
        self.assertFalse(result["unique_anchor_verified"])
        self.assertIn("lane_not_in_one_group", result["reasons"])

    def test_audit_does_not_mutate_source_graphs(self):
        graphs = example_graphs()
        original_raw = copy.deepcopy(graphs[0])
        original_compact = {k: v.copy() for k, v in graphs[2]["lanes"].items()}
        self.audit(graphs)
        self.assertEqual(graphs[0], original_raw)
        for group_id, points in original_compact.items():
            self.assertTrue((graphs[2]["lanes"][group_id] == points).all())


class OriginalStopParserTests(unittest.TestCase):
    def test_stop_xyz_and_state_come_from_target_timestep(self):
        point = (field(1, 1, struct.pack("<d", 101.0))
                 + field(2, 1, struct.pack("<d", 200.0))
                 + field(3, 1, struct.pack("<d", -1.5)))
        lane_state = (field(1, 0, 102) + field(2, 0, 4)
                      + field(3, 2, point))
        payload = (field(7, 2, field(1, 2, field(1, 0, 102)))
                   + field(7, 2, field(1, 2, lane_state)))
        result = parse_stop_points(payload, 1, (102,))
        self.assertEqual(result[102]["state_code"], 4)
        self.assertEqual(result[102]["stop_point_global_xyz"],
                         (101.0, 200.0, -1.5))

    def test_missing_stop_point_is_rejected(self):
        lane_state = field(1, 0, 102) + field(2, 0, 4)
        payload = field(7, 2, field(1, 2, lane_state))
        with self.assertRaisesRegex(ValueError, "Stop Point"):
            parse_stop_points(payload, 0, (102,))


if __name__ == "__main__":
    unittest.main()
