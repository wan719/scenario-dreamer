import unittest

from research.traffic_logic.lane_lineage import (
    build_lane_lineage,
    diagnose_controlled_lanes,
)


class LaneLineageTests(unittest.TestCase):
    def setUp(self):
        self.groups = {7: ["101", 102, "103"]}
        self.lanes = {
            101: [(0, 0), (1, 0), (2, 0)],
            "102": [(2, 0), (3, 0), (4, 0)],
            103: [(4, 0), (5, 0), (6, 0)],
        }

    def test_middle_lane_red_maps_to_group_without_marking_whole_group(self):
        lineage = build_lane_lineage(self.groups, self.lanes)
        report = diagnose_controlled_lanes(lineage, ["102"])
        self.assertEqual(
            report,
            [{
                "raw_lane_id": 102,
                "group_id": 7,
                "source_point_range": (1, 3),
                "compact_point_range": (3, 5),
                "compact_span_range": (2, 5),
            }],
        )
        self.assertEqual(lineage[101]["group_id"], 7)
        self.assertEqual(lineage[103]["group_id"], 7)

    def test_integer_and_string_signal_ids_give_same_mapping(self):
        lineage = build_lane_lineage(self.groups, self.lanes)
        self.assertEqual(
            diagnose_controlled_lanes(lineage, [102]),
            diagnose_controlled_lanes(lineage, ["102"]),
        )

    def test_missing_signal_lane_is_explicitly_unmatched(self):
        lineage = build_lane_lineage(self.groups, self.lanes)
        self.assertEqual(
            diagnose_controlled_lanes(lineage, [999]),
            [{"raw_lane_id": 999, "group_id": None}],
        )

    def test_empty_signal_list_has_no_controlled_lanes(self):
        lineage = build_lane_lineage(self.groups, self.lanes)
        self.assertEqual(diagnose_controlled_lanes(lineage, []), [])

    def test_multiple_controlled_lanes_keep_separate_intervals(self):
        lineage = build_lane_lineage(self.groups, self.lanes)
        report = diagnose_controlled_lanes(lineage, [101, "103"])
        self.assertEqual(
            report,
            [
                {
                    "raw_lane_id": 101,
                    "group_id": 7,
                    "source_point_range": (0, 3),
                    "compact_point_range": (0, 3),
                    "compact_span_range": (0, 3),
                },
                {
                    "raw_lane_id": 103,
                    "group_id": 7,
                    "source_point_range": (1, 3),
                    "compact_point_range": (5, 7),
                    "compact_span_range": (4, 7),
                },
            ],
        )

    def test_group_member_missing_from_source_lanes_fails(self):
        with self.assertRaises(KeyError):
            build_lane_lineage({7: [101, 404]}, self.lanes)


if __name__ == "__main__":
    unittest.main()
