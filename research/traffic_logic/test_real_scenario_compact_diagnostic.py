import unittest

from research.traffic_logic.real_scenario_compact_diagnostic import classify_lane


class CompactClassificationTests(unittest.TestCase):
    def test_merged_lane_retains_its_group_membership(self):
        result = classify_lane(
            443,
            {443: {"type": "TYPE_SURFACE_STREET"}},
            {"lanes": {443: object()}},
            {"lanes": {11: object()}},
            {11: [541, 443]},
        )
        self.assertEqual(result, ("merged", 11, [541, 443]))

    def test_filtered_bike_lane_names_loss_stage(self):
        result = classify_lane(
            526,
            {526: {"type": "TYPE_BIKE_LANE"}},
            {"lanes": {}},
            {"lanes": {}},
            {},
        )
        self.assertEqual(result, ("dropped:process_lanegraph:TYPE_BIKE_LANE", None, []))


if __name__ == "__main__":
    unittest.main()
