"""CPU contract tests for read-only E5-D sampled Lane lineage."""

import copy
import unittest

import numpy as np

try:
    from research.traffic_logic.e5d_row_lineage import (
        classify_lane_sample,
        invert_simulated_permutation,
        trace_resampling,
    )
except ModuleNotFoundError as exc:
    if exc.name != "research.traffic_logic.e5d_row_lineage":
        raise
    classify_lane_sample = None
    invert_simulated_permutation = None
    trace_resampling = None


def _source(lane_id, compact_index, source_index, *, reasons=()):
    return {
        "raw_lane_candidates": [] if lane_id is None else [lane_id],
        "compact_source_indices": [] if compact_index is None else [compact_index],
        "source_index": source_index,
        "reasons": list(reasons),
    }


class ResamplingLineageTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(trace_resampling, "E5-D trace_resampling is not implemented")
        self.assertIsNotNone(classify_lane_sample,
                             "E5-D classify_lane_sample is not implemented")

    def test_same_lane_segment_has_unique_local_source(self):
        points = np.array([[0.0, 0.0], [2.0, 0.0]])
        sources = [_source(443, 119, 0), _source(443, 120, 1)]
        before = copy.deepcopy(sources)

        result = trace_resampling(points, sources, 3)

        np.testing.assert_allclose(result["points"],
                                   [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        self.assertEqual(sources, before)
        self.assertEqual(len(result["samples"]), 3)
        midpoint = result["samples"][1]
        self.assertEqual(midpoint["sample_index"], 1)
        self.assertEqual(tuple(midpoint["input_indices"]), (0, 1))
        self.assertEqual(midpoint["raw_lane_candidates"], [443])
        self.assertEqual(midpoint["compact_source_indices"], [119, 120])
        self.assertEqual(midpoint["source_status"], "verified")
        self.assertEqual(midpoint["unique_raw_lane_id"], 443)
        self.assertEqual(classify_lane_sample(midpoint, 443), "retained")

    def test_cross_member_interpolation_is_partial_for_each_member(self):
        points = np.array([[0.0, 0.0], [2.0, 0.0]])
        sources = [_source(541, 118, 0), _source(443, 119, 1)]

        midpoint = trace_resampling(points, sources, 3)["samples"][1]

        self.assertEqual(midpoint["raw_lane_candidates"], [443, 541])
        self.assertEqual(midpoint["source_status"], "partial")
        self.assertIsNone(midpoint["unique_raw_lane_id"])
        self.assertEqual(classify_lane_sample(midpoint, 443), "partial")
        self.assertEqual(classify_lane_sample(midpoint, 541), "partial")

    def test_fov_gap_prevents_unique_source_even_with_one_lane_id(self):
        points = np.array([[0.0, 0.0], [2.0, 0.0]])
        sources = [_source(455, 119, 0), _source(455, 121, 2)]

        midpoint = trace_resampling(points, sources, 3)["samples"][1]

        self.assertEqual(midpoint["raw_lane_candidates"], [455])
        self.assertEqual(midpoint["source_status"], "uncertain")
        self.assertIsNone(midpoint["unique_raw_lane_id"])
        self.assertEqual(classify_lane_sample(midpoint, 455), "uncertain")

    def test_zero_length_segment_keeps_affected_samples_uncertain(self):
        points = np.array([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
        sources = [_source(443, 119, 0), _source(443, 120, 1),
                   _source(443, 121, 2)]

        samples = trace_resampling(points, sources, 3)["samples"]

        self.assertTrue(any(sample["source_status"] == "uncertain" and
                            sample["unique_raw_lane_id"] is None
                            for sample in samples))

    def test_partition_shared_endpoint_is_not_a_unique_local_witness(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        sources = [_source(443, 119, 0), _source(443, 120, 1),
                   _source(443, 121, 2)]

        samples = trace_resampling(
            points, sources, 3, shared_input_indices=(1,))["samples"]

        self.assertTrue(any(1 in sample["input_indices"] and
                            sample["source_status"] == "uncertain" and
                            sample["unique_raw_lane_id"] is None
                            for sample in samples))

    def test_shared_endpoint_cannot_restore_an_absent_lane(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        sources = [_source(541, 110, 0), _source(541, 111, 1),
                   _source(541, 112, 2)]

        sample = trace_resampling(
            points, sources, 3, shared_input_indices=(1,))["samples"][1]

        self.assertEqual(sample["source_status"], "uncertain")
        self.assertEqual(sample["raw_lane_candidates"], [541])
        self.assertEqual(sample["reasons"], ["partition_shared_endpoint"])
        self.assertEqual(classify_lane_sample(sample, 443), "filtered")

    def test_absent_lane_stays_uncertain_when_sources_may_be_missing(self):
        cases = (
            ([_source(541, 110, 0), _source(541, 112, 2)],
             "fov_crop_gap"),
            ([_source(541, 110, 0), _source(None, None, 1)],
             "missing_source_evidence"),
        )
        for sources, reason in cases:
            with self.subTest(reason=reason):
                sample = trace_resampling(
                    np.array([[0.0, 0.0], [2.0, 0.0]]),
                    sources, 3)["samples"][1]
                self.assertIn(reason, sample["reasons"])
                self.assertEqual(classify_lane_sample(sample, 443), "uncertain")

    def test_non_stop_side_is_filtered_when_only_other_lane_is_supported(self):
        points = np.array([[0.0, 0.0], [2.0, 0.0]])
        sources = [_source(541, 117, 0), _source(541, 118, 1)]

        sample = trace_resampling(points, sources, 3)["samples"][1]

        self.assertEqual(sample["source_status"], "verified")
        self.assertEqual(classify_lane_sample(sample, 443), "filtered")
        self.assertEqual(classify_lane_sample(sample, 541), "retained")

    def test_missing_lane_source_is_uncertain_not_verified_or_filtered(self):
        points = np.array([[0.0, 0.0], [2.0, 0.0]])
        sources = [_source(None, None, 0), _source(None, None, 1)]

        sample = trace_resampling(points, sources, 3)["samples"][1]

        self.assertEqual(sample["raw_lane_candidates"], [])
        self.assertEqual(sample["source_status"], "uncertain")
        self.assertIsNone(sample["unique_raw_lane_id"])
        self.assertEqual(classify_lane_sample(sample, 443), "uncertain")

    def test_one_missing_endpoint_cannot_make_other_owner_unique(self):
        points = np.array([[0.0, 0.0], [2.0, 0.0]])
        sources = [_source(443, 119, 0), _source(None, None, 1)]

        midpoint = trace_resampling(points, sources, 3)["samples"][1]

        self.assertEqual(midpoint["source_status"], "uncertain")
        self.assertIsNone(midpoint["unique_raw_lane_id"])
        self.assertEqual(classify_lane_sample(midpoint, 443), "uncertain")
        self.assertEqual(classify_lane_sample(midpoint, 541), "uncertain")

    def test_near_vertex_target_still_includes_other_interpolation_endpoint(self):
        # The middle target is 5e-13 m past vertex 1. np.interp therefore
        # uses the 443-to-541 edge, even though it is numerically very close
        # to vertex 1; a tolerance-based exact-vertex shortcut is unsafe.
        points = np.array([[0.0, 0.0], [1.0, 0.0],
                           [2.0 + 1e-12, 0.0]])
        sources = [_source(443, 118, 0), _source(443, 119, 1),
                   _source(541, 120, 2)]

        result = trace_resampling(points, sources, 3)
        midpoint = result["samples"][1]

        self.assertGreater(result["points"][1, 0], 1.0)
        self.assertEqual(midpoint["raw_lane_candidates"], [443, 541])
        self.assertEqual(midpoint["source_status"], "partial")
        self.assertIsNone(midpoint["unique_raw_lane_id"])

    def test_two_resampling_stages_preserve_local_indices_per_raw_lane(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        sources = [
            {**_source(443, 119, 0),
             "raw_lane_local_point_indices": {"443": [0]}},
            {**_source(443, 120, 1),
             "raw_lane_local_point_indices": {"443": [1]}},
            {**_source(541, 121, 2),
             "raw_lane_local_point_indices": {"541": [0]}},
        ]

        parent = trace_resampling(points, sources, 5)
        final = trace_resampling(parent["points"], parent["samples"], 4)

        lane443_only = final["samples"][1]  # x = 2/3
        mixed_boundary = final["samples"][2]  # x = 4/3
        self.assertEqual(lane443_only["source_status"], "verified")
        self.assertEqual(lane443_only["raw_lane_local_point_indices"],
                         {"443": [0, 1]})
        self.assertEqual(mixed_boundary["source_status"], "partial")
        self.assertEqual(mixed_boundary["raw_lane_candidates"], [443, 541])
        self.assertEqual(mixed_boundary["raw_lane_local_point_indices"],
                         {"443": [1], "541": [0]})
        self.assertIsNone(mixed_boundary["unique_raw_lane_id"])

    def test_second_stage_detects_fov_gap_hidden_between_parent_samples(self):
        # The first stage has no target on the cropped 101-to-104 edge.
        # Its neighboring samples each look locally unique, but the second
        # stage must not bridge the omitted Compact points as one Lane region.
        points = np.array([[0.0, 0.0], [1.0, 0.0],
                           [1.1, 0.0], [1.2, 0.0]])
        sources = [
            {**_source(443, compact_index, index),
             "raw_lane_local_point_indices": {"443": [index]}}
            for index, compact_index in enumerate((100, 101, 104, 105))
        ]
        first = trace_resampling(points, sources, 3)
        self.assertEqual(first["samples"][1]["source_status"], "verified")
        self.assertEqual(first["samples"][2]["source_status"], "verified")

        second = trace_resampling(first["points"], first["samples"], 5)
        bridged = second["samples"][3]
        self.assertEqual(bridged["input_indices"], [1, 2])
        self.assertEqual(bridged["compact_source_indices"], [100, 101, 105])
        self.assertEqual(bridged["source_status"], "uncertain")
        self.assertIsNone(bridged["unique_raw_lane_id"])
        self.assertEqual(classify_lane_sample(bridged, 443), "uncertain")

    def test_local_point_map_cannot_claim_another_lane_or_negative_index(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0]])
        bad_maps = ({"541": [0]}, {"443": [-1]})
        for bad_map in bad_maps:
            with self.subTest(raw_lane_local_point_indices=bad_map):
                sources = [
                    {**_source(443, 119, 0),
                     "raw_lane_local_point_indices": bad_map},
                    {**_source(443, 120, 1),
                     "raw_lane_local_point_indices": {"443": [1]}},
                ]
                with self.assertRaises(ValueError):
                    trace_resampling(points, sources, 3)

    def test_strict_local_region_requires_every_supporting_raw_index(self):
        points = np.array([[0.0, 0.0], [2.0, 0.0]])
        complete = [
            {**_source(443, 119, 0),
             "raw_lane_local_point_indices": {"443": [1]}},
            {**_source(443, 120, 1),
             "raw_lane_local_point_indices": {"443": [2]}},
        ]
        for sources in (
                [_source(443, 119, 0), complete[1]],
                [complete[0], _source(443, 120, 1)]):
            with self.subTest(sources=sources):
                sample = trace_resampling(
                    points, sources, 3,
                    require_local_indices=True)["samples"][1]
                self.assertEqual(sample["source_status"], "uncertain")
                self.assertIsNone(sample["unique_raw_lane_id"])
                self.assertEqual(classify_lane_sample(sample, 443), "uncertain")

        complete_sample = trace_resampling(
            points, complete, 3,
            require_local_indices=True)["samples"][1]
        self.assertEqual(complete_sample["source_status"], "verified")
        self.assertEqual(complete_sample["unique_raw_lane_id"], 443)


class SimulatedPermutationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            invert_simulated_permutation,
            "E5-D invert_simulated_permutation is not implemented")

    def test_branch_specific_new_to_pre_arrays_invert_independently(self):
        regular = invert_simulated_permutation([2, 0, 1], 3)
        partitioned = invert_simulated_permutation([1, 0], 2)

        self.assertEqual(regular, {0: 1, 1: 2, 2: 0})
        self.assertEqual(partitioned, {0: 1, 1: 0})
        self.assertEqual(invert_simulated_permutation([0, 1, 2], 3),
                         {0: 0, 1: 1, 2: 2})

    def test_non_bijection_or_invalid_row_count_is_rejected(self):
        invalid = (
            ([0, 0], 2), ([2, 0], 2), ([-1, 0], 2),
            ([True, 0], 2), ([0], 2), ([0, 1], 0),
            ([0], True),
        )
        for values, count in invalid:
            with self.subTest(values=values, row_count=count):
                with self.assertRaises(ValueError):
                    invert_simulated_permutation(values, count)


if __name__ == "__main__":
    unittest.main()
