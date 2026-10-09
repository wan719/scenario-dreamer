import unittest

from research.traffic_logic.tensor_rows import map_lane_ids_to_rows


class TensorRowMappingTests(unittest.TestCase):
    def test_retained_lane_uses_selected_row_not_group_id(self):
        self.assertEqual(
            map_lane_ids_to_rows([7], [7], [0]),
            {7: 0},
        )

    def test_lane_removed_by_fov_has_no_row(self):
        self.assertEqual(
            map_lane_ids_to_rows([7, 8], [8], [0]),
            {7: None, 8: 0},
        )

    def test_lane_omitted_by_max_num_lanes_has_no_row(self):
        self.assertEqual(
            map_lane_ids_to_rows([7, 8, 9], [7, 8, 9], [2, 0]),
            {7: 1, 8: None, 9: 0},
        )

    def test_different_distance_order_changes_rows(self):
        candidates = [7, 8, 9]
        graph_order = [9, 7, 8]
        self.assertEqual(
            map_lane_ids_to_rows(candidates, graph_order, [1, 2, 0]),
            {7: 0, 8: 1, 9: 2},
        )
        self.assertEqual(
            map_lane_ids_to_rows(candidates, graph_order, [2, 0, 1]),
            {7: 2, 8: 0, 9: 1},
        )

    def test_training_permutation_changes_final_rows(self):
        self.assertEqual(
            map_lane_ids_to_rows(
                [7, 8, 9], [9, 7, 8], [1, 2, 0],
                lane_permutation=[2, 0, 1],
            ),
            {7: 1, 8: 2, 9: 0},
        )

    def test_invalid_selected_index_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "selected index"):
            map_lane_ids_to_rows([7], [7], [1])

    def test_duplicate_selected_indices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "selected index"):
            map_lane_ids_to_rows([7, 8], [7, 8], [0, 0])

    def test_selected_indices_outside_both_bounds_are_rejected(self):
        for selected in ([-1], [2]):
            with self.subTest(selected=selected):
                with self.assertRaisesRegex(ValueError, "selected index"):
                    map_lane_ids_to_rows([7, 8], [7, 8], selected)

    def test_duplicate_lane_permutation_indices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "lane permutation"):
            map_lane_ids_to_rows(
                [7, 8], [7, 8], [0, 1], lane_permutation=[0, 0]
            )

    def test_lane_permutation_indices_outside_both_bounds_are_rejected(self):
        for permutation in ([-1, 0], [0, 2]):
            with self.subTest(permutation=permutation):
                with self.assertRaisesRegex(ValueError, "lane permutation"):
                    map_lane_ids_to_rows(
                        [7, 8], [7, 8], [0, 1],
                        lane_permutation=permutation,
                    )

    def test_lane_permutation_length_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "lane permutation length"):
            map_lane_ids_to_rows(
                [7, 8], [7, 8], [0, 1], lane_permutation=[0]
            )

    def test_normalized_duplicate_candidate_lane_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate normalized Lane ID in candidates"):
            map_lane_ids_to_rows([7, "7"], [7], [0])

    def test_normalized_duplicate_graph_lane_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate normalized Lane ID in graph"):
            map_lane_ids_to_rows([7], [7, "7"], [0])


if __name__ == "__main__":
    unittest.main()
