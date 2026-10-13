import copy
import unittest

import numpy as np
import torch

from research.traffic_logic import signal_sampling
from research.traffic_logic.signal_lane_adapter import SignalLaneAdapter
from research.traffic_logic import test_signal_sampling as sampling_fixture


def geometry_fixture():
    roads = np.array([[[x, y] for y in np.linspace(-20, 20, 20)] for x in (0., 4., 10.)])
    features = np.zeros((3, 11), dtype=np.float32)
    features[:, 4] = 1
    features[:, 9] = np.array([0, 4, 10]) / 32
    masks = np.zeros((3, 3, 20), bool)
    for i in range(2): masks[i, i, 10:] = True
    arrays = dict(road_points=roads / 32, signal_features=features, signal_point_mask=masks,
                  signal_active=np.array([True, True, False]), lane_new_to_ae=np.array([2, 0, 1]))
    signals = []
    evidence = []
    for i in range(3):
        signal = dict(raw_lane_id=100+i, group_id=i, state_code=4, state_name='STOP',
                      stop_point_ego_xy_m=[float(roads[i, 0, 0]), 0.],
                      stop_point_model_xy=features[i, 9:].tolist(),
                      regular_candidate_route=dict(batch_row=int(arrays['lane_new_to_ae'][i])))
        signals.append(signal)
        if i < 2:
            evidence.append(dict(raw_lane_id=100+i, group_id=i, batch_row=int(arrays['lane_new_to_ae'][i]),
                                 valid_point_indices=list(range(10, 20)),
                                 samples=[dict(raw_lane_candidates=[200+i if j < 10 else 100+i], source_status='verified', reasons=[]) for j in range(20)]))
    scale = dict(min_lane_x=-32, max_lane_x=32, min_lane_y=-32, max_lane_y=32)
    metadata = dict(scenario_id='scene', time_index=10, lane_new_to_ae=[2, 0, 1])
    doc = dict(scenario_id='scene', time_index=10, signals=signals)
    mask = dict(signal_input_sha256='hash', regular_source_point_mask_verified=True, evidence=evidence)
    return arrays, metadata, doc, mask, scale


def scene(states):
    return {'agent_states': torch.tensor(states, dtype=torch.float32), 'agent_types': torch.zeros(len(states), dtype=torch.long)}


class SingleSignalConditionsTest(unittest.TestCase):
    def test_only_one_active_signal_state_changes_and_all_geometry_is_preserved(self):
        from research.traffic_logic.signal_locality import single_signal_variants
        arrays, _, doc, _, _ = geometry_fixture()
        condition = tuple(torch.from_numpy(arrays[k]) for k in ('signal_features', 'signal_point_mask', 'road_points', 'signal_active'))
        saved = [t.clone() for t in condition]
        variants = single_signal_variants(condition, doc['signals'])
        self.assertEqual([v[0] for v in variants], ['stop', 'stop_repeat', 'go_100', 'go_101'])
        for index, (_, changed) in enumerate(variants[2:]):
            self.assertEqual((changed[0][:, :9] != condition[0][:, :9]).any(1).nonzero().flatten().tolist(), [index])
            self.assertEqual(changed[0][index, :9].argmax().item(), 6)
            self.assertEqual(changed[0][index, :9].sum().item(), 1)
            for a, b in zip(changed[1:], condition[1:]): torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(changed[0][:, 9:], condition[0][:, 9:], rtol=0, atol=0)
        for a, b in zip(condition, saved): torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_active_go_or_non_one_hot_input_is_rejected(self):
        from research.traffic_logic.signal_locality import single_signal_variants
        arrays, _, doc, _, _ = geometry_fixture()
        condition = tuple(torch.from_numpy(arrays[k]) for k in ('signal_features', 'signal_point_mask', 'road_points', 'signal_active'))
        condition[0][0, 4] = 0; condition[0][0, 6] = 1
        with self.assertRaises(ValueError): single_signal_variants(condition, doc['signals'])

    def test_official_sampler_accepts_stop_repeats_and_two_single_signal_variants(self):
        from research.traffic_logic.signal_locality import single_signal_variants
        ldm, reference, condition = sampling_fixture.OfficialSamplingTest().fixture()
        features, masks, roads, active = [v.clone() for v in condition]
        active[:2] = True
        active[2:] = False
        variants = single_signal_variants((features, masks, roads, active), [dict(raw_lane_id=100+i) for i in range(len(features))])
        adapter = SignalLaneAdapter(32)
        torch.nn.init.normal_(adapter.output.weight, std=.01)
        original = ldm.model
        outputs, report = signal_sampling.sample_condition_variants(ldm, reference['data'], adapter, variants, seed=23)
        self.assertEqual(list(outputs), ['stop', 'stop_repeat', 'go_100', 'go_101'])
        self.assertIs(ldm.model, original)
        self.assertTrue(report['paired_torch_noise_verified'])
        self.assertTrue(report['adapter_parameters_unchanged'])
        for trace in report['variants'].values(): self.assertEqual(trace['injection_calls'], 10)
        torch.testing.assert_close(outputs['stop'][0], outputs['stop_repeat'][0], rtol=0, atol=0)

    def test_duplicate_variant_name_is_rejected_before_sampling(self):
        ldm, reference, condition = sampling_fixture.OfficialSamplingTest().fixture()
        with self.assertRaises(ValueError):
            signal_sampling.sample_condition_variants(ldm, reference['data'], SignalLaneAdapter(32), [('stop', condition), ('stop', condition)])


class LocalityGeometryTest(unittest.TestCase):
    def context(self):
        from research.traffic_logic.signal_locality_geometry import build_geometry_context
        arrays, metadata, doc, mask, scale = geometry_fixture()
        return build_geometry_context(arrays, metadata, doc, mask, scale, 'hash')

    def test_upstream_vehicle_is_associated_even_outside_signal_source_mask(self):
        from research.traffic_logic.signal_locality_geometry import classify_approaches
        result = classify_approaches(scene([[.1, -10, 2, 0, 1, 4, 2]]), self.context())[0]
        self.assertEqual(result['signal_lane_id'], 100)
        self.assertEqual(result['source_lane_id'], 200)
        self.assertFalse(result['in_source_mask'])
        self.assertAlmostEqual(result['distance_m'], .1, places=6)
        self.assertAlmostEqual(result['distance_to_stop_m'], 10, places=6)

    def test_opposite_heading_and_downstream_vehicle_are_not_approach_candidates(self):
        from research.traffic_logic.signal_locality_geometry import classify_approaches
        states = [[0, -10, 2, 0, -1, 4, 2], [0, 10, 2, 0, 1, 4, 2], [10, -10, 2, 0, 1, 4, 2]]
        self.assertTrue(all(x['signal_lane_id'] is None for x in classify_approaches(scene(states), self.context())))

    def test_close_parallel_candidates_are_marked_ambiguous(self):
        from research.traffic_logic.signal_locality_geometry import classify_approaches
        result = classify_approaches(scene([[2, -10, 2, 0, 1, 4, 2]]), self.context())[0]
        self.assertIsNone(result['signal_lane_id'])
        self.assertEqual(result['reason'], 'ambiguous_aligned_candidates')

    def test_mixed_or_unverified_source_segment_is_not_assigned_to_signal(self):
        from research.traffic_logic.signal_locality_geometry import classify_approaches
        for change in ('mixed', 'unverified', 'missing_status'):
            with self.subTest(change=change):
                context = self.context()
                sample = context['rows'][0]['samples'][4]
                if change == 'mixed': sample['raw_lane_candidates'] = [999]
                elif change == 'unverified': sample.update(source_status='ambiguous', reasons=['fov_crop_gap'])
                else: sample.pop('source_status')
                result = classify_approaches(scene([[.1, -10, 2, 0, 1, 4, 2]]), context)[0]
                self.assertIsNone(result['signal_lane_id'])
                self.assertEqual(result['reason'], 'unverified_source_segment')

    def test_stale_source_hash_and_wrong_permutation_are_rejected(self):
        from research.traffic_logic.signal_locality_geometry import build_geometry_context
        arrays, metadata, doc, mask, scale = geometry_fixture()
        with self.assertRaises(ValueError): build_geometry_context(arrays, metadata, doc, mask, scale, 'wrong')
        arrays['lane_new_to_ae'] = np.array([0, 1, 2])
        with self.assertRaises(ValueError): build_geometry_context(arrays, metadata, doc, mask, scale, 'hash')

    def test_retained_mask_point_with_other_lane_source_is_rejected(self):
        from research.traffic_logic.signal_locality_geometry import build_geometry_context
        arrays, metadata, doc, mask, scale = geometry_fixture()
        mask['evidence'][0]['samples'][10]['raw_lane_candidates'] = [999]
        with self.assertRaisesRegex(ValueError, 'retained'):
            build_geometry_context(arrays, metadata, doc, mask, scale, 'hash')

    def test_counterfactual_switch_is_retained_in_fixed_target_group_and_empty_groups_are_defined(self):
        from research.traffic_logic.signal_locality_geometry import analyze_locality
        anchor = scene([[0, -10, 2, 0, 1, 4, 2], [4, -10, 3, 0, 1, 4, 2]])
        changed = scene([[4, -10, 1, 0, 1, 4, 2], [4, -10, 3.1, 0, 1, 4, 2]])
        summaries, rows, _ = analyze_locality({'stop': anchor, 'stop_repeat': anchor, 'go_100': changed, 'go_101': anchor}, self.context())
        first = summaries['go_100']
        self.assertEqual(first['groups']['target']['slots'], [0])
        self.assertEqual(first['association_switch_slots'], [0])
        self.assertEqual(first['groups']['target']['mean_signed_speed_delta_mps'], -1)
        self.assertEqual(first['groups']['remaining']['count'], 0)
        self.assertIsNone(first['groups']['remaining']['mean_position_delta_m'])
        self.assertEqual(rows[0]['group'], 'target')

    def test_no_target_and_nonfinite_scene_are_explicit(self):
        from research.traffic_logic.signal_locality_geometry import analyze_locality, classify_approaches
        anchor = scene([[10, -10, 2, 0, 1, 4, 2]])
        summary, _, _ = analyze_locality({'stop': anchor, 'stop_repeat': anchor, 'go_100': anchor, 'go_101': anchor}, self.context())
        self.assertEqual(summary['go_100']['target_status'], 'no_target_approach_candidates')
        anchor['agent_states'][0, 0] = float('nan')
        with self.assertRaises(ValueError): classify_approaches(anchor, self.context())

    def test_remaining_slot_candidate_row_switch_is_reported(self):
        from research.traffic_logic.signal_locality_geometry import analyze_locality
        before = scene([[0, 10, 2, 0, 1, 4, 2]])
        after = scene([[4, 10, 2, 0, 1, 4, 2]])
        summary, rows, _ = analyze_locality({'stop': before, 'stop_repeat': before, 'go_100': after}, self.context())
        self.assertEqual(summary['go_100']['candidate_row_switch_slots'], [0])
        self.assertFalse(rows[0]['same_candidate_row'])


if __name__ == '__main__': unittest.main()
