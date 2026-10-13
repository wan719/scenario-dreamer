import unittest

import torch
from omegaconf import open_dict

from research.traffic_logic.test_e6_signal_adapter_probe import small_model_fixture
from research.traffic_logic.signal_lane_adapter import SignalLaneAdapter
from research.traffic_logic import signal_sampling as sampling


class OfficialSamplingTest(unittest.TestCase):
    def fixture(self):
        ldm, reference, condition = small_model_fixture()
        with open_dict(ldm.cfg.train):
            ldm.cfg.train.guidance_scale = 1.5
        return ldm, reference, condition

    def test_disabled_and_zero_adapter_match_original_official_loop(self):
        ldm, reference, condition = self.fixture()
        adapter = SignalLaneAdapter(32)
        state = torch.random.get_rng_state().clone()
        with torch.random.fork_rng():
            torch.manual_seed(23)
            official = ldm(reference['data'].clone(), mode='lane_conditioned')
        outputs, report = sampling.sample_paired(ldm, reference['data'], adapter, condition, seed=23)
        torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
        for values in outputs.values():
            for actual, expected in zip(values, official):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(report['paired_torch_noise_verified'])
        self.assertTrue(report['pretrained_parameters_unchanged'])

    def test_signal_is_injected_only_into_conditional_calls_at_all_official_steps(self):
        ldm, reference, condition = self.fixture()
        adapter = SignalLaneAdapter(32)
        torch.nn.init.normal_(adapter.output.weight, std=0.01)
        original_model = ldm.model
        outputs, report = sampling.sample_paired(ldm, reference['data'], adapter, condition, seed=31)
        self.assertIs(ldm.model, original_model)
        self.assertEqual(set(outputs), {'baseline', 'baseline_repeat', 'stop', 'go'})
        for label, trace in report['variants'].items():
            self.assertEqual(trace['conditional_calls'], 10)
            self.assertEqual(trace['unconditional_calls'], 10)
            self.assertEqual(trace['injection_calls'], 10 if label in ('stop', 'go') else 0)
            self.assertEqual(trace['lane_timesteps'], list(range(9, -1, -1)))
            self.assertTrue(trace['fixed_road_latents_verified'])
        self.assertFalse(original_model.lane_embedder._forward_hooks)
        self.assertFalse(report['quality_improvement_verified'])

    def test_failure_restores_model_and_removes_signal_hook(self):
        ldm, reference, condition = self.fixture()
        adapter = SignalLaneAdapter(32)
        torch.nn.init.constant_(adapter.output.weight, float('nan'))
        original_model = ldm.model
        with self.assertRaisesRegex(AssertionError, 'Non-finite'):
            sampling.sample_paired(ldm, reference['data'], adapter, condition, seed=17)
        self.assertIs(ldm.model, original_model)
        self.assertFalse(original_model.lane_embedder._forward_hooks)


if __name__ == '__main__':
    unittest.main()
