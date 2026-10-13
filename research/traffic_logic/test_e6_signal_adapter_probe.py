import unittest
import contextlib
import io
import json
from pathlib import Path
import pickle
import subprocess
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from nn_modules.ldm import LDM
from research.traffic_logic.e6_signal_adapter_probe import main, run_learning_probe, sha256, validate_reference
from research.traffic_logic.test_signal_lane_adapter import inputs
from utils.data_container import ScenarioDreamerData
from utils.pyg_helpers import get_edge_index_bipartite, get_edge_index_complete_graph


def small_model_fixture(lanes=3, agents=2):
    torch.set_num_threads(1)
    torch.manual_seed(13)
    cfg = OmegaConf.create({
        'model': {'hidden_dim': 32, 'agent_hidden_dim': 16, 'num_heads': 4,
                  'agent_num_heads': 4, 'num_factorized_dit_blocks': 1,
                  'num_l2l_blocks': 1, 'lane_latent_dim': 6, 'agent_latent_dim': 4,
                  'dropout': 0, 'label_dropout': 0.1, 'n_diffusion_timesteps': 10,
                  'lane_sampling_temperature': 0.75, 'diffusion_clip': 5},
        'dataset': {'num_map_ids': 2, 'max_num_lanes': max(5, lanes), 'max_num_agents': max(4, agents)},
        'train': {'loss_type': 'l2', 'ema_decay': 0.99}})
    ldm = LDM(cfg).eval()
    # Create nonzero attention/noise paths, as in a learned checkpoint.
    for module in ldm.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.1)
            if module.bias is not None:
                torch.nn.init.normal_(module.bias, std=0.03)
    data = ScenarioDreamerData()
    data['num_lanes'], data['num_agents'], data['map_id'], data['lg_type'] = lanes, agents, 0, 0
    data['lane'].x = torch.randn(lanes, 6)
    data['lane'].latents = data['lane'].x.clone()
    data['agent'].x = torch.randn(agents, 4)
    data['agent'].latents = data['agent'].x.clone()
    data['lane', 'to', 'lane'].edge_index = get_edge_index_complete_graph(lanes)
    data['agent', 'to', 'agent'].edge_index = get_edge_index_complete_graph(agents)
    data['lane', 'to', 'agent'].edge_index = get_edge_index_bipartite(lanes, agents)
    batch = Batch.from_data_list([data])
    xa, xl = torch.randn(agents, 1, 4), batch['lane'].latents.unsqueeze(1)
    ta, tl = torch.full((agents,), 5, dtype=torch.long), torch.zeros(lanes, dtype=torch.long)
    with torch.no_grad():
        ca, cl = ldm.model(xa, xl, batch, ta, tl, unconditional=False)
        ua, ul = ldm.model(xa, xl, batch, ta, tl, unconditional=True)
    reference = {'data': batch, 'x_agent': xa, 'x_lane': xl, 't_agent': ta, 't_lane': tl,
                 'conditional_agent': ca, 'conditional_lane': cl,
                 'unconditional_agent': ua, 'unconditional_lane': ul}
    return ldm, reference, inputs()


class RealDiTProbeTest(unittest.TestCase):
    def test_audited_fp32_replay_noise_does_not_block_learning(self):
        ldm, reference, condition = small_model_fixture()
        for key in ('conditional_lane', 'unconditional_lane'):
            value = reference[key].clone()
            index = value.abs().flatten().argmin()
            value.flatten()[index] += 4.1e-6
            reference[key] = value
        report, _ = run_learning_probe(ldm, reference, condition, steps=2, learning_rate=0.01)
        self.assertTrue(report['baseline_replayed'])
        self.assertEqual(report['steps_completed'], 2)

    def test_noise_sized_positive_changes_are_not_verified_as_condition_response(self):
        ldm, reference, condition = small_model_fixture()
        report, _ = run_learning_probe(ldm, reference, condition, steps=1, learning_rate=1e-7)
        self.assertGreater(report['updated_agent_max_abs'], 0)
        self.assertFalse(report['condition_response_verified'])

    def test_materially_wrong_historical_prediction_is_still_rejected(self):
        ldm, reference, condition = small_model_fixture()
        reference['conditional_lane'] = reference['conditional_lane'] + 1e-3
        progress = {'model_condition_injected': False, 'weights_updated': False}
        with self.assertRaises(AssertionError):
            run_learning_probe(ldm, reference, condition, progress=progress)
        self.assertFalse(progress['model_condition_injected'])
        self.assertFalse(progress['weights_updated'])

    def test_reference_lane_order_must_match_condition_evidence(self):
        ldm, reference, _ = small_model_fixture()
        arrays = {'lane_mu': reference['data']['lane'].x.numpy().copy(),
                  'agent_mu': reference['data']['agent'].x.numpy().copy()}
        stats = {'agent_latents_mean': 0., 'agent_latents_std': 1.,
                 'lane_latents_mean': 0., 'lane_latents_std': 1.}
        reference['data']['lane'].x = reference['data']['lane'].x.flip(0)
        with self.assertRaises(AssertionError):
            validate_reference(reference, arrays, stats, ldm.cfg)

    def test_fixed_road_input_must_match_lane_latents(self):
        ldm, reference, _ = small_model_fixture()
        arrays = {'lane_mu': reference['data']['lane'].x.numpy().copy(),
                  'agent_mu': reference['data']['agent'].x.numpy().copy()}
        stats = {'agent_latents_mean': 0., 'agent_latents_std': 1.,
                 'lane_latents_mean': 0., 'lane_latents_std': 1.}
        reference['x_lane'] = reference['x_lane'] + 0.1
        with self.assertRaises(AssertionError):
            validate_reference(reference, arrays, stats, ldm.cfg)

    def test_actual_dit_consumes_condition_with_frozen_base_and_learning_response(self):
        torch.set_num_threads(1)
        ldm, reference, condition = small_model_fixture()
        report, artifacts = run_learning_probe(ldm, reference, condition, steps=5, learning_rate=0.01)
        self.assertTrue(report['model_condition_injected'])
        self.assertTrue(report['zero_residual_baseline_verified'])
        self.assertTrue(report['pretrained_parameters_unchanged'])
        self.assertGreater(report['projection_grad_norms'][0], 0)
        self.assertGreater(report['state_perturbation_agent_max_abs'], 0)
        self.assertGreater(report['updated_agent_max_abs'], 0)
        self.assertEqual(len(report['step_losses']), 5)
        self.assertIn('adapter_state_dict', artifacts)

    def test_cli_round_trip_with_real_dit_and_synthetic_checkpoint(self):
        from torch_ema import ExponentialMovingAverage
        ldm, reference, _ = small_model_fixture(24, 18)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original, baseline = root / 'original', root / 'baseline'
            original.mkdir(); baseline.mkdir()
            features = np.zeros((12, 11), dtype=np.float32)
            features[:, 4] = 1
            masks = np.zeros((12, 24, 20), dtype=bool)
            for index, lane in enumerate([8, 7, 6, 5, 4, 2]):
                masks[index, lane, 12 if index == 0 else 11:] = True
            arrays = {'signal_features': features, 'signal_point_mask': masks,
                      'signal_active': np.arange(12) < 6, 'signal_batch_index': np.zeros(12, dtype=np.int64),
                      'road_points': np.random.default_rng(3).normal(size=(24,20,2)).astype(np.float32),
                      'lane_mu': reference['data']['lane'].x.numpy(),
                      'agent_mu': reference['data']['agent'].x.numpy()}
            for target in (original, baseline):
                np.savez_compressed(target / 'regular_latents_and_conditions.npz', **arrays)
            checkpoint = root / 'synthetic.ckpt'
            ema = ExponentialMovingAverage(ldm.parameters(), decay=0.99)
            torch.save({'hyper_parameters': {'cfg': ldm.cfg},
                        'state_dict': {'diff_model.' + key: value for key, value in ldm.state_dict().items()},
                        'ema_state_dict': ema.state_dict()}, checkpoint)
            stats_path = root / 'stats.pkl'
            stats = {'agent_latents_mean': 0., 'agent_latents_std': 1.,
                     'lane_latents_mean': 0., 'lane_latents_std': 1.}
            stats_path.write_bytes(pickle.dumps(stats))
            code = Path(__file__).resolve().parents[2]
            source = subprocess.check_output(['git','hash-object','nn_modules/dit.py'], cwd=code, text=True).strip()
            metadata = {'status': 'pretrained_regular_forward_verified', 'scenario_id': '637f20cafde22ff8',
                        'time_index': 10, 'official_num_map_ids': 2, 'previous_output_dir': str(original),
                        'ae_latent_source_sha256': sha256(original / 'regular_latents_and_conditions.npz'),
                        'source_git_blobs': {'nn_modules/dit.py': source},
                        'latent_stats': {'path': str(stats_path), 'sha256': sha256(stats_path), 'values': stats},
                        'ldm_checkpoint': {'path': str(checkpoint), 'sha256': sha256(checkpoint)}}
            (baseline / 'model_base_report.json').write_text(json.dumps(metadata))
            torch.save(reference, baseline / 'ldm_forward_cpu.pt')
            argv = ['probe', '--baseline-dir', str(baseline), '--device', 'cpu', '--steps', '2', '--learning-rate', '0.01']
            captured = io.StringIO()
            with patch('sys.argv', argv), contextlib.redirect_stdout(captured):
                status = main()
            self.assertEqual(status, 0, captured.getvalue())
            output = next(root.glob('signal_adapter_*'))
            report = json.loads((output / 'signal_adapter_report.json').read_text())
            self.assertEqual(report['status'], 'signal_adapter_probe_verified')
            self.assertTrue(report['model_condition_injected'])
            self.assertTrue((output / 'adapter_probe.pt').is_file())


if __name__ == '__main__':
    unittest.main()
