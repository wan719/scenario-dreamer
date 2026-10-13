import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from research.traffic_logic import test_e7_signal_sampling_probe as e7_fixture
from research.traffic_logic.e6_signal_adapter_probe import sha256


def saved_locality_fixture(root):
    baseline, adapter = e7_fixture.saved_fixture(root)
    path = baseline / 'regular_latents_and_conditions.npz'
    with np.load(path) as source: arrays = {key: source[key] for key in source.files}
    arrays['lane_new_to_ae'] = np.arange(24)
    for folder in (baseline, root / 'original'): np.savez_compressed(folder / path.name, **arrays)
    metadata = json.loads((baseline / 'model_base_report.json').read_text())
    metadata['lane_new_to_ae'] = arrays['lane_new_to_ae'].tolist()
    metadata['ae_latent_source_sha256'] = sha256(path)
    (baseline / 'model_base_report.json').write_text(json.dumps(metadata))
    e6 = json.loads((adapter / 'signal_adapter_report.json').read_text()); e6['condition_npz_sha256'] = sha256(path)
    (adapter / 'signal_adapter_report.json').write_text(json.dumps(e6))
    signals = []; evidence = []
    for i in range(12):
        row = [8, 7, 6, 5, 4, 2][i] if i < 6 else [0, 1, 3, 9, 10, 11][i-6]
        signals.append(dict(raw_lane_id=443+i, group_id=row, state_code=4, state_name='STOP',
                            stop_point_ego_xy_m=[0., 0.], stop_point_model_xy=[0., 0.],
                            regular_candidate_route=dict(batch_row=row)))
        if i < 6:
            keep=np.flatnonzero(arrays['signal_point_mask'][i, row]).tolist()
            evidence.append(dict(raw_lane_id=443+i, group_id=row, batch_row=row, valid_point_indices=keep,
                                 samples=[dict(raw_lane_candidates=[1000+i if j < keep[0] else 443+i], source_status='verified', reasons=[]) for j in range(20)]))
    signals_path = root / 'signals.json'; mask_path=root / 'mask.json'
    signals_path.write_text(json.dumps(dict(scenario_id=metadata['scenario_id'], time_index=10, signals=signals)))
    mask_path.write_text(json.dumps(dict(signal_input_sha256=sha256(signals_path), regular_source_point_mask_verified=True, evidence=evidence)))
    return baseline, adapter, signals_path, mask_path


class LocalityCLIIntegrationTest(unittest.TestCase):
    def test_incomplete_official_source_fingerprints_are_rejected(self):
        from research.traffic_logic import e8_signal_locality_probe as probe
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);baseline,adapter,signals,mask=saved_locality_fixture(root)
            path=baseline/'model_base_report.json';metadata=json.loads(path.read_text())
            metadata['source_git_blobs'].pop('nn_modules/autoencoder.py');path.write_text(json.dumps(metadata))
            argv=['probe','--baseline-dir',str(baseline),'--adapter-dir',str(adapter), '--signals-json',str(signals), '--mask-report',str(mask), '--device','cpu']
            with patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()): status=probe.main()
            self.assertEqual(status,1)
            report=json.loads((next(root.glob('signal_locality_*'))/'locality_report.json').read_text())
            self.assertFalse(report['full_sampling_run'])
            self.assertIn('source fingerprints',report['error_traceback'])

    def test_cli_uses_saved_models_for_eight_paired_groups_and_writes_fixed_population_results(self):
        from research.traffic_logic import e8_signal_locality_probe as probe
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);baseline,adapter,signals,mask=saved_locality_fixture(root)
            argv=['probe','--baseline-dir',str(baseline),'--adapter-dir',str(adapter), '--signals-json',str(signals), '--mask-report',str(mask), '--device','cpu']
            stream=io.StringIO()
            with patch('sys.argv',argv), contextlib.redirect_stdout(stream): status=probe.main()
            self.assertEqual(status,0,stream.getvalue())
            out=next(root.glob('signal_locality_*')); report=json.loads((out/'locality_report.json').read_text())
            self.assertEqual(report['status'],'signal_locality_pipeline_verified')
            self.assertEqual(len(report['variants']),8)
            self.assertTrue(report['paired_torch_noise_verified'])
            self.assertTrue(report['ae_parameters_unchanged'])
            self.assertFalse(report['quality_improvement_verified'])
            self.assertFalse(report['weights_updated'])
            self.assertEqual(len(report['single_signal_comparisons']),6)
            for trace in report['variants'].values():
                self.assertEqual(trace['conditional_calls'],10)
                self.assertEqual(trace['unconditional_calls'],10)
                self.assertEqual(trace['injection_calls'],10)
            scenes=torch.load(out/'generated_scenes.pt',map_location='cpu')
            self.assertEqual(set(scenes),{'stop','stop_repeat',*[f'go_{443+i}' for i in range(6)]})
            self.assertTrue((out/'signal_group_differences.csv').is_file())
            self.assertTrue((out/'signal_slot_differences.csv').is_file())
            self.assertTrue((out/'comparison_443.svg').is_file())
            self.assertEqual((out/'comparison_443.svg').read_text().count('fill="#16a34a"'),1)

    def test_stale_signal_source_fails_before_model_loading_and_writes_failure_report(self):
        from research.traffic_logic import e8_signal_locality_probe as probe
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);baseline,adapter,signals,mask=saved_locality_fixture(root)
            doc=json.loads(signals.read_text());doc['time_index']=11;signals.write_text(json.dumps(doc))
            argv=['probe','--baseline-dir',str(baseline),'--adapter-dir',str(adapter), '--signals-json',str(signals), '--mask-report',str(mask), '--device','cpu']
            with patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()): status=probe.main()
            self.assertEqual(status,1)
            out=next(root.glob('signal_locality_*'));report=json.loads((out/'locality_report.json').read_text())
            self.assertFalse(report['full_sampling_run'])
            self.assertNotIn('ldm_checkpoint',report)
            self.assertIn('source evidence',report['error_traceback'])


if __name__=='__main__': unittest.main()
