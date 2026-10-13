import contextlib
import io
import json
import pickle
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch_ema import ExponentialMovingAverage

from nn_modules.autoencoder import AutoEncoder
from research.traffic_logic.e6_signal_adapter_probe import run_learning_probe, sha256
from research.traffic_logic.test_e6_signal_adapter_probe import small_model_fixture
from research.traffic_logic.test_signal_sampling_decode import decoder_fixture
from research.traffic_logic import e7_signal_sampling_probe as probe


def saved_fixture(root):
    ldm, ref, _=small_model_fixture(24,18)
    _,_,_,scale,stats=decoder_fixture()
    for key in stats: stats[key]=0. if key.endswith('mean') else 1.
    with open_dict(ldm.cfg.train): ldm.cfg.train.guidance_scale=1.5
    with open_dict(ldm.cfg.dataset):
        for key in scale: ldm.cfg.dataset[key]=scale[key]
        ldm.cfg.dataset.num_points_per_lane=20
    baseline,original,adapterdir=[root/name for name in ('baseline','original','adapter')]
    for folder in (baseline,original,adapterdir): folder.mkdir()
    f=np.zeros((12,11),dtype=np.float32);f[:,4]=1
    masks=np.zeros((12,24,20),dtype=bool)
    for index,lane in enumerate([8,7,6,5,4,2]): masks[index,lane,12 if index==0 else 11:]=True
    arrays={'signal_features':f,'signal_point_mask':masks,'signal_active':np.arange(12)<6,
            'signal_batch_index':np.zeros(12,dtype=np.int64),
            'road_points':np.random.default_rng(3).normal(size=(24,20,2)).astype(np.float32),
            'lane_mu':ref['data']['lane'].x.numpy(),'agent_mu':ref['data']['agent'].x.numpy()}
    for folder in (baseline,original): np.savez_compressed(folder/'regular_latents_and_conditions.npz',**arrays)
    torch.save(ref,baseline/'ldm_forward_cpu.pt')
    checkpoint=root/'ldm.ckpt'; ema=ExponentialMovingAverage(ldm.parameters(),decay=.99)
    torch.save({'hyper_parameters':{'cfg':ldm.cfg},'state_dict':{'diff_model.'+k:v for k,v in ldm.state_dict().items()},
                'ema_state_dict':ema.state_dict()},checkpoint)
    aecfg=decoder_fixture()[0].cfg
    aecfg.num_points_per_lane=20; aecfg.max_num_lanes=30
    ae=AutoEncoder(aecfg).eval(); aepath=root/'ae.ckpt'
    scale.num_points_per_lane=20
    fullcfg=OmegaConf.create({'model':OmegaConf.to_container(aecfg),'dataset':OmegaConf.to_container(scale)})
    torch.save({'hyper_parameters':{'cfg':fullcfg},'state_dict':{'model.'+k:v for k,v in ae.state_dict().items()}},aepath)
    statpath=root/'stats.pkl';statpath.write_bytes(pickle.dumps(stats))
    code=Path(__file__).resolve().parents[2]
    blobs={p:subprocess.check_output(['git','hash-object',p],cwd=code,text=True).strip()
           for p in ('nn_modules/dit.py','nn_modules/autoencoder.py','nn_modules/ldm.py',
                     'utils/data_helpers.py','utils/data_container.py')}
    metadata={'status':'pretrained_regular_forward_verified','scenario_id':'637f20cafde22ff8','time_index':10,
              'official_num_map_ids':2,'source_git_blobs':blobs,'previous_output_dir':str(original),
              'ae_latent_source_sha256':sha256(original/'regular_latents_and_conditions.npz'),
              'latent_stats':{'path':str(statpath),'sha256':sha256(statpath),'values':stats},
              'ldm_checkpoint':{'path':str(checkpoint),'sha256':sha256(checkpoint)},
              'ae_checkpoint':{'path':str(aepath),'sha256':sha256(aepath)}}
    (baseline/'model_base_report.json').write_text(json.dumps(metadata))
    condition=tuple(torch.from_numpy(arrays[k]) for k in ('signal_features','signal_point_mask','road_points','signal_active'))
    report,artifacts=run_learning_probe(ldm,ref,condition,steps=2,learning_rate=.01)
    report.update({'status':'signal_adapter_probe_verified','baseline_dir':str(baseline),
                   'ldm_checkpoint':metadata['ldm_checkpoint'],
                   'condition_npz_sha256':sha256(baseline/'regular_latents_and_conditions.npz'),
                   'reference_forward_sha256':sha256(baseline/'ldm_forward_cpu.pt')})
    (adapterdir/'signal_adapter_report.json').write_text(json.dumps(report))
    torch.save(artifacts,adapterdir/'adapter_probe.pt')
    return baseline,adapterdir


class SamplingCLITest(unittest.TestCase):
    def test_full_cli_uses_saved_real_models_and_writes_scene_comparisons(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);baseline,adapterdir=saved_fixture(root)
            argv=['probe','--baseline-dir',str(baseline),'--adapter-dir',str(adapterdir),'--device','cpu']
            captured=io.StringIO()
            with patch('sys.argv',argv),contextlib.redirect_stdout(captured): status=probe.main()
            self.assertEqual(status,0,captured.getvalue())
            out=next(root.glob('signal_sampling_*'))
            report=json.loads((out/'sampling_report.json').read_text())
            self.assertEqual(report['status'],'signal_sampling_pipeline_verified')
            self.assertTrue(report['full_sampling_run'])
            self.assertTrue(report['ae_decoded'])
            self.assertTrue(report['paired_torch_noise_verified'])
            self.assertFalse(report['weights_updated'])
            self.assertTrue(report['adapter_parameters_unchanged'])
            self.assertTrue((out/'comparison.svg').is_file())
            self.assertTrue((out/'agent_slot_differences.csv').is_file())
            scenes=torch.load(out/'generated_scenes.pt',map_location='cpu')
            self.assertEqual(scenes['stop']['agent_states'].shape,(18,7))
            self.assertEqual(scenes['go']['road_points'].shape,(24,20,2))

    def test_wrong_adapter_condition_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            baseline,adapterdir=saved_fixture(Path(folder))
            path=adapterdir/'signal_adapter_report.json'
            report=json.loads(path.read_text());report['condition_npz_sha256']='wrong'
            path.write_text(json.dumps(report))
            argv=['probe','--baseline-dir',str(baseline),'--adapter-dir',str(adapterdir),'--device','cpu']
            with patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()): status=probe.main()
            self.assertEqual(status,1)
            report=json.loads((next(Path(folder).glob('signal_sampling_*'))/'sampling_report.json').read_text())
            self.assertFalse(report['full_sampling_run'])
            self.assertIn('condition',report['error_traceback'])


if __name__=='__main__': unittest.main()
