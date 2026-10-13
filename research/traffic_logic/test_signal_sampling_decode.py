import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from omegaconf import OmegaConf

from nn_modules.autoencoder import AutoEncoder
from research.traffic_logic.test_e6_signal_adapter_probe import small_model_fixture
from research.traffic_logic import signal_sampling_decode as decode


def decoder_fixture():
    ldm, reference, condition = small_model_fixture()
    cfg = OmegaConf.create({
        'hidden_dim':32, 'num_encoder_blocks':1, 'num_decoder_blocks':1,
        'lane_attr':2, 'num_heads':4, 'dropout':0., 'dim_f':64,
        'state_dim':7, 'num_agent_types':3, 'lane_conn_attr':6, 'num_lane_types':0,
        'agent_hidden_dim':16, 'agent_num_heads':4, 'agent_dim_f':32,
        'lane_conn_hidden_dim':8, 'lane_latent_dim':6, 'agent_latent_dim':4,
        'num_points_per_lane':4, 'max_num_lanes':5})
    ae = AutoEncoder(cfg).eval()
    scale = OmegaConf.create({'fov':64, 'min_speed':0, 'max_speed':30,
        'min_length':0, 'max_length':10, 'min_width':0, 'max_width':4,
        'min_lane_x':-32, 'max_lane_x':32, 'min_lane_y':-32, 'max_lane_y':32,
        'num_points_per_lane':4})
    stats = {'agent_latents_mean':0.2, 'agent_latents_std':0.8,
             'lane_latents_mean':0.1, 'lane_latents_std':1.2}
    return ae, reference, condition, scale, stats


class DecodeTest(unittest.TestCase):
    def test_real_decoder_matches_original_unnormalization_without_mutating_input(self):
        from utils.data_helpers import unnormalize_scene
        ae, reference, _, scale, stats = decoder_fixture()
        latent = (reference['data']['agent'].latents.clone(), reference['data']['lane'].latents.clone())
        saved = tuple(t.clone() for t in latent)
        with torch.no_grad():
            raw = ae.forward_decoder(latent[0]*0.8+0.2, latent[1]*1.2+0.1, reference['data'])
            expected = unnormalize_scene(raw[0].clone(),raw[1].clone(),**{k:scale[k] for k in decode.SCALE_KEYS})
        actual = decode.decode_scene(ae, latent, reference['data'], stats, scale)
        torch.testing.assert_close(actual['agent_states'], expected[0])
        torch.testing.assert_close(actual['road_points'], expected[1])
        for a,b in zip(latent,saved):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_metrics_use_meters_speed_and_wrapped_heading_with_slot_scope(self):
        baseline=torch.tensor([[0.,0.,2.,-1.,0.,4.,2.],[1.,1.,3.,1.,0.,4.,2.]])
        changed=baseline.clone(); changed[0,:3]=torch.tensor([3.,4.,5.])
        changed[0,3:5]=torch.tensor([-1.,-0.001])
        result, rows=decode.compare_scenes({'agent_states':baseline,'road_points':torch.zeros(3,4,2)},
                                         {'agent_states':changed,'road_points':torch.zeros(3,4,2)})
        self.assertEqual(rows[0]['position_delta_m'],5.)
        self.assertEqual(rows[0]['speed_delta_mps'],3.)
        self.assertLess(abs(rows[0]['heading_delta_deg']),0.1)
        self.assertEqual(result['agent_pairing'],'latent slots, not real-world track identities')

    def test_svg_outputs_parse_and_keep_stop_markers_separate_from_lane_geometry(self):
        ae, ref, condition, scale, stats=decoder_fixture()
        latent=(ref['data']['agent'].latents,ref['data']['lane'].latents)
        scene=decode.decode_scene(ae,latent,ref['data'],stats,scale)
        with tempfile.TemporaryDirectory() as directory:
            paths=decode.write_scene_views(Path(directory),{k:scene for k in ('baseline','stop','go')},condition,scale)
            self.assertEqual(len(paths),4)
            for filename in paths:
                ET.parse(Path(directory)/filename)
            self.assertIn('Stop point', (Path(directory)/'stop.svg').read_text())


if __name__=='__main__':
    unittest.main()
