"""E7: saved signal adapter, paired official sampling, AE decoding and scene views."""
import argparse
import csv
import gc
import json
from pathlib import Path
import tempfile
import traceback

import torch
from omegaconf import OmegaConf

from nn_modules.autoencoder import AutoEncoder
from research.traffic_logic.e6_signal_adapter_probe import load_verified_baseline, sha256
from research.traffic_logic.signal_lane_adapter import SignalLaneAdapter
from research.traffic_logic.signal_sampling import sample_paired
from research.traffic_logic.signal_sampling_decode import (
    SCALE_KEYS, decode_scene, compare_scenes, scene_diagnostics, write_scene_views)


def validate_adapter_report(directory, baseline, metadata):
    report = json.loads((directory/'signal_adapter_report.json').read_text())
    required = ('model_condition_injected','adapter_weights_updated','pretrained_parameters_unchanged','condition_response_verified')
    if report['status'] != 'signal_adapter_probe_verified' or not all(report.get(k) is True for k in required):
        raise ValueError('Expected the verified E6 adapter result')
    if Path(report['baseline_dir']).resolve() != baseline:
        raise ValueError('Adapter baseline directory differs')
    for key, filename in [('condition_npz_sha256','regular_latents_and_conditions.npz'),
                          ('reference_forward_sha256','ldm_forward_cpu.pt')]:
        if report[key] != sha256(baseline/filename):
            raise ValueError('Adapter condition/reference evidence differs: '+key)
    if report['ldm_checkpoint']['sha256'] != metadata['ldm_checkpoint']['sha256']:
        raise ValueError('Adapter checkpoint evidence differs')
    return report


def load_adapter(directory, ldm, condition, evidence):
    path = directory/'adapter_probe.pt'
    saved = torch.load(path,map_location='cpu')
    if int(saved['adapter_output_dim']) != int(ldm.cfg.model.hidden_dim) or int(saved['adapter_hidden_dim']) != 64:
        raise ValueError('Adapter architecture differs from the witnessed projection')
    adapter = SignalLaneAdapter(saved['adapter_output_dim'],saved['adapter_hidden_dim'])
    adapter.load_state_dict(saved['adapter_state_dict'],strict=True)
    if sum(p.numel() for p in adapter.parameters()) != int(evidence['adapter_parameter_count']):
        raise ValueError('Adapter parameter count differs')
    if not all(torch.isfinite(p).all() for p in adapter.parameters()):
        raise ValueError('Non-finite saved adapter parameters')
    device = next(ldm.parameters()).device
    adapter.to(device).eval().requires_grad_(False)
    with torch.no_grad():
        actual = adapter(*(t.to(device) for t in condition)).cpu()
    torch.testing.assert_close(actual,saved['lane_residual'].cpu(),rtol=1e-5,atol=1e-5)
    return adapter, {'path':str(path),'sha256':sha256(path),'saved_residual_reproduced':True}


def load_autoencoder(metadata, ldm, arrays, device):
    description = metadata['ae_checkpoint']
    path = Path(description['path'])
    if sha256(path) != description['sha256']:
        raise ValueError('AE checkpoint changed')
    ckpt = torch.load(path,map_location='cpu')
    cfg = ckpt['hyper_parameters']['cfg']
    if not OmegaConf.is_config(cfg): cfg=OmegaConf.create(cfg)
    for key in SCALE_KEYS:
        if cfg.dataset[key] != ldm.cfg.dataset[key]:
            raise ValueError('AE/LDM coordinate scale differs: '+key)
    if int(cfg.model.num_points_per_lane) != arrays['road_points'].shape[1]:
        raise ValueError('AE point count differs from condition geometry')
    for role in ('agent','lane'):
        if int(cfg.model[role+'_latent_dim']) != int(ldm.cfg.model[role+'_latent_dim']):
            raise ValueError('AE/LDM latent dimensions differ')
    ae = AutoEncoder(cfg.model)
    weights={key[len('model.'):]:value for key,value in ckpt['state_dict'].items() if key.startswith('model.')}
    ae.load_state_dict(weights,strict=True)
    ae.to(device).eval().requires_grad_(False)
    del ckpt, weights
    gc.collect()
    return ae,cfg.dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir',type=Path,required=True)
    parser.add_argument('--adapter-dir',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261013)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=parser.parse_args()
    baseline,adapterdir=args.baseline_dir.resolve(),args.adapter_dir.resolve()
    out=Path(tempfile.mkdtemp(prefix='signal_sampling_',dir=baseline.parent))
    report={'status':'signal_sampling_pipeline_failed','output_dir':str(out),
            'baseline_dir':str(baseline),'adapter_dir':str(adapterdir),
            'model_condition_injected':False,'weights_updated':False,
            'quality_improvement_verified':False,'full_sampling_run':False,'ae_decoded':False}
    try:
        metadata=json.loads((baseline/'model_base_report.json').read_text())
        adapter_evidence=validate_adapter_report(adapterdir,baseline,metadata)
        print('Loading witnessed EMA LDM and input evidence...',flush=True)
        ldm,reference,condition,metadata,arrays,stats=load_verified_baseline(baseline,args.device,report)
        adapter,adapter_description=load_adapter(adapterdir,ldm,condition,adapter_evidence)
        report['adapter_checkpoint']=adapter_description
        # These two source files provide the official loop and physical decoder.
        if not all(path in metadata['source_git_blobs'] for path in (
                'nn_modules/ldm.py','nn_modules/autoencoder.py','utils/data_helpers.py','utils/data_container.py')):
            raise ValueError('Missing official sampler/AE source fingerprints')
        print('Loading original AE checkpoint for decoding...',flush=True)
        ae,scale=load_autoencoder(metadata,ldm,arrays,args.device)
        report.update({'ae_checkpoint':metadata['ae_checkpoint'], 'ae_weight_view':'checkpoint model state_dict, matching witnessed AE',
                       'coordinate_scale':{key:float(scale[key]) for key in SCALE_KEYS},
                       'verification_scope':'one scene, one paired seed, full official diffusion and AE decoding',
                       'counterfactual_scope':'synthetic GO perturbation; no traffic-validity claim',
                       'adapter_training_scope':adapter_evidence['objective']})
        ae_versions=tuple(p._version for p in ae.parameters())
        if args.device=='cuda': torch.cuda.reset_peak_memory_stats()
        latent_pairs,_=sample_paired(ldm,reference['data'],adapter,condition,args.seed,progress=report,
                                    on_variant=lambda name:print('Sampling '+name+' through '+str(ldm.n_timesteps)+' official diffusion steps...',flush=True))
        # Preserve full per-call evidence separately so the console report stays readable.
        rng_traces={}
        for label,trace in report['variants'].items():
            digests=trace.pop('rng_call_digests')
            timesteps=trace.pop('lane_timesteps')
            rng_traces[label]={'rng_call_digests':digests,'lane_timesteps':timesteps}
            trace['rng_call_count']=len(digests)
            trace['lane_timestep_first_last']=[timesteps[0],timesteps[-1]]
        (out/'sampling_rng_trace.json').write_text(json.dumps(rng_traces,indent=2))
        report['sampling_rng_trace_sha256']=sha256(out/'sampling_rng_trace.json')
        report['guidance_scale']=float(ldm.cfg.train.guidance_scale)
        scenes={label:decode_scene(ae,pair,reference['data'],stats,scale) for label,pair in latent_pairs.items()}
        report['ae_decoded']=True
        if tuple(p._version for p in ae.parameters()) != ae_versions or any(p.grad is not None for p in ae.parameters()):
            raise AssertionError('Frozen AE changed')
        report['ae_parameters_unchanged']=True
        reference_roads=condition[2].cpu().clone()
        reference_roads[...,0]=(reference_roads[...,0]+1)/2*(scale.max_lane_x-scale.min_lane_x)+scale.min_lane_x
        reference_roads[...,1]=(reference_roads[...,1]+1)/2*(scale.max_lane_y-scale.min_lane_y)+scale.min_lane_y
        comparisons={}; csv_rows=[]
        for name,left,right in [('baseline_repeat','baseline','baseline_repeat'),
                                 ('adapter_vs_baseline','baseline','stop'),('go_vs_stop','stop','go')]:
            summary,rows=compare_scenes(scenes[left],scenes[right])
            comparisons[name]=summary
            csv_rows.extend(dict(comparison=name,**row) for row in rows)
        report['physical_scene_comparisons']=comparisons
        report['scene_diagnostics']={name:scene_diagnostics(scene,reference_roads,scale.fov) for name,scene in scenes.items()}
        torch.save(scenes,out/'generated_scenes.pt')
        torch.save(latent_pairs,out/'generated_normalized_latents.pt')
        with (out/'agent_slot_differences.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['comparison','agent_slot','position_delta_m','speed_delta_mps','heading_delta_deg'])
            writer.writeheader(); writer.writerows(csv_rows)
        report['scene_views']=write_scene_views(out,scenes,condition,scale)
        report['output_files']=['generated_scenes.pt','generated_normalized_latents.pt','sampling_rng_trace.json','agent_slot_differences.csv',*report['scene_views']]
        if args.device=='cuda': report['peak_gpu_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
        report['status']='signal_sampling_pipeline_verified'
        report['quality_improvement_verified']=False
    except Exception:
        report['error_traceback']=traceback.format_exc()
        print(report['error_traceback'])
    (out/'sampling_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='signal_sampling_pipeline_verified' else 1


if __name__=='__main__':
    raise SystemExit(main())
