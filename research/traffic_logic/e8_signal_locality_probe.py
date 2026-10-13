"""E8: one-signal synthetic GO, paired official sampling and response locality."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import traceback

import numpy as np
import torch

from research.traffic_logic.e6_signal_adapter_probe import load_verified_baseline, sha256
from research.traffic_logic.e7_signal_sampling_probe import validate_adapter_report, load_adapter, load_autoencoder
from research.traffic_logic.signal_locality import single_signal_variants
from research.traffic_logic.signal_locality_geometry import build_geometry_context, analyze_locality
from research.traffic_logic.signal_sampling import sample_condition_variants
from research.traffic_logic.signal_sampling_decode import SCALE_KEYS, decode_scene, scene_diagnostics, _panel


def _write_csv(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _write_views(out, scenes, condition, variants, context, scale):
    paths = []
    by_label = dict(variants)
    for signal in context['rows'].values():
        if not signal['active']: continue
        lane, index = signal['signal_lane_id'], signal['signal_index']
        label = 'go_' + str(lane)
        anchor = _panel(scenes['stop'], 'Adapter_STOP', condition, scale)
        changed = _panel(scenes[label], 'Signal_' + str(lane) + '_synthetic_GO', by_label[label], scale,
                         go_signal_indices=[index])
        filename = 'comparison_' + str(lane) + '.svg'
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="736" height="465" viewBox="0 0 736 465">'
               '<rect width="736" height="465" fill="white"/>' + anchor +
               '<g transform="translate(368,0)">' + changed + '</g>' +
               '<text x="368" y="448" text-anchor="middle" font-size="11">One paired seed; slots are not track IDs; Stop Points are not stop lines.</text></svg>')
        (out / filename).write_text(svg); paths.append(filename)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--adapter-dir', type=Path, required=True)
    parser.add_argument('--signals-json', type=Path, required=True)
    parser.add_argument('--mask-report', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    baseline, adapter_dir = args.baseline_dir.resolve(), args.adapter_dir.resolve()
    out = Path(tempfile.mkdtemp(prefix='signal_locality_', dir=baseline.parent))
    report = dict(status='signal_locality_pipeline_failed', output_dir=str(out), baseline_dir=str(baseline),
                  adapter_dir=str(adapter_dir), model_condition_injected=False, weights_updated=False,
                  quality_improvement_verified=False, full_sampling_run=False, ae_decoded=False)
    try:
        metadata = json.loads((baseline / 'model_base_report.json').read_text())
        required_sources = ('nn_modules/ldm.py', 'nn_modules/dit.py', 'nn_modules/autoencoder.py',
                            'utils/data_helpers.py', 'utils/data_container.py')
        if not all(path in metadata['source_git_blobs'] for path in required_sources):
            raise ValueError('Missing official sampler/decoder source fingerprints')
        with np.load(baseline / 'regular_latents_and_conditions.npz', allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        signals = json.loads(args.signals_json.read_text())
        masks = json.loads(args.mask_report.read_text())
        # Witnessed Waymo square coordinate schema. Confirm against AE/LDM config below.
        preliminary_scale = dict(min_lane_x=-32., max_lane_x=32., min_lane_y=-32., max_lane_y=32.)
        context = build_geometry_context(arrays, metadata, signals, masks, preliminary_scale, sha256(args.signals_json))
        adapter_evidence = validate_adapter_report(adapter_dir, baseline, metadata)
        print('Loading witnessed EMA LDM, adapter and original AE...', flush=True)
        ldm, reference, condition, metadata, loaded_arrays, stats = load_verified_baseline(baseline, args.device, report)
        if set(arrays) != set(loaded_arrays) or any(not np.array_equal(arrays[k], loaded_arrays[k]) for k in arrays):
            raise ValueError('Input arrays changed during model loading')
        adapter, description = load_adapter(adapter_dir, ldm, condition, adapter_evidence)
        ae, scale = load_autoencoder(metadata, ldm, arrays, args.device)
        context = build_geometry_context(arrays, metadata, signals, masks, scale, sha256(args.signals_json))
        for key, expected in preliminary_scale.items():
            if float(scale[key]) != expected: raise ValueError('Unsupported witnessed coordinate schema')
        report.update(adapter_checkpoint=description, ae_checkpoint=metadata['ae_checkpoint'],
                      coordinate_scale={k: float(scale[k]) for k in SCALE_KEYS},
                      verification_scope='one scene/seed; six single-state interventions; descriptive geometry locality',
                      counterfactual_scope='synthetic GO; no traffic-validity claim',
                      adapter_training_scope=adapter_evidence['objective'],
                      geometry_policy=context['policy'],
                      geometry_input_sha256=dict(signals_json=sha256(args.signals_json), mask_report=sha256(args.mask_report)),
                      grouping_scope='fixed STOP candidate population; association switches retained; remaining is not known uncontrolled')
        variants = single_signal_variants(condition, signals['signals'])
        if len(variants) != 8: raise ValueError('Expected the witnessed six active signals and two STOP runs')
        report['interventions'] = {name: {'signal_lane_id': int(name[3:]), 'changed_state_rows': 1}
                                   for name, _ in variants if name.startswith('go_')}
        versions = tuple(p._version for p in ae.parameters())
        if args.device == 'cuda': torch.cuda.reset_peak_memory_stats()
        pairs, _ = sample_condition_variants(ldm, reference['data'], adapter, variants, args.seed, report,
                                            lambda name: print('Sampling ' + name + ' through official diffusion...', flush=True))
        rng = {}
        for name, trace in report['variants'].items():
            digests, times = trace.pop('rng_call_digests'), trace.pop('lane_timesteps')
            rng[name] = dict(rng_call_digests=digests, lane_timesteps=times)
            trace['rng_call_count'] = len(digests); trace['lane_timestep_first_last'] = [times[0], times[-1]]
        (out / 'sampling_rng_trace.json').write_text(json.dumps(rng, indent=2))
        report['sampling_rng_trace_sha256'] = sha256(out / 'sampling_rng_trace.json')
        scenes = {name: decode_scene(ae, pair, reference['data'], stats, scale) for name, pair in pairs.items()}
        if tuple(p._version for p in ae.parameters()) != versions or any(p.grad is not None for p in ae.parameters()):
            raise AssertionError('Frozen AE changed')
        report.update(ae_decoded=True, ae_parameters_unchanged=True, guidance_scale=float(ldm.cfg.train.guidance_scale))
        summaries, records, audit = analyze_locality(scenes, context)
        report['single_signal_comparisons'] = summaries
        report['numeric_repeat_audit'] = {k: v for k, v in audit.items() if k != 'associations'}
        (out / 'approach_associations.json').write_text(json.dumps(audit['associations'], indent=2))
        report['approach_associations_sha256'] = sha256(out / 'approach_associations.json')
        reference_roads = torch.as_tensor(context['roads'])
        report['scene_diagnostics'] = {name: scene_diagnostics(scene, reference_roads, scale.fov) for name, scene in scenes.items()}
        torch.save(scenes, out / 'generated_scenes.pt'); torch.save(pairs, out / 'generated_normalized_latents.pt')
        _write_csv(out / 'signal_slot_differences.csv', records, list(records[0]))
        grouped = []
        for comparison, summary in summaries.items():
            for group, values in summary['groups'].items():
                grouped.append(dict(comparison=comparison, changed_signal_lane_id=summary['changed_signal_lane_id'],
                                    group=group, **{k: json.dumps(v) if isinstance(v, list) else v for k,v in values.items()}))
        _write_csv(out / 'signal_group_differences.csv', grouped, list(grouped[0]))
        report['scene_views'] = _write_views(out, scenes, condition, variants, context, scale)
        report['output_files'] = ['generated_scenes.pt', 'generated_normalized_latents.pt', 'sampling_rng_trace.json',
                                  'approach_associations.json', 'signal_slot_differences.csv', 'signal_group_differences.csv', *report['scene_views']]
        if args.device == 'cuda': report['peak_gpu_allocated_mib'] = torch.cuda.max_memory_allocated() / 1024**2
        report['status'] = 'signal_locality_pipeline_verified'
    except Exception:
        report['error_traceback'] = traceback.format_exc(); print(report['error_traceback'])
    (out / 'locality_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['status'] == 'signal_locality_pipeline_verified' else 1


if __name__ == '__main__': raise SystemExit(main())
