"""Exploratory directed approach candidates and fixed-population E8 statistics."""
import math

import numpy as np


def build_geometry_context(arrays, metadata, signal_doc, mask_doc, scale, signal_sha):
    if mask_doc['signal_input_sha256'] != signal_sha or mask_doc.get('regular_source_point_mask_verified') is not True:
        raise ValueError('Signal source evidence differs or is unverified')
    if signal_doc['scenario_id'] != metadata['scenario_id'] or signal_doc['time_index'] != metadata['time_index']:
        raise ValueError('Signal scenario/time differs')
    roads = np.asarray(arrays['road_points'], dtype=float).copy()
    features, masks, active = (np.asarray(arrays[k]) for k in ('signal_features', 'signal_point_mask', 'signal_active'))
    perm = np.asarray(arrays['lane_new_to_ae'])
    n, points, _ = roads.shape
    if (roads.shape[-1] != 2 or not np.isfinite(roads).all() or
            not np.array_equal(np.sort(perm), np.arange(n)) or not np.array_equal(perm, metadata['lane_new_to_ae'])):
        raise ValueError('Invalid road geometry or witnessed permutation differs')
    signals = signal_doc['signals']
    if features.shape != (len(signals), 11) or masks.shape != (len(signals), n, points) or active.shape != (len(signals),):
        raise ValueError('Signal geometry dimensions differ')
    for axis, label in enumerate(('x', 'y')):
        low, high = float(scale['min_lane_' + label]), float(scale['max_lane_' + label])
        if not high > low: raise ValueError('Invalid physical coordinate range')
        roads[..., axis] = (roads[..., axis] + 1) * (high - low) / 2 + low
    rows = {}
    for index, signal in enumerate(signals):
        match = np.flatnonzero(perm == signal['regular_candidate_route']['batch_row'])
        if len(match) != 1: raise ValueError('Signal route missing from LDM permutation')
        row = int(match[0])
        if row in rows: raise ValueError('Multiple signals routed to one compact row are unsupported')
        if (not np.allclose(features[index, 9:], signal['stop_point_model_xy'], rtol=0, atol=1e-7)
                or features[index, :9].argmax() != signal['state_code']):
            raise ValueError('Signal state/Stop Point feature order differs')
        stop = np.asarray(signal['stop_point_ego_xy_m'], dtype=float)
        normalized = np.asarray(signal['stop_point_model_xy'], dtype=float)
        # This source uses ego coordinates scaled by the 64 m square FoV.
        if not np.isfinite(stop).all() or not np.allclose(stop, normalized * 32, rtol=0, atol=1e-5):
            raise ValueError('Stop Point physical/model coordinates differ')
        rows[row] = dict(signal_index=index, signal_lane_id=int(signal['raw_lane_id']),
                         compact_group_id=int(signal['group_id']), active=bool(active[index]), stop=stop)
    found = set()
    for evidence in mask_doc['evidence']:
        match = np.flatnonzero(perm == evidence['batch_row'])
        if len(match) != 1: raise ValueError('Source mask route missing')
        row = int(match[0]); description = rows.get(row)
        if not description or not description['active'] or description['signal_lane_id'] != evidence['raw_lane_id']:
            raise ValueError('Source Lane and active signal route differ')
        if description['compact_group_id'] != evidence['group_id'] or row in found:
            raise ValueError('Duplicate or mismatched compact source group')
        index = description['signal_index']
        if not np.array_equal(np.flatnonzero(masks[index, row]), evidence['valid_point_indices']) or masks[index, np.arange(n) != row].any():
            raise ValueError('Signal source mask differs from evidence')
        if len(evidence['samples']) != points: raise ValueError('Resampling source evidence point count differs')
        if not masks[index, row].any(): raise ValueError('Active source mask has no retained points')
        for point in evidence['valid_point_indices']:
            sample = evidence['samples'][point]
            if (sample['raw_lane_candidates'] != [description['signal_lane_id']]
                    or sample.get('source_status') != 'verified' or sample.get('reasons') != []):
                raise ValueError('Unverified or mismatched retained source point')
        description['samples'] = evidence['samples']; description['point_mask'] = masks[index, row]
        found.add(row)
    if found != {r for r, d in rows.items() if d['active']}:
        raise ValueError('Missing active source evidence')
    return dict(roads=roads, rows=rows, signals=signals,
                policy=dict(max_distance_m=2., max_heading_difference_deg=30., min_candidate_margin_m=.5,
                            stop_point_is_not_stop_line=True, exploratory_not_ground_truth=True))


def project(point, line):
    vector = np.diff(line, axis=0); square = np.square(vector).sum(1)
    good = square > 1e-12
    if not good.any(): return None
    fractions = np.clip(((point - line[:-1]) * vector).sum(1) / np.where(good, square, 1), 0, 1)
    distance = np.linalg.norm(point - line[:-1] - fractions[:, None] * vector, axis=1)
    distance[~good] = np.inf
    index = int(distance.argmin()); lengths = np.sqrt(square)
    return dict(distance_m=float(distance[index]), segment=index,
                tangent_deg=math.degrees(math.atan2(vector[index, 1], vector[index, 0])),
                arc_s_m=float(lengths[:index].sum() + fractions[index] * lengths[index]))


def classify_approaches(scene, context):
    states = scene['agent_states'].detach().cpu().double().numpy()
    if states.ndim != 2 or states.shape[1] != 7 or not np.isfinite(states).all():
        raise ValueError('Expected finite [Agent,7] physical states')
    results = []; policy = context['policy']
    for slot, state in enumerate(states):
        result = dict(slot=slot, signal_lane_id=None, reason='no_aligned_candidate')
        if math.hypot(state[3], state[4]) <= 1e-6:
            result['reason'] = 'invalid_heading'; results.append(result); continue
        heading = math.degrees(math.atan2(state[4], state[3])); candidates = []
        for row, road in enumerate(context['roads']):
            candidate = project(state[:2], road)
            if candidate is None: continue
            candidate['row'] = row
            candidate['heading_difference_deg'] = abs((heading - candidate['tangent_deg'] + 180) % 360 - 180)
            if candidate['heading_difference_deg'] <= policy['max_heading_difference_deg']: candidates.append(candidate)
        candidates.sort(key=lambda c: c['distance_m'])
        if not candidates: results.append(result); continue
        best = candidates[0]; result.update(best)
        margin = candidates[1]['distance_m'] - best['distance_m'] if len(candidates) > 1 else None
        result['candidate_margin_m'] = margin
        if margin is not None and margin < policy['min_candidate_margin_m']: result['reason'] = 'ambiguous_aligned_candidates'
        elif best['distance_m'] > policy['max_distance_m']: result['reason'] = 'too_far'
        else:
            description = context['rows'].get(best['row'])
            if not description or not description['active']: result['reason'] = 'not_active_signal_group'
            else:
                stop = project(description['stop'], context['roads'][best['row']])
                result['distance_to_stop_m'] = stop['arc_s_m'] - best['arc_s_m']
                if result['distance_to_stop_m'] <= 0: result['reason'] = 'downstream_of_stop_reference'
                else:
                    segment = best['segment']; samples = description['samples']
                    a, b = [samples[i]['raw_lane_candidates'] for i in (segment, segment + 1)]
                    source = a[0] if len(a) == len(b) == 1 and a == b else None
                    verified = all(samples[i].get('source_status') == 'verified' and samples[i].get('reasons') == []
                                   for i in (segment, segment + 1))
                    if source is None or not verified:
                        result['reason'] = 'unverified_source_segment'; results.append(result); continue
                    result.update(signal_lane_id=description['signal_lane_id'], signal_index=description['signal_index'],
                                  source_lane_id=source, compact_group_id=description['compact_group_id'],
                                  in_source_mask=bool(description['point_mask'][segment:segment+2].all()), reason='approach_candidate')
        results.append(result)
    return results


def _deltas(first, second):
    a, b = [s['agent_states'].detach().cpu().double().numpy() for s in (first, second)]
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Physical scenes differ in shape or contain nonfinite values')
    heading = np.arctan2(b[:, 4], b[:, 3]) - np.arctan2(a[:, 4], a[:, 3])
    return dict(position_delta_m=np.linalg.norm(b[:, :2] - a[:, :2], axis=1),
                speed_delta_mps=b[:, 2] - a[:, 2], heading_delta_deg=np.degrees(np.arctan2(np.sin(heading), np.cos(heading))))


def analyze_locality(scenes, context):
    associations = {name: classify_approaches(scene, context) for name, scene in scenes.items()}
    anchor = associations['stop']; noise = _deltas(scenes['stop'], scenes['stop_repeat'])
    floors = {key: max(.001, 5 * float(np.abs(value).max())) for key, value in noise.items()}
    summaries = {}; records = []
    for name, scene in scenes.items():
        if not name.startswith('go_'): continue
        lane = int(name[3:]); values = _deltas(scenes['stop'], scene)
        groups = dict(target=[], other_approach=[], remaining=[])
        for slot, before in enumerate(anchor):
            group = 'target' if before['signal_lane_id'] == lane else ('other_approach' if before['signal_lane_id'] is not None else 'remaining')
            groups[group].append(slot)
            records.append(dict(comparison=name, changed_signal_lane_id=lane, agent_slot=slot, group=group,
                                stop_signal_lane_id=before['signal_lane_id'], go_signal_lane_id=associations[name][slot]['signal_lane_id'],
                                same_association=before['signal_lane_id'] == associations[name][slot]['signal_lane_id'],
                                stop_candidate_row=before.get('row'), go_candidate_row=associations[name][slot].get('row'),
                                same_candidate_row=before.get('row') == associations[name][slot].get('row'),
                                **{key: float(value[slot]) for key, value in values.items()}))
        statistics = {}
        for group, slots in groups.items():
            statistics[group] = dict(slots=slots, count=len(slots),
                mean_position_delta_m=float(values['position_delta_m'][slots].mean()) if slots else None,
                max_position_delta_m=float(values['position_delta_m'][slots].max()) if slots else None,
                mean_signed_speed_delta_mps=float(values['speed_delta_mps'][slots].mean()) if slots else None,
                mean_absolute_speed_delta_mps=float(np.abs(values['speed_delta_mps'][slots]).mean()) if slots else None,
                speed_positive_above_noise_slots=[s for s in slots if values['speed_delta_mps'][s] > floors['speed_delta_mps']],
                speed_negative_above_noise_slots=[s for s in slots if values['speed_delta_mps'][s] < -floors['speed_delta_mps']],
                position_above_noise_slots=[s for s in slots if values['position_delta_m'][s] > floors['position_delta_m']])
        switches = [s for s, before in enumerate(anchor) if before['signal_lane_id'] != associations[name][s]['signal_lane_id']]
        row_switches = [s for s, before in enumerate(anchor) if before.get('row') != associations[name][s].get('row')]
        summaries[name] = dict(changed_signal_lane_id=lane, groups=statistics, association_switch_slots=switches,
                               candidate_row_switch_slots=row_switches,
                               target_stable_association_slots=[s for s in groups['target'] if s not in switches],
                               target_status='target_candidates_present' if groups['target'] else 'no_target_approach_candidates',
                               quality_improvement_verified=False)
    return summaries, records, dict(associations=associations, repeat_max_absolute={k: float(np.abs(v).max()) for k,v in noise.items()},
                                    descriptive_response_floors=floors, numeric_floor_scope='one STOP repeat; not a statistical test')
