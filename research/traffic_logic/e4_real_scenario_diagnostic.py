"""Trace one real Scenario from original Lane IDs to pre-shuffle tensor rows.

This diagnostic imports the official modules but never calls ``get_data``:
that method selects a random valid frame and writes training caches. Instead,
it supplies the verified ego state at an explicit time index to the same
official geometry methods in their normal order. Only the explicit
``--output`` and ``--log`` files are written.
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from .e4_geometry_audit import audit_lane_fov, mask_to_runs, plan_partition_lineage
from .lane_lineage import build_lane_lineage
from .tensor_rows import map_lane_ids_to_rows


SCENARIO_ID = "637f20cafde22ff8"
EXPECTED_SIGNALS = {
    443: ("STOP", 4), 445: ("STOP", 4),
    448: ("STOP", 4), 449: ("STOP", 4),
    455: ("ARROW_STOP", 1), 456: ("ARROW_STOP", 1),
    431: ("UNKNOWN", 0), 432: ("UNKNOWN", 0),
    446: ("UNKNOWN", 0), 447: ("UNKNOWN", 0),
    450: ("UNKNOWN", 0), 457: ("UNKNOWN", 0),
}


def sample_source_support(points, mask, count):
    """Conservatively bracket each resampled point by retained compact IDs.

    Brackets crossing removed original points or zero-length segments are
    marked ambiguous. This is provenance evidence, not an exact inverse of
    ``np.interp`` when its cumulative-distance inputs repeat.
    """
    points = np.asarray(points)
    kept = np.flatnonzero(mask)
    if len(kept) == 0:
        return []
    if len(kept) == 1:
        return [(int(kept[0]), int(kept[0]), False, True)] * count
    cropped = points[kept]
    lengths = np.linalg.norm(np.diff(cropped, axis=0), axis=1)
    cumulative = np.insert(np.cumsum(lengths), 0, 0.0)
    targets = np.linspace(0, cumulative[-1], count)
    left = np.searchsorted(cumulative, targets, side="right") - 1
    left = np.clip(left, 0, len(kept) - 2)
    return [
        (int(kept[i]), int(kept[i + 1]),
         bool(kept[i + 1] != kept[i] + 1), bool(lengths[i] == 0))
        for i in left
    ]


def child_source_evidence(support, compact_range, child_range):
    """Count direct and boundary-mixed source brackets in a child slice."""
    start, end = compact_range
    child_start, child_end = child_range
    direct = mixed = gap = zero = 0
    for left, right, discontinuous, repeated in support[child_start:child_end]:
        left_local = start <= left < end
        right_local = start <= right < end
        if left_local and right_local:
            direct += 1
        elif left_local or right_local:
            mixed += 1
        if left_local or right_local:
            gap += int(discontinuous)
            zero += int(repeated)
    return {
        "direct_samples": direct,
        "mixed_boundary_samples": mixed,
        "cross_gap_samples": gap,
        "zero_length_samples": zero,
    }


def final_source_evidence(parent_support, final_support, compact_range,
                          parent_start=0):
    """Compose the 1000-point and 20-point interpolation source brackets."""
    start, end = compact_range
    direct = mixed = ambiguous = 0
    for left, right, gap, zero in final_support:
        if parent_start + max(left, right) >= len(parent_support):
            raise ValueError("20-point source bracket is outside the parent")
        parent_brackets = (
            parent_support[parent_start + left],
            parent_support[parent_start + right],
        )
        source_ids = (parent_brackets[0][0], parent_brackets[0][1],
                      parent_brackets[1][0], parent_brackets[1][1])
        local_count = sum(start <= source_id < end for source_id in source_ids)
        if local_count == len(source_ids):
            direct += 1
        elif local_count:
            mixed += 1
        if local_count and (gap or zero or
                            any(bracket[2] or bracket[3]
                                for bracket in parent_brackets)):
            ambiguous += 1
    return {
        "direct_tensor_points": direct,
        "mixed_tensor_points": mixed,
        "ambiguous_tensor_points": ambiguous,
    }


def capture_official_rows(dataset_class, proxy, graph, num_points, max_lanes):
    """Capture the actual np.argsort result in one official tensor call."""
    calls = []
    original_argsort = np.argsort

    def capture_argsort(values, *args, **kwargs):
        ordered = original_argsort(values, *args, **kwargs)
        calls.append((np.asarray(values).copy(), np.asarray(ordered).copy()))
        return ordered

    np.argsort = capture_argsort
    try:
        output = dataset_class.get_road_points_adj(proxy, graph)
    finally:
        np.argsort = original_argsort
    if len(calls) != 1:
        raise RuntimeError(f"Expected one distance sort, got {len(calls)}")
    distances, full_order = calls[0]
    road_points, *_, num_lanes = output
    graph_ids = list(graph["lanes"])
    selected = [int(index) for index in full_order[:num_lanes]]
    if num_lanes != min(len(graph_ids), max_lanes):
        raise RuntimeError("Official max_num_lanes result disagrees with cfg")
    if len(distances) != len(graph_ids):
        raise RuntimeError("Official distance sort has wrong length")
    if road_points.shape != (num_lanes, num_points, 2):
        raise RuntimeError(f"Unexpected official road_points shape: {road_points.shape}")
    from utils.lane_graph_helpers import resample_polyline
    expected_points = np.asarray([
        resample_polyline(graph["lanes"][lane_id], num_points=num_points)
        for lane_id in graph_ids
    ])[selected]
    if not np.array_equal(road_points, expected_points):
        raise RuntimeError("Official tensor rows disagree with captured sort indices")
    rows = map_lane_ids_to_rows(graph_ids, graph_ids, selected)
    if len(set(selected)) != len(selected):
        raise RuntimeError("Official distance sort repeated an input index")
    return {
        "graph_ids_in_iteration_order": graph_ids,
        "minimum_distances": distances.tolist(),
        "full_argsort_indices": [int(index) for index in full_order],
        "selected_indices": selected,
        "selected_lane_ids": [graph_ids[index] for index in selected],
        "lane_id_to_tensor_row": rows,
        "num_lanes": int(num_lanes),
        "road_points": road_points,
    }


def run(tfrecord, repo, time_index=10, observer=None):
    """Run the complete read-only E4-A trace on a single Scenario record."""
    if time_index != 10:
        raise ValueError("This verified E4-A scenario requires time_index=10")
    repo = Path(repo).resolve()
    tfrecord = Path(tfrecord).resolve()
    sys.path.insert(0, str(repo))
    import tensorflow as tf
    from hydra import compose, initialize_config_dir
    from google.protobuf.json_format import MessageToDict
    from waymo_open_dataset.protos import scenario_pb2
    from data_processing.waymo import generate_waymo_dataset as preprocessor
    from datasets.waymo.dataset_autoencoder_waymo import WaymoDatasetAutoEncoder
    from utils import lane_graph_helpers as lane_helpers
    from utils.data_helpers import extract_raw_waymo_data, modify_agent_states

    log = []
    with initialize_config_dir(config_dir=str(repo / "cfgs"), version_base=None):
        cfg = compose(config_name="config").ae.dataset
    cfg_values = {
        key: cfg[key] for key in (
            "fov", "upsample_lane_num_points", "num_points_per_lane",
            "max_num_lanes", "preprocess", "generate_only_vehicles",
            "remove_offroad_agents", "max_num_agents", "offroad_threshold",
        )
    }
    proxy = SimpleNamespace(cfg=cfg)
    log.append(f"cfg={cfg_values}")
    official_files = (
        "data_processing/waymo/generate_waymo_dataset.py",
        "datasets/waymo/dataset_autoencoder_waymo.py",
        "utils/lane_graph_helpers.py",
        "utils/data_helpers.py",
        "cfgs/config.yaml",
        "cfgs/dataset/waymo_base.yaml",
    )
    repo_status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "-sb"], text=True).strip()
    source_blobs = {
        path: subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", f"HEAD:{path}"],
            text=True,
        ).strip()
        for path in official_files
    }
    log.append(f"official_repo_status={repo_status}; source_blobs={source_blobs}")

    payload = next(iter(tf.data.TFRecordDataset(
        [str(tfrecord)], compression_type=""))).numpy()
    proto = scenario_pb2.Scenario()
    proto.ParseFromString(payload)
    if proto.scenario_id != SCENARIO_ID or proto.current_time_index != time_index:
        raise RuntimeError("Scenario ID or current_time_index differs from baseline")
    signals = {int(signal.lane): int(signal.state)
               for signal in proto.dynamic_map_states[time_index].lane_states}
    if signals != {lane_id: state for lane_id, (_, state) in EXPECTED_SIGNALS.items()}:
        raise RuntimeError("t=10 controlled Lane IDs or states differ from baseline")
    scenario = MessageToDict(proto)
    road_info = preprocessor.get_road_info([scenario], 0)
    data = {"road_info": road_info}
    data["graph"] = preprocessor.get_lane_graph(data)
    processed = preprocessor.process_lanegraph(data)
    objects, av_index = preprocessor.get_objects([scenario], 0)
    agent_states_all, agent_types_all = extract_raw_waymo_data(objects)
    valid_timesteps = np.flatnonzero(agent_states_all[av_index, :, -1] == 1)
    if not proto.tracks[proto.sdc_track_index].states[time_index].valid:
        raise RuntimeError("Raw Proto ego state is invalid at t=10")
    if time_index not in valid_timesteps:
        raise RuntimeError("Official extracted ego state is invalid at t=10")
    normalize_dict = {
        "center": agent_states_all[av_index, time_index, :2].copy(),
        "yaw": agent_states_all[av_index, time_index, 4].copy(),
    }
    log.append(f"ego_t10_valid=True center={normalize_dict['center'].tolist()} "
               f"yaw={float(normalize_dict['yaw'])}")

    captured = []
    original_find = lane_helpers.find_lane_groups

    def capture_groups(pre_pairs, suc_pairs):
        groups = original_find(pre_pairs, suc_pairs)
        captured.append(copy.deepcopy(groups))
        return groups

    lane_helpers.find_lane_groups = capture_groups
    try:
        compact = lane_helpers.get_compact_lane_graph(
            copy.deepcopy({"lane_graph": processed}))
    finally:
        lane_helpers.find_lane_groups = original_find
    if len(captured) != 1:
        raise RuntimeError("find_lane_groups was not captured exactly once")
    groups = captured[0]
    raw_ids = {int(lane_id) for lane_id in road_info["lane"]}
    processed_ids = set(processed["lanes"])
    grouped_ids = {lane_id for members in groups.values() for lane_id in members}
    if (len(raw_ids), len(processed_ids), len(groups)) != (199, 198, 135):
        raise RuntimeError("Raw/processed/compact counts differ from E3-C")
    if raw_ids - processed_ids != {526} or grouped_ids != processed_ids:
        raise RuntimeError("Original-to-compact Lane set reconciliation failed")
    if road_info["lane"]["526"]["type"] != "TYPE_BIKE_LANE":
        raise RuntimeError("Lane 526 filter reason differs from E3-C")
    log.append("compact=199 raw / 198 processed / 135 groups; capture_calls=1")

    full_lineage_error = None
    try:
        build_lane_lineage(groups, processed["lanes"])
    except ValueError as error:
        full_lineage_error = str(error)
    if full_lineage_error != "Source Lane needs at least 2 points: 529":
        raise RuntimeError(f"Unexpected full-graph lineage result: {full_lineage_error}")
    invalid_lanes = {
        lane_id for lane_id, points in processed["lanes"].items()
        if len(points) < 2 or not np.isfinite(points).all()
        or np.linalg.norm(np.diff(points, axis=0), axis=1).sum() == 0
    }
    invalid_groups = {
        group_id for group_id, members in groups.items()
        if invalid_lanes.intersection(members)
    }
    valid_groups = {group_id: members for group_id, members in groups.items()
                    if group_id not in invalid_groups}
    valid_ids = {lane_id for members in valid_groups.values() for lane_id in members}
    lineage = build_lane_lineage(
        valid_groups, {lane_id: processed["lanes"][lane_id] for lane_id in valid_ids})
    if valid_ids != set(lineage) or valid_ids | grouped_ids.intersection(invalid_lanes) != processed_ids:
        raise RuntimeError("Valid lineage and exception ledger do not reconcile")
    if invalid_lanes != {529} or invalid_groups != {0} or groups[0] != [529]:
        raise RuntimeError("Lane 529 exception differs from E3-C")
    if not set(EXPECTED_SIGNALS).issubset(lineage):
        raise RuntimeError("A controlled Lane is missing from strict lineage")
    log.append("lineage=197 valid Lanes; exception Group 0 [529]")

    normalized = WaymoDatasetAutoEncoder.normalize_compact_lane_graph(
        proxy, copy.deepcopy(compact), normalize_dict)
    masks = {
        group_id: ((np.abs(points[:, 0]) < cfg.fov / 2)
                   & (np.abs(points[:, 1]) < cfg.fov / 2))
        for group_id, points in normalized["lanes"].items()
    }
    fov = WaymoDatasetAutoEncoder.get_lane_graph_within_fov(proxy, normalized)
    if set(fov["lanes"]) != {group_id for group_id, mask in masks.items() if mask.any()}:
        raise RuntimeError("FoV Group IDs do not match official point masks")
    for group_id, points in fov["lanes"].items():
        expected_points = lane_helpers.resample_polyline(
            normalized["lanes"][group_id][masks[group_id]],
            num_points=cfg.upsample_lane_num_points)
        if not np.array_equal(points, expected_points):
            raise RuntimeError(f"FoV resampling differs for Group {group_id}")
    log.append(f"fov_groups={len(fov['lanes'])}")

    partition_plan = plan_partition_lineage(fov["lanes"])
    partitioned = WaymoDatasetAutoEncoder.partition_compact_lane_graph(
        proxy, copy.deepcopy(fov))
    expected_partition_ids = set()
    for parent_id, entry in partition_plan.items():
        for child_id, (start, end) in entry["children"]:
            expected_partition_ids.add(child_id)
            if child_id not in partitioned["lanes"]:
                raise RuntimeError(f"Partition child {child_id} was not produced")
            if not np.array_equal(partitioned["lanes"][child_id],
                                  fov["lanes"][parent_id][start:end]):
                raise RuntimeError(f"Partition child {child_id} has wrong parent slice")
    if set(partitioned["lanes"]) != expected_partition_ids:
        raise RuntimeError("Official Partition IDs differ from captured slice plan")
    log.append(f"partition_lanes={len(partitioned['lanes'])}; verified_parent_slices=True")

    # Mirror the get_data validity gate at the explicit time index. The
    # official get_data method itself randomly chooses a frame and writes a
    # cache, so it must not be invoked for this fixed-time diagnostic.
    existing = copy.deepcopy(agent_states_all[:, time_index, -1]).astype(bool)
    if cfg.generate_only_vehicles:
        supported_type = copy.deepcopy(agent_types_all[:, 1]).astype(bool)
    else:
        supported_type = copy.deepcopy(
            agent_types_all[:, 1] + agent_types_all[:, 2] + agent_types_all[:, 3]
        ).astype(bool)
    agent_mask = existing * supported_type
    agent_states = copy.deepcopy(agent_states_all[agent_mask, time_index])
    agent_types = copy.deepcopy(agent_types_all[agent_mask])
    selected_count = len(agent_states)
    agent_states, agent_types = WaymoDatasetAutoEncoder.get_agents_within_fov(
        proxy, agent_states, agent_types, normalize_dict)
    fov_agent_count = len(agent_states)
    if cfg.remove_offroad_agents:
        agent_states, agent_types = WaymoDatasetAutoEncoder.remove_offroad_agents(
            proxy, agent_states, agent_types, fov["lanes"])
    offroad_agent_count = len(agent_states)
    agent_states = modify_agent_states(agent_states)
    agent_gate = {
        "existing_count": int(existing.sum()),
        "existing_supported_type_count": int(selected_count),
        "after_fov_count": int(fov_agent_count),
        "after_offroad_count": int(offroad_agent_count),
        "after_modify_count": int(len(agent_states)),
        "would_pass_get_data_gate": bool(len(agent_states) > 0 and len(fov["lanes"]) > 0),
    }
    log.append(f"fixed_t10_agent_gate={agent_gate}")
    if not agent_gate["would_pass_get_data_gate"]:
        raise RuntimeError("The fixed t=10 scene fails the official get_data agent/lane gate")

    regular_rows = capture_official_rows(
        WaymoDatasetAutoEncoder, proxy, fov,
        cfg.num_points_per_lane, cfg.max_num_lanes)
    partition_rows = capture_official_rows(
        WaymoDatasetAutoEncoder, proxy, partitioned,
        cfg.num_points_per_lane, cfg.max_num_lanes)
    log.append(f"tensor_rows_regular={regular_rows['num_lanes']} "
               f"partitioned={partition_rows['num_lanes']}; pre_shuffle=True")

    cases = []
    for lane_id in (*EXPECTED_SIGNALS, 529):
        group_id = next(group_id for group_id, members in groups.items()
                        if lane_id in members)
        group_mask = masks[group_id]
        group_kept, group_runs = mask_to_runs(group_mask)
        group_state = ("filtered" if not group_kept else
                       "retained" if len(group_kept) == len(group_mask) else "partial")
        if lane_id == 529:
            compact_range = (0, len(compact["lanes"][group_id]))
            source_range = (0, len(processed["lanes"][lane_id]))
        else:
            compact_range = lineage[lane_id]["compact_point_range"]
            source_range = lineage[lane_id]["source_point_range"]
        audit = audit_lane_fov(
            normalized["lanes"][group_id], group_mask, compact_range,
            source_point_count=len(processed["lanes"][lane_id]))
        support = sample_source_support(
            normalized["lanes"][group_id], group_mask,
            cfg.upsample_lane_num_points)
        regular_source_evidence = child_source_evidence(
            support, compact_range, (0, len(support)))
        child_records = []
        if group_id in partition_plan:
            for child_id, child_range in partition_plan[group_id]["children"]:
                evidence = child_source_evidence(support, compact_range, child_range)
                if evidence["direct_samples"] or evidence["mixed_boundary_samples"] or lane_id == 529:
                    child_points = partitioned["lanes"][child_id]
                    final_support = sample_source_support(
                        child_points, np.ones(len(child_points), dtype=bool),
                        cfg.num_points_per_lane)
                    child_records.append({
                        "child_id": child_id,
                        "parent_upsampled_point_range": child_range,
                        "tensor_row": partition_rows["lane_id_to_tensor_row"].get(child_id),
                        "source_evidence": evidence,
                        "final_20_point_source_evidence": final_source_evidence(
                            support, final_support, compact_range,
                            parent_start=child_range[0]),
                    })
        if group_id in fov["lanes"]:
            regular_points = fov["lanes"][group_id]
            regular_support = sample_source_support(
                regular_points, np.ones(len(regular_points), dtype=bool),
                cfg.num_points_per_lane)
            regular_final_evidence = final_source_evidence(
                support, regular_support, compact_range)
        else:
            regular_final_evidence = {
                "direct_tensor_points": 0,
                "mixed_tensor_points": 0,
                "ambiguous_tensor_points": 0,
            }
        regular_source_uncertain = bool(
            audit["uncertain"] or
            regular_source_evidence["mixed_boundary_samples"] or
            regular_source_evidence["cross_gap_samples"] or
            regular_source_evidence["zero_length_samples"] or
            regular_final_evidence["mixed_tensor_points"] or
            regular_final_evidence["ambiguous_tensor_points"]
        )
        partition_source_uncertain = bool(
            audit["uncertain"] or
            (audit["kept_local_point_count"] and not child_records) or
            any(child["source_evidence"]["mixed_boundary_samples"] or
                child["source_evidence"]["cross_gap_samples"] or
                child["source_evidence"]["zero_length_samples"]
                for child in child_records) or
            any(child["final_20_point_source_evidence"]["mixed_tensor_points"] or
                child["final_20_point_source_evidence"]["ambiguous_tensor_points"]
                for child in child_records)
        )
        source_uncertain = regular_source_uncertain or partition_source_uncertain
        row_regular = regular_rows["lane_id_to_tensor_row"].get(group_id)
        has_partition_row = any(child["tensor_row"] is not None
                                for child in child_records)
        regular_final_supported = bool(
            regular_final_evidence["direct_tensor_points"] or
            regular_final_evidence["mixed_tensor_points"])
        partition_final_supported = any(
            child["tensor_row"] is not None and
            (child["final_20_point_source_evidence"]["direct_tensor_points"] or
             child["final_20_point_source_evidence"]["mixed_tensor_points"])
            for child in child_records)
        if lane_id == 529:
            final_status = "invalid"
        elif audit["point_status"] == "filtered":
            final_status = "filtered"
        elif row_regular is None and not has_partition_row:
            final_status = "filtered"
        elif source_uncertain or not (regular_final_supported or partition_final_supported):
            final_status = "uncertain"
        else:
            final_status = audit["point_status"]
        reasons = list(audit["reasons"])
        if audit["point_status"] == "partial":
            reasons.append("local_compact_points_partly_outside_fov")
        if audit["point_status"] == "filtered" and group_state != "filtered":
            reasons.append("group_in_fov_but_local_points_filtered")
        if any(child["source_evidence"]["mixed_boundary_samples"]
               for child in child_records) or regular_source_evidence["mixed_boundary_samples"]:
            reasons.append("resampling_brackets_cross_raw_lane_boundary")
        if any(child["source_evidence"]["cross_gap_samples"]
               for child in child_records) or regular_source_evidence["cross_gap_samples"]:
            reasons.append("resampling_brackets_cross_removed_fov_points")
        if any(child["source_evidence"]["zero_length_samples"]
               for child in child_records) or regular_source_evidence["zero_length_samples"]:
            reasons.append("resampling_brackets_zero_length_segment")
        if (regular_final_evidence["mixed_tensor_points"] or
                any(child["final_20_point_source_evidence"]["mixed_tensor_points"]
                    for child in child_records)):
            reasons.append("final_resampling_brackets_cross_raw_lane_boundary")
        if (regular_final_evidence["ambiguous_tensor_points"] or
                any(child["final_20_point_source_evidence"]["ambiguous_tensor_points"]
                    for child in child_records)):
            reasons.append("final_resampling_has_ambiguous_source_segment")
        if group_id in fov["lanes"] and row_regular is None:
            reasons.append("regular_max_num_lanes")
        if child_records and all(child["tensor_row"] is None for child in child_records):
            reasons.append("partitioned_max_num_lanes")
        if audit["kept_local_point_count"] and not child_records:
            reasons.append("no_resampled_child_source_support")
        if audit["kept_local_point_count"] and row_regular is not None and not regular_final_supported:
            reasons.append("no_regular_20_point_local_source_support")
        if audit["kept_local_point_count"] and has_partition_row and not partition_final_supported:
            reasons.append("no_partitioned_20_point_local_source_support")
        if lane_id == 529 and group_state == "filtered":
            reasons.append("single_point_group_filtered_by_fov")
        if source_uncertain and not reasons:
            reasons.append("resampling_source_ambiguous")
        if lane_id == 529:
            regular_status = partition_status = "invalid"
        else:
            regular_status = ("filtered" if row_regular is None or
                              audit["point_status"] == "filtered" else
                              "uncertain" if regular_source_uncertain or not regular_final_supported else
                              audit["point_status"])
            partition_status = ("filtered" if not has_partition_row or
                                audit["point_status"] == "filtered" else
                                "uncertain" if partition_source_uncertain or not partition_final_supported else
                                audit["point_status"])
        case = {
            "raw_lane_id": lane_id,
            "signal_state": EXPECTED_SIGNALS.get(lane_id, (None, None))[0],
            "group_id": group_id,
            "group_members": groups[group_id],
            "source_point_range": source_range,
            "compact_point_range": compact_range,
            "group_fov_status": group_state,
            "group_fov_mask_bits": "".join("1" if bit else "0" for bit in group_mask),
            "group_fov_kept_compact_indices": group_kept,
            "group_fov_contiguous_spans": group_runs,
            "local_fov_status": audit["status"],
            "local_fov_point_status": audit["point_status"],
            "local_kept_compact_indices": [compact_range[0] + offset
                                           for offset in audit["kept_local_point_offsets"]],
            "local_kept_source_indices": [source_range[0] + offset
                                          for offset in audit["kept_local_point_offsets"]],
            "local_kept_offset_spans": audit["kept_local_offset_runs"],
            "local_kept_source_spans": [
                (source_range[0] + start, source_range[0] + end)
                for start, end in audit["kept_local_offset_runs"]
            ],
            "local_kept_point_count": audit["kept_local_point_count"],
            "local_contributed_point_count": audit["local_point_count"],
            "source_uncertain_after_resampling": source_uncertain,
            "regular_source_uncertain": regular_source_uncertain,
            "partitioned_source_uncertain": partition_source_uncertain,
            "regular_tensor_row": row_regular,
            "regular_1000_point_source_evidence": regular_source_evidence,
            "regular_final_20_point_source_evidence": regular_final_evidence,
            "regular_input_status": regular_status,
            "partition_children": child_records,
            "partitioned_input_status": partition_status,
            "final_status": final_status,
            "reasons": reasons,
        }
        if lane_id == 529:
            case["fov_resampled_unique_points"] = (
                len(np.unique(fov["lanes"][group_id], axis=0))
                if group_id in fov["lanes"] else None)
            case["regular_tensor_unique_points"] = (
                len(np.unique(regular_rows["road_points"][row_regular], axis=0))
                if row_regular is not None else None)
            case["partition_tensor_unique_points"] = {
                str(child["child_id"]): len(np.unique(
                    partition_rows["road_points"][child["tensor_row"]], axis=0))
                for child in child_records if child["tensor_row"] is not None
            }
        cases.append(case)
    log.append("controlled_lanes=12; lane529_audited=True")

    def strip_road_points(branch):
        return {key: value for key, value in branch.items() if key != "road_points"}

    if observer is not None:
        observer({
            "proto": proto,
            "processed": processed,
            "groups": groups,
            "lineage": lineage,
            "compact": compact,
            "normalized": normalized,
            "fov_masks": masks,
            "fov": fov,
            "partitioned": partitioned,
            "partition_plan": partition_plan,
            "regular_rows": regular_rows,
            "partitioned_rows": partition_rows,
            "normalize_dict": normalize_dict,
            "cfg": cfg,
        })

    return {
        "run_status": "pass",
        "environment": {
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "gpu_visible": len(tf.config.list_physical_devices("GPU")),
            "repo": str(repo),
            "repo_commit": subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
            "repo_status": repo_status,
            "official_source_blobs": source_blobs,
            "cfg": cfg_values,
        },
        "input": {
            "tfrecord": str(tfrecord),
            "tfrecord_bytes": tfrecord.stat().st_size,
            "record_index": 0,
            "record_payload_sha256": hashlib.sha256(payload).hexdigest(),
            "scenario_id": proto.scenario_id,
            "time_index": time_index,
            "av_index": int(av_index),
            "valid_timestep_count": len(valid_timesteps),
            "ego_t10_center": normalize_dict["center"].tolist(),
            "ego_t10_yaw": float(normalize_dict["yaw"]),
            "signal_states": signals,
        },
        "counts": {
            "raw_lanes": len(raw_ids),
            "processed_lanes": len(processed_ids),
            "compact_groups": len(groups),
            "fov_groups": len(fov["lanes"]),
            "partitioned_lanes": len(partitioned["lanes"]),
            "regular_tensor_rows": regular_rows["num_lanes"],
            "partitioned_tensor_rows": partition_rows["num_lanes"],
        },
        "lineage_audit": {
            "full_graph_error": full_lineage_error,
            "strict_valid_lane_count": len(lineage),
            "exception_lane_ids": sorted(invalid_lanes),
            "exception_group_ids": sorted(invalid_groups),
        },
        "agent_gate": agent_gate,
        "partition_parent_children": partition_plan,
        "regular": strip_road_points(regular_rows),
        "partitioned": strip_road_points(partition_rows),
        "lanes": cases,
        "log": log,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfrecord", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--time-index", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.tfrecord, args.repo, args.time_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False,
                                      allow_nan=False) + "\n", encoding="utf-8")
    args.log.write_text("\n".join(result["log"]) + "\n", encoding="utf-8")
    print("\n".join(result["log"]))
    print(f"output={args.output} log={args.log}")


if __name__ == "__main__":
    main()
