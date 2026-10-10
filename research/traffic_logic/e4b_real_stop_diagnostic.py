"""Read-only real Stop Point audit using one E4-A official processing run.

The official road tensor contains polylines, not traffic-signal features.
This script asks whether each true Stop Point's Lane position has defensible
geometric provenance in those polylines. It never calls random-frame get_data.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .e4_real_scenario_diagnostic import (
    EXPECTED_SIGNALS,
    run as run_e4a,
    sample_source_support,
)
from .e4b_stop_geometry import (
    audit_stop_fov,
    audit_stop_resampled_support,
    project_stop_point,
)


TARGET_LANES = (443, 445, 448, 449, 455, 456)
EXPECTED_RECORD_SHA256 = (
    "7af5edce831d6aec3eae0f665ee137b3eaf97a949cec6c9ea1e0127e1be46442"
)


def _projection_summary(projection):
    if projection is None:
        return None
    return {
        "status": projection["status"],
        "reasons": projection["reasons"],
        "nearest_segment_index": projection["nearest_segment_index"],
        "candidate_segment_indices": projection["candidate_segment_indices"],
        "candidate_projections": projection["candidate_projections"],
        "segment_fraction": projection["segment_fraction"],
        "projected_xy": projection["projected_xy"],
        "distance_m": projection["distance_m"],
        "arc_length_m": projection["arc_length_m"],
        "total_arc_length_m": projection["total_arc_length_m"],
        "position_ratio": projection["position_ratio"],
    }


def _branch_evidence(fov_audit, parent_support, branch_points, road_points,
                     tensor_row, stop_ego, *, child_range=None,
                     shared_endpoints=()):
    """Audit the actual 20-point row at the Stop Point's source position."""
    final_support = sample_source_support(
        branch_points, np.ones(len(branch_points), dtype=bool), 20)
    source = audit_stop_resampled_support(
        fov_audit, parent_support, final_support, tensor_row,
        child_range=child_range, shared_endpoints=shared_endpoints)
    relevant = set(source["exact_sample_indices"])
    for left, right in source["candidate_adjacent_sample_pairs"]:
        relevant.update((left, right))
    bounds = [entry for entry in source["final_sample_source_bounds"]
              if entry["sample_index"] in relevant]
    final_projection = (project_stop_point(
        road_points[tensor_row], stop_ego, on_centerline_tolerance_m=None)
                        if tensor_row is not None else None)
    status = source["status"]
    reasons = list(source["reasons"])
    if (status != "filtered" and final_projection is not None and
            final_projection["status"] != "retained"):
        if status == "attributable":
            status = "uncertain"
        reasons.append("nonunique_or_degenerate_final_geometry_projection")
    exact_vertex = (bool(np.any(np.all(
        road_points[tensor_row] == stop_ego, axis=1)))
        if tensor_row is not None else False)
    return {
        "tensor_row": tensor_row,
        "tensor_row_exists": tensor_row is not None,
        "stop_source_status": status,
        "reasons": tuple(dict.fromkeys(reasons)),
        "candidate_adjacent_20_point_pairs": source["candidate_adjacent_sample_pairs"],
        "proven_adjacent_20_point_pairs": source["proven_adjacent_sample_pairs"],
        "exact_source_sample_indices": source["exact_sample_indices"],
        "partition_shared_endpoint_candidates": source["shared_endpoint_candidates"],
        "relevant_20_point_source_bounds": bounds,
        "stop_exactly_equals_final_vertex": exact_vertex,
        "stop_projection_on_final_polyline": _projection_summary(final_projection),
    }


def audit_one_lane(context, baseline_case, signal):
    """Audit one Proto signal against graphs captured from that same run."""
    from utils.geometry import apply_se2_transform

    lane_id = int(signal.lane)
    state_name, state_code = EXPECTED_SIGNALS[lane_id]
    if int(signal.state) != state_code or not signal.HasField("stop_point"):
        raise RuntimeError(f"Lane {lane_id} signal state or Stop Point changed")
    stop_xyz = np.array((signal.stop_point.x, signal.stop_point.y,
                         signal.stop_point.z), dtype=float)
    if not np.isfinite(stop_xyz).all():
        raise RuntimeError(f"Lane {lane_id} Stop Point is nonfinite")
    proto_features = [feature for feature in context["proto"].map_features
                      if int(feature.id) == lane_id and feature.HasField("lane")]
    if len(proto_features) != 1:
        raise RuntimeError(f"Lane {lane_id} Proto centerline is not unique")
    raw_xyz = np.array([(point.x, point.y, point.z)
                        for point in proto_features[0].lane.polyline], dtype=float)
    processed_xy = context["processed"]["lanes"][lane_id]
    if raw_xyz.shape[0] != len(processed_xy) or not np.allclose(
            raw_xyz[:, :2], processed_xy, rtol=0, atol=1e-9):
        raise RuntimeError(f"Lane {lane_id} official processed centerline differs from Proto")
    projection = project_stop_point(raw_xyz, stop_xyz)

    normalize = context["normalize_dict"]
    yaw = normalize["yaw"]
    angle = np.pi / 2 + np.sign(-yaw) * np.abs(yaw)
    stop_ego = apply_se2_transform(
        stop_xyz[:2][np.newaxis, np.newaxis, :],
        normalize["center"][np.newaxis, np.newaxis, :], angle)[0, 0]
    half_fov = context["cfg"].fov / 2
    stop_inside_fov = bool(np.all(np.abs(stop_ego) < half_fov))

    group_id = baseline_case["group_id"]
    members = context["groups"][group_id]
    member_index = members.index(lane_id)
    source_range = context["lineage"][lane_id]["source_point_range"]
    compact_range = context["lineage"][lane_id]["compact_point_range"]
    if (tuple(source_range) != tuple(baseline_case["source_point_range"]) or
            tuple(compact_range) != tuple(baseline_case["compact_point_range"])):
        raise RuntimeError(f"Lane {lane_id} lineage differs within one run")
    mask = context["fov_masks"][group_id]
    kept = np.flatnonzero(mask)
    fov_audit = audit_stop_fov(
        projection, source_range, compact_range, kept, stop_inside_fov)

    boundary_index = (compact_range[0] - 1
                      if source_range[0] and compact_range[0] else None)
    predecessor_id = members[member_index - 1] if member_index else None
    predecessor_end_distance = (
        float(np.linalg.norm(stop_xyz[:2] -
                             context["processed"]["lanes"][predecessor_id][-1]))
        if predecessor_id is not None else None)
    boundary_ego_distance = (
        float(np.linalg.norm(context["normalized"]["lanes"][group_id][boundary_index]
                             - stop_ego))
        if boundary_index is not None else None)
    boundary_kept = (bool(mask[boundary_index])
                     if boundary_index is not None else None)
    shared_boundary_coordinate = bool(
        predecessor_id is not None and boundary_index is not None and
        predecessor_end_distance <= 1e-9 and boundary_ego_distance <= 1e-9)

    normalized_group = context["normalized"]["lanes"][group_id]
    cropped_group = normalized_group[mask]
    duplicate_cropped_edges = int(np.sum(
        np.linalg.norm(np.diff(cropped_group, axis=0), axis=1) == 0))
    parent_support = sample_source_support(
        normalized_group, mask, context["cfg"].upsample_lane_num_points)
    fov_group = context["fov"]["lanes"].get(group_id)
    parent_projection = (project_stop_point(
        fov_group, stop_ego, on_centerline_tolerance_m=None)
                         if fov_group is not None and len(fov_group) >= 2 else None)
    parent_near_source = None
    if parent_projection is not None:
        index = parent_projection["nearest_segment_index"]
        parent_near_source = (parent_support[index], parent_support[index + 1])

    regular_row = context["regular_rows"]["lane_id_to_tensor_row"].get(group_id)
    regular = _branch_evidence(
        fov_audit, parent_support, fov_group,
        context["regular_rows"]["road_points"], regular_row, stop_ego)

    plan = context["partition_plan"][group_id]
    partitioned = []
    for child_id, child_range in plan["children"]:
        child_points = context["partitioned"]["lanes"][child_id]
        child_row = context["partitioned_rows"]["lane_id_to_tensor_row"].get(child_id)
        branch = _branch_evidence(
            fov_audit, parent_support, child_points,
            context["partitioned_rows"]["road_points"], child_row,
            stop_ego, child_range=child_range,
            shared_endpoints=plan["shared_endpoints"])
        branch.update({
            "child_id": child_id,
            "parent_1000_point_range": child_range,
            "contains_parent_stop_projection_segment": bool(
                parent_projection is not None and
                child_range[0] <= parent_projection["nearest_segment_index"] and
                parent_projection["nearest_segment_index"] + 1 < child_range[1]),
        })
        partitioned.append(branch)

    return {
        "raw_lane_id": lane_id,
        "signal_state": state_name,
        "signal_state_code": state_code,
        "stop_point_global_xyz": stop_xyz.tolist(),
        "stop_point_ego_xy": stop_ego.tolist(),
        "original_lane_projection": _projection_summary(projection),
        "raw_stop_equals_lane_first_point": bool(np.array_equal(stop_xyz, raw_xyz[0])),
        "stop_to_lane_first_xy_distance_m": float(
            np.linalg.norm(stop_xyz[:2] - raw_xyz[0, :2])),
        "stop_to_lane_first_xyz_distance_m": float(
            np.linalg.norm(stop_xyz - raw_xyz[0])),
        "raw_lane_point_count": len(raw_xyz),
        "compact_contribution_fov_status": baseline_case["local_fov_point_status"],
        "compact_contribution_kept_source_spans": baseline_case["local_kept_source_spans"],
        "group_id": group_id,
        "group_members": members,
        "member_index": member_index,
        "compact_point_range": compact_range,
        "source_point_range": source_range,
        "shared_boundary": {
            "predecessor_lane_id": predecessor_id,
            "compact_vertex_index": boundary_index,
            "predecessor_end_distance_m": predecessor_end_distance,
            "stop_to_compact_boundary_ego_distance_m": boundary_ego_distance,
            "same_coordinate": shared_boundary_coordinate,
            "compact_vertex_kept_by_fov": boundary_kept,
        },
        "stop_point_inside_fov": stop_inside_fov,
        "stop_fov_source_audit": fov_audit,
        "fov_group_duplicate_edge_count": duplicate_cropped_edges,
        "stop_projection_on_fov_1000_polyline": _projection_summary(parent_projection),
        "stop_nearby_1000_source_brackets": parent_near_source,
        "regular": regular,
        "partitioned": partitioned,
    }


def run(tfrecord, repo, time_index=10):
    """Run E4-A once, then audit six Stop Points from its live objects."""
    captures = []
    baseline = run_e4a(tfrecord, repo, time_index, observer=captures.append)
    if len(captures) != 1:
        raise RuntimeError("E4-A observer did not capture exactly one processing instance")
    if baseline["input"]["record_payload_sha256"] != EXPECTED_RECORD_SHA256:
        raise RuntimeError("TFRecord record 0 differs from the verified E4-A input")
    context = captures[0]
    signals = {}
    for signal in context["proto"].dynamic_map_states[time_index].lane_states:
        lane_id = int(signal.lane)
        if lane_id in TARGET_LANES:
            if lane_id in signals:
                raise RuntimeError(f"Duplicate signal Lane {lane_id}")
            signals[lane_id] = signal
    if set(signals) != set(TARGET_LANES):
        raise RuntimeError("The six target signals were not all present")
    baseline_cases = {case["raw_lane_id"]: case for case in baseline["lanes"]}
    lanes = [audit_one_lane(context, baseline_cases[lane_id], signals[lane_id])
             for lane_id in TARGET_LANES]
    log = [
        f"scenario={baseline['input']['scenario_id']} record=0 t={time_index}",
        f"record_sha256={baseline['input']['record_payload_sha256']}",
        f"repo={baseline['environment']['repo_commit']} status={baseline['environment']['repo_status']}",
        f"cfg={baseline['environment']['cfg']}",
        f"counts={baseline['counts']} agent_gate={baseline['agent_gate']}",
    ]
    for lane in lanes:
        children = [(child["child_id"], child["tensor_row"],
                     child["stop_source_status"]) for child in lane["partitioned"]]
        log.append(
            f"lane={lane['raw_lane_id']} signal={lane['signal_state']} "
            f"projection={lane['original_lane_projection']['status']} "
            f"fov_stop_source={lane['stop_fov_source_audit']['status']} "
            f"regular=({lane['regular']['tensor_row']},"
            f"{lane['regular']['stop_source_status']}) children={children}")
    return {
        "run_status": "pass",
        "scope": "fixed_t10_pre_shuffle_road_geometry_only",
        "environment": baseline["environment"],
        "input": baseline["input"],
        "counts": baseline["counts"],
        "agent_gate": baseline["agent_gate"],
        "target_signal_count": len(lanes),
        "lanes": lanes,
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
