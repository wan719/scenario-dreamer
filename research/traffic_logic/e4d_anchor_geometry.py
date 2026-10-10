"""Independent global-XY audit of raw Stop Points and Compact joins.

``groups`` and ``compact_graph`` must come from the same ``build_graphs`` call.
The audit does not transform points to ego coordinates or alter road geometry.
"""

import hashlib
import math
import struct

import numpy as np

from .lane_lineage import _lane_id, build_lane_lineage
from .real_scenario_helpers import _fields


def _distance(first, second):
    return float(np.linalg.norm(np.asarray(first)[:2] - np.asarray(second)[:2]))


def _raw_endpoint(raw_lanes, lane_id, position):
    polyline = raw_lanes[lane_id]["polyline"]
    point = polyline[position]
    return [float(point["x"]), float(point["y"])]


def parse_stop_points(payload, timestep, target_ids):
    """Read target LaneState stop points directly from one Scenario payload."""
    if timestep < 0:
        raise ValueError("timestep must be nonnegative")
    targets = {_lane_id(lane_id) for lane_id in target_ids}
    dynamic_states = [value for number, wire, value in _fields(payload)
                      if number == 7 and wire == 2]
    if timestep >= len(dynamic_states):
        raise ValueError(f"Dynamic map timestep {timestep} is missing")
    result = {}
    for number, wire, lane_state in _fields(dynamic_states[timestep]):
        if number != 1 or wire != 2:
            continue
        fields = list(_fields(lane_state))
        ids = [value for field, field_wire, value in fields
               if field == 1 and field_wire == 0]
        if len(ids) != 1 or ids[0] not in targets:
            continue
        lane_id = ids[0]
        if lane_id in result:
            raise ValueError(f"Duplicate LaneState for Lane {lane_id}")
        states = [value for field, field_wire, value in fields
                  if field == 2 and field_wire == 0]
        points = [value for field, field_wire, value in fields
                  if field == 3 and field_wire == 2]
        if len(states) != 1 or len(points) != 1:
            raise ValueError(f"Lane {lane_id} lacks one state or Stop Point")
        coordinates = {}
        for field, field_wire, value in _fields(points[0]):
            if field in (1, 2, 3) and field_wire == 1:
                if field in coordinates:
                    raise ValueError(f"Lane {lane_id} has duplicate Stop Point coordinate")
                coordinates[field] = struct.unpack("<d", value)[0]
        if set(coordinates) != {1, 2, 3} or not all(
                math.isfinite(value) for value in coordinates.values()):
            raise ValueError(f"Lane {lane_id} has invalid Stop Point coordinates")
        result[lane_id] = {
            "state_code": int(states[0]),
            "stop_point_global_xyz": tuple(coordinates[index]
                                           for index in (1, 2, 3)),
        }
    missing = targets - result.keys()
    if missing:
        raise ValueError(f"Missing Stop Point LaneState for {sorted(missing)}")
    return result


def audit_anchor_global(raw_lanes, lane_graph, compact_graph, groups,
                        lane_id, stop_point_global_xyz, *, tolerance_m=1e-9):
    """Recompute one join from raw endpoints and scan its whole Compact Group.

    The selected index comes from strict lineage of just the target's complete
    Group, since another Group (e.g. Lane 529) may contain a one-point Lane.
    Full-Group duplicate coordinates are retained as separate candidates.
    """
    lane_id = _lane_id(lane_id)
    stop = tuple(float(value) for value in stop_point_global_xyz)
    if len(stop) != 3 or not all(math.isfinite(value) for value in stop):
        raise ValueError("stop_point_global_xyz must be finite XYZ")
    tolerance = float(tolerance_m)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance_m must be finite and nonnegative")
    result = {
        "raw_lane_id": lane_id,
        "coordinate_frame": "global_xy",
        "xy_tolerance_m": tolerance,
        "stop_point_global_xyz": list(stop),
        "group_id": None,
        "group_members": None,
        "predecessor_lane_id": None,
        "shared_compact_vertex_index": None,
        "raw_downstream_first_xy": None,
        "raw_predecessor_last_xy": None,
        "compact_shared_vertex_global_xy": None,
        "distances_m": {},
        "topology_edges": {},
        "topology_verified": False,
        "selected_vertex_consistent": False,
        "geometric_candidate_scan_performed": False,
        "geometric_candidate_vertex_indices": None,
        "geometric_candidate_distances_m": None,
        "geometric_candidates_count": None,
        "unique_anchor_verified": False,
        "compact_point_count": None,
        "compact_geometry_sha256": None,
        "reasons": [],
    }
    reasons = result["reasons"]
    memberships = [(int(group_id), [_lane_id(member) for member in members])
                   for group_id, members in groups.items()
                   if lane_id in {_lane_id(member) for member in members}]
    if len(memberships) != 1:
        reasons.append("lane_not_in_one_group")
        return result
    group_id, members = memberships[0]
    result["group_id"] = group_id
    result["group_members"] = members
    if members.count(lane_id) != 1 or members.index(lane_id) == 0:
        reasons.append("missing_unique_compact_predecessor")
        return result
    predecessor = members[members.index(lane_id) - 1]
    result["predecessor_lane_id"] = predecessor
    if (lane_id not in raw_lanes or predecessor not in raw_lanes or
            lane_id not in lane_graph["lanes"] or
            predecessor not in lane_graph["lanes"] or
            group_id not in compact_graph["lanes"]):
        reasons.append("source_lane_or_group_missing")
        return result
    if (len(raw_lanes[lane_id]["polyline"]) < 2 or
            len(raw_lanes[predecessor]["polyline"]) < 2):
        reasons.append("degenerate_raw_lane")
        return result
    try:
        lineage = build_lane_lineage(
            {group_id: members},
            {member: lane_graph["lanes"][member] for member in members})
    except (ValueError, KeyError) as exc:
        reasons.append(f"nonrepresentable_group:{exc}")
        return result
    vertex_index = lineage[lane_id]["compact_point_range"][0] - 1
    result["shared_compact_vertex_index"] = vertex_index
    compact = np.asarray(compact_graph["lanes"][group_id], dtype=np.float64)
    if compact.ndim != 2 or compact.shape[1] != 2 or vertex_index >= len(compact):
        reasons.append("invalid_compact_geometry")
        return result
    result["compact_point_count"] = len(compact)
    result["compact_geometry_sha256"] = hashlib.sha256(
        np.asarray(compact, dtype="<f8").tobytes()).hexdigest()
    downstream_first = _raw_endpoint(raw_lanes, lane_id, 0)
    predecessor_last = _raw_endpoint(raw_lanes, predecessor, -1)
    selected = compact[vertex_index].tolist()
    result["raw_downstream_first_xy"] = downstream_first
    result["raw_predecessor_last_xy"] = predecessor_last
    result["compact_shared_vertex_global_xy"] = selected
    processed_downstream_first = lane_graph["lanes"][lane_id][0].tolist()
    processed_predecessor_last = lane_graph["lanes"][predecessor][-1].tolist()
    distances = {
        "stop_to_downstream_first": _distance(stop, downstream_first),
        "stop_to_predecessor_last": _distance(stop, predecessor_last),
        "stop_to_selected_compact": _distance(stop, selected),
        "predecessor_to_downstream": _distance(predecessor_last,
                                               downstream_first),
        "downstream_to_selected_compact": _distance(downstream_first,
                                                    selected),
        "raw_to_processed_downstream_first": _distance(
            downstream_first, processed_downstream_first),
        "raw_to_processed_predecessor_last": _distance(
            predecessor_last, processed_predecessor_last),
    }
    result["distances_m"] = distances
    selected_consistent = all(value <= tolerance for value in distances.values())
    result["selected_vertex_consistent"] = selected_consistent
    if not selected_consistent:
        reasons.append("selected_vertex_xy_mismatch")

    raw_entry = [_lane_id(value) for value in
                 raw_lanes[lane_id].get("entryLanes", [])]
    raw_exit = [_lane_id(value) for value in
                raw_lanes[predecessor].get("exitLanes", [])]
    processed_pre = [_lane_id(value) for value in
                     lane_graph["pre_pairs"].get(lane_id, [])]
    processed_suc = [_lane_id(value) for value in
                     lane_graph["suc_pairs"].get(predecessor, [])]
    result["topology_edges"] = {
        "raw_downstream_entry_lanes": raw_entry,
        "raw_predecessor_exit_lanes": raw_exit,
        "processed_downstream_predecessors": processed_pre,
        "processed_predecessor_successors": processed_suc,
    }
    if raw_entry != [predecessor]:
        reasons.append("raw_entry_mismatch")
    if raw_exit != [lane_id]:
        reasons.append("raw_exit_mismatch")
    if processed_pre != [predecessor]:
        reasons.append("processed_predecessor_mismatch")
    if processed_suc != [lane_id]:
        reasons.append("processed_successor_mismatch")
    result["topology_verified"] = bool(
        raw_entry == [predecessor] and raw_exit == [lane_id] and
        processed_pre == [predecessor] and processed_suc == [lane_id])

    all_distances = np.linalg.norm(compact - np.asarray(stop[:2]), axis=1)
    candidates = np.flatnonzero(all_distances <= tolerance).tolist()
    result["geometric_candidate_scan_performed"] = True
    result["geometric_candidate_vertex_indices"] = candidates
    result["geometric_candidate_distances_m"] = [
        float(all_distances[index]) for index in candidates]
    result["geometric_candidates_count"] = len(candidates)
    if len(candidates) > 1:
        reasons.append("multiple_same_xy_vertices")
    elif not candidates:
        reasons.append("no_same_xy_vertex")
    result["unique_anchor_verified"] = bool(
        selected_consistent and result["topology_verified"] and
        candidates == [vertex_index])
    return result
