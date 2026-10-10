"""Audit an explicit Stop Anchor at a Compact Lane member boundary.

Input is one E4-B lane record produced by the same E4-A processing instance.
Its recorded distances are measurements of live source geometry, not a fresh
reconstruction of the source polylines. This module changes no road geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

from .lane_lineage import _lane_id


@dataclass(frozen=True)
class StopAnchor:
    raw_lane_id: int
    signal_state: str
    stop_point: tuple
    group_id: int
    predecessor_lane_id: int | None
    shared_compact_vertex_index: int | None
    topology_verified: bool
    geometry_verified: bool
    shared_vertex_verified: bool
    shared_vertex_fov_retained: bool | None
    verification_status: str
    final_tensor_stop_provenance_verified: bool | None
    final_tensor_stop_provenance_status: str
    regular_tensor_row: int | None
    partition_stop_side_rows: tuple
    connection_xy_upper_bound_m: float | None
    xy_tolerance_m: float
    geometric_candidate_scan_performed: bool
    geometric_candidate_vertex_indices: tuple | None
    reasons: tuple


def _index(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _range(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a half-open pair")
    start, end = (_index(item, name) for item in value)
    if end < start:
        raise ValueError(f"{name} has reversed bounds")
    return start, end


def _distance(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite nonnegative distance")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative distance")
    return result


def _stop_point(value):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("stop_point_global_xyz must have three coordinates")
    point = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in point):
        raise ValueError("stop_point_global_xyz must be finite")
    return point


def _xy(value, name):
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        raise ValueError(f"{name} must contain XY coordinates")
    point = tuple(float(item) for item in value[:2])
    if not all(math.isfinite(item) for item in point):
        raise ValueError(f"{name} must have finite XY coordinates")
    return point


def _row(value):
    return None if value is None else _index(value, "tensor_row")


def audit_e4b_stop_anchor(case, *, xy_tolerance_m=1e-9,
                          candidate_shared_vertex_indices=None,
                          compact_group_points=None):
    """Verify one recorded Compact join and retain downstream uncertainty.

    The XY bound is d(stop, downstream first) + d(stop, predecessor last),
    which bounds the predecessor-to-downstream connection by the triangle
    inequality. E4-B separately measured stop-to-Compact-vertex in ego XY.
    A tensor row or E4-B ``attributable`` status cannot alone prove the
    *raw Lane identity* of a first point omitted by Compact concatenation.

    Supply ``compact_group_points`` to scan all same-XY vertices directly;
    otherwise optional ``candidate_shared_vertex_indices`` can carry a scan
    done elsewhere. With neither, global candidate uniqueness is untested.
    """
    tolerance = _distance(xy_tolerance_m, "xy_tolerance_m")
    raw_lane_id = _lane_id(case["raw_lane_id"])
    group_id = _lane_id(case["group_id"])
    signal_state = case["signal_state"]
    if not isinstance(signal_state, str) or not signal_state:
        raise ValueError("signal_state must be a nonempty string")
    stop = _stop_point(case["stop_point_global_xyz"])
    raw_count = _index(case["raw_lane_point_count"], "raw_lane_point_count")
    members = tuple(_lane_id(member) for member in case["group_members"])
    member_index = _index(case["member_index"], "member_index")
    source_start, source_end = _range(case["source_point_range"],
                                      "source_point_range")
    compact_start, compact_end = _range(case["compact_point_range"],
                                        "compact_point_range")
    boundary = case["shared_boundary"]
    predecessor = boundary.get("predecessor_lane_id")
    predecessor = _lane_id(predecessor) if predecessor is not None else None
    vertex = boundary.get("compact_vertex_index")
    vertex = _index(vertex, "compact_vertex_index") if vertex is not None else None

    invalid = []
    uncertain = []
    filtered = []
    downstream = []
    if not members or len(set(members)) != len(members):
        invalid.append("duplicate_group_member" if members else "empty_group")
    if member_index >= len(members) or members[member_index] != raw_lane_id:
        invalid.append("member_index_mismatch")
    if member_index == 0 or predecessor is None or member_index >= len(members):
        invalid.append("missing_compact_predecessor")
    elif members[member_index - 1] != predecessor:
        invalid.append("predecessor_not_adjacent")
    if source_start != 1 or source_end != raw_count:
        invalid.append("source_range_not_omitted_first_point")
    if compact_end - compact_start != source_end - source_start:
        invalid.append("compact_contribution_length_mismatch")
    if compact_start == 0 or vertex != compact_start - 1:
        invalid.append("shared_vertex_index_mismatch")
    topology_verified = not invalid

    if raw_count < 2:
        invalid.append("degenerate_raw_lane")
    projection = case["original_lane_projection"]
    total_arc = _distance(projection["total_arc_length_m"],
                          "projection.total_arc_length_m")
    arc = _distance(projection["arc_length_m"],
                    "projection.arc_length_m")
    fraction = _distance(projection["segment_fraction"],
                         "projection.segment_fraction")
    segment_index = _index(projection["nearest_segment_index"],
                           "projection.nearest_segment_index")
    if total_arc <= 0:
        invalid.append("zero_total_raw_lane_arclength")
    else:
        position_ratio = _distance(projection["position_ratio"],
                                   "projection.position_ratio")
        if position_ratio > 1e-8:
            invalid.append("stop_not_at_raw_lane_first_point")
    candidates = tuple(_index(item, "projection.candidate_segment_index")
                       for item in projection["candidate_segment_indices"])
    if len(candidates) != 1 or projection["status"] != "retained":
        uncertain.append("nonunique_stop_projection")
    if segment_index != 0 or fraction > 1e-8 or arc > tolerance:
        invalid.append("stop_not_at_raw_lane_first_point")
    if _distance(projection["distance_m"], "projection.distance_m") > tolerance:
        invalid.append("stop_off_raw_lane_centerline")

    lane_first_distance = _distance(
        case["stop_to_lane_first_xy_distance_m"],
        "stop_to_lane_first_xy_distance_m")
    predecessor_distance = _distance(
        boundary["predecessor_end_distance_m"],
        "predecessor_end_distance_m")
    compact_distance = _distance(
        boundary["stop_to_compact_boundary_ego_distance_m"],
        "stop_to_compact_boundary_ego_distance_m")
    connection_bound = lane_first_distance + predecessor_distance
    if connection_bound > tolerance:
        invalid.append("predecessor_downstream_xy_mismatch")
    if compact_distance > tolerance:
        invalid.append("shared_compact_vertex_xy_mismatch")
    if compact_group_points is not None and candidate_shared_vertex_indices is not None:
        raise ValueError("Supply Compact points or candidate indices, not both")
    scanned = (compact_group_points is not None or
               candidate_shared_vertex_indices is not None)
    if scanned:
        if compact_group_points is not None:
            stop_ego = _xy(case["stop_point_ego_xy"], "stop_point_ego_xy")
            points = tuple(_xy(point, "compact_group_point")
                           for point in compact_group_points)
            if vertex is None or vertex >= len(points):
                invalid.append("shared_vertex_out_of_bounds")
            elif abs(math.dist(stop_ego, points[vertex]) - compact_distance) > tolerance:
                invalid.append("recorded_compact_distance_mismatch")
            vertex_candidates = tuple(
                index for index, point in enumerate(points)
                if math.dist(stop_ego, point) <= tolerance)
        else:
            vertex_candidates = tuple(
                _index(item, "candidate_shared_vertex_index")
                for item in candidate_shared_vertex_indices)
        if len(set(vertex_candidates)) != len(vertex_candidates):
            invalid.append("duplicate_shared_vertex_candidate")
        elif len(vertex_candidates) != 1:
            uncertain.append("multiple_shared_vertex_candidates"
                             if vertex_candidates else "no_shared_vertex_candidate")
        elif vertex_candidates[0] != vertex:
            invalid.append("shared_vertex_candidate_mismatch")
    else:
        vertex_candidates = None

    geometry_verified = not any(reason in invalid for reason in (
        "degenerate_raw_lane", "zero_total_raw_lane_arclength",
        "stop_not_at_raw_lane_first_point", "stop_off_raw_lane_centerline",
        "predecessor_downstream_xy_mismatch",
        "shared_compact_vertex_xy_mismatch", "shared_vertex_out_of_bounds",
        "recorded_compact_distance_mismatch"))

    shared_verified = bool(topology_verified and geometry_verified and
                           not invalid and not uncertain)
    stop_in_fov = case["stop_point_inside_fov"]
    vertex_kept = boundary["compact_vertex_kept_by_fov"]
    if not isinstance(stop_in_fov, bool) or not isinstance(vertex_kept, bool):
        raise ValueError("FoV flags must be boolean")
    fov_retained = bool(stop_in_fov and vertex_kept)
    if not stop_in_fov:
        filtered.append("stop_point_outside_fov")
    if not vertex_kept:
        filtered.append("shared_vertex_filtered_by_fov")

    allowed_source_statuses = {"filtered", "uncertain", "attributable", "invalid"}
    if case["regular"]["stop_source_status"] not in allowed_source_statuses:
        raise ValueError("Unknown regular stop_source_status")
    regular_row = _row(case["regular"]["tensor_row"])
    partition_rows = []
    for child in case["partitioned"]:
        child_row = _row(child["tensor_row"])
        child_status = child["stop_source_status"]
        if child_status not in allowed_source_statuses:
            raise ValueError("Unknown partition stop_source_status")
        if child["partition_shared_endpoint_candidates"]:
            downstream.append("partition_shared_endpoint")
        if child_status == "invalid":
            downstream.append("invalid_partition_source_audit")
        if child_status in ("uncertain", "attributable") and child_row is not None:
            partition_rows.append((_lane_id(child["child_id"]), child_row))
    if len(partition_rows) > 1:
        downstream.append("multiple_partition_stop_sides")

    if invalid:
        status = "invalid"
    elif uncertain:
        status = "uncertain"
    elif filtered:
        status = "filtered"
    else:
        status = "verified"
    if invalid:
        final_status = "invalid"
        final_verified = False
    elif filtered or (regular_row is None and not partition_rows):
        final_status = "filtered"
        final_verified = False
    else:
        final_status = "uncertain"
        final_verified = None
    reasons = tuple(dict.fromkeys((*invalid, *uncertain, *filtered, *downstream)))
    return StopAnchor(
        raw_lane_id=raw_lane_id,
        signal_state=signal_state,
        stop_point=stop,
        group_id=group_id,
        predecessor_lane_id=predecessor,
        shared_compact_vertex_index=vertex,
        topology_verified=topology_verified,
        geometry_verified=geometry_verified,
        shared_vertex_verified=shared_verified,
        shared_vertex_fov_retained=fov_retained,
        verification_status=status,
        final_tensor_stop_provenance_verified=final_verified,
        final_tensor_stop_provenance_status=final_status,
        regular_tensor_row=regular_row,
        partition_stop_side_rows=tuple(partition_rows),
        connection_xy_upper_bound_m=connection_bound,
        xy_tolerance_m=tolerance,
        geometric_candidate_scan_performed=scanned,
        geometric_candidate_vertex_indices=vertex_candidates,
        reasons=reasons,
    )
