"""Read-only geometry provenance helpers for one lane-graph instance.

All ranges are zero-based and half-open. The caller must use the actual
official compact graph, FoV mask, and ordered graph from the same scene run.
This module never claims that interpolated points have exact raw-Lane owners.
"""

from numbers import Integral

import numpy as np


def _runs_from_indices(indices):
    runs = []
    for index in indices:
        if runs and index == runs[-1][1]:
            runs[-1] = (runs[-1][0], index + 1)
        else:
            runs.append((index, index + 1))
    return tuple(runs)


def mask_to_runs(mask):
    """Return kept point indices and their ordered, half-open index runs."""
    values = np.asarray(mask)
    if values.ndim != 1 or values.dtype != np.dtype(bool):
        raise ValueError("FoV mask must be a one-dimensional boolean array")
    indices = tuple(int(index) for index in np.flatnonzero(values))
    return indices, _runs_from_indices(indices)


def audit_lane_fov(compact_points, mask, compact_point_range,
                   source_point_count=None):
    """Audit point survival for a raw Lane's contributed compact interval.

    ``compact_point_range`` is build_lane_lineage's compact_point_range, not
    compact_span_range: a preceding Lane's connection marker is not evidence
    that this raw Lane's own first point survived. Kept local offsets can be
    translated back to raw source indices by adding source_point_range[0].
    ``uncertain`` flags provenance ambiguity introduced by the official crop
    then interpolation; it is independent of partial point retention.
    """
    points = np.asarray(compact_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("compact_points must have shape (N, 2)")
    kept_group, group_runs = mask_to_runs(mask)
    if len(mask) != len(points):
        raise ValueError("FoV mask length must match compact point count")
    if (not isinstance(compact_point_range, (tuple, list))
            or len(compact_point_range) != 2
            or any(isinstance(i, bool) or not isinstance(i, Integral)
                   for i in compact_point_range)):
        raise ValueError("compact_point_range must be a pair of indices")
    start, end = (int(i) for i in compact_point_range)
    if start < 0 or end < start or end > len(points):
        raise ValueError("compact_point_range is outside compact_points")
    if (source_point_count is not None
            and (isinstance(source_point_count, bool)
                 or not isinstance(source_point_count, Integral)
                 or source_point_count < 0)):
        raise ValueError("source_point_count must be a nonnegative integer")

    kept_local_offsets = tuple(index - start for index in kept_group
                               if start <= index < end)
    local_runs = _runs_from_indices(kept_local_offsets)
    local_count = end - start
    reasons = []
    if local_count == 0:
        reasons.append("empty_compact_contribution")
    if source_point_count is not None and source_point_count < 2:
        reasons.append("source_lane_has_fewer_than_two_points")
    if not np.isfinite(points[start:end]).all():
        reasons.append("nonfinite_source_geometry")

    invalid = bool(reasons)
    if invalid:
        point_status = "invalid"
    elif not kept_local_offsets:
        point_status = "filtered"
    elif len(kept_local_offsets) < local_count:
        point_status = "partial"
    else:
        point_status = "retained"

    uncertain = False
    if point_status in ("partial", "retained"):
        if len(group_runs) > 1:
            uncertain = True
            reasons.append("discontinuous_fov_points")
        cropped = points[list(kept_group)]
        if len(cropped) < 2:
            uncertain = True
            reasons.append("fewer_than_two_kept_group_points")
        elif np.any(np.all(np.diff(cropped, axis=0) == 0, axis=1)):
            uncertain = True
            reasons.append("repeated_kept_points")
        if len(kept_local_offsets) < 2:
            uncertain = True
            reasons.append("fewer_than_two_kept_local_points")

    status = "uncertain" if point_status == "retained" and uncertain else point_status
    return {
        "status": status,
        "point_status": point_status,
        "uncertain": uncertain,
        "reasons": tuple(reasons),
        "kept_group_indices": kept_group,
        "group_runs": group_runs,
        "kept_local_point_offsets": kept_local_offsets,
        "kept_local_offset_runs": local_runs,
        "local_point_count": local_count,
        "kept_local_point_count": len(kept_local_offsets),
    }


def plan_partition_lineage(lanes):
    """Mirror the official partition index/ID allocation without mutation.

    ``lanes`` must preserve the ordered IDs of the official FoV output.
    Children are ``(id, (start, end))`` slices of each FoV-resampled parent.
    ``unused_ids`` exposes the official one-segment allocation-and-skip case.
    Verify each planned child array against the official output before using
    the plan to claim provenance for a real scenario.
    """
    if not lanes:
        raise ValueError("Partition input has no lanes")
    ids = list(lanes)
    if any(isinstance(i, bool) or not isinstance(i, Integral) for i in ids):
        raise ValueError("Partition Lane IDs must be integers")
    next_id = max(ids) + 1
    plan = {}
    for lane_id in ids:
        lane = np.asarray(lanes[lane_id], dtype=float)
        if lane.ndim != 2 or lane.shape[1] != 2 or len(lane) == 0:
            raise ValueError(f"Partition Lane {lane_id} must have shape (N, 2), N >= 1")
        if not np.isfinite(lane).all():
            raise ValueError(f"Partition Lane {lane_id} has nonfinite coordinates")
        crossings = tuple(int(i) for i in
                          np.flatnonzero(np.signbit(lane[1:, 1])
                                         != np.signbit(lane[:-1, 1])) + 1)
        generated = []
        start = 0
        for crossing in crossings:
            generated.append((next_id, (start, crossing + 1)))
            start = crossing
            next_id += 1
        if crossings and crossings[-1] < len(lane) - 1:
            generated.append((next_id, (start, len(lane))))
            next_id += 1
        if len(generated) <= 1:
            children = ((int(lane_id), (0, len(lane))),)
            unused_ids = tuple(child_id for child_id, _ in generated)
            shared_endpoints = ()
        else:
            children = tuple(generated)
            unused_ids = ()
            shared_endpoints = tuple(point_range[0]
                                     for _, point_range in children[1:])
        plan[int(lane_id)] = {
            "crossings": crossings,
            "children": children,
            "shared_endpoints": shared_endpoints,
            "unused_ids": unused_ids,
        }
    return plan
