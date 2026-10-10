"""Conservative, read-only Stop Point geometry and source audits.

All source ranges are zero-based and half-open. A tensor row means that a
Group/child exists; it does not establish that the Stop Point survived.
``parent_support`` and ``final_support`` use E4-A's four-field source
brackets: (left_index, right_index, crop_gap, zero_length_segment).
"""

from numbers import Integral

import numpy as np


def _xy_points(points, name, minimum):
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] < 2 or len(array) < minimum:
        raise ValueError(f"{name} must have shape (N, 2+) with N >= {minimum}")
    xy = array[:, :2]
    if not np.isfinite(xy).all():
        raise ValueError(f"{name} has nonfinite XY coordinates")
    return xy


def _index_range(value, name, upper=None):
    if (not isinstance(value, (tuple, list)) or len(value) != 2 or
            any(isinstance(x, bool) or not isinstance(x, Integral) for x in value)):
        raise ValueError(f"{name} must be a half-open integer pair")
    start, end = (int(x) for x in value)
    if start < 0 or end < start or (upper is not None and end > upper):
        raise ValueError(f"{name} is out of bounds")
    return start, end


def project_stop_point(lane_points, stop_point, *, tie_tolerance=1e-6,
                       vertex_tolerance=1e-8,
                       on_centerline_tolerance_m=1e-6):
    """Project a true Stop Point onto the original Lane XY polyline.

    Equal-distance segments remain explicit candidates. Distance beyond
    ``on_centerline_tolerance_m`` is geometry-only evidence: no engineering
    association radius is inferred from a nearest-centerline projection.
    Use ``None`` when measuring distance to a resampled polyline; its
    nonzero interpolation error must not downgrade the original projection.
    """
    lane = _xy_points(lane_points, "lane_points", 2)
    stop = np.asarray(stop_point, dtype=float)
    if stop.ndim != 1 or len(stop) < 2 or not np.isfinite(stop[:2]).all():
        raise ValueError("stop_point must have at least two finite coordinates")
    if (not np.isfinite(tie_tolerance) or tie_tolerance < 0 or
            not np.isfinite(vertex_tolerance) or vertex_tolerance < 0 or
            (on_centerline_tolerance_m is not None and
             (not np.isfinite(on_centerline_tolerance_m) or
              on_centerline_tolerance_m < 0))):
        raise ValueError("projection tolerances must be finite and nonnegative")
    stop_xy = stop[:2]
    vectors = np.diff(lane, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    cumulative = np.insert(np.cumsum(lengths), 0, 0.0)
    candidates = []
    for index, (origin, vector, length) in enumerate(
            zip(lane[:-1], vectors, lengths)):
        raw_fraction = (float(np.dot(stop_xy - origin, vector) / length**2)
                        if length > 0 else 0.0)
        fraction = float(np.clip(raw_fraction, 0.0, 1.0))
        projected = origin + fraction * vector
        candidates.append({
            "segment_index": index,
            "segment_fraction": fraction,
            "unclamped_segment_fraction": raw_fraction,
            "projected_xy": tuple(float(x) for x in projected),
            "distance_m": float(np.linalg.norm(stop_xy - projected)),
            "arc_length_m": float(cumulative[index] + fraction * length),
            "zero_length_segment": bool(length == 0),
        })
    minimum = min(candidate["distance_m"] for candidate in candidates)
    nearest = tuple(candidate for candidate in candidates
                    if candidate["distance_m"] <= minimum + tie_tolerance)
    selected = nearest[0]
    reasons = []
    if cumulative[-1] == 0:
        reasons.append("zero_total_lane_arclength")
    if len(nearest) > 1:
        reasons.append("multiple_nearest_segments")
    if any(candidate["zero_length_segment"] for candidate in nearest):
        reasons.append("zero_length_nearest_segment")
    if any(candidate["unclamped_segment_fraction"] < -vertex_tolerance or
           candidate["unclamped_segment_fraction"] > 1 + vertex_tolerance
           for candidate in nearest if not candidate["zero_length_segment"]):
        reasons.append("projection_clamped_beyond_segment")
    if (on_centerline_tolerance_m is not None and
            minimum > on_centerline_tolerance_m):
        reasons.append("stop_off_original_centerline")
    status = ("invalid" if cumulative[-1] == 0 else
              "uncertain" if reasons else "retained")
    return {
        "status": status,
        "reasons": tuple(reasons),
        "stop_xy": tuple(float(x) for x in stop_xy),
        "source_point_count": len(lane),
        "nearest_segment_index": selected["segment_index"],
        "candidate_segment_indices": tuple(
            candidate["segment_index"] for candidate in nearest),
        "candidate_projections": nearest,
        "segment_fraction": selected["segment_fraction"],
        "unclamped_segment_fraction": selected["unclamped_segment_fraction"],
        "projected_xy": selected["projected_xy"],
        "distance_m": selected["distance_m"],
        "arc_length_m": selected["arc_length_m"],
        "total_arc_length_m": float(cumulative[-1]),
        "position_ratio": (float(selected["arc_length_m"] / cumulative[-1])
                           if cumulative[-1] > 0 else None),
        "vertex_tolerance": float(vertex_tolerance),
        "on_centerline_tolerance_m": (
            float(on_centerline_tolerance_m)
            if on_centerline_tolerance_m is not None else None),
    }


def audit_stop_fov(projection, source_point_range, compact_point_range,
                   kept_compact_indices, stop_in_fov):
    """Check the Stop Point's own raw edge against the actual point crop.

    ``source_point_range`` and ``compact_point_range`` come from the same
    E4-A ``build_lane_lineage`` call. The omitted first point of a merged
    Lane is never borrowed from its preceding Group member.
    """
    source_start, source_end = _index_range(
        source_point_range, "source_point_range", projection["source_point_count"])
    compact_start, compact_end = _index_range(
        compact_point_range, "compact_point_range")
    if source_end - source_start != compact_end - compact_start:
        raise ValueError("source and compact contribution lengths disagree")
    if not isinstance(stop_in_fov, (bool, np.bool_)):
        raise ValueError("stop_in_fov must be boolean")
    kept = set()
    for value in kept_compact_indices:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError("kept_compact_indices must contain nonnegative integers")
        kept.add(int(value))
    segment = projection["nearest_segment_index"]
    fraction = projection["segment_fraction"]
    tolerance = projection["vertex_tolerance"]
    if fraction <= tolerance:
        required_source = (segment,)
    elif 1.0 - fraction <= tolerance:
        required_source = (segment + 1,)
    else:
        required_source = (segment, segment + 1)
    required_compact = tuple(compact_start + index - source_start
                             for index in required_source
                             if source_start <= index < source_end)
    omitted = tuple(index for index in required_source
                    if index < source_start or index >= source_end)
    kept_required = tuple(index for index in required_compact if index in kept)
    candidate_compact_positions = tuple(
        float(compact_start + candidate["segment_index"] +
              candidate["segment_fraction"] - source_start)
        for candidate in projection["candidate_projections"])
    retained_vertex_edges = ()
    if len(required_source) == 1:
        vertex = required_source[0]
        adjacent = (vertex - 1, vertex + 1)
        retained_vertex_edges = tuple(
            (min(vertex, neighbor), max(vertex, neighbor))
            for neighbor in adjacent
            if (0 <= neighbor < projection["source_point_count"] and
                source_start <= vertex < source_end and
                source_start <= neighbor < source_end and
                compact_start + vertex - source_start in kept and
                compact_start + neighbor - source_start in kept)
        )
    compact_position = compact_start + segment + fraction - source_start
    reasons = list(projection["reasons"])
    if projection["status"] == "invalid":
        status = "invalid"
    elif not stop_in_fov:
        status = "filtered"
        reasons.append("stop_point_outside_fov")
    elif len(candidate_compact_positions) > 1:
        status = "uncertain"
    elif omitted:
        status = "uncertain"
        reasons.append("source_point_omitted_by_compact")
    elif not kept_required:
        status = "filtered"
        reasons.append("original_stop_edge_filtered_by_fov")
    elif len(kept_required) < len(required_compact):
        status = "uncertain"
        reasons.append("source_edge_broken_by_fov_crop")
    elif len(required_source) == 1 and not retained_vertex_edges:
        status = "uncertain"
        reasons.append("no_continuous_original_edge_at_stop")
    elif projection["status"] == "uncertain":
        status = "uncertain"
    else:
        status = "retained"
    return {
        "status": status,
        "reasons": tuple(reasons),
        "stop_in_fov": bool(stop_in_fov),
        "compact_position": float(compact_position),
        "candidate_compact_positions": candidate_compact_positions,
        "required_source_indices": required_source,
        "required_compact_indices": required_compact,
        "omitted_source_indices": omitted,
        "kept_required_compact_indices": kept_required,
        "retained_vertex_source_edges": retained_vertex_edges,
    }


def _support_bracket(value, upper, name):
    if (len(value) != 4 or
            any(isinstance(index, bool) or not isinstance(index, Integral)
                for index in value[:2]) or
            any(not isinstance(flag, (bool, np.bool_)) for flag in value[2:])):
        raise ValueError(f"{name} must be (left, right, crop_gap, zero_length)")
    left, right = int(value[0]), int(value[1])
    if left < 0 or right < left or (upper is not None and right >= upper):
        raise ValueError(f"{name} source indices are out of bounds")
    return left, right, bool(value[2]), bool(value[3])


def audit_stop_resampled_support(fov_audit, parent_support, final_support,
                                 tensor_row, *, child_range=None,
                                 shared_endpoints=(), tolerance=1e-8):
    """Bound a Stop Point between adjacent 20-point sample source ranges.

    Parent support maps 1000-point samples to compact indices; final
    support maps 20-point samples to a regular parent or Partition child.
    Proven adjacent samples must lie on opposite sides of the projected
    compact position. Overlapping source ranges, crop gaps, and Partition
    shared endpoints remain uncertain, even if a tensor row exists.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    parent_count = len(parent_support)
    child_start, child_end = (_index_range(child_range, "child_range", parent_count)
                              if child_range is not None else (0, parent_count))
    if child_start == child_end:
        raise ValueError("child_range cannot be empty")
    normalized_parent = tuple(
        _support_bracket(value, None, "parent_support")
        for value in parent_support)
    if tensor_row is not None and (
            isinstance(tensor_row, bool) or not isinstance(tensor_row, Integral)
            or tensor_row < 0):
        raise ValueError("tensor_row must be a nonnegative integer or None")
    shared = tuple(int(index) for index in shared_endpoints)
    if any(isinstance(index, bool) or not isinstance(index, Integral)
           or index < 0 or index >= parent_count for index in shared_endpoints):
        raise ValueError("shared_endpoints must index the 1000-point parent")
    position = float(fov_audit["compact_position"])
    positions = tuple(float(value) for value in
                      fov_audit.get("candidate_compact_positions", (position,)))
    if not positions or not all(np.isfinite(value) for value in positions):
        raise ValueError("candidate_compact_positions must be finite and nonempty")
    child_source_bounds = (
        min(item[0] for item in normalized_parent[child_start:child_end]),
        max(item[1] for item in normalized_parent[child_start:child_end]),
    )
    no_local_support = all(
        value < child_source_bounds[0] - tolerance or
        value > child_source_bounds[1] + tolerance
        for value in positions)
    evidence = []
    for sample_index, raw in enumerate(final_support):
        left, right, gap, zero = _support_bracket(
            raw, child_end - child_start, "final_support")
        parent_left = child_start + left
        parent_right = child_start + right
        endpoints = (normalized_parent[parent_left],
                     normalized_parent[parent_right])
        lower = min(endpoints[0][0], endpoints[1][0])
        upper = max(endpoints[0][1], endpoints[1][1])
        evidence.append({
            "sample_index": sample_index,
            "parent_point_indices": (parent_left, parent_right),
            "compact_source_bounds": (lower, upper),
            "crop_gap": bool(gap or endpoints[0][2] or endpoints[1][2]),
            "zero_length": bool(zero or endpoints[0][3] or endpoints[1][3]),
        })
    candidate_pairs = []
    proven_pairs = []
    gap_between_pairs = []
    zero_between_pairs = []
    for left, right in zip(evidence[:-1], evidence[1:]):
        a_lo, a_hi = left["compact_source_bounds"]
        b_lo, b_hi = right["compact_source_bounds"]
        if any(a_lo - tolerance <= value <= b_hi + tolerance
               for value in positions):
            pair = (left["sample_index"], right["sample_index"])
            candidate_pairs.append(pair)
            between = normalized_parent[
                min(*left["parent_point_indices"],
                    *right["parent_point_indices"]):
                max(*left["parent_point_indices"],
                    *right["parent_point_indices"]) + 1]
            pair_gap = any(item[2] for item in between)
            pair_zero = any(item[3] for item in between)
            if pair_gap:
                gap_between_pairs.append(pair)
            if pair_zero:
                zero_between_pairs.append(pair)
            if (any(a_hi + tolerance < value < b_lo - tolerance
                    for value in positions) and
                    not (left["crop_gap"] or right["crop_gap"] or
                         left["zero_length"] or right["zero_length"] or
                         pair_gap or pair_zero)):
                proven_pairs.append(pair)
    exact_samples = tuple(item["sample_index"] for item in evidence
                          if any(item["compact_source_bounds"] == (value, value)
                                 for value in positions)
                          and not item["crop_gap"] and not item["zero_length"])
    shared_candidates = tuple(index for index in shared
                              if any(normalized_parent[index][0] - tolerance <= value
                                     <= normalized_parent[index][1] + tolerance
                                     for value in positions))
    near_gap = any(item["crop_gap"] and
                   any(item["compact_source_bounds"][0] - tolerance <= value
                       <= item["compact_source_bounds"][1] + tolerance
                       for value in positions)
                   for item in evidence)
    near_zero = any(item["zero_length"] and
                    any(item["compact_source_bounds"][0] - tolerance <= value
                        <= item["compact_source_bounds"][1] + tolerance
                        for value in positions)
                    for item in evidence)
    reasons = list(fov_audit["reasons"])
    if shared_candidates:
        reasons.append("partition_shared_endpoint")
    if near_gap:
        reasons.append("fov_crop_gap_in_support")
    if gap_between_pairs:
        reasons.append("fov_crop_gap_between_final_samples")
    if near_zero:
        reasons.append("zero_length_resampling_support")
    if zero_between_pairs:
        reasons.append("zero_length_between_final_samples")
    if no_local_support:
        reasons.append("no_local_stop_support")
    if not proven_pairs and not exact_samples:
        reasons.append("no_proven_adjacent_final_samples")
    if fov_audit["status"] == "invalid":
        status = "invalid"
    elif tensor_row is None or fov_audit["status"] == "filtered" or no_local_support:
        status = "filtered"
        if tensor_row is None:
            reasons.append("no_tensor_row")
    elif (fov_audit["status"] == "retained" and
          (proven_pairs or exact_samples) and
          not (shared_candidates or near_gap or near_zero or
               gap_between_pairs or zero_between_pairs)):
        status = "attributable"
    else:
        status = "uncertain"
    return {
        "status": status,
        "reasons": tuple(dict.fromkeys(reasons)),
        "tensor_row": int(tensor_row) if tensor_row is not None else None,
        "compact_position": position,
        "candidate_compact_positions": positions,
        "child_compact_source_bounds": child_source_bounds,
        "child_range": (child_start, child_end),
        "candidate_adjacent_sample_pairs": tuple(candidate_pairs),
        "proven_adjacent_sample_pairs": tuple(proven_pairs),
        "exact_sample_indices": exact_samples,
        "shared_endpoint_candidates": shared_candidates,
        "crop_gap_between_candidate_pairs": tuple(gap_between_pairs),
        "zero_length_between_candidate_pairs": tuple(zero_between_pairs),
        "final_sample_source_bounds": tuple(evidence),
    }
