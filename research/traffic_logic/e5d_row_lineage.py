"""Conservative source brackets for research-side Lane resampling audits.

The coordinates come from the repository's unchanged resample_polyline body.
Brackets are explanatory evidence, not an inverse identity for np.interp.
"""

from pathlib import Path

import numpy as np

from .lane_lineage import _lane_id
from .real_scenario_helpers import _source_functions


_OFFICIAL = {"np": np}
_source_functions(
    Path(__file__).absolute().parents[2] / "utils/lane_graph_helpers.py",
    ("resample_polyline",), _OFFICIAL)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def trace_resampling(points, sources, count, *, shared_input_indices=(),
                     require_local_indices=False):
    """Replay official sampling and attach conservative per-point source sets.

    ``sources[i].source_index`` is the index before this stage, so a crop gap
    remains visible. Later stages receive this function's ``samples`` directly.
    All reported owner candidates are possibilities, never a control broadcast.
    """
    if not isinstance(require_local_indices, bool):
        raise ValueError("require_local_indices must be boolean")
    points = np.asarray(points, dtype=float)
    count = _integer(count, "count")
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or
            not np.isfinite(points).all() or count < 2 or
            len(sources) != len(points)):
        raise ValueError("Need at least two finite XY points, matching sources and count >= 2")
    source_indices = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Each source must be a dictionary")
        source_indices.append(_integer(source["source_index"], "source_index"))
        if not isinstance(source.get("raw_lane_candidates"), list):
            raise ValueError("raw_lane_candidates must be a list")
        if not isinstance(source.get("compact_source_indices"), list):
            raise ValueError("compact_source_indices must be a list")
        owners = {_lane_id(owner) for owner in source["raw_lane_candidates"]}
        if any(_integer(index, "compact_source_index") < 0
               for index in source["compact_source_indices"]):
            raise ValueError("compact_source_indices must be nonnegative")
        local = source.get("raw_lane_local_point_indices", {})
        if not isinstance(local, dict):
            raise ValueError("raw_lane_local_point_indices must be a dictionary")
        for raw_lane_id, indices in local.items():
            if _lane_id(raw_lane_id) not in owners:
                raise ValueError("Local point Lane ID is absent from source candidates")
            if not isinstance(indices, list) or any(
                    _integer(index, "raw local point index") < 0 for index in indices):
                raise ValueError("Raw local point indices must be nonnegative integers")
    if len(set(source_indices)) != len(source_indices) or any(
            right <= left for left, right in zip(source_indices, source_indices[1:])):
        raise ValueError("Source indices must be strictly increasing")
    shared = {_integer(value, "shared_input_index") for value in shared_input_indices}
    if any(value < 0 or value >= len(points) for value in shared):
        raise ValueError("shared_input_index is outside input points")

    sampled = _OFFICIAL["resample_polyline"](points, num_points=count)
    distances = np.sqrt(((points[1:] - points[:-1]) ** 2).sum(axis=1))
    cumulative = np.insert(np.cumsum(distances), 0, 0.0)
    targets = np.linspace(0, cumulative[-1], num=count)
    samples = []
    for output_index, target in enumerate(targets):
        # Exact vertices use every coincident cumulative-distance index. This
        # handles np.interp's duplicate-x ambiguity without guessing a winner.
        exact = np.flatnonzero(cumulative == target)
        if len(exact):
            support = [int(index) for index in exact]
        else:
            right = int(np.searchsorted(cumulative, target, side="right"))
            support = [right - 1, right]
        upstream = [sources[index] for index in support]
        owners = sorted({int(owner) for item in upstream
                         for owner in item["raw_lane_candidates"]})
        compact_indices = sorted({int(index) for item in upstream
                                  for index in item["compact_source_indices"]})
        local_indices = {}
        for item in upstream:
            for lane_id, indices in item.get("raw_lane_local_point_indices", {}).items():
                key = str(_lane_id(lane_id))
                local_indices.setdefault(key, set()).update(
                    _integer(index, "raw local point index") for index in indices)
        reasons = {str(reason) for item in upstream
                   for reason in item.get("reasons", [])}
        if len(exact) > 1 or any(distances[index] == 0 for index in support
                                 if index < len(distances)):
            reasons.add("zero_length_or_duplicate_distance")
        if any(index in shared for index in support):
            reasons.add("partition_shared_endpoint")
        if len(support) > 1 and any(
                source_indices[right] != source_indices[left] + 1
                for left, right in zip(support, support[1:])):
            reasons.add("fov_crop_gap")
        if len(support) > 1 and any(
                min(sources[right]["compact_source_indices"]) >
                max(sources[left]["compact_source_indices"]) + 1
                for left, right in zip(support, support[1:])
                if sources[left]["compact_source_indices"] and
                sources[right]["compact_source_indices"]):
            reasons.add("fov_crop_gap")
        if any(item.get("source_status", "verified") == "uncertain"
               for item in upstream):
            reasons.add("upstream_uncertain")
            if any(item.get("source_status") == "uncertain" and
                   not item.get("reasons") for item in upstream):
                reasons.add("unexplained_upstream_uncertainty")
        if len(owners) > 1 or any(item.get("source_status") == "partial"
                                  for item in upstream):
            reasons.add("member_boundary_mixed")
        if any(not item["raw_lane_candidates"] or
               not item["compact_source_indices"] for item in upstream):
            reasons.add("missing_source_evidence")
        if require_local_indices and any(
                not item.get("raw_lane_local_point_indices", {}).get(
                    str(_lane_id(owner)))
                for item in upstream for owner in item["raw_lane_candidates"]):
            reasons.add("missing_local_point_evidence")
        hard_uncertainty = reasons - {"member_boundary_mixed"}
        if hard_uncertainty:
            status = "uncertain"
        elif len(owners) > 1 or "member_boundary_mixed" in reasons:
            status = "partial"
        else:
            status = "verified"
        samples.append({
            "sample_index": output_index,
            "source_index": output_index,
            "input_indices": support,
            "raw_lane_candidates": owners,
            "compact_source_indices": compact_indices,
            "raw_lane_local_point_indices": {
                lane_id: sorted(indices) for lane_id, indices in sorted(local_indices.items())},
            "source_status": status,
            "unique_raw_lane_id": owners[0] if status == "verified" else None,
            "reasons": sorted(reasons),
        })
    return {"points": sampled, "samples": samples}


def classify_lane_sample(sample, lane_id):
    """Classify one candidate Lane at one resampled point, conservatively."""
    lane_id = _integer(lane_id, "lane_id")
    owners = sample["raw_lane_candidates"]
    status = sample["source_status"]
    reasons = set(sample.get("reasons", ()))
    if not owners:
        return "uncertain"
    if lane_id not in owners:
        if reasons & {"missing_source_evidence", "fov_crop_gap",
                      "unexplained_upstream_uncertainty"}:
            return "uncertain"
        return "filtered"
    if status == "uncertain":
        return "uncertain"
    if status == "partial":
        return "partial"
    return "retained" if sample["unique_raw_lane_id"] == lane_id else "uncertain"


def invert_simulated_permutation(new_to_pre, row_count):
    """Invert a supplied synthetic permutation; never labels an actual Batch."""
    row_count = _integer(row_count, "row_count")
    if row_count < 1:
        raise ValueError("row_count must be positive")
    values = [_integer(value, "row") for value in new_to_pre]
    if len(values) != row_count or set(values) != set(range(row_count)):
        raise ValueError("new_to_pre must be a complete permutation")
    return {pre_row: new_row for new_row, pre_row in enumerate(values)}
