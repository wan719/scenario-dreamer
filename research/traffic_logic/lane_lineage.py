"""Offline, dependency-free diagnostics for original Waymo lane lineage.

Ranges are zero-based and half-open. They describe point indices before any
normalization, FOV crop, partition, or resampling.
"""


def _lane_id(value):
    """Normalize the integer IDs and decimal strings emitted by MessageToDict."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"Lane ID must be an integer or decimal string: {value!r}")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Lane ID must be decimal: {value!r}") from exc


def build_lane_lineage(lane_groups, lanes):
    """Map original lanes to supplied compact groups and concatenation ranges.

    ``lane_groups`` must be the ordered membership produced for the same raw
    graph by ``find_lane_groups``. ``lanes`` maps raw IDs to point sequences.
    The first lane contributes all points; later lanes contribute points[1:].
    The span additionally includes the preceding compact point as a connection
    marker; it does not assert that the dropped source point is identical.
    """
    normalized_lanes = {}
    for raw_id, points in lanes.items():
        lane_id = _lane_id(raw_id)
        if lane_id in normalized_lanes:
            raise ValueError(f"Duplicate normalized Lane ID: {lane_id}")
        normalized_lanes[lane_id] = points

    lineage = {}
    for raw_group_id, members in lane_groups.items():
        group_id = _lane_id(raw_group_id)
        offset = 0
        for index, raw_id in enumerate(members):
            lane_id = _lane_id(raw_id)
            if lane_id in lineage:
                raise ValueError(f"Lane ID occurs in multiple groups: {lane_id}")
            if lane_id not in normalized_lanes:
                raise KeyError(f"Group member has no source Lane: {lane_id}")
            point_count = len(normalized_lanes[lane_id])
            if point_count == 0:
                raise ValueError(f"Source Lane has no points: {lane_id}")
            source_start = 0 if index == 0 else 1
            retained_count = point_count - source_start
            lineage[lane_id] = {
                "group_id": group_id,
                "source_point_range": (source_start, point_count),
                "compact_point_range": (offset, offset + retained_count),
                "compact_span_range": (offset - (index > 0), offset + retained_count),
            }
            offset += retained_count
    return lineage


def diagnose_controlled_lanes(lineage, controlled_lane_ids):
    """Report each controlled raw Lane; never infer a group-wide light state."""
    report = []
    for raw_id in controlled_lane_ids:
        lane_id = _lane_id(raw_id)
        record = {"raw_lane_id": lane_id}
        record.update(lineage.get(lane_id, {"group_id": None}))
        report.append(record)
    return report
