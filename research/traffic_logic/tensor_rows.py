"""Pure Python observer for Lane ID to tensor row mapping."""

from numbers import Integral

from .lane_lineage import _lane_id


def _unique_lane_ids(values, label):
    ids = [_lane_id(value) for value in values]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate normalized Lane ID in {label}")
    return ids


def _indices(values, size, label):
    indices = list(values)
    if (any(isinstance(i, bool) or not isinstance(i, Integral)
            or i < 0 or i >= size for i in indices)
            or len(indices) != len(set(indices))):
        raise ValueError(f"Invalid {label}")
    return indices


def map_lane_ids_to_rows(candidate_lane_ids, graph_lane_ids, selected_indices,
                         lane_permutation=None):
    """Return Lane ID -> final row or None for lanes excluded from the tensor.

    ``graph_lane_ids`` is the iteration order of ``lg['lanes']`` passed to
    ``get_road_points_adj``. ``selected_indices`` is that call's actual
    ``np.argsort(dist_to_origin)[:num_lanes]``, indexing graph order. An
    optional training permutation indexes the selected rows in their new
    order, as in ``road_points = road_points[lane_perm]``. No distance or
    Group ID ordering is inferred here.
    """
    candidates = _unique_lane_ids(candidate_lane_ids, "candidates")
    graph_ids = _unique_lane_ids(graph_lane_ids, "graph")
    selected = _indices(selected_indices, len(graph_ids), "selected index")
    selected_ids = [graph_ids[index] for index in selected]
    if lane_permutation is not None:
        permutation = _indices(lane_permutation, len(selected_ids),
                               "lane permutation")
        if len(permutation) != len(selected_ids):
            raise ValueError("Invalid lane permutation length")
        selected_ids = [selected_ids[index] for index in permutation]

    rows = {lane_id: None for lane_id in candidates}
    rows.update({lane_id: None for lane_id in graph_ids})
    rows.update({lane_id: row for row, lane_id in enumerate(selected_ids)})
    return rows
