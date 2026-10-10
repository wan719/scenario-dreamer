"""Read-only, fixed-time replay of the official Waymo road source bodies.

The local Python environment does not have the official TensorFlow/Torch stack.
This module extracts selected function bodies from the checked official source
files with AST and runs them on CPU. It is a source-body replay, not an import
of the official dataset module or a captured training batch.
"""

import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np

from .e4_geometry_audit import plan_partition_lineage
from .real_scenario_helpers import (build_graphs, parse_scenario,
                                    read_tfrecord_record)


ROOT = Path(__file__).absolute().parents[2]
SCENARIO_ID = "637f20cafde22ff8"
RECORD_SHA256 = "7af5edce831d6aec3eae0f665ee137b3eaf97a949cec6c9ea1e0127e1be46442"
RECORD_INDEX = 0
TIME_INDEX = 10
OFFICIAL_SOURCES = (
    "data_processing/waymo/generate_waymo_dataset.py",
    "datasets/waymo/dataset_autoencoder_waymo.py",
    "utils/lane_graph_helpers.py",
    "utils/data_helpers.py",
    "utils/geometry.py",
    "cfgs/config.yaml",
    "cfgs/dataset/waymo_base.yaml",
)
ROAD_METHODS = (
    "normalize_compact_lane_graph",
    "get_lane_graph_within_fov",
    "partition_compact_lane_graph",
    "get_road_points_adj",
)


class _NumpySortProbe:
    """Expose NumPy normally while recording source-body distance sorts."""

    def __init__(self):
        self.sort_calls = []

    def __getattr__(self, name):
        return getattr(np, name)

    def argsort(self, values, *args, **kwargs):
        result = np.argsort(values, *args, **kwargs)
        self.sort_calls.append((np.asarray(values).copy(),
                                np.asarray(result).copy()))
        return result


def _load_json(value):
    if isinstance(value, (str, Path)):
        return json.loads(Path(value).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        return value
    raise TypeError("E4 evidence must be a JSON path or decoded dictionary")


def _git_blob(path):
    return subprocess.check_output(
        ["git", "hash-object", str(ROOT / path)], cwd=ROOT,
        text=True).strip()


def _extract_functions(path, names, namespace, *, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if class_name is None:
        nodes = tree.body
    else:
        classes = [node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == class_name]
        if len(classes) != 1:
            raise RuntimeError(f"Expected one {class_name} in {path}")
        nodes = classes[0].body
    functions = [node for node in nodes
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise RuntimeError(f"Required official function bodies missing from {path}")
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)


def _official_namespace():
    probe = _NumpySortProbe()
    namespace = {"np": probe, "copy": copy,
                 "Any": Any, "Dict": Dict, "Tuple": Tuple}
    _extract_functions(ROOT / "utils/geometry.py",
                       ("make_2d_rotation_matrix", "apply_se2_transform"),
                       namespace)
    _extract_functions(ROOT / "utils/lane_graph_helpers.py",
                       ("resample_polyline",), namespace)
    _extract_functions(ROOT / "datasets/waymo/dataset_autoencoder_waymo.py",
                       ROAD_METHODS, namespace,
                       class_name="WaymoDatasetAutoEncoder")
    return namespace, probe


def _assert_same(label, actual, expected):
    if actual != expected:
        raise ValueError(f"{label} differs from saved E4-A evidence")


def _capture_rows(namespace, probe, proxy, graph, baseline, label):
    probe.sort_calls.clear()
    output = namespace["get_road_points_adj"](proxy, graph)
    if len(probe.sort_calls) != 1:
        raise RuntimeError(f"{label}: expected exactly one official np.argsort")
    distances, full_order = probe.sort_calls[0]
    road_points, pre, suc, left, right, num_lanes = output
    graph_ids = list(graph["lanes"])
    selected = [int(index) for index in full_order[:num_lanes]]
    selected_ids = [graph_ids[index] for index in selected]
    if num_lanes != min(len(graph_ids), proxy.cfg.max_num_lanes):
        raise RuntimeError(f"{label}: official row count differs from cfg")
    if distances.shape != (len(graph_ids),):
        raise RuntimeError(f"{label}: official distance vector shape differs")
    if road_points.shape != (num_lanes, proxy.cfg.num_points_per_lane, 2):
        raise RuntimeError(f"{label}: official road tensor shape differs")
    unsorted = np.asarray([
        namespace["resample_polyline"](
            graph["lanes"][lane_id], num_points=proxy.cfg.num_points_per_lane)
        for lane_id in graph_ids
    ])
    if not np.array_equal(road_points, unsorted[selected]):
        raise RuntimeError(f"{label}: returned rows disagree with captured sort")
    row_by_id = {lane_id: None for lane_id in graph_ids}
    row_by_id.update({lane_id: row for row, lane_id in enumerate(selected_ids)})
    result = {
        "graph_ids_in_iteration_order": graph_ids,
        "minimum_distances": distances.tolist(),
        "full_argsort_indices": [int(index) for index in full_order],
        "selected_indices": selected,
        "selected_lane_ids": selected_ids,
        "lane_id_to_tensor_row": row_by_id,
        "num_lanes": int(num_lanes),
        "road_points": road_points,
        "pre_road_adj": pre,
        "suc_road_adj": suc,
        "left_road_adj": left,
        "right_road_adj": right,
    }
    for field in ("graph_ids_in_iteration_order", "full_argsort_indices",
                  "selected_indices", "selected_lane_ids", "num_lanes"):
        _assert_same(f"{label}.{field}", result[field], baseline[field])
    baseline_distances = np.asarray(baseline["minimum_distances"], dtype=float)
    if (baseline_distances.shape != distances.shape or
            not np.isfinite(baseline_distances).all() or
            not np.isfinite(distances).all() or
            not np.allclose(distances, baseline_distances,
                            rtol=0.0, atol=1e-12)):
        raise ValueError(f"{label}.minimum_distances differs from E4-A")
    expected_rows = {int(lane_id): row for lane_id, row in
                     baseline["lane_id_to_tensor_row"].items()}
    _assert_same(f"{label}.lane_id_to_tensor_row", row_by_id, expected_rows)
    return result


def _plan_to_json_shape(plan):
    return {
        str(parent): {
            "crossings": list(entry["crossings"]),
            "children": [[child, list(point_range)]
                         for child, point_range in entry["children"]],
            "shared_endpoints": list(entry["shared_endpoints"]),
            "unused_ids": list(entry["unused_ids"]),
        }
        for parent, entry in plan.items()
    }


def replay_fixed_scene(tfrecord, e4a, e4d):
    """Replay record 0 at t=10 and return inspected graphs and pre-shuffle rows.

    ``e4a`` and ``e4d`` can be decoded JSON dictionaries or paths. This reads
    the TFRecord and evidence but never writes caches, invokes ``get_data``,
    constructs a PyG Batch, or imports the official dataset module.
    """
    e4a, e4d = _load_json(e4a), _load_json(e4d)
    expected_context = (SCENARIO_ID, RECORD_INDEX, TIME_INDEX, RECORD_SHA256)
    a_context = e4a["input"]
    if e4a["run_status"] != "pass" or (
            a_context["scenario_id"], a_context["record_index"],
            a_context["time_index"], a_context["record_payload_sha256"]
    ) != expected_context:
        raise ValueError("E4-A evidence has a different record or timestep")
    if e4d["run_status"] != "pass" or (
            e4d["scenario_id"], e4d["record_index"],
            e4d["time_index"], e4d["record_payload_sha256"]
    ) != expected_context:
        raise ValueError("E4-D evidence has a different record or timestep")

    local_blobs = {}
    expected_blobs = e4a["environment"]["official_source_blobs"]
    for path in OFFICIAL_SOURCES:
        local_blobs[path] = _git_blob(path)
        official_commit_blob = subprocess.check_output(
            ["git", "rev-parse", f"{e4a['environment']['repo_commit']}:{path}"],
            cwd=ROOT, text=True).strip()
        if local_blobs[path] != official_commit_blob:
            raise ValueError(f"Official source changed since E4-A commit: {path}")
        if path in expected_blobs and local_blobs[path] != expected_blobs[path]:
            raise ValueError(f"Official source changed since E4-A: {path}")
        if path in e4d["official_source_blobs_local"] and (
                local_blobs[path] != e4d["official_source_blobs_local"][path]):
            raise ValueError(f"Official source changed since E4-D: {path}")

    payload = read_tfrecord_record(tfrecord, RECORD_INDEX)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    if payload_sha256 != RECORD_SHA256:
        raise ValueError("TFRecord payload differs from fixed-scene evidence")
    parsed = parse_scenario(payload, TIME_INDEX)
    if (parsed["scenario_id"], parsed["current_time_index"]) != (
            SCENARIO_ID, TIME_INDEX):
        raise ValueError("Parsed Scenario ID or current time differs")
    expected_states = {int(lane_id): int(state)
                       for lane_id, state in a_context["signal_states"].items()}
    _assert_same("t=10 LaneState map", parsed["lane_states"], expected_states)

    # Exactly one grouping; build_graphs itself executes the official source
    # bodies for raw-to-processed and Compact construction.
    processed, compact, groups = build_graphs(parsed["raw_lanes"])
    counts = e4a["counts"]
    for label, value in (("raw_lanes", len(parsed["raw_lanes"])),
                         ("processed_lanes", len(processed["lanes"])),
                         ("compact_groups", len(compact["lanes"]))):
        _assert_same(label, value, counts[label])
    for lane in e4d["lanes"]:
        group_id = lane["group_id"]
        _assert_same(f"Lane {lane['raw_lane_id']} Group ID",
                     next(group for group, members in groups.items()
                          if lane["raw_lane_id"] in members), group_id)
        _assert_same(f"Group {group_id} members", groups[group_id],
                     lane["group_members"])
        geometry = np.asarray(compact["lanes"][group_id], dtype="<f8")
        _assert_same(f"Group {group_id} point count", len(geometry),
                     lane["compact_point_count"])
        _assert_same(f"Group {group_id} geometry fingerprint",
                     hashlib.sha256(geometry.tobytes()).hexdigest(),
                     lane["compact_geometry_sha256"])

    namespace, probe = _official_namespace()
    cfg_values = e4a["environment"]["cfg"]
    proxy = SimpleNamespace(cfg=SimpleNamespace(**cfg_values))
    normalize_dict = {
        "center": np.asarray(a_context["ego_t10_center"], dtype=np.float64),
        "yaw": np.float64(a_context["ego_t10_yaw"]),
    }
    normalized = namespace["normalize_compact_lane_graph"](
        proxy, copy.deepcopy(compact), normalize_dict)
    fov_masks = {
        group_id: ((np.abs(points[:, 0]) < proxy.cfg.fov / 2)
                   & (np.abs(points[:, 1]) < proxy.cfg.fov / 2))
        for group_id, points in normalized["lanes"].items()
    }
    for lane in e4a["lanes"]:
        group_id = lane["group_id"]
        mask_bits = "".join("1" if value else "0"
                            for value in fov_masks[group_id])
        _assert_same(f"Group {group_id} FoV mask", mask_bits,
                     lane["group_fov_mask_bits"])
    fov = namespace["get_lane_graph_within_fov"](proxy, normalized)
    _assert_same("FoV Group order", list(fov["lanes"]),
                 e4a["regular"]["graph_ids_in_iteration_order"])
    _assert_same("FoV Group count", len(fov["lanes"]), counts["fov_groups"])
    for group_id, points in fov["lanes"].items():
        expected = namespace["resample_polyline"](
            normalized["lanes"][group_id][fov_masks[group_id]],
            num_points=proxy.cfg.upsample_lane_num_points)
        if not np.array_equal(points, expected):
            raise RuntimeError(f"Group {group_id} 1000-point output differs")

    partition_plan = plan_partition_lineage(fov["lanes"])
    _assert_same("Partition plan", _plan_to_json_shape(partition_plan),
                 e4a["partition_parent_children"])
    partitioned = namespace["partition_compact_lane_graph"](
        proxy, copy.deepcopy(fov))
    expected_child_ids = set()
    for parent_id, entry in partition_plan.items():
        for child_id, (start, end) in entry["children"]:
            expected_child_ids.add(child_id)
            if child_id not in partitioned["lanes"] or not np.array_equal(
                    partitioned["lanes"][child_id],
                    fov["lanes"][parent_id][start:end]):
                raise RuntimeError(f"Partition child {child_id} differs")
    _assert_same("Partition child ID set", set(partitioned["lanes"]),
                 expected_child_ids)
    _assert_same("Partition graph order", list(partitioned["lanes"]),
                 e4a["partitioned"]["graph_ids_in_iteration_order"])
    _assert_same("Partition lane count", len(partitioned["lanes"]),
                 counts["partitioned_lanes"])

    regular_rows = _capture_rows(namespace, probe, proxy, fov,
                                 e4a["regular"], "regular")
    partitioned_rows = _capture_rows(namespace, probe, proxy, partitioned,
                                     e4a["partitioned"], "partitioned")
    return {
        "payload_sha256": payload_sha256,
        "parsed": parsed,
        "processed": processed,
        "compact": compact,
        "groups": groups,
        "normalized": normalized,
        "fov_masks": fov_masks,
        "fov": fov,
        "partitioned": partitioned,
        "partition_plan": partition_plan,
        "regular_rows": regular_rows,
        "partitioned_rows": partitioned_rows,
        "execution_scope": "research_side_ast_source_body_replay_not_official_module_import",
        "official_source_blobs": local_blobs,
    }
