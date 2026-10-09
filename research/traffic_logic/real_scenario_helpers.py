"""Minimal read-only WOMD reader and official lane-graph function adapter.

Only protobuf fields needed for lane grouping are decoded. The official
preprocessor imports TensorFlow, Hydra and Waymo packages at module load time;
this adapter executes its selected function definitions without those imports.
"""

import ast
import copy
import struct
from pathlib import Path

import numpy as np


LANE_TYPES = {
    0: "TYPE_UNDEFINED",
    1: "TYPE_FREEWAY",
    2: "TYPE_SURFACE_STREET",
    3: "TYPE_BIKE_LANE",
}


def read_tfrecord_record(path, index=0):
    """Read one uncompressed TFRecord payload; framing CRCs are not checked."""
    if index < 0:
        raise ValueError("Record index must be nonnegative")
    with open(path, "rb") as stream:
        for record_index in range(index + 1):
            header = stream.read(12)
            if not header:
                raise IndexError(f"TFRecord has no record at index {index}")
            if len(header) != 12:
                raise ValueError("Truncated TFRecord header")
            size = struct.unpack("<Q", header[:8])[0]
            payload = stream.read(size)
            footer = stream.read(4)
            if len(payload) != size or len(footer) != 4:
                raise ValueError(f"Truncated TFRecord payload at {record_index}")
    return payload


def _varint(data, position):
    result = 0
    for shift in range(0, 70, 7):
        if position >= len(data):
            raise ValueError("Truncated protobuf varint")
        byte = data[position]
        position += 1
        result |= (byte & 127) << shift
        if byte < 128:
            return result, position
    raise ValueError("Protobuf varint exceeds 10 bytes")


def _fields(data):
    position = 0
    while position < len(data):
        tag, position = _varint(data, position)
        number, wire = tag >> 3, tag & 7
        if number == 0:
            raise ValueError("Invalid protobuf field zero")
        if wire == 0:
            value, position = _varint(data, position)
        elif wire in (1, 5):
            end = position + (8 if wire == 1 else 4)
            value = data[position:end]
            position = end
        elif wire == 2:
            size, position = _varint(data, position)
            end = position + size
            value = data[position:end]
            position = end
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        if wire in (1, 5) and len(value) != (8 if wire == 1 else 4):
            raise ValueError("Truncated protobuf fixed-width field")
        if position > len(data):
            raise ValueError("Truncated protobuf field")
        yield number, wire, value


def _packed_ints(value, wire):
    if wire == 0:
        return [value]
    values = []
    position = 0
    while position < len(value):
        item, position = _varint(value, position)
        values.append(item)
    return values


def _point(data):
    point = {"x": 0.0, "y": 0.0, "z": 0.0}
    for number, wire, value in _fields(data):
        if number in (1, 2, 3) and wire == 1:
            point[("x", "y", "z")[number - 1]] = struct.unpack("<d", value)[0]
    return point


def _neighbor(data):
    for number, wire, value in _fields(data):
        if number == 1 and wire == 0:
            return {"featureId": str(value)}
    raise ValueError("Lane neighbor lacks feature_id")


def _lane(data):
    lane = {"type": "TYPE_UNDEFINED", "polyline": []}
    for number, wire, value in _fields(data):
        if number == 2 and wire == 0:
            lane["type"] = LANE_TYPES.get(value, f"TYPE_{value}")
        elif number == 8 and wire == 2:
            lane["polyline"].append(_point(value))
        elif number in (9, 10) and wire in (0, 2):
            key = "entryLanes" if number == 9 else "exitLanes"
            lane.setdefault(key, []).extend(str(item) for item in _packed_ints(value, wire))
        elif number in (11, 12) and wire == 2:
            key = "leftNeighbors" if number == 11 else "rightNeighbors"
            lane.setdefault(key, []).append(_neighbor(value))
    return lane


def _map_lane(data):
    lane_id = None
    lane_data = None
    for number, wire, value in _fields(data):
        if number == 1 and wire == 0:
            lane_id = value
        elif number == 3 and wire == 2:
            lane_data = value
    if lane_id is None or lane_data is None:
        return None
    return lane_id, _lane(lane_data)


def _signals(data):
    signals = {}
    count = 0
    for number, wire, value in _fields(data):
        if number != 1 or wire != 2:
            continue
        count += 1
        lane_id = None
        state = None
        for field_number, field_wire, field_value in _fields(value):
            if field_number == 1 and field_wire == 0:
                lane_id = field_value
            elif field_number == 2 and field_wire == 0:
                state = field_value
        if lane_id is not None:
            if lane_id in signals:
                raise ValueError(f"Duplicate signal Lane ID: {lane_id}")
            signals[lane_id] = state
    return signals, count


def parse_scenario(payload, timestep):
    """Decode the Scenario fields required by this compact-only diagnostic."""
    scenario_id = None
    current_time_index = None
    raw_lanes = {}
    dynamic_state_count = 0
    lane_states = None
    lane_state_count = 0
    for number, wire, value in _fields(payload):
        if number == 5 and wire == 2:
            scenario_id = value.decode("utf-8")
        elif number == 10 and wire == 0:
            current_time_index = value
        elif number == 8 and wire == 2:
            parsed = _map_lane(value)
            if parsed is not None:
                lane_id, lane = parsed
                if lane_id in raw_lanes:
                    raise ValueError(f"Duplicate map Lane ID: {lane_id}")
                raw_lanes[lane_id] = lane
        elif number == 7 and wire == 2:
            if dynamic_state_count == timestep:
                lane_states, lane_state_count = _signals(value)
            dynamic_state_count += 1
    if scenario_id is None or lane_states is None:
        raise ValueError("Scenario ID or requested dynamic map timestep is missing")
    return {
        "scenario_id": scenario_id,
        "current_time_index": current_time_index,
        "raw_lanes": raw_lanes,
        "lane_states": lane_states,
        "lane_state_count": lane_state_count,
        "dynamic_state_count": dynamic_state_count,
    }


def _source_functions(path, names, namespace):
    source = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [node for node in source.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise RuntimeError(f"Required official functions missing from {path}")
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)


def build_graphs(raw_lanes):
    """Run the repository's exact lane-graph and compact function bodies."""
    root = Path(__file__).absolute().parents[2]
    preprocess = root / "data_processing/waymo/generate_waymo_dataset.py"
    compact_source = root / "utils/lane_graph_helpers.py"
    namespace = {"np": np, "copy": copy}
    _source_functions(preprocess, (
        "poly_gon_and_line", "road_info_lane", "get_lane_pairs",
        "get_engage_lanes", "get_lane_graph", "process_lanegraph",
    ), namespace)
    _source_functions(compact_source, (
        "get_compact_lane_graph", "find_lane_groups", "find_lane_group_id",
    ), namespace)

    features = [{"id": str(lane_id), "lane": lane}
                for lane_id, lane in raw_lanes.items()]
    data = {"road_info": {"lane": namespace["road_info_lane"](features)}}
    data["graph"] = namespace["get_lane_graph"](data)
    lane_graph = namespace["process_lanegraph"](data)

    captured_groups = []
    actual_find_groups = namespace["find_lane_groups"]

    def capture_groups(pre_pairs, suc_pairs):
        groups = actual_find_groups(pre_pairs, suc_pairs)
        captured_groups.append(groups)
        return groups

    namespace["find_lane_groups"] = capture_groups
    compact_graph = namespace["get_compact_lane_graph"](
        copy.deepcopy({"lane_graph": lane_graph}))
    if len(captured_groups) != 1:
        raise RuntimeError("Compact graph did not produce exactly one grouping")
    return lane_graph, compact_graph, captured_groups[0]
