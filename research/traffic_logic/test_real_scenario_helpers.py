import struct
import tempfile
import unittest
from pathlib import Path

from research.traffic_logic.real_scenario_helpers import (
    build_graphs,
    parse_scenario,
    read_tfrecord_record,
)


def varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def field(number, wire, value):
    payload = varint(value) if wire == 0 else value
    if wire == 2:
        payload = varint(len(payload)) + payload
    return varint((number << 3) | wire) + payload


class RealScenarioHelperTests(unittest.TestCase):
    def test_read_only_first_record_framing(self):
        payload = b"scenario-zero"
        record = struct.pack("<Q", len(payload)) + b"\0" * 4 + payload + b"\0" * 4
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.tfrecord"
            path.write_bytes(record)
            self.assertEqual(read_tfrecord_record(path, 0), payload)

    def test_parse_lane_topology_and_timestep_signal(self):
        point = field(1, 1, struct.pack("<d", 1.0)) + field(2, 1, struct.pack("<d", 2.0))
        lane = (
            field(2, 0, 2)
            + field(8, 2, point) + field(8, 2, point)
            + field(9, 2, varint(101))
            + field(10, 0, 103)
        )
        feature = field(1, 0, 102) + field(3, 2, lane)
        signal = field(1, 0, 102) + field(2, 0, 4)
        scenario = (
            field(5, 2, b"scenario-zero")
            + field(8, 2, feature)
            + field(7, 2, field(1, 2, signal))
        )
        result = parse_scenario(scenario, timestep=0)
        self.assertEqual(result["scenario_id"], "scenario-zero")
        self.assertEqual(result["lane_states"], {102: 4})
        self.assertEqual(result["raw_lanes"][102]["type"], "TYPE_SURFACE_STREET")
        self.assertEqual(result["raw_lanes"][102]["entryLanes"], ["101"])
        self.assertEqual(result["raw_lanes"][102]["exitLanes"], ["103"])
        self.assertEqual(result["raw_lanes"][102]["polyline"],
                         [{"x": 1.0, "y": 2.0, "z": 0.0}] * 2)

    def test_build_graphs_uses_same_compact_group_for_chain(self):
        lanes = {
            101: {"type": "TYPE_SURFACE_STREET", "polyline": [
                {"x": 0, "y": 0, "z": 0}, {"x": 1, "y": 0, "z": 0}],
                "exitLanes": ["102"]},
            102: {"type": "TYPE_SURFACE_STREET", "polyline": [
                {"x": 1, "y": 0, "z": 0}, {"x": 2, "y": 0, "z": 0}],
                "entryLanes": ["101"], "exitLanes": ["103"]},
            103: {"type": "TYPE_SURFACE_STREET", "polyline": [
                {"x": 2, "y": 0, "z": 0}, {"x": 3, "y": 0, "z": 0}],
                "entryLanes": ["102"]},
        }
        raw_graph, compact_graph, groups = build_graphs(lanes)
        self.assertEqual(set(raw_graph["lanes"]), {101, 102, 103})
        self.assertEqual(len(compact_graph["lanes"]), 1)
        self.assertEqual(list(groups.values()), [[101, 102, 103]])


if __name__ == "__main__":
    unittest.main()
