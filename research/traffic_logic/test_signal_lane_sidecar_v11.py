"""Synthetic compatibility and time-axis checks for Sidecar schema v1.1.0."""

import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from research.traffic_logic.signal_lane_sidecar import (
    dumps_sidecar, loads_sidecar, signal_for_lane, validate_sidecar,
    verify_evidence_files, verify_original_time_axis)
from research.traffic_logic.test_signal_lane_sidecar import fixture


STATE_NAMES = {
    0: "UNKNOWN", 1: "ARROW_STOP", 2: "ARROW_CAUTION",
    3: "ARROW_GO", 4: "STOP", 5: "CAUTION", 6: "GO",
    7: "FLASHING_STOP", 8: "FLASHING_CAUTION",
}


def _varint(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _field(number, wire, value):
    tag = _varint((number << 3) | wire)
    return tag + (_varint(len(value)) + value if wire == 2 else value)


def raw_scenario_payload(timestamp_count=12, dynamic_count=12):
    return b"".join(
        [_field(5, 2, b"synthetic-scene"),
         _field(10, 0, _varint(10))] +
        [_field(1, 1, struct.pack("<d", float(i)))
         for i in range(timestamp_count)] +
        [_field(7, 2, b"") for _ in range(dynamic_count)])


def v11_fixture(time_index=10, current_time_index=10, count=12,
                mapped=False):
    document = copy.deepcopy(fixture())
    document.update(
        schema_version="1.1.0", time_index=time_index,
        current_time_index=current_time_index,
        dynamic_map_states_count=count,
        temporal_role=("history" if time_index < current_time_index else
                       "future_label" if time_index > current_time_index else
                       "current"),
        available_at_observation_cutoff=time_index <= current_time_index,
        mapping_evidence_time_index=time_index if mapped else None,
    )
    for record in document["signals"]:
        record["context"]["time_index"] = time_index
        record["mapping_audit_status"] = "verified" if mapped else "not_run"
        if not mapped:
            for field in ("compact", "anchor", "fov", "road_tensor"):
                record[field] = None
            for field in ("group_and_rows", "anchor", "stop_fov", "final_stop"):
                record["evidence_refs"][field] = None
    if not mapped:
        document["evidence_sha256"] = {}
        document["evidence_paths"] = {}
        document["row_index_stage"] = None
    return document


class SidecarV11Tests(unittest.TestCase):
    def test_nine_official_enum_codes_keep_distinct_names(self):
        for code, name in STATE_NAMES.items():
            with self.subTest(code=code):
                document = v11_fixture()
                document["signals"][0]["signal"].update(
                    state_code=code, state_name=name)
                self.assertEqual(loads_sidecar(dumps_sidecar(document)), document)
        document = v11_fixture()
        self.assertEqual(signal_for_lane(document, 3)["signal"]["state_name"],
                         "UNKNOWN")
        self.assertIsNone(signal_for_lane(document, 4))

    def test_unsupported_code_and_wrong_name_are_rejected(self):
        for code, name in [(9, "GO"), (2, "GO"), (0, "GO"), (-1, "UNKNOWN")]:
            with self.subTest(code=code, name=name):
                document = v11_fixture()
                document["signals"][0]["signal"].update(
                    state_code=code, state_name=name)
                with self.assertRaisesRegex(ValueError, "signal state|state_code"):
                    validate_sidecar(document)

    def test_local_retained_is_valid_without_upgrading_stop_identity(self):
        document = v11_fixture(mapped=True)
        regular = document["signals"][0]["road_tensor"]["regular"]
        regular["local_lane_source_status"] = "retained"
        child = document["signals"][0]["road_tensor"]["partitioned"][1]
        child["local_lane_source_status"] = "retained"
        self.assertEqual(loads_sidecar(dumps_sidecar(document)), document)
        self.assertEqual(regular["stop_identity_status"], "uncertain")
        for field_path in ("stop_identity_status", "anchor.audit_status",
                           "final_stop_identity_status"):
            with self.subTest(field=field_path):
                changed = copy.deepcopy(document)
                signal = changed["signals"][0]
                if field_path == "stop_identity_status":
                    signal["road_tensor"]["regular"][field_path] = "retained"
                elif field_path == "anchor.audit_status":
                    signal["anchor"]["audit_status"] = "retained"
                else:
                    signal["road_tensor"][field_path] = "retained"
                with self.assertRaisesRegex(ValueError, "status"):
                    validate_sidecar(changed)
        changed = copy.deepcopy(document)
        filtered = changed["signals"][0]["road_tensor"]["partitioned"][0]
        filtered["tensor_row"] = None
        filtered["row_status"] = "filtered"
        filtered["local_lane_source_status"] = "retained"
        with self.assertRaisesRegex(ValueError, "filtered row"):
            validate_sidecar(changed)
        legacy = fixture()
        legacy["signals"][0]["road_tensor"]["regular"][
            "local_lane_source_status"] = "retained"
        with self.assertRaisesRegex(ValueError, "local_lane_source"):
            validate_sidecar(legacy)

    def test_history_current_and_future_labels_round_trip(self):
        for t, role, visible in [(3, "history", True), (10, "current", True),
                                 (11, "future_label", False)]:
            with self.subTest(t=t):
                document = v11_fixture(t, 10, 12)
                self.assertEqual(document["temporal_role"], role)
                self.assertIs(document["available_at_observation_cutoff"],
                              visible)
                self.assertEqual(loads_sidecar(dumps_sidecar(document)), document)
                self.assertIsNone(document["mapping_evidence_time_index"])
                self.assertIsNone(document["signals"][0]["road_tensor"])
                self.assertIsNone(document["row_index_stage"])

    def test_time_indices_must_fit_original_dynamic_map_length(self):
        for t, current, count in [(12, 10, 12), (10, 12, 12), (0, 0, 0),
                                  (-1, 0, 12), (0, -1, 12)]:
            with self.subTest(t=t, current=current, count=count):
                document = v11_fixture(t, current, count)
                with self.assertRaisesRegex(ValueError, "time_index|dynamic_map"):
                    validate_sidecar(document)

    def test_raw_proto_confirms_declared_time_axis(self):
        payload = raw_scenario_payload()
        document = v11_fixture(11, 10, 12)
        digest = hashlib.sha256(payload).hexdigest()
        document["record_payload_sha256"] = digest
        for record in document["signals"]:
            record["context"]["record_payload_sha256"] = digest
        self.assertEqual(verify_original_time_axis(document, payload), 12)
        changed = copy.deepcopy(document)
        changed["dynamic_map_states_count"] = 13
        with self.assertRaisesRegex(ValueError, "dynamic_map_states_count"):
            verify_original_time_axis(changed, payload)
        broken_payload = raw_scenario_payload(timestamp_count=11)
        changed = copy.deepcopy(document)
        digest = hashlib.sha256(broken_payload).hexdigest()
        changed["record_payload_sha256"] = digest
        for record in changed["signals"]:
            record["context"]["record_payload_sha256"] = digest
        with self.assertRaisesRegex(ValueError, "timestamps_seconds"):
            verify_original_time_axis(changed, broken_payload)

    def test_future_label_cannot_be_marked_current_available(self):
        document = v11_fixture(11, 10, 12)
        document["available_at_observation_cutoff"] = True
        with self.assertRaisesRegex(ValueError, "available_at_observation_cutoff"):
            validate_sidecar(document)
        document = v11_fixture(11, 10, 12)
        document["temporal_role"] = "current"
        with self.assertRaisesRegex(ValueError, "temporal_role"):
            validate_sidecar(document)
        document = v11_fixture(11, 10, 12)
        document["model_condition_injected"] = True
        with self.assertRaisesRegex(ValueError, "model_condition_injected"):
            validate_sidecar(document)

    def test_ten_o_clock_mapping_evidence_cannot_follow_time_mutation(self):
        document = v11_fixture(mapped=True)
        document["time_index"] = 11
        document["temporal_role"] = "future_label"
        document["available_at_observation_cutoff"] = False
        for record in document["signals"]:
            record["context"]["time_index"] = 11
        with self.assertRaisesRegex(ValueError, "mapping_evidence_time_index"):
            validate_sidecar(document)
        document = v11_fixture(11, 10, 12, mapped=False)
        document["signals"][0]["road_tensor"] = copy.deepcopy(
            fixture()["signals"][0]["road_tensor"])
        with self.assertRaisesRegex(ValueError, "not_run mapping"):
            validate_sidecar(document)
        document = v11_fixture(11, 10, 12, mapped=False)
        document["row_index_stage"] = "pre_training_random_permutation"
        with self.assertRaisesRegex(ValueError, "row_index_stage"):
            validate_sidecar(document)
        document = v11_fixture(11, 10, 12, mapped=False)
        document["evidence_sha256"] = {"e4a": "4" * 64}
        document["evidence_paths"] = {"e4a": "output/e4a.json"}
        with self.assertRaisesRegex(ValueError, "not_run mapping"):
            validate_sidecar(document)

    def test_e4_evidence_context_rejects_relabelled_ten_o_clock_rows(self):
        document = v11_fixture(mapped=True)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            context = {
                "scenario_id": document["scenario_id"],
                "record_index": document["record_index"],
                "time_index": 10,
                "record_payload_sha256": document["record_payload_sha256"],
            }
            for artifact, relative in document["evidence_paths"].items():
                path = base / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                source = ({"input": context} if artifact in ("e4a", "e4b")
                          else context)
                content = (b"E3-C" if artifact == "e3c" else
                           json.dumps(source).encode("utf-8"))
                path.write_bytes(content)
                document["evidence_sha256"][artifact] = hashlib.sha256(
                    content).hexdigest()
            verify_evidence_files(document, base)
            changed = copy.deepcopy(document)
            changed.update(time_index=11, temporal_role="future_label",
                           available_at_observation_cutoff=False,
                           mapping_evidence_time_index=11)
            for record in changed["signals"]:
                record["context"]["time_index"] = 11
            validate_sidecar(changed)
            with self.assertRaisesRegex(ValueError, "e4a.*time"):
                verify_evidence_files(changed, base)

    def test_v1_legacy_round_trip_and_real_sample_remain_unchanged(self):
        legacy = fixture()
        self.assertEqual(loads_sidecar(dumps_sidecar(legacy)), legacy)
        legacy["signals"][0]["signal"].update(
            state_code=2, state_name="ARROW_CAUTION")
        with self.assertRaisesRegex(ValueError, "signal state"):
            validate_sidecar(legacy)
        sample = Path(__file__).parent / "output" / "e5a_sidecar.json"
        if sample.is_file():
            saved = sample.read_text(encoding="utf-8")
            decoded = loads_sidecar(saved)
            self.assertEqual(decoded["schema_version"], "1.0.0")
            self.assertEqual(dumps_sidecar(decoded), saved)
            self.assertEqual(len(decoded["signals"]), 12)
            self.assertIs(decoded["model_condition_injected"], False)


if __name__ == "__main__":
    unittest.main()
