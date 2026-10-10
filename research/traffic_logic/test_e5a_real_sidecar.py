"""One-record integration check for the E5-A read-only Sidecar builder."""

import unittest
import copy
import hashlib
from pathlib import Path

from research.traffic_logic.e5a_real_sidecar import (
    build_real_sidecar, verify_real_sidecar)
from research.traffic_logic.e4d_real_anchor_diagnostic import DEFAULT_TFRECORD
from research.traffic_logic.real_scenario_helpers import read_tfrecord_record
from research.traffic_logic.signal_lane_sidecar import (
    dumps_sidecar, loads_sidecar, signal_for_lane,
    verify_evidence_files, verify_original_time_axis)


class RealSidecarTests(unittest.TestCase):
    def test_original_record_confirms_ninety_one_time_steps(self):
        if not Path(DEFAULT_TFRECORD).is_file():
            self.skipTest("Local original WOMD record is unavailable")
        document = copy.deepcopy(build_real_sidecar(DEFAULT_TFRECORD))
        document.update(
            schema_version="1.1.0", dynamic_map_states_count=91,
            temporal_role="current", available_at_observation_cutoff=True,
            mapping_evidence_time_index=10)
        for record in document["signals"]:
            record["mapping_audit_status"] = "verified"
        payload = read_tfrecord_record(DEFAULT_TFRECORD, 0)
        self.assertEqual(verify_original_time_axis(document, payload), 91)
        self.assertEqual(len(verify_evidence_files(
            document, Path(__file__).parent)), 5)

    def test_saved_v1_sample_is_byte_stable_and_matches_raw_evidence(self):
        sample = Path(__file__).parent / "output" / "e5a_sidecar.json"
        if not sample.is_file() or not Path(DEFAULT_TFRECORD).is_file():
            self.skipTest("Saved Sidecar or original WOMD record is unavailable")
        saved = sample.read_bytes()
        self.assertEqual(
            hashlib.sha256(saved).hexdigest(),
            "95cc8e182d524a9cf8b9df3943f569ec4bf2dd3cc4373a79058eeceb7bb908c3")
        document = loads_sidecar(saved.decode("utf-8"))
        self.assertEqual(document["schema_version"], "1.0.0")
        self.assertEqual(dumps_sidecar(document),
                         sample.read_text(encoding="utf-8"))
        self.assertEqual(document, build_real_sidecar(DEFAULT_TFRECORD))
        verify_real_sidecar(document, DEFAULT_TFRECORD)

    def test_record_zero_preserves_twelve_signal_records_and_provenance(self):
        if not Path(DEFAULT_TFRECORD).is_file():
            self.skipTest("Local original WOMD record is unavailable")
        document = build_real_sidecar(DEFAULT_TFRECORD)
        self.assertEqual(document["scenario_id"], "637f20cafde22ff8")
        self.assertEqual(document["time_index"], 10)
        self.assertEqual(len(document["signals"]), 12)
        self.assertEqual(
            {record["signal"]["state_name"] for record in document["signals"]},
            {"STOP", "ARROW_STOP", "UNKNOWN"})
        self.assertEqual(
            [record["raw_lane_id"] for record in document["signals"]
             if record["anchor"]["audit_status"] == "verified"],
            [443, 445, 448, 449, 455, 456])
        self.assertEqual(signal_for_lane(document, 431)["anchor"]["audit_status"],
                         "not_run")
        self.assertEqual(signal_for_lane(document, 431)["signal"]["state_name"],
                         "UNKNOWN")
        self.assertIsNone(signal_for_lane(document, 529))
        lane443 = signal_for_lane(document, 443)
        self.assertEqual(lane443["compact"]["group_id"], 11)
        self.assertEqual(lane443["anchor"]["shared_compact_vertex_index"], 118)
        self.assertEqual(lane443["fov"]["stop_point_coordinate_status"],
                         "retained")
        self.assertEqual(lane443["fov"]["stop_source_status"], "uncertain")
        self.assertEqual(lane443["road_tensor"]["regular"]["tensor_row"], 12)
        self.assertEqual(
            [(item["child_id"], item["tensor_row"], item["stop_identity_status"])
             for item in lane443["road_tensor"]["partitioned"]],
            [(132, 19, "filtered"), (133, 20, "uncertain")])
        self.assertIsNone(lane443["road_tensor"]
                          ["final_tensor_stop_provenance_verified"])
        self.assertFalse(document["model_condition_injected"])
        verify_real_sidecar(document, DEFAULT_TFRECORD)

    def test_saved_e4_evidence_rejects_unsupported_status_upgrade(self):
        if not Path(DEFAULT_TFRECORD).is_file():
            self.skipTest("Local original WOMD record is unavailable")
        document = build_real_sidecar(DEFAULT_TFRECORD)
        changed = copy.deepcopy(document)
        lane443 = signal_for_lane(changed, 443)
        lane443["road_tensor"]["final_stop_identity_status"] = "verified"
        lane443["road_tensor"]["final_tensor_stop_provenance_verified"] = True
        with self.assertRaisesRegex(ValueError, "final_stop_identity_status"):
            verify_real_sidecar(changed, DEFAULT_TFRECORD)
        changed = copy.deepcopy(document)
        lane431 = signal_for_lane(changed, 431)
        lane431["signal"]["state_code"] = 4
        lane431["signal"]["state_name"] = "STOP"
        with self.assertRaisesRegex(ValueError, "state_code"):
            verify_real_sidecar(changed, DEFAULT_TFRECORD)


if __name__ == "__main__":
    unittest.main()
