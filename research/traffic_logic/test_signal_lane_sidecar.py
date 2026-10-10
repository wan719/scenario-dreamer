"""Synthetic contract tests for the independent E5-A Signal–Lane Sidecar."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from research.traffic_logic.signal_lane_sidecar import (
    dumps_sidecar,
    loads_sidecar,
    signal_for_lane,
    validate_sidecar,
    verify_evidence_files,
)


def fixture():
    context = {
        "scenario_id": "synthetic-scene",
        "record_index": 0,
        "time_index": 10,
        "record_payload_sha256": "a" * 64,
    }
    stop = {
        "context": context.copy(),
        "raw_lane_id": 2,
        "signal": {
            "presence": "present", "state_code": 4, "state_name": "STOP",
            "stop_point_global_xyz": [5.0, 6.0, 0.0],
        },
        "compact": {
            "group_id": 7, "ordered_member_lane_ids": [1, 2],
            "source_point_range": [1, 3],
            "compact_point_range": [2, 4],
        },
        "anchor": {
            "audit_status": "verified", "applies_to_raw_lane_id": 2,
            "predecessor_lane_id": 1, "shared_compact_vertex_index": 1,
            "shared_vertex_global_xy": [5.0, 6.0],
            "selected_vertex_consistent": True, "topology_verified": True,
            "geometric_candidate_vertex_indices": [1],
            "geometric_candidates_count": 1,
            "unique_anchor_verified": True,
            "xy_tolerance_m": 1e-9,
        },
        "fov": {
            "local_contribution_status": "partial",
            "stop_point_coordinate_status": "retained",
            "stop_source_status": "uncertain",
        },
        "road_tensor": {
            "regular": {
                "lane_id": 7, "tensor_row": 0, "row_status": "retained",
                "local_lane_source_status": "uncertain",
                "stop_identity_status": "uncertain",
            },
            "partitioned": [
                {"child_id": 70, "parent_upsampled_point_range": [0, 501],
                 "tensor_row": 0, "row_status": "retained",
                 "local_lane_source_status": "filtered",
                 "stop_identity_status": "filtered"},
                {"child_id": 71, "parent_upsampled_point_range": [500, 1000],
                 "tensor_row": 1, "row_status": "retained",
                 "local_lane_source_status": "uncertain",
                 "stop_identity_status": "uncertain"},
            ],
            "final_stop_identity_status": "uncertain",
            "final_tensor_stop_provenance_verified": None,
        },
        "evidence_refs": {
            "raw_signal": "record_payload", "group_and_rows": "e4a",
            "anchor": "e4d", "stop_fov": "e4b", "final_stop": "e4c",
        },
    }
    unknown = copy.deepcopy(stop)
    unknown["raw_lane_id"] = 3
    unknown["signal"] = {
        "presence": "present", "state_code": 0, "state_name": "UNKNOWN",
        "stop_point_global_xyz": [20.0, 30.0, 0.0],
    }
    unknown["compact"] = {
        "group_id": 8, "ordered_member_lane_ids": [3],
        "source_point_range": [0, 2], "compact_point_range": [0, 2],
    }
    unknown["anchor"] = {
        "audit_status": "not_run", "applies_to_raw_lane_id": 3,
        "predecessor_lane_id": None, "shared_compact_vertex_index": None,
        "shared_vertex_global_xy": None,
        "selected_vertex_consistent": None, "topology_verified": None,
        "geometric_candidate_vertex_indices": None,
        "geometric_candidates_count": None,
        "unique_anchor_verified": None,
        "xy_tolerance_m": None,
    }
    unknown["fov"] = {
        "local_contribution_status": "retained",
        "stop_point_coordinate_status": "not_run",
        "stop_source_status": "not_run",
    }
    unknown["road_tensor"] = {
        "regular": {"lane_id": 8, "tensor_row": None,
                    "row_status": "filtered",
                    "local_lane_source_status": "filtered",
                    "stop_identity_status": "not_run"},
        "partitioned": [{"child_id": 8,
                         "parent_upsampled_point_range": [0, 1000],
                         "tensor_row": None, "row_status": "filtered",
                         "local_lane_source_status": "filtered",
                         "stop_identity_status": "not_run"}],
        "final_stop_identity_status": "not_run",
        "final_tensor_stop_provenance_verified": None,
    }
    unknown["evidence_refs"] = {
        "raw_signal": "record_payload", "group_and_rows": "e4a",
        "anchor": None, "stop_fov": None, "final_stop": None,
    }
    return {
        "schema_name": "scenario-dreamer.signal-lane-sidecar",
        "schema_version": "1.0.0",
        **context,
        "current_time_index": 10,
        "official_repo_commit": "1" * 40,
        "official_source_blobs": {"official.py": "2" * 40},
        "evidence_sha256": {key: digit * 64 for key, digit in
                            [("e3c", "3"), ("e4a", "4"), ("e4b", "5"),
                             ("e4c", "6"), ("e4d", "7")]},
        "evidence_paths": {key: f"output/{key}.json" for key in
                           ("e3c", "e4a", "e4b", "e4c", "e4d")},
        "raw_lane_ids": [1, 2, 3, 4],
        "coordinate_frame": "waymo_global_xyz",
        "row_index_stage": "pre_training_random_permutation",
        "model_condition_injected": False,
        "signals": [stop, unknown],
    }


class SignalLaneSidecarContractTests(unittest.TestCase):
    def test_round_trip_preserves_signal_semantics_and_rows(self):
        document = fixture()
        encoded = dumps_sidecar(document)
        decoded = loads_sidecar(encoded)
        self.assertEqual(decoded, document)
        self.assertEqual(encoded, dumps_sidecar(decoded))
        self.assertIs(decoded["model_condition_injected"], False)
        self.assertEqual(decoded["signals"][0]["road_tensor"]["regular"]["tensor_row"], 0)
        self.assertEqual(decoded["signals"][0]["road_tensor"]["partitioned"][0]["stop_identity_status"], "filtered")
        self.assertEqual(decoded["signals"][0]["road_tensor"]["final_stop_identity_status"], "uncertain")

    def test_unknown_state_is_a_record_and_absent_signal_is_none(self):
        document = fixture()
        self.assertEqual(signal_for_lane(document, 3)["signal"]["state_name"],
                         "UNKNOWN")
        self.assertIsNone(signal_for_lane(document, 4))
        with self.assertRaisesRegex(ValueError, "raw Lane"):
            signal_for_lane(document, 99)

    def test_expected_scenario_and_time_guard_deserialization(self):
        encoded = dumps_sidecar(fixture())
        with self.assertRaisesRegex(ValueError, "scenario_id"):
            loads_sidecar(encoded, expected_scenario_id="different-scene")
        with self.assertRaisesRegex(ValueError, "time_index"):
            loads_sidecar(encoded, expected_time_index=11)
        with self.assertRaisesRegex(ValueError, "record_index"):
            loads_sidecar(encoded, expected_record_index=1)

    def test_cross_scenario_or_time_record_is_rejected(self):
        for field, value in [("scenario_id", "other"), ("time_index", 11),
                             ("record_payload_sha256", "b" * 64)]:
            with self.subTest(field=field):
                document = fixture()
                document["signals"][0]["context"][field] = value
                with self.assertRaisesRegex(ValueError, "context"):
                    validate_sidecar(document)

        document = fixture()
        document["current_time_index"] = 11
        with self.assertRaisesRegex(ValueError, "current_time_index"):
            validate_sidecar(document)

    def test_invalid_raw_lane_or_group_member_reference_is_rejected(self):
        document = fixture()
        document["signals"][0]["raw_lane_id"] = 99
        with self.assertRaisesRegex(ValueError, "raw Lane"):
            validate_sidecar(document)
        document = fixture()
        document["signals"][0]["compact"]["ordered_member_lane_ids"] = [1, 99, 2]
        with self.assertRaisesRegex(ValueError, "group member"):
            validate_sidecar(document)

    def test_predecessor_must_be_adjacent_group_member(self):
        document = fixture()
        document["signals"][0]["anchor"]["predecessor_lane_id"] = 3
        with self.assertRaisesRegex(ValueError, "predecessor"):
            validate_sidecar(document)

    def test_unrecognized_state_code_or_mismatched_name_is_rejected(self):
        for code, name in [(99, "UNKNOWN"), (4, "UNKNOWN"), (0, "GREEN")]:
            with self.subTest(code=code, name=name):
                document = fixture()
                document["signals"][0]["signal"]["state_code"] = code
                document["signals"][0]["signal"]["state_name"] = name
                with self.assertRaisesRegex(ValueError, "signal state"):
                    validate_sidecar(document)

    def test_missing_mapping_or_anchor_source_evidence_is_rejected(self):
        for artifact in ("e4a", "e4d"):
            with self.subTest(artifact=artifact):
                document = fixture()
                del document["evidence_sha256"][artifact]
                with self.assertRaisesRegex(ValueError, "evidence"):
                    validate_sidecar(document)
        document = fixture()
        document["signals"][0]["evidence_refs"]["anchor"] = None
        with self.assertRaisesRegex(ValueError, "anchor evidence"):
            validate_sidecar(document)
        document = fixture()
        del document["evidence_paths"]["e4d"]
        with self.assertRaisesRegex(ValueError, "evidence path"):
            validate_sidecar(document)

    def test_verified_claim_cannot_cite_mapping_as_anchor_evidence(self):
        document = fixture()
        document["signals"][0]["evidence_refs"].update(
            anchor="e4a", stop_fov="e4a", final_stop="e4a")
        for artifact in ("e4b", "e4c", "e4d"):
            del document["evidence_sha256"][artifact]
            del document["evidence_paths"][artifact]
        with self.assertRaisesRegex(ValueError, "anchor evidence"):
            validate_sidecar(document)

    def test_same_group_cannot_disagree_on_members_or_rows(self):
        document = fixture()
        other = document["signals"][1]
        other["compact"]["group_id"] = 7
        other["road_tensor"]["regular"]["lane_id"] = 7
        with self.assertRaisesRegex(ValueError, "Group 7"):
            validate_sidecar(document)

        document = fixture()
        other = document["signals"][1]
        other["raw_lane_id"] = 1
        other["anchor"]["applies_to_raw_lane_id"] = 1
        other["compact"] = {
            "group_id": 7, "ordered_member_lane_ids": [1, 2],
            "source_point_range": [0, 2], "compact_point_range": [0, 2],
        }
        other["road_tensor"]["regular"]["lane_id"] = 7
        other["road_tensor"]["partitioned"] = copy.deepcopy(
            document["signals"][0]["road_tensor"]["partitioned"])
        for child in other["road_tensor"]["partitioned"]:
            child["stop_identity_status"] = "not_run"
        with self.assertRaisesRegex(ValueError, "Group 7"):
            validate_sidecar(document)

    def test_filtered_tensor_row_cannot_claim_verified_source(self):
        document = fixture()
        regular = document["signals"][0]["road_tensor"]["regular"]
        regular["tensor_row"] = None
        regular["row_status"] = "filtered"
        regular["local_lane_source_status"] = "verified"
        regular["stop_identity_status"] = "verified"
        with self.assertRaisesRegex(ValueError, "filtered row"):
            validate_sidecar(document)

    def test_evidence_file_verifier_rejects_missing_and_changed_bytes(self):
        document = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for artifact, relative_path in document["evidence_paths"].items():
                path = base / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                content = artifact.encode("ascii")
                path.write_bytes(content)
                document["evidence_sha256"][artifact] = hashlib.sha256(
                    content).hexdigest()
            verify_evidence_files(document, base)
            changed = base / document["evidence_paths"]["e4d"]
            changed.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "digest"):
                verify_evidence_files(document, base)
            changed.unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                verify_evidence_files(document, base)

    def test_not_run_anchor_cannot_carry_verified_claims(self):
        document = fixture()
        document["signals"][1]["anchor"]["unique_anchor_verified"] = True
        with self.assertRaisesRegex(ValueError, "not_run"):
            validate_sidecar(document)

    def test_partial_anchor_still_validates_candidate_lane_and_types(self):
        document = fixture()
        anchor = document["signals"][0]["anchor"]
        anchor["audit_status"] = "partial"
        anchor["unique_anchor_verified"] = False
        anchor["predecessor_lane_id"] = 999
        anchor["shared_compact_vertex_index"] = "bogus"
        anchor["topology_verified"] = "yes"
        with self.assertRaisesRegex(ValueError, "predecessor"):
            validate_sidecar(document)
        anchor["predecessor_lane_id"] = 1
        with self.assertRaisesRegex(ValueError, "shared_compact_vertex_index"):
            validate_sidecar(document)
        anchor["shared_compact_vertex_index"] = 1
        with self.assertRaisesRegex(ValueError, "topology_verified"):
            validate_sidecar(document)

    def test_candidate_count_bool_and_unhashable_status_are_rejected(self):
        document = fixture()
        document["signals"][0]["anchor"]["geometric_candidates_count"] = True
        with self.assertRaisesRegex(ValueError, "geometric_candidates_count"):
            validate_sidecar(document)
        document = fixture()
        document["signals"][0]["fov"]["stop_point_coordinate_status"] = []
        with self.assertRaisesRegex(ValueError, "fov.stop_point_coordinate"):
            validate_sidecar(document)

    def test_stop_coordinate_retention_is_separate_from_fov_source(self):
        document = fixture()
        self.assertEqual(document["signals"][0]["fov"]
                         ["stop_point_coordinate_status"], "retained")
        self.assertEqual(document["signals"][0]["fov"]
                         ["stop_source_status"], "uncertain")
        del document["signals"][0]["fov"]["stop_source_status"]
        with self.assertRaisesRegex(ValueError, "stop_source_status"):
            validate_sidecar(document)
        document = fixture()
        document["signals"][0]["fov"]["stop_source_status"] = "not_run"
        with self.assertRaisesRegex(ValueError, "stop source.*audit"):
            validate_sidecar(document)

    def test_row_presence_does_not_upgrade_stop_identity(self):
        document = fixture()
        stop = document["signals"][0]
        self.assertEqual(stop["road_tensor"]["partitioned"][0]["tensor_row"], 0)
        self.assertEqual(stop["road_tensor"]["partitioned"][0]["stop_identity_status"], "filtered")
        self.assertIsNone(stop["road_tensor"]["final_tensor_stop_provenance_verified"])
        self.assertEqual(loads_sidecar(dumps_sidecar(document))["signals"][0]["road_tensor"]["final_stop_identity_status"], "uncertain")

    def test_model_condition_flag_cannot_be_true(self):
        document = fixture()
        document["model_condition_injected"] = True
        with self.assertRaisesRegex(ValueError, "model_condition_injected"):
            validate_sidecar(document)

    def test_duplicate_json_keys_are_rejected(self):
        encoded = dumps_sidecar(fixture())
        encoded = encoded.replace('"schema_version": "1.0.0",',
                                  '"schema_version": "1.0.0", "schema_version": "1.0.0",')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            loads_sidecar(encoded)


if __name__ == "__main__":
    unittest.main()
