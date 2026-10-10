"""CPU tests for the offline E5-B Signal–Road candidate projection."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from research.traffic_logic.e5a_real_sidecar import DEFAULT_TFRECORD
from research.traffic_logic.signal_candidate_projection import project_sidecar
from research.traffic_logic.signal_lane_sidecar import dumps_sidecar, validate_sidecar
from research.traffic_logic.test_signal_lane_sidecar import fixture
from research.traffic_logic.test_signal_lane_sidecar_v11 import v11_fixture


MASK_KEYS = {
    "time_available",
    "state_known",
    "anchor_unique_verified",
    "stop_coordinate_in_fov",
    "candidate_road_row_exists",
    "local_lane_source_traceable",
    "final_stop_identity_verified",
    "candidate_relation_eligible",
    "condition_candidate_usable",
    "tensor_control_identity_proven",
}


def _token(result, lane_id):
    return next(token for token in result["tokens"]
                if token["raw_lane_id"] == lane_id)


def _relation(result, lane_id, branch, child_id=None):
    return next(relation for relation in result["candidate_relations"]
                if relation["raw_lane_id"] == lane_id and
                relation["road_branch"] == branch and
                relation["child_id"] == child_id)


def _shared_group_fixture():
    """Two signals share a Group and road rows, but own separate local scopes."""
    document = fixture()
    first = copy.deepcopy(document["signals"][0])
    first["raw_lane_id"] = 1
    first["signal"] = {
        "presence": "present", "state_code": 0, "state_name": "UNKNOWN",
        "stop_point_global_xyz": [1.0, 2.0, 0.0],
    }
    first["compact"]["source_point_range"] = [0, 2]
    first["compact"]["compact_point_range"] = [0, 2]
    first["anchor"] = {
        "audit_status": "not_run", "applies_to_raw_lane_id": 1,
        "predecessor_lane_id": None, "shared_compact_vertex_index": None,
        "shared_vertex_global_xy": None,
        "selected_vertex_consistent": None, "topology_verified": None,
        "geometric_candidate_vertex_indices": None,
        "geometric_candidates_count": None,
        "unique_anchor_verified": None, "xy_tolerance_m": None,
    }
    first["fov"]["stop_point_coordinate_status"] = "not_run"
    first["fov"]["stop_source_status"] = "not_run"
    first["road_tensor"]["regular"]["stop_identity_status"] = "not_run"
    for child in first["road_tensor"]["partitioned"]:
        child["stop_identity_status"] = "not_run"
    first["road_tensor"]["final_stop_identity_status"] = "not_run"
    first["road_tensor"]["final_tensor_stop_provenance_verified"] = None
    first["evidence_refs"].update(anchor=None, stop_fov=None, final_stop=None)
    document["signals"].append(first)
    validate_sidecar(document)
    return document


class SignalCandidateProjectionTests(unittest.TestCase):
    def test_dict_and_json_inputs_preserve_raw_signal_and_masks(self):
        document = fixture()
        before = copy.deepcopy(document)
        as_dict = project_sidecar(document)
        as_json = project_sidecar(dumps_sidecar(document))
        self.assertEqual(as_dict, as_json)
        self.assertEqual(document, before)
        self.assertEqual(as_dict["verification_level"], "structure_only")
        self.assertIs(as_dict["model_condition_injected"], False)
        self.assertEqual(len(as_dict["tokens"]), 2)
        for entry in as_dict["tokens"] + as_dict["candidate_relations"]:
            self.assertEqual(set(entry["masks"]), MASK_KEYS)
            self.assertTrue(all(isinstance(value, bool)
                                for value in entry["masks"].values()))
        stop = _token(as_dict, 2)
        self.assertEqual((stop["state_code"], stop["state_name"]), (4, "STOP"))
        self.assertEqual(stop["stop_point_global_xyz"], [5.0, 6.0, 0.0])
        self.assertEqual(stop["temporal_role"], "current")
        self.assertTrue(stop["masks"]["time_available"])
        self.assertTrue(stop["masks"]["state_known"])
        self.assertTrue(stop["masks"]["anchor_unique_verified"])
        self.assertTrue(stop["masks"]["stop_coordinate_in_fov"])
        self.assertFalse(stop["masks"]["final_stop_identity_verified"])
        self.assertFalse(stop["masks"]["tensor_control_identity_proven"])

    def test_shared_group_does_not_broadcast_stop_to_other_raw_lane(self):
        result = project_sidecar(_shared_group_fixture())
        stop = _token(result, 2)
        unknown = _token(result, 1)
        self.assertEqual(stop["state_name"], "STOP")
        self.assertEqual(unknown["state_name"], "UNKNOWN")
        self.assertTrue(stop["masks"]["anchor_unique_verified"])
        self.assertFalse(unknown["masks"]["anchor_unique_verified"])
        self.assertFalse(unknown["masks"]["condition_candidate_usable"])
        stop_relation = _relation(result, 2, "regular")
        unknown_relation = _relation(result, 1, "regular")
        self.assertEqual(stop_relation["group_id"], 7)
        self.assertEqual(unknown_relation["group_id"], 7)
        self.assertEqual(stop_relation["tensor_row"], 0)
        self.assertEqual(unknown_relation["tensor_row"], 0)
        self.assertEqual(stop_relation["source_point_range"], [1, 3])
        self.assertEqual(unknown_relation["source_point_range"], [0, 2])
        self.assertEqual(stop_relation["compact_point_range"], [2, 4])
        self.assertEqual(unknown_relation["compact_point_range"], [0, 2])
        self.assertEqual(unknown_relation["relation_status"], "unsupported")

    def test_partition_row_can_exist_when_stop_source_is_filtered(self):
        result = project_sidecar(fixture())
        regular = _relation(result, 2, "regular")
        filtered_stop = _relation(result, 2, "partitioned", 70)
        stop_side = _relation(result, 2, "partitioned", 71)
        self.assertEqual((regular["group_id"], regular["tensor_row"]), (7, 0))
        self.assertIsNone(regular["child_id"])
        self.assertEqual(filtered_stop["tensor_row"], 0)
        self.assertEqual(stop_side["tensor_row"], 1)
        self.assertEqual(filtered_stop["relation_status"], "unsupported")
        self.assertTrue(filtered_stop["masks"]["candidate_road_row_exists"])
        self.assertFalse(filtered_stop["masks"]["candidate_relation_eligible"])
        self.assertFalse(filtered_stop["masks"]["condition_candidate_usable"])
        self.assertFalse(filtered_stop["masks"]["tensor_control_identity_proven"])
        self.assertEqual(regular["relation_status"], "candidate_only")
        self.assertEqual(stop_side["relation_status"], "candidate_only")
        self.assertTrue(stop_side["masks"]["candidate_relation_eligible"])
        self.assertFalse(stop_side["masks"]["condition_candidate_usable"])
        self.assertFalse(stop_side["masks"]["tensor_control_identity_proven"])

    def test_verified_anchor_and_row_never_imply_final_stop_identity(self):
        result = project_sidecar(fixture())
        stop = _token(result, 2)
        self.assertTrue(stop["masks"]["anchor_unique_verified"])
        self.assertTrue(stop["masks"]["candidate_road_row_exists"])
        self.assertTrue(stop["masks"]["candidate_relation_eligible"])
        self.assertFalse(stop["masks"]["condition_candidate_usable"])
        self.assertFalse(stop["masks"]["final_stop_identity_verified"])
        self.assertFalse(stop["masks"]["tensor_control_identity_proven"])
        self.assertFalse(stop["masks"]["local_lane_source_traceable"])
        for relation in result["candidate_relations"]:
            if relation["raw_lane_id"] == 2:
                self.assertFalse(
                    relation["masks"]["tensor_control_identity_proven"])

    def test_self_reported_verified_stop_is_not_a_verified_mask(self):
        document = fixture()
        stop = document["signals"][0]
        stop["fov"]["stop_source_status"] = "verified"
        stop["road_tensor"]["regular"]["local_lane_source_status"] = "verified"
        stop["road_tensor"]["regular"]["stop_identity_status"] = "verified"
        stop["road_tensor"]["final_stop_identity_status"] = "verified"
        stop["road_tensor"]["final_tensor_stop_provenance_verified"] = True
        validate_sidecar(document)
        result = project_sidecar(document)
        relation = _relation(result, 2, "regular")
        self.assertEqual(relation["final_stop_identity_status"], "verified")
        self.assertTrue(relation["masks"]["candidate_relation_eligible"])
        self.assertFalse(relation["masks"]["final_stop_identity_verified"])
        self.assertFalse(relation["masks"]["condition_candidate_usable"])
        self.assertFalse(relation["masks"]["tensor_control_identity_proven"])

    def test_explicit_unknown_is_not_absent_lane_state(self):
        result = project_sidecar(fixture())
        self.assertEqual(result["raw_lane_ids"], [1, 2, 3, 4])
        self.assertEqual(result["signal_absent_raw_lane_ids"], [1, 4])
        self.assertEqual({token["raw_lane_id"] for token in result["tokens"]},
                         {2, 3})
        self.assertNotIn(4, {token["raw_lane_id"]
                             for token in result["tokens"]})
        unknown = _token(result, 3)
        self.assertEqual((unknown["state_code"], unknown["state_name"]),
                         (0, "UNKNOWN"))
        self.assertFalse(unknown["masks"]["state_known"])
        self.assertFalse(unknown["masks"]["condition_candidate_usable"])
        filtered_relation = _relation(result, 3, "regular")
        self.assertIsNone(filtered_relation["tensor_row"])
        self.assertEqual(filtered_relation["relation_status"], "filtered")
        self.assertFalse(filtered_relation["masks"]["candidate_road_row_exists"])

    def test_v11_retained_local_source_does_not_promote_stop_identity(self):
        document = v11_fixture(mapped=True)
        road = document["signals"][0]["road_tensor"]
        road["regular"]["local_lane_source_status"] = "retained"
        road["partitioned"][1]["local_lane_source_status"] = "retained"
        result = project_sidecar(document)
        for relation in (_relation(result, 2, "regular"),
                         _relation(result, 2, "partitioned", 71)):
            self.assertTrue(relation["masks"]["local_lane_source_traceable"])
            self.assertFalse(relation["masks"]["final_stop_identity_verified"])
            self.assertFalse(relation["masks"]["tensor_control_identity_proven"])

    def test_all_nine_signal_codes_stay_distinct(self):
        names = ("UNKNOWN", "ARROW_STOP", "ARROW_CAUTION", "ARROW_GO",
                 "STOP", "CAUTION", "GO", "FLASHING_STOP",
                 "FLASHING_CAUTION")
        for code, name in enumerate(names):
            with self.subTest(code=code):
                document = v11_fixture(mapped=False)
                document["signals"][0]["signal"].update(
                    state_code=code, state_name=name)
                result = project_sidecar(document)
                token = _token(result, 2)
                self.assertEqual((token["state_code"], token["state_name"]),
                                 (code, name))
                self.assertIs(token["masks"]["state_known"], code != 0)
                self.assertEqual(result["candidate_relations"], [])

    def test_future_mapped_label_has_rows_but_no_current_condition(self):
        # Structural mode sees only the Sidecar's self-reported future row.
        document = v11_fixture(time_index=11, current_time_index=10,
                               count=12, mapped=True)
        result = project_sidecar(document)
        token = _token(result, 2)
        relation = _relation(result, 2, "regular")
        self.assertEqual(token["temporal_role"], "future_label")
        self.assertFalse(token["masks"]["time_available"])
        self.assertFalse(token["masks"]["condition_candidate_usable"])
        self.assertEqual(relation["tensor_row"], 0)
        self.assertTrue(relation["masks"]["candidate_road_row_exists"])
        self.assertFalse(relation["masks"]["time_available"])
        self.assertFalse(relation["masks"]["condition_candidate_usable"])
        self.assertIs(result["model_condition_injected"], False)

    def test_future_row_rejects_stale_ten_clock_e4_evidence(self):
        document = v11_fixture(time_index=11, current_time_index=10,
                               count=12, mapped=True)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            old_context = {
                "scenario_id": document["scenario_id"],
                "record_index": document["record_index"],
                "time_index": 10,
                "record_payload_sha256": document["record_payload_sha256"],
            }
            for artifact, relative in document["evidence_paths"].items():
                path = base / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                content = (b"E3-C" if artifact == "e3c" else json.dumps(
                    {"input": old_context} if artifact in ("e4a", "e4b")
                    else old_context).encode("utf-8"))
                path.write_bytes(content)
                document["evidence_sha256"][artifact] = hashlib.sha256(
                    content).hexdigest()
            with self.assertRaisesRegex(ValueError, "time_index"):
                project_sidecar(document, evidence_base_dir=base)

    def test_not_run_mapping_never_emits_a_real_road_row(self):
        result = project_sidecar(v11_fixture(mapped=False))
        self.assertEqual(result["candidate_relations"], [])
        self.assertEqual(result["verification_level"], "structure_only")
        self.assertEqual(len(result["tokens"]), 2)
        for token in result["tokens"]:
            self.assertFalse(token["masks"]["candidate_road_row_exists"])
            self.assertFalse(token["masks"]["condition_candidate_usable"])

    def test_expected_context_and_payload_fingerprint_are_enforced(self):
        overrides = (
            ("expected_scenario_id", "other-scene"),
            ("expected_record_index", 1),
            ("expected_time_index", 11),
            ("expected_current_time_index", 11),
            ("expected_record_sha256", "b" * 64),
        )
        for name, value in overrides:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    project_sidecar(fixture(), **{name: value})

    def test_evidence_files_are_a_separate_level_and_changed_bytes_fail(self):
        document = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for artifact, relative in document["evidence_paths"].items():
                path = base / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                content = artifact.encode("ascii")
                path.write_bytes(content)
                document["evidence_sha256"][artifact] = hashlib.sha256(
                    content).hexdigest()
            result = project_sidecar(document, evidence_base_dir=base)
            self.assertEqual(result["verification_level"],
                             "evidence_files_consistent")
            self.assertFalse(_relation(result, 2, "regular")["masks"]
                             ["tensor_control_identity_proven"])
            (base / document["evidence_paths"]["e4d"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "digest"):
                project_sidecar(document, evidence_base_dir=base)

    def test_duplicate_signal_and_illegal_lane_reference_fail_closed(self):
        document = fixture()
        document["signals"].append(copy.deepcopy(document["signals"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            project_sidecar(document)
        document = fixture()
        document["signals"][0]["raw_lane_id"] = 999
        with self.assertRaisesRegex(ValueError, "raw Lane"):
            project_sidecar(document)

    def test_fixed_real_sidecar_rebuilds_twelve_lanes_without_upgrading_stop(self):
        sidecar_path = Path(__file__).parent / "output" / "e5a_sidecar.json"
        if not sidecar_path.is_file() or not DEFAULT_TFRECORD.is_file():
            self.skipTest("fixed E5-A Sidecar or original TFRecord unavailable")
        result = project_sidecar(
            sidecar_path.read_text(encoding="utf-8"),
            expected_scenario_id="637f20cafde22ff8",
            expected_record_index=0,
            expected_time_index=10,
            expected_current_time_index=10,
            evidence_base_dir=Path(__file__).parent,
            real_tfrecord=DEFAULT_TFRECORD,
        )
        self.assertEqual(result["verification_level"],
                         "fixed_scene_lane_rebuilt")
        self.assertEqual(len(result["tokens"]), 12)
        self.assertEqual(len(result["candidate_relations"]), 30)
        self.assertIs(result["model_condition_injected"], False)
        self.assertEqual({token["state_name"] for token in result["tokens"]},
                         {"UNKNOWN", "STOP", "ARROW_STOP"})
        for lane_id in (443, 445, 448, 449, 455, 456):
            token = _token(result, lane_id)
            self.assertTrue(token["masks"]["anchor_unique_verified"])
            self.assertFalse(token["masks"]["final_stop_identity_verified"])
            self.assertFalse(token["masks"]["tensor_control_identity_proven"])
            regular = _relation(result, lane_id, "regular")
            self.assertEqual(regular["relation_status"], "candidate_only")
            children = [relation for relation in result["candidate_relations"]
                        if relation["raw_lane_id"] == lane_id and
                        relation["road_branch"] == "partitioned"]
            self.assertEqual(len(children), 2)
            self.assertEqual({child["relation_status"] for child in children},
                             {"candidate_only", "unsupported"})
        for lane_id in (431, 432, 446, 447, 450, 457):
            token = _token(result, lane_id)
            self.assertEqual(token["state_name"], "UNKNOWN")
            self.assertFalse(token["masks"]["condition_candidate_usable"])

    def test_real_file_hashes_do_not_verify_a_tampered_lane_row(self):
        sidecar_path = Path(__file__).parent / "output" / "e5a_sidecar.json"
        if not sidecar_path.is_file() or not DEFAULT_TFRECORD.is_file():
            self.skipTest("fixed E5-A Sidecar or original TFRecord unavailable")
        document = json.loads(sidecar_path.read_text(encoding="utf-8"))
        stop = next(record for record in document["signals"]
                    if record["raw_lane_id"] == 443)
        stop["road_tensor"]["regular"]["tensor_row"] = 999
        structurally_valid = project_sidecar(
            document, evidence_base_dir=Path(__file__).parent)
        self.assertEqual(structurally_valid["verification_level"],
                         "evidence_files_consistent")
        self.assertEqual(_relation(structurally_valid, 443, "regular")
                         ["tensor_row"], 999)
        with self.assertRaisesRegex(ValueError, "claim differs"):
            project_sidecar(document,
                            evidence_base_dir=Path(__file__).parent,
                            real_tfrecord=DEFAULT_TFRECORD)


if __name__ == "__main__":
    unittest.main()
