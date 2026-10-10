"""CPU tests for the independent E5-C Signal Token interface simulation."""

import copy
import json
import math
from pathlib import Path
import unittest

from research.traffic_logic.e5a_real_sidecar import DEFAULT_TFRECORD
from research.traffic_logic.signal_candidate_projection import project_sidecar
from research.traffic_logic.signal_interface_simulation import (
    global_to_ego_xy,
    simulate_interface,
)
from research.traffic_logic.test_signal_candidate_projection import (
    _shared_group_fixture,
)
from research.traffic_logic.test_signal_lane_sidecar import fixture
from research.traffic_logic.test_signal_lane_sidecar_v11 import v11_fixture


RESEARCH_DIR = Path(__file__).absolute().parent
STOP_LANES = {443, 445, 448, 449, 455, 456}
STATE_NAMES = (
    "UNKNOWN", "ARROW_STOP", "ARROW_CAUTION", "ARROW_GO", "STOP",
    "CAUTION", "GO", "FLASHING_STOP", "FLASHING_CAUTION",
)


def _token(result, lane_id):
    return next(token for token in result["tokens"]
                if token["raw_lane_id"] == lane_id)


def _alignment(result, lane_id, branch, child_id=None):
    return next(entry for entry in result["alignments"]
                if entry["raw_lane_id"] == lane_id and
                entry["road_branch"] == branch and
                entry["child_id"] == child_id)


def _context(document):
    return {key: document[key] for key in (
        "scenario_id", "record_index", "time_index",
        "record_payload_sha256")}


def _pose(document):
    return {
        "context": _context(document),
        "source_frame": "waymo_global_xyz",
        "target_frame": "waymo_ego_xy",
        "center_global_xy": [0.0, 0.0],
        "yaw_rad": 0.0,
    }


def _claim(relation):
    claim = {key: relation[key] for key in (
        "raw_lane_id", "group_id", "road_branch", "child_id",
        "source_point_range", "compact_point_range", "row_index_stage")}
    claim["tensor_row"] = relation["tensor_row"]
    return claim


def _context_claim(relation, document):
    claim = _claim(relation)
    claim["context"] = _context(document)
    return claim


class SignalInterfaceSimulationTests(unittest.TestCase):
    def test_global_to_ego_matches_official_waymo_rotation_convention(self):
        actual = global_to_ego_xy([12.0, -3.0], [10.0, -3.0], 0.0)
        self.assertAlmostEqual(actual[0], 0.0, places=12)
        self.assertAlmostEqual(actual[1], 2.0, places=12)
        rotated = global_to_ego_xy([5.0, 6.0], [0.0, 0.0], 0.0)
        self.assertAlmostEqual(rotated[0], -6.0, places=12)
        self.assertAlmostEqual(rotated[1], 5.0, places=12)

    def test_observed_stop_token_is_separate_from_unverified_road_alignment(self):
        document = fixture()
        before = copy.deepcopy(document)
        previous = project_sidecar(document)
        result = simulate_interface(document)
        self.assertEqual(document, before)
        self.assertIs(result["model_condition_injected"], False)
        self.assertEqual(result["verification_level"], "structure_only")
        self.assertEqual(len(result["tokens"]), 2)
        self.assertEqual(len(result["alignments"]), 5)
        token = _token(result, 2)
        self.assertEqual((token["raw_lane_id"], token["state_code"],
                          token["state_name"]), (2, 4, "STOP"))
        self.assertEqual(token["stop_point_global_xyz"], [5.0, 6.0, 0.0])
        self.assertIs(token["explicit_signal_observed"], True)
        self.assertIs(token["temporal_condition_available"], True)
        self.assertIs(token["raw_lane_scope_structurally_consistent"], True)
        self.assertIs(token["raw_lane_scope_known"], False)
        self.assertIs(token["anchor_alignment_verified"], False)
        self.assertEqual(token["anchor_audit_status"], "verified")
        self.assertIs(token["anchor_unique_verified"], True)
        self.assertEqual(token["local_fov_status"], "partial")
        self.assertEqual(token["stop_coordinate_fov_status"], "retained")
        self.assertEqual(token["stop_source_fov_status"], "uncertain")
        self.assertEqual(token["final_stop_identity_status"], "uncertain")
        self.assertIs(token["new_interface_candidate"], True)
        self.assertIs(token["new_interface_verified"], False)
        self.assertEqual(token["e5b_masks"], previous["tokens"][0]["masks"])
        self.assertIs(token["e5b_masks"]["tensor_control_identity_proven"],
                      False)
        regular = _alignment(result, 2, "regular")
        self.assertEqual(regular["road_row_alignment_status"], "uncertain")
        self.assertIs(regular["new_interface_candidate"], True)
        self.assertIs(regular["new_interface_verified"], False)
        self.assertIsNone(regular["simulated_post_shuffle_row"])
        self.assertEqual(regular["e5b_masks"],
                         previous["candidate_relations"][0]["masks"])

    def test_partitioned_non_stop_side_is_filtered_despite_row_presence(self):
        result = simulate_interface(fixture())
        regular = _alignment(result, 2, "regular")
        non_stop = _alignment(result, 2, "partitioned", 70)
        stop_side = _alignment(result, 2, "partitioned", 71)
        self.assertEqual((regular["group_id"],
                          regular["pre_shuffle_tensor_row"]), (7, 0))
        self.assertEqual((non_stop["pre_shuffle_tensor_row"],
                          stop_side["pre_shuffle_tensor_row"]), (0, 1))
        self.assertEqual(non_stop["road_row_alignment_status"], "filtered")
        self.assertIs(non_stop["new_interface_candidate"], False)
        self.assertIs(non_stop["e5b_masks"]["candidate_road_row_exists"],
                      True)
        self.assertEqual(stop_side["road_row_alignment_status"], "uncertain")
        self.assertIs(stop_side["new_interface_candidate"], True)
        self.assertIs(stop_side["new_interface_verified"], False)

    def test_fov_filtered_stop_does_not_leave_uncertain_row_candidate(self):
        document = fixture()
        stop = document["signals"][0]
        stop["fov"]["stop_point_coordinate_status"] = "filtered"
        stop["fov"]["stop_source_status"] = "filtered"
        result = simulate_interface(document)
        regular = _alignment(result, 2, "regular")
        self.assertEqual(regular["road_row_alignment_status"], "filtered")
        self.assertIs(regular["new_interface_candidate"], False)

    def test_invalid_local_source_remains_invalid(self):
        document = fixture()
        document["signals"][0]["road_tensor"]["regular"][
            "local_lane_source_status"] = "invalid"
        result = simulate_interface(document)
        regular = _alignment(result, 2, "regular")
        self.assertEqual(regular["road_row_alignment_status"], "invalid")
        self.assertIs(regular["new_interface_candidate"], False)

    def test_unrun_local_fov_audit_cannot_create_candidate(self):
        document = fixture()
        document["signals"][0]["fov"]["local_contribution_status"] = "not_run"
        result = simulate_interface(document)
        for branch, child in (("regular", None), ("partitioned", 71)):
            alignment = _alignment(result, 2, branch, child)
            self.assertEqual(alignment["road_row_alignment_status"], "not_run")
            self.assertIs(alignment["new_interface_candidate"], False)
            self.assertIs(alignment["e5b_masks"]["candidate_relation_eligible"],
                          True)

    def test_unrun_stop_fov_audit_remains_not_run(self):
        document = fixture()
        document["signals"][0]["fov"].update(
            stop_point_coordinate_status="not_run", stop_source_status="not_run")
        document["signals"][0]["evidence_refs"]["stop_fov"] = None
        result = simulate_interface(document)
        regular = _alignment(result, 2, "regular")
        self.assertEqual(regular["road_row_alignment_status"], "not_run")
        self.assertIs(regular["new_interface_candidate"], False)

    def test_new_candidate_does_not_require_old_final_stop_audit(self):
        document = fixture()
        stop = document["signals"][0]
        stop["road_tensor"]["final_stop_identity_status"] = "not_run"
        stop["evidence_refs"]["final_stop"] = None
        result = simulate_interface(document)
        regular = _alignment(result, 2, "regular")
        self.assertIs(regular["e5b_masks"]["candidate_relation_eligible"],
                      False)
        self.assertIs(regular["new_interface_candidate"], True)
        self.assertIs(regular["new_interface_verified"], False)

    def test_shared_group_keeps_state_and_local_scope_per_raw_lane(self):
        result = simulate_interface(_shared_group_fixture())
        stop = _token(result, 2)
        unknown = _token(result, 1)
        self.assertEqual((stop["state_code"], stop["state_name"]), (4, "STOP"))
        self.assertEqual((unknown["state_code"], unknown["state_name"]),
                         (0, "UNKNOWN"))
        self.assertIs(stop["new_interface_candidate"], True)
        self.assertIs(unknown["new_interface_candidate"], False)
        stop_row = _alignment(result, 2, "regular")
        unknown_row = _alignment(result, 1, "regular")
        self.assertEqual(stop_row["group_id"], unknown_row["group_id"])
        self.assertEqual(stop_row["pre_shuffle_tensor_row"],
                         unknown_row["pre_shuffle_tensor_row"])
        self.assertEqual(stop_row["source_point_range"], [1, 3])
        self.assertEqual(unknown_row["source_point_range"], [0, 2])
        self.assertEqual(stop_row["compact_point_range"], [2, 4])
        self.assertEqual(unknown_row["compact_point_range"], [0, 2])
        self.assertIs(unknown_row["new_interface_candidate"], False)

    def test_explicit_unknown_has_token_and_absent_lane_has_none(self):
        result = simulate_interface(fixture())
        self.assertEqual({item["raw_lane_id"] for item in result["tokens"]},
                         {2, 3})
        self.assertNotIn(4, {item["raw_lane_id"] for item in result["tokens"]})
        unknown = _token(result, 3)
        self.assertEqual((unknown["state_code"], unknown["state_name"]),
                         (0, "UNKNOWN"))
        self.assertIs(unknown["explicit_signal_observed"], True)
        self.assertIs(unknown["temporal_condition_available"], True)
        self.assertIs(unknown["new_interface_candidate"], False)
        self.assertIs(unknown["e5b_masks"]["state_known"], False)
        self.assertEqual(_alignment(result, 3, "regular")
                         ["road_row_alignment_status"], "filtered")

    def test_nine_original_signal_enums_remain_distinct(self):
        for code, name in enumerate(STATE_NAMES):
            with self.subTest(code=code):
                document = v11_fixture(mapped=False)
                document["signals"][0]["signal"].update(
                    state_code=code, state_name=name)
                result = simulate_interface(document)
                token = _token(result, 2)
                self.assertEqual((token["state_code"], token["state_name"]),
                                 (code, name))
                self.assertIs(token["explicit_signal_observed"], True)
                self.assertIs(token["new_interface_candidate"], False)
                self.assertEqual(result["alignments"], [])

    def test_future_label_keeps_observation_but_blocks_current_condition(self):
        document = v11_fixture(time_index=11, current_time_index=10,
                               count=12, mapped=True)
        result = simulate_interface(document)
        stop = _token(result, 2)
        self.assertIs(stop["explicit_signal_observed"], True)
        self.assertIs(stop["temporal_condition_available"], False)
        self.assertIs(stop["new_interface_candidate"], False)
        self.assertIs(stop["new_interface_verified"], False)
        regular = _alignment(result, 2, "regular")
        self.assertEqual(regular["pre_shuffle_tensor_row"], 0)
        self.assertIs(regular["new_interface_candidate"], False)

    def test_mapping_not_run_keeps_token_without_real_road_link(self):
        result = simulate_interface(v11_fixture(mapped=False))
        self.assertEqual(len(result["tokens"]), 2)
        self.assertEqual(result["alignments"], [])
        stop = _token(result, 2)
        self.assertIs(stop["explicit_signal_observed"], True)
        self.assertIs(stop["temporal_condition_available"], True)
        self.assertIs(stop["raw_lane_scope_known"], False)
        self.assertIs(stop["anchor_alignment_verified"], False)
        self.assertEqual(stop["mapping_audit_status"], "not_run")
        self.assertEqual(stop["anchor_audit_status"], "not_run")
        self.assertEqual(stop["stop_coordinate_fov_status"], "not_run")
        self.assertIs(stop["new_interface_candidate"], False)

    def test_synthetic_pose_can_be_consistent_without_real_verification(self):
        document = fixture()
        result = simulate_interface(document, ego_pose=_pose(document))
        stop = _token(result, 2)
        self.assertAlmostEqual(stop["stop_point_ego_xy"][0], -6.0, places=12)
        self.assertAlmostEqual(stop["stop_point_ego_xy"][1], 5.0, places=12)
        self.assertIs(stop["anchor_alignment_consistent"], True)
        self.assertIs(stop["anchor_alignment_verified"], False)
        self.assertIs(stop["new_interface_verified"], False)

    def test_matching_identity_claims_do_not_promote_alignment_to_verified(self):
        document = fixture()
        projection = project_sidecar(document)
        claims = [_context_claim(relation, document) for relation in
                  projection["candidate_relations"]]
        result = simulate_interface(document, alignment_claims=claims)
        self.assertEqual(len(result["alignments"]), len(claims))
        self.assertTrue(_alignment(result, 2, "regular")
                        ["new_interface_candidate"])
        self.assertTrue(all(not entry["new_interface_verified"]
                            for entry in result["alignments"]))
        self.assertTrue(all(not item["new_interface_verified"]
                            for item in result["tokens"]))

    def test_wrong_lane_group_child_row_scope_or_stage_claim_is_rejected(self):
        document = fixture()
        relation = next(item for item in project_sidecar(document)
                        ["candidate_relations"]
                        if item["raw_lane_id"] == 2 and
                        item["road_branch"] == "partitioned" and
                        item["child_id"] == 71)
        baseline = _context_claim(relation, document)
        mutations = (
            ("raw_lane_id", 3),
            ("group_id", 8),
            ("child_id", 70),
            ("tensor_row", 0),
            ("source_point_range", [0, 2]),
            ("compact_point_range", [0, 2]),
            ("row_index_stage", "post_training_random_permutation"),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                changed = copy.deepcopy(baseline)
                changed[field] = value
                with self.assertRaises(ValueError):
                    simulate_interface(document, alignment_claims=[changed])

    def test_duplicate_or_incomplete_alignment_claim_is_rejected(self):
        document = fixture()
        claim = _context_claim(
            project_sidecar(document)["candidate_relations"][0], document)
        with self.assertRaises(ValueError):
            simulate_interface(document, alignment_claims=[claim, claim])
        missing = copy.deepcopy(claim)
        del missing["compact_point_range"]
        with self.assertRaises(ValueError):
            simulate_interface(document, alignment_claims=[missing])

    def test_alignment_claim_requires_exact_scenario_time_and_payload(self):
        document = fixture()
        claim = _context_claim(
            project_sidecar(document)["candidate_relations"][0], document)
        for field, wrong in (("scenario_id", "neighbor-scene"),
                             ("record_index", 1), ("time_index", 9),
                             ("record_payload_sha256", "b" * 64)):
            with self.subTest(field=field):
                changed = copy.deepcopy(claim)
                changed["context"][field] = wrong
                with self.assertRaises(ValueError):
                    simulate_interface(document, alignment_claims=[changed])
        missing = copy.deepcopy(claim)
        del missing["context"]
        with self.assertRaises(ValueError):
            simulate_interface(document, alignment_claims=[missing])
        bool_record = copy.deepcopy(claim)
        bool_record["context"]["record_index"] = False
        with self.assertRaises(ValueError):
            simulate_interface(document, alignment_claims=[bool_record])

    def test_pose_frame_context_and_nonfinite_values_are_rejected(self):
        document = fixture()
        baseline = _pose(document)
        mutations = (
            ("source_frame", "waymo_ego_xy"),
            ("target_frame", "waymo_global_xyz"),
            ("center_global_xy", [math.nan, 0.0]),
            ("yaw_rad", math.inf),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                pose = copy.deepcopy(baseline)
                pose[field] = value
                with self.assertRaises(ValueError):
                    simulate_interface(document, ego_pose=pose)
        pose = copy.deepcopy(baseline)
        pose["context"]["scenario_id"] = "neighbor-scene"
        with self.assertRaises(ValueError):
            simulate_interface(document, ego_pose=pose)
        pose = copy.deepcopy(baseline)
        pose["context"]["time_index"] = 9
        with self.assertRaises(ValueError):
            simulate_interface(document, ego_pose=pose)
        pose = copy.deepcopy(baseline)
        pose["context"]["record_index"] = False
        with self.assertRaises(ValueError):
            simulate_interface(document, ego_pose=pose)

    def test_missing_evidence_cannot_claim_real_verification(self):
        document = fixture()
        with self.assertRaises(ValueError):
            simulate_interface(document, real_tfrecord=DEFAULT_TFRECORD)
        with self.assertRaises((ValueError, FileNotFoundError)):
            simulate_interface(document,
                               evidence_base_dir=RESEARCH_DIR / "not-a-directory")

    def test_pre_shuffle_rows_need_explicit_branch_specific_permutation(self):
        document = fixture()
        context = _context(document)
        result = simulate_interface(document, row_permutations={
            "regular": {"context": context, "row_count": 2,
                        "new_to_pre": [1, 0]},
            "partitioned": {"context": context, "row_count": 2,
                            "new_to_pre": [0, 1]},
        })
        regular = _alignment(result, 2, "regular")
        first_child = _alignment(result, 2, "partitioned", 70)
        second_child = _alignment(result, 2, "partitioned", 71)
        self.assertEqual(regular["pre_shuffle_tensor_row"], 0)
        self.assertEqual(regular["simulated_post_shuffle_row"], 1)
        self.assertEqual(first_child["simulated_post_shuffle_row"], 0)
        self.assertEqual(second_child["simulated_post_shuffle_row"], 1)
        self.assertIs(regular["new_interface_verified"], False)

    def test_invalid_permutation_or_wrong_context_is_rejected(self):
        document = fixture()
        context = _context(document)
        for values in ([0, 0], [2, 0], [-1, 0], [True, 0]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    simulate_interface(document, row_permutations={
                        "regular": {"context": context,
                                    "row_count": 2,
                                    "new_to_pre": values}})
        wrong = copy.deepcopy(context)
        wrong["record_payload_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            simulate_interface(document, row_permutations={
                "regular": {"context": wrong, "row_count": 2,
                            "new_to_pre": [1, 0]}})
        with self.assertRaises(ValueError):
            simulate_interface(document, row_permutations={
                "regular": {"context": context, "row_count": 3,
                            "new_to_pre": [1, 0]}})
        with self.assertRaises(ValueError):
            simulate_interface(document, row_permutations={
                "regular": {"context": context,
                            "new_to_pre": [1, 0]}})
        bool_context = copy.deepcopy(context)
        bool_context["record_index"] = False
        with self.assertRaises(ValueError):
            simulate_interface(document, row_permutations={
                "regular": {"context": bool_context, "row_count": 2,
                            "new_to_pre": [1, 0]}})

    def test_expected_sidecar_context_mismatch_is_rejected(self):
        mismatches = (
            ("expected_scenario_id", "adjacent-scene"),
            ("expected_record_index", 1),
            ("expected_time_index", 9),
            ("expected_current_time_index", 11),
            ("expected_record_sha256", "b" * 64),
        )
        for name, value in mismatches:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    simulate_interface(fixture(), **{name: value})

    def test_fixed_real_six_stop_lanes_keep_old_provenance_masks_false(self):
        sidecar_path = RESEARCH_DIR / "output" / "e5a_sidecar.json"
        e4a_path = RESEARCH_DIR / "output" / "e4a_autodl" / "e4a_result.json"
        e4b_path = RESEARCH_DIR / "output" / "e4b_autodl" / "e4b_result.json"
        e4d_path = RESEARCH_DIR / "output" / "e4d_result.json"
        if not all(path.is_file() for path in
                   (sidecar_path, e4a_path, e4b_path, e4d_path,
                    DEFAULT_TFRECORD)):
            self.skipTest("fixed E5-A/E4 evidence or original TFRecord unavailable")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        e4a = json.loads(e4a_path.read_text(encoding="utf-8"))
        e4b = json.loads(e4b_path.read_text(encoding="utf-8"))
        e4d = json.loads(e4d_path.read_text(encoding="utf-8"))
        for evidence in (e4a, e4b):
            self.assertEqual(evidence["input"]["scenario_id"],
                             sidecar["scenario_id"])
            self.assertEqual(evidence["input"]["record_index"],
                             sidecar["record_index"])
            self.assertEqual(evidence["input"]["time_index"],
                             sidecar["time_index"])
            self.assertEqual(evidence["input"]["record_payload_sha256"],
                             sidecar["record_payload_sha256"])
        for name in ("scenario_id", "record_index", "time_index",
                     "record_payload_sha256"):
            self.assertEqual(e4d[name], sidecar[name])
        pose = {
            "context": _context(sidecar),
            "source_frame": "waymo_global_xyz",
            "target_frame": "waymo_ego_xy",
            "center_global_xy": e4a["input"]["ego_t10_center"],
            "yaw_rad": e4a["input"]["ego_t10_yaw"],
        }
        result = simulate_interface(
            sidecar,
            evidence_base_dir=RESEARCH_DIR,
            real_tfrecord=DEFAULT_TFRECORD,
            ego_pose=pose,
            expected_scenario_id="637f20cafde22ff8",
            expected_record_index=0,
            expected_time_index=10,
            expected_current_time_index=10,
            expected_record_sha256=sidecar["record_payload_sha256"],
        )
        self.assertEqual(result["verification_level"],
                         "fixed_scene_lane_rebuilt")
        self.assertIs(result["model_condition_injected"], False)
        self.assertEqual(len(result["tokens"]), 12)
        self.assertEqual(len(result["alignments"]), 30)
        observed = {case["raw_lane_id"]: case for case in e4b["lanes"]}
        anchors = {case["raw_lane_id"]: case for case in e4d["lanes"]}
        for lane_id in STOP_LANES:
            with self.subTest(lane_id=lane_id):
                self.assertIs(anchors[lane_id]["unique_anchor_verified"], True)
                token = _token(result, lane_id)
                self.assertIs(token["explicit_signal_observed"], True)
                self.assertIs(token["temporal_condition_available"], True)
                self.assertIs(token["anchor_alignment_consistent"], True)
                self.assertIs(token["anchor_alignment_verified"], True)
                self.assertIs(token["raw_lane_scope_known"], True)
                self.assertIs(token["new_interface_candidate"], True)
                self.assertIs(token["new_interface_verified"], False)
                self.assertIs(token["e5b_masks"]
                              ["tensor_control_identity_proven"], False)
                for computed, saved in zip(token["stop_point_ego_xy"],
                                           observed[lane_id]["stop_point_ego_xy"]):
                    self.assertAlmostEqual(computed, saved, places=8)
                rows = [entry for entry in result["alignments"]
                        if entry["raw_lane_id"] == lane_id]
                self.assertEqual(len(rows), 3)
                self.assertEqual(sum(entry["new_interface_candidate"]
                                     for entry in rows), 2)
                self.assertEqual(sum(entry["road_row_alignment_status"]
                                     == "filtered" for entry in rows), 1)
                self.assertTrue(all(not entry["new_interface_verified"]
                                    for entry in rows))
                self.assertTrue(all(not entry["e5b_masks"]
                                    ["tensor_control_identity_proven"]
                                    for entry in rows))
        with self.assertRaisesRegex(ValueError, "row_count"):
            simulate_interface(
                sidecar, evidence_base_dir=RESEARCH_DIR,
                real_tfrecord=DEFAULT_TFRECORD,
                row_permutations={"regular": {
                    "context": _context(sidecar), "row_count": 2,
                    "new_to_pre": [0, 1]}},
                expected_scenario_id="637f20cafde22ff8",
                expected_record_index=0, expected_time_index=10,
                expected_current_time_index=10,
                expected_record_sha256=sidecar["record_payload_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
