"""E5-C CPU-only simulation of independent signals and local road alignment.

This is an audit object, not an AE/LDM input. E5-B masks are copied unchanged.
No existing road row proves the original Stop identity in its final 20 points.
"""

import copy
import json
import math
from pathlib import Path

from .signal_candidate_projection import project_sidecar
from .signal_lane_sidecar import dumps_sidecar, loads_sidecar


INTERFACE_SCHEMA_NAME = "scenario-dreamer.signal-token-interface-simulation"
INTERFACE_SCHEMA_VERSION = "0.1.0"
_CONTEXT_FIELDS = ("scenario_id", "record_index", "time_index",
                   "record_payload_sha256")
_CLAIM_FIELDS = ("raw_lane_id", "group_id", "road_branch", "child_id",
                 "tensor_row", "source_point_range", "compact_point_range",
                 "row_index_stage")
_STOP_NAMES = frozenset(("STOP", "ARROW_STOP"))


def _finite_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _xy(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or
            any(not _finite_number(item) for item in value)):
        raise ValueError(f"{name} needs two finite coordinates")
    return [float(value[0]), float(value[1])]


def global_to_ego_xy(global_xy, center_global_xy, yaw_rad):
    """Match official subtract-center then SE(2) rotation by pi/2 - yaw."""
    point = _xy(global_xy, "global_xy")
    center = _xy(center_global_xy, "center_global_xy")
    if not _finite_number(yaw_rad):
        raise ValueError("yaw_rad must be finite")
    dx, dy = point[0] - center[0], point[1] - center[1]
    angle = math.pi / 2 - yaw_rad
    cosine, sine = math.cos(angle), math.sin(angle)
    return [cosine * dx - sine * dy, sine * dx + cosine * dy]


def _context(document):
    return {name: document[name] for name in _CONTEXT_FIELDS}


def _same_context(actual, expected):
    return (isinstance(actual, dict) and set(actual) == set(expected) and
            all(type(actual[name]) is type(value) and actual[name] == value
                for name, value in expected.items()))


def _validate_pose(pose, context):
    if (not isinstance(pose, dict) or
            not _same_context(pose.get("context"), context)):
        raise ValueError("Ego pose context differs from Sidecar")
    if (pose.get("source_frame") != "waymo_global_xyz" or
            pose.get("target_frame") != "waymo_ego_xy"):
        raise ValueError("Ego pose coordinate frames differ from official path")
    _xy(pose.get("center_global_xy"), "Ego center_global_xy")
    if not _finite_number(pose.get("yaw_rad")):
        raise ValueError("Ego pose yaw_rad must be finite")
    return copy.deepcopy(pose)


def _distance(left, right):
    return math.dist(_xy(left, "left XY"), _xy(right, "right XY"))


def _anchor_consistent(record):
    compact = record["compact"]
    if compact is None:
        return False
    anchor = record["anchor"]
    members = compact["ordered_member_lane_ids"]
    lane = record["raw_lane_id"]
    if (anchor["audit_status"] != "verified" or
            anchor["topology_verified"] is not True or
            anchor["selected_vertex_consistent"] is not True or
            anchor["unique_anchor_verified"] is not True or
            anchor["geometric_candidates_count"] != 1 or
            anchor["geometric_candidate_vertex_indices"] != [
                anchor["shared_compact_vertex_index"]] or
            anchor["applies_to_raw_lane_id"] != lane or lane not in members):
        return False
    member_index = members.index(lane)
    if (member_index == 0 or
            members[member_index - 1] != anchor["predecessor_lane_id"]):
        return False
    selected = anchor["shared_compact_vertex_index"]
    if selected != compact["compact_point_range"][0] - 1:
        return False
    tolerance = anchor["xy_tolerance_m"]
    return (tolerance is not None and
            _distance(record["signal"]["stop_point_global_xyz"][:2],
                      anchor["shared_vertex_global_xy"]) <= tolerance)


def _scope_structurally_consistent(record):
    compact = record["compact"]
    if compact is None:
        return False
    source = compact["source_point_range"]
    points = compact["compact_point_range"]
    return (record["raw_lane_id"] in compact["ordered_member_lane_ids"] and
            source[0] < source[1] and points[0] < points[1] and
            source[1] - source[0] == points[1] - points[0])


def _read_real_evidence(document, base_dir):
    """Load only hash-checked E4 files after project_sidecar has verified them."""
    paths = document["evidence_paths"]
    evidence = {}
    for key in ("e4a", "e4b", "e4d"):
        if key not in paths:
            raise ValueError(f"Missing {key} real alignment evidence")
        evidence[key] = json.loads((Path(base_dir) / paths[key]).read_text(
            encoding="utf-8"))
    expected = _context(document)
    for key in ("e4a", "e4b"):
        actual = {name: evidence[key]["input"][name]
                  for name in _CONTEXT_FIELDS}
        if actual != expected:
            raise ValueError(f"{key} Scenario/record/time/payload mismatch")
    d = evidence["e4d"]
    if ({name: d[name] for name in _CONTEXT_FIELDS} != expected or
            d["official_autodl_commit"] != document["official_repo_commit"]):
        raise ValueError("E4-D Scenario/record/time/payload/source mismatch")
    a_pose = (evidence["e4a"]["input"]["ego_t10_center"],
              evidence["e4a"]["input"]["ego_t10_yaw"])
    b_pose = (evidence["e4b"]["input"]["ego_t10_center"],
              evidence["e4b"]["input"]["ego_t10_yaw"])
    if a_pose != b_pose:
        raise ValueError("E4-A/B Ego poses differ")
    pose = {
        "context": expected,
        "source_frame": "waymo_global_xyz",
        "target_frame": "waymo_ego_xy",
        "center_global_xy": a_pose[0],
        "yaw_rad": a_pose[1],
    }
    return evidence, _validate_pose(pose, expected)


def _real_anchor_checks(record, evidence, pose):
    """Cross-check saved E4-A/B/D claims without treating row proximity as proof."""
    lane = record["raw_lane_id"]
    d_cases = {case["raw_lane_id"]: case for case in evidence["e4d"]["lanes"]}
    a_cases = {case["raw_lane_id"]: case for case in evidence["e4a"]["lanes"]}
    b_cases = {case["raw_lane_id"]: case for case in evidence["e4b"]["lanes"]}
    if lane not in d_cases or lane not in a_cases or lane not in b_cases:
        raise ValueError(f"Lane {lane} lacks fixed real E4-A/B/D evidence")
    a, b, d = a_cases[lane], b_cases[lane], d_cases[lane]
    compact, anchor, road = record["compact"], record["anchor"], record["road_tensor"]
    if (not _anchor_consistent(record) or
            a["group_id"] != compact["group_id"] or
            a["group_members"] != compact["ordered_member_lane_ids"] or
            a["source_point_range"] != compact["source_point_range"] or
            a["compact_point_range"] != compact["compact_point_range"] or
            a["regular_tensor_row"] != road["regular"]["tensor_row"] or
            b["group_id"] != compact["group_id"] or
            b["group_members"] != compact["ordered_member_lane_ids"] or
            b["compact_point_range"] != compact["compact_point_range"] or
            b["source_point_range"] != compact["source_point_range"] or
            d["group_id"] != compact["group_id"] or
            d["group_members"] != compact["ordered_member_lane_ids"] or
            d["predecessor_lane_id"] != anchor["predecessor_lane_id"] or
            d["shared_compact_vertex_index"] != anchor[
                "shared_compact_vertex_index"] or
            d["geometric_candidate_vertex_indices"] != anchor[
                "geometric_candidate_vertex_indices"] or
            not all(d[field] is True for field in (
                "topology_verified", "selected_vertex_consistent",
                "geometric_candidate_scan_performed", "unique_anchor_verified"))):
        raise ValueError(f"Lane {lane} E4 Compact/Anchor/Regular mismatch")
    stop = record["signal"]["stop_point_global_xyz"]
    xy_tolerance = anchor["xy_tolerance_m"]
    for point in (b["stop_point_global_xyz"][:2],
                  d["stop_point_global_xyz"][:2],
                  d["raw_downstream_first_xy"],
                  d["raw_predecessor_last_xy"],
                  d["compact_shared_vertex_global_xy"]):
        if _distance(stop[:2], point) > xy_tolerance:
            raise ValueError(f"Lane {lane} E4 global Stop/Anchor geometry mismatch")
    if (b["stop_point_global_xyz"] != stop or
            d["stop_point_global_xyz"] != stop or
            d["compact_shared_vertex_global_xy"] !=
            anchor["shared_vertex_global_xy"] or
            not b["stop_point_inside_fov"]):
        raise ValueError(f"Lane {lane} E4 Stop coordinate/FoV mismatch")
    b_rows = {(child["child_id"], child["tensor_row"])
              for child in b["partitioned"]}
    sidecar_rows = {(child["child_id"], child["tensor_row"])
                    for child in road["partitioned"]}
    if (b["regular"]["tensor_row"] != road["regular"]["tensor_row"] or
            b_rows != sidecar_rows):
        raise ValueError(f"Lane {lane} E4-B road rows differ")
    ego = global_to_ego_xy(stop[:2], pose["center_global_xy"],
                           pose["yaw_rad"])
    error = _distance(ego, b["stop_point_ego_xy"])
    if error > 1e-8:
        raise ValueError(f"Lane {lane} official Ego Stop transform mismatch")
    return error


def _validate_claims(claims, relations, context):
    if claims is None:
        return
    if not isinstance(claims, list):
        raise ValueError("alignment_claims must be a list")
    lookup = {(item["raw_lane_id"], item["road_branch"], item["child_id"]):
              item for item in relations}
    if len(lookup) != len(relations):
        raise ValueError("E5-B supplied duplicate relations")
    seen = set()
    for claim in claims:
        if (not isinstance(claim, dict) or
                any(field not in claim for field in _CLAIM_FIELDS) or
                not _same_context(claim.get("context"), context)):
            raise ValueError("Incomplete alignment claim")
        key = (claim["raw_lane_id"], claim["road_branch"], claim["child_id"])
        if key in seen or key not in lookup:
            raise ValueError("Duplicate or unknown Lane/branch/child claim")
        seen.add(key)
        reference = lookup[key]
        for field in _CLAIM_FIELDS:
            if (type(claim[field]) is not type(reference[field]) or
                    claim[field] != reference[field]):
                raise ValueError(f"Alignment claim {key} differs at {field}")


def _validate_permutations(permutations, relations, context,
                           official_row_counts=None):
    if permutations is None:
        return {}
    if not isinstance(permutations, dict) or set(permutations) - {
            "regular", "partitioned"}:
        raise ValueError("row_permutations needs branch-specific entries")
    inverses = {}
    for branch, entry in permutations.items():
        if (not isinstance(entry, dict) or
                not _same_context(entry.get("context"), context)):
            raise ValueError(f"{branch} permutation context mismatch")
        values = entry.get("new_to_pre")
        count = entry.get("row_count")
        if (type(count) is not int or count <= 0 or
                not isinstance(values, list) or len(values) != count or
                any(type(index) is not int for index in values) or
                sorted(values) != list(range(len(values)))):
            raise ValueError(f"{branch} permutation needs row_count and a full bijection")
        if (official_row_counts is not None and
                count != official_row_counts[branch]):
            raise ValueError(f"{branch} row_count differs from official E4-A")
        used_rows = [item["tensor_row"] for item in relations
                     if item["road_branch"] == branch and
                     item["tensor_row"] is not None]
        if used_rows and max(used_rows) >= len(values):
            raise ValueError(f"{branch} permutation omits a referenced row")
        inverses[branch] = {old: new for new, old in enumerate(values)}
    return inverses


def _row_alignment_status(relation, record):
    statuses = (record["fov"]["local_contribution_status"],
                record["fov"]["stop_point_coordinate_status"],
                relation["fov_stop_source_status"],
                relation["local_lane_source_status"],
                relation["row_stop_identity_status"],
                relation["final_stop_identity_status"])
    if "invalid" in statuses:
        return "invalid"
    if relation["tensor_row"] is None or "filtered" in statuses:
        return "filtered"
    if (record["anchor"]["audit_status"] == "not_run" or
            "not_run" in statuses[:-1]):
        return "not_run"
    # No complete final-20-point original Lane/Stop witness exists here.
    return "uncertain"


def simulate_interface(sidecar_source, *, evidence_base_dir=None,
                       real_tfrecord=None, ego_pose=None,
                       alignment_claims=None, row_permutations=None,
                       expected_scenario_id=None, expected_record_index=None,
                       expected_time_index=None,
                       expected_current_time_index=None,
                       expected_record_sha256=None):
    """Validate Sidecar, preserve E5-B masks, and emit offline E5-C hypotheses."""
    expectations = {
        "expected_scenario_id": expected_scenario_id,
        "expected_record_index": expected_record_index,
        "expected_time_index": expected_time_index,
        "expected_current_time_index": expected_current_time_index,
        "expected_record_sha256": expected_record_sha256,
    }
    projection = project_sidecar(
        sidecar_source, evidence_base_dir=evidence_base_dir,
        real_tfrecord=real_tfrecord, **expectations)
    source = (sidecar_source if isinstance(sidecar_source, str) else
              dumps_sidecar(sidecar_source))
    document = loads_sidecar(source, expected_scenario_id=expected_scenario_id,
                             expected_record_index=expected_record_index,
                             expected_time_index=expected_time_index,
                             expected_record_sha256=expected_record_sha256)
    context = _context(document)
    evidence = None
    official_pose = None
    if real_tfrecord is not None:
        evidence, official_pose = _read_real_evidence(document, evidence_base_dir)
    if ego_pose is not None:
        pose = _validate_pose(ego_pose, context)
        if official_pose is not None and (
                _distance(pose["center_global_xy"],
                          official_pose["center_global_xy"]) > 1e-9 or
                abs(pose["yaw_rad"] - official_pose["yaw_rad"]) > 1e-12):
            raise ValueError("Supplied Ego pose differs from official E4-A/B pose")
    else:
        pose = official_pose
    _validate_claims(alignment_claims, projection["candidate_relations"],
                     context)
    official_row_counts = None
    if evidence is not None:
        counts = evidence["e4a"]["counts"]
        official_row_counts = {
            "regular": counts["regular_tensor_rows"],
            "partitioned": counts["partitioned_tensor_rows"],
        }
        for branch in official_row_counts:
            if (official_row_counts[branch] !=
                    evidence["e4a"][branch]["num_lanes"]):
                raise ValueError(f"E4-A {branch} row counts differ")
    inverses = _validate_permutations(row_permutations,
                                      projection["candidate_relations"],
                                      context, official_row_counts)
    records = {record["raw_lane_id"]: record for record in
               document["signals"]}
    if len(records) != len(document["signals"]):
        raise ValueError("Duplicate LaneState in Sidecar")
    grouped_relations = {lane: [] for lane in records}
    for relation in projection["candidate_relations"]:
        grouped_relations[relation["raw_lane_id"]].append(relation)
    tokens = []
    alignments = []
    max_ego_error = None
    for projected_token in projection["tokens"]:
        lane = projected_token["raw_lane_id"]
        record = records[lane]
        signal = record["signal"]
        anchor_consistent = _anchor_consistent(record)
        scope_structural = _scope_structurally_consistent(record)
        scope_known = (scope_structural and projection[
            "verification_level"] == "fixed_scene_lane_rebuilt")
        anchor_verified = False
        if evidence is not None and signal["state_name"] in _STOP_NAMES:
            error = _real_anchor_checks(record, evidence, pose)
            max_ego_error = error if max_ego_error is None else max(max_ego_error,
                                                                    error)
            anchor_verified = True
        stop = signal["stop_point_global_xyz"]
        ego = (global_to_ego_xy(stop[:2], pose["center_global_xy"],
                                pose["yaw_rad"]) if pose is not None else None)
        lane_alignments = []
        for relation in grouped_relations[lane]:
            branch = relation["road_branch"]
            old_row = relation["tensor_row"]
            status = _row_alignment_status(relation, record)
            candidate = (anchor_consistent and scope_structural and
                         projected_token["masks"]["time_available"] and
                         signal["state_code"] != 0 and
                         record["fov"]["stop_point_coordinate_status"] in
                         ("retained", "verified") and
                         old_row is not None and status == "uncertain")
            alignment = {
                "raw_lane_id": lane,
                "group_id": relation["group_id"],
                "road_branch": branch,
                "child_id": relation["child_id"],
                "pre_shuffle_tensor_row": old_row,
                "simulated_post_shuffle_row": (
                    inverses[branch][old_row] if branch in inverses and
                    old_row is not None else None),
                "row_index_stage": relation["row_index_stage"],
                "source_point_range": copy.deepcopy(
                    relation["source_point_range"]),
                "compact_point_range": copy.deepcopy(
                    relation["compact_point_range"]),
                "parent_upsampled_point_range": copy.deepcopy(
                    relation["parent_upsampled_point_range"]),
                "stop_anchor": copy.deepcopy(relation["stop_anchor"]),
                "relation_status_e5b": relation["relation_status"],
                "road_row_alignment_status": status,
                "new_interface_candidate": candidate,
                "new_interface_verified": False,
                "source_uncertainty_reasons": copy.deepcopy(
                    relation["uncertainty_reasons"]),
                "e5b_masks": copy.deepcopy(relation["masks"]),
            }
            alignments.append(alignment)
            lane_alignments.append(alignment)
        tokens.append({
            "raw_lane_id": lane,
            "state_code": projected_token["state_code"],
            "state_name": projected_token["state_name"],
            "stop_point_global_xyz": copy.deepcopy(stop),
            "coordinate_frame": "waymo_global_xyz",
            "stop_point_ego_xy": ego,
            "ego_coordinate_frame": "waymo_ego_xy" if pose else None,
            "time_index": document["time_index"],
            "current_time_index": document["current_time_index"],
            "temporal_role": projected_token["temporal_role"],
            "mapping_audit_status": projected_token[
                "mapping_audit_status"],
            "explicit_signal_observed": True,
            "temporal_condition_available": projected_token["masks"][
                "time_available"],
            "anchor_audit_status": projected_token[
                "anchor_audit_status"],
            "anchor_unique_verified": projected_token[
                "anchor_unique_verified"],
            "anchor_alignment_consistent": anchor_consistent,
            "anchor_alignment_verified": anchor_verified,
            "raw_lane_scope_structurally_consistent": scope_structural,
            "raw_lane_scope_known": scope_known,
            "local_fov_status": projected_token["local_fov_status"],
            "stop_coordinate_fov_status": projected_token[
                "stop_coordinate_fov_status"],
            "stop_source_fov_status": projected_token[
                "stop_source_fov_status"],
            "final_stop_identity_status": projected_token[
                "final_stop_identity_status"],
            "new_interface_candidate": any(item["new_interface_candidate"]
                                           for item in lane_alignments),
            "new_interface_verified": False,
            "e5b_masks": copy.deepcopy(projected_token["masks"]),
        })
    return {
        "schema_name": INTERFACE_SCHEMA_NAME,
        "schema_version": INTERFACE_SCHEMA_VERSION,
        "source_sidecar_schema_version": document["schema_version"],
        "source_sidecar_sha256": projection["source_sidecar_sha256"],
        **context,
        "current_time_index": document["current_time_index"],
        "official_repo_commit": document["official_repo_commit"],
        "verification_level": projection["verification_level"],
        "coordinate_transform": {
            "source": ("official_e4a_e4b_t10_pose" if official_pose is not None
                       else "caller_supplied_pose" if pose is not None else
                       "not_run"),
            "pose": copy.deepcopy(pose),
            "max_e4b_ego_xy_error_m": max_ego_error,
            "xy_tolerance_m": 1e-8 if evidence is not None else None,
        },
        "row_index_stage": document["row_index_stage"],
        "simulated_permutation_supplied": sorted(inverses),
        "signal_absent_raw_lane_ids": copy.deepcopy(projection[
            "signal_absent_raw_lane_ids"]),
        "model_condition_injected": False,
        "tokens": tokens,
        "alignments": alignments,
    }
