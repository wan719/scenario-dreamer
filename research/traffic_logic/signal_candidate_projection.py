"""Read-only E5-B projection of Sidecar observations to road-row candidates.

This module records candidate associations, not model inputs. In particular,
neither a Compact Group nor an extant road row proves original Stop identity.
"""

import hashlib

from .signal_lane_sidecar import (dumps_sidecar, loads_sidecar,
                                  verify_evidence_files)


PROJECTION_SCHEMA_NAME = "scenario-dreamer.signal-road-candidate-projection"
PROJECTION_SCHEMA_VERSION = "0.1.0"
_TRACEABLE_LOCAL_STATUSES = frozenset(("retained", "verified"))
_IN_FOV_STATUSES = frozenset(("retained", "verified"))
_UNSUPPORTED_STATUSES = frozenset(("filtered", "invalid", "not_run"))


def _relation_masks(document, record, row, relation_status, verification_level):
    """Keep seven audit questions separate from candidate and proof gates."""
    anchor = record["anchor"]
    fov = record["fov"]
    road = record["road_tensor"]
    time_available = (document["available_at_observation_cutoff"]
                      if document["schema_version"] == "1.1.0" else True)
    state_known = record["signal"]["state_code"] != 0
    anchor_unique = (anchor["audit_status"] == "verified" and
                     anchor["unique_anchor_verified"] is True)
    stop_in_fov = fov["stop_point_coordinate_status"] in _IN_FOV_STATUSES
    row_exists = row["tensor_row"] is not None
    local_traceable = (row_exists and row["local_lane_source_status"]
                       in _TRACEABLE_LOCAL_STATUSES)
    final_claim = (row_exists and row["stop_identity_status"] == "verified" and
                   road["final_stop_identity_status"] == "verified" and
                   road["final_tensor_stop_provenance_verified"] is True)
    final_verified = (final_claim and anchor_unique and stop_in_fov and
                      fov["stop_source_status"] == "verified" and
                      verification_level == "fixed_scene_lane_rebuilt")
    candidate_eligible = (time_available and state_known and anchor_unique and
                          stop_in_fov and row_exists and
                          relation_status == "candidate_only")
    control_proven = local_traceable and final_verified
    condition_usable = candidate_eligible and control_proven
    return {
        "time_available": time_available,
        "state_known": state_known,
        "anchor_unique_verified": anchor_unique,
        "stop_coordinate_in_fov": stop_in_fov,
        "candidate_road_row_exists": row_exists,
        "local_lane_source_traceable": local_traceable,
        "final_stop_identity_verified": final_verified,
        "candidate_relation_eligible": candidate_eligible,
        "condition_candidate_usable": condition_usable,
        "tensor_control_identity_proven": control_proven,
    }


def _relation_status(record, row):
    if row["tensor_row"] is None:
        return "filtered"
    if (record["anchor"]["audit_status"] != "verified" or
            record["fov"]["stop_point_coordinate_status"] not in
            _IN_FOV_STATUSES or
            record["fov"]["stop_source_status"] in _UNSUPPORTED_STATUSES or
            row["local_lane_source_status"] in _UNSUPPORTED_STATUSES or
            row["stop_identity_status"] in _UNSUPPORTED_STATUSES or
            record["road_tensor"]["final_stop_identity_status"] in
            ("filtered", "invalid", "not_run")):
        return "unsupported"
    return "candidate_only"


def _uncertainty_reasons(record, row, relation_status, masks):
    reasons = []
    if not masks["time_available"]:
        reasons.append("future_label_unavailable_at_observation_cutoff")
    if not masks["state_known"]:
        reasons.append("explicit_unknown_signal_state")
    if not masks["anchor_unique_verified"]:
        reasons.append("unique_stop_anchor_not_verified")
    if not masks["stop_coordinate_in_fov"]:
        reasons.append("stop_coordinate_not_confirmed_in_fov")
    if not masks["candidate_road_row_exists"]:
        reasons.append("road_row_filtered")
    for field, status in (
            ("fov_stop_source", record["fov"]["stop_source_status"]),
            ("local_lane_source", row["local_lane_source_status"]),
            ("row_stop_identity", row["stop_identity_status"]),
            ("final_stop_identity",
             record["road_tensor"]["final_stop_identity_status"])):
        if status != "verified" and status != "retained":
            reasons.append(f"{field}_{status}")
    if relation_status == "candidate_only" and not masks[
            "final_stop_identity_verified"]:
        reasons.append("candidate_row_does_not_prove_stop_identity")
    return reasons


def _make_relation(document, record, branch, row, verification_level):
    compact = record["compact"]
    anchor = record["anchor"]
    relation_status = _relation_status(record, row)
    masks = _relation_masks(document, record, row, relation_status,
                            verification_level)
    return {
        "raw_lane_id": record["raw_lane_id"],
        "state_code": record["signal"]["state_code"],
        "state_name": record["signal"]["state_name"],
        "group_id": compact["group_id"],
        "road_branch": branch,
        "tensor_row": row["tensor_row"],
        "child_id": row["child_id"] if branch == "partitioned" else None,
        "source_point_range": compact["source_point_range"],
        "compact_point_range": compact["compact_point_range"],
        "parent_upsampled_point_range": (
            row["parent_upsampled_point_range"]
            if branch == "partitioned" else None),
        "stop_anchor": {
            "audit_status": anchor["audit_status"],
            "shared_compact_vertex_index": anchor[
                "shared_compact_vertex_index"],
            "shared_vertex_global_xy": anchor["shared_vertex_global_xy"],
            "unique_anchor_verified": anchor["unique_anchor_verified"],
        },
        "row_status": row["row_status"],
        "local_lane_source_status": row["local_lane_source_status"],
        "row_stop_identity_status": row["stop_identity_status"],
        "fov_stop_source_status": record["fov"]["stop_source_status"],
        "final_stop_identity_status": record["road_tensor"][
            "final_stop_identity_status"],
        "relation_status": relation_status,
        "uncertainty_reasons": _uncertainty_reasons(
            record, row, relation_status, masks),
        "evidence_verification_level": verification_level,
        "row_index_stage": document["row_index_stage"],
        "masks": masks,
    }


def _token_masks(document, record, relations):
    time_available = (document["available_at_observation_cutoff"]
                      if document["schema_version"] == "1.1.0" else True)
    mapped = record["compact"] is not None
    anchor = record["anchor"] if mapped else None
    fov = record["fov"] if mapped else None
    return {
        "time_available": time_available,
        "state_known": record["signal"]["state_code"] != 0,
        "anchor_unique_verified": (anchor is not None and
                                   anchor["audit_status"] == "verified" and
                                   anchor["unique_anchor_verified"] is True),
        "stop_coordinate_in_fov": (fov is not None and
                                   fov["stop_point_coordinate_status"]
                                   in _IN_FOV_STATUSES),
        "candidate_road_row_exists": any(
            relation["masks"]["candidate_road_row_exists"]
            for relation in relations),
        "local_lane_source_traceable": any(
            relation["masks"]["local_lane_source_traceable"]
            for relation in relations),
        "final_stop_identity_verified": any(
            relation["masks"]["final_stop_identity_verified"]
            for relation in relations),
        "candidate_relation_eligible": any(
            relation["masks"]["candidate_relation_eligible"]
            for relation in relations),
        "condition_candidate_usable": any(
            relation["masks"]["condition_candidate_usable"]
            for relation in relations),
        "tensor_control_identity_proven": any(
            relation["masks"]["tensor_control_identity_proven"]
            for relation in relations),
    }


def _make_token(document, record, relations, verification_level):
    mapped = record["compact"] is not None
    anchor = record["anchor"] if mapped else None
    fov = record["fov"] if mapped else None
    return {
        "raw_lane_id": record["raw_lane_id"],
        "state_code": record["signal"]["state_code"],
        "state_name": record["signal"]["state_name"],
        "signal_observed": True,
        "stop_point_global_xyz": record["signal"]["stop_point_global_xyz"],
        "coordinate_frame": document["coordinate_frame"],
        "time_index": document["time_index"],
        "current_time_index": document["current_time_index"],
        "temporal_role": (document["temporal_role"]
                          if document["schema_version"] == "1.1.0"
                          else "current"),
        "mapping_audit_status": (record["mapping_audit_status"]
                                 if document["schema_version"] == "1.1.0"
                                 else "legacy_recorded"),
        "anchor_audit_status": anchor["audit_status"] if mapped else "not_run",
        "anchor_unique_verified": (anchor["unique_anchor_verified"]
                                   if mapped else None),
        "local_fov_status": (fov["local_contribution_status"]
                             if mapped else "not_run"),
        "stop_coordinate_fov_status": (fov["stop_point_coordinate_status"]
                                       if mapped else "not_run"),
        "stop_source_fov_status": (fov["stop_source_status"]
                                   if mapped else "not_run"),
        "final_stop_identity_status": (
            record["road_tensor"]["final_stop_identity_status"]
            if mapped else "not_run"),
        "evidence_verification_level": verification_level,
        "masks": _token_masks(document, record, relations),
    }


def project_sidecar(source, *, expected_scenario_id=None,
                    expected_record_index=None, expected_time_index=None,
                    expected_current_time_index=None,
                    expected_record_sha256=None, evidence_base_dir=None,
                    real_tfrecord=None):
    """Validate and project Sidecar v1.0/v1.1 without altering road tensors.

    Supplying evidence_base_dir checks referenced file hashes. Supplying both
    evidence_base_dir and real_tfrecord additionally invokes the fixed t=10
    per-Lane E5-A rebuild; this stronger level is unavailable generically.
    """
    expected = {
        "expected_scenario_id": expected_scenario_id,
        "expected_record_index": expected_record_index,
        "expected_time_index": expected_time_index,
        "expected_record_sha256": expected_record_sha256,
    }
    if isinstance(source, str):
        document = loads_sidecar(source, **expected)
    elif isinstance(source, dict):
        document = loads_sidecar(dumps_sidecar(source), **expected)
    else:
        raise ValueError("source must be a Sidecar JSON string or dict")
    if (expected_current_time_index is not None and
            document["current_time_index"] != expected_current_time_index):
        raise ValueError("Sidecar current_time_index differs from expected context")
    if real_tfrecord is not None and evidence_base_dir is None:
        raise ValueError("real_tfrecord needs evidence_base_dir")
    verification_level = "structure_only"
    if evidence_base_dir is not None:
        if not document["evidence_paths"]:
            raise ValueError("No evidence files to check for this Sidecar")
        verify_evidence_files(document, evidence_base_dir)
        verification_level = "evidence_files_consistent"
    if real_tfrecord is not None:
        from .e5a_real_sidecar import verify_real_sidecar

        verify_real_sidecar(document, real_tfrecord)
        verification_level = "fixed_scene_lane_rebuilt"

    relations = []
    tokens = []
    for record in document["signals"]:
        lane_relations = []
        if record["compact"] is not None:
            road = record["road_tensor"]
            lane_relations.append(_make_relation(
                document, record, "regular", road["regular"],
                verification_level))
            for child in road["partitioned"]:
                lane_relations.append(_make_relation(
                    document, record, "partitioned", child,
                    verification_level))
        relations.extend(lane_relations)
        tokens.append(_make_token(document, record, lane_relations,
                                  verification_level))
    canonical = dumps_sidecar(document).encode("utf-8")
    observed_lane_ids = {record["raw_lane_id"] for record in document["signals"]}
    return {
        "schema_name": PROJECTION_SCHEMA_NAME,
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "source_sidecar_schema_version": document["schema_version"],
        "source_sidecar_sha256": hashlib.sha256(canonical).hexdigest(),
        "scenario_id": document["scenario_id"],
        "record_index": document["record_index"],
        "time_index": document["time_index"],
        "current_time_index": document["current_time_index"],
        "record_payload_sha256": document["record_payload_sha256"],
        "official_repo_commit": document["official_repo_commit"],
        "raw_lane_ids": document["raw_lane_ids"],
        "signal_absent_raw_lane_ids": [
            lane_id for lane_id in document["raw_lane_ids"]
            if lane_id not in observed_lane_ids],
        "row_index_stage": document["row_index_stage"],
        "verification_level": verification_level,
        "model_condition_injected": False,
        "tokens": tokens,
        "candidate_relations": relations,
    }
