"""Versioned, model-independent Signal–Lane Sidecar contract (E5-A).

The contract stores observations and evidence claims. It never infers a Stop
identity from a Group ID or tensor row and never writes production caches.
"""

import hashlib
import json
import math
from pathlib import Path
import re


SCHEMA_NAME = "scenario-dreamer.signal-lane-sidecar"
SCHEMA_VERSION = "1.0.0"
STATE_NAMES = {0: "UNKNOWN", 1: "ARROW_STOP", 4: "STOP"}
AUDIT_STATUSES = frozenset((
    "verified", "partial", "uncertain", "filtered", "invalid", "not_run"))
FOV_STATUSES = AUDIT_STATUSES | {"retained"}
ROW_STATUSES = frozenset(("retained", "filtered"))


def _field(mapping, name, path):
    if not isinstance(mapping, dict) or name not in mapping:
        raise ValueError(f"Missing {path}.{name}")
    return mapping[name]


def _nonnegative_int(value, path):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a nonnegative integer")
    return value


def _hex(value, length, path):
    if not isinstance(value, str) or re.fullmatch(
            rf"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError(f"{path} needs a {length}-character lowercase hex digest")


def _finite_vector(value, length, path):
    if (not isinstance(value, list) or len(value) != length or
            any(isinstance(item, bool) or not isinstance(item, (int, float)) or
                not math.isfinite(item) for item in value)):
        raise ValueError(f"{path} needs {length} finite coordinates")


def _point_range(value, path):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{path} needs a half-open point range")
    start, end = (_nonnegative_int(item, path) for item in value)
    if end < start:
        raise ValueError(f"{path} has reversed range")
    return start, end


def _status(value, allowed, path):
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"Invalid {path} status: {value!r}")


def _artifact_ref(refs, field, evidence, *, expected, required):
    value = _field(refs, field, "evidence_refs")
    if value is None:
        if required:
            raise ValueError(f"Missing {field} evidence reference")
        return
    if value != expected or value not in evidence:
        raise ValueError(f"Invalid {field} evidence role or digest")


def _validate_row(row, path):
    row_index = _field(row, "tensor_row", path)
    status = _field(row, "row_status", path)
    _status(status, ROW_STATUSES, f"{path}.row")
    if row_index is None:
        if status != "filtered":
            raise ValueError(f"{path}: missing row must be filtered")
    else:
        _nonnegative_int(row_index, f"{path}.tensor_row")
        if status != "retained":
            raise ValueError(f"{path}: present row must be retained")
    local_source = _field(row, "local_lane_source_status", path)
    stop_source = _field(row, "stop_identity_status", path)
    _status(local_source, AUDIT_STATUSES, f"{path}.local_lane_source")
    _status(stop_source, AUDIT_STATUSES, f"{path}.stop_identity")
    if status == "filtered" and (local_source not in ("filtered", "not_run")
                                 or stop_source not in ("filtered", "not_run")):
        raise ValueError(f"{path}: filtered row cannot claim retained source")


def _validate_signal(record, document, raw_ids, evidence):
    raw_id = _nonnegative_int(_field(record, "raw_lane_id", "signal"),
                              "signal.raw_lane_id")
    if raw_id not in raw_ids:
        raise ValueError(f"Signal references unknown raw Lane {raw_id}")
    context = _field(record, "context", f"signal[{raw_id}]")
    for field in ("scenario_id", "record_index", "time_index",
                  "record_payload_sha256"):
        if _field(context, field, f"signal[{raw_id}].context") != document[field]:
            raise ValueError(f"Signal {raw_id} context {field} differs")

    signal = _field(record, "signal", f"signal[{raw_id}]")
    if _field(signal, "presence", "signal") != "present":
        raise ValueError("A Sidecar signal record must have presence=present")
    code = _nonnegative_int(_field(signal, "state_code", "signal"),
                            "signal.state_code")
    if STATE_NAMES.get(code) != _field(signal, "state_name", "signal"):
        raise ValueError(f"Unsupported or mismatched signal state for Lane {raw_id}")
    stop = _field(signal, "stop_point_global_xyz", "signal")
    _finite_vector(stop, 3, "signal.stop_point_global_xyz")

    compact = _field(record, "compact", f"signal[{raw_id}]")
    group_id = _nonnegative_int(_field(compact, "group_id", "compact"),
                                "compact.group_id")
    members = _field(compact, "ordered_member_lane_ids", "compact")
    if (not isinstance(members, list) or not members or
            any(isinstance(member, bool) or not isinstance(member, int)
                for member in members) or len(set(members)) != len(members)):
        raise ValueError(f"Lane {raw_id} needs unique ordered group members")
    if any(member not in raw_ids for member in members):
        raise ValueError(f"Lane {raw_id} has an unknown group member")
    if raw_id not in members:
        raise ValueError(f"Raw Lane {raw_id} absent from its Group")
    source_start, source_end = _point_range(
        _field(compact, "source_point_range", "compact"),
        "compact.source_point_range")
    compact_start, compact_end = _point_range(
        _field(compact, "compact_point_range", "compact"),
        "compact.compact_point_range")
    member_index = members.index(raw_id)
    if (source_start != (0 if member_index == 0 else 1) or
            source_end - source_start != compact_end - compact_start):
        raise ValueError(f"Lane {raw_id} has inconsistent compact contribution range")

    anchor = _field(record, "anchor", f"signal[{raw_id}]")
    if _field(anchor, "applies_to_raw_lane_id", "anchor") != raw_id:
        raise ValueError(f"Lane {raw_id} anchor scope must remain raw-Lane-local")
    anchor_status = _field(anchor, "audit_status", "anchor")
    _status(anchor_status, AUDIT_STATUSES, "anchor.audit")
    candidate_keys = (
        "predecessor_lane_id", "shared_compact_vertex_index",
        "shared_vertex_global_xy", "selected_vertex_consistent",
        "topology_verified", "geometric_candidate_vertex_indices",
        "geometric_candidates_count", "unique_anchor_verified",
        "xy_tolerance_m")
    anchor_values = {key: _field(anchor, key, "anchor") for key in candidate_keys}
    if anchor_status == "not_run":
        if any(value is not None for value in anchor_values.values()):
            raise ValueError(f"Lane {raw_id} not_run anchor carries an audit claim")
    else:
        predecessor = anchor_values["predecessor_lane_id"]
        if predecessor is not None:
            _nonnegative_int(predecessor, "anchor.predecessor_lane_id")
            if predecessor not in raw_ids or predecessor not in members:
                raise ValueError(f"Lane {raw_id} predecessor is not a Group member")
        index = anchor_values["shared_compact_vertex_index"]
        if index is not None:
            _nonnegative_int(index, "anchor.shared_compact_vertex_index")
        xy = anchor_values["shared_vertex_global_xy"]
        if xy is not None:
            _finite_vector(xy, 2, "anchor.shared_vertex_global_xy")
        for key in ("selected_vertex_consistent", "topology_verified",
                    "unique_anchor_verified"):
            value = anchor_values[key]
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"anchor.{key} must be a boolean or null")
        candidates = anchor_values["geometric_candidate_vertex_indices"]
        count = anchor_values["geometric_candidates_count"]
        if count is not None:
            _nonnegative_int(count, "anchor.geometric_candidates_count")
        if candidates is not None:
            if not isinstance(candidates, list) or any(
                    isinstance(item, bool) or not isinstance(item, int) or item < 0
                    for item in candidates) or len(set(candidates)) != len(candidates):
                raise ValueError("anchor.geometric_candidate_vertex_indices invalid")
            if count != len(candidates):
                raise ValueError("anchor.geometric_candidates_count differs")
        elif count is not None:
            raise ValueError("anchor.geometric_candidates_count without a scan")
        tolerance = anchor_values["xy_tolerance_m"]
        if tolerance is not None and (
                isinstance(tolerance, bool) or
                not isinstance(tolerance, (int, float)) or
                not math.isfinite(tolerance) or tolerance <= 0):
            raise ValueError("anchor.xy_tolerance_m must be positive")
    if anchor_status == "verified":
        if member_index == 0 or compact_start == 0:
            raise ValueError(f"Lane {raw_id} has no Compact predecessor")
        if anchor_values["predecessor_lane_id"] != members[member_index - 1]:
            raise ValueError(f"Lane {raw_id} predecessor is not adjacent")
        index = anchor_values["shared_compact_vertex_index"]
        if index != compact_start - 1:
            raise ValueError(f"Lane {raw_id} selected Compact vertex differs")
        _finite_vector(anchor_values["shared_vertex_global_xy"], 2,
                       "anchor.shared_vertex_global_xy")
        tolerance = anchor_values["xy_tolerance_m"]
        if tolerance is None:
            raise ValueError("Verified anchor needs positive XY tolerance")
        xy = anchor_values["shared_vertex_global_xy"]
        if math.hypot(stop[0] - xy[0], stop[1] - xy[1]) > tolerance:
            raise ValueError(f"Lane {raw_id} Stop Point differs from shared anchor")
        if (anchor_values["selected_vertex_consistent"] is not True or
                anchor_values["topology_verified"] is not True or
                anchor_values["unique_anchor_verified"] is not True or
                anchor_values["geometric_candidate_vertex_indices"] != [index] or
                anchor_values["geometric_candidates_count"] != 1):
            raise ValueError(f"Lane {raw_id} lacks unique verified anchor geometry")
    elif anchor_values["unique_anchor_verified"] is True:
        raise ValueError(f"Lane {raw_id} anchor cannot claim uniqueness at {anchor_status}")

    fov = _field(record, "fov", f"signal[{raw_id}]")
    _status(_field(fov, "local_contribution_status", "fov"), FOV_STATUSES,
            "fov.local_contribution")
    stop_coordinate_fov = _field(fov, "stop_point_coordinate_status", "fov")
    _status(stop_coordinate_fov, FOV_STATUSES, "fov.stop_point_coordinate")
    stop_source_fov = _field(fov, "stop_source_status", "fov")
    _status(stop_source_fov, AUDIT_STATUSES, "fov.stop_source")
    if (stop_coordinate_fov == "not_run") != (stop_source_fov == "not_run"):
        raise ValueError("stop source and coordinate need the same audit scope")

    tensor = _field(record, "road_tensor", f"signal[{raw_id}]")
    regular = _field(tensor, "regular", "road_tensor")
    if _field(regular, "lane_id", "road_tensor.regular") != group_id:
        raise ValueError(f"Lane {raw_id} Regular lane ID differs from Group ID")
    _validate_row(regular, "road_tensor.regular")
    children = _field(tensor, "partitioned", "road_tensor")
    if not isinstance(children, list) or not children:
        raise ValueError(f"Lane {raw_id} needs Partitioned child records")
    seen_children = set()
    for child in children:
        child_id = _nonnegative_int(_field(child, "child_id", "partitioned"),
                                    "partitioned.child_id")
        if child_id in seen_children:
            raise ValueError(f"Lane {raw_id} repeats Partitioned child ID")
        seen_children.add(child_id)
        _point_range(_field(child, "parent_upsampled_point_range", "partitioned"),
                     "partitioned.parent_upsampled_point_range")
        _validate_row(child, f"partitioned[{child_id}]")
    final_status = _field(tensor, "final_stop_identity_status", "road_tensor")
    _status(final_status, AUDIT_STATUSES, "road_tensor.final_stop_identity")
    final_verified = _field(tensor, "final_tensor_stop_provenance_verified",
                            "road_tensor")
    if final_status == "verified":
        if final_verified is not True:
            raise ValueError("Verified final Stop identity needs true provenance")
    elif final_verified is not None:
        raise ValueError("Nonverified final Stop identity must preserve null provenance")
    if anchor_status == "not_run" and (
            stop_coordinate_fov != "not_run" or
            stop_source_fov != "not_run" or final_status != "not_run" or
            regular["stop_identity_status"] != "not_run" or
            any(child["stop_identity_status"] != "not_run" for child in children)):
        raise ValueError(f"Lane {raw_id} not_run Stop audit cannot inherit row status")

    refs = _field(record, "evidence_refs", f"signal[{raw_id}]")
    if _field(refs, "raw_signal", "evidence_refs") != "record_payload":
        raise ValueError("Missing raw signal evidence reference")
    if _field(refs, "group_and_rows", "evidence_refs") != "e4a":
        raise ValueError("Missing E4-A group/row evidence reference")
    _artifact_ref(refs, "group_and_rows", evidence,
                  expected="e4a", required=True)
    _artifact_ref(refs, "anchor", evidence,
                  expected="e4d",
                  required=anchor_status != "not_run")
    if anchor_status == "not_run" and refs["anchor"] is not None:
        raise ValueError("not_run anchor cannot cite verification evidence")
    _artifact_ref(refs, "stop_fov", evidence, expected="e4b",
                  required=stop_source_fov != "not_run")
    if stop_source_fov == "not_run" and refs["stop_fov"] is not None:
        raise ValueError("not_run Stop FoV cannot cite verification evidence")
    _artifact_ref(refs, "final_stop", evidence, expected="e4c",
                  required=final_status != "not_run")
    if final_status == "not_run" and refs["final_stop"] is not None:
        raise ValueError("not_run final Stop cannot cite verification evidence")


def validate_sidecar(document, *, expected_scenario_id=None,
                     expected_record_index=None, expected_time_index=None,
                     expected_record_sha256=None):
    """Reject mixed contexts, unsupported semantics and unsupported claims."""
    if _field(document, "schema_name", "sidecar") != SCHEMA_NAME:
        raise ValueError("Unsupported Sidecar schema_name")
    if _field(document, "schema_version", "sidecar") != SCHEMA_VERSION:
        raise ValueError("Unsupported Sidecar schema_version")
    scenario_id = _field(document, "scenario_id", "sidecar")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario_id must be a nonempty string")
    _nonnegative_int(_field(document, "record_index", "sidecar"),
                     "record_index")
    _nonnegative_int(_field(document, "time_index", "sidecar"), "time_index")
    current_time_index = _nonnegative_int(
        _field(document, "current_time_index", "sidecar"),
        "current_time_index")
    if current_time_index != document["time_index"]:
        raise ValueError("current_time_index differs from audited time_index")
    _hex(_field(document, "record_payload_sha256", "sidecar"), 64,
         "record_payload_sha256")
    _hex(_field(document, "official_repo_commit", "sidecar"), 40,
         "official_repo_commit")
    if _field(document, "coordinate_frame", "sidecar") != "waymo_global_xyz":
        raise ValueError("Sidecar needs waymo_global_xyz coordinates")
    if _field(document, "row_index_stage", "sidecar") != "pre_training_random_permutation":
        raise ValueError("Road rows must identify their pre-shuffle stage")
    if _field(document, "model_condition_injected", "sidecar") is not False:
        raise ValueError("model_condition_injected must be false in Sidecar v1")
    expected = {
        "scenario_id": expected_scenario_id,
        "record_index": expected_record_index,
        "time_index": expected_time_index,
        "record_payload_sha256": expected_record_sha256,
    }
    for field, value in expected.items():
        if value is not None and document[field] != value:
            raise ValueError(f"Sidecar {field} differs from expected context")
    sources = _field(document, "official_source_blobs", "sidecar")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("official_source_blobs must be nonempty")
    for name, digest in sources.items():
        if not isinstance(name, str) or not name:
            raise ValueError("official_source_blobs has empty source name")
        _hex(digest, 40, f"official_source_blobs.{name}")
    evidence = _field(document, "evidence_sha256", "sidecar")
    if not isinstance(evidence, dict) or "e4a" not in evidence:
        raise ValueError("Missing E4-A evidence digest")
    for name, digest in evidence.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Invalid evidence artifact name")
        _hex(digest, 64, f"evidence_sha256.{name}")
    evidence_paths = _field(document, "evidence_paths", "sidecar")
    if not isinstance(evidence_paths, dict) or set(evidence_paths) != set(evidence):
        raise ValueError("Each evidence digest needs an evidence path")
    for name, path in evidence_paths.items():
        if (not isinstance(path, str) or not path or path.startswith(("/", "\\"))
                or ":" in path or "\\" in path or
                ".." in path.split("/")):
            raise ValueError(f"Invalid evidence path for {name}")
    raw_lane_ids = _field(document, "raw_lane_ids", "sidecar")
    if (not isinstance(raw_lane_ids, list) or
            any(isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in raw_lane_ids) or
            len(raw_lane_ids) != len(set(raw_lane_ids))):
        raise ValueError("raw_lane_ids must be unique nonnegative Lane IDs")
    records = _field(document, "signals", "sidecar")
    if not isinstance(records, list):
        raise ValueError("signals must be a list of observed LaneStates")
    seen = set()
    group_signatures = {}
    raw_ids = set(raw_lane_ids)
    for record in records:
        _validate_signal(record, document, raw_ids, evidence)
        raw_id = record["raw_lane_id"]
        if raw_id in seen:
            raise ValueError(f"Duplicate raw signal Lane {raw_id}")
        seen.add(raw_id)
        group_id = record["compact"]["group_id"]
        regular = record["road_tensor"]["regular"]
        children = record["road_tensor"]["partitioned"]
        signature = (
            tuple(record["compact"]["ordered_member_lane_ids"]),
            (regular["lane_id"], regular["tensor_row"], regular["row_status"]),
            tuple((child["child_id"],
                   tuple(child["parent_upsampled_point_range"]),
                   child["tensor_row"], child["row_status"])
                  for child in children),
        )
        if group_id in group_signatures and signature != group_signatures[group_id]:
            raise ValueError(f"Group {group_id} has inconsistent members or rows")
        group_signatures[group_id] = signature


def dumps_sidecar(document):
    """Serialize a valid Sidecar without changing evidence or null states."""
    validate_sidecar(document)
    return json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False) + "\n"


def _unique_pairs(pairs):
    mapping = {}
    for key, value in pairs:
        if key in mapping:
            raise ValueError(f"Duplicate JSON key: {key}")
        mapping[key] = value
    return mapping


def _invalid_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def loads_sidecar(text, *, expected_scenario_id=None,
                  expected_record_index=None, expected_time_index=None,
                  expected_record_sha256=None):
    """Deserialize, validate and optionally pin the expected scenario."""
    document = json.loads(text, object_pairs_hook=_unique_pairs,
                          parse_constant=_invalid_constant)
    validate_sidecar(document,
                     expected_scenario_id=expected_scenario_id,
                     expected_record_index=expected_record_index,
                     expected_time_index=expected_time_index,
                     expected_record_sha256=expected_record_sha256)
    return document


def signal_for_lane(document, lane_id):
    """Return an observed LaneState; ``None`` means no signal record."""
    validate_sidecar(document)
    lane_id = _nonnegative_int(lane_id, "raw Lane ID")
    if lane_id not in document["raw_lane_ids"]:
        raise ValueError(f"Unknown raw Lane {lane_id}")
    return next((record for record in document["signals"]
                 if record["raw_lane_id"] == lane_id), None)


def verify_evidence_files(document, base_dir):
    """Rehash referenced files; ``loads_sidecar`` alone checks structure only."""
    validate_sidecar(document)
    base = Path(base_dir).absolute()
    digests = {}
    for artifact, relative_path in document["evidence_paths"].items():
        path = base / relative_path
        if not path.is_file():
            raise ValueError(f"{artifact} evidence file is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != document["evidence_sha256"][artifact]:
            raise ValueError(f"{artifact} evidence digest differs: {path}")
        digests[artifact] = digest
    return digests
