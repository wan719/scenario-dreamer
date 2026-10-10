"""Build one read-only E5-A Sidecar from raw record 0 and saved E4 evidence.

No Compact/FoV/Partition algorithm is rerun here. Earlier research outputs
retain their own provenance; this builder checks their scenario, hashes and
lane-level claims before copying them into a standalone data contract.
"""

import argparse
import hashlib
import json
from pathlib import Path

from .e4d_anchor_geometry import parse_stop_points
from .e4d_real_anchor_diagnostic import (
    DEFAULT_TFRECORD, RECORD_SHA256, SCENARIO_ID, SOURCE_COMMIT,
    e3c_baselines,
)
from .real_scenario_helpers import parse_scenario, read_tfrecord_record
from .signal_lane_sidecar import (SCHEMA_NAME, SCHEMA_VERSION,
                                  LEGACY_STATE_NAMES as STATE_NAMES,
                                  dumps_sidecar, loads_sidecar,
                                  verify_evidence_files)


RESEARCH_DIR = Path(__file__).absolute().parent
OUTPUT_DIR = RESEARCH_DIR / "output"
EVIDENCE_PATHS = {
    "e3c": "output/e3c_official_crosscheck_report.txt",
    "e4a": "output/e4a_autodl/e4a_result.json",
    "e4b": "output/e4b_autodl/e4b_result.json",
    "e4c": "output/e4c_anchor_records.json",
    "e4d": "output/e4d_result.json",
}


def _unique_lanes(cases, label):
    ids = [case["raw_lane_id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} has duplicate raw Lane IDs")
    return {case["raw_lane_id"]: case for case in cases}


def _checked_documents(scenario, payload_hash):
    source_bytes = {key: (RESEARCH_DIR / path).read_bytes()
                    for key, path in EVIDENCE_PATHS.items()}
    hashes = {key: hashlib.sha256(value).hexdigest()
              for key, value in source_bytes.items()}
    a, b, c, d = (json.loads(source_bytes[key]) for key in
                  ("e4a", "e4b", "e4c", "e4d"))
    context = (SCENARIO_ID, 0, 10, payload_hash)
    if payload_hash != RECORD_SHA256:
        raise ValueError("Record payload differs from E3-C/E4 baseline")
    if (scenario["scenario_id"], scenario["current_time_index"]) != (
            SCENARIO_ID, 10):
        raise ValueError("Raw Scenario ID or current_time_index differs")
    for name, doc in (("E4-A", a), ("E4-B", b)):
        source = doc["input"]
        actual = (source["scenario_id"], source["record_index"],
                  source["time_index"], source["record_payload_sha256"])
        if actual != context or doc["run_status"] != "pass":
            raise ValueError(f"{name} context or run status differs")
        if doc["environment"]["repo_commit"] != SOURCE_COMMIT:
            raise ValueError(f"{name} official source commit differs")
        raw_states = {str(lane_id): code for lane_id, code in
                      scenario["lane_states"].items()}
        if source["signal_states"] != raw_states:
            raise ValueError(f"{name} raw signal state inventory differs")
    if (c["scenario_id"], c["record_index"], c["time_index"],
            c["record_payload_sha256"], c["official_repo_commit"]) != (
                *context, SOURCE_COMMIT):
        raise ValueError("E4-C context/source differs")
    if (d["scenario_id"], d["record_index"], d["time_index"],
            d["record_payload_sha256"], d["official_autodl_commit"]) != (
                *context, SOURCE_COMMIT) or d["run_status"] != "pass":
        raise ValueError("E4-D context/source/status differs")
    if (c["e4b_json_sha256"] != hashes["e4b"] or
            d["e4b_json_sha256"] != hashes["e4b"] or
            d["e4c_json_sha256"] != hashes["e4c"]):
        raise ValueError("E4 evidence SHA-256 chain differs")
    if a["environment"]["official_source_blobs"] != b["environment"][
            "official_source_blobs"]:
        raise ValueError("E4-A/B official source blobs differ")
    for name, digest in d["official_source_blobs_local"].items():
        if a["environment"]["official_source_blobs"].get(name) != digest:
            raise ValueError(f"E4-D official source blob differs: {name}")
    report = source_bytes["e3c"].decode("utf-8")
    if payload_hash not in report or SOURCE_COMMIT[:7] not in report:
        raise ValueError("E3-C record/source fingerprint differs")
    return a, b, c, d, hashes


def build_real_sidecar(tfrecord=DEFAULT_TFRECORD):
    """Create one validated Sidecar in memory from fixed t=10 evidence."""
    payload = read_tfrecord_record(tfrecord, 0)
    payload_hash = hashlib.sha256(payload).hexdigest()
    scenario = parse_scenario(payload, 10)
    a, b, c, d, evidence_hashes = _checked_documents(scenario, payload_hash)
    states = scenario["lane_states"]
    if len(states) != 12 or any(code not in STATE_NAMES for code in states.values()):
        raise ValueError("Expected exactly 12 supported raw LaneState enums")
    raw_stop_points = parse_stop_points(payload, 10, states.keys())
    a_cases = _unique_lanes(a["lanes"], "E4-A")
    b_cases = _unique_lanes(b["lanes"], "E4-B")
    c_cases = _unique_lanes(c["anchors"], "E4-C")
    d_cases = _unique_lanes(d["lanes"], "E4-D")
    stop_ids = {lane_id for lane_id, code in states.items() if code in (1, 4)}
    if (set(a_cases) != set(states) | {529} or
            set(b_cases) != stop_ids or set(c_cases) != stop_ids or
            set(d_cases) != stop_ids):
        raise ValueError("E4 evidence Lane inventories differ")
    e3c = e3c_baselines(RESEARCH_DIR / EVIDENCE_PATHS["e3c"])
    if set(e3c) != stop_ids:
        raise ValueError("E3-C Stop Lane inventory differs")
    if a["counts"]["raw_lanes"] != len(scenario["raw_lanes"]):
        raise ValueError("E4-A raw Lane count differs")
    if (a["counts"]["raw_lanes"], a["counts"]["processed_lanes"],
            a["counts"]["compact_groups"]) != (199, 198, 135):
        raise ValueError("E4-A Compact counts differ from E3-C")

    context = {
        "scenario_id": SCENARIO_ID,
        "record_index": 0,
        "time_index": 10,
        "record_payload_sha256": payload_hash,
    }
    regular_rows = a["regular"]["lane_id_to_tensor_row"]
    partition_rows = a["partitioned"]["lane_id_to_tensor_row"]
    records = []
    for lane_id in sorted(states):
        code = states[lane_id]
        case = a_cases[lane_id]
        stop = raw_stop_points[lane_id]
        if (stop["state_code"] != code or
                case["signal_state"] != STATE_NAMES[code]):
            raise ValueError(f"Lane {lane_id} raw/E4-A signal state differs")
        group_id = case["group_id"]
        members = case["group_members"]
        if (lane_id not in members or
                case["source_point_range"][1] != len(
                    scenario["raw_lanes"][lane_id]["polyline"])):
            raise ValueError(f"Lane {lane_id} Group or raw point range differs")
        regular_row = regular_rows.get(str(group_id))
        if case["regular_tensor_row"] != regular_row:
            raise ValueError(f"Lane {lane_id} Regular row differs from E4-A order")
        plan = a["partition_parent_children"][str(group_id)]["children"]
        local_children = {child["child_id"]: child
                          for child in case["partition_children"]}
        planned_ids = {child_id for child_id, _ in plan}
        if not set(local_children).issubset(planned_ids):
            raise ValueError(f"Lane {lane_id} local Partition children differ")
        b_case = b_cases.get(lane_id)
        b_children = ({child["child_id"]: child for child in b_case["partitioned"]}
                      if b_case else {})
        if b_case and set(b_children) != planned_ids:
            raise ValueError(f"Lane {lane_id} E4-B Partition children differ")
        if b_case and (b_case["signal_state"] != STATE_NAMES[code] or
                       b_case["source_point_range"] !=
                       case["source_point_range"] or
                       b_case["compact_point_range"] !=
                       case["compact_point_range"]):
            raise ValueError(f"Lane {lane_id} E4-A/B Lane interval differs")
        partitioned = []
        for child_id, parent_range in plan:
            row = partition_rows.get(str(child_id))
            if child_id in local_children and (
                    local_children[child_id]["tensor_row"] != row or
                    local_children[child_id][
                        "parent_upsampled_point_range"] != parent_range):
                raise ValueError(f"Lane {lane_id} local Partition row differs")
            if b_case and (
                    b_children[child_id]["tensor_row"] != row or
                    b_children[child_id][
                        "parent_1000_point_range"] != parent_range):
                raise ValueError(f"Lane {lane_id} Stop-side Partition row differs")
            partitioned.append({
                "child_id": child_id,
                "parent_upsampled_point_range": parent_range,
                "tensor_row": row,
                "row_status": "retained" if row is not None else "filtered",
                "local_lane_source_status": (
                    case["partitioned_input_status"] if child_id in local_children
                    else "filtered"),
                "stop_identity_status": (
                    b_children[child_id]["stop_source_status"] if b_case
                    else "not_run"),
            })
        if b_case:
            c_case, d_case, e3c_case = (c_cases[lane_id], d_cases[lane_id],
                                        e3c[lane_id])
            point = list(stop["stop_point_global_xyz"])
            if (b_case["signal_state_code"] != code or
                    c_case["signal_state"] != STATE_NAMES[code] or
                    d_case["signal_state_code"] != code or
                    d_case["signal_state"] != STATE_NAMES[code] or
                    b_case["stop_point_global_xyz"] != point or
                    c_case["stop_point"] != point or
                    d_case["stop_point_global_xyz"] != point):
                raise ValueError(f"Lane {lane_id} Stop XYZ/state differs")
            if (not d_case["baseline_consistent"] or
                    d_case["e4d_verification_status"] != "verified" or
                    not d_case["geometric_candidate_scan_performed"] or
                    not d_case["selected_vertex_consistent"] or
                    not d_case["topology_verified"] or
                    not d_case["unique_anchor_verified"] or
                    d_case["geometric_candidates_count"] != 1 or
                    d_case["geometric_candidate_vertex_indices"] != [
                        d_case["shared_compact_vertex_index"]]):
                raise ValueError(f"Lane {lane_id} lacks E4-D unique Anchor proof")
            if (group_id != b_case["group_id"] or
                    group_id != c_case["group_id"] or
                    group_id != d_case["group_id"] or
                    group_id != e3c_case["group_id"] or
                    members != b_case["group_members"] or
                    members != d_case["group_members"] or
                    members != e3c_case["group_members"] or
                    d_case["compact_geometry_sha256"] !=
                    e3c_case["compact_geometry_sha256"] or
                    d_case["compact_point_count"] !=
                    e3c_case["compact_point_count"] or
                    e3c_case["signal_state"] != STATE_NAMES[code] or
                    d_case["shared_compact_vertex_index"] !=
                    c_case["shared_compact_vertex_index"] or
                    d_case["predecessor_lane_id"] !=
                    c_case["predecessor_lane_id"]):
                raise ValueError(f"Lane {lane_id} E3-C/E4 Group or Anchor differs")
            if (b_case["regular"]["tensor_row"] != regular_row or
                    b_case["regular"]["stop_source_status"] != "uncertain" or
                    c_case["final_tensor_stop_provenance_verified"] is not None or
                    c_case["final_tensor_stop_provenance_status"] != "uncertain" or
                    d_case["final_tensor_stop_provenance_verified"] is not None or
                    d_case["final_tensor_stop_provenance_status"] != "uncertain"):
                raise ValueError(f"Lane {lane_id} final Stop source claim differs")
            if (not b_case["stop_point_inside_fov"] or
                    not b_case["shared_boundary"]["compact_vertex_kept_by_fov"] or
                    not d_case["stop_point_fov_retained"] or
                    b_case["stop_fov_source_audit"]["status"] != "uncertain"):
                raise ValueError(f"Lane {lane_id} Stop FoV evidence differs")
            anchor = {
                "audit_status": "verified",
                "applies_to_raw_lane_id": lane_id,
                "predecessor_lane_id": d_case["predecessor_lane_id"],
                "shared_compact_vertex_index": d_case[
                    "shared_compact_vertex_index"],
                "shared_vertex_global_xy": d_case[
                    "compact_shared_vertex_global_xy"],
                "selected_vertex_consistent": True,
                "topology_verified": True,
                "geometric_candidate_vertex_indices": d_case[
                    "geometric_candidate_vertex_indices"],
                "geometric_candidates_count": 1,
                "unique_anchor_verified": True,
                "xy_tolerance_m": d_case["xy_tolerance_m"],
            }
            stop_coordinate_fov = "retained"
            stop_source_fov = b_case["stop_fov_source_audit"]["status"]
            regular_stop_status = b_case["regular"]["stop_source_status"]
            final_stop_status = c_case["final_tensor_stop_provenance_status"]
            final_verified = c_case["final_tensor_stop_provenance_verified"]
            refs = {"raw_signal": "record_payload", "group_and_rows": "e4a",
                    "anchor": "e4d", "stop_fov": "e4b", "final_stop": "e4c"}
        else:
            anchor = {
                "audit_status": "not_run", "applies_to_raw_lane_id": lane_id,
                "predecessor_lane_id": None,
                "shared_compact_vertex_index": None,
                "shared_vertex_global_xy": None,
                "selected_vertex_consistent": None,
                "topology_verified": None,
                "geometric_candidate_vertex_indices": None,
                "geometric_candidates_count": None,
                "unique_anchor_verified": None,
                "xy_tolerance_m": None,
            }
            stop_coordinate_fov = "not_run"
            stop_source_fov = "not_run"
            regular_stop_status = "not_run"
            final_stop_status = "not_run"
            final_verified = None
            refs = {"raw_signal": "record_payload", "group_and_rows": "e4a",
                    "anchor": None, "stop_fov": None, "final_stop": None}
        records.append({
            "context": context.copy(),
            "raw_lane_id": lane_id,
            "signal": {
                "presence": "present", "state_code": code,
                "state_name": STATE_NAMES[code],
                "stop_point_global_xyz": list(stop["stop_point_global_xyz"]),
            },
            "compact": {
                "group_id": group_id,
                "ordered_member_lane_ids": members,
                "source_point_range": case["source_point_range"],
                "compact_point_range": case["compact_point_range"],
            },
            "anchor": anchor,
            "fov": {
                "local_contribution_status": case["local_fov_point_status"],
                "stop_point_coordinate_status": stop_coordinate_fov,
                "stop_source_status": stop_source_fov,
            },
            "road_tensor": {
                "regular": {
                    "lane_id": group_id,
                    "tensor_row": regular_row,
                    "row_status": ("retained" if regular_row is not None
                                   else "filtered"),
                    "local_lane_source_status": case["regular_input_status"],
                    "stop_identity_status": regular_stop_status,
                },
                "partitioned": partitioned,
                "final_stop_identity_status": final_stop_status,
                "final_tensor_stop_provenance_verified": final_verified,
            },
            "evidence_refs": refs,
        })
    document = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        **context,
        "current_time_index": scenario["current_time_index"],
        "official_repo_commit": SOURCE_COMMIT,
        "official_source_blobs": a["environment"]["official_source_blobs"],
        "evidence_sha256": evidence_hashes,
        "evidence_paths": EVIDENCE_PATHS.copy(),
        "raw_lane_ids": sorted(scenario["raw_lanes"]),
        "coordinate_frame": "waymo_global_xyz",
        "row_index_stage": "pre_training_random_permutation",
        "model_condition_injected": False,
        "signals": records,
    }
    # A full serialize/deserialize pass also enforces the public contract.
    result = loads_sidecar(
        dumps_sidecar(document), expected_scenario_id=SCENARIO_ID,
        expected_record_index=0, expected_time_index=10,
        expected_record_sha256=payload_hash)
    verify_evidence_files(result, RESEARCH_DIR)
    return result


def _first_difference(actual, expected, path="sidecar"):
    if type(actual) is not type(expected):
        return path
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            return f"{path}.keys"
        for key in sorted(expected):
            difference = _first_difference(actual[key], expected[key],
                                           f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(actual) != len(expected):
            return f"{path}.length"
        for index, (value, reference) in enumerate(zip(actual, expected)):
            difference = _first_difference(value, reference,
                                           f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    return None if actual == expected else path


def verify_real_sidecar(document, tfrecord=DEFAULT_TFRECORD):
    """Rebuild fixed record 0 from raw Proto/E4 evidence and compare claims."""
    # The structural round trip also rejects unsupported status combinations.
    actual = loads_sidecar(dumps_sidecar(document))
    expected = build_real_sidecar(tfrecord)
    difference = _first_difference(actual, expected)
    if difference is not None:
        raise ValueError(f"E5-A real Sidecar claim differs at {difference}")


def render_report(document):
    """Summarize claims without conflating row existence with Stop identity."""
    rows = []
    for record in document["signals"]:
        signal = record["signal"]
        anchor = record["anchor"]
        road = record["road_tensor"]
        partition = ", ".join(
            f"{child['child_id']}→{child['tensor_row']} "
            f"({child['stop_identity_status']})"
            for child in road["partitioned"])
        rows.append(
            f"| {record['raw_lane_id']} | {signal['state_name']} "
            f"({signal['state_code']}) | G{record['compact']['group_id']} "
            f"{record['compact']['source_point_range']}→"
            f"{record['compact']['compact_point_range']} | "
            f"{anchor['audit_status']} | "
            f"{record['fov']['local_contribution_status']} / "
            f"{record['fov']['stop_point_coordinate_status']} / "
            f"{record['fov']['stop_source_status']} | "
            f"{road['regular']['tensor_row']} "
            f"({road['regular']['stop_identity_status']}) | "
            f"{partition} | {road['final_stop_identity_status']} |")
    state_counts = {name: sum(record["signal"]["state_name"] == name
                              for record in document["signals"])
                    for name in STATE_NAMES.values()}
    verified = sum(record["anchor"]["audit_status"] == "verified"
                   for record in document["signals"])
    return "\n".join([
        "# E5-A：Signal–Lane Sidecar 固定场景核对",
        "",
        f"Schema `{SCHEMA_VERSION}`；Scenario `{document['scenario_id']}`，"
        f"record `{document['record_index']}`，固定 `t={document['time_index']}`。"
        f"原始 payload SHA-256 `{document['record_payload_sha256']}`；"
        f"官方源码 commit `{document['official_repo_commit']}`。"
        "12 条原始 LaneState 逐一核对："
        f"STOP {state_counts['STOP']}、ARROW_STOP {state_counts['ARROW_STOP']}、"
        f"UNKNOWN {state_counts['UNKNOWN']}；唯一 Anchor 审计 verified "
        f"{verified} 条。",
        "",
        "| 原 Lane | 原始信号 | Group 原始贡献点区间→Compact 区间 | Anchor | 局部 FoV / Stop 坐标 FoV / Stop 来源 FoV | Regular 行 (Stop 来源) | Partition 子 ID→行 (Stop 来源) | 最终 Stop 身份 |",
        "|---:|---|---|---|---|---|---|---|",
        *rows,
        "",
        "六条 STOP/ARROW_STOP 的 Anchor `verified` 源于 E4-D 的唯一同 XY "
        "候选、原始相邻拓扑与全局几何复核。六条 UNKNOWN 的原始状态代码 `0` "
        "及 Stop Point XYZ 直接来自 TFRecord，但其 Anchor、Stop FoV 和最终 "
        "Stop 身份审计均为 `not_run`；UNKNOWN 不表示绿灯。原始图中的 Lane 529 "
        "没有 LaneState，因此没有 Sidecar 信号记录，也未被改写为 UNKNOWN。",
        "",
        "六条 Stop Point 坐标在 FoV 内，但 E4-B 的原始 Stop 来源 FoV 审计"
        "仍为 `uncertain`，因为拼接时下游 Lane 首点被省略。"
        "所有行号都来自 E4-A 固定 `t=10` 的官方道路分支、训练随机排列之前。"
        "Partition 列包含停止信号的非 Stop 侧子行：即使该子行有张量行，"
        "其 Stop 来源仍为 `filtered`。六条停止信号的最终 20 点原始 Stop"
        " 身份继续为 `uncertain`；本 Sidecar 没有向 AE/LDM 注入信号条件，"
        "`model_condition_injected=false`。",
        "",
        "**交叉核对：** 原始 record 与 E4-A/B/C/D 的 Scenario、record、时间"
        "和 payload 指纹一致；E4-A/B 的六个官方源码 blob 一致，E4-D 的"
        "两个 Compact 源码 blob 匹配；E4-C/D 保存的 B/C JSON SHA-256 "
        "链一致。12 条 Group、半开区间与两路行从 E4-A 逐项核对，六条 "
        "Stop/Arrow Stop 的 Stop XYZ、Group、唯一 Anchor 和分区 Stop 来源"
        "再与 E4-B/C/D 核对。所有证据文件路径和 SHA-256 见 Sidecar JSON。",
        "",
        "**复核边界：** `loads_sidecar` 校验结构，`verify_evidence_files` "
        "校验文件 SHA-256；针对本固定场景，`verify_real_sidecar` 从原始 "
        "TFRecord 与 E4 证据重建并逐字段比对，防止事后升级审计状态。",
        "",
        "**限制：** 本轮没有重新执行官方 FoV、Partition 或张量重采样；"
        "Sidecar 只是只读证据契约，并不证明最终 Stop 原始身份。"
        "未修改官方预处理、AE、LDM、Checkpoint，也未训练或使用 GPU。",
        "",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfrecord", type=Path, default=DEFAULT_TFRECORD)
    parser.add_argument("--sidecar", type=Path,
                        default=OUTPUT_DIR / "e5a_sidecar.json")
    parser.add_argument("--report", type=Path,
                        default=OUTPUT_DIR / "e5a_report.md")
    args = parser.parse_args()
    document = build_real_sidecar(args.tfrecord)
    args.sidecar.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.sidecar.write_text(dumps_sidecar(document), encoding="utf-8")
    args.report.write_text(render_report(document), encoding="utf-8")
    state_counts = {name: sum(record["signal"]["state_name"] == name
                              for record in document["signals"])
                    for name in STATE_NAMES.values()}
    print(f"scenario={document['scenario_id']} record=0 t=10 "
          f"signals={len(document['signals'])} states={state_counts} "
          f"model_condition_injected={document['model_condition_injected']}")
    print(f"sidecar={args.sidecar} report={args.report}")


if __name__ == "__main__":
    main()
