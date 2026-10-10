"""Rebuild record-0 Compact joins and independently audit six Stop Anchors.

Run with ``python -m research.traffic_logic.e4d_real_anchor_diagnostic``.
The TFRecord and all earlier reports are read only; only E4-D outputs are
written. The local adapter executes exact official function bodies on CPU.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

import numpy as np

from .e4d_anchor_geometry import audit_anchor_global, parse_stop_points
from .real_scenario_helpers import (build_graphs, parse_scenario,
                                    read_tfrecord_record)


SCENARIO_ID = "637f20cafde22ff8"
RECORD_SHA256 = "7af5edce831d6aec3eae0f665ee137b3eaf97a949cec6c9ea1e0127e1be46442"
SOURCE_COMMIT = "675423469766bf2fd8a6b569ef1869a6f1e76993"
TARGET_STATES = {
    443: (4, "STOP"), 445: (4, "STOP"),
    448: (4, "STOP"), 449: (4, "STOP"),
    455: (1, "ARROW_STOP"), 456: (1, "ARROW_STOP"),
}
SOURCE_FILES = (
    "data_processing/waymo/generate_waymo_dataset.py",
    "utils/lane_graph_helpers.py",
)
DEFAULT_TFRECORD = Path(
    r"E:\ScenarioDreamer\datasets\raw_womd\v1.1.0\validation"
    r"\uncompressed_scenario_validation_validation.tfrecord-00000-of-00150")
ROOT = Path(__file__).absolute().parents[2]
OUTPUT = Path(__file__).absolute().parent / "output"


def git_blob_sha1(path):
    """Hash a worktree source through Git's CRLF filter without storing it."""
    return subprocess.check_output(
        ["git", "hash-object", str(path)], cwd=ROOT, text=True).strip()


def e3c_baselines(path):
    """Read the six ordered Group, shape and geometry claims from E3-C."""
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = [part.strip() for part in line.split("|")]
        if len(fields) != 6 or fields[0] not in ("STOP", "ARROW_STOP"):
            continue
        state, lane, group, members, shape, geometry_hash = fields
        lane_id = int(lane)
        if lane_id in rows:
            raise ValueError(f"Duplicate E3-C Lane row: {lane_id}")
        match = re.fullmatch(r"(\d+)x2", shape)
        if match is None or not re.fullmatch(r"[0-9a-f]{64}", geometry_hash):
            raise ValueError(f"Malformed E3-C Group row: {line}")
        rows[lane_id] = {
            "signal_state": state,
            "group_id": int(group),
            "group_members": [int(item) for item in members.split(",")],
            "compact_point_count": int(match.group(1)),
            "compact_geometry_sha256": geometry_hash,
        }
    if set(rows) != set(TARGET_STATES):
        raise ValueError("E3-C report lacks exactly the six target Lanes")
    return rows


def _validate_prior(e4b, e4c, e4b_sha256):
    source = e4b["input"]
    expected = (SCENARIO_ID, 0, 10, RECORD_SHA256)
    actual = (source["scenario_id"], source["record_index"],
              source["time_index"], source["record_payload_sha256"])
    if e4b["run_status"] != "pass" or actual != expected:
        raise ValueError("E4-B baseline does not match this record and timestep")
    if e4b["environment"]["repo_commit"] != SOURCE_COMMIT:
        raise ValueError("E4-B official source commit differs")
    if (e4c["scenario_id"], e4c["record_index"], e4c["time_index"],
            e4c["record_payload_sha256"], e4c["official_repo_commit"]) != (
                *expected, SOURCE_COMMIT):
        raise ValueError("E4-C baseline does not match this record or source")
    if e4c["e4b_json_sha256"] != e4b_sha256:
        raise ValueError("E4-C was generated from a different E4-B JSON")
    b_ids = [case["raw_lane_id"] for case in e4b["lanes"]]
    c_ids = [case["raw_lane_id"] for case in e4c["anchors"]]
    if len(b_ids) != len(TARGET_STATES) or set(b_ids) != set(TARGET_STATES):
        raise ValueError("E4-B does not contain exactly the six target Lanes")
    if len(c_ids) != len(TARGET_STATES) or set(c_ids) != set(TARGET_STATES):
        raise ValueError("E4-C does not contain exactly the six target Lanes")


def run(tfrecord=DEFAULT_TFRECORD, output_dir=OUTPUT):
    """Run one read-only source reconstruction and write its new evidence."""
    tfrecord = Path(tfrecord)
    output_dir = Path(output_dir)
    e4b_path = OUTPUT / "e4b_autodl" / "e4b_result.json"
    e4c_path = OUTPUT / "e4c_anchor_records.json"
    e3c_path = OUTPUT / "e3c_official_crosscheck_report.txt"
    e4b_raw, e4c_raw = e4b_path.read_bytes(), e4c_path.read_bytes()
    e4b, e4c = json.loads(e4b_raw), json.loads(e4c_raw)
    _validate_prior(e4b, e4c, hashlib.sha256(e4b_raw).hexdigest())
    e3c = e3c_baselines(e3c_path)
    expected_blobs = e4b["environment"]["official_source_blobs"]
    local_blobs = {name: git_blob_sha1(ROOT / name) for name in SOURCE_FILES}
    if any(local_blobs[name] != expected_blobs[name] for name in SOURCE_FILES):
        raise ValueError("Local official processing source differs from AutoDL")
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if branch != "research/traffic-logic":
        raise ValueError(f"Unexpected research branch: {branch}")
    payload = read_tfrecord_record(tfrecord, 0)
    payload_hash = hashlib.sha256(payload).hexdigest()
    if payload_hash != RECORD_SHA256:
        raise ValueError("Record-0 payload fingerprint differs from E3-C")
    scenario = parse_scenario(payload, 10)
    if (scenario["scenario_id"], scenario["current_time_index"]) != (
            SCENARIO_ID, 10):
        raise ValueError("Scenario ID or current_time_index differs")
    stop_points = parse_stop_points(payload, 10, TARGET_STATES)
    for lane_id, (state_code, _) in TARGET_STATES.items():
        if (scenario["lane_states"].get(lane_id) != state_code or
                stop_points[lane_id]["state_code"] != state_code):
            raise ValueError(f"Lane {lane_id} signal state differs")

    # Exactly one official grouping and Compact construction for this record.
    lane_graph, compact_graph, groups = build_graphs(scenario["raw_lanes"])
    counts = {
        "raw_lanes": len(scenario["raw_lanes"]),
        "processed_lanes": len(lane_graph["lanes"]),
        "compact_groups": len(compact_graph["lanes"]),
    }
    if counts != {"raw_lanes": 199, "processed_lanes": 198,
                  "compact_groups": 135}:
        raise ValueError(f"Graph counts differ from E3-C: {counts}")
    lane529_group_id = next(group_id for group_id, members in groups.items()
                            if members == [529])
    lane529 = {
        "processed_lane_point_count": len(lane_graph["lanes"][529]),
        "group_id": lane529_group_id,
        "compact_point_count": len(compact_graph["lanes"][lane529_group_id]),
    }
    if lane529 != {"processed_lane_point_count": 1, "group_id": 0,
                   "compact_point_count": 1}:
        raise ValueError(f"Lane 529 exception differs from E3-C: {lane529}")
    e4b_cases = {case["raw_lane_id"]: case for case in e4b["lanes"]}
    e4c_cases = {case["raw_lane_id"]: case for case in e4c["anchors"]}
    audits = []
    for lane_id, (_, state_name) in TARGET_STATES.items():
        stop = stop_points[lane_id]
        audit = audit_anchor_global(
            scenario["raw_lanes"], lane_graph, compact_graph, groups,
            lane_id, stop["stop_point_global_xyz"], tolerance_m=1e-9)
        old_b, old_c, old_3 = e4b_cases[lane_id], e4c_cases[lane_id], e3c[lane_id]
        audit["signal_state"] = state_name
        audit["signal_state_code"] = stop["state_code"]
        checks = {
            "e3c_signal_state": state_name == old_3["signal_state"],
            "e3c_group_id": audit["group_id"] == old_3["group_id"],
            "e3c_ordered_members": audit["group_members"] == old_3["group_members"],
            "e3c_compact_point_count": (
                audit["compact_point_count"] == old_3["compact_point_count"]),
            "e3c_full_compact_geometry_sha256": (
                audit["compact_geometry_sha256"] ==
                old_3["compact_geometry_sha256"]),
            "e4c_group_id": audit["group_id"] == old_c["group_id"],
            "e4c_predecessor_lane_id": (
                audit["predecessor_lane_id"] == old_c["predecessor_lane_id"]),
            "e4c_shared_vertex_index": (
                audit["shared_compact_vertex_index"] ==
                old_c["shared_compact_vertex_index"]),
            "e4c_stop_point_xyz": (
                audit["stop_point_global_xyz"] == old_c["stop_point"]),
            "e4b_stop_point_xyz": (
                audit["stop_point_global_xyz"] == old_b["stop_point_global_xyz"]),
            "e4b_group_id": audit["group_id"] == old_b["group_id"],
            "e4b_ordered_members": (
                audit["group_members"] == old_b["group_members"]),
            "e4b_predecessor_lane_id": (
                audit["predecessor_lane_id"] ==
                old_b["shared_boundary"]["predecessor_lane_id"]),
            "e4b_shared_vertex_index": (
                audit["shared_compact_vertex_index"] ==
                old_b["shared_boundary"]["compact_vertex_index"]),
        }
        audit["baseline_crosschecks"] = checks
        audit["baseline_consistent"] = all(checks.values())
        audit["e3c_compact_geometry_sha256"] = old_3["compact_geometry_sha256"]
        audit["e4c_selected_shared_vertex_verified"] = old_c[
            "shared_vertex_verified"]
        audit["e4c_geometric_candidate_scan_performed"] = old_c[
            "geometric_candidate_scan_performed"]
        audit["stop_point_fov_retained"] = bool(
            old_b["stop_point_inside_fov"] and
            old_b["shared_boundary"]["compact_vertex_kept_by_fov"])
        audit["stop_point_fov_evidence"] = "E4-B official fixed-t10 mask; not rerun in E4-D"
        audit["final_tensor_stop_provenance_verified"] = old_c[
            "final_tensor_stop_provenance_verified"]
        audit["final_tensor_stop_provenance_status"] = old_c[
            "final_tensor_stop_provenance_status"]
        if (audit["final_tensor_stop_provenance_verified"] is not None or
                audit["final_tensor_stop_provenance_status"] != "uncertain"):
            raise ValueError(f"Lane {lane_id} E4-C final provenance changed")
        audit["e4d_verification_status"] = (
            "verified" if audit["baseline_consistent"] and
            audit["selected_vertex_consistent"] and audit["topology_verified"] and
            audit["unique_anchor_verified"] else "inconsistent")
        audits.append(audit)

    output = {
        "run_status": ("pass" if all(item["e4d_verification_status"] == "verified"
                                     for item in audits) else "inconsistent"),
        "scope": "independent_global_xy_compact_anchor_uniqueness_only",
        "scenario_id": SCENARIO_ID,
        "record_index": 0,
        "time_index": 10,
        "record_payload_bytes": len(payload),
        "record_payload_sha256": payload_hash,
        "tfrecord_path": str(tfrecord),
        "tfrecord_bytes": tfrecord.stat().st_size,
        "local_branch": branch,
        "local_head": head,
        "runtime": {"platform": platform.platform(),
                    "python": sys.version.split()[0], "numpy": np.__version__},
        "official_autodl_commit": SOURCE_COMMIT,
        "official_source_blobs_local": local_blobs,
        "e4b_json_sha256": hashlib.sha256(e4b_raw).hexdigest(),
        "e4c_json_sha256": hashlib.sha256(e4c_raw).hexdigest(),
        "counts": counts,
        "lane529_exception": lane529,
        "xy_tolerance_m": 1e-9,
        "coordinate_frame": "global_xy",
        "tfrecord_framing_crc_checked": False,
        "lanes": audits,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "e4d_result.json"
    report_path = output_dir / "e4d_report.md"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    report_path.write_text(render_report(output), encoding="utf-8")
    return output, json_path, report_path


def render_report(result):
    distance_keys = (
        "stop_to_downstream_first", "stop_to_predecessor_last",
        "stop_to_selected_compact", "predecessor_to_downstream")
    max_join_distance = max(
        item["distances_m"][key] for item in result["lanes"]
        for key in distance_keys if key in item["distances_m"]
    ) if all(item["distances_m"] for item in result["lanes"]) else None
    lines = [
        "# E4-D：Stop Anchor 独立几何与拓扑复核",
        "",
        f"状态：**{result['run_status']}**。Scenario `{result['scenario_id']}`，"
        f"Waymo v1.1 validation record 0，固定 `t=10`。原始 record payload "
        f"SHA-256 `{result['record_payload_sha256']}`；本地 HEAD "
        f"`{result['local_head'][:7]}`，AutoDL 官方基线 "
        f"`{result['official_autodl_commit'][:7]}`。",
        "",
        f"数据 `{result['tfrecord_path']}`（{result['tfrecord_bytes']} bytes）；"
        f"本机 `{result['runtime']['platform']}`，Python "
        f"`{result['runtime']['python']}`，NumPy "
        f"`{result['runtime']['numpy']}`。",
        "",
        "本次在 Windows CPU 上从同一原始 record 独立读取 Stop Point，并在同一次"
        " `build_graphs()` 调用中执行官方函数体，得到 199 原始 Lane → 198 预处理"
        " Lane → 135 Compact Group。以下端点距离与整组候选扫描由本次全局 XY "
        "几何重新计算，容差均为 `1e-9 m`；未混用 E4-C 的 ego XY 接口。",
        "",
        "| Lane / 信号 | Group·前驱·选定下标 | E4-C 选定点 | E4-D 选定点 / 拓扑 | 全组同 XY 候选数及下标 | 几何唯一 | Stop FoV¹ | 最终 20 点来源 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in result["lanes"]:
        prior = "verified" if item["e4c_selected_shared_vertex_verified"] else "unverified"
        selected = "verified" if item["selected_vertex_consistent"] else "inconsistent"
        topology = "verified" if item["topology_verified"] else "inconsistent"
        unique = "verified" if item["unique_anchor_verified"] else "uncertain"
        fov = "retained" if item["stop_point_fov_retained"] else "filtered"
        lines.append(
            f"| {item['raw_lane_id']} {item['signal_state']} | "
            f"G{item['group_id']}·{item['predecessor_lane_id']}·"
            f"{item['shared_compact_vertex_index']} | {prior} | "
            f"{selected} / {topology} | "
            f"{item['geometric_candidates_count']}: "
            f"{item['geometric_candidate_vertex_indices']} | {unique} | "
            f"{fov} | {item['final_tensor_stop_provenance_status']} |")
    lines.append("")
    if result["run_status"] == "pass":
        lines.extend([
            "**新旧交叉核对：** 六条 Lane 的 E3-C Group ID、完整有序成员、"
            "Compact 点数及整条折线 SHA-256，和 E4-C Group ID、紧邻前驱、共享顶点"
            "下标与 Stop Point 均逐项一致。每条重新计算的 Stop→原始下游首点、"
            "Stop→原始前驱末点、Stop→选定 Compact 顶点、前驱末点→下游首点"
            f"的 XY 距离最大为 `{max_join_distance:.3g} m`；"
            "官方原始 entry/exit 与处理后 pre/suc "
            "双向连接均一致。完整逐项布尔值、端点坐标、距离和候选下标见 JSON。",
            "",
            "**候选结论：** 六条目标 Lane 的整条 Compact 折线扫描均只找到"
            "表中一个同 XY 顶点，故本次 `unique_anchor_verified=true`。"
            "若未来出现多个候选，拓扑选定点仍可一致，但几何唯一性须保持未验证。",
            "",
        ])
    else:
        failed = [item for item in result["lanes"]
                  if item["e4d_verification_status"] != "verified"]
        for item in failed:
            false_checks = [name for name, value in
                            item["baseline_crosschecks"].items() if not value]
            lines.append(
                f"**不一致 Lane {item['raw_lane_id']}：** 旧证据差异 "
                f"`{false_checks}`；新证据原因 `{item['reasons']}`；"
                f"候选 `{item['geometric_candidate_vertex_indices']}`。")
        lines.append("")
    lines.extend([
        "¹ `stop_point_fov_retained` 沿用 E4-B 的同场景固定 `t=10` 官方"
        " FoV 点掩码（Stop 在 FoV 内且共享 Compact 顶点保留）；E4-D 未重新执行"
        " FoV、Partition 或两次重采样。六条 `final_tensor_stop_provenance_verified`"
        " 保持 E4-C 的 `null`，状态均为 `uncertain`。拓扑与几何唯一不证明"
        "最终 20 点仍可归因于原始受控 Lane，亦不将信号扩散到整个 Group。",
        "",
        "**范围与限制：** Windows 无 TensorFlow/Waymo 包，本地适配器从未修改的"
        "官方源码加载相关函数体；这些源码的 Git blob 与 AutoDL 官方基线匹配。"
        "TFRecord framing CRC 未校验，但 record payload SHA-256 与 E3-C"
        " 官方导入验证相同。Lane 529 的单点 Group 0 仍保留在 135 个 Group 中，"
        "未为 lineage 静默丢弃。未修改生产代码、模型输入或权重，未使用 GPU 或训练。",
        "",
        "运行：`python -m research.traffic_logic.e4d_real_anchor_diagnostic`。",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfrecord", type=Path, default=DEFAULT_TFRECORD)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result, json_path, report_path = run(args.tfrecord, args.output_dir)
    print(f"status={result['run_status']} record={result['scenario_id']} "
          f"t={result['time_index']} counts={result['counts']}")
    for item in result["lanes"]:
        print(f"Lane {item['raw_lane_id']}: G{item['group_id']} "
              f"predecessor={item['predecessor_lane_id']} "
              f"selected={item['shared_compact_vertex_index']} "
              f"candidates={item['geometric_candidate_vertex_indices']} "
              f"unique={item['unique_anchor_verified']} "
              f"final={item['final_tensor_stop_provenance_status']}")
    print(f"json={json_path} report={report_path}")
    if result["run_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
