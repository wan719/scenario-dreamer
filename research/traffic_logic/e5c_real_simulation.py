"""Run the fixed t=10 Signal Token interface simulation outside the model.

Inputs are the saved Sidecar, E4 evidence, and original WOMD record 0. This
module writes only an E5-C research JSON and report; no model input is made.
"""

import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path

from .e4d_real_anchor_diagnostic import DEFAULT_TFRECORD, RECORD_SHA256, SCENARIO_ID
from .signal_interface_simulation import global_to_ego_xy, simulate_interface


RESEARCH_DIR = Path(__file__).absolute().parent
OUTPUT_DIR = RESEARCH_DIR / "output"
SIDECAR_PATH = OUTPUT_DIR / "e5a_sidecar.json"
E4A_PATH = OUTPUT_DIR / "e4a_autodl" / "e4a_result.json"
E4B_PATH = OUTPUT_DIR / "e4b_autodl" / "e4b_result.json"
E4D_PATH = OUTPUT_DIR / "e4d_result.json"
E5B_PATH = OUTPUT_DIR / "e5b_projection.json"
OUTPUT_PATH = OUTPUT_DIR / "e5c_simulation.json"
REPORT_PATH = RESEARCH_DIR / "e5c_report.md"
STOP_LANES = {
    443: "STOP", 445: "STOP", 448: "STOP", 449: "STOP",
    455: "ARROW_STOP", 456: "ARROW_STOP",
}
EXPECTED_SIGNAL_LANES = {431, 432, 443, 445, 446, 447, 448, 449,
                         450, 455, 456, 457}


def _load_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Fixed-scene evidence is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _context(document):
    return (document["scenario_id"], document["record_index"],
            document["time_index"], document["record_payload_sha256"])


def _relation_key(relation):
    return (relation["raw_lane_id"], relation["road_branch"],
            relation["child_id"])


def _check_baseline_masks(simulation, previous):
    """Confirm that the independent route does not rewrite E5-B history."""
    old_tokens = {item["raw_lane_id"]: item for item in previous["tokens"]}
    old_relations = {_relation_key(item): item for item in
                     previous["candidate_relations"]}
    if (len(old_tokens) != len(simulation["tokens"]) or
            len(old_relations) != len(simulation["alignments"])):
        raise ValueError("E5-B token or relation inventory differs")
    for token in simulation["tokens"]:
        old = old_tokens.get(token["raw_lane_id"])
        if old is None or token["e5b_masks"] != old["masks"]:
            raise ValueError("E5-B token Mask was changed")
    for alignment in simulation["alignments"]:
        old = old_relations.get(_relation_key(alignment))
        if old is None or alignment["e5b_masks"] != old["masks"]:
            raise ValueError("E5-B relation Mask was changed")
        for new_name, old_name in (
                ("pre_shuffle_tensor_row", "tensor_row"),
                ("group_id", "group_id"),
                ("source_point_range", "source_point_range"),
                ("compact_point_range", "compact_point_range")):
            if alignment[new_name] != old[old_name]:
                raise ValueError(f"E5-B relation {new_name} was changed")


def build_fixed_simulation(tfrecord=DEFAULT_TFRECORD):
    """Rebuild fixed-scene claims, then check the saved historical projection."""
    tfrecord = Path(tfrecord)
    if not tfrecord.is_file():
        raise FileNotFoundError(f"Original WOMD TFRecord is missing: {tfrecord}")
    sidecar = _load_json(SIDECAR_PATH)
    e4a = _load_json(E4A_PATH)
    e4b = _load_json(E4B_PATH)
    e4d = _load_json(E4D_PATH)
    previous = _load_json(E5B_PATH)
    expected = (SCENARIO_ID, 0, 10, RECORD_SHA256)
    if _context(sidecar) != expected or _context(e4d) != expected:
        raise ValueError("Sidecar or E4-D fixed-scene context differs")
    for evidence in (e4a, e4b):
        if _context(evidence["input"]) != expected:
            raise ValueError("E4-A/B fixed-scene context differs")
    if _context(previous) != expected:
        raise ValueError("E5-B fixed-scene context differs")
    ego_pose = {
        "context": {key: sidecar[key] for key in (
            "scenario_id", "record_index", "time_index",
            "record_payload_sha256")},
        "source_frame": "waymo_global_xyz",
        "target_frame": "waymo_ego_xy",
        "center_global_xy": e4a["input"]["ego_t10_center"],
        "yaw_rad": e4a["input"]["ego_t10_yaw"],
    }
    simulation = simulate_interface(
        sidecar, evidence_base_dir=RESEARCH_DIR, real_tfrecord=tfrecord,
        ego_pose=ego_pose, expected_scenario_id=SCENARIO_ID,
        expected_record_index=0, expected_time_index=10,
        expected_current_time_index=10,
        expected_record_sha256=RECORD_SHA256)
    if simulation["verification_level"] != "fixed_scene_lane_rebuilt":
        raise ValueError("Fixed-scene Lane evidence was not rebuilt")
    if simulation["model_condition_injected"] is not False:
        raise ValueError("E5-C must not claim model condition injection")
    if (simulation["source_sidecar_sha256"] !=
            previous["source_sidecar_sha256"] or
            simulation["official_repo_commit"] !=
            previous["official_repo_commit"] or
            simulation["row_index_stage"] !=
            previous["row_index_stage"] or
            simulation["row_index_stage"] !=
            "pre_training_random_permutation" or
            simulation["simulated_permutation_supplied"] != []):
        raise ValueError("E5-B source or pre-shuffle row stage differs")
    tokens = simulation["tokens"]
    alignments = simulation["alignments"]
    if (len(tokens), len(alignments)) != (12, 30):
        raise ValueError("Fixed-scene Token/alignment inventory differs")
    if {token["raw_lane_id"] for token in tokens} != EXPECTED_SIGNAL_LANES:
        raise ValueError("Fixed-scene LaneState set differs")
    _check_baseline_masks(simulation, previous)

    e4b_lanes = {item["raw_lane_id"]: item for item in e4b["lanes"]}
    e4d_lanes = {item["raw_lane_id"]: item for item in e4d["lanes"]}
    token_by_lane = {item["raw_lane_id"]: item for item in tokens}
    max_transform_error = 0.0
    for lane_id, state in STOP_LANES.items():
        token = token_by_lane[lane_id]
        original = e4b_lanes[lane_id]
        anchor = e4d_lanes[lane_id]
        if token["state_name"] != state or original["signal_state"] != state:
            raise ValueError(f"Lane {lane_id} signal state differs")
        if (token["stop_point_global_xyz"] !=
                anchor["stop_point_global_xyz"] or
                token["stop_point_global_xyz"] !=
                original["stop_point_global_xyz"]):
            raise ValueError(f"Lane {lane_id} global Stop differs")
        calculated = global_to_ego_xy(
            token["stop_point_global_xyz"][:2], ego_pose["center_global_xy"],
            ego_pose["yaw_rad"])
        error = math.dist(calculated, original["stop_point_ego_xy"])
        max_transform_error = max(max_transform_error, error)
        if error > 1e-9 or math.dist(
                token["stop_point_ego_xy"], calculated) > 1e-9:
            raise ValueError(f"Lane {lane_id} official Ego transform differs")
        if not all((token["explicit_signal_observed"],
                    token["temporal_condition_available"],
                    token["anchor_alignment_verified"],
                    token["raw_lane_scope_known"],
                    token["new_interface_candidate"])):
            raise ValueError(f"Lane {lane_id} independent Token evidence differs")
        lane_alignments = [entry for entry in alignments
                           if entry["raw_lane_id"] == lane_id]
        if (len(lane_alignments) != 3 or
                sum(entry["new_interface_candidate"]
                    for entry in lane_alignments) != 2 or
                sum(entry["road_row_alignment_status"] == "filtered"
                    for entry in lane_alignments) != 1):
            raise ValueError(f"Lane {lane_id} branch alignment differs")
        regular = next(entry for entry in lane_alignments
                       if entry["road_branch"] == "regular")
        if (regular["group_id"] != anchor["group_id"] or
                regular["compact_point_range"] != original["compact_point_range"] or
                regular["source_point_range"] != original["source_point_range"] or
                regular["compact_point_range"][0] - 1 !=
                anchor["shared_compact_vertex_index"] or
                regular["stop_anchor"]["shared_compact_vertex_index"] !=
                anchor["shared_compact_vertex_index"] or
                anchor["unique_anchor_verified"] is not True or
                anchor["topology_verified"] is not True):
            raise ValueError(f"Lane {lane_id} local Anchor scope differs")
    if (any(item["new_interface_verified"] for item in tokens) or
            any(item["new_interface_verified"] for item in alignments) or
            any(item["e5b_masks"]["tensor_control_identity_proven"]
                for item in alignments)):
        raise ValueError("Unproved Stop identity was promoted")
    if any(item["simulated_post_shuffle_row"] is not None
           for item in alignments):
        raise ValueError("Unprovided training row permutation was invented")
    return simulation, ego_pose, max_transform_error


def render_report(simulation, ego_pose, max_transform_error, tfrecord):
    """Render a compact side-by-side audit of old and independent routes."""
    tokens = {item["raw_lane_id"]: item for item in simulation["tokens"]}
    alignments = simulation["alignments"]
    by_lane = {lane_id: [entry for entry in alignments
                         if entry["raw_lane_id"] == lane_id]
               for lane_id in STOP_LANES}
    status_counts = Counter(entry["road_row_alignment_status"]
                            for entry in alignments)
    candidate_count = sum(entry["new_interface_candidate"]
                          for entry in alignments)
    verified_count = sum(entry["new_interface_verified"]
                         for entry in alignments)
    lines = [
        "# E5-C：独立 Signal Token 与局部道路对齐离线模拟",
        "",
        (f"固定 Scenario `{simulation['scenario_id']}`、record "
         f"`{simulation['record_index']}`、t=`{simulation['time_index']}`；"
         f"原始 payload SHA-256 `{simulation['record_payload_sha256']}`，"
         f"官方源码 `{simulation['official_repo_commit']}`。"
         "本轮仍为模型外只读模拟，`model_condition_injected=false`。"),
        "",
        "## 输入与核验",
        "",
        f"- 原始 TFRecord：`{Path(tfrecord)}`，record 0。",
        (f"- E5-A Sidecar `output/e5a_sidecar.json`；E4-A/B/D 与 E5-B "
         "保存证据均匹配同一 Scenario、时刻和 payload 指纹，"
         f"核验等级 `{simulation['verification_level']}`。"),
        ("- Ego 姿态来自 E4-A 固定 t=10 官方 agent 提取："
         f"center=({ego_pose['center_global_xy'][0]}, "
         f"{ego_pose['center_global_xy'][1]}) m，"
         f"yaw={ego_pose['yaw_rad']} rad。按官方 SE(2) 重算六条 Stop "
         f"Ego XY，与 E4-B 值的最大差 `{max_transform_error:.3g} m`。"
         "E4-D 的 Anchor 核验使用全局 XY；坐标系未混用。"),
        (f"- 独立 Token `{len(tokens)}` 个、Road Alignment "
         f"`{len(alignments)}` 条；新接口候选 `{candidate_count}` 条，"
         f"已验证 `{verified_count}` 条。对齐状态："
         + "、".join(f"{status} {count}" for status, count in
                    sorted(status_counts.items())) + "。"),
        "",
        "## 六条 STOP/ARROW_STOP Lane",
        "",
        ("| 原始 Lane / 状态 | Group·Anchor | 原始→Compact 局部半开区间 | Stop Ego XY (m) | "
         "Regular pre-row / 新对齐 | Partition 子ID→pre-row / 新对齐 | "
         "独立 Token 候选/已验证 | 旧张量身份已证明 |"),
        "|---|---|---|---|---|---|---|---|",
    ]
    for lane_id, state in STOP_LANES.items():
        token = tokens[lane_id]
        rows = by_lane[lane_id]
        regular = next(item for item in rows if item["road_branch"] == "regular")
        children = sorted((item for item in rows
                           if item["road_branch"] == "partitioned"),
                          key=lambda item: item["child_id"])
        anchor = regular["stop_anchor"]["shared_compact_vertex_index"]
        parts = ", ".join(
            f"{item['child_id']}→{item['pre_shuffle_tensor_row']} "
            f"{item['road_row_alignment_status']}"
            for item in children)
        xy = token["stop_point_ego_xy"]
        source = regular["source_point_range"]
        scope = regular["compact_point_range"]
        lines.append(
            f"| {lane_id} / {state} | G{regular['group_id']}·{anchor} | "
            f"[{source[0]},{source[1]})→[{scope[0]},{scope[1]}) | "
            f"({xy[0]:.3f}, {xy[1]:.3f}) | "
            f"{regular['pre_shuffle_tensor_row']} / "
            f"{regular['road_row_alignment_status']} | {parts} | "
            f"{str(token['new_interface_candidate']).lower()}/"
            f"{str(token['new_interface_verified']).lower()} | "
            f"{str(token['e5b_masks']['tensor_control_identity_proven']).lower()} |")
    lines.extend([
        "",
        ("六条信号的 `explicit_signal_observed`、"
         "`temporal_condition_available`、`anchor_alignment_verified`、"
         "`raw_lane_scope_known` 均为 `true`。Regular 与 Stop 侧 Partition "
         "只是局部候选；非 Stop 侧虽有行号，其局部来源被过滤。"
         "Group 共用不扩散灯态到其他原始 Lane。"),
        "",
        "## 两条条件证据路线",
        "",
        "| 路线 | 已成立 | 仍未成立 |",
        "|---|---|---|",
        ("| 旧道路张量身份 | E5-B 的 12 条离线候选关系及所有 Mask "
         "逐项保持原值 | 最终 20 点的原始 Stop 身份仍为 `uncertain`；"
         "`tensor_control_identity_proven=false`，可用条件 Mask 为 `false` |"),
        ("| 新独立 Token | 六条停止信号原始语义、Stop 坐标、时刻、"
         "唯一 Anchor、原 Lane 局部范围可独立保存；Regular/Stop 侧 "
         "Partition 可列候选 | Token 到具体最终道路行的局部来源尚未由新真实证据"
         "唯一证明；`new_interface_verified=false` |"),
        "",
        "## 限制与复现",
        "",
        ("- 显式 `UNKNOWN` 六条仍各有 Token，但 Anchor 未同级审计，"
         "不可视为 GO；没有 LaneState 的原始 Lane 不生成 Token。"
         "`future_label` 仅在合成测试中验证，不能用作当前条件。"),
        ("- 原 Lane、Compact、FoV 1000 点与最终 20 点属不同采样域，"
         "不能直接相交数字区间；Stop 附近的最终 20 点投影不证明来源。"
         "没有给出真实训练行随机排列，JSON 的 post-shuffle 行号为 `null`。"),
        ("- 运行 `python -m research.traffic_logic.e5c_real_simulation`；"
         f"环境 `{platform.platform()}`、Python `{sys.version.split()[0]}`，"
         "Windows CPU。无模型条件输入、无 AE/LDM 或官方预处理修改、"
         "无 GPU/训练。"),
    ])
    return "\n".join(lines) + "\n"


def main():
    simulation, ego_pose, error = build_fixed_simulation()
    OUTPUT_PATH.write_text(json.dumps(
        simulation, ensure_ascii=False, sort_keys=True, indent=2,
        allow_nan=False) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render_report(
        simulation, ego_pose, error, DEFAULT_TFRECORD), encoding="utf-8")
    print(f"scenario={simulation['scenario_id']} "
          f"tokens={len(simulation['tokens'])} "
          f"alignments={len(simulation['alignments'])} "
          f"verified={sum(item['new_interface_verified'] for item in simulation['alignments'])} "
          f"verification={simulation['verification_level']} "
          f"model_condition_injected={simulation['model_condition_injected']}")
    print(f"json={OUTPUT_PATH}")
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
