"""Generate the fixed E5-B offline candidate projection from audited E5-A data.

The read-only inputs are the saved Sidecar, its E4 evidence files, and record 0
of the original WOMD TFRecord. Only the E5-B diagnostic JSON/report are written.
"""

import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

from .e4d_real_anchor_diagnostic import DEFAULT_TFRECORD, RECORD_SHA256, SCENARIO_ID
from .signal_candidate_projection import project_sidecar


RESEARCH_DIR = Path(__file__).absolute().parent
SIDECAR_PATH = RESEARCH_DIR / "output" / "e5a_sidecar.json"
OUTPUT_PATH = RESEARCH_DIR / "output" / "e5b_projection.json"
REPORT_PATH = RESEARCH_DIR / "e5b_report.md"
STOP_LANES = {
    443: "STOP",
    445: "STOP",
    448: "STOP",
    449: "STOP",
    455: "ARROW_STOP",
    456: "ARROW_STOP",
}
SIGNAL_LANES = {431, 432, 443, 445, 446, 447, 448, 449, 450, 455, 456, 457}


def build_fixed_projection(sidecar_path=SIDECAR_PATH, tfrecord=DEFAULT_TFRECORD):
    """Require the exact t=10 record and independently rebuild per-Lane claims."""
    sidecar_path = Path(sidecar_path)
    tfrecord = Path(tfrecord)
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"Saved E5-A Sidecar is missing: {sidecar_path}")
    if not tfrecord.is_file():
        raise FileNotFoundError(f"Original WOMD TFRecord is missing: {tfrecord}")
    source = sidecar_path.read_text(encoding="utf-8")
    context = {
        "expected_scenario_id": SCENARIO_ID,
        "expected_record_index": 0,
        "expected_time_index": 10,
        "expected_current_time_index": 10,
        "expected_record_sha256": RECORD_SHA256,
    }
    structural = project_sidecar(source, **context)
    files_checked = project_sidecar(
        source, evidence_base_dir=RESEARCH_DIR, **context)
    rebuilt = project_sidecar(
        source, evidence_base_dir=RESEARCH_DIR,
        real_tfrecord=tfrecord, **context)
    levels = [entry["verification_level"] for entry in
              (structural, files_checked, rebuilt)]
    if levels != ["structure_only", "evidence_files_consistent",
                  "fixed_scene_lane_rebuilt"]:
        raise ValueError("Projection verification levels differ")
    if len({entry["source_sidecar_sha256"] for entry in
            (structural, files_checked, rebuilt)}) != 1:
        raise ValueError("Projection levels used different Sidecar contents")
    tokens = rebuilt["tokens"]
    relations = rebuilt["candidate_relations"]
    if (len(tokens) != 12 or {item["raw_lane_id"] for item in tokens}
            != SIGNAL_LANES or len(relations) != 30):
        raise ValueError("Fixed Scenario signal/road inventory differs")
    names = {item["raw_lane_id"]: item["state_name"] for item in tokens}
    if any(names[lane] != state for lane, state in STOP_LANES.items()):
        raise ValueError("Fixed Scenario STOP/ARROW_STOP states differ")
    if (Counter(item["state_name"] for item in tokens) !=
            {"STOP": 4, "ARROW_STOP": 2, "UNKNOWN": 6}):
        raise ValueError("Fixed Scenario complete signal states differ")
    if (rebuilt["model_condition_injected"] is not False or
            rebuilt["row_index_stage"] != "pre_training_random_permutation"):
        raise ValueError("Projection cannot claim model injection or shuffled rows")
    unsupported = [item for item in relations
                   if item["relation_status"] == "unsupported"]
    unknown_unsupported = sum(item["state_name"] == "UNKNOWN"
                              for item in unsupported)
    filtered_stop_side = sum(
        item["state_name"] in ("STOP", "ARROW_STOP") and
        item["road_branch"] == "partitioned" and
        item["row_stop_identity_status"] == "filtered"
        for item in unsupported)
    if (len(unsupported), unknown_unsupported, filtered_stop_side) != (
            18, 12, 6):
        raise ValueError("Fixed Scenario unsupported relation evidence differs")
    if any(item["masks"]["tensor_control_identity_proven"]
           for item in relations):
        raise ValueError("Final Stop identity cannot be proven by E4 evidence")
    return rebuilt, levels, hashlib.sha256(sidecar_path.read_bytes()).hexdigest()


def render_report(projection, levels, sidecar_file_sha256, tfrecord=DEFAULT_TFRECORD):
    """Report all six stop-associated Lane rows without upgrading provenance."""
    tokens = projection["tokens"]
    relations = projection["candidate_relations"]
    state_counts = Counter(item["state_name"] for item in tokens)
    status_counts = Counter(item["relation_status"] for item in relations)
    branch_counts = Counter(item["road_branch"] for item in relations)
    lines = [
        "# E5-B：固定真实场景离线候选条件投影",
        "",
        (f"Scenario `{projection['scenario_id']}`，record `{projection['record_index']}`，"
         f"t=`{projection['time_index']}`，current=`{projection['current_time_index']}`，"
         f"官方源码 commit `{projection['official_repo_commit']}`，原始 payload SHA-256 "
         f"`{projection['record_payload_sha256']}`；Sidecar "
         f"`{projection['source_sidecar_schema_version']}`。"
         "本结果仅是离线研究候选关系，`model_condition_injected=false`。"),
        "",
        "## 输入与可信等级",
        "",
        f"- 原始 TFRecord：`{Path(tfrecord)}`，record 0。",
        f"- 已存 Sidecar：`{SIDECAR_PATH}`，文件 SHA-256 `{sidecar_file_sha256}`；"
        f"规范化内容 SHA-256 `{projection['source_sidecar_sha256']}`。",
        ("- 依次完成 `structure_only`（契约结构，行号仅是 Sidecar 声明）、"
         "`evidence_files_consistent`（E4 证据文件哈希）、"
         "`fixed_scene_lane_rebuilt`（从原始 TFRecord 和 E4 证据逐 Lane 重建）。"
         f"本次保存等级：`{projection['verification_level']}`；"
         f"实际执行等级：`{' → '.join(levels)}`。"),
        (f"- 原始信号 Token {len(tokens)} 条：STOP {state_counts['STOP']}、"
         f"ARROW_STOP {state_counts['ARROW_STOP']}、UNKNOWN {state_counts['UNKNOWN']}。"
         "UNKNOWN 是显式 LaneState，不能视为 GO；无 LaneState 不生成 Token。"),
        (f"- 原始图 Lane {len(projection['raw_lane_ids'])} 条，其中 "
         f"{len(projection['signal_absent_raw_lane_ids'])} 条在 t=10 "
         "没有 LaneState；缺席 ID 已在投影 JSON 单独列出。"),
        (f"- 候选道路关系 {len(relations)} 条：Regular {branch_counts['regular']}、"
         f"Partitioned {branch_counts['partitioned']}；状态 "
         + "、".join(f"{name} {count}" for name, count in
                    sorted(status_counts.items())) + "。"),
        ("- `unsupported` 18 条由两类组成：六条 STOP/ARROW_STOP 的非 Stop 侧 "
         "Partition 子行局部来源及 Stop 身份为 `filtered`；另 12 条是六条显式 "
         "UNKNOWN 信号的 Regular/Partition 关系，Anchor 与 Stop FoV 尚未审计。"
         "UNKNOWN 不表示 GO。"),
        "",
        "## 六条 STOP/ARROW_STOP Lane 的逐行证据",
        "",
        ("| 原 Lane / 状态 | Group | 原 Lane→Compact 半开区间 | Anchor 顶点 | "
         "分支 / 子 ID / 行 | 局部 Lane 来源 | "
         "FoV Stop 来源 / 行内 Stop 身份 | 关系 | 离线候选资格 Mask | "
         "可用条件 Mask | 最终 Stop 身份 Mask | 控制身份已证明 Mask |"),
        "|---|---:|---|---:|---|---|---|---|---|---|---|---|",
    ]
    for relation in relations:
        lane = relation["raw_lane_id"]
        if lane not in STOP_LANES:
            continue
        branch = (f"Regular / — / {relation['tensor_row']}" if
                  relation["road_branch"] == "regular" else
                  f"Partitioned / {relation['child_id']} / {relation['tensor_row']}")
        masks = relation["masks"]
        source = relation["source_point_range"]
        compact = relation["compact_point_range"]
        lines.append(
            f"| {lane} / {relation['state_name']} | {relation['group_id']} | "
            f"[{source[0]},{source[1]})→[{compact[0]},{compact[1]}) | "
            f"{relation['stop_anchor']['shared_compact_vertex_index']} | "
            f"{branch} | {relation['local_lane_source_status']} | "
            f"{relation['fov_stop_source_status']} / "
            f"{relation['row_stop_identity_status']} | "
            f"{relation['relation_status']} | "
            f"{str(masks['candidate_relation_eligible']).lower()} | "
            f"{str(masks['condition_candidate_usable']).lower()} | "
            f"{str(masks['final_stop_identity_verified']).lower()} | "
            f"{str(masks['tensor_control_identity_proven']).lower()} |")
    lines.extend([
        "",
        ("六条 Lane 的 Stop Anchor 均由 E4-D 在各自 Compact Group 内验证唯一；"
         "这只支持候选关联。各组只有原始 Lane 的局部区间受控，"
         "STOP 不扩散到整个 Group。六条非 Stop 侧 Partition 子行虽然存在，"
         "其局部来源和行内 Stop 身份均为 `filtered`，因此关系为 `unsupported`。"),
        "",
        "## Token Mask 汇总",
        "",
        "| Mask | true / 12 |",
        "|---|---:|",
    ])
    for name in tokens[0]["masks"]:
        true_count = sum(item["masks"][name] is True for item in tokens)
        lines.append(f"| `{name}` | {true_count} |")
    lines.extend([
        "",
        ("`candidate_relation_eligible` 只是离线候选资格；"
         "`condition_candidate_usable` 还要求真实逐 Lane 复核和最终 Stop 身份证明。"
         "真实 E4 证据没有证明最终 20 点 Stop 原始身份；"
         "本场景的可用条件、最终 Stop 身份与 `tensor_control_identity_proven` "
         "均为 `false`。"
         "所有行号均处于训练随机排列之前的阶段，不能直接当作训练批次行号。"),
        "",
        "## 复现与边界",
        "",
        "- 生成命令：`python -m research.traffic_logic.e5b_real_projection`。",
        "- 完整测试命令：`python -m unittest discover -s research/traffic_logic -p \"test_*.py\"`。",
        f"- 运行环境：`{platform.platform()}`，Python `{sys.version.split()[0]}`，CPU 只读诊断。",
        ("- 本次复核只针对 v1.0.0 固定 t=10 原始记录和既存 E4 文件；"
         "其他时刻或 Scenario 必须取得对应时间的道路映射与逐 Lane 证据。"
         "表中的原 Lane/Compact 区间位于 FoV 裁剪及两次重采样之前，"
         "不能当作最终 20 点的有效来源区间。"
         "局部来源 `partial/uncertain`、FoV Stop 来源 `uncertain` "
         "与最终 20 点 Stop 身份 `uncertain` 均维持原判。"
         "没有运行官方模型、训练或 GPU，也没有条件注入。"),
    ])
    return "\n".join(lines) + "\n"


def main():
    projection, levels, sidecar_sha = build_fixed_projection()
    output = json.dumps(projection, ensure_ascii=False, sort_keys=True,
                        indent=2, allow_nan=False) + "\n"
    report = render_report(projection, levels, sidecar_sha)
    OUTPUT_PATH.write_text(output, encoding="utf-8")
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"scenario={projection['scenario_id']} tokens={len(projection['tokens'])} "
          f"relations={len(projection['candidate_relations'])} "
          f"verification={projection['verification_level']} "
          f"model_condition_injected={projection['model_condition_injected']}")
    print(f"json={OUTPUT_PATH}")
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
