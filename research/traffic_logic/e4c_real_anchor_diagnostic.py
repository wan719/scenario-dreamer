"""Read-only E4-C audit of the six real E4-B Stop Point records.

The input JSON was captured from one E4-A/E4-B official processing instance.
This tool does not reread TFRecord or rerun Compact/FoV/Partition/resampling.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .e4c_stop_anchor import audit_e4b_stop_anchor


SCENARIO_ID = "637f20cafde22ff8"
RECORD_SHA256 = "7af5edce831d6aec3eae0f665ee137b3eaf97a949cec6c9ea1e0127e1be46442"
OFFICIAL_COMMIT_PREFIX = "6754234"
EXPECTED_STATES = {
    443: "STOP", 445: "STOP", 448: "STOP", 449: "STOP",
    455: "ARROW_STOP", 456: "ARROW_STOP",
}


def audit_result(document):
    """Reject mixed scenarios; derive every Group and row from E4-B evidence."""
    source = document["input"]
    if (document["run_status"] != "pass" or
            source["scenario_id"] != SCENARIO_ID or
            source["record_index"] != 0 or
            source["time_index"] != 10 or
            source["record_payload_sha256"] != RECORD_SHA256 or
            not document["environment"]["repo_commit"].startswith(
                OFFICIAL_COMMIT_PREFIX)):
        raise ValueError("E4-B evidence is not the verified scenario/run")
    cases = document["lanes"]
    ids = [case["raw_lane_id"] for case in cases]
    if len(ids) != len(EXPECTED_STATES) or set(ids) != set(EXPECTED_STATES):
        raise ValueError("E4-B evidence does not contain exactly six target Lanes")
    if any(case["signal_state"] != EXPECTED_STATES[case["raw_lane_id"]]
           for case in cases):
        raise ValueError("E4-B signal states differ from the verified baseline")
    return [audit_e4b_stop_anchor(case) for case in cases]


def render_report(document, anchors, source_path, source_hash):
    rows = []
    verified_ids = [str(anchor.raw_lane_id) for anchor in anchors
                    if anchor.verification_status == "verified"]
    final_uncertain_ids = [str(anchor.raw_lane_id) for anchor in anchors
                           if anchor.final_tensor_stop_provenance_status == "uncertain"]
    for anchor in anchors:
        partition = ", ".join(
            f"{child}→{row}" for child, row in anchor.partition_stop_side_rows)
        rows.append(
            f"| {anchor.raw_lane_id} | {anchor.signal_state} | "
            f"G{anchor.group_id} / {anchor.predecessor_lane_id} / "
            f"{anchor.shared_compact_vertex_index} | "
            f"{anchor.connection_xy_upper_bound_m:.3g} | "
            f"{'保留' if anchor.shared_vertex_fov_retained else '过滤'} | "
            f"{anchor.verification_status} | {anchor.regular_tensor_row} | "
            f"{partition or '无'} | "
            f"{anchor.final_tensor_stop_provenance_status} |")
    source = document["input"]
    official = document["environment"]["repo_commit"]
    return "\n".join([
        "# E4-C：真实 Stop Anchor 显式血缘审计",
        "",
        f"场景 `{source['scenario_id']}`，validation record 0，固定 `t=10`；"
        f"官方源码 `{official[:7]}`。依据同一次 E4-B 正式处理输出 "
        f"`{source_path.name}`（SHA-256 `{source_hash}`），原始 TFRecord record "
        f"SHA-256 `{source['record_payload_sha256']}`。仅审计元数据，不重跑官方流程。",
        "",
        "| 原 Lane | 信号 | Group / 紧邻前驱 / 共享顶点索引 | 上下游 XY 距离上界 m | 共享点 FoV | 选定接点状态 | Regular 行 | Partition Stop 侧子 ID→行 | 最终来源 |",
        "|---:|---|---|---:|---|---|---:|---|---|",
        *rows,
        "",
        f"**已验证：** {len(verified_ids)}/{len(anchors)} 个选定 Compact Anchor"
        f"（Lane {', '.join(verified_ids) or '无'}）。表中 `verified` 项的"
        "前驱与组员顺序一致，原始来源从点 1 开始，共享顶点索引等于"
        "该 Lane 贡献区间起点减 1。E4-B 同次运行实测的 Stop→Lane 首点、"
        "Stop→前驱末点与 Stop→ego Compact 共享顶点的 XY 距离均在"
        " `1e-9 m` 容差内；表中的上界由前两项经三角不等式得到。"
        "`verified` 还要求 Stop 和共享顶点均通过 FoV，仅指选定的"
        " Compact 连接点，不证明整组坐标唯一，也不将 STOP/ARROW_STOP"
        " 扩散到 Group。",
        "",
        f"**不确定：** 最终来源为 `uncertain` 的 Lane："
        f"{', '.join(final_uncertain_ids) or '无'}。这六条原始 Lane 的首点"
        "都由 Compact 的 `polyline[1:]` 省略；本次六条均没有最终 20 点"
        "的原 Lane 身份证明。Regular 行与表中的 Partition Stop 侧行"
        "存在；Partition 另一侧的 Stop 来源为 `filtered`。"
        "共享顶点通过验证不会自动令 `final_tensor_stop_provenance_verified`"
        " 通过：JSON 中该字段为 `null`（未证实）。",
        "",
        "**无法从现有 JSON 独立复核：** 前驱原始端点坐标、整条 Compact"
        " 折线及其全部同 XY 候选未被保存，故本次核验依赖 E4-B 同次运行"
        "记录的实测距离和组员顺序，不能重新计算端点或排除组内其他重合"
        "坐标。六条 JSON Anchor 的 `geometric_candidate_scan_performed=false`。"
        "两次重采样和 Partition 后的 Anchor 身份传播仍需显式来源标签"
        "或同次 live 对象的追加证据。没有新增交通信号模型输入，也没有训练。",
        "",
        "验证命令：`python -m unittest discover -s research/traffic_logic "
        "-p 'test_*.py'`；`git diff --check`。完整逐 Lane Anchor 元数据见"
        " `e4c_anchor_records.json`。",
        "",
    ])


def main():
    output_dir = Path(__file__).absolute().parent / "output"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e4b-json", type=Path,
                        default=output_dir / "e4b_autodl" / "e4b_result.json")
    parser.add_argument("--report", type=Path,
                        default=output_dir / "e4c_report.md")
    parser.add_argument("--anchor-json", type=Path,
                        default=output_dir / "e4c_anchor_records.json")
    args = parser.parse_args()
    raw = args.e4b_json.read_bytes()
    document = json.loads(raw)
    anchors = audit_result(document)
    output = {
        "scenario_id": document["input"]["scenario_id"],
        "record_index": 0,
        "time_index": 10,
        "record_payload_sha256": document["input"]["record_payload_sha256"],
        "official_repo_commit": document["environment"]["repo_commit"],
        "e4b_json_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": "selected_compact_shared_vertex_and_fov_not_final_tensor_provenance",
        "anchors": [asdict(anchor) for anchor in anchors],
    }
    args.anchor_json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.anchor_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    args.report.write_text(
        render_report(document, anchors, args.e4b_json,
                      output["e4b_json_sha256"]), encoding="utf-8")
    for anchor in anchors:
        print(f"Lane {anchor.raw_lane_id}: G{anchor.group_id} "
              f"predecessor={anchor.predecessor_lane_id} "
              f"vertex={anchor.shared_compact_vertex_index} "
              f"shared={anchor.verification_status} "
              f"final={anchor.final_tensor_stop_provenance_status}")
    print(f"report={args.report} anchor_json={args.anchor_json}")


if __name__ == "__main__":
    main()
