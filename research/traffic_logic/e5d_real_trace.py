"""Read-only E5-D fixed-scene Lane-to-row provenance replay on Windows CPU.

Run as ``python -m research.traffic_logic.e5d_real_trace``. This never calls
the official random-time get_data path or makes a training Batch.
"""

import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from .e4d_real_anchor_diagnostic import DEFAULT_TFRECORD, RECORD_SHA256, SCENARIO_ID
from .e5d_official_replay import replay_fixed_scene
from .e5d_row_lineage import (classify_lane_sample,
                              invert_simulated_permutation, trace_resampling)
from .lane_lineage import build_lane_lineage


ROOT = Path(__file__).absolute().parent
OUTPUT = ROOT / "output"
E4A_PATH = OUTPUT / "e4a_autodl" / "e4a_result.json"
E4D_PATH = OUTPUT / "e4d_result.json"
E5C_PATH = OUTPUT / "e5c_simulation.json"
RESULT_PATH = OUTPUT / "e5d_trace.json"
REPORT_PATH = ROOT / "e5d_report.md"
TARGET_LANES = (443, 455, 445, 448, 449, 456)
STOP_STATES = {443: "STOP", 445: "STOP", 448: "STOP", 449: "STOP",
               455: "ARROW_STOP", 456: "ARROW_STOP"}


def _read_evidence(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _context(document):
    return (document["scenario_id"], document["record_index"],
            document["time_index"], document["record_payload_sha256"])


def _status_runs(statuses):
    """Encode every 1000-point status without a bulky repeated JSON object."""
    runs = []
    for index, status in enumerate(statuses):
        if runs and runs[-1][0] == status:
            runs[-1][2] = index + 1
        else:
            runs.append([status, index, index + 1])
    return runs


def _compact_sources(group_id, replay):
    """Assign raw member point indices to this invocation's Compact vertices."""
    members = replay["groups"][group_id]
    processed = replay["processed"]["lanes"]
    lineage = build_lane_lineage(
        {group_id: members}, {member: processed[member] for member in members})
    compact = replay["compact"]["lanes"][group_id]
    sources = [{
        "source_index": index,
        "raw_lane_candidates": [],
        "compact_source_indices": [index],
        "raw_lane_local_point_indices": {},
        "reasons": [],
    } for index in range(len(compact))]
    for member_index, lane_id in enumerate(members):
        source_start, source_end = lineage[lane_id]["source_point_range"]
        compact_start, compact_end = lineage[lane_id]["compact_point_range"]
        if compact_end - compact_start != source_end - source_start:
            raise ValueError(f"Group {group_id} Lane {lane_id} point spans differ")
        for source_offset, compact_index in enumerate(range(compact_start, compact_end)):
            source = sources[compact_index]
            source["raw_lane_candidates"].append(lane_id)
            source["raw_lane_local_point_indices"][str(lane_id)] = [
                source_start + source_offset]
        if member_index:
            # Official concat omits the downstream first point. The shared
            # predecessor vertex is a known connection, not an exclusive
            # geometric witness for either member.
            shared = sources[compact_start - 1]
            shared["raw_lane_candidates"].append(lane_id)
            shared["raw_lane_local_point_indices"][str(lane_id)] = [0]
            shared["reasons"].append("shared_compact_member_vertex")
    if any(not source["raw_lane_candidates"] for source in sources):
        raise ValueError(f"Group {group_id} has an unassigned Compact vertex")
    for source in sources:
        source["raw_lane_candidates"].sort()
    return lineage, sources


def _first_resampling(group_id, sources, replay):
    mask = replay["fov_masks"][group_id]
    kept = np.flatnonzero(mask)
    if len(kept) < 2:
        raise ValueError(f"Group {group_id} lacks two FoV points for this audit")
    result = trace_resampling(
        replay["normalized"]["lanes"][group_id][mask],
        [sources[int(index)] for index in kept],
        replay["fov"]["lanes"][group_id].shape[0],
        require_local_indices=True)
    if not np.array_equal(result["points"], replay["fov"]["lanes"][group_id]):
        raise ValueError(f"Group {group_id} 1000-point replay differs")
    return result


def _branch_trace(group_id, alignment, first, replay):
    branch = alignment["road_branch"]
    if branch == "regular":
        graph_id = group_id
        rows = replay["regular_rows"]
        input_points = first["points"]
        input_sources = first["samples"]
        shared = ()
        parent_range = [0, len(input_points)]
    elif branch == "partitioned":
        graph_id = alignment["child_id"]
        rows = replay["partitioned_rows"]
        plan = replay["partition_plan"][group_id]
        matching = [point_range for child_id, point_range in plan["children"]
                    if child_id == graph_id]
        if len(matching) != 1:
            raise ValueError(f"Group {group_id} child {graph_id} is not unique")
        start, end = matching[0]
        input_points = replay["partitioned"]["lanes"][graph_id]
        if not np.array_equal(input_points, first["points"][start:end]):
            raise ValueError(f"Partition child {graph_id} source slice differs")
        input_sources = first["samples"][start:end]
        shared = tuple(index - start for index in plan["shared_endpoints"]
                       if start <= index < end)
        parent_range = [start, end]
    else:
        raise ValueError(f"Unknown road branch {branch}")
    row = rows["lane_id_to_tensor_row"].get(graph_id)
    if row != alignment["pre_shuffle_tensor_row"]:
        raise ValueError(f"{branch} graph ID {graph_id} row differs from E5-C")
    if row is None:
        return {"graph_id": graph_id, "pre_shuffle_row": None,
                "parent_1000_point_range": parent_range,
                "shared_parent_indices": list(shared), "sample_points": [],
                "per_sample_local_source": [], "per_lane_status_counts": {},
                "source_body_replay_matches_20_points": None}
    result = trace_resampling(
        input_points, input_sources,
        rows["road_points"].shape[1], shared_input_indices=shared,
        require_local_indices=True)
    matches = bool(np.array_equal(result["points"], rows["road_points"][row]))
    if not matches:
        raise ValueError(f"{branch} graph ID {graph_id} 20 points differ from rows")
    return {"graph_id": graph_id, "pre_shuffle_row": row,
            "parent_1000_point_range": parent_range,
            "shared_parent_indices": list(shared),
            "sample_points": result["points"].tolist(),
            "per_sample_local_source": result["samples"],
            "source_body_replay_matches_20_points": matches}


def build_fixed_trace(tfrecord=DEFAULT_TFRECORD):
    """Reconcile one raw record and saved E4/E5 evidence, then trace rows."""
    e4a, e4a_sha = _read_evidence(E4A_PATH)
    e4d, e4d_sha = _read_evidence(E4D_PATH)
    e5c, e5c_sha = _read_evidence(E5C_PATH)
    expected = (SCENARIO_ID, 0, 10, RECORD_SHA256)
    if _context(e4a["input"]) != expected or _context(e4d) != expected or \
            _context(e5c) != expected:
        raise ValueError("E4/E5 evidence is from a different Scenario or timestep")
    if (e5c["official_repo_commit"] != e4a["environment"]["repo_commit"] or
            e5c["model_condition_injected"] is not False or
            e5c["row_index_stage"] != "pre_training_random_permutation" or
            e5c["simulated_permutation_supplied"]):
        raise ValueError("E5-C source, row stage or model boundary differs")
    tokens = e5c["tokens"]
    token_by_lane = {token["raw_lane_id"]: token for token in tokens}
    if len(tokens) != 12 or len(token_by_lane) != 12:
        raise ValueError("Expected 12 distinct observed LaneState tokens")
    for token in tokens:
        if token["state_name"] == "UNKNOWN" and token["new_interface_candidate"]:
            raise ValueError("UNKNOWN was promoted to an available candidate")
        if not token["temporal_condition_available"] and token["new_interface_candidate"]:
            raise ValueError("Unavailable time was promoted to a current candidate")
        if token["new_interface_verified"] or token["e5b_masks"]["tensor_control_identity_proven"]:
            raise ValueError("E5-C unexpectedly claims verified routing or old Stop identity")
    if set(e5c["signal_absent_raw_lane_ids"]) & set(token_by_lane):
        raise ValueError("Absent LaneState was treated as an observed token")
    replay = replay_fixed_scene(tfrecord, e4a, e4d)
    if replay["payload_sha256"] != RECORD_SHA256:
        raise ValueError("Raw Scenario fingerprint differs")
    cases = []
    group_cache = {}
    baseline_lanes = {case["raw_lane_id"]: case for case in e4a["lanes"]}
    anchor_lanes = {case["raw_lane_id"]: case for case in e4d["lanes"]}
    for lane_id in TARGET_LANES:
        token = token_by_lane[lane_id]
        old = baseline_lanes[lane_id]
        anchor = anchor_lanes[lane_id]
        group_id = old["group_id"]
        if (token["state_name"] != STOP_STATES[lane_id] or
                replay["parsed"]["lane_states"].get(lane_id) != token["state_code"] or
                anchor["group_id"] != group_id or
                not anchor["unique_anchor_verified"] or
                not token["anchor_alignment_verified"]):
            raise ValueError(f"Lane {lane_id} state, Group or anchor differs")
        if group_id not in group_cache:
            lineage, sources = _compact_sources(group_id, replay)
            group_cache[group_id] = (lineage, _first_resampling(
                group_id, sources, replay))
        lineage, first = group_cache[group_id]
        if (list(lineage[lane_id]["source_point_range"]) != old["source_point_range"] or
                list(lineage[lane_id]["compact_point_range"]) != old["compact_point_range"]):
            raise ValueError(f"Lane {lane_id} local span differs from E4-A")
        first_statuses = [classify_lane_sample(sample, lane_id)
                          for sample in first["samples"]]
        first_counts = Counter(first_statuses)
        first_ambiguous = [
            {"sample_index": sample["sample_index"],
             "status": first_statuses[index],
             "input_indices": sample["input_indices"],
             "compact_source_indices": sample["compact_source_indices"],
             "raw_lane_candidates": sample["raw_lane_candidates"],
             "reasons": sample["reasons"]}
            for index, sample in enumerate(first["samples"])
            if first_statuses[index] in ("partial", "uncertain")]
        alignments = [alignment for alignment in e5c["alignments"]
                      if alignment["raw_lane_id"] == lane_id]
        if not alignments:
            raise ValueError(f"Lane {lane_id} has no E5-C candidate records")
        branches = []
        for alignment in alignments:
            if (alignment["group_id"] != group_id or
                    alignment["new_interface_verified"] or
                    alignment["e5b_masks"]["tensor_control_identity_proven"]):
                raise ValueError(f"Lane {lane_id} relation was promoted")
            trace = _branch_trace(group_id, alignment, first, replay)
            counts = Counter(classify_lane_sample(sample, lane_id)
                             for sample in trace["per_sample_local_source"])
            trace["per_lane_status_counts"] = dict(sorted(counts.items()))
            if trace["pre_shuffle_row"] is None:
                local_replay_status = "filtered"
            elif counts["retained"] and not counts["partial"] and not counts["uncertain"]:
                local_replay_status = "partial" if counts["filtered"] else "retained"
            elif counts["retained"] or counts["partial"]:
                local_replay_status = "partial"
            elif counts["uncertain"]:
                local_replay_status = "uncertain"
            else:
                local_replay_status = "filtered"
            if (alignment["road_row_alignment_status"] == "filtered" and
                    counts["retained"]):
                raise ValueError(f"Lane {lane_id} filtered side gained direct local points")
            row_count = (replay["regular_rows"]["num_lanes"] if
                         alignment["road_branch"] == "regular" else
                         replay["partitioned_rows"]["num_lanes"])
            synthetic_reverse = invert_simulated_permutation(
                list(reversed(range(row_count))), row_count)
            branches.append({
                "road_branch": alignment["road_branch"],
                "child_id": alignment["child_id"],
                "e5c_alignment_status": alignment["road_row_alignment_status"],
                "new_interface_candidate": alignment["new_interface_candidate"],
                "local_source_replay_status": local_replay_status,
                "synthetic_reverse_post_shuffle_row_example": (
                    synthetic_reverse[trace["pre_shuffle_row"]] if
                    trace["pre_shuffle_row"] is not None else None),
                "actual_post_shuffle_row": None,
                "actual_post_shuffle_row_status": "not_run",
                "actual_batch_row": None,
                "actual_batch_row_status": "not_run",
                "audit": {
                    "pre_shuffle_row_verified": trace["pre_shuffle_row"] is not None,
                    "batch_row_permutation_verified": False,
                    "lane_local_region_verified": False,
                    "legacy_final_stop_identity_verified": False,
                    "new_interface_verified": False,
                    "actual_model_consumption_verified": False,
                },
                **trace,
            })
        cases.append({
            "raw_lane_id": lane_id,
            "signal_state": token["state_name"],
            "group_id": group_id,
            "group_members": replay["groups"][group_id],
            "source_point_range": old["source_point_range"],
            "compact_point_range": old["compact_point_range"],
            "shared_compact_vertex_index": anchor["shared_compact_vertex_index"],
            "anchor_unique_verified": anchor["unique_anchor_verified"],
            "temporal_condition_available": token["temporal_condition_available"],
            "explicit_signal_observed": token["explicit_signal_observed"],
            "new_interface_candidate": token["new_interface_candidate"],
            "new_interface_verified": False,
            "legacy_final_stop_identity_verified": False,
            "fov_mask_kept_compact_indices": [int(i) for i in
                                               np.flatnonzero(replay["fov_masks"][group_id])],
            "first_1000_sample_status_counts": dict(sorted(first_counts.items())),
            "first_1000_sample_status_runs": _status_runs(first_statuses),
            "first_1000_ambiguous_samples": first_ambiguous,
            "branches": branches,
        })
    return {
        "schema_name": "e5d_read_only_row_lineage_audit",
        "schema_version": "0.1.0",
        "execution_scope": replay["execution_scope"],
        "scenario_id": SCENARIO_ID,
        "record_index": 0,
        "time_index": 10,
        "current_time_index": replay["parsed"]["current_time_index"],
        "record_payload_sha256": replay["payload_sha256"],
        "tfrecord_path": str(Path(tfrecord)),
        "runtime": {"platform": platform.platform(),
                    "python": platform.python_version(),
                    "numpy": np.__version__, "gpu_used": False},
        "local_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT.parents[1],
            text=True).strip(),
        "local_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT.parents[1],
            text=True).strip(),
        "official_repo_commit": e4a["environment"]["repo_commit"],
        "official_source_blobs": replay["official_source_blobs"],
        "ego_pose_source": "E4-A fixed-t10 official extracted ego pose; same payload SHA",
        "e4a_json_sha256": e4a_sha,
        "e4d_json_sha256": e4d_sha,
        "e5c_json_sha256": e5c_sha,
        "source_body_replay_matches_saved_official_row_order": True,
        "official_imported_runtime_this_round": False,
        "actual_training_batch_captured": False,
        "model_condition_injected": False,
        "old_tensor_stop_identity_proven": False,
        "observed_signal_token_count": len(tokens),
        "explicit_unknown_token_count": sum(token["state_name"] == "UNKNOWN"
                                            for token in tokens),
        "signal_absent_raw_lane_count": len(e5c["signal_absent_raw_lane_ids"]),
        "focus_lanes_first": [443, 455],
        "extended_lanes": [445, 448, 449, 456],
        "counts": {
            "raw_lanes": len(replay["parsed"]["raw_lanes"]),
            "processed_lanes": len(replay["processed"]["lanes"]),
            "compact_groups": len(replay["compact"]["lanes"]),
            "fov_groups": len(replay["fov"]["lanes"]),
            "partitioned_lanes": len(replay["partitioned"]["lanes"]),
            "regular_pre_shuffle_rows": replay["regular_rows"]["num_lanes"],
            "partitioned_pre_shuffle_rows": replay["partitioned_rows"]["num_lanes"],
        },
        "lanes": cases,
    }


def render_report(result):
    lines = [
        "# E5-D：官方道路行重排与原始 Lane 局部来源只读追踪",
        "",
        ("固定 Scenario `637f20cafde22ff8`、record 0、t=10；"
         f"payload SHA-256 `{result['record_payload_sha256']}`。"
         "本轮在 Windows CPU 以受指纹约束的官方函数源码体研究侧复演；"
         "未正式导入官方数据集运行、未取得 t=10 实际 Batch。"),
        (f"研究分支 `{result['local_branch']}`、HEAD `{result['local_head'][:8]}`；"
         f"E4-A 官方基线 `{result['official_repo_commit'][:8]}`。"
         "Ego 姿态复用同 payload 的 E4-A 固定 t=10 正式提取值。"),
        f"原始输入：`{result['tfrecord_path']}`（record 0）；Windows Python {result['runtime']['python']}、NumPy {result['runtime']['numpy']}，GPU 未使用。",
        "",
        "## 官方链路与行号含义",
        "",
        "- `data_processing/waymo/generate_waymo_dataset.py:319-383`：TFRecord → `MessageToDict`（328）→ `get_road_info`（289）→ `get_lane_graph`（149）→ `process_lanegraph`（162）；`utils/lane_graph_helpers.py:5,130` 的 `get_compact_lane_graph`/`find_lane_groups` 建组并拼接。第一阶段 pickle 只保存 lane graph、objects、av_index。",
        "- `datasets/waymo/dataset_autoencoder_waymo.py:587-710`：`get_data` 原本在 644-647 随机选择有效帧；本审计显式固定 t=10，按 `normalize_compact_lane_graph`（201）→ `get_lane_graph_within_fov`（237，点裁剪、1000 点）→ `partition_compact_lane_graph`（95）→ Regular/Partitioned 各自 `get_road_points_adj`（330，20 点、距离排序、`max_num_lanes` 截断）复演。两次采样都调用 `utils/lane_graph_helpers.py:194`。",
        "- 之后 `datasets/waymo/dataset_autoencoder_waymo.py:877-980` 调用 `normalize_scene`（`utils/data_helpers.py:337`）缩放数值，训练模式在 894 调用 `randomize_indices`（`utils/data_helpers.py:421`）施加随机 Lane 排列，`datamodules/waymo/waymo_datamodule_autoencoder.py:34-47` 的 PyG DataLoader 再拼 Batch，`models/scenario_dreamer_autoencoder.py:187` 消费。Group ID、pre-shuffle Row、post-shuffle Row 与 Batch 全局 Row 是四种编号。排列定义 `lane_perm[new_row]=pre_row`，反查用 `argsort(lane_perm)`；Batch Row 还需加样本前缀偏移。",
        "- 本轮六个官方源码/配置指纹与 E4-A 保存的官方版本相同；原始记录、Group 有序成员/几何、FoV mask、Partition 子数组及两路完整排序均与 E4 证据交叉核对。研究侧复演的 20 点数值没有独立的 t=10 官方缓存或实际 Batch 数值见证。",
        "",
        "## 固定真实场景追踪",
        "",
        "| 原始 Lane | 状态 | Group | Regular pre-row：20 点局部来源计数 | Partition child→pre-row：20 点局部来源计数 | post-shuffle Row | Batch Row |",
        "|---:|---|---:|---|---|---|---|",
    ]
    for case in result["lanes"]:
        regular = next(branch for branch in case["branches"]
                       if branch["road_branch"] == "regular")
        partition = [branch for branch in case["branches"]
                     if branch["road_branch"] == "partitioned"]
        def summary(branch):
            counts = branch["per_lane_status_counts"]
            return (f"{branch['pre_shuffle_row']} "
                    f"(R{counts.get('retained', 0)}/P{counts.get('partial', 0)}/"
                    f"U{counts.get('uncertain', 0)}/F{counts.get('filtered', 0)})")
        partition_text = "; ".join(
            f"{branch['child_id']}→{summary(branch)}" for branch in partition)
        lines.append(f"| {case['raw_lane_id']} | {case['signal_state']} | "
                     f"{case['group_id']} | {summary(regular)} | "
                     f"{partition_text} | 未捕获 | 未捕获 |")
    lines.extend([
        "",
        "R/P/U/F = retained/partial/uncertain/filtered，均指**研究侧逐点来源括号**；JSON 保存 1000 点逐点状态的半开区间编码及歧义点证据、完整 20 点数组和每点原始 Lane/原始点下标候选。跨原 Lane 插值、FoV 缺口、零长段和 Partition 共享端点不会被提升为唯一来源。非 Stop 侧 Partition 子行即使存在，也不广播该 Lane 的控制状态。",
        "",
        "## 证据边界",
        "",
        "| 命题 | 本轮状态 | 根据 |",
        "|---|---|---|",
        "| pre_shuffle_row_verified | 六条目标 Lane 的现存行均 true | E4-A 同一固定场景正式调用捕获的实际排序，与本轮源码体复演相符 |",
        "| batch_row_permutation_verified | false | 无 t=10 的真实训练排列或 Batch 捕获；JSON 的反序示例仅是合成演示 |",
        "| lane_local_region_verified | false | 已给出研究侧逐点候选，但缺少独立官方 t=10 20 点数组与 Batch 行级见证 |",
        "| legacy_final_stop_identity_verified | false | 旧 20 点原始 Stop 身份在 E4-B/E5-C 仍 uncertain |",
        "| new_interface_verified | false | 尚无绑定真实训练排列及目标局部道路行的新增 Token 路由证据；它不以旧 Stop 身份为默认前提 |",
        "| actual_model_consumption_verified | false | 未注入模型，也未运行训练或 GPU |",
        "",
        "`UNKNOWN` 仍是六条显式 LaneState，缺席 LaneState 不生成 Token；未来 `future_label` 不可作为当前在线条件。本场景 12 个 Token 均在 t=10，但新旧条件 Mask 没有被升级。`model_condition_injected=false`。",
        "",
        "## 未验证与下一步",
        "",
        "本机 record 0 的 AE pickle 仅有 t=81 两路缓存，不能替代 t=10；当前 Python 缺 torch/PyG，AutoDL SSH 域名解析失败。取得同一固定场景 t=10 的官方 20 点缓存/观察钩子和真实 `lane_perm`、PyG Batch `lane.ptr`/样本位次后，逐行比对 20 点及来源括号，再核验 pre→post→Batch 路由。保留旧 Stop 身份独立审计，不能凭 Row 存在宣称其已进入旧张量。",
        "",
        "## 复现命令",
        "",
        "`python -m research.traffic_logic.e5d_real_trace`；`python -m unittest discover -s research/traffic_logic -p \"test_*.py\"`；`git diff --check`。本轮只写研究目录下的报告与 JSON，未提交或推送。",
    ])
    return "\n".join(lines) + "\n"


def main():
    result = build_fixed_trace()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render_report(result), encoding="utf-8")
    print(f"lanes={len(result['lanes'])} regular={result['counts']['regular_pre_shuffle_rows']} "
          f"partitioned={result['counts']['partitioned_pre_shuffle_rows']}")
    print(f"result={RESULT_PATH}\nreport={REPORT_PATH}")


if __name__ == "__main__":
    main()
