# E5-B：固定真实场景离线候选条件投影

Scenario `637f20cafde22ff8`，record `0`，t=`10`，current=`10`，官方源码 commit `675423469766bf2fd8a6b569ef1869a6f1e76993`，原始 payload SHA-256 `7af5edce831d6aec3eae0f665ee137b3eaf97a949cec6c9ea1e0127e1be46442`；Sidecar `1.0.0`。本结果仅是离线研究候选关系，`model_condition_injected=false`。

## 输入与可信等级

- 原始 TFRecord：`E:\ScenarioDreamer\datasets\raw_womd\v1.1.0\validation\uncompressed_scenario_validation_validation.tfrecord-00000-of-00150`，record 0。
- 已存 Sidecar：`D:\OneDrive\documents\Desktop\GitHub_repositories\scenario-dreamer\research\traffic_logic\output\e5a_sidecar.json`，文件 SHA-256 `95cc8e182d524a9cf8b9df3943f569ec4bf2dd3cc4373a79058eeceb7bb908c3`；规范化内容 SHA-256 `ab098eedc94622142b060eac2ef572461af1c1213d94d8ecf54ce443f425dca8`。
- 依次完成 `structure_only`（契约结构，行号仅是 Sidecar 声明）、`evidence_files_consistent`（E4 证据文件哈希）、`fixed_scene_lane_rebuilt`（从原始 TFRecord 和 E4 证据逐 Lane 重建）。本次保存等级：`fixed_scene_lane_rebuilt`；实际执行等级：`structure_only → evidence_files_consistent → fixed_scene_lane_rebuilt`。
- 原始信号 Token 12 条：STOP 4、ARROW_STOP 2、UNKNOWN 6。UNKNOWN 是显式 LaneState，不能视为 GO；无 LaneState 不生成 Token。
- 原始图 Lane 199 条，其中 187 条在 t=10 没有 LaneState；缺席 ID 已在投影 JSON 单独列出。
- 候选道路关系 30 条：Regular 12、Partitioned 18；状态 candidate_only 12、unsupported 18。
- `unsupported` 18 条由两类组成：六条 STOP/ARROW_STOP 的非 Stop 侧 Partition 子行局部来源及 Stop 身份为 `filtered`；另 12 条是六条显式 UNKNOWN 信号的 Regular/Partition 关系，Anchor 与 Stop FoV 尚未审计。UNKNOWN 不表示 GO。

## 六条 STOP/ARROW_STOP Lane 的逐行证据

| 原 Lane / 状态 | Group | 原 Lane→Compact 半开区间 | Anchor 顶点 | 分支 / 子 ID / 行 | 局部 Lane 来源 | FoV Stop 来源 / 行内 Stop 身份 | 关系 | 离线候选资格 Mask | 可用条件 Mask | 最终 Stop 身份 Mask | 控制身份已证明 Mask |
|---|---:|---|---:|---|---|---|---|---|---|---|---|
| 443 / STOP | 11 | [1,49)→[119,167) | 118 | Regular / — / 12 | uncertain | uncertain / uncertain | candidate_only | true | false | false | false |
| 443 / STOP | 11 | [1,49)→[119,167) | 118 | Partitioned / 132 / 19 | filtered | uncertain / filtered | unsupported | false | false | false | false |
| 443 / STOP | 11 | [1,49)→[119,167) | 118 | Partitioned / 133 / 20 | uncertain | uncertain / uncertain | candidate_only | true | false | false | false |
| 445 / STOP | 17 | [1,111)→[120,230) | 119 | Regular / — / 5 | uncertain | uncertain / uncertain | candidate_only | true | false | false | false |
| 445 / STOP | 17 | [1,111)→[120,230) | 119 | Partitioned / 138 / 11 | filtered | uncertain / filtered | unsupported | false | false | false | false |
| 445 / STOP | 17 | [1,111)→[120,230) | 119 | Partitioned / 139 / 10 | uncertain | uncertain / uncertain | candidate_only | true | false | false | false |
| 448 / STOP | 28 | [1,111)→[213,323) | 212 | Regular / — / 3 | uncertain | uncertain / uncertain | candidate_only | true | false | false | false |
| 448 / STOP | 28 | [1,111)→[213,323) | 212 | Partitioned / 140 / 7 | filtered | uncertain / filtered | unsupported | false | false | false | false |
| 448 / STOP | 28 | [1,111)→[213,323) | 212 | Partitioned / 141 / 6 | uncertain | uncertain / uncertain | candidate_only | true | false | false | false |
| 449 / STOP | 1 | [1,110)→[201,310) | 200 | Regular / — / 1 | uncertain | uncertain / uncertain | candidate_only | true | false | false | false |
| 449 / STOP | 1 | [1,110)→[201,310) | 200 | Partitioned / 130 / 3 | filtered | uncertain / filtered | unsupported | false | false | false | false |
| 449 / STOP | 1 | [1,110)→[201,310) | 200 | Partitioned / 131 / 2 | uncertain | uncertain / uncertain | candidate_only | true | false | false | false |
| 455 / ARROW_STOP | 14 | [1,110)→[119,228) | 118 | Regular / — / 0 | uncertain | uncertain / uncertain | candidate_only | true | false | false | false |
| 455 / ARROW_STOP | 14 | [1,110)→[119,228) | 118 | Partitioned / 134 / 1 | filtered | uncertain / filtered | unsupported | false | false | false | false |
| 455 / ARROW_STOP | 14 | [1,110)→[119,228) | 118 | Partitioned / 135 / 0 | uncertain | uncertain / uncertain | candidate_only | true | false | false | false |
| 456 / ARROW_STOP | 15 | [1,99)→[119,217) | 118 | Regular / — / 2 | uncertain | uncertain / uncertain | candidate_only | true | false | false | false |
| 456 / ARROW_STOP | 15 | [1,99)→[119,217) | 118 | Partitioned / 136 / 5 | filtered | uncertain / filtered | unsupported | false | false | false | false |
| 456 / ARROW_STOP | 15 | [1,99)→[119,217) | 118 | Partitioned / 137 / 4 | uncertain | uncertain / uncertain | candidate_only | true | false | false | false |

六条 Lane 的 Stop Anchor 均由 E4-D 在各自 Compact Group 内验证唯一；这只支持候选关联。各组只有原始 Lane 的局部区间受控，STOP 不扩散到整个 Group。六条非 Stop 侧 Partition 子行虽然存在，其局部来源和行内 Stop 身份均为 `filtered`，因此关系为 `unsupported`。

## Token Mask 汇总

| Mask | true / 12 |
|---|---:|
| `time_available` | 12 |
| `state_known` | 6 |
| `anchor_unique_verified` | 6 |
| `stop_coordinate_in_fov` | 6 |
| `candidate_road_row_exists` | 12 |
| `local_lane_source_traceable` | 0 |
| `final_stop_identity_verified` | 0 |
| `candidate_relation_eligible` | 6 |
| `condition_candidate_usable` | 0 |
| `tensor_control_identity_proven` | 0 |

`candidate_relation_eligible` 只是离线候选资格；`condition_candidate_usable` 还要求真实逐 Lane 复核和最终 Stop 身份证明。真实 E4 证据没有证明最终 20 点 Stop 原始身份；本场景的可用条件、最终 Stop 身份与 `tensor_control_identity_proven` 均为 `false`。所有行号均处于训练随机排列之前的阶段，不能直接当作训练批次行号。

## 复现与边界

- 生成命令：`python -m research.traffic_logic.e5b_real_projection`。
- 完整测试命令：`python -m unittest discover -s research/traffic_logic -p "test_*.py"`。
- 运行环境：`Windows-10-10.0.26100-SP0`，Python `3.9.1`，CPU 只读诊断。
- 本次复核只针对 v1.0.0 固定 t=10 原始记录和既存 E4 文件；其他时刻或 Scenario 必须取得对应时间的道路映射与逐 Lane 证据。表中的原 Lane/Compact 区间位于 FoV 裁剪及两次重采样之前，不能当作最终 20 点的有效来源区间。局部来源 `partial/uncertain`、FoV Stop 来源 `uncertain` 与最终 20 点 Stop 身份 `uncertain` 均维持原判。没有运行官方模型、训练或 GPU，也没有条件注入。
