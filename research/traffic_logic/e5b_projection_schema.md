# E5-B：离线 Signal–Road 候选投影契约

`signal_candidate_projection.py` 从已验证的 Signal–Lane Sidecar 生成独立 JSON 对象；Schema 为 `scenario-dreamer.signal-road-candidate-projection` / `0.1.0`。输出只表达候选关系及审计边界，不写官方缓存或模型输入，`model_condition_injected` 固定为 `false`。

## 输入与上下文

`project_sidecar(source, ...)` 接受 Sidecar 字典或 JSON 字符串，先调用现有契约的序列化/反序列化校验。接受 Sidecar `1.0.0` 和 `1.1.0`，不迁移或改写原件；输出的 `source_sidecar_schema_version` 与 `source_sidecar_sha256` 绑定原件规范序列化内容。可显式要求 Scenario ID、record、信号审计 `time_index`、观察分界 `current_time_index` 和原始 payload SHA-256 一致。输出还保留这些字段、官方源码 commit 与 `row_index_stage`。

输出保留 `raw_lane_ids`，并以 `signal_absent_raw_lane_ids` 明列原始图中本时刻没有 LaneState 的 Lane。v1.0.0 的两个时间索引相同，Token 的 `temporal_role=current`。v1.1.0 依其 `history/current/future_label` 和 `available_at_observation_cutoff` 标志生成时间 Mask；`future_label` 的当前可用 Mask 必为 `false`。`mapping_audit_status=not_run` 仍产生原始信号 Token，但不产生道路关联或真实行号。不能把 t=10 的 E4 道路行移植到其他信号审计时刻。

## Signal Token

每条原始 LaneState 产生一个 Token，保存 `raw_lane_id`、原始 `state_code/state_name`、`signal_observed=true`、`stop_point_global_xyz`、`coordinate_frame=waymo_global_xyz`、两个时间索引及时间角色，并保留 Anchor、局部 FoV、Stop 坐标 FoV、Stop 来源 FoV、最终 Stop 身份的原审计状态。九种 Waymo 枚举 `UNKNOWN`、`ARROW_STOP`、`ARROW_CAUTION`、`ARROW_GO`、`STOP`、`CAUTION`、`GO`、`FLASHING_STOP`、`FLASHING_CAUTION` 各自保留；代码 `0` 的显式 `UNKNOWN` 仍有 Token，无 LaneState 则没有 Token。`state_known` 对 UNKNOWN 为 `false`，不将其当作 GO，也不将其他状态二分类。

## Signal–Road Candidate Relation

每条已映射信号按 `road_branch=regular/partitioned` 保存各道路行，包括被过滤的空行记录。每条 Relation 带 `raw_lane_id`、原始枚举、`group_id`、`tensor_row`、Partition 的 `child_id`、Stop Anchor 顶点及审计状态、局部来源与 Stop 来源状态、证据核验等级、Mask 和 `uncertainty_reasons`。同 Group 的多个原始 Lane 可指向同一个道路行，但仍是各自独立的 Token 和局部 Relation；`group_id` 从不当作 `tensor_row`。

`source_point_range` 是原始 Lane 对 Compact 拼接的源点半开区间；`compact_point_range` 是它贡献给 Group 的半开区间。Partition 的 `parent_upsampled_point_range` 是父 Group **重采样后**的区间，坐标域不同；没有逐点来源映射时，不能把这些区间直接相交以证明 Stop 身份。行号属于 `pre_training_random_permutation` 阶段，不能直接解释为训练随机排列后批次的行号。

`relation_status` 的含义：

| 值 | 条件与解释 |
|---|---|
| `filtered` | 道路行号为空；没有可用候选行。 |
| `unsupported` | 行存在，但 Anchor 或 Stop FoV 未审计，或 Stop/局部来源明确为 `filtered/invalid/not_run`；保留行事实，不授予 Stop 候选资格。 |
| `candidate_only` | 行存在，且未遇上述排除证据；`partial/uncertain` 仍保持不确定，绝不自动证明最终 Stop 来源。 |

## Mask 语义

Token 和 Relation 使用同名的 10 个布尔 Mask。Relation 按本行计算；Token 的行相关 Mask 为其各 Relation 的 `any` 聚合。`candidate_relation_eligible` 表示离线候选资格；`condition_candidate_usable` 是更严格的已核验门槛。两者都不是实际模型条件注入。

| Mask | 为 `true` 的精确含义 |
|---|---|
| `time_available` | 信号审计时刻不晚于观察分界。 |
| `state_known` | 原始状态代码不是显式 UNKNOWN `0`。 |
| `anchor_unique_verified` | Anchor 审计为 `verified` 且唯一性复核为真。 |
| `stop_coordinate_in_fov` | Stop 坐标 FoV 状态为 `retained` 或 `verified`；不代表来源已证明。 |
| `candidate_road_row_exists` | 该行的 `tensor_row` 非空；行号 `0` 有效。结构核验时这只是 Sidecar 的行号声明，未独立核实。 |
| `local_lane_source_traceable` | 行存在且局部 Lane 来源为 `retained` 或 `verified`。 |
| `final_stop_identity_verified` | 本行和整体最终 Stop 身份均为 `verified`，最终来源布尔证明为真，唯一 Anchor、Stop 坐标 FoV、FoV Stop 来源均验证，并达到 `fixed_scene_lane_rebuilt`；局部来源另由独立 Mask 判断。 |
| `candidate_relation_eligible` | 时间可用、状态已知、唯一 Anchor、Stop 坐标在 FoV、道路行存在，且本关系为 `candidate_only`。局部来源 `uncertain` 可保留这一离线候选资格。 |
| `tensor_control_identity_proven` | `final_stop_identity_verified` 且本行局部 Lane 来源可追溯；仍须与时间可用性和信号状态分开判断。 |
| `condition_candidate_usable` | 同时满足 `candidate_relation_eligible` 与 `tensor_control_identity_proven` 的严格门槛；结构或文件哈希核验模式下恒为假。 |

## 核验等级与边界

| `verification_level` | 已做的检查 |
|---|---|
| `structure_only` | Sidecar Schema、内部引用、状态和上下文结构校验。 |
| `evidence_files_consistent` | 再检查所引 E4 文件的 SHA-256；v1.1 已映射记录同时检查证据时间与场景上下文。文件一致仍不证明逐 Lane 映射内容。 |
| `fixed_scene_lane_rebuilt` | 前两层之外，调用固定场景 `verify_real_sidecar`，从原始 TFRecord 和 E4 结果重建并逐字段核对。仅适用于已实现的 `637f20cafde22ff8`、record 0、t=10 审计。 |

只有显式提供 `evidence_base_dir` 才核验文件；逐 Lane 模式还必须提供 `real_tfrecord`。六条 STOP/ARROW_STOP 的 Compact Anchor 已唯一验证，但 E5-A 固定样例中最终 20 点的原始 Stop 身份仍是 `uncertain`；候选 Relation 不把 STOP 扩散至整个 Group，也不因 Partition 行存在而升级 Stop 来源。本模块没有修改 AE、LDM、预处理或权重，没有训练或条件注入。
