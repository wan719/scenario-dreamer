# E5-A.1：Signal–Lane Sidecar 兼容性加固

分支 `research/traffic-logic`，基线 HEAD `7aa5b68`。本轮只改研究目录；未修改官方预处理、AE、LDM、Checkpoint，也未训练、调用 GPU、提交或推送。

## 变更与版本策略

| 问题 | 处理 |
|---|---|
| v1.0.0 只识别 0、1、4 | 显式新增 v1.1.0，按 [官方 map.proto](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/map.proto) 保留 0–8 九个独立枚举；UNKNOWN 与无信号记录仍分开。旧版按原有三种代码验证。 |
| 局部几何来源 `retained` 被误拒 | v1.1.0 的 Regular/Partitioned `local_lane_source_status` 单独允许 `retained`；Stop 身份、Anchor 审计和最终身份仍不接受该值。过滤行不能声称局部来源保留。v1.0.0 校验语义不变。 |
| `time_index` 被强制等于 `current_time_index` | v1.1.0 用 `dynamic_map_states_count` 限定两个索引范围，并严格派生 `history/current/future_label` 与 `available_at_observation_cutoff`。未来标签的可见性必须为 false，`model_condition_injected` 始终为 false。 |
| t=10 道路证据可能被改写成其他时刻 | v1.1.0 将已执行道路映射绑定到 `mapping_evidence_time_index`；`verify_evidence_files` 除 SHA-256 外还核对 E4-A/B/C/D 的实际场景、record、时间和 payload 指纹。没有对应时刻映射时，`compact/anchor/fov/road_tensor`、道路引用与行阶段均为 `null`，`mapping_audit_status=not_run`。 |

v1.0.0 的版本常量、只读 t=10 构建器和既存 JSON 均保持原样；加载时按文件内版本分派，**不自动迁移**。v1.1.0 字段及条件见 [新 Schema](e5a1_sidecar_schema.md)，原 [v1.0.0 Schema](e5a_sidecar_schema.md) 保留。

## 原始数据与回归

[官方 scenario.proto](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/scenario.proto) 将 `current_time_index` 作为观察分界，并要求动态状态与时间戳按索引对齐。原始目标 record 只读核对得到两种时间数组各 **91** 项、`current_time_index=10`，有效审计索引为 **0–90**。`verify_original_time_axis` 可用 Proto 原始字节复核该长度、场景 ID、当前索引及 payload 摘要。E4-A/B 的道路行仅在 **t=10** 验证；本轮没有将其用于其他真实时刻。

旧版真实样例 `output/e5a_sidecar.json` 保持 v1.0.0，文件 SHA-256 仍为 `95cc8e182d524a9cf8b9df3943f569ec4bf2dd3cc4373a79058eeceb7bb908c3`。它与原始 record/E4 输出重建结果逐字段相同：12 条原始信号、六条 STOP/ARROW_STOP 的已验证 Anchor、两路道路行、FoV 与来源状态均未改变，`model_condition_injected=false`。

## 验证与限制

运行 `python -m unittest discover -s research/traffic_logic -p "test_*.py"`：**133 项通过**。新增测试覆盖九枚举、非法代码、局部 `retained` 与 Stop 身份反例、历史/当前/未来、越界索引、原始 Proto 时间轴、未来标签可见性、E4 证据时间错配、v1.0.0/v1.1.0 往返及旧版真实样例。`git diff --check` 通过；新文件额外检查了行尾空格。最终工作区状态见任务回复。

本轮没有生成其他真实时刻的 Sidecar 道路映射。v1.1.0 允许写入 `local_lane_source_status=retained`，但结构与文件上下文校验**不独立证明**该 Lane 的局部几何来源；仍需逐 Lane 证据复核。固定 t=10 的既有 `verify_real_sidecar` 继续承担该场景逐字段审计。E5-B 条件 Tensor 投影尚未开始。
