# E5-A.1 Sidecar v1.1.0 契约

`schema_name` 沿用 `scenario-dreamer.signal-lane-sidecar`，新版本显式写 `schema_version=1.1.0`。v1.0.0 的字段和校验路径仍按 [原契约](e5a_sidecar_schema.md)处理；`loads_sidecar` 根据**文件内版本**分派，不自动转换。现有固定 t=10 JSON 仍是 v1.0.0，E5-A 只读构建器继续输出 v1.0.0；本轮没有提供版本迁移函数或生成其他真实时刻的映射。

## 原始信号枚举

v1.1.0 保存 [Waymo `map.proto`](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/map.proto) 的原始数字和名称，不合并颜色或控制动作。

| 代码 | 名称 | 代码 | 名称 | 代码 | 名称 |
|---:|---|---:|---|---:|---|
| 0 | UNKNOWN | 1 | ARROW_STOP | 2 | ARROW_CAUTION |
| 3 | ARROW_GO | 4 | STOP | 5 | CAUTION |
| 6 | GO | 7 | FLASHING_STOP | 8 | FLASHING_CAUTION |

`signals[]` 只列原始 `LaneState` 中实际存在的信号；原始 Lane ID 在 `raw_lane_ids` 中但不在 `signals` 中表示该时刻**无信号记录**。显式 `UNKNOWN` 是代码 0 的信号记录，既不是 GO，也不是信号缺席。v1.0.0 继续只识别其原有的 0、1、4；其他代码须使用显式 v1.1.0。

## 时间与道路证据

[Waymo `scenario.proto`](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/scenario.proto) 的 `current_time_index` 是观察时刻；`dynamic_map_states[i]` 与 `timestamps_seconds[i]` 对齐。v1.1.0 新增以下根字段：

| 字段 | 约束 |
|---|---|
| `dynamic_map_states_count` | 从原始 Scenario 的 `dynamic_map_states` 长度取得，必须大于零；`time_index` 与 `current_time_index` 都必须位于 `[0, count)`。 |
| `temporal_role` | 严格由两种索引派生：审计时刻较早为 `history`，相同为 `current`，较晚为 `future_label`。 |
| `available_at_observation_cutoff` | `history/current=true`，`future_label=false`。仅说明该标签相对观察分界的时间可见性；**不表示已进入模型**，`model_condition_injected` 仍须为 `false`。 |
| `mapping_evidence_time_index` | 有任一已审计 Compact/道路行时必须等于本文件的 `time_index`；全部未执行道路映射时为 `null`。 |

每条 v1.1.0 信号另有 `mapping_audit_status`：

- `verified`：保留 v1.0.0 的 `compact`、`anchor`、`fov`、`road_tensor` 和分角色证据引用。这里的 `verified` 只表示**道路映射审计已做**，不提升 `stop_identity_status` 或最终 20 点 Stop 身份。E4 证据实际场景、record、时间和 payload 指纹须在 `verify_evidence_files` 中与 Sidecar 一致。
- `not_run`：上述四个映射对象均为 `null`，道路证据引用、`mapping_evidence_time_index`、`row_index_stage` 及道路证据文件列表均为空；原始信号状态与 Stop Point XYZ 仍可保存。这样可以表示其他时刻的信号，而不借用 t=10 的道路行。

v1.1.0 的 `local_lane_source_status` 在 Regular/Partitioned 行上另允许 `retained`，表示局部几何有保留证据。`stop_identity_status`、`anchor.audit_status`、`final_stop_identity_status` 仍只允许 `verified/partial/uncertain/filtered/invalid/not_run`；`retained` 不可用于原始 Stop 身份。v1.0.0 的旧规则保持不变。

`validate_sidecar` 只按声明的 `dynamic_map_states_count` 做结构与范围检查。拿到原始 Proto 字节时，调用 `verify_original_time_axis(document, payload)` 核对 payload SHA-256、Scenario ID、当前索引、动态状态数及时间戳数；调用 `verify_evidence_files(document, research_dir)` 核对证据文件 SHA-256，并对 v1.1.0 **已映射**记录检查 E4-A/B/C/D 的实际时间上下文。它们不取代逐 Lane 几何/来源内容复核；固定 t=10 的 v1.0.0 样例仍由 `verify_real_sidecar` 从原始 record 和 E4 输出重建后逐字段比较。

目标真实场景的两个时间数组各 91 项，`current_time_index=10`，有效审计索引为 0–90。E4-A/B 的真实道路行只在 t=10 验证；本轮没有为其他时刻生成道路映射或 Stop 来源结论，也没有模型条件注入。
