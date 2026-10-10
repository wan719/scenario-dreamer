# E5-A Signal–Lane Sidecar 契约

`schema_name=scenario-dreamer.signal-lane-sidecar`、`schema_version=1.0.0`。这是独立研究 JSON；每份文件只对应一个 `(scenario_id, record_index, time_index)`，不改变道路张量或模型接口。`model_condition_injected` 在本版本必须为 `false`。

## 字段和来源

| 层级 | 字段 | 含义与证据 |
|---|---|---|
| 根 | `scenario_id`, `record_index`, `time_index`, `current_time_index`, `record_payload_sha256` | 固定原始 Scenario 与时间；SHA-256 取 record 0 的 Proto payload 原始字节。每条信号的 `context` 必须完全一致。 |
| 根 | `official_repo_commit`, `official_source_blobs` | E4-A/B 已记录的官方源码 commit 与六个 Git blob SHA-1；生成时交叉核对 E4-D 的 Compact 源码 blob。 |
| 根 | `evidence_paths`, `evidence_sha256` | E3-C、E4-A/B/C/D 已保存文件的相对路径及文件 SHA-256。`verify_evidence_files(document, research_dir)` 重新读取文件核验；单纯反序列化只验证结构，不代表证据文件仍在。 |
| 根 | `raw_lane_ids`, `signals` | 原始图的 Lane ID 清单及当前帧实际存在的 LaneState 列表。某 ID 在前者但不在后者表示**无信号记录**，与原始枚举 `UNKNOWN` 不同。 |
| 根 | `coordinate_frame`, `row_index_stage` | Stop Point 与 Anchor 使用 `waymo_global_xyz`；行号取训练随机排列前的 `pre_training_random_permutation` 阶段。 |
| `signals[]` | `raw_lane_id`, `signal.presence/state_code/state_name/stop_point_global_xyz` | 从原始 Proto 读取的 LaneState。代码 `4=STOP`、`1=ARROW_STOP`、`0=UNKNOWN`；不把 UNKNOWN 解释为绿灯。Stop Point 原始全局 XYZ 不转换为张量来源断言。 |
| `signals[].compact` | `group_id`, `ordered_member_lane_ids`, `source_point_range`, `compact_point_range` | E4-A 的同组有序成员、该原始 Lane 的半开点区间 `[start,end)` 及拼接后区间。下游 Lane 首点被拼接省略，因此区间不单独证明 Stop Anchor 来源。Group ID 不是张量行。 |
| `signals[].anchor` | `applies_to_raw_lane_id`, `audit_status`, `predecessor_lane_id`, `shared_compact_vertex_index`, `shared_vertex_global_xy`, `selected_vertex_consistent`, `topology_verified`, `geometric_candidate_vertex_indices/count`, `unique_anchor_verified`, `xy_tolerance_m` | 只作用于指定原始 Lane。六条停止信号引用 E4-D 的唯一候选、相邻拓扑及 1e-9 m 容差复核；其余六条 UNKNOWN 的 Anchor 审计为 `not_run`，候选字段为 `null`。不能把 STOP 广播到整组。 |
| `signals[].fov` | `local_contribution_status`, `stop_point_coordinate_status`, `stop_source_status` | 前者来自 E4-A 的原始 Lane 局部几何证据。后两者分别表示 E4-B 的 Stop Point **坐标**在 FoV 内，以及原始 Stop **来源**通过 FoV 的可归因程度。本场景六条停止信号依次为 `retained` / `uncertain`；拼接省点后坐标可见不等于来源已证实。 |
| `signals[].road_tensor` | `regular`, `partitioned[]`, `final_stop_identity_status`, `final_tensor_stop_provenance_verified` | E4-A 实际距离排序和截断后的 Regular 行、Partition 子 ID 与行、父折线上采样点半开区间。每一路分开存 `row_status`、`local_lane_source_status`、`stop_identity_status`。行存在只证明道路几何行存在，不证明最终 20 点包含原始 Stop 身份；该来源目前为 `uncertain` 或未审计。 |
| `signals[].evidence_refs` | `raw_signal`, `group_and_rows`, `anchor`, `stop_fov`, `final_stop` | 每条声明的证据角色分别是原始 payload、E4-A、E4-D、E4-B、E4-C。未审计的字段使用 `null`，不可借用其他角色文件填充。 |

## 状态语义

审计状态是 `verified`（有直接可核对证据）、`partial`（只覆盖部分）、`uncertain`（来源无法唯一归因）、`filtered`（在指定阶段被过滤）、`invalid`（输入或几何无效）、`not_run`（该项未执行审计）。FoV 字段另允许 `retained` 表示观察到几何或点通过视野；这不是原始身份验证。道路行的 `row_status` 仅为 `retained` 或 `filtered`，过滤行的 `tensor_row=null`。`final_tensor_stop_provenance_verified` 对非 `verified` 状态必须为 `null`，不可用 `false` 掩盖尚未证实的原因。

`validate_sidecar` 检查版本、单场景上下文、状态枚举、原始 Lane 引用、同组成员及行的一致性、Anchor 声明和证据角色；`dumps_sidecar`/`loads_sidecar` 作无损 JSON 往返，并拒绝重复 JSON 键或非有限数。读取外来 JSON 时调用者须传预期场景、record、时间和 payload 指纹，并另调用 `verify_evidence_files` 验证文件摘要。**结构校验加文件 SHA 不等于逐 Lane 证据内容复核。**本固定场景可调用 `verify_real_sidecar(document, tfrecord)`：它从原始 record 与保存的 E4 结果重新构造 Sidecar，逐字段比较，能发现事后把 `uncertain` 改为 `verified` 或改写 UNKNOWN 的伪声明。现有真实样例只涵盖固定场景 `637f20cafde22ff8` 的 `t=10`；其他场景须建立对应的真实证据构建器，不能沿用本场景的结论。
