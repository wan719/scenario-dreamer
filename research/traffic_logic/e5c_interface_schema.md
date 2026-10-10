# E5-C：独立 Signal Token 与局部道路对齐模拟契约

`signal_interface_simulation.py` 的 `simulate_interface(sidecar_source, *, evidence_base_dir=None, real_tfrecord=None, ego_pose=None, alignment_claims=None, row_permutations=None, expected_...=None)` 只在研究目录中产生离线 JSON 对象。它不修改 Sidecar、E5-B 投影、道路张量或模型接口；顶层 `model_condition_injected` 恒为 `false`。输出分为独立 `tokens` 和逐分支 `alignments`，并记录输入 Scenario、record、审计时间、观察分界、原始记录指纹、证据核验等级及行号阶段。

## 两条证据路径

| 路径 | 已能表达的事实 | 尚需证明的事实 |
|---|---|---|
| 旧道路张量身份 | E5-B 的 `tensor_control_identity_proven`、`condition_candidate_usable` 等 Mask 原样保存在 `e5b_masks`，含义与历史值不变。 | 最终 20 点是否携带原始 Stop 身份；E5-B 固定场景仍为 `uncertain`，不能由行存在或点接近 Stop 推导。 |
| 新独立接口 | 每条 LaneState 单独保留原始枚举、Stop 坐标、时间与原始 Lane 局部范围；可建立有明确局部约束的离线候选对齐。 | Token 到最终道路行的真实局部来源和模型批次位置；本轮没有新证据可将候选提升为已验证对齐。 |

`structure_only` 只确认 Sidecar 自洽；`evidence_files_consistent` 还确认所引 E4 文件的哈希与适用上下文；固定真实场景的 `fixed_scene_lane_rebuilt` 还逐 Lane 对照原始 TFRecord。文件哈希本身不能证明 Sidecar 的行号正确。E5-C 的全局到 Ego 变换及 E4-A/B/D 交叉核对另记在对齐证据中，不篡改 E5-B 的核验等级或 Mask。

## Signal Token

每条**实际出现**的原始 LaneState 生成一个 Token，不按 Compact Group 聚合。Token 保存 `raw_lane_id`、`state_code` 与九种原始 `state_name`、`stop_point_global_xyz`、`coordinate_frame=waymo_global_xyz`、`time_index`、`current_time_index`、`temporal_role`，以及可选的 `stop_point_ego_xy` 和 Ego 姿态来源。它还逐项保留 `mapping_audit_status`、`anchor_audit_status`、`anchor_unique_verified`、局部与 Stop 的 FoV 状态、最终 Stop 身份审计状态。显式 `UNKNOWN`（代码 0）有 Token，`state_known=false`；某 Lane 没有 LaneState 时没有 Token，不能将缺席解释成 UNKNOWN 或 GO。

Token 的状态彼此独立：

| 字段 | 含义 |
|---|---|
| `explicit_signal_observed` | 原始 LaneState 确实存在；不等于状态已知。 |
| `temporal_condition_available` | `time_index <= current_time_index`，且 v1.1.0 Sidecar 标记可在观察分界取得。`future_label` 恒为 `false`。 |
| `anchor_alignment_consistent` | 给定 Stop、共享顶点、局部成员关系及坐标变换的结构或几何一致性检查通过；这是候选条件，不代表独立重建。 |
| `anchor_alignment_verified` | 固定真实场景的原始记录与 E4-A/B/D 证据经新的官方同式 SE(2) 交叉复核通过；不能仅复制 Sidecar 的 Anchor `verified`。 |
| `raw_lane_scope_structurally_consistent` | Sidecar 中有原 Lane 的成员、源点及 Compact 局部区间，且半开区间长度相容；合成样本也可满足。 |
| `raw_lane_scope_known` | 已由同场景、同时间的真实分组和血缘证据核对原始 Lane 的局部 Compact 范围；仅有 Group ID 或自报区间不充分。 |

Token 可独立保存观察到的信号，即使其道路行被过滤或映射尚未执行。状态代码不预先压缩为红绿二分类。`mapping_audit_status=not_run` 时没有带真实行号的对齐。

## 局部 Road Alignment

每条 E5-B Regular 或 Partitioned 关系分别生成一条局部对齐记录，携带 `raw_lane_id`、`group_id`、`road_branch`、`child_id`、`source_point_range`、`compact_point_range`、`parent_upsampled_point_range`、共享 Anchor 下标及证据等级。`pre_shuffle_tensor_row` 是 E4-A `get_road_points_adj` 距离排序和数量截断后的行号；Group ID 不等于该行号。局部控制范围属于**原始 Lane**，同 Group 的其他 Lane 不继承其 STOP 或 ARROW_STOP。

`road_row_alignment_status` 使用审计状态：`not_run` 表示未执行该行的局部核验，`filtered` 表示候选行或 Stop 侧局部来源被过滤，`uncertain` 表示有候选行但采样来源不能唯一归因，`partial` 表示只能证实部分局部来源，`verified` 必须有新增的逐行真实血缘证据，`invalid` 表示上下文、拓扑或几何冲突。本轮固定场景的合格候选仍为 `uncertain`；非 Stop 侧且来源 `filtered` 的 Partition 行为 `filtered`；尚未审计 Anchor 的 UNKNOWN 行为 `not_run`。这些值不改写 E5-B `relation_status`。

`new_interface_candidate` 只表示值得继续验证的离线假设：信号已观察、时间可用、状态已知、Anchor/原始局部范围在当前证据下结构相容、Stop 坐标在 FoV 内、该分支存在未被过滤且局部审计已执行的候选行。它独立计算，可在旧最终 Stop 身份未审计时成立；结构型合成样本也可能有候选，但此时 `raw_lane_scope_known=false`。它不要求或暗示旧 20 点 Stop 身份已证明。`new_interface_verified` 需另有与该原始 Lane、分支、子 ID、具体最终行、局部几何和时间绑定的**新真实对齐证据**；不能从 `new_interface_candidate`、E5-B 的候选 Mask、Anchor 唯一性或 Stop 点与 20 点的距离复制为真。本轮没有这种新证据，所有 `new_interface_verified` 均为 `false`。每条对齐记录原样保存其 E5-B `e5b_masks`，供两条路径对照。

## 坐标、采样域与行号

全局 Stop XYZ 来自原始 Scenario Proto。Ego 中心和 yaw 必须绑定相同 Scenario 与 t；官方 `normalize_compact_lane_graph` 使用 `angle = pi/2 + sign(-yaw) * abs(yaw)`（等价于 `pi/2 - yaw`）及 `apply_se2_transform`。E5-C 用这一变换重算 Stop Ego XY，再与 E4-B 保存值和 E4-D 全局 Anchor 对照，记录来源、容差与残差。不能把全局 XY 直接同 Ego XY 比较。`get_road_points_adj` 的 Ego 米空间 20 点在模型数据路径上还会经 `normalize_scene` 缩放；未经对应变换的坐标不得互比。

`source_point_range` 是原始 Lane 源点半开区间，`compact_point_range` 是 Compact 拼接后的贡献区间，`parent_upsampled_point_range` 是 FoV 后父 Group 的 1000 点采样区间。裁剪可能使保留点不连续，拼接省略下游首点，Partition 可共享端点。不同采样域的数字区间不能直接相交以证明局部 Stop 归属；没有逐点来源血缘时保持 `uncertain`。

可选 `row_permutations` 仅模拟某次训练排列，必须按 Regular/Partitioned 分支提供 `row_count`、完整无重复且长度匹配的 `new_to_pre` 双射，并绑定具体处理上下文。固定真实场景还要求 `row_count` 与 E4-A 官方该分支实际行数一致；纯结构型合成样本的行数仅是调用者声明，不能当作真实模型批次证明。官方训练路径满足 `lane_perm[new_row] = pre_shuffle_row`，因此 `simulated_post_shuffle_row = argsort(lane_perm)[pre_shuffle_tensor_row]`。没有排列时 post 行号为 `null`；即使给出合成排列，post 行号也只是离线模拟，不能冒充任意真实模型批次的行号。可选 `alignment_claims` 每条均须带 Scenario、record、time、payload 指纹，并逐字段匹配既有原始 Lane/Group/子 ID/行号/局部区间；重复、缺失或不匹配均拒绝。关系键 `(raw_lane_id, road_branch, child_id)` 唯一。

## 固定场景的审计边界

Scenario `637f20cafde22ff8`、record 0、t=10 的 E5-B 输入有 12 个 Token、30 条关系；六条 STOP/ARROW_STOP Lane 的 Compact Anchor 已被 E4-D 验证组内唯一。E4-B 给出全局及 Ego Stop 坐标，E4-A 给出两路 pre-shuffle 行号。E5-C 可核对这六条的显式 Token 与局部候选对齐；非 Stop 侧 Partition 行应保持 `filtered`。现有文件未提供足以证明最终 20 点逐点原始 Stop 来源及具体训练批次排列的完整证据，因此本轮不宣称新接口已可用于模型，也不生成真实模型条件输入。
