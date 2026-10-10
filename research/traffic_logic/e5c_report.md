# E5-C：独立 Signal Token 与局部道路对齐离线模拟

固定 Scenario `637f20cafde22ff8`、record `0`、t=`10`；原始 payload SHA-256 `7af5edce831d6aec3eae0f665ee137b3eaf97a949cec6c9ea1e0127e1be46442`，官方源码 `675423469766bf2fd8a6b569ef1869a6f1e76993`。本轮仍为模型外只读模拟，`model_condition_injected=false`。

## 输入与核验

- 原始 TFRecord：`E:\ScenarioDreamer\datasets\raw_womd\v1.1.0\validation\uncompressed_scenario_validation_validation.tfrecord-00000-of-00150`，record 0。
- E5-A Sidecar `output/e5a_sidecar.json`；E4-A/B/D 与 E5-B 保存证据均匹配同一 Scenario、时刻和 payload 指纹，核验等级 `fixed_scene_lane_rebuilt`。
- Ego 姿态来自 E4-A 固定 t=10 官方 agent 提取：center=(-7785.916487577568, -6683.40586769982) m，yaw=-1.5457615000000002 rad。按官方 SE(2) 重算六条 Stop Ego XY，与 E4-B 值的最大差 `5.55e-17 m`。E4-D 的 Anchor 核验使用全局 XY；坐标系未混用。
- 独立 Token `12` 个、Road Alignment `30` 条；新接口候选 `12` 条，已验证 `0` 条。对齐状态：filtered 6、not_run 12、uncertain 12。

## 六条 STOP/ARROW_STOP Lane

| 原始 Lane / 状态 | Group·Anchor | 原始→Compact 局部半开区间 | Stop Ego XY (m) | Regular pre-row / 新对齐 | Partition 子ID→pre-row / 新对齐 | 独立 Token 候选/已验证 | 旧张量身份已证明 |
|---|---|---|---|---|---|---|---|
| 443 / STOP | G11·118 | [1,49)→[119,167) | (12.660, 3.125) | 12 / uncertain | 132→19 filtered, 133→20 uncertain | true/false | false |
| 445 / STOP | G17·119 | [1,111)→[120,230) | (9.306, 3.053) | 5 / uncertain | 138→11 filtered, 139→10 uncertain | true/false | false |
| 448 / STOP | G28·212 | [1,111)→[213,323) | (6.041, 3.168) | 3 / uncertain | 140→7 filtered, 141→6 uncertain | true/false | false |
| 449 / STOP | G1·200 | [1,110)→[201,310) | (2.714, 3.440) | 1 / uncertain | 130→3 filtered, 131→2 uncertain | true/false | false |
| 455 / ARROW_STOP | G14·118 | [1,110)→[119,228) | (-0.436, 3.675) | 0 / uncertain | 134→1 filtered, 135→0 uncertain | true/false | false |
| 456 / ARROW_STOP | G15·118 | [1,99)→[119,217) | (-3.294, 3.702) | 2 / uncertain | 136→5 filtered, 137→4 uncertain | true/false | false |

六条信号的 `explicit_signal_observed`、`temporal_condition_available`、`anchor_alignment_verified`、`raw_lane_scope_known` 均为 `true`。Regular 与 Stop 侧 Partition 只是局部候选；非 Stop 侧虽有行号，其局部来源被过滤。Group 共用不扩散灯态到其他原始 Lane。

## 两条条件证据路线

| 路线 | 已成立 | 仍未成立 |
|---|---|---|
| 旧道路张量身份 | E5-B 的 12 条离线候选关系及所有 Mask 逐项保持原值 | 最终 20 点的原始 Stop 身份仍为 `uncertain`；`tensor_control_identity_proven=false`，可用条件 Mask 为 `false` |
| 新独立 Token | 六条停止信号原始语义、Stop 坐标、时刻、唯一 Anchor、原 Lane 局部范围可独立保存；Regular/Stop 侧 Partition 可列候选 | Token 到具体最终道路行的局部来源尚未由新真实证据唯一证明；`new_interface_verified=false` |

## 限制与复现

- 显式 `UNKNOWN` 六条仍各有 Token，但 Anchor 未同级审计，不可视为 GO；没有 LaneState 的原始 Lane 不生成 Token。`future_label` 仅在合成测试中验证，不能用作当前条件。
- 原 Lane、Compact、FoV 1000 点与最终 20 点属不同采样域，不能直接相交数字区间；Stop 附近的最终 20 点投影不证明来源。没有给出真实训练行随机排列，JSON 的 post-shuffle 行号为 `null`。
- 运行 `python -m research.traffic_logic.e5c_real_simulation`；环境 `Windows-10-10.0.26100-SP0`、Python `3.9.1`，Windows CPU。无模型条件输入、无 AE/LDM 或官方预处理修改、无 GPU/训练。
