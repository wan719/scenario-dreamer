# E5-D：官方道路行重排与原始 Lane 局部来源只读追踪

固定 Scenario `637f20cafde22ff8`、record 0、t=10；payload SHA-256 `7af5edce831d6aec3eae0f665ee137b3eaf97a949cec6c9ea1e0127e1be46442`。本轮在 Windows CPU 以受指纹约束的官方函数源码体研究侧复演；未正式导入官方数据集运行、未取得 t=10 实际 Batch。
研究分支 `research/traffic-logic`、HEAD `8951efea`；E4-A 官方基线 `67542346`。Ego 姿态复用同 payload 的 E4-A 固定 t=10 正式提取值。
原始输入：`E:\ScenarioDreamer\datasets\raw_womd\v1.1.0\validation\uncompressed_scenario_validation_validation.tfrecord-00000-of-00150`（record 0）；Windows Python 3.9.1、NumPy 1.26.4，GPU 未使用。

## 官方链路与行号含义

- `data_processing/waymo/generate_waymo_dataset.py:319-383`：TFRecord → `MessageToDict`（328）→ `get_road_info`（289）→ `get_lane_graph`（149）→ `process_lanegraph`（162）；`utils/lane_graph_helpers.py:5,130` 的 `get_compact_lane_graph`/`find_lane_groups` 建组并拼接。第一阶段 pickle 只保存 lane graph、objects、av_index。
- `datasets/waymo/dataset_autoencoder_waymo.py:587-710`：`get_data` 原本在 644-647 随机选择有效帧；本审计显式固定 t=10，按 `normalize_compact_lane_graph`（201）→ `get_lane_graph_within_fov`（237，点裁剪、1000 点）→ `partition_compact_lane_graph`（95）→ Regular/Partitioned 各自 `get_road_points_adj`（330，20 点、距离排序、`max_num_lanes` 截断）复演。两次采样都调用 `utils/lane_graph_helpers.py:194`。
- 之后 `datasets/waymo/dataset_autoencoder_waymo.py:877-980` 调用 `normalize_scene`（`utils/data_helpers.py:337`）缩放数值，训练模式在 894 调用 `randomize_indices`（`utils/data_helpers.py:421`）施加随机 Lane 排列，`datamodules/waymo/waymo_datamodule_autoencoder.py:34-47` 的 PyG DataLoader 再拼 Batch，`models/scenario_dreamer_autoencoder.py:187` 消费。Group ID、pre-shuffle Row、post-shuffle Row 与 Batch 全局 Row 是四种编号。排列定义 `lane_perm[new_row]=pre_row`，反查用 `argsort(lane_perm)`；Batch Row 还需加样本前缀偏移。
- 本轮六个官方源码/配置指纹与 E4-A 保存的官方版本相同；原始记录、Group 有序成员/几何、FoV mask、Partition 子数组及两路完整排序均与 E4 证据交叉核对。研究侧复演的 20 点数值没有独立的 t=10 官方缓存或实际 Batch 数值见证。

## 固定真实场景追踪

| 原始 Lane | 状态 | Group | Regular pre-row：20 点局部来源计数 | Partition child→pre-row：20 点局部来源计数 | post-shuffle Row | Batch Row |
|---:|---|---:|---|---|---|---|
| 443 | STOP | 11 | 12 (R8/P0/U0/F12) | 132→19 (R0/P0/U0/F20); 133→20 (R17/P0/U1/F2) | 未捕获 | 未捕获 |
| 455 | ARROW_STOP | 14 | 0 (R9/P0/U0/F11) | 134→1 (R0/P0/U0/F20); 135→0 (R17/P0/U1/F2) | 未捕获 | 未捕获 |
| 445 | STOP | 17 | 5 (R9/P0/U0/F11) | 138→11 (R0/P0/U0/F20); 139→10 (R17/P0/U1/F2) | 未捕获 | 未捕获 |
| 448 | STOP | 28 | 3 (R9/P0/U0/F11) | 140→7 (R0/P0/U0/F20); 141→6 (R17/P0/U1/F2) | 未捕获 | 未捕获 |
| 449 | STOP | 1 | 1 (R9/P0/U0/F11) | 130→3 (R0/P0/U0/F20); 131→2 (R17/P0/U1/F2) | 未捕获 | 未捕获 |
| 456 | ARROW_STOP | 15 | 2 (R9/P0/U0/F11) | 136→5 (R0/P0/U0/F20); 137→4 (R17/P0/U1/F2) | 未捕获 | 未捕获 |

R/P/U/F = retained/partial/uncertain/filtered，均指**研究侧逐点来源括号**；JSON 保存 1000 点逐点状态的半开区间编码及歧义点证据、完整 20 点数组和每点原始 Lane/原始点下标候选。跨原 Lane 插值、FoV 缺口、零长段和 Partition 共享端点不会被提升为唯一来源。非 Stop 侧 Partition 子行即使存在，也不广播该 Lane 的控制状态。

## 证据边界

| 命题 | 本轮状态 | 根据 |
|---|---|---|
| pre_shuffle_row_verified | 六条目标 Lane 的现存行均 true | E4-A 同一固定场景正式调用捕获的实际排序，与本轮源码体复演相符 |
| batch_row_permutation_verified | false | 无 t=10 的真实训练排列或 Batch 捕获；JSON 的反序示例仅是合成演示 |
| lane_local_region_verified | false | 已给出研究侧逐点候选，但缺少独立官方 t=10 20 点数组与 Batch 行级见证 |
| legacy_final_stop_identity_verified | false | 旧 20 点原始 Stop 身份在 E4-B/E5-C 仍 uncertain |
| new_interface_verified | false | 尚无绑定真实训练排列及目标局部道路行的新增 Token 路由证据；它不以旧 Stop 身份为默认前提 |
| actual_model_consumption_verified | false | 未注入模型，也未运行训练或 GPU |

`UNKNOWN` 仍是六条显式 LaneState，缺席 LaneState 不生成 Token；未来 `future_label` 不可作为当前在线条件。本场景 12 个 Token 均在 t=10，但新旧条件 Mask 没有被升级。`model_condition_injected=false`。

## 未验证与下一步

本机 record 0 的 AE pickle 仅有 t=81 两路缓存，不能替代 t=10；当前 Python 缺 torch/PyG，AutoDL SSH 域名解析失败。取得同一固定场景 t=10 的官方 20 点缓存/观察钩子和真实 `lane_perm`、PyG Batch `lane.ptr`/样本位次后，逐行比对 20 点及来源括号，再核验 pre→post→Batch 路由。保留旧 Stop 身份独立审计，不能凭 Row 存在宣称其已进入旧张量。

## 复现命令

`python -m research.traffic_logic.e5d_real_trace`；`python -m unittest discover -s research/traffic_logic -p "test_*.py"`；`git diff --check`。本轮只写研究目录下的报告与 JSON，未提交或推送。
