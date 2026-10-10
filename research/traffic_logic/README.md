# E1-B / E1-C: Traffic-Light ↔ Lane 映射诊断

## 用法与范围

`lane_lineage.py` 只依赖 Python 标准库。传入**同一次压缩**中 `find_lane_groups` 得到的有序 `group_id -> [raw_lane_id, ...]` 和原始 `lane_graph['lanes']`，调用 `build_lane_lineage(groups, lanes)`；把某一时刻信号所关联的原始 Lane ID 列表传给 `diagnose_controlled_lanes(lineage, ids)`。工具不读取 TFRecord、不判定灯色，也不将信号状态赋给整个 Group。

原始 Lane 至少须有两个点；规范化后重复的 Group ID（如 `7` 与 `"7"`）会报错。

点区间均为从零开始的半开区间。`source_point_range` 是被保留的原始 Lane 点索引；`compact_point_range` 是压缩后、裁剪与重采样前这些点在 Group 内的索引；`compact_span_range` 额外包含后续 Lane 连接处的前一个 Group 点，仅供定位片段，不能证明两个原始点几何重合。101、102、103 每条三点并入 Group 7 时，102 的保留点是原始 `[1,3)`、Group `[3,5)`，片段跨度是 Group `[2,5)`。因此 102 受红灯控制的报告仍指向原始 Lane 102。

运行：`python -m unittest discover -s research/traffic_logic -p 'test_*.py' -v`

## 源码链与当前缺口

- `collect_data` 用 `MessageToDict(proto)` 读取场景，但最终 pickle 仅写入 `lane_graph`、`objects` 和 `av_idx`；交通灯动态状态没有进入该产物。`process_lanegraph` 把 Lane ID 转成整数，并按车道类型过滤。
- `get_compact_lane_graph` 调用 `find_lane_groups`，以组内首条 Lane 的全部点及后续 Lane 的 `[1:]` 拼接；返回的图只有 Group 几何和边，没有原始成员关系。Group 编号来自当次分组顺序，不能用另一遍运行的编号冒充同一次压缩的 ID。
- `partition_compact_lane_graph` 在场景归一化、FOV 裁剪和重采样之后，按 `y=0` 穿越点切分 Group；新 ID 从当时最大 ID 加一开始，原 Group ID 会被删除。若扩展到 partitioned ID，须在**该次切分时**记录 `new_id -> parent_group_id + scene-polyline [start,end)`，并把与原始 Lane 的重叠区间组合起来。裁剪和重采样已改变点索引，精确回溯还需同步保留坐标/弧长对应关系；仅凭当前缓存无法恢复。

## E1-C：最终张量行

`get_data` 的实际顺序是 Compact → 按自车位置和朝向归一化 → FoV 按点过滤并重采样 → 为 in-painting 另建 Partition 图 → 对 regular 和 partitioned 图分别再次重采样、按距原点最近点排序并截取 `max_num_lanes` 条。缓存仅保留 `road_points` 等张量，不含 Lane ID。加载缓存后，训练模式还通过 `randomize_indices` 随机重排 Lane 行；坐标的后续数值归一化不改变行序。

`map_lane_ids_to_rows(candidate_lane_ids, graph_lane_ids, selected_indices, lane_permutation=None)` 接收候选 ID、进入 `get_road_points_adj` 时 `lg['lanes']` 的**实际迭代顺序**、该调用实际得到的 `closest_lane_ids` 原位置索引，以及可选的训练行排列 `lane_perm`，返回候选与图中 ID 并集的 `{lane_id: row_index | None}`。在 regular 分支，候选 ID 可取 FoV 前的 Group ID：`None` 表示该 ID 未进入最终张量，可能是在 FoV 或数量截断时被过滤。传入的选择索引必须与 `np.argsort` 的实际结果一致，尤其距离相等时不能自行假定稳定顺序。partitioned 分支应使用本次生成的子 Lane ID；父 Group 到子 ID 需单独记录，父 ID 对应 `None` 不能解释为没有子 Lane，Group ID 也不能视为行号。该接口只映射**同一次场景、同一图分支**的 ID，不从现有缓存倒推。

几何血缘的下一阶段接口应逐层保存：原始 Lane ID 与点索引、同次压缩的 Group 成员及拼接偏移；FoV 点掩码对应的**有序原索引和连续段边界**；两次重采样每个输出点对应的输入线段端点、插值权重或弧长位置，并显式处理重复点造成的零长线段；Partition 每个新 ID 的父 ID、切片 `[start,end)` 和共享交叉点；最终图顺序、实际排序/截断索引、训练行排列、场景时间步与 regular/partitioned 类型。FoV 若保留不连续的点，当前实现会把它们相邻重采样，单一原始点区间不足以表达这段新连线。

上述 E1 阶段仅有合成样本；后续已针对固定 Scenario `637f20cafde22ff8`、record 0、t=10 完成 E3–E5-A 的真实证据审计。E5-B 的独立候选投影见 [Schema](e5b_projection_schema.md) 与 [固定场景报告](e5b_report.md)。这些证据仅适用于该场景和时刻，最终 20 点的原始 Stop 身份仍未证实，不代表通用数据链已贯通。
