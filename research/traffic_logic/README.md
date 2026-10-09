# E1-B: Traffic-Light ↔ Lane 最小映射验证

## 用法与范围

`lane_lineage.py` 只依赖 Python 标准库。传入**同一次压缩**中 `find_lane_groups` 得到的有序 `group_id -> [raw_lane_id, ...]` 和原始 `lane_graph['lanes']`，调用 `build_lane_lineage(groups, lanes)`；把某一时刻信号所关联的原始 Lane ID 列表传给 `diagnose_controlled_lanes(lineage, ids)`。工具不读取 TFRecord、不判定灯色，也不将信号状态赋给整个 Group。

点区间均为从零开始的半开区间，位于压缩后、裁剪与重采样前的 polyline。`source_point_range` 是被保留的原始 Lane 点索引；`compact_point_range` 是这些点在 Group 内的索引；`compact_span_range` 额外包含后续 Lane 连接处的前一个 Group 点，仅供定位片段，不能证明两个原始点几何重合。101、102、103 每条三点并入 Group 7 时，102 的保留点是原始 `[1,3)`、Group `[3,5)`，片段跨度是 Group `[2,5)`。因此 102 受红灯控制的报告仍指向原始 Lane 102。

运行：`python -m unittest discover -s research/traffic_logic -p 'test_*.py' -v`

## 源码链与当前缺口

- `collect_data` 用 `MessageToDict(proto)` 读取场景，但最终 pickle 仅写入 `lane_graph`、`objects` 和 `av_idx`；交通灯动态状态没有进入该产物。`process_lanegraph` 把 Lane ID 转成整数，并按车道类型过滤。
- `get_compact_lane_graph` 调用 `find_lane_groups`，以组内首条 Lane 的全部点及后续 Lane 的 `[1:]` 拼接；返回的图只有 Group 几何和边，没有原始成员关系。Group 编号来自当次分组顺序，不能用另一遍运行的编号冒充同一次压缩的 ID。
- `partition_compact_lane_graph` 在场景归一化、FOV 裁剪和重采样之后，按 `y=0` 穿越点切分 Group；新 ID 从当时最大 ID 加一开始，原 Group ID 会被删除。若扩展到 partitioned ID，须在**该次切分时**记录 `new_id -> parent_group_id + scene-polyline [start,end)`，并把与原始 Lane 的重叠区间组合起来。裁剪和重采样已改变点索引，精确回溯还需同步保留坐标/弧长对应关系；仅凭当前缓存无法恢复。

真实验证下一步需要获授权可用的原始 Scenario TFRecord（含 map feature 与逐时刻信号状态）、其对应的原始 `lane_graph`/场景标识，以及同次压缩和分割产生的成员与区间记录。当前只有合成样本，无法报告真实映射覆盖率或称数据链已贯通。
