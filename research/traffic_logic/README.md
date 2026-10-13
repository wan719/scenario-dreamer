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

## E6：真实DiT信号残差与短步学习实验

`signal_lane_adapter.py`将九状态one-hot、归一化Stop XY和有效点Mask选择的局部几何均值组成13维输入，经13→64→DiT道路hidden维度的MLP输出残差；末层零初始化。无有效点、inactive信号和不受支持的道路行不产生残差；同一行的多个有效信号取均值。

`LaneSignalResidual`在现有lane_embedder输出后临时挂接分支，进入原有位置编码与注意力；退出上下文会移除hook。原AE/LDM源码和checkpoint格式保持原状。新增adapter权重单独保存，不会自动接入原eval.py。

输入必须来自已核验的单一regular场景record0/t=10。服务器已有AE→LDM排序、53个有效点、预训练strict加载和EMA去噪证据；实验入口再次检查源码blob、checkpoint/AE来源SHA、原始与归一化latent、固定道路输入、时间步及Mask。

测试（PyTorch 2.2、PyG 2.6.1、OmegaConf、Hydra、torch-ema）：
```bash
python -m unittest research.traffic_logic.test_signal_lane_adapter research.traffic_logic.test_e6_signal_adapter_probe -v
```

服务器实验：
```bash
OMP_NUM_THREADS=1 /root/autodl-tmp/conda-envs/scenario-dreamer/bin/python -m research.traffic_logic.e6_signal_adapter_probe \
  --baseline-dir /root/autodl-tmp/e5e_t10_2xe3rd_i/model_resume_3744mega \
  --steps 5 --learning-rate 0.001 --device cuda
```

输出到同级`signal_adapter_*`目录：`signal_adapter_report.json`记录基线一致性、真实hook调用、梯度、条件响应与冻结检查；`adapter_probe.pt`保存adapter参数、实际预测、残差及训练噪声。

2026-10-13 A100原模型重复性检查发现FP32道路预测两两最大绝对差4.083e-6。因此仅预测输出比较改用rtol=1e-5、atol=1e-5，输入latent、源码及SHA证据仍按原规则核验。容差只用于数值一致性，不用于宣称条件有效。

条件响应采用原版、训练后、合成GO三组各3次前向；车辆输出的组内两两最大差记为jitter，门槛为max(1e-5, 5×三组最大jitter)。原版→训练后、训练后→合成GO的全部跨组配对差异（每对取最大绝对差）的最小值都必须超过门槛。否则保存已训练adapter与预测，报告signal_adapter_probe_no_state_response。三组重复预测、实测波动、门槛和差异均留存；本条件只是保守接口探测，三次重复不能给出统计显著性或泛化结论。

成功标志`signal_adapter_probe_verified`仅表示模型实际读取条件、分支可学习且对合成灯态扰动有响应。训练目标是固定时间步、AE均值latent、固定道路下的车辆epsilon MSE，并非官方完整训练协议。GO扰动是接口敏感性测试；单场景六条停止类信号不能证明交通遵循、生成质量或闯红灯率，代码不执行完整采样。原模型参数冻结，`weights_updated`仅指adapter。

## E7：已训练adapter的完整采样与AE解码对照

`e7_signal_sampling_probe.py`复用E6核验过的EMA权重、输入与已保存adapter，运行原`LDM.p_sample_loop(mode='lane_conditioned')`全部扩散步，采用checkpoint原有CFG和时间步设置。条件调用挂接信号，CFG无条件调用不挂接；退出或失败恢复原模型。原采样循环、AE与坐标还原源码未修改。E6加载逻辑抽成`load_verified_baseline`共用，不重复训练。

同一道路、同一个seed比较原版、原停止类信号adapter、合成GO adapter，额外跑一次原版重复作为累计数值波动参照。逐调用核对Torch CPU/CUDA RNG状态、初始车辆噪声及最终RNG状态；每步恢复的道路latent必须精确等于给定道路。三个条件只改变adapter启用及active灯态，初始和各步采样噪声保持配对。

通过原`AutoEncoder.forward_decoder`及`unnormalize_latents/unnormalize_scene`还原物理状态。保存CSV（位置米、速度米/秒、环绕处理的朝向角差），以及生成场景/latent、完整随机状态摘要和SVG场景图。车辆对比按latent slot，不视为真实track身份。解码道路可能因车辆latent交互而偏移，报告同时保留道路差异、FoV外点、非正尺寸和近零方向向量诊断。

```bash
OMP_NUM_THREADS=1 /root/autodl-tmp/conda-envs/scenario-dreamer/bin/python -m research.traffic_logic.e7_signal_sampling_probe \
  --baseline-dir /root/autodl-tmp/e5e_t10_2xe3rd_i/model_resume_3744mega \
  --adapter-dir /root/autodl-tmp/e5e_t10_2xe3rd_i/signal_adapter_x2zy1bvr \
  --seed 20261013 --device cuda
```

输出`signal_sampling_*`：`sampling_report.json`、`generated_scenes.pt`、`generated_normalized_latents.pt`、`sampling_rng_trace.json`、`agent_slot_differences.csv`及baseline/stop/go/comparison.svg。SVG直接打开；不要求Matplotlib、CairoSVG或图像生成服务。

`signal_sampling_pipeline_verified`只验收完整采样、配对、冻结与解码/输出链路。响应差异与一次原版重复波动的比较是描述性结果，不是统计验证，生成质量标志始终false。当前adapter只在一个场景t_agent=50/t_lane=0、固定噪声上学习5步，在其他扩散步的表现可能退化；不能预先保证GO更快、红灯遵循或泛化。官方eval.py不自动消费adapter；本次完整接入限定在新研究入口。E6历史结果仍由原入口维护。
