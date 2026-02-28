# TwinBrain — AI Agent Instructions
> **This file must be read before any other file in every session.**

---

## 一、身份与角色定位

你是一个拥有海量知识储备的高级研究伙伴，而非执行指令的工具。请始终以此自我定位：

- **不是劳工**：拒绝只做最低限度的完成任务或简单修复 bug。
- **是思考者**：理解项目本质目标，主动识别深层问题。
- **是设计者**：提出系统性的架构改进，而非局部补丁。
- **是合作者**：必要时从根本上挑战和改进用户的想法。

## 二、行为准则

1. **优先理解目标**：先思考"这个问题的本质是什么？是否有更优雅的解法？"
2. **稳定性优先**：所有优化必须可验证、稳定，不引入新风险。
3. **主动创新**：发现更优解时主动提出并说明理由。
4. **诚实评估**：用户方案有根本性缺陷时，明确指出并给出替代方案。
5. **记录错误**：每次发现值得记录的 bug 或陷阱，立即写入本文件§三。

## 三、错误记录（Error Log）

> **记录原则**：重在**思维上的盲区**，而非具体报错信息。每条记录应能回答：*"下次遇到类似问题，我应该先想到什么？"*

---

### [2026-02-20] CUDA OOM：序列长度 + 时间循环两个叠加根因

**思维误区**：以为"截断序列 = 内存优化"，没有追问为什么要 T 个时间步逐一调用 propagate()。

**根因（双重）**：
1. EEG 以 250Hz 采样，完整序列 T≈75000，产生 `[N, T, 128]` 巨张量。
2. ST-GCN 对每个时间步独立 `propagate()`，T=300×4层×3边 = 3600 次 CUDA 内核启动，Python 循环是串行瓶颈（GPU 实际空闲）。
3. 显存碎片化：reserved 但 unallocated 的块不足以响应新分配。

**正确思路**：
- `max_seq_len` 截断全序列；`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 消碎片；`empty_cache()` 每 epoch 归还块。
- 时间循环改为"时序虚拟节点"向量化：将 edge_index 扩展 T 次，单次 propagate 处理 T×E 条消息，GPU 满载，理论加速 100×。

---

### [2026-02-21] "声明了但从未真正工作"的功能（通用反模式，多次复现）

**思维误区**：看到代码里有某功能的声明/参数，就以为它在工作。

**正确思路**：对每个"已实现"的功能，追问三点：
1. 调用入口存在吗？（`set_epoch` 在 `train_epoch` 里有没有被调用？）
2. 内部实现能执行吗？（`autograd.grad` 在 `backward` 后会崩溃）
3. 形状与调用者匹配吗？（predictor 接 4-D 还是 3-D？）

**具体案例（已修复）**：
- `AdaptiveLossBalancer.update_weights` 用 `autograd.grad`，但 `backward()` 已释放计算图 → 崩溃；`set_epoch()` 从未调用 → warmup 永不结束 → 上述 bug 反而被保护。
- `use_gradient_checkpointing` 调用 `HeteroConv.gradient_checkpointing_enable()`，该方法不存在。
- `modality_energy_ratios` 存入 `AdaptiveLossBalancer` 但从不参与计算（死存储）。
- `v5_optimization` 配置块中所有参数均被硬编码默认值覆盖，从未从 YAML 读取。
- EEG handler 里的 `eeg_info['regularization_loss']` 从未加入 `total_loss`。

**规则**：凡是"存储但不使用"的属性，立即问：*这个值应该在什么时候、以什么方式被使用？*

---

### [2026-02-21] fMRI 实际只有 1 个节点（架构上的空话）

**思维误区**：看到"图原生"就以为 fMRI 真的有空间结构，没有打印 N_fmri。

**根因**：`process_fmri_timeseries` 对所有体素 `mean(axis=0)` 后 `reshape(1, -1)` → N_fmri=1，空间信息完全丢失。

**修复**：用 `NiftiLabelsMasker` 应用 Schaefer200 图谱，提取 ~190 个 ROI 时序（实际数量因 EPI 覆盖而异，少于 200 是正常的，不要填充）。

---

### [2026-02-21] 跨模态边的两个叠加 N 不匹配 bug

**思维误区**：只修了可见的 guard，没有追问 guard 能否被触发。

**根因**：
1. `update()` 无条件做 `aggr_out + lin_self(x_self)`。跨模态边 N_src≠N_dst，PyTorch 广播导致输出 N=N_src（错误值），污染后续所有层。
2. `propagate()` 未传 `size=(N_src, N_dst)`，PyG 默认 `size=(N_src, N_src)`，使 `aggr_out` 形状本身就错误，guard 永远不会触发。

**修复**：① `update()` 检查 `aggr_out.shape[0] != x_self.shape[0]` 时跳过 self-connection；② `propagate()` 显式传入 `size`；③ `compute_loss` 在 `huber_loss` 前 raise RuntimeError 防止广播静默掩盖。

---

### [2026-02-21] EEG enhancement 原地修改 data 导致跨 epoch "backward 两次" 错误

**思维误区**：以为 `.to(device)` 会重新创建张量，切断梯度链。实际上 `.to()` 对已在目标设备的张量是 no-op。

**根因**：`data['eeg'].x = eeg_x_enhanced` 永久改写了 `data_list` 里的原始对象。Epoch 1 的 `backward()` 释放了 saved tensors，Epoch 2 在已释放的图上构建新图 → 报错。

**修复**：`try-finally` 块保证步骤结束后（无论异常）恢复 `original_eeg_x`。`original_eeg_x` 仅在**实际修改前**才保存（non-None ↔ 已修改）。

---

### [2026-02-21] log_weights 参与 backward 图导致梯度累积

**思维误区**：看到 `nn.Parameter` 就以为"有梯度是正常的"，没有追问这个梯度会被使用吗。

**根因**：`log_weights` 通过 `update_weights()` 手动更新，完全不依赖 `.grad`。让它参与 backward 图只有负面效果：浪费显存、梯度无限累积。

**规则**：对**手动更新**的参数（不通过 optimizer），forward 中用 `.detach()` 排除；向后处理函数传入张量时用 `.detach()` 明确语义。

---

### [2026-02-21] Decoder ConvTranspose1d(stride=1) 悄悄改变时序长度

**正确思路**：ConvTranspose1d 设计用于上采样（stride>1）。stride=1 时输出长度 = `T + kernel_size - 2*padding`（不等于 T）。**stride=1 应用 Conv1d，不应用 ConvTranspose1d。**

---

### [2026-02-23] 硬编码顺序假设破坏 EEG→fMRI 方向

**思维误区**：`edge_types.append((node_types[0], 'projects_to', node_types[1]))`，以为 EEG 一定排第一。

**规则**：有方向性的设计意图，永远用**模态名**而非位置索引确定方向：`if 'eeg' in node_types and 'fmri' in node_types: append(('eeg', 'projects_to', 'fmri'))`。

---

### [2026-02-24 / 2026-02-26] HeteroConv 内部 key 不可预期 → 改用 nn.ModuleDict

**思维误区**：以为 PyG `HeteroConv.convs` 是标准 Python dict，用 `'__'.join(edge_type)` 可以直接访问。

**根因**：`HeteroConv.convs` 是 PyG 自定义 `ModuleDict`，其 `to_internal_key()` 实现因版本而异，导致存入 key=A，读取时得到 B → KeyError。

**规则**：对任何第三方库容器，不能假设其访问语义与标准 dict 相同。正确做法：用 `nn.ModuleDict({'__'.join(et): conv})` 替换 `HeteroConv`，避免任何隐式 key 转换。

---

### [2026-02-24] 替换 self.model 后 optimizer 参数组失效

**根因**：`super().__init__(model=base_model)` 后立即 `self.model = new_model`，optimizer 在构造时捕获参数快照，不随 `self.model` 更新。增强模块参数有梯度但永远不会被更新。

**规则**：每次 `self.model = new_model`，必须重建 optimizer（或 `add_param_group()`）。

---

### [2026-02-25] 1:N EEG→fMRI 对齐是「设计缺失」而非「设计完成」

**场景**：2 个 EEG 条件（GRADON/GRADOFF）共享 1 个 fMRI 扫描（task-CB）。

**思维误区**：`_discover_tasks()` 把"BIDS 文件名中出现的所有 task"等同于"有效 run"，导致出现 fMRI-only 幽灵任务，GRADON/GRADOFF 静默回退到"任意 bold"文件。

**修复**：
1. `fmri_task_mapping` 支持显式映射（V5.15）。
2. `_load_fmri()` 新增 ON/OFF 后缀自动剥离（V5.16，GRADON→GRA 匹配 GRA bold）。
3. `_discover_tasks()` 改为只扫描 EEG 文件，彻底杜绝幽灵任务。
4. 查找链：显式 mapping → 同名直接匹配 → ON/OFF 剥离重试 → 任意 bold 回退。

---

### [2026-02-26] 缓存命中路径的 `continue` 绕过必要副作用

**通用规则**：每次在主循环中加 `continue`/`return`/`break`，都必须问：**循环尾部有没有必须执行的副作用（metadata 写入、索引赋值）？** 如有，必须在 `continue` 前补全。

**案例**：缓存命中直接 `continue` 跳过了 `subject_idx`、`run_idx`、`task_id`、`subject_id_str` 的赋值，导致从缓存加载的图缺少个性化嵌入索引和训练数据追踪信息。

---

### [2026-02-26] 异质图「框架使用完整」≠「特性使用完整」

**排查方法**：为每个 `HeteroData` 属性画「SET → READ」追踪表，找出只 SET 从未 READ 的属性。

**发现（已修复）**：
- `eeg.labels`、`fmri.labels`、`eeg.atlas_mapping`、`fmri.pos`：死存储，never READ。
- 跨模态边无 `edge_attr`（同模态边有相关性权重，不一致）。
- DTI 接口完全缺失：无 `_load_dti()`、无 `('fmri','structural','fmri')` 边类型。

**修复原则**：DTI 不设独立节点类型（无时序特征），而是在 fMRI 节点上新增结构连通性边 `('fmri','structural','fmri')`。编码器已有 `if edge_type in edge_index_dict` 保护，DTI 文件缺失时自动降级。

---

### [2026-02-26] V5.21 全流程审查四个隐患

1. **`max_grad_norm` 静默忽略**：YAML 中存在该参数，代码里却硬编码 1.0。修复：`GraphNativeTrainer.__init__` 新增 `max_grad_norm` 参数，`train_model` 传入。

2. **Checkpoint 不保存调度器状态**：恢复训练时 LR 从头重启。**规则**：恢复训练必须保存所有有状态组件：model + optimizer + scheduler + loss_balancer + history。修复：`save_checkpoint` 新增 `scheduler_state_dict`。

3. **无消息时残差连接 ≈ 2x**：`messages=[]` 时 `x_new = x`，然后无条件执行 `x_new = x + dropout(x_new)` → 幅度 ×2，4 层后 ×16。修复：有消息时 `x + dropout(mean(messages))`，无消息时直接 `x`（passthrough）。

4. **窗口模式按窗口划分训练/验证集 → 数据泄漏**：相邻窗口 50% 重叠，同 run 的窗口同时进入两集。修复：引入 `run_idx`，以 run 为单位划分，保证同 run 所有窗口进入同一集合。

---

### [2026-02-26] SpatialTemporalGraphConv 注意力归一化用 global softmax

**根因**：`softmax(alpha, dim=0)` 对形状 `[E, 1]` 是跨所有 E 条边归一化，而非每目标节点独立。E≈4000 时每条边权重≈0.00025，消息几乎消失，GNN 退化为每节点独立 MLP。

**修复**：用 `pyg_softmax(alpha, index, num_nodes=size_i)` 实现每目标节点归一化。

---

### [2026-02-26] 配置文件路径是"死字符串"（Atlas 案例）

**规则**：配置文件中的文件路径每次改动后必须和实际文件系统一一对照验证。`exists()` 检查只是运行时保护，不能替代"写正确的路径"。警告被忽视会导致系统长期静默运行错误配置（Atlas 未找到 → N_fmri=1，空间信息全丢）。

---

### [2026-02-27] 跨条件边界时序污染（ON/OFF 共享 fMRI 场景）

**问题**：GRADON/GRADOFF 共享同一 fMRI 文件，截断和窗口均从 t=0 开始，导致 GRADOFF 条件使用 GRADON 时段的 fMRI 数据；窗口模式下还会跨越条件边界（模型被迫学习"预测实验条件切换"）。

**修复**：新增 `fmri_condition_bounds` 配置项，在图构建前（连通性估计前）先切片 fMRI 时序到当前条件的时间段。缓存 key 包含此配置，修改边界自动失效。

```yaml
fmri_condition_bounds:
  GRADON:  [0,   150]   # CB fMRI TR 0–149
  GRADOFF: [150, 300]   # CB fMRI TR 150–299
```

1:1 标准场景设 `null`，行为不变。

---

### [2026-02-27] 多被试多任务联合训练三项静默缺陷

> `graphs` 列表中每个 `HeteroData` 均为独立的 (被试, 任务) 图，不存在跨被试/跨任务的节点或特征混合。`train_epoch` 以 batch_size=1 逐一独立处理。

1. **`train_epoch` 缺少逐 epoch 打乱**：`train_model` 只打乱一次，之后每 epoch 顺序相同，optimizer 动量导致末尾被试获得更新偏差。修复：`random.Random(epoch).shuffle(copy)` 每轮独立打乱。

2. **EEG handler 通道数跨被试不匹配**：handler 按第一个样本初始化，不同 N_eeg 的被试静默传入错误形状张量。修复：记录 `_eeg_n_channels`，不匹配时跳过增强并 debug 日志。

3. **`task_id` / `subject_id_str` 未存储**：图构建后来源信息丢失，无法验证训练数据分布。修复：在 `build_graphs` 和缓存路径均写入这两个属性，`extract_windowed_samples` 传播到窗口，`log_training_summary` 展示每任务/被试样本数。

---

### [2026-02-27] 预测器以节点为 batch 维度 → 节点间完全独立，刺激单脑区无法传播

**思维误区**：以为"图模型 + 时间序列预测"就等于"系统级预测"。实际上 `EnhancedMultiStepPredictor(h)` 把 `h[N, T, H]` 的 N 视为 batch 维度，对 N 个节点完全独立预测——图拓扑在预测阶段完全缺失。

**根因**：编码器（ST-GCN）在编码阶段通过图边传递了跨区域信息，但预测头对每个节点单独运行 Transformer，图结构在预测时被丢弃。刺激节点 A 只改变 A 的历史表征 → 只影响 A 的预测 → 其他节点预测轨迹完全不变。

**正确思路**：预测分两步：
1. **Per-node 时间预测**（`EnhancedMultiStepPredictor`）：各节点独立预测初始轨迹
2. **系统级图传播**（`GraphPredictionPropagator`）：在预测空间 `{node_type: [N, pred_steps, H]}` 上运行 N 轮 ST-GCN 消息传递，让预测变化通过图拓扑传播

**规则**：任何"对节点集合的预测"，都必须追问：*预测之后，图连通性如何影响相邻节点？* 如果没有图传播步骤，预测就不是系统级的。

**修复版本**：V5.25。`GraphPredictionPropagator`（`num_prop_layers=2`，覆盖 ≥2 跳邻居）注册为 `model.prediction_propagator`，在 `forward()` 和 `compute_loss()` 的预测步骤后均调用。

---

### [2026-02-27] AdaptiveLossBalancer 双重修正 + 梯度方向反转

**思维误区1（双重修正）**：误以为「能量初始化」+「按初始损失归一化」是两个独立增强，实为同一问题的双重修正，叠加后结果远超预期。

**根因**：
- 能量初始化：`w_eeg = 50 × w_fmri`（补偿 EEG 振幅小 50 倍）
- 初始损失归一化：`loss_norm = loss / L0`（L0_eeg ≈ 0.001，L0_fmri ≈ 0.5）
- 叠加效果：EEG 梯度比例 = 50/0.001 = **50,000**，fMRI = 1/0.5 = 2 → **25,000 倍失衡**，训练完全由 EEG 主导，fMRI 实质上被忽略

**思维误区2（方向反转）**：把"减小困难任务的权重，避免其主导梯度"误写为 GradNorm 的正确方向。

**根因**：GradNorm 的正确逻辑是：收敛慢的任务（相对损失 > 均值）→ **提高**权重 → 模型分配更多梯度信号 → 追上进度。现有代码取负号，困难任务权重被**降低**，导致模型放弃困难任务（fMRI），只学习简单任务（EEG）。

**修复**：
1. `forward()`：移除初始损失归一化，仅保留能量初始化（单一修正）
2. `update_weights()`：(a) 计算相对困难度时先除以 `initial_losses` 做量纲归一化；(b) 将符号改正：`weight_update = +lr * (rel_loss - 1.0)`

**规则**：能量初始化（解决振幅差异）和 GradNorm 自适应更新（解决收敛速度差异）是正交机制，不能叠加到同一个归一化步骤中。

---

### [2026-02-27] 训练优化：四个关联盲区（V5.28）

**问题 1 — validate() 返回值变更后调用方同步规则**

`validate()` 从 `float` 改为 `Tuple[float, Dict]` 后，所有调用方必须同步解包，否则在 `np.isnan(val_loss)` 等处触发 `TypeError: ufunc 'isnan' not supported for the input types`。

**规则**：改变函数返回类型前，先用 grep 找出所有调用方，确保全部同步修改。

**问题 2 — AMP scaler 在梯度累积边界的调用顺序**

`scaler.step()` 和 `scaler.update()` 必须成对调用，且只在真正执行 `optimizer.step()` 的那一步（边界步）调用。`scaler.unscale_(optimizer)` 也应仅在边界步调用，否则梯度被提前反缩放导致数值不稳定。

**规则**：梯度累积中，AMP scaler 的 `unscale_/step/update` 三个调用均受 `do_optimizer_step` 条件保护。

**问题 3 — SWA 对 AveragedModel 的属性访问**

`AveragedModel` 是一个 `nn.Module` 包装器，`forward()` 正确透传，但自定义属性（如 `use_prediction`、`compute_loss`）不自动代理，直接访问会 `AttributeError`。

**规则**：凡需要访问被 AveragedModel 包装的原始模型属性或方法，应先 `_orig_model = trainer.model`（AveragedModel 创建前），然后通过 `_orig_model` 访问，不通过 `swa_model` 访问。

**问题 4 — EEG 频谱相干性改变 edge_index，必须纳入缓存 key**

将 `eeg_connectivity_method` 从 `'correlation'` 切换到 `'coherence'` 会产生完全不同的连边和权重。若缓存 key 不包含此参数，旧缓存（用 correlation 权重）会被 coherence 模式无声使用，导致图结构静默错误。

**规则**：任何影响 edge_index 或 edge_attr 的参数（包括连通性计算方法、阈值、k 近邻数）必须全部纳入 `_graph_cache_key()` 的哈希计算。

---

### [2026-02-27] 训练科学可信度三项盲区（V5.29）

**盲区 1 — R² < 0 无报警（最严重）**

`validate()` 返回 R²，但训练循环仅将其格式化为日志字符串，没有任何语义级判断。
R² < 0 表示模型重建误差 > 信号总方差，即比"永远预测均值"还差——这是模型科学失效的明确信号，非专业用户无法从裸数字识别。

**正确思路**：每次验证后检查所有模态的 R²，若任一为负则打印 ⛔ 警告并解释含义（"尚未从数据中学到有效信号"）。

**修复**：在 validation 代码块内，遍历 `r2_dict`，`if r2v < 0: logger.warning(⛔...)`。

---

**盲区 2 — 最佳 epoch 的 R² 未被追踪**

`best_val_loss` 被追踪并在恢复日志中显示，但对应的 R²（评估模型重建意义的关键指标）未被记录。
用户仅能知道"最低损失是 X"，无法知道"最低损失时模型实际重建能力如何"。

**修复**：引入 `best_r2_dict: dict = {}`，在 `val_loss < best_val_loss` 时同步 `best_r2_dict = r2_dict.copy()`；写入"保存最佳模型"日志行和恢复日志行。

---

**盲区 3 — 训练结束无可信度总结**

训练结束后日志末尾只有"最佳验证损失: X.XXXX"。用户面对一个数字无法判断：
1. 模型是否真正有效（R² > 0）；
2. 哪个模态重建可信、哪个不可信；
3. 是否存在过拟合风险（val/train 比率）。

**修复**：训练循环结束后打印"训练可信度摘要（📊）"，含：
- 每个模态的 R² 及评级（✅ ≥0.3 / ⚠️ 0-0.3 / ⛔ <0）；
- 过拟合检测（`val_loss/train_loss > 3.0` 时警告）；
- `val_frequency × patience` 等效 epoch 耐心值（训练开始前打印）；
- 综合结论一行。

**规则**：任何向非专业用户报告的数字，必须附带含义解释；任何可能失败的科学指标（R² < 0），必须有明确告警而非默认忽略。

---

### [2026-02-27] 跨模态边存入缓存导致的双重问题（V5.30 修复）

**思维误区**：认为"缓存图应完整保存所有边"——包括跨模态边。

**根因（两个叠加问题）**：
1. `k_cross_modal` 进入了缓存键哈希 → 修改 k 会使所有旧缓存失效，只有原始数据在才能重建 → 若用户只保留了缓存文件（已删原始数据），修改 k 导致数据无法恢复。
2. 下游代码加载缓存后看到 `graph.node_types = ['eeg', 'fmri']`，无直接方法获取 numpy 数组形式的时序数据，需要了解 PyG HeteroData API。

**规则**：
- **只缓存"原始数据衍生物"**（节点特征和同模态边）；不缓存"特征的函数"（跨模态边，代价低、参数化、随配置变化）。
- 如果一个图属性 X 满足 `X = f(cached_node_features, config_params)`，X 不应写入缓存；应在每次加载时重建。
- 判断一个参数是否应进入缓存键：若该参数改变只需调用已缓存特征的计算而非重读原始文件，则不进入缓存键。

**修复**：
- 跨模态边从缓存保存中剥离；每次加载时用 `create_simple_cross_modal_edges()` 从缓存节点特征重建。
- `k_cross_modal` 从缓存键哈希中删除（修改 k 后无需清缓存）。
- 新增 `load_subject_graph_from_cache(path)` 工具函数，返回 `{'eeg_timeseries': ndarray, 'fmri_timeseries': ndarray, ...}`，下游无需了解 PyG API。



**项目永远只保留以下四个 MD 文件：**

| 文件 | 用途 |
|------|------|
| `AGENTS.md` | **本文件**：AI Agent 指令 + 错误记录 |
| `SPEC.md` | 项目规范说明：目标、设计理念、架构 |
| `USERGUIDE.md` | 项目使用说明：面向非专业人士 |
| `CHANGELOG.md` | 更新日志：所有版本变更 + 待优化事项 |

**严禁**：在 `docs/`、根目录或任何位置创建第五个 MD 文件。

---

## 六、EEG→fMRI 设计理念与逻辑链（必读，每次编码前确认）

> **核心原则**：EEG 电极（较少节点）向 fMRI ROI（较多节点）投射信号。N_eeg < N_fmri 是整个跨模态设计的前提。

### 数据形状全链路

| 阶段 | EEG | fMRI |
|------|-----|------|
| 原始数据 | `[N_ch, N_times]` (e.g. 63×75000) | `[X, Y, Z, T]` (e.g. 64×64×40×190) |
| 图节点特征 | `[N_eeg, T_eeg, 1]` (e.g. 63×300×1) | `[N_fmri, T_fmri, 1]` (e.g. ~190×190×1，Schaefer200，实际数因 EPI 覆盖而异) |
| 编码器输入投影 | `[N_eeg, T_eeg, H]` | `[N_fmri, T_fmri, H]` |
| ST-GCN 跨模态消息 | 源 → propagate(size=(N_eeg, N_fmri)) → `[N_fmri, T_eeg, H]` → interpolate → `[N_fmri, T_fmri, H]` | 目标 |
| 解码器输出 | `[N_eeg, T_eeg, 1]` | `[N_fmri, T_fmri, 1]` |
| 损失目标 | `data['eeg'].x` | `data['fmri'].x` |

### 关键不变量（违反即 bug）

1. `N_eeg < N_fmri`
2. 跨模态边类型必须是 `('eeg', 'projects_to', 'fmri')`，**不依赖 config 列表顺序**
3. `propagate(size=(N_eeg, N_fmri))` 必须显式传入 size
4. `recon.shape[0] == target.shape[0]`，违反时 raise RuntimeError 而非广播

### 典型症状
- `Using a target size ([63, 190, 1]) different from input ([1, 190, 1])` → 不变量 3 或 4 被违反
- `Trying to backward through the graph a second time` → data_list 中对象被原地修改

---

## 七、训练数据设计原则

### max_seq_len=300 是错误的训练单元

`max_seq_len=300` @ 250Hz = 1.2 秒。从 1.2 秒 EEG 估计 Pearson 相关性统计上不可靠（需至少 10-30 秒）。图的 edge_index 建立在统计噪声之上。

### 正确范式：动态功能连接（dFC）滑动窗口

| 概念 | 图的哪个部分 | 如何计算 |
|------|-------------|---------|
| 结构连通性 | `edge_index` | 完整 run 的相关矩阵 |
| 动态脑状态 | 节点特征 `x` | 时间窗口切片 |

**关键约束**：
1. `windowed_sampling.enabled: true` 时，**必须设 `max_seq_len: null`**。
2. 缓存存储**完整 run 图**，窗口在运行时从缓存切片。
3. `edge_index` 在同一 run 的所有窗口间**共享同一对象**（不复制）。
4. 窗口模式下以 **run** 为单位做训练/验证集划分（防止重叠窗口数据泄漏）。

---

## 八、损失函数体系（防止死代码）

```
train_step()
    eeg_handler() → eeg_info['regularization_loss']  ← 必须加入 total_loss
    model.forward(return_encoded=True) → reconstructed, _, encoded
    model.compute_loss(data, reconstructed, encoded=encoded)
        ├── recon_{node_type}: 重建损失
        └── pred_{node_type}: 潜空间预测损失
    loss_balancer(losses)       ← 平衡 recon_* 和 pred_*
    total_loss += eeg_reg       ← 固定权重，不参与平衡
    total_loss.backward()
```

| 损失名 | 空间 | 目的 | 权重 |
|--------|------|------|------|
| `recon_eeg` | 原始 C=1 | EEG 信号重建 | 自适应 |
| `recon_fmri` | 原始 C=1 | fMRI 信号重建 | 自适应 |
| `pred_eeg` | 潜空间 H | EEG 潜向量未来预测 | 自适应 |
| `pred_fmri` | 潜空间 H | fMRI 潜向量未来预测 | 自适应 |
| `eeg_reg` | 原始 C=1 | 防止 EEG 通道崩塌 | 固定 0.01×3 |

> `h_fmri` 已含 EEG 跨模态消息，因此 `pred_fmri` 已隐式包含跨模态预测，无需专用跨模态预测头（future work）。

---

### [2026-02-27] 预测功能系统性审查：三个叠加盲区

**背景**：要求以"哪都可能有问题"的批判态度对预测功能进行完整审查。

---

**盲区 1 — HierarchicalPredictor 整数除法产生空张量（P0 bug）**

**思维误区**：以为 `future_steps // scale_factor` 总是正整数。

**根因**：当 `prediction_steps=1`（NPI 风格）且 `scale_factor=4` 时，`1 // 4 = 0`。
`future_init = x_down[:, -1:, :].repeat(1, 0, 1)` 产生形状 `[N, 0, H]` 的空张量，
Transformer 对空序列的输出仍是空张量，`pred_full[:, -0:, :]` = 完整序列（Python 负索引！），
导致 fusion 的输入维度从 `input_dim * 3` 变为 `input_dim * 3 + 全序列 * 2`，报 shape mismatch。

**触发条件**：`prediction_steps` < `max(scale_factors)` = 4（默认 num_scales=3）。
即 `prediction_steps ∈ {1, 2, 3}` 都会在前两个尺度上产生空预测。

**修复**：`future_steps_scaled = max(1, future_steps // scale_factor)`。
每个尺度至少预测 1 步，粗尺度用下采样后的 context 预测 1 步后再上采样到 future_steps，
物理语义为"低分辨率趋势预测"，融合后给细尺度提供全局先验。

**规则**：**所有整数除法，若分子可能小于分母，必须用 max(1, ...) 或 ceiling 除法。**

---

**盲区 2 — validate() pred_R² 使用非因果 context（P1 科学问题）**

**思维误区**：以为 `h[:, :T_ctx, :]` 是"上下文的表示"，没有追问编码器是否是因果的。

**根因**：编码器使用：
- `Conv1d(kernel=3, padding=1)`：对称填充，`h[t]` 包含 `signal[t-1, t, t+1]` 的信息
- `TemporalAttention(is_causal=False)`：全局双向注意力，每个时间步可看到所有其他时间步

因此 `h[:, T_ctx-1, :]`（context 最后一步）包含来自 `T_ctx, T_ctx+1` 的少量未来信息。
旧 validate() 用这个"污染"的 context 做预测，pred_r2 会偏乐观。

**影响评估**：对 `kernel_size=3`, `T_ctx=200` 的情况，泄漏量约为 0.5%（仅边界步受影响）。
训练时接受此近似（避免 2× forward pass cost）。**但评估指标必须严格，不能接受偏差。**

**修复**：validate() 中重新编码仅含 T_ctx 步的数据：
```python
context_data = data.clone()
context_data[nt].x = data[nt].x[:, :T_ctx, :]  # 切片原始信号
_, _, h_ctx = self.model(context_data, return_encoded=True)  # 编码器只见 T_ctx 步
pred_latents = predictor.predict_next(h_ctx[nt])  # 真正因果预测
```
代价：验证时多一次无反向传播的 forward pass（可接受）。

**规则**：**科学评估指标不能有任何信息泄漏。训练效率 vs 评估严格性的权衡，必须让评估端严格。**

---

**盲区 3 — 声明"因果"但未追问编码器是否真的因果**

**思维误区**：看到注释"causal prediction using predict_next()"就以为整个流程是因果的。

**根因**：`predict_next()` 本身是因果的（只用 context 的最后 context_length 步），
但如果 context 是由非因果编码器产生的（包含未来信息），因果预测器也无济于事。
"因果"需要端到端保证：**编码器因果 + 预测器因果 + 评估因果**，缺一不可。

**现状**：
- 训练：编码器非因果（工程权衡，边界泄漏极小）→ 预测器因果 → 近似因果（可接受）
- 评估：**编码器强制因果**（重新编码 T_ctx 步）→ 预测器因果 → 完全因果（严格）

**规则**：**对任何标榜"因果"的预测功能，逐一检查信息流的每一步：原始信号 → 编码 → 预测 → 解码 → 评估，看哪一步有未来信息的访问权限。**

---

**附：StratifiedWindowSampler.forward() 是训练中的死代码**

`EnhancedMultiStepPredictor.forward()`（使用 StratifiedWindowSampler 采样多个窗口）
在当前训练循环中**从未被调用**。训练用 `predict_next()`，验证也用 `predict_next()`。

`forward()` 是 API 的一部分（向后兼容），保留不删除，但新用户不应依赖它进行训练。

---

### [2026-02-27] 预测功能第二轮审查：归一化层选型盲区 + 配置误导

**背景**：对预测系统进行独立第二轮审查，假设所有地方都可能有问题。

---

**盲区 1 — BatchNorm1d 在 prediction_steps_scaled=1 时静默归零粗尺度（P0 静默失效）**

**思维误区**：以为"前一轮修复了整数除法 → 预测步数不再为零 → 一切正常"，
没有追问"steps=1 时，归一化层是否还有问题"。

**根因**：上一轮修复将 `future_steps_scaled = max(1, future_steps // scale_factor)` 后，
预测步数不再为零。但上采样器中的 `BatchNorm1d(input_dim)` 接收形状 `[batch=1, input_dim=128, time=1]`：
- 每通道仅有 N×L = 1×1 = **1 个样本**
- var = 0 → normalized = (x − x) / sqrt(eps) = **0**
- output = gamma × 0 + beta = **0**（BatchNorm 初始化 gamma=1, beta=0）

粗尺度和中等尺度的上采样输出全为零，Fusion 输入 `cat([zeros, zeros, valid_fine])`，
层级预测**静默退化为单尺度**——在 NPI 风格（prediction_steps=1）这一最重要的科学配置下，
多尺度层级架构完全失效，但不会有任何报错或警告。

**修复**：`nn.BatchNorm1d(input_dim)` → `nn.GroupNorm(1, input_dim)`

GroupNorm(num_groups=1, num_channels=C) 将所有 C 个通道作为一个组归一化，
每个样本使用 C×L×N 个值计算统计量。对 `[1, 128, 1]`：128 个值 → 统计量有意义。

归一化层比较（对 [batch=1, channels=128, time=1]）：
| 层 | 每次统计的元素数 | 适用性 |
|----|----|------|
| BatchNorm1d(C) | N×L = 1×1 = 1 | 🔴 退化 |
| InstanceNorm1d(C) | L = 1 | 🔴 退化 |
| GroupNorm(C, C) = InstanceNorm | L = 1 | 🔴 退化 |
| GroupNorm(1, C) | C×L×N = 128 | ✅ 正常 |

**规则**：**在 batch_size=1 的图神经网络场景中，凡 L（时间/空间维度）可能为 1 的地方，
必须选用不依赖 batch 维度和时间维度的归一化层（GroupNorm(1, C) 或 LayerNorm）。**

---

**盲区 2 — 配置注释声称"启用 NLL 损失"但实际从未执行**

**思维误区**：以为"配置里写 use_uncertainty: true + NLL 损失"就代表 NLL 损失在工作。

**根因**：
1. `predict_next()` 硬编码 `return_uncertainty=False`
2. `compute_loss()` → `predict_next()` → uncertainty_head 从不被调用
3. `uncertainty_head` 参数存在于模型中但永远不会收到梯度
4. `default.yaml` 注释"启用 NLL 损失"是**错误的描述**

**后果**：
- 用户相信已启用不确定性估计，实际上什么都没发生
- 浪费 ~input_dim×1.5 个永不更新的参数
- 如果用户依赖不确定性估计做临床决策，将产生虚假安全感

**修复**：
- 更正 `default.yaml` 注释，明确说明"当前版本为后处理工具，不参与训练损失"
- 将默认值从 `true` 改为 `false`（诚实接口原则：默认配置只启用真正工作的功能）

**规则**：**凡是"声明了但从未真正工作"的配置选项，默认值必须为 false，且注释必须如实描述实际状态。**
（参见 AGENTS.md §三 [2026-02-21] "声明了但从未真正工作"的功能）

---

### [2026-02-27] 预测功能第三轮审查：YAML 与代码默认值不同步 + GRU 自回归维度崩溃

**背景**：第三轮以全新视角独立审查，验证前两轮已修复项的正确性，并寻找新盲区。

---

**盲区 1 — use_uncertainty Python 默认值未与 YAML 同步（P1 逻辑不一致）**

**思维误区**：以为"上一轮改了 YAML 就完成了"，没有检查 Python 代码层的 fallback 默认值。

**根因**：Round 2 已将 `default.yaml` 中 `use_uncertainty` 改为 `false`，并在 AGENTS.md 中记录了意图。但 `graph_native_system.py:446` 中的 Python fallback 默认值仍为 `True`：

```python
use_uncertainty=pred_cfg.get('use_uncertainty', True),  # ← 仍为 True！
```

任何不通过 YAML 创建模型的代码（单元测试、API用户、`predictor_config` 缺少该键时）会得到 `True`，创建从未训练的 `uncertainty_head` 参数，浪费显存且误导用户。

**修复**：`True` → `False`（1行改动）。

**规则**：**配置项的 Python fallback 默认值必须与 YAML 默认值保持一致。每次修改 YAML 默认值时，必须同时检索代码中所有对应的 `.get(key, python_default)` 调用并同步修改。**

---

**盲区 2 — GRU 自回归滚动在第2步后崩溃（P0 运行时崩溃）**

**思维误区**：以为"GRU 是简单替代品，没有复杂的维度问题"，没有追问反馈维度是否和输入维度匹配。

**根因**：`EnhancedMultiStepPredictor` 创建时 `hidden_dim = input_dim × 2`（例如 128 → 256）。GRU 的 `input_size=input_dim=128`，`hidden_size=hidden_dim=256`。自回归滚动代码（3处）：

```python
current = context[:, -1:, :]          # [batch, 1, 128] ✓
output, hidden = predictor(current)    # output: [batch, 1, 256]
current = output                       # BUG: 256 ≠ 128 → 第2步崩溃
```

第2步开始 GRU 收到 256 维输入但只接受 128 维 → `RuntimeError`。

**影响的3处位置**：
1. `HierarchicalPredictor._autoregressive_predict()` — 触发条件: `use_transformer=False`
2. `EnhancedMultiStepPredictor.predict_next()` GRU分支 — `use_hierarchical=False, use_transformer=False`
3. `EnhancedMultiStepPredictor.forward()` GRU分支 — 同上

**修复**：为 GRU 添加输出投影层 `nn.Linear(hidden_dim, input_dim)`，在反馈前将 GRU 输出投影回 input_dim。这是标准 seq2seq 解码器设计模式（隐藏空间 ≠ 输入空间，学习投影桥接两者）。

- `HierarchicalPredictor.__init__`: 添加 `self.gru_output_projs = nn.ModuleList([nn.Linear(hidden_dim, input_dim) × num_scales])`（仅 predictor_type='gru' 时）
- `_autoregressive_predict()`: 接受 `output_proj` 参数，使用 `current = proj(output) if proj else output`
- `HierarchicalPredictor.forward()`: 传入 `self.gru_output_projs[i]`
- `EnhancedMultiStepPredictor.__init__`: 添加 `self.gru_output_proj = nn.Linear(hidden_dim, input_dim)`（仅简单GRU模式）
- `predict_next()` / `forward()` GRU分支: 使用 `self.gru_output_proj`

**规则**：**任何 GRU/RNN 自回归滚动，第一步的输入维度和输出维度可能不同。反馈前必须确认 output.shape[-1] == input_size。若不同，必须添加投影层。**

---

### [2026-02-27] 预测功能第四轮审查：GRU 累积输出维度未投影（Round 3 遗漏的对称问题）

**背景**：第四轮独立审查，假设所有已修复项都可能存在遗漏。

---

**盲区 — GRU 累积预测维度错误（P0 运行时崩溃）**

**思维误区**：以为"Round 3 修复了 GRU 输出投影"就等于"GRU 自回归循环完全正确"。没有追问：**修复了反馈路径，累积路径是否也修了？**

**根因**：Round 3 的修复代码：

```python
output, hidden = predictor(current, hidden)  # output: [B, 1, 256]
predictions.append(output)                   # ← 仍然累积 256 维！（遗漏）
current = output_proj(output)                # ← 反馈正确投影到 128 维 ✓
```

`predictions` 累积的是 `hidden_dim=256` 维的原始 GRU 输出，而下游所有消费者都期待 `input_dim=128`：
- `upsamplers[i]` = `ConvTranspose1d(in_channels=128)` → 收到 256 → **CRASH**
- `fusion` = `Linear(input_dim*3=384)` → 收到 768 → **CRASH**
- `prediction_propagator` = `SpatialTemporalGraphConv(in_channels=128)` → 收到 256 → **CRASH**

**影响的3处位置**：
1. `HierarchicalPredictor._autoregressive_predict()` — GRU 模式（`use_transformer=False`）
2. `EnhancedMultiStepPredictor.predict_next()` GRU分支
3. `EnhancedMultiStepPredictor.forward()` GRU分支

**修复**：在所有3处，将"累积"与"反馈"统一为同一个投影后的变量：

```python
# 修复后（每处完全对称）：
output, hidden = predictor(current, hidden)
projected = output_proj(output) if output_proj is not None else output
predictions.append(projected)   # ← 累积 128 维 ✓
current = projected              # ← 反馈 128 维 ✓
```

**根本规则**：**自回归循环中"用于累积（append）"的张量和"用于反馈（current=）"的张量必须是同一个对象。** 如果两者不同，必须追问：两者的维度是否一致，下游消费者期待哪个维度？

---

### [2026-02-27] 预测功能第五轮审查：avg_pool1d 崩溃前守卫 + pred_enc 初始化 + N守卫 + 训练状态污染

**背景**：第五轮独立审查，重新审视每个函数，假设所有已修复项都可能仍有遗漏。

---

**盲区 1 — avg_pool1d 在返回前内部崩溃（P2 潜在崩溃）**

**思维误区**：以为"检查 x_down.shape[1] == 0 就能保护空张量"——没有追问：**PyTorch 是在返回前还是返回后抛出异常？**

**根因**：`F.avg_pool1d` 在计算出 `output_len ≤ 0` 时**直接在函数内部**抛出 "Output size is too small"，*在*返回之前。所以"在 avg_pool1d 调用之后检查结果"永远不会执行。触发条件：`_PRED_MIN_SEQ_LEN=4`、`_PRED_CONTEXT_RATIO=2/3` → T_ctx=2 < scale_factor=4。

```python
# 错误修复（Round 4 分析的方案）：
x_down = F.avg_pool1d(...)   # ← 已经崩溃，走不到下面
if x_down.shape[1] == 0:     # ← 永远不会执行
    ...

# 正确修复（Round 5）：
if x.shape[1] < scale_factor:   # ← 在调用前检查
    logger.debug(f"... zero prediction for this scale ...")
    pred_up = torch.zeros(...)
    scale_predictions.append(pred_up)
    continue
x_down = F.avg_pool1d(...)      # ← 安全
```

**规则**：**对所有可能抛出内部异常的函数（不仅仅是返回错误值），必须在调用前就验证前置条件，不能依赖检查返回值。**

---

**盲区 2 — validate() pred_enc 用原始信号初始化导致 decoder 维度崩溃（P2 潜在崩溃）**

**根因**：
```python
pred_enc = data.clone()  # x shape = [N, T, C=1] — 原始信号！
for nt, pred_lat in pred_latents.items():
    pred_enc[nt].x = pred_lat  # 只覆盖了 pred_latents 中的 modality
# 未被覆盖的 modality: pred_enc[nt].x = [N, T, 1]
# decoder: Conv1d(in_channels=128) 收到 1 通道 → CRASH
```
触发条件：某个 modality 因 T < `_PRED_MIN_SEQ_LEN` 被跳过。

**修复**：先从 `h_ctx_dict` 初始化（正确 H=128 维），再用 pred_latents 覆盖。
未被预测的 modality 的 context latent 喂给 decoder 不影响 R² 评估（`pred_T_ctx.get(nt) is None` 在 Step 6 跳过它们）。

---

**盲区 3 — compute_loss() pred_loss 缺少 N 不匹配守卫（P3 不一致）**

`recon_loss` 有 `if recon.shape[0] != target.shape[0]: raise RuntimeError(...)` 守卫；`pred_loss` 没有。**修复**：添加 `if pred_mean.shape[0] != future_target.shape[0]: logger.warning(...); continue`，与 recon_loss 保持一致的防御性编程。

---

**盲区 4 — UncertaintyAwarePredictor 强制 .train() 污染 eval 状态（P2）**

MC dropout 路径中 `self.base_predictor.train()` 被无条件调用，在 validate() 的 `@torch.no_grad()` 上下文中也会执行，污染 BatchNorm running stats。**修复**：`was_training = self.base_predictor.training`，try/finally 恢复。

---

### [2026-02-27] 预测功能第六轮审查：非层级 TransformerPredictor 错误分派 + UAP shape 错误 + h_ctx_dict 初始化

**背景**：第六轮独立审查，系统性验证所有配置组合。

---

**盲区 1 — 非层级 TransformerPredictor 分派到 GRU 分支（P2 运行时崩溃/静默错误）**

**思维误区**：以为"修复了 GRU 投影就修好了所有 else 分支"，没有追问：**else 分支覆盖了哪些配置？**

**根因**：`EnhancedMultiStepPredictor.__init__()` 存储了 `self.use_hierarchical` 和 `self.use_uncertainty`，但**未存储 `self.use_transformer`**。当 `use_hierarchical=False, use_transformer=True, use_uncertainty=False` 时：
- `self.predictor = TransformerPredictor(...)` (返回单一 Tensor)
- `predict_next()` 和 `forward()` 的 `else` 分支执行 `_, hidden = self.predictor(context)`
- TransformerPredictor 返回 `[B, T, H]`（不是 `(output, hidden)`）
- batch_size ≠ 2 时崩溃；batch_size = 2 时静默产生错误结果

**修复**：
1. `__init__` 中保存 `self.use_transformer = use_transformer`
2. 在 `predict_next()` 和 `forward()` 中添加 `elif self.use_transformer:` 分支
3. 提取模块级辅助函数 `_transformer_seq2seq_predict(predictor, context, n_steps)` — 消除3处重复的 seq2seq 扩展逻辑（extend context + causal mask + take last n_steps）

---

**盲区 2 — UncertaintyAwarePredictor 包装 TransformerPredictor 时返回错误形状（P2 静默错误）**

**思维误区**：以为"修复了 EnhancedMultiStepPredictor 的分派就覆盖了 UAP 的分派"，没有追问：**UAP.forward() 的 isinstance 分支是否正确处理 TransformerPredictor？**

**根因**：`UncertaintyAwarePredictor.forward()` 中 gaussian/dropout 路径只检查 `isinstance(self.base_predictor, HierarchicalPredictor)`；对非 Hierarchical 的情况调用 `self.base_predictor(x)`（TransformerPredictor 返回 `[B, ctx_len, H]`，不是 `[B, future_steps, H]`）。

**修复**：提取 `UncertaintyAwarePredictor._base_predict(x, future_steps)` 辅助方法，使用 `_transformer_seq2seq_predict` 正确处理 TransformerPredictor；gaussian/dropout 所有分支统一调用 `_base_predict`，消除重复的 isinstance 分散逻辑。

---

**盲区 3 — validate() h_ctx_dict 未初始化（P3 鲁棒性）**

`h_ctx_dict` 仅在 `if ctx_T_map:` 块内赋值，在 `if pred_latents:` 块内使用。当前逻辑保证安全（pred_latents 非空 → ctx_T_map 非空 → h_ctx_dict 已赋值），但这是隐式不变式。

**修复**：在 `if ctx_T_map:` 之前添加 `h_ctx_dict: Dict[str, torch.Tensor] = {}` 显式初始化，使不变式变得显式，防止未来代码修改破坏此假设。

**配置空间全覆盖审查结果（Round 6）**：
| 配置 | 状态 |
|------|------|
| use_hierarchical=True（默认） | ✅ 正确 |
| use_hierarchical=False, use_transformer=False (GRU) | ✅ 正确 |
| use_hierarchical=False, use_transformer=True | **已修复 V5.36** |
| use_uncertainty=True 包装任意 base | **已修复 V5.36**（UAP._base_predict） |

---

### [2026-02-28] 预测功能第七轮审查：GRU+不确定性彻底破损 + validate() decoder 潜在崩溃

**背景**：第七轮独立审查，假设所有已修复项都可能存在遗漏，穷举所有配置组合的完整调用链。

---

**盲区 1 — GRU + UncertaintyAwarePredictor 组合三重破损（P2 运行时崩溃）**

**配置**：`use_hierarchical=False, use_transformer=False, use_uncertainty=True`

**思维误区**：以为"Round 6 修复了 UAP._base_predict 就覆盖了所有 else 分支"，没有追问：**当 base_predictor = nn.GRU 时，`_base_predict` else 分支是否还会崩溃？**

**三重根因**：
1. `_base_predict` else 分支调用 `return self.base_predictor(x)` = `nn.GRU(x)` → 返回 `(output, h_n)` **tuple**，不是 Tensor → TypeError
2. 即使解包 `output`：`nn.GRU(x)` 返回 `[N, ctx_len, hidden_dim]`，不是 `[N, future_steps, input_dim]`（时间维度长度错误，特征维度 256≠128）
3. `uncertainty_head` 期待 `input_dim=128`，收到 `hidden_dim=256` → 形状崩溃

**修复**：
1. `EnhancedMultiStepPredictor.__init__` 对不支持的组合提前抛出 `ValueError`（构造时即报告，而非运行时神秘崩溃）
2. `UAP._base_predict` else 分支防御性解包：`result = self.base_predictor(x); return result[0] if isinstance(result, tuple) else result`（保护直接实例化 UAP 的场景，如单元测试）

---

**盲区 2 — validate() decoder 在 T < _PRED_MIN_SEQ_LEN 时崩溃（P3 潜在崩溃）**

**根因**：`pred_enc = data.clone()` 会保留 T < 4 的节点的原始信号 `[N, T, C=1]`。`h_ctx_dict` 不包含这些节点（因 T < _PRED_MIN_SEQ_LEN）。decoder 的 `Conv1d(in_channels=128)` 应用于 `[N, 1, T]` → 通道不匹配崩溃。

**修复**：在调用 decoder 前删除 pred_enc 中不在 h_ctx_dict 中的节点类型：
```python
for nt in list(pred_enc.node_types):
    if nt not in h_ctx_dict:
        logger.debug(f"validate: removing '{nt}' from pred_enc (T < _PRED_MIN_SEQ_LEN)")
        del pred_enc[nt]
```

**配置空间完整覆盖审查结果（Round 7）**：
| 配置 | 状态 |
|------|------|
| use_hierarchical=True（默认） | ✅ |
| use_hierarchical=True, use_transformer=False | ✅ |
| use_hierarchical=False, use_transformer=True | ✅ |
| use_hierarchical=False, use_transformer=False | ✅ |
| + use_uncertainty=True（上述前三种） | ✅ |
| **use_hierarchical=False, use_transformer=False, use_uncertainty=True** | **✅ 已明确禁止（ValueError）V5.37** |

**新规则**：**每次新增 isinstance 分支时，必须追问：`else` 分支现在还有哪些类型会落入？每种落入类型的返回值格式是否兼容后续代码？** 特别是 `nn.GRU` 等 PyTorch 原生模块的返回格式（tuple），永远不应假设与自定义 nn.Module 相同。

---

## 十、数字孪生脑架构状态（持续更新）

| 维度 | 状态 | 实现版本 |
|------|------|---------|
| 个性化（被试特异性嵌入） | ✅ 已实现 | V5.19–V5.20 |
| 动态拓扑（DynamicGraphConstructor） | ✅ 已实现 | V5.14 |
| 系统级预测传播（GraphPredictionPropagator） | ✅ 已实现 | V5.25 |
| 梯度累积（gradient_accumulation_steps） | ✅ 已实现 | V5.28 |
| 随机权重平均（SWA） | ✅ 已实现（可选） | V5.28 |
| 验证集 R² 指标 | ✅ 已实现 | V5.28 |
| EEG 频谱相干性连接（configurable） | ✅ 已实现 | V5.28 |
| 相关性跨模态边（NVC 感知，top-k |r|） | ✅ 已实现 | V5.30 |
| 训练曲线可视化（loss + R² PNGs） | ✅ 已实现 | V5.30 |
| 断点续训（--resume） | ✅ 已实现 | V5.30 |
| 时序数据增强（noise + scale，可选） | ✅ 已实现（默认关闭） | V5.30 |
| NPI 风格 context_length 可配置 | ✅ 已实现 | V5.31 |
| HierarchicalPredictor NPI 兼容（prediction_steps=1 整数除法） | ✅ 已修复 | V5.31 |
| validate() 真因果 pred_R²（因果编码） | ✅ 已修复 | V5.31 |
| HierarchicalPredictor 上采样器 BatchNorm1d→GroupNorm（静默归零修复） | ✅ 已修复 | V5.32 |
| use_uncertainty 配置文档准确化（默认关闭） | ✅ 已修复 | V5.32 |
| use_uncertainty Python 默认值同步 YAML（True→False） | ✅ 已修复 | V5.33 |
| GRU 自回归滚动输出投影——反馈维度修复 | ✅ 已修复 | V5.33 |
| GRU 自回归滚动输出投影——累积维度修复（对称完整） | ✅ 已修复 | V5.34 |
| avg_pool1d 调用前守卫（短 context 崩溃修复） | ✅ 已修复 | V5.35 |
| validate() pred_enc 从 h_ctx_dict 初始化（decoder 维度修复） | ✅ 已修复 | V5.35 |
| pred_loss N 不匹配守卫（防御性一致性） | ✅ 已修复 | V5.35 |
| UncertaintyAwarePredictor MC dropout eval 状态恢复 | ✅ 已修复 | V5.35 |
| 非层级 TransformerPredictor 错误分派修复（predict_next/forward） | ✅ 已修复 | V5.36 |
| UAP._base_predict 统一分派（TransformerPredictor shape 修复） | ✅ 已修复 | V5.36 |
| _transformer_seq2seq_predict 模块级辅助函数（DRY 消重） | ✅ 已实现 | V5.36 |
| validate() h_ctx_dict 显式初始化（鲁棒性） | ✅ 已修复 | V5.36 |
| GRU+uncertainty 不支持组合 ValueError（构造时明确拒绝） | ✅ 已修复 | V5.37 |
| UAP._base_predict GRU tuple 防御性解包 | ✅ 已修复 | V5.37 |
| validate() decoder 在 T < _PRED_MIN_SEQ_LEN 时删除节点守卫 | ✅ 已修复 | V5.37 |
| 跨会话预测 | ⚡ 部分（within-run） | — |
| 干预响应、自我演化 | ❌ Future work | — |

### 被试特异性嵌入全链路（V5.19–V5.20）

`subject_to_idx` → `built_graph.subject_idx`（含缓存路径）→ `extract_windowed_samples` 传播 → `nn.Embedding(N, H)` → `x_proj += embed.view(1,1,-1)`

**⚠️ 注意**：缓存命中路径须在 `extract_windowed_samples` 前先写入 `subject_idx`（V5.20 修复）。

### DynamicGraphConstructor 使用建议
- 小数据集（< 50 样本）谨慎：额外参数可能过拟合。
- `k_dynamic_neighbors`：fMRI(N≈190) 设 10；EEG(N≤64) 设 5。
