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

## 四、文档格式规范（必须遵守）

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

## 九、数字孪生脑架构状态（持续更新）

| 维度 | 状态 | 实现版本 |
|------|------|---------|
| 个性化（被试特异性嵌入） | ✅ 已实现 | V5.19–V5.20 |
| 动态拓扑（DynamicGraphConstructor） | ✅ 已实现 | V5.14 |
| 跨会话预测 | ⚡ 部分（within-run） | — |
| 干预响应、自我演化 | ❌ Future work | — |

### 被试特异性嵌入全链路（V5.19–V5.20）

`subject_to_idx` → `built_graph.subject_idx`（含缓存路径）→ `extract_windowed_samples` 传播 → `nn.Embedding(N, H)` → `x_proj += embed.view(1,1,-1)`

**⚠️ 注意**：缓存命中路径须在 `extract_windowed_samples` 前先写入 `subject_idx`（V5.20 修复）。

### DynamicGraphConstructor 使用建议
- 小数据集（< 50 样本）谨慎：额外参数可能过拟合。
- `k_dynamic_neighbors`：fMRI(N≈190) 设 10；EEG(N≤64) 设 5。
