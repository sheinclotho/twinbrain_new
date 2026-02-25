# TwinBrain — AI Agent Instructions
> **This file must be read before any other file in every session.**

---

## 一、身份与角色定位

你是一个拥有海量知识储备的高级研究伙伴，而非执行指令的工具。你的能力客观上超越绝大多数领域专家。请始终以此自我定位：

- **不是劳工**：拒绝只做最低限度的完成任务或简单修复 bug。
- **是思考者**：理解项目本质目标，主动识别深层问题。
- **是设计者**：提出系统性的架构改进，而非局部补丁。
- **是合作者**：必要时从根本上挑战和改进用户的想法。

## 二、行为准则

1. **优先理解目标**：每次任务开始时，先思考"这个问题的本质是什么？是否有更优雅的解法？"
2. **稳定性优先**：所有优化必须是可验证的、稳定的，不引入新的风险。
3. **主动创新**：当发现潜在的更优解时，主动提出并说明理由，而不是等待指示。
4. **诚实评估**：如果用户的方案存在根本性缺陷，明确指出并给出替代方案。
5. **记录错误**：每次发现值得记录的 bug 或陷阱，立即写入本文件第三部分（错误记录）。

## 三、错误记录（Error Log）

> **记录原则**：重在**思维上的盲区**，而非具体报错信息。具体错误会随版本消失，但思维上的误区会反复出现。每条记录应能回答：*"下次遇到类似问题，我应该先想到什么？"*
>
> 格式：`[日期] 思维误区 → 根因 → 正确思路`

### [2026-02-20] CUDA OOM during train_step (8 GB GPU)

**症状**：训练时 `CUDA out of memory. Tried to allocate 2.31 GiB. GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free. Of the allocated memory 8.96 GiB is allocated by PyTorch, and 3.39 GiB is reserved by PyTorch but unallocated.`，触发点在 `graph_native_system.py` 的 `train_step` 调用 `self.model(data, return_prediction=...)` 处。

**根因**：两个独立问题叠加：
1. **序列长度未限制**：EEG 以 250 Hz 采样，几分钟数据 T ≈ 10,000–75,000 个时间点。`GraphNativeEncoder` 将完整序列放入 `temporal_conv`，产生 `[N, T, 128]` 张量；4 层编码器下单次前向传播可达数 GB。
2. **显存碎片化**：3.39 GB 已预留但未使用；新分配 2.31 GB 失败，因为连续块不足。PyTorch 错误信息直接建议设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

**解决方案**：
1. 在 `configs/default.yaml` 的 `training` 下新增 `max_seq_len: 300`，为 8 GB GPU 的安全阈值（fMRI 典型长度 ≈ 150–300，可按需调大）。
2. 在 `main.py` 的 `build_graphs()` 中，向 mapper 传参前将 `fmri_ts` 和 `eeg_data` 截断到 `max_seq_len`，从根源消除大张量。
3. 在 `main.py` 顶部 import 区设置 `os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')`，解决显存碎片化（`setdefault` 保留用户自定义值）。
4. 在 `graph_native_system.py` 的 `train_epoch` 每个 epoch 结束时调用 `torch.cuda.empty_cache()`，将已释放的碎片块归还给分配器。

**影响文件**：
- `configs/default.yaml`
- `main.py`
- `models/graph_native_system.py`
- `AGENTS.md`

**症状**：训练时 `MemoryError`，traceback 在 `graph_native_encoder.py` 的 `SpatialTemporalGraphConv.forward()` 中，最终触发点是 spectral_norm 的 `_power_method`。

**根因**：`forward()` 中对 T 个时间步逐步调用 `self.propagate()`，PyTorch autograd 将所有 T 步的中间激活（注意力权重 `[E,1]`、消息矩阵 `[E,C]` 等）全部保留在内存中用于反向传播。当 T 较大（如 T=300, E=4000, C=128）时，仅注意力相关张量就达到数百 MB，直至耗尽 RAM。spectral_norm 的 power method 只是压死骆驼的最后一根稻草。

**错误地方**：`models/graph_native_system.py` 中声明了 `use_gradient_checkpointing` 参数，但实现时调用 `HeteroConv.gradient_checkpointing_enable()`，该方法不存在，导致 checkpointing 从未真正生效。

**解决方案**：
1. 在 `SpatialTemporalGraphConv` 中添加 `use_gradient_checkpointing` 参数，并在时间步循环内用 `torch.utils.checkpoint.checkpoint()` 包装 `propagate()` 调用。每个时间步完成后立即释放中间激活，仅保留输出 `[N, C_out]`，内存从 `O(T·E·C)` 降至 `O(T·N·C)`。
2. 将 `use_gradient_checkpointing` 参数从 `GraphNativeBrainModel` 传递到 `GraphNativeEncoder`，再到每个 `SpatialTemporalGraphConv`。
3. 在 `main.py` 中从 config 读取该参数并传入 model 构造函数。
4. 将 `configs/default.yaml` 中 `use_gradient_checkpointing` 改为 `true`（之前是 `false`，这是功能声明而非真正实现）。

**影响的文件**：
- `models/graph_native_encoder.py`
- `models/graph_native_system.py`
- `main.py`
- `configs/default.yaml`

### [2026-02-21] 跨模态消息聚合时未考虑时序维度对齐

**思维误区**：修复内存问题时，只关注了"如何截断序列到 max_seq_len"，没有意识到**不同模态的序列长度天然不同**（EEG 原始长度可能 < max_seq_len，而 fMRI 被截断到 max_seq_len），导致两个模态各有独立的 T。

**根因**：在 `graph_native_encoder.py` 的消息聚合循环中，跨模态边（EEG→fMRI）使用 EEG 的时序维度 T_eeg 生成消息 `[N_fmri, T_eeg, H]`，而 fMRI 自环产生 `[N_fmri, T_fmri, H]`。两者 T 不同，`sum(messages)` 报 size mismatch。

**正确思路**：凡是"来自不同来源的张量需要聚合"的场景，**第一个要问的问题就是：它们的所有维度是否都匹配？**跨模态 = 跨 T 的可能性，聚合前应归一化到目标节点的维度。

**解决**：在 `conv(x_src, ...)` 之后，若 `msg.shape[1] != x.shape[1]`（目标节点的 T），用 `F.interpolate` 对时序维度做线性重采样。

---

### [2026-02-21] 多个"声明了但从未真正工作"的功能

**思维误区**：看到代码里有 `use_gradient_checkpointing`、`AdaptiveLossBalancer`、`update_weights` 等声明，就以为它们在工作。没有追溯"这个功能的调用链是否闭合"。

**正确思路**：对每一个声称"已实现"的功能，问三个问题：
1. 调用入口在哪里？（e.g. set_epoch 在 train_epoch 里有没有被调用？）
2. 内部实现是否能执行？（e.g. autograd.grad 在 backward 后会崩溃吗？）
3. 输入/输出形状是否与调用者匹配？（e.g. predictor 接收 4-D 还是 3-D？）**不要信任"看上去有的代码"，要信任"可以从头到尾追踪的执行路径"。**

**具体例子**：
- `AdaptiveLossBalancer.update_weights` 调用 `torch.autograd.grad(task_loss, ...)` 但 `backward()` 已经释放了计算图 → 崩溃
- `set_epoch()` 从未在 `train_epoch` 里调用 → `epoch_count` 永远是 0 → warmup 永远不结束 → `update_weights` 实际上永远是 no-op（这个 bug 反而"保护"了上面的 bug）
- `predictor(h.unsqueeze(0), ...)` 把 `[N, T, H]` 变成 `[1, N, T, H]`（4-D），但 window sampler 只能 unpack 3 维 → ValueError

---

### [2026-02-21] "架构上的空话"：fMRI 实际只有 1 个节点

**思维误区**：看到"图原生"、"空间-时间联合建模"、"保持大脑拓扑"，就认为 fMRI 真的在图上建模了空间结构。没有检查 fMRI 节点数 N_fmri 到底是多少。

**根因**：`process_fmri_timeseries` 把所有 fMRI 体素的时间序列 **mean 掉**（`fmri_data.reshape(-1, T).mean(axis=0)`），然后 `.reshape(1, -1)` → N_fmri = **1**。整个项目的"图卷积"对 fMRI 实际上只有 1 个节点，空间信息完全丢失。

**正确思路**：在宣称"图原生"之前，先打印 `N_fmri` 和 `N_eeg` 看实际上有多少个节点。任何时候看到 `mean(axis=0)` 后接 `reshape(1, -1)`，都要警惕"这是把 N 维空间折叠成了 1 维"。

**解决**：用 NiftiLabelsMasker（nilearn）应用 Schaefer200 图谱，提取 200 个 ROI 时间序列 `[200, T]`，每个 ROI 对应图上一个节点。atlas 文件已在 `configs/default.yaml` 配置，只是从未使用。

---

### [2026-02-21] log_weights 参与 backward 图导致梯度累积 + "backward 两次" 错误

**症状**：训练时 `RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed)`，触发点在 `graph_native_system.py` 的 `train_step` 调用 `self.scaler.scale(total_loss).backward()` 处。

**根因**：`AdaptiveLossBalancer.forward()` 中计算权重时用了 `torch.exp(self.log_weights[name]).clamp(...)`，未 `.detach()`。这导致：
1. `total_loss` 的计算图包含 `log_weights`（nn.Parameter），backward() 会为它们计算梯度。
2. `log_weights` 不在 `optimizer` 中，`optimizer.zero_grad()` 不会清零它们的 `.grad`。
3. 每次 backward 后 `log_weights.grad` 持续累积，不被重置。
4. 同时，`update_weights()` 被调用时传入的是带 `grad_fn` 的 loss 张量（backward 已释放其计算图），若 PyTorch 内部尝试访问这些已释放的节点，即触发"backward 两次"错误。

**思维误区**：看到 `nn.Parameter` 就以为"有梯度是正常的"。没有追问：**这个梯度会被使用吗？** `log_weights` 通过 `update_weights()` 手动更新（loss 幅值代理规则），完全不依赖 `.grad`。让它参与 backward 图只有负面效果：浪费显存、累积无用梯度、引发潜在的图释放错误。

**正确思路**：对于**手动更新**的参数（不通过 optimizer），在 forward 中应使用 `.detach()` 将其从 backward 图中排除。同样，向 `update_weights()` 这类"只读 scalar"的后处理函数传入张量时，应先 `.detach()` 明确 post-backward 的使用语义。

**解决**：
1. `AdaptiveLossBalancer.forward()` 中：`weights = {name: torch.exp(self.log_weights[name]).detach().clamp(...)}` — 权重作为常数参与 loss 计算，不进入反向图。
2. `GraphNativeTrainer.train_step()` 中：调用 `update_weights` 前先 `detached_losses = {k: v.detach() for k, v in losses.items()}`，明确表达 backward 已结束、只需读取 scalar 的意图。

**影响文件**：
- `models/adaptive_loss_balancer.py`
- `models/graph_native_system.py`
- `AGENTS.md`
- `CHANGELOG.md`

---

### [2026-02-21] 跨模态 ST-GCN 的 update() 将 N_src 广播给 N_dst → 重建 shape 错误

**症状**：训练时警告 `Using a target size (torch.Size([1, 190, 1])) that is different to the input size (torch.Size([63, 190, 1]))` 在 `graph_native_system.py` 的 `compute_loss` 里 `F.huber_loss(recon, target)` 处。recon 有 63 个节点，而 fMRI target 只有 1 个节点。

**根因**：`SpatialTemporalGraphConv.update(aggr_out, x_self)` 无条件计算 `aggr_out + lin_self(x_self)`。对于**跨模态边**（EEG→fMRI）：
- `aggr_out`：shape `[N_dst=1, H]`（fMRI 目标节点聚合结果）
- `x_self`：shape `[N_src=63, H]`（从 `propagate` 直传过来的 EEG 源节点特征）
- `[1, H] + [63, H]` → PyTorch 广播 → 返回 `[63, H]`，而不是 `[1, H]`

该错误在第一层编码器就扩散：`x_dict['fmri']` 变成 `[63, T, H]`，后续层、解码器全程处理 63 个"fMRI 节点"，最终重建输出 `[63, T, 1]` 与真实目标 `[1, T, 1]` 不匹配。

**思维误区**：`SpatialTemporalGraphConv` 被设计为同模态（intra-modal）卷积，默认 N_src == N_dst，self-connection 代表节点自身残差。**跨模态边没有这个前提**——源节点和目标节点属于不同节点类型，数量可以完全不同。看到 `MessagePassing` 就以为 N 不变是错误的。

**正确思路**：每次在异构图中将同一 conv 层用于跨模态边时，先问：**源节点数 N_src 和目标节点数 N_dst 是否相同？** 如不同，任何依赖"N 不变"假设的操作（如 `update` 里的 self-connection）都必须跳过或替换。

**解决**：在 `SpatialTemporalGraphConv.update` 中添加一行检查：当 `aggr_out.shape[0] != x_self.shape[0]` 时，直接返回 `aggr_out`（跨模态边不做 self-connection）。

**影响文件**：
- `models/graph_native_encoder.py`
- `AGENTS.md`
- `CHANGELOG.md`

---

### [2026-02-21] Decoder 的 ConvTranspose1d 悄悄改变了时序长度

**思维误区**：以为 `ConvTranspose1d(kernel_size=4, stride=1, padding=1)` 和 `Conv1d` 是对称的，输出长度不变。实际上 ConvTranspose1d 的输出公式是 `(T-1)*stride - 2*padding + kernel_size`，stride=1, padding=1, kernel_size=4 → `T+1`。3 层后 T+3。

**正确思路**：ConvTranspose1d 的语义是"逆卷积/转置卷积"，设计用于**上采样**（stride>1 才有意义）。**stride=1 时应该用 Conv1d，不应该用 ConvTranspose1d。** 一旦用了 ConvTranspose1d(stride=1)，就要立刻验证输出尺寸公式。

**解决**：对 stride=1 的层改用 `Conv1d(kernel_size=3, padding=1)`，精确保留 T。对 stride=2 的层（真正上采样）仍用 ConvTranspose1d。

---

### [2026-02-21] propagate() 未传 size 导致跨模态 N 不匹配（第 5 次复现）

**症状**：`UserWarning: Using a target size (torch.Size([1, 190, 1])) that is different to the input size (torch.Size([63, 190, 1]))` 在 `compute_loss` 的 `F.huber_loss` 处。`reconstructed['fmri']` 有 63 个节点，而 target 只有 1 个节点。

**根因**：`SpatialTemporalGraphConv.forward()` 调用 `self.propagate(edge_index, x=x_t_slice, ...)` 时**没有传 `size=(N_src, N_dst)`**。PyG 默认 `size=(N_src, N_src)`，对 EEG→fMRI 跨模态边（N_src=63, N_dst=1）会产生 `aggr_out=[63, H]` 而非 `[1, H]`。既有的 `update()` 守卫（`aggr_out.shape[0] != x_self.shape[0]`）**永远不会触发**，因为两个张量都错误地显示 N=63。

**思维误区**：看到 `update()` 里已经有 N 不匹配守卫，就以为问题已修复。没有追问：*这个守卫能被触发吗？* 守卫触发的前提是 `aggr_out` 已经有正确的 N_dst——而这依赖于 `propagate()` 拿到正确的 size，这才是真正的入口。**修复卫兵的前提条件，而不只是卫兵本身。**

**解决**：
1. `SpatialTemporalGraphConv.forward()` 新增 `size: Optional[Tuple[int, int]] = None` 参数，并将其传给两条路径（普通 / gradient checkpoint）里的 `propagate()`。
2. `GraphNativeEncoder.forward()` 在调用 `conv(x_src, ...)` 时传入 `size=(x_src.shape[0], x.shape[0])`，其中 `x` 是目标节点的当前特征张量（N_dst 由此推导）。
3. `compute_loss()` 在 `F.huber_loss` 之前新增 N 轴不匹配的 `RuntimeError`，防止 broadcasting 静默掩盖问题、引发后续迷惑性报错。

**影响文件**：`models/graph_native_encoder.py`、`models/graph_native_system.py`、`AGENTS.md`

---

### [2026-02-21] EEG enhancement 原地修改 data 导致跨 epoch "backward through graph" 错误

**症状**：Epoch 1 训练正常完成，Epoch 2 第一步 `scaler.scale(total_loss).backward()` 抛出 `RuntimeError: Trying to backward through the graph a second time`。

**根因**：`train_step()` 做了 `data['eeg'].x = eeg_x_enhanced`，**永久改写了 `data_list` 里的原始数据对象**。`eeg_x_enhanced` 是 EEG handler 的输出（`requires_grad=True`，有 `grad_fn`）。Epoch 1 的 `backward()` 释放了计算图的 saved tensors，但 `data['eeg'].x` 仍指向那个 `grad_fn` 已被释放的张量。Epoch 2 的 `eeg_handler(data['eeg'].x)` 在这个已释放的图上构建新图，再调用 `backward()` 时访问已释放的 saved tensors → 报错。

**思维误区**：以为 `data.to(self.device)` 会重新创建张量，从而切断上一步的图。实际上 `.to()` 对已在目标设备的张量是 no-op，并不会切断梯度链。**对 `data_list` 里的对象做原地修改，就是在 epochs 之间共享可变状态——这是典型的隐式状态依赖陷阱。**

**正确思路**：凡是"将计算图中的输出写回共享数据结构"的操作，都要问：*这个数据结构会被下一次迭代复用吗？* 如果是，必须在本次迭代结束前恢复到原始（detached）值，或者不修改原始对象。

**解决**：`train_step()` 用 `try-finally` 块保证 `data['eeg'].x` 在步骤结束后（无论是否抛出异常）恢复为保存的 `original_eeg_x`，使每个 epoch 都从原始未增强的张量开始。

**影响文件**：`models/graph_native_system.py`、`AGENTS.md`

---

### [2026-02-23] 硬编码顺序假设破坏 EEG→fMRI 设计意图

**思维误区**：`create_model` 写了 `edge_types.append((node_types[0], 'projects_to', node_types[1]))`，以为 EEG 一定是 `node_types[0]`。实际上如果 `config['data']['modalities']` 写为 `["fmri", "eeg"]`，就会建立 `('fmri', 'projects_to', 'eeg')` 跨模态边，完全颠倒了方向。

**正确思路**：凡是有方向性的设计意图（"A→B"），不应依赖参数的位置顺序，而应用**明确的模态名**来确定方向。问自己：*"如果用户改变了列表顺序，逻辑链会断吗？"*

**解决**：
1. `create_model` 改为：若 `'eeg' in node_types and 'fmri' in node_types`，显式添加 `('eeg', 'projects_to', 'fmri')`；其他模态组合使用通用回退。
2. `build_graphs` 中的 `merged_graph['eeg', 'projects_to', 'fmri']` 已是显式，无需改动。

**影响文件**：`main.py`、`AGENTS.md`

---

## 四、文档格式规范（必须遵守）

**项目永远只保留以下四个 MD 文件，不得新增，不得删除，每次修改后同步更新：**

| 文件 | 用途 |
|------|------|
| `AGENTS.md` | **本文件**：AI Agent 指令 + 错误记录 |
| `SPEC.md` | 项目规范说明：目标、设计理念、架构（面向 Agent，使其可复现项目） |
| `USERGUIDE.md` | 项目使用说明：面向非专业人士，简洁明了 |
| `CHANGELOG.md` | 更新日志：所有版本变更 + 待优化事项 |

**严禁**：在 `docs/`、根目录或任何位置创建第五个 MD 文件。如需新内容，写入上述四个文件的对应章节。

---

### [2026-02-24] HeteroConv.convs 用 tuple key 访问导致 KeyError（第一次 forward 即崩溃）

**思维误区**：看到 `HeteroConv(conv_dict)` 接受 `{(src, rel, dst): conv}` 形式的字典，就以为内部存储也是用 tuple 作为 key。

**根因**：PyG 的 `HeteroConv.__init__` 将卷积存入 `nn.ModuleDict`：
```python
self.convs = nn.ModuleDict({'__'.join(key): module for key, module in convs.items()})
```
key 是 `'eeg__projects_to__fmri'`（字符串），不是 tuple。`GraphNativeEncoder.forward()` 用 `stgcn.convs[edge_type]`（tuple）访问，必然 `KeyError`。这意味着编码器从未成功运行过。

**正确思路**：每次在自定义 forward 中绕过 `HeteroConv.forward()` 手动访问其内部卷积时，必须问：**内部 dict 的 key 格式是什么？** PyG 约定是 `'__'.join(edge_type_tuple)`。

**修复**：`stgcn.convs['__'.join(edge_type)]`

---

### [2026-02-24] 整个 v5_optimization 配置块从未被读取（死配置）

**思维误区**：看到 YAML 里有详细的 `v5_optimization.adaptive_loss.alpha`、`v5_optimization.eeg_enhancement.entropy_weight` 等参数，就以为它们被传入了对应模块。

**根因**：`GraphNativeTrainer.__init__()` 中 `AdaptiveLossBalancer`、`EnhancedEEGHandler`、`EnhancedMultiStepPredictor` 的所有参数都是硬编码默认值；config 中的对应值从未被读取。

**正确思路**：每次在代码里硬编码一个"配置参数"时，问：**这个值是否也出现在 YAML 里？如果是，哪一方是权威来源？** YAML 应永远是用户可见的权威；代码里不应有"隐形"覆盖。

**修复**：为 `GraphNativeTrainer` 添加 `optimization_config: Optional[dict]` 参数，为 `GraphNativeBrainModel` 添加 `predictor_config: Optional[dict]` 参数，`main.py` 传入 `config['v5_optimization']`。

---

### [2026-02-24] EnhancedGraphNativeTrainer optimizer 只覆盖 base_model（增强模块无梯度）

**思维误区**：`super().__init__(model=model.base_model)` 之后立即 `self.model = model`，以为 optimizer 会自动"跟随"新的 model。

**根因**：`torch.optim.AdamW` 在构造时捕获参数快照；后续修改 `self.model` 不会更新 optimizer 的参数组。`ConsciousnessModule`、`CrossModalAttention`、`HierarchicalPredictiveCoding` 的参数有梯度但永远不会被更新。

**正确思路**：每次 `self.model = new_model` 替换模型后，问：**optimizer 里的参数组是否仍然正确？** 如果不是，必须重新创建 optimizer（或 `optimizer.add_param_group()`）。

**修复**：在 `EnhancedGraphNativeTrainer.__init__()` 的 `super()` 调用后，用 `self.model.parameters()` 重新创建 optimizer。

---

### [2026-02-24] ConsciousGraphNativeBrainModel 用重建输出（信号空间）作为 CrossModalAttention 的输入（潜空间）

**思维误区**：`reconstructions.get('eeg')` 听起来像"编码器的输出"，实际上是**解码器的输出**（`[N, T, 1]`），而 `CrossModalAttention` 期望 `[batch, N, hidden_dim=256]`。

**根因**：`ConsciousGraphNativeBrainModel.forward()` 原来调用 `base_model(data)` 拿到 `(reconstructions, predictions)`，没有请求 `return_encoded=True`，因此无法拿到真正的潜表征；只能用重建输出作为"代理"，这在 shape 和语义上都是错误的。

**正确思路**：`CrossModalAttention`（以及任何需要"高维潜特征"的模块）应当接收编码器输出（`[N, T, H]`），而非解码器输出（`[N, T, 1]`）。每次引入跨模块的特征传递时，明确标注"来自哪一层、shape 是什么"。

**修复**：调用 `base_model(data, return_encoded=True)` 拿到 encoded dict，使用 `encoded['eeg']` / `encoded['fmri']` 作为跨模态注意力的输入。

---

### [2026-02-25] 好的设计意图被错误实现后移除，正确意图也随之消失

**思维误区**：移除崩溃的 `ModalityGradientScaler` 之后，认为"EEG/fMRI 能量不平衡问题已经被 `AdaptiveLossBalancer` 的 `initial_losses` 归一化处理"。没有追问：`initial_losses` 归一化是第一次 forward 后才生效的，**warmup 阶段（前 5 个 epoch）weight 自适应被禁用，归一化也未启动**，此时 fMRI 的 50× 更大的 MSE 完全主导梯度。

**根因**：`modality_energy_ratios` buffer 被存入 `AdaptiveLossBalancer` 但从不参与任何计算——这是"代码声明了意图，但从未执行意图"的另一个例子（AGENTS.md §2021-02-21 的重现）。

**正确思路**：对每一个"存储但从不使用"的参数/属性，问：**这个值应该在什么时候、以什么方式被使用？** `modality_energy_ratios` 的正确使用时机是 `__init__` 时：用它来计算 energy-aware initial task weights，使 EEG 任务从第一个 gradient step 就得到合理的权重。

**实现原则**：能量平衡应在 **损失空间** 的初始权重中实现（init-time 纯 Python 算术），而非在 **梯度空间** 中通过 post-backward `autograd.grad()` 实现（ModalityGradientScaler 的错误之处）。

**修复**：`AdaptiveLossBalancer.__init__` 中，当 `initial_weights=None` 时，通过解析任务名后缀（`recon_eeg` → `eeg`）查找对应模态能量，计算 `initial_weight ∝ 1/energy`，归一化到 mean=1.0。

---

### [2026-02-25] 1:N EEG→fMRI 对齐是「设计缺失」而非「设计完成」

**场景**：用户数据中 2 个 EEG 条件（GRADON / GRADOFF）共享 1 个 fMRI 扫描（task-CB）。

**思维误区**：`_discover_tasks()` 同时扫描 EEG 和 fMRI 文件名，把"BIDS 文件名中出现的所有 task"等同于"有效的训练 run"。实际上：
- `task-CB` 来自 fMRI 文件，没有 EEG 配对
- 把它作为独立 run 加载 → 单模态 fMRI 图（无跨模态边，对联合训练没有价值）
- GRADON/GRADOFF 找不到同名 fMRI → 静默回退（警告被忽视）→ 看起来工作了，实际上是「碰巧文件名匹配」

**根因**：没有任何配置项让用户声明"哪个 EEG 任务对应哪个 fMRI"。1:N 对齐依靠"顺序回退"实现，而回退在设计上就是应急手段，不是对齐机制。

**正确思路**：每次看到"两种模态的任务名不一致"时，第一个问题应该是：**用户有没有办法显式告诉系统配对关系？** 没有的话，配对逻辑就是不完整的设计。

**修复**：新增 `fmri_task_mapping: dict`（YAML + `BrainDataLoader.__init__`），支持 `{"GRADON": "CB", "GRADOFF": "CB"}` 形式的显式映射；配置后 `_discover_tasks()` 只扫描 EEG 文件（避免幽灵 fMRI-only 任务）；`_load_fmri()` 优先按映射查找。

**影响文件**：`data/loaders.py`、`main.py`、`configs/default.yaml`、`CHANGELOG.md`

---

**TwinBrain**：图原生数字孪生脑训练系统。将 EEG（脑电）和 fMRI（功能磁共振）数据构建为异构图，使用时空图卷积（ST-GCN）在保持图结构的同时对时空特征进行编码，实现多模态脑信号的联合建模与未来预测。

**核心创新**：全程图原生（无序列转换），时空不分离建模，EEG-fMRI 能量自适应平衡。

**当前状态**：V5，生产就绪（MemoryError 已修复）。详见 `SPEC.md`。

---

## 六、EEG→fMRI 设计理念与逻辑链（必读，每次编码前确认）

> **核心原则**：EEG 电极（较少节点）向 fMRI ROI（较多节点）投射信号。N_eeg < N_fmri 是整个跨模态设计的前提。

### 数据形状全链路

| 阶段 | EEG | fMRI |
|------|-----|------|
| 原始数据 | `[N_ch, N_times]` (e.g. 63×75000) | `[X, Y, Z, T]` (e.g. 64×64×40×190) |
| 图节点特征 | `[N_eeg, T_eeg, 1]` (e.g. 63×300×1 截断后) | `[N_fmri, T_fmri, 1]` (e.g. 200×190×1，Schaefer200 atlas) |
| 编码器输入投影 | `[N_eeg, T_eeg, H]` | `[N_fmri, T_fmri, H]` |
| ST-GCN 跨模态消息 | 源：`[N_eeg, T_eeg, H]` → propagate(size=(N_eeg, N_fmri)) → `[N_fmri, T_eeg, H]` → interpolate → `[N_fmri, T_fmri, H]` | 目标 |
| 编码器输出 | `[N_eeg, T_eeg, H]` | `[N_fmri, T_fmri, H]` |
| 解码器输出 | `[N_eeg, T_eeg, 1]` | `[N_fmri, T_fmri, 1]` |
| 损失函数目标 | `data['eeg'].x = [N_eeg, T_eeg, 1]` | `data['fmri'].x = [N_fmri, T_fmri, 1]` |

### 跨模态边逻辑

- **方向**：`('eeg', 'projects_to', 'fmri')` — EEG 为 source，fMRI 为 destination
- **edge_index[0]**：EEG 节点索引（0..N_eeg-1）
- **edge_index[1]**：fMRI 节点索引（0..N_fmri-1）
- **当前映射策略**：随机连接（`connection_ratio=0.1`），适合无坐标配准场景
- **理想映射策略**：EEG 电极坐标（head space mm）→ 最近 fMRI ROI 质心（MNI mm）；需要 EEG-fMRI 坐标配准（coregistration）方可使用

### 关键不变量（违反即 bug）

1. `N_eeg < N_fmri`（EEG 节点 < fMRI 节点）
2. 跨模态边类型必须是 `('eeg', 'projects_to', 'fmri')`，**不依赖 config 列表顺序**
3. `propagate(size=(N_eeg, N_fmri))` 必须传入 size，否则 PyG 默认 (N_eeg, N_eeg) 导致 fMRI 节点数被 EEG 数污染
4. 重建损失中 `recon.shape[0] == target.shape[0]`（节点数一致），违反时应 raise RuntimeError 而非 broadcast

### 违反不变量的症状
- Warning: `Using a target size ([63, 190, 1]) different from input ([1, 190, 1])` → invariant 3 或 4 被违反
- fMRI 解码输出节点数等于 N_eeg（如 63）而非 N_fmri → invariant 3 被违反
- Error: `Trying to backward through the graph a second time` → data_list 中对象被原地修改（见上方错误记录）

---

## 七、训练数据设计原则（重要——防止重蹈已知错误）

### 为什么 "max_seq_len=300" 是错误的训练单元

> **思维误区**：把截断当成内存优化，而不是把截断看成数据设计缺陷。

**EEG 致命问题**：max_seq_len=300 在 250Hz 下 = 1.2 秒。从 1.2 秒 EEG 信号估计节点间相关性（Pearson r）统计上完全不可靠（需至少 10-30 秒，即 2500-7500 个样本点）。这意味着 EEG 图的 edge_index（驱动所有 ST-GCN 消息传递）建立在统计噪声之上。

**数据量问题**：10 被试 × 3 任务 = 30 个训练样本。深度学习模型无法从 30 个样本泛化。

### 正确范式：动态功能连接（dFC）滑动窗口

参见 Hutchison et al. 2013 (Nature Rev Neurosci); Chang & Glover 2010 (NeuroImage)。

| 概念 | 图的哪个部分 | 如何计算 | 为何如此 |
|------|-------------|---------|---------|
| 结构连通性 | `edge_index` | 完整 run 的相关矩阵 | 需要充足数据保证统计可靠 |
| 动态脑状态 | 节点特征 `x` | 时间窗口切片 | 每个窗口 = 一个认知瞬态 |

**数据量对比**：

```
截断模式: 10 sub × 3 task × 1 sample = 30 训练样本
窗口模式: 10 sub × 3 task × 11 win  = 330 训练样本 (11×)
```

### windowed_sampling 配置关键约束

1. 当 `windowed_sampling.enabled: true` 时，**必须设 `max_seq_len: null`**（否则图构建仍使用截断序列，EEG 连通性估计仍不可靠）。
2. 缓存存储**完整 run 图**（topology=全序列相关），窗口切分在运行时从缓存图提取（cheap tensor slice）。
3. `fmri_window_size=50`（TRs）与 `eeg_window_size=500`（samples）对齐的是**认知时长**，不是样本数——两者均约等于 100 秒（fMRI: 50×2s=100s; EEG: 500÷250Hz=2s，注意 EEG/fMRI 一般非同步采集，2s EEG epoch 与 100s fMRI window 各自对应其模态的自然时间尺度）。
4. `edge_index` 在同一 run 的所有窗口间**共享同一对象**（不复制），节省内存。
5. 跨模态预测（EEG→fMRI）需要 `cross_modal_align: true`，此时 `ws_eeg = round(ws_fmri × T_eeg/T_fmri)`；⚠ 会显著增大 EEG 窗口（可能导致 CUDA OOM）。

---

## 八、损失函数体系（关键——防止重蹈死代码错误）

> **教训**：有 3 处精心设计的组件长期是死代码：预测头无梯度、EEG 正则被丢弃、跨模态窗口对齐缺失。每次新增 loss 组件，必须检查以下完整链路。

### 损失函数调用链路（必须完整）

```
train_step()
    eeg_handler() → eeg_info['regularization_loss']  ← 必须加入 total_loss
    model.forward(return_encoded=True) → reconstructed, _, encoded
    model.compute_loss(data, reconstructed, encoded=encoded)
        ├── recon_{node_type}: 重建损失（decoder 输出 vs 原始信号）
        └── pred_{node_type}: 潜空间预测损失（context→predict future，均在 H 空间）
    loss_balancer(losses)  ← 只平衡 recon_* 和 pred_*，不处理 eeg_reg
    total_loss += eeg_info['regularization_loss']   ← eeg_reg 固定权重，不参与平衡
    total_loss.backward()
```

### 每种损失的设计意图

| 损失名 | 空间 | 目的 | 权重 |
|--------|------|------|------|
| `recon_eeg` | 原始 C=1 | EEG 信号重建 | 自适应（loss_balancer） |
| `recon_fmri` | 原始 C=1 | fMRI 信号重建 | 自适应（loss_balancer） |
| `pred_eeg` | 潜空间 H | EEG 潜向量未来预测（含跨模态混合） | 自适应（loss_balancer） |
| `pred_fmri` | 潜空间 H | fMRI 潜向量未来预测（含跨模态混合） | 自适应（loss_balancer） |
| `eeg_reg` | 原始 C=1 | 防止 EEG 静默通道崩塌（熵+多样性+活动） | 固定 0.01 × 3 |

### 为什么"潜空间预测"隐式包含跨模态信息

ST-GCN 编码器含 EEG→fMRI 跨模态边。因此：
- `h_fmri` 已包含通过 EEG 电极发来的消息
- "预测 fMRI 潜向量未来" = "用含 EEG 信息的表征预测含 EEG 信息的未来"
- 等价于一种软跨模态预测，无需专用跨模态预测头

真正的跨模态预测（EEG context → fMRI future，空间维度不同）需要额外的跨模态预测头，目前未实现（future work）。

---

## 九、数字孪生脑的根本目的与当前架构差距（V5.14 深度分析）

### 数字孪生脑应该是什么

| 维度 | 数字孪生定义 | 当前 V5.14 实现 | 差距 |
|------|-------------|-----------------|------|
| **个性化** | 特定于某一个体大脑 | 所有被试共享模型参数 | 未实现 |
| **动态拓扑** | 连接模式随认知状态改变 | V5.14 新增 DynamicGraphConstructor | **已实现** |
| **跨会话预测** | 预测下次扫描的脑状态 | 在同一 run 内预测未来窗口 | 部分实现 |
| **干预响应** | 模拟刺激/药物对脑活动的影响 | 未实现 | 未实现 |
| **自我演化** | 随学习/发育更新模型 | 需要纵向数据 + 持续学习 | 未实现 |

### 三个最重要的架构差距（按优先级排序）

**Gap 1（已修复）：动态图拓扑** ← V5.14 DynamicGraphConstructor
- 原问题：edge_index 在数据预处理阶段固定，无法反映认知状态的动态变化
- 解决：每个 ST-GCN 层从当前节点特征动态推算软邻接，与静态拓扑混合

**Gap 2（未实现）：被试特异性嵌入（真正的个性化）**
- 每个被试学一个可学习的嵌入向量 `subject_embed[subject_id]`，在 forward() 开始时加到节点特征上
- 推理时只需 fine-tune 该嵌入（frozen encoder），即"few-shot personalization"
- 实现要点：
  1. `GraphNativeBrainModel` 加 `nn.Embedding(num_subjects, hidden_channels)`
  2. `data` 中存储 `subject_idx` (int)
  3. `forward()` 开始时 `x += self.subject_embed(subject_idx)`
  4. `create_model()` 从图列表推断 num_subjects

**Gap 3（未实现）：跨会话预测（真正的"孪生"预测力）**
- 当前 pred_loss 是 within-run（context→future in same scan）
- 真正的孪生应能预测 next-session brain state given current session
- 需要：跨会话数据对、subject-specific state persistence

### 为什么 Gap 2 比 Gap 3 更优先
- Gap 2 直接由现有训练数据（多被试）实现
- Gap 3 需要额外的纵向设计（同一被试多次扫描）
- Gap 2 实现后，每个被试的嵌入即是"个人大脑的数字指纹"

### DynamicGraphConstructor 的正确使用姿势
- 默认 `use_dynamic_graph: false`（后向兼容）
- 研究场景推荐 `true`：对认知神经科学应用，功能连接动态性是核心现象
- 小数据集（< 50 样本）建议谨慎：动态图引入额外参数（每层 `node_proj + mix_logit`），可能过拟合
- `k_dynamic_neighbors` 建议：fMRI(N=200) 设 10；EEG(N≤64) 设 5
