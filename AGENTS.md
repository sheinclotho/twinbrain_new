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

### [2026-02-21] Decoder 的 ConvTranspose1d 悄悄改变了时序长度

**思维误区**：以为 `ConvTranspose1d(kernel_size=4, stride=1, padding=1)` 和 `Conv1d` 是对称的，输出长度不变。实际上 ConvTranspose1d 的输出公式是 `(T-1)*stride - 2*padding + kernel_size`，stride=1, padding=1, kernel_size=4 → `T+1`。3 层后 T+3。

**正确思路**：ConvTranspose1d 的语义是"逆卷积/转置卷积"，设计用于**上采样**（stride>1 才有意义）。**stride=1 时应该用 Conv1d，不应该用 ConvTranspose1d。** 一旦用了 ConvTranspose1d(stride=1)，就要立刻验证输出尺寸公式。

**解决**：对 stride=1 的层改用 `Conv1d(kernel_size=3, padding=1)`，精确保留 T。对 stride=2 的层（真正上采样）仍用 ConvTranspose1d。

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

## 五、项目背景速览

**TwinBrain**：图原生数字孪生脑训练系统。将 EEG（脑电）和 fMRI（功能磁共振）数据构建为异构图，使用时空图卷积（ST-GCN）在保持图结构的同时对时空特征进行编码，实现多模态脑信号的联合建模与未来预测。

**核心创新**：全程图原生（无序列转换），时空不分离建模，EEG-fMRI 能量自适应平衡。

**当前状态**：V5，生产就绪（MemoryError 已修复）。详见 `SPEC.md`。
