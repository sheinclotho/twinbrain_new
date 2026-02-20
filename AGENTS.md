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

> 每次发现重要 bug 或陷阱时，追加到此节。格式：`[日期] 问题摘要 → 根因 → 解决方案`

### [2026-02-20] MemoryError in ST-GCN temporal loop

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
