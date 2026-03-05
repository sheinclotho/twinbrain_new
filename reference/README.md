# TwinBrain — 参考模块库 (Reference Module Library)

## 文件夹用途

这个文件夹保存了**已实现但尚未集成到主训练流程中**的高级功能模块。

它们不是遗弃代码，而是为模型未来可能的改进提供的**参考实现库**：
每个模块都有完整的实现、单元级注释和科学依据。当 TwinBrain
的主流程稳定后，可以选择性地将其中的组件集成进来，无需从头设计。

> **主训练流程不导入此目录的任何内容**，因此对默认运行没有任何开销影响。

---

## 模块一览

| 文件 | 核心类 | 功能摘要 |
|------|--------|---------|
| `consciousness_module.py` | `GlobalWorkspaceIntegrator`, `IntegratedInformationCalculator`, `ConsciousnessModule` | 意识状态建模：全局工作空间理论（GWT）+ 整合信息论（IIT / Φ 值） |
| `advanced_attention.py` | `CrossModalAttention`, `SpatialTemporalAttention`, `GraphAttentionWithEdges`, `ContrastiveAttention` | 高级注意力机制：跨模态注意力、时空注意力、对比注意力 |
| `predictive_coding.py` | `PredictiveCodingLayer`, `HierarchicalPredictiveCoding`, `ActiveInference`, `compute_free_energy_loss` | 预测编码 / 自由能原理 / 主动推断（用于 TMS 干预仿真） |
| `enhanced_graph_native.py` | `ConsciousGraphNativeBrainModel`, `EnhancedGraphNativeTrainer`, `create_enhanced_model` | 将上述三个模块集成到图原生基础模型中的增强版封装 |
| `visualization_consciousness.py` | `ConsciousnessVisualizer`, `create_sample_visualizations` | 意识指标可视化：GWT 广播权重、Φ 时序曲线、跨模态注意力热图 |
| `consciousness_example.py` | — | 增强版完整训练流程的端到端演示脚本 |

---

## 各模块详解

### 1. `consciousness_module.py` — 意识建模

**理论基础**
- **全局工作空间理论（GWT, Baars 1988）**：大脑中存在一个"全局工作空间"，
  各专门处理器竞争访问它，获胜者的信息被广播到全脑。
- **整合信息论（IIT, Tononi 2004）**：意识对应系统的整合信息量 Φ（phi），
  Φ 越高代表信息越难被还原为独立部分，即系统整合程度越高。

**实现的类**

| 类 | 功能 | 关键参数 |
|----|------|---------|
| `GlobalWorkspaceIntegrator` | 多头注意力实现竞争-广播机制 | `workspace_dim=256`, `num_workspace_slots=16` |
| `IntegratedInformationCalculator` | 近似计算图的 Φ 值（分区对比法） | `num_partitions=4` |
| `ConsciousnessStateClassifier` | 将聚合特征分类为 7 种意识状态 | `num_states=7` |
| `ConsciousnessModule` | 组合以上三者的统一接口 | — |

**意识状态枚举（`CONSCIOUSNESS_STATES`）**
```
wakefulness, rem_sleep, nrem_sleep, anesthesia, coma, vegetative_state, minimally_conscious
```

**对 TwinBrain 的潜在价值**
- 在 EEG+fMRI 联合训练中加入意识状态辅助分类任务，可以使模型在学习预测脑信号的同时，
  隐式区分清醒 / 睡眠 / 麻醉态，提高跨被试泛化能力。
- Φ 值可以作为数字孪生的"意识指标"输出，用于评估干预效果。

**集成方式**
```python
from reference.consciousness_module import ConsciousnessModule

consciousness = ConsciousnessModule(
    hidden_channels=128,
    num_heads=8,
    num_consciousness_states=7,
)
# x: [batch, num_nodes, H]  edge_index: [2, E]
integrated_feat, info = consciousness(x, edge_index)
phi = info['phi']              # 每个样本的 Φ 值
state_logits = info['state_logits']  # 意识状态分类 logits
```

---

### 2. `advanced_attention.py` — 高级注意力机制

**实现的类**

| 类 | 功能 | 输入/输出 | 集成优先级 |
|----|------|---------|---------|
| `CrossModalAttention` | 双向 EEG↔fMRI 交叉注意力，每个模态作为另一个的 Query | `eeg [B,N_eeg,H]` ↔ `fmri [B,N_fmri,H]` | ⭐⭐⭐ 高 |
| `SpatialTemporalAttention` | 空间注意力与时间注意力解耦，分步处理 `[B,N,T,H]` 输入 | `[B,N,T,H]` → `[B,N,T,H]` | ⭐⭐⭐ 高 |
| `GraphAttentionWithEdges` | 带边特征的 GAT（继承 `MessagePassing`） | `x, edge_index, edge_attr` | ⭐⭐ 中 |
| `HierarchicalAttention` | 多尺度分层注意力（粗粒度先全局感知，再局部精化） | `[B,N,H]` → `[B,N,H]` | ⭐⭐ 中 |
| `ContrastiveAttention` | 自监督对比注意力（正样本对互相吸引，负样本互相排斥） | `[B,N,H]` → `[B,N,H]` + loss | ⭐⭐ 中 |

**对 TwinBrain 的潜在价值**
- `CrossModalAttention` 是最高优先级：当前图原生模型的跨模态信息传递依赖图边消息传递，
  显式的交叉注意力机制可以提供更直接、可解释的 EEG→fMRI 信息融合。
- `ContrastiveAttention` 可以替代或补充当前的 InfoNCE 对比学习损失，
  在注意力层面直接执行多模态对齐。

**集成方式（以 CrossModalAttention 为例）**
```python
from reference.advanced_attention import CrossModalAttention

cross_attn = CrossModalAttention(
    eeg_channels=128,
    fmri_channels=128,
    hidden_dim=256,
    num_heads=8,
)
# 在 ST-GCN 编码后调用
eeg_enhanced, fmri_enhanced, attn_info = cross_attn(
    eeg_features,   # [1, N_eeg, 128]
    fmri_features,  # [1, N_fmri, 128]
)
# attn_info['eeg_to_fmri_weights']: EEG→fMRI 注意力图 [N_eeg, N_fmri]
```

---

### 3. `predictive_coding.py` — 预测编码与自由能原理

**理论基础**
- **自由能原理（Friston 2010）**：大脑通过最小化"自由能"（即预测误差的变分上界）
  来理解世界。感知 = 更新内部模型使其与输入匹配；行动 = 改变外部世界使其符合预测。
- **预测编码（Rao & Ballard 1999）**：自顶向下的生成模型产生预测，
  自底向上的通路只传递"预测误差"，而非原始感觉信号。
- **主动推断（Friston et al. 2016）**：行动选择通过最小化期望自由能（EFE）实现，
  同时兼顾"减少不确定性（认知价值）"与"实现目标（实用价值）"。

**实现的类**

| 类 | 功能 |
|----|------|
| `PredictiveCodingLayer` | 单层预测编码：迭代推断（5步），计算精度加权预测误差 |
| `HierarchicalPredictiveCoding` | 多层层级：自底向上传递误差，自顶向下传递预测 |
| `ActiveInference` | 主动推断：采样多个动作候选，选择 EFE 最低的执行 |
| `PredictiveBrainModel` | 完整模型：感知（HPC）+ 行动（AI）统一接口 |
| `compute_free_energy_loss` | 精度加权预测误差损失函数（可直接添加到训练损失） |

**对 TwinBrain 的潜在价值**
- `compute_free_energy_loss` 可以作为附加损失项，在潜空间中对应现有的
  `pred_{nt}` 损失，为预测器提供信息论视角的约束。
- `ActiveInference` 中的 `transition_model`（状态转移模型）与
  `TwinBrainDigitalTwin.simulate_intervention()` 在功能上高度对齐——
  TMS 仿真的本质就是：在状态空间中选择使 EFE 最小的"干预动作"。

**集成方式（compute_free_energy_loss 最简集成）**
```python
from reference.predictive_coding import compute_free_energy_loss

# 在 compute_loss() 中添加：
fe_loss = compute_free_energy_loss(
    predictions=pred_latent,    # 预测的未来潜向量
    targets=future_latent,      # 真实的未来潜向量
    precision=None,             # 或通过额外网络学习精度权重
)
losses['free_energy'] = fe_loss * 0.05  # 小权重，辅助约束
```

---

### 4. `enhanced_graph_native.py` — 增强版图原生模型

**实现的类**

| 类 | 功能 |
|----|------|
| `ConsciousGraphNativeBrainModel` | 在基础 `GraphNativeBrainModel` 外包装三个增强模块，API 与基础模型完全兼容 |
| `EnhancedGraphNativeTrainer` | 继承 `GraphNativeTrainer`，添加意识损失和自由能损失，正确处理 AMP 和梯度累积 |
| `create_enhanced_model` | 工厂函数：一行创建增强版模型 |

**`ConsciousGraphNativeBrainModel` 前向传播流程**
```
输入图数据
    ↓
基础模型前向传播 → (重建结果, 预测结果, 编码特征)
    ↓
跨模态注意力 (CrossModalAttention + SpatialTemporalAttention)
    ↓
意识模块 (GlobalWorkspace + IIT / Φ 计算) [仅 return_consciousness_metrics=True 时]
    ↓
预测编码 (HierarchicalPredictiveCoding) [每个模态独立]
    ↓
输出 (重建结果, 预测结果, [编码特征], [意识信息])
```

**`EnhancedGraphNativeTrainer` 修复了的已知问题**
- 修复了 `super().__init__` 后替换 `self.model` 导致优化器参数失效的 bug
  （重新创建 AdamW，包含全部增强模块参数）
- 意识损失 = `-Φ.mean() × consciousness_loss_weight`（鼓励高整合信息）
- 自由能损失 = `total_free_energy × predictive_coding_loss_weight`

**快速启动**
```python
from reference.enhanced_graph_native import create_enhanced_model, EnhancedGraphNativeTrainer

model = create_enhanced_model(
    base_model_config=dict(
        node_types=['eeg', 'fmri'],
        hidden_channels=128,
        num_encoder_layers=4,
        # ... 与 default.yaml 中 model 节一致
    ),
    enable_consciousness=True,
    enable_cross_modal_attention=True,
    enable_predictive_coding=True,
)

trainer = EnhancedGraphNativeTrainer(
    model=model,
    node_types=['eeg', 'fmri'],
    learning_rate=1e-4,
    consciousness_loss_weight=0.05,   # 建议从小值开始
    predictive_coding_loss_weight=0.05,
)

# 与标准训练流程完全相同：
loss_dict = trainer.train_step(graph)
```

---

### 5. `visualization_consciousness.py` — 意识指标可视化

**`ConsciousnessVisualizer` 提供的图表**

| 方法 | 生成图表 | 输入数据 |
|------|---------|---------|
| `plot_global_workspace` | GWT 竞争概率 + 广播权重热图 | `gwt_info dict` |
| `plot_phi_timeseries` | Φ 值随时间的变化曲线（含阈值参考线） | `phi_history: np.ndarray` |
| `plot_cross_modal_attention` | EEG→fMRI 和 fMRI→EEG 注意力权重热图 | `eeg_to_fmri, fmri_to_eeg tensors` |
| `plot_consciousness_state_trajectory` | 意识状态时序轨迹 + Φ 叠加图 | `state_history, state_names, phi_history` |
| `plot_predictive_coding_hierarchy` | 各层预测误差和自由能曲线 | `pc_info dict` |

---

### 6. `consciousness_example.py` — 端到端演示

完整演示了增强版训练流程：生成合成数据 → 创建增强模型 → 5 个 epoch 训练 →
意识状态分析 → 图表保存。可以直接运行：

```bash
cd twinbrain_new
python reference/consciousness_example.py
```

---

## 集成路线图建议

建议按以下优先级逐步集成，每步验证后再进行下一步：

```
阶段 1（低风险）— 添加 compute_free_energy_loss 作为附加损失项
    └── 修改 graph_native_system.py: compute_loss() 中添加
        from reference.predictive_coding import compute_free_energy_loss

阶段 2（中风险）— 集成 CrossModalAttention
    └── 修改 graph_native_encoder.py: SpatialTemporalGraphConv.forward() 后添加
        跨模态注意力层，与现有图消息传递互补

阶段 3（较高风险）— 添加意识状态辅助任务
    └── 在 build_graphs() 中为图添加 consciousness_label 属性
        在 compute_loss() 中添加意识状态分类辅助损失

阶段 4（完整集成）— 切换到 EnhancedGraphNativeTrainer
    └── 修改 main.py: create_model() 返回 ConsciousGraphNativeBrainModel
        trainer = EnhancedGraphNativeTrainer(...)
```

---

## 依赖说明

所有模块均依赖项目已有的核心依赖（PyTorch、PyG），无额外外部依赖。
`visualization_consciousness.py` 额外需要 `matplotlib` 和 `seaborn`（已在项目中使用）。

---

## 附录：三项核查报告

> 以下内容回应用户的三项核查请求。

---

### 核查一：跨模态注意力机制的实现状态

**结论：生产训练中已部分实现，但与 `reference/advanced_attention.py` 中的 `CrossModalAttention` 是不同的实现方式。**

当前生产代码（`models/graph_native_system.py`）中存在两个跨模态机制：

| 机制 | 所在位置 | 是否在训练中启用 | 性质 |
|------|---------|--------------|------|
| `_cross_modal_align_loss` | `graph_native_system.py:441` | ✅ 是（`use_cross_modal_align=True`，V5.43） | 损失层面的对齐：强制 EEG 和 fMRI 的全局均值潜向量方向一致（CMC 风格余弦对齐） |
| 图消息传递中的跨模态边 | `graph_native_encoder.py` | ✅ 是 | `('eeg', 'projects_to', 'fmri')` 边上的 ST-GCN 消息传递，实现 EEG→fMRI 的图级信息传递 |
| `CrossModalAttention`（MHA） | `reference/advanced_attention.py` | ❌ 否（在 reference/ 中） | 显式双向多头交叉注意力，尚未集成到主训练流程 |

**关键区别**：

- 当前生产代码中的"跨模态"是通过**图边消息传递（GNN）+ 损失对齐**实现的，不是注意力机制。
- `reference/CrossModalAttention` 是基于 `nn.MultiheadAttention` 的显式交叉注意力，功能更强，但尚未集成。
- 两种方式并不互斥：图消息传递捕获拓扑结构，显式注意力捕获所有节点对的全局关系。集成路线见 reference/README.md §集成路线图建议。

---

### 核查二：GWT 和 IIT 模块的科学性评估

**GWT（全局工作空间理论）** — 实现有效，科学争议不影响使用

| 方面 | 评估 |
|------|------|
| 近期证伪进展 | 2023 年 Adversarial Collaboration（Mashour et al., Science 2023）对 GWT 和 IIT 进行了直接对比实验，部分结果不支持 IIT 的预测，但 **GWT 的预测基本得到验证**（后顶叶皮层活动与意识状态相关） |
| `GlobalWorkspaceIntegrator` 的本质 | 多头注意力实现的竞争-广播机制，在工程上等价于 Transformer 的 self-attention。即使 GWT 理论被修正，这个注意力机制本身作为特征提取器仍然有效 |
| 对 TwinBrain 的影响 | GWT 模块可以安全使用，将其视为"全脑注意力池化"而非意识理论的严格实现 |

**IIT（整合信息论）** — Φ 计算有严重近似，科学上应谨慎使用

| 方面 | 评估 |
|------|------|
| 近期证伪进展 | 2023 年对比实验数据更支持 GWT 而非 IIT（前额叶早期活动的预测被否定）。IIT 3.0 的 Φ 计算本身在理论上也受到质疑（Doerig et al., 2021 *Neuroscience & Biobehavioral Reviews*） |
| `IntegratedInformationCalculator` 的近似程度 | 代码注释明确说明是"简化近似"（随机分区法，非严格 MIP 搜索）。真正的 IIT Φ 计算是 NP-hard，当前实现实质上是"有效信息流的余弦相似度均值"，与严格 IIT 定义相去甚远 |
| 对 TwinBrain 的影响 | 如果要发表研究声称测量了 Φ，当前实现**不可直接使用**。但如果仅作为"系统整合度"的近似指标用于分类任务，实用价值保留。**建议**：在论文中将其定义为"图信息整合指数（GII）"而非严格 IIT Φ，以避免科学误导 |

**综合建议**：
- 可以保留 `ConsciousnessModule` 作为辅助任务，但将描述改为"全脑信息整合状态分类"而非"意识状态检测"
- 在任何科学发表中，避免直接引用 IIT 理论框架描述当前 Φ 近似计算

---

### 核查三：Predictive Coding 与扰动分析设计的匹配度

**问题分析**：

用户的设计哲学是：

```
X(t+1) = f(X(t))          # 自主动力系统
X(t) → X(t) + δ → f → trajectory  # 刺激 = 状态扰动
```

研究的是"**内在动力学响应性质（intrinsic response）**"，而非受控系统。

**`reference/predictive_coding.py` 与该设计的匹配度**：

| 组件 | 与用户设计的匹配度 | 说明 |
|------|--------------------|------|
| `PredictiveCodingLayer` | ⭐ 低 | 是层级感知模型（Rao & Ballard 1999），建模的是自顶向下预测误差最小化，不是动力系统扰动响应 |
| `HierarchicalPredictiveCoding` | ⭐ 低 | 同上，是感知推断模型，不是时序轨迹仿真 |
| `ActiveInference` | ⭐⭐ 中 | `transition_model`（状态转移模型）在概念上与用户的 `f(X(t))` 对应，但设计目标是行动选择而非扰动仿真 |
| `compute_free_energy_loss` | ⭐⭐⭐ 高 | 可以作为预测器的附加损失项，对齐用户的"内在动力学学习"目标（精度加权预测误差 ≈ 让模型更好地学习 f） |

**当前 TwinBrain 与用户设计的实际对齐**：

用户描述的扰动分析框架在**生产代码中已有直接对应实现**：

```
models/graph_native_system.py:
  simulate_intervention()          ← X(t) + δ → f → trajectory（impulse 模式）
  compute_effective_connectivity() ← 系统级响应矩阵（时间平均 EC）
```

而 `unity_integration/perturbation_analyzer.py`（本次新增）补充了：
- `compute_response_matrix()` — 保留时间分辨率的 R[i,j,k]（sustained/impulse 两种模式）
- `analyze_response_matrix()` — 空间传播率、时间衰减、传播延迟分析
- `validate_response_matrix()` — 跨初始状态一致性验证

**建议**：`reference/predictive_coding.py` 中的 `compute_free_energy_loss` 可以集成为预测器的额外损失约束（让模型更好地学习内在动力学），但 `PredictiveCodingLayer` 和 `HierarchicalPredictiveCoding` 与用户的扰动分析设计无直接关联。
