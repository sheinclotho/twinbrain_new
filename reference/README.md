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
