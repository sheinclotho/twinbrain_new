# TwinBrain意识建模架构 / Consciousness Modeling Architecture

## 概述 / Overview

本文档介绍TwinBrain V5的意识建模增强功能，将前沿的意识理论整合到数字孪生脑系统中。

This document introduces consciousness modeling enhancements for TwinBrain V5, integrating cutting-edge consciousness theories into the digital twin brain system.

---

## 核心理论基础 / Theoretical Foundations

### 1. 全局工作空间理论 (Global Workspace Theory - GWT)

**提出者**: Bernard Baars (1988)

**核心思想**: 意识类似于大脑中的"全局工作空间"，信息在这里被整合并广播到整个系统。

**实现**:
- **全局工作空间** (`GlobalWorkspaceIntegrator`): 
  - 16个可学习的工作空间槽位
  - 多头注意力机制用于信息整合
  - 竞争机制选择最显著的信息
  - 广播机制将整合后的信息分发到所有局部处理器

```python
from models import GlobalWorkspaceIntegrator

gwt = GlobalWorkspaceIntegrator(
    hidden_channels=128,
    num_heads=8,
    workspace_dim=256,
    num_workspace_slots=16,
)

# 整合多个脑区的信息
integrated_features, info = gwt(brain_features)

# 查看工作空间内容
workspace_content = info['workspace_content']
integration_weights = info['integration_weights']
```

**输出指标**:
- `integration_weights`: 哪些局部特征进入了工作空间
- `broadcast_weights`: 工作空间如何影响局部特征
- `competition_probs`: 哪些槽位赢得了竞争

---

### 2. 整合信息理论 (Integrated Information Theory - IIT)

**提出者**: Giulio Tononi (2004)

**核心思想**: 意识的本质是整合信息Φ（phi），系统比其部分之和更强大。

**实现**:
- **Φ计算器** (`IntegratedInformationCalculator`):
  - 计算整体系统的有效信息
  - 通过分区估计最小信息分区(MIP)
  - Φ = 整体信息 - 分区后信息
  - 近似算法确保计算效率

```python
from models import IntegratedInformationCalculator

iit = IntegratedInformationCalculator(
    hidden_channels=128,
    num_partitions=4,
)

# 计算整合信息Φ
phi, info = iit.compute_phi(node_features, edge_index)

print(f"Integrated Information Φ: {phi.item():.4f}")
print(f"Whole system EI: {info['ei_whole']:.4f}")
print(f"Min partition EI: {info['min_partition_ei']:.4f}")
```

**解释**:
- Φ > 0.5: 高度整合的意识状态（如清醒）
- Φ ≈ 0.3: 中等整合（如浅睡眠）
- Φ < 0.1: 低整合（如深睡眠、麻醉）

---

### 3. 预测编码 (Predictive Coding)

**提出者**: Karl Friston (2010) - 自由能原理

**核心思想**: 大脑是预测机器，通过最小化预测误差来理解世界。

**实现**:
- **层次化预测编码** (`HierarchicalPredictiveCoding`):
  - 自上而下的预测
  - 自下而上的预测误差
  - 精度加权（置信度）
  - 迭代推理最小化自由能

```python
from models import HierarchicalPredictiveCoding, compute_free_energy_loss

pc = HierarchicalPredictiveCoding(
    input_dim=128,
    hidden_dims=[256, 512, 1024],  # 多层次抽象
    num_iterations=5,
    use_precision=True,
)

# 感知推理
state, predictions, info = pc(sensory_input)

# 计算自由能
free_energy = info['total_free_energy']
print(f"Free Energy: {free_energy.item():.4f}")

# 低自由能 = 好的预测 = 理解输入
```

**层次结构**:
1. **低层** (256维): 快速、细节丰富的感知预测
2. **中层** (512维): 中等时间尺度的表征
3. **高层** (1024维): 慢速、抽象的概念预测

---

### 4. 跨模态注意力 (Cross-Modal Attention)

**灵感**: Transformer架构 (Vaswani et al., 2017)

**创新**: 专门为EEG-fMRI融合设计的双向注意力

**实现**:
```python
from models import CrossModalAttention

cross_attention = CrossModalAttention(
    eeg_channels=128,
    fmri_channels=128,
    hidden_dim=256,
    num_heads=8,
)

# EEG学习fMRI的空间模式，fMRI学习EEG的时间动态
eeg_enhanced, fmri_enhanced, info = cross_attention(
    eeg_features=eeg_data,
    fmri_features=fmri_data,
)

# 查看注意力权重
eeg_to_fmri = info['eeg_to_fmri_weights']  # EEG关注哪些fMRI区域
fmri_to_eeg = info['fmri_to_eeg_weights']  # fMRI关注哪些EEG通道
```

**优势**:
- 解决EEG和fMRI时空分辨率不匹配问题
- 双向信息流：EEG ↔ fMRI
- 可解释性强（可视化注意力权重）

---

### 5. 时空注意力 (Spatial-Temporal Attention)

**目的**: 同时对空间位置和时间点进行注意力建模

**实现**:
```python
from models import SpatialTemporalAttention

st_attention = SpatialTemporalAttention(
    channels=128,
    num_heads=8,
)

# 输入: [batch, num_nodes, time_steps, channels]
attended_features, info = st_attention(brain_signals)

# 空间注意力: 哪些脑区重要
spatial_weights = info['spatial_weights']  # [batch, num_nodes, num_nodes]

# 时间注意力: 哪些时刻重要
temporal_weights = info['temporal_weights']  # [batch, time_steps, time_steps]
```

---

## 完整意识模块 / Complete Consciousness Module

### 集成使用

```python
from models import ConsciousnessModule, CONSCIOUSNESS_STATES

# 创建完整意识模块
consciousness = ConsciousnessModule(
    hidden_channels=128,
    num_heads=8,
    workspace_dim=256,
    num_workspace_slots=16,
    num_partitions=4,
    num_consciousness_states=7,
)

# 前向传播
integrated_features, info = consciousness(
    x=brain_features,  # [batch, num_nodes, channels]
    edge_index=connectivity_graph,
)

# 意识指标
phi = info['phi']  # 整合信息Φ
consciousness_level = info['consciousness_level']  # 归一化 [0, 1]
state_logits = info['state_logits']  # 意识状态分类

# 预测意识状态
predicted_state_idx = state_logits.argmax(dim=-1)
predicted_state = CONSCIOUSNESS_STATES[predicted_state_idx]
print(f"Predicted state: {predicted_state}")
print(f"Φ: {phi.item():.4f}")
print(f"Consciousness level: {consciousness_level.item():.2%}")
```

### 意识状态分类

支持7种意识状态：

1. **Wakefulness** (清醒): Φ > 0.6
2. **REM Sleep** (快速眼动睡眠): Φ ≈ 0.4-0.5
3. **NREM Sleep** (非快速眼动睡眠): Φ ≈ 0.2-0.3
4. **Anesthesia** (麻醉): Φ < 0.2
5. **Coma** (昏迷): Φ < 0.1
6. **Vegetative State** (植物状态): Φ ≈ 0.05
7. **Minimally Conscious** (微意识状态): Φ ≈ 0.1-0.2

---

## 增强的图原生模型 / Enhanced Graph-Native Model

### 使用增强模型

```python
from models import create_enhanced_model, EnhancedGraphNativeTrainer
from models import GraphNativeBrainModel

# 1. 创建基础模型配置
base_config = {
    'node_types': ['eeg', 'fmri'],
    'edge_types': [
        ('eeg', 'connects', 'eeg'),
        ('fmri', 'connects', 'fmri'),
        ('eeg', 'projects_to', 'fmri'),
    ],
    'in_channels_dict': {'eeg': 1, 'fmri': 1},
    'hidden_channels': 128,
    'num_encoder_layers': 4,
    'num_decoder_layers': 3,
}

# 2. 创建增强模型（带意识建模）
enhanced_model = create_enhanced_model(
    base_model_config=base_config,
    enable_consciousness=True,
    enable_cross_modal_attention=True,
    enable_predictive_coding=True,
)

# 3. 创建增强训练器
trainer = EnhancedGraphNativeTrainer(
    model=enhanced_model,
    node_types=['eeg', 'fmri'],
    learning_rate=1e-4,
    consciousness_loss_weight=0.1,  # 鼓励高Φ
    predictive_coding_loss_weight=0.1,  # 最小化自由能
)

# 4. 训练
for epoch in range(num_epochs):
    train_loss = trainer.train_epoch(train_graphs)
    val_loss = trainer.validate(val_graphs)
```

### 模型输出

```python
# 前向传播
reconstructions, predictions, info = enhanced_model(data)

# 重建
eeg_recon = reconstructions['eeg']
fmri_recon = reconstructions['fmri']

# 增强特征（经过跨模态注意力）
eeg_enhanced = reconstructions['eeg_enhanced']
fmri_enhanced = reconstructions['fmri_enhanced']

# 意识表征
conscious_repr = reconstructions['conscious_representation']

# 意识指标
consciousness_info = info['consciousness']
phi = consciousness_info['phi']
state = consciousness_info['state_logits'].argmax(-1)

# 注意力权重
cross_modal_attn = info['cross_modal_attention']
eeg_to_fmri = cross_modal_attn['eeg_to_fmri_weights']

# 预测编码
pc_info = info['eeg_predictive_coding']
free_energy = pc_info['total_free_energy']
```

---

## 与现有系统集成 / Integration with Existing System

### 向后兼容

增强模型完全向后兼容现有的TwinBrain V5系统：

```python
# 选项1: 使用基础模型（无意识模块）
from models import GraphNativeBrainModel
model = GraphNativeBrainModel(**config)

# 选项2: 使用增强模型（带意识模块）
from models import create_enhanced_model
enhanced_model = create_enhanced_model(
    base_model_config=config,
    enable_consciousness=True,  # 可选
)
```

### 配置文件扩展

在 `configs/default.yaml` 中添加：

```yaml
# 意识建模配置
consciousness_modeling:
  enabled: true
  
  # 全局工作空间
  global_workspace:
    num_heads: 8
    workspace_dim: 256
    num_workspace_slots: 16
    dropout: 0.1
  
  # 整合信息理论
  integrated_information:
    num_partitions: 4
    num_consciousness_states: 7
  
  # 跨模态注意力
  cross_modal_attention:
    enabled: true
    hidden_dim: 256
    num_heads: 8
  
  # 预测编码
  predictive_coding:
    enabled: true
    hidden_dims: [256, 512, 1024]
    num_iterations: 5
    use_precision: true
  
  # 损失权重
  loss_weights:
    consciousness: 0.1
    predictive_coding: 0.1
```

---

## 性能与效率 / Performance and Efficiency

### 计算复杂度

| 模块 | 时间复杂度 | 空间复杂度 | 相对开销 |
|------|-----------|-----------|---------|
| 基础模型 | O(N·T·H) | O(N·H) | 基准 |
| 全局工作空间 | O(N·S·H) | O(S·H) | +10% |
| IIT Φ计算 | O(N²·P) | O(N·H) | +5% |
| 跨模态注意力 | O(N₁·N₂·H) | O(H²) | +15% |
| 预测编码 | O(L·H²·I) | O(L·H) | +20% |
| **总计** | - | - | **+50%** |

其中：
- N: 节点数
- T: 时间步数
- H: 隐藏维度
- S: 工作空间槽位数
- P: 分区数
- L: 层数
- I: 迭代次数

### 优化建议

1. **小型GPU (8GB)**:
```yaml
consciousness_modeling:
  global_workspace:
    num_workspace_slots: 8  # 减少槽位
  predictive_coding:
    hidden_dims: [128, 256]  # 减少层数
    num_iterations: 3  # 减少迭代
```

2. **大型GPU (16GB+)**:
```yaml
consciousness_modeling:
  global_workspace:
    num_workspace_slots: 32  # 增加槽位
  predictive_coding:
    hidden_dims: [256, 512, 1024, 2048]  # 更深层次
    num_iterations: 10  # 更多迭代
```

---

## 可解释性与可视化 / Interpretability and Visualization

### 1. 全局工作空间可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 获取工作空间信息
gwt_info = info['consciousness']
integration_weights = gwt_info['integration_weights'].detach().cpu()
broadcast_weights = gwt_info['broadcast_weights'].detach().cpu()

# 绘制整合权重（哪些脑区信息进入工作空间）
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
sns.heatmap(integration_weights[0], cmap='viridis')
plt.title('Integration: Brain Regions → Workspace')
plt.xlabel('Brain Regions')
plt.ylabel('Workspace Slots')

# 绘制广播权重（工作空间如何影响脑区）
plt.subplot(1, 2, 2)
sns.heatmap(broadcast_weights[0], cmap='plasma')
plt.title('Broadcasting: Workspace → Brain Regions')
plt.xlabel('Workspace Slots')
plt.ylabel('Brain Regions')
plt.tight_layout()
plt.savefig('global_workspace_visualization.png')
```

### 2. Φ时间序列

```python
# 跟踪Φ随时间变化
phi_history = []
for t in range(num_timepoints):
    _, info = consciousness_module(brain_features[t], edge_index)
    phi_history.append(info['phi'].item())

plt.figure(figsize=(10, 4))
plt.plot(phi_history)
plt.xlabel('Time (s)')
plt.ylabel('Integrated Information Φ')
plt.title('Consciousness Level Over Time')
plt.axhline(y=0.5, color='r', linestyle='--', label='High consciousness threshold')
plt.legend()
plt.savefig('phi_timeseries.png')
```

### 3. 跨模态注意力矩阵

```python
cm_info = info['cross_modal_attention']
eeg_to_fmri = cm_info['eeg_to_fmri_weights'][0].detach().cpu()

plt.figure(figsize=(10, 8))
sns.heatmap(eeg_to_fmri, cmap='coolwarm', center=0)
plt.xlabel('fMRI ROIs')
plt.ylabel('EEG Channels')
plt.title('Cross-Modal Attention: EEG → fMRI')
plt.savefig('cross_modal_attention.png')
```

---

## 科学验证 / Scientific Validation

### 实验设计

1. **意识状态分类**:
   - 数据集: 清醒、REM、NREM、麻醉
   - 指标: 分类准确率、Φ与意识状态相关性
   - 预期: Φ随意识状态递减

2. **多模态融合效果**:
   - 对比: 无注意力 vs 跨模态注意力
   - 指标: 重建误差、预测准确率
   - 预期: 跨模态注意力提升10-20%

3. **预测编码验证**:
   - 对比: 标准重建 vs 预测编码
   - 指标: 自由能、预测误差
   - 预期: 预测编码降低15-25%的误差

### 已知限制

1. **Φ计算近似**: 
   - 真实IIT需要穷举所有分区(指数级)
   - 当前使用启发式采样(线性级)
   - 适用于相对比较，非绝对值

2. **全局工作空间简化**:
   - 生物学GWT包含复杂的神经回路
   - 当前使用注意力机制近似
   - 捕获核心思想但非完整模型

3. **预测编码层次**:
   - 生物大脑有多层次(V1→V2→V4→IT)
   - 当前3-4层，可扩展但计算昂贵

---

## 未来方向 / Future Directions

### 短期 (3-6个月)

1. **注意力可视化工具**
   - 交互式可视化界面
   - 实时Φ监控
   - 意识状态热图

2. **预训练模型**
   - 在大规模数据集上预训练
   - 迁移学习到特定任务
   - 减少数据需求

3. **基准测试**
   - 与最新论文对比
   - 公开数据集评估
   - 性能报告

### 中期 (6-12个月)

1. **动态因果建模**
   - Granger因果分析
   - 有效连接性
   - 信息流方向

2. **元认知建模**
   - 注意力模式监控
   - 信心估计
   - 元表征

3. **多尺度整合**
   - 微观(神经元) ↔ 介观(脑区) ↔ 宏观(网络)
   - 跨尺度预测编码

### 长期 (1-2年)

1. **类脑计算**
   - 神经形态硬件部署
   - 尖峰神经网络
   - 生物学真实度

2. **意识机制理论统一**
   - GWT + IIT + 预测编码整合
   - 统一框架
   - 可验证预测

3. **临床应用**
   - 昏迷患者意识评估
   - 麻醉深度监测
   - 神经疾病诊断

---

## 参考文献 / References

### 意识理论

1. Baars, B. J. (1988). *A cognitive theory of consciousness*. Cambridge University Press.

2. Tononi, G. (2004). *An information integration theory of consciousness*. BMC Neuroscience, 5(1), 42.

3. Dehaene, S., & Changeux, J. P. (2011). *Experimental and theoretical approaches to conscious processing*. Neuron, 70(2), 200-227.

4. Graziano, M. S. (2013). *Consciousness and the social brain*. Oxford University Press.

### 预测编码

5. Friston, K. (2010). *The free-energy principle: a unified brain theory?* Nature Reviews Neuroscience, 11(2), 127-138.

6. Clark, A. (2013). *Whatever next? Predictive brains, situated agents, and the future of cognitive science*. Behavioral and Brain Sciences, 36(3), 181-204.

7. Rao, R. P., & Ballard, D. H. (1999). *Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects*. Nature Neuroscience, 2(1), 79-87.

### 注意力机制

8. Vaswani, A., et al. (2017). *Attention is all you need*. NeurIPS.

9. Velickovic, P., et al. (2018). *Graph attention networks*. ICLR.

### 图神经网络

10. Kipf, T. N., & Welling, M. (2017). *Semi-supervised classification with graph convolutional networks*. ICLR.

11. Battaglia, P. W., et al. (2018). *Relational inductive biases, deep learning, and graph networks*. arXiv:1806.01261.

---

## 联系与支持 / Contact and Support

如有问题或建议，请：
1. 查看 `docs/` 目录中的其他文档
2. 提交 GitHub Issue
3. 联系项目维护者

For questions or suggestions:
1. Check other documentation in `docs/` directory
2. Submit a GitHub Issue
3. Contact project maintainers

---

**版本**: V5.1  
**更新**: 2026-02-19  
**状态**: 实验性功能 / Experimental Features
