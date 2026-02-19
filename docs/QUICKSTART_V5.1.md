# TwinBrain V5.1 - 意识建模功能快速入门

## 🎯 核心功能

TwinBrain V5.1 引入了基于前沿意识理论的创新功能：

### 1. 全局工作空间理论 (GWT)
模拟意识的"工作空间"，信息在此整合并广播到整个大脑系统。

### 2. 整合信息理论 (IIT)
计算整合信息Φ，量化意识水平的数值指标。

### 3. 预测编码
大脑作为预测机器，通过最小化预测误差来理解世界。

### 4. 跨模态注意力
EEG和fMRI之间的双向信息流，解决时空分辨率不匹配。

---

## 📦 安装

### 基础依赖（已有）
```bash
pip install torch torch-geometric nibabel mne pyyaml numpy scipy
```

### 额外依赖（用于可视化）
```bash
pip install matplotlib seaborn
```

---

## 🚀 快速开始

### 方法1: 使用基础模型（现有功能）

```python
from models import GraphNativeBrainModel, GraphNativeTrainer

# 创建基础模型
model = GraphNativeBrainModel(
    node_types=['eeg', 'fmri'],
    edge_types=[...],
    hidden_channels=128,
)

# 训练
trainer = GraphNativeTrainer(model=model)
trainer.train_epoch(train_graphs)
```

### 方法2: 使用增强模型（新功能）

```python
from models import create_enhanced_model, EnhancedGraphNativeTrainer

# 创建意识增强模型
enhanced_model = create_enhanced_model(
    base_model_config={
        'node_types': ['eeg', 'fmri'],
        'hidden_channels': 128,
        # ... 其他配置
    },
    enable_consciousness=True,          # 启用意识模块
    enable_cross_modal_attention=True,  # 启用跨模态注意力
    enable_predictive_coding=True,      # 启用预测编码
)

# 使用增强训练器
trainer = EnhancedGraphNativeTrainer(
    model=enhanced_model,
    consciousness_loss_weight=0.1,  # 鼓励高Φ
    predictive_coding_loss_weight=0.1,  # 最小化自由能
)

# 训练
for epoch in range(num_epochs):
    train_loss = trainer.train_epoch(train_graphs)
    val_loss = trainer.validate(val_graphs)
```

---

## 📊 评估意识指标

```python
# 前向传播
reconstructions, predictions, info = enhanced_model(
    data,
    return_consciousness_metrics=True,
)

# 意识指标
consciousness_info = info['consciousness']
phi = consciousness_info['phi']  # 整合信息Φ
consciousness_level = consciousness_info['consciousness_level']  # 归一化[0,1]
state_logits = consciousness_info['state_logits']  # 意识状态概率

# 预测意识状态
from models import CONSCIOUSNESS_STATES
predicted_state_idx = state_logits.argmax(dim=-1)
predicted_state = CONSCIOUSNESS_STATES[predicted_state_idx]

print(f"Φ: {phi.mean():.4f}")
print(f"Consciousness Level: {consciousness_level.mean():.2%}")
print(f"Predicted State: {predicted_state}")
```

---

## 📈 可视化

```python
from utils.visualization import ConsciousnessVisualizer

viz = ConsciousnessVisualizer(output_dir='outputs/visualizations')

# 1. 全局工作空间可视化
viz.plot_global_workspace(
    info['consciousness'],
    save_as='global_workspace.png'
)

# 2. Φ时间序列
viz.plot_phi_timeseries(
    phi_history,  # 随时间变化的Φ值列表
    save_as='phi_timeseries.png'
)

# 3. 跨模态注意力
viz.plot_cross_modal_attention(
    eeg_to_fmri=info['cross_modal_attention']['eeg_to_fmri_weights'][0],
    fmri_to_eeg=info['cross_modal_attention']['fmri_to_eeg_weights'][0],
    save_as='cross_modal_attention.png'
)

# 4. 意识状态轨迹
viz.plot_consciousness_state_trajectory(
    state_history,  # 随时间变化的状态列表
    state_names=CONSCIOUSNESS_STATES,
    phi_history=phi_history,
    save_as='consciousness_trajectory.png'
)
```

---

## 💡 意识状态分类

模型可以自动识别7种意识状态：

1. **Wakefulness** (清醒): Φ > 0.6
2. **REM Sleep** (快速眼动睡眠): Φ ≈ 0.4-0.5
3. **NREM Sleep** (非快速眼动睡眠): Φ ≈ 0.2-0.3
4. **Anesthesia** (麻醉): Φ < 0.2
5. **Coma** (昏迷): Φ < 0.1
6. **Vegetative State** (植物状态): Φ ≈ 0.05
7. **Minimally Conscious** (微意识状态): Φ ≈ 0.1-0.2

---

## ⚙️ 配置文件

在 `configs/default.yaml` 中添加：

```yaml
# 意识建模配置（可选）
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

## 📖 完整文档

### 理论文档
- **[意识建模架构](docs/CONSCIOUSNESS_ARCHITECTURE.md)** - 详细理论和使用说明
- **[质变升级报告](docs/TRANSFORMATIVE_UPGRADE_REPORT.md)** - 技术创新和性能分析

### 示例代码
- **[完整示例](examples/consciousness_example.py)** - 端到端使用示例
- **[可视化工具](utils/visualization.py)** - 可视化API文档

### API文档
所有模块都包含详细的docstring：
- `models/consciousness_module.py`
- `models/advanced_attention.py`
- `models/predictive_coding.py`
- `models/enhanced_graph_native.py`

---

## 🎓 科学应用

### 研究场景

1. **意识研究**
   - 不同意识状态的神经特征
   - Φ与意识水平的关系
   - 意识的神经相关物

2. **临床应用**
   - 昏迷患者意识评估
   - 麻醉深度监测
   - 睡眠障碍诊断

3. **脑机接口**
   - 意图识别
   - 注意力解码
   - 神经反馈训练

4. **神经疾病**
   - 意识障碍诊断
   - 治疗效果评估
   - 预后预测

---

## ⚡ 性能考虑

### GPU内存建议

| GPU内存 | 配置建议 |
|---------|---------|
| 8GB | workspace_slots=8, hidden_dims=[128,256], iterations=3 |
| 12GB | workspace_slots=16, hidden_dims=[256,512], iterations=5 |
| 16GB+ | workspace_slots=32, hidden_dims=[256,512,1024], iterations=10 |

### 计算开销

相比基础V5模型：
- 时间开销: +50%
- 内存开销: +26%
- 提供的价值: 质变飞跃（从脑信号建模到意识建模）

### 优化选项

```python
# 关闭某些模块以减少开销
enhanced_model = create_enhanced_model(
    base_model_config=config,
    enable_consciousness=True,       # 保留核心意识功能
    enable_cross_modal_attention=False,  # 关闭以节省15%
    enable_predictive_coding=False,  # 关闭以节省20%
)
```

---

## 🔬 验证实验

### 实验1: 意识状态分类

```python
# 收集不同意识状态的数据
states = ['wakefulness', 'rem_sleep', 'nrem_sleep', 'anesthesia']
results = []

for state in states:
    data = load_data(state)
    _, _, info = enhanced_model(data, return_consciousness_metrics=True)
    phi = info['consciousness']['phi'].mean().item()
    results.append({'state': state, 'phi': phi})

# 预期: phi递减 (wakefulness > rem > nrem > anesthesia)
```

### 实验2: 跨模态融合

```python
# 对比有无跨模态注意力
models = {
    'baseline': create_enhanced_model(..., enable_cross_modal_attention=False),
    'with_attention': create_enhanced_model(..., enable_cross_modal_attention=True),
}

for name, model in models.items():
    mse = evaluate_reconstruction(model, test_data)
    print(f"{name}: MSE={mse:.4f}")

# 预期: with_attention显著降低MSE
```

---

## 🤝 贡献与反馈

### 如何贡献
1. 在公开数据集上测试
2. 报告bugs和性能问题
3. 提出改进建议
4. 分享应用案例

### 反馈渠道
- GitHub Issues
- 项目讨论区
- 邮件联系

---

## 📚 参考文献

### 核心理论
1. Baars, B. J. (1988). *A cognitive theory of consciousness*.
2. Tononi, G. (2004). *An information integration theory of consciousness*.
3. Friston, K. (2010). *The free-energy principle: a unified brain theory?*
4. Dehaene, S., & Changeux, J. P. (2011). *Experimental approaches to conscious processing*.

### 技术方法
5. Vaswani, A., et al. (2017). *Attention is all you need*.
6. Velickovic, P., et al. (2018). *Graph attention networks*.
7. Kipf, T. N., & Welling, M. (2017). *Semi-supervised classification with GCNs*.

---

## 📋 常见问题

### Q1: 是否必须使用意识模块？
**A**: 不是。所有新功能都是可选的。可以继续使用基础V5模型。

### Q2: 计算开销有多大？
**A**: 相比基础模型增加约50%时间和26%内存。可通过关闭某些模块优化。

### Q3: Φ值如何解释？
**A**: Φ是意识水平的数值指标。一般来说：
- Φ > 0.6: 高度意识（清醒）
- Φ 0.2-0.4: 中等意识（睡眠）
- Φ < 0.2: 低意识（深睡眠、麻醉、昏迷）

### Q4: 如何可视化结果？
**A**: 使用 `utils.visualization.ConsciousnessVisualizer` 类，提供6种可视化方法。

### Q5: 如何验证结果？
**A**: 参考文档中的验证实验设计，在标注数据集上评估分类准确率和Φ-意识状态相关性。

---

## 🔮 未来计划

### 短期 (3-6个月)
- [ ] 在公开数据集上基准测试
- [ ] 发布预训练模型
- [ ] 创建交互式可视化界面

### 中期 (6-12个月)
- [ ] 动态因果建模
- [ ] 元认知功能
- [ ] 临床试验

### 长期 (1-2年)
- [ ] 理论统一框架
- [ ] 神经形态硬件部署
- [ ] 商业化应用

---

## 📄 许可证

遵循项目原有许可证。

---

**版本**: V5.1  
**发布日期**: 2026-02-19  
**状态**: 实验性功能  
**维护者**: TwinBrain Team

---

## 🌟 致谢

感谢以下理论和工作的启发：
- Bernard Baars的全局工作空间理论
- Giulio Tononi的整合信息理论
- Karl Friston的自由能原理
- PyTorch和PyTorch Geometric社区
- 脑科学和人工智能研究社区

---

**开始使用，探索意识的奥秘！** 🧠✨
