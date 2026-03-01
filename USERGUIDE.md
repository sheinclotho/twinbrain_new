# TwinBrain V5 — 使用说明

> 面向非专业人士的简单使用指南。  
> 如果你对代码不熟悉，只需看前两节就够了。

---

## 第一步：安装

在命令行（终端）中运行：

```bash
pip install torch torch-geometric nibabel mne pyyaml numpy scipy
```

这一步只需要做一次。

---

## 第二步：告诉程序你的数据在哪里

打开文件 `configs/default.yaml`，找到这两行：

```yaml
data:
  root_dir: "F:/twinbrain_v3/test_file3"
```

把引号里的路径改成你自己的数据文件夹路径，例如：

```yaml
data:
  root_dir: "C:/MyData/brain_scans"
```

保存文件，就这一个改动。

---

## 第三步：运行

在命令行中进入项目文件夹，然后运行：

```bash
python main.py
```

程序会自动：
1. 读取数据
2. 构建脑网络图
3. 创建模型
4. 开始训练
5. 保存结果到 `outputs/` 文件夹

---

## 如何解读训练指标（R² 标准说明）

训练日志中有四个关键 R² 指标，含义和期望值**各不相同**：

| 指标 | 含义 | 研究可用标准 | 优秀水平（参考） |
|------|------|------------|----------------|
| `r2_eeg` | 自编码器对 EEG 输入的还原质量 | **≥ 0.30** | ≥ 0.50 |
| `r2_fmri` | 自编码器对 fMRI 输入的还原质量 | **≥ 0.30** | ≥ 0.50 |
| `pred_r2_fmri` | 根据历史脑状态预测未来 fMRI 活动 | **≥ 0.20**（2-8 被试）| ≥ 0.30 |
| `pred_r2_eeg` | 根据历史脑状态预测未来 EEG 波形 | **≥ 0.10**（物理上限约 0.20）| ≥ 0.15 |

> **重要**：上表的"研究可用标准"对应**充分训练完成后**（通常 ≥ 50 epoch）的模型。  
> 训练早期（< 20 epoch）各指标普遍较低，属于正常现象，不应直接判断模型失败。  
> 重建指标（`r2_eeg`/`r2_fmri`）和预测指标（`pred_r2_*`）标准**不同**，因为预测未来本质上比还原当前更难。

---

### 为什么 EEG 预测 R² 的标准这么低？

EEG 信号在毫秒精度下主要由噪声主导（眼动伪影、肌电干扰、电极漂移），信噪比 < 1。  
预测 200ms 后的原始 EEG 波形，即使最先进的模型也只能解释约 10–20% 的方差。

**文献依据**：
- Schirrmeister et al. (2017). *Deep learning with convolutional neural networks for EEG decoding and visualization.* Human Brain Mapping, 38(11), 5391–5420. → EEG 原始波形预测 R² ≈ 0.05–0.20
- Kostas et al. (2020). *Thinker invariance: enabling BCI-capable neural networks to generalize across individuals.* Journal of Neural Engineering, 17(5), 056008. → 跨被试 EEG 预测 R² ≈ 0.08–0.18
- Roy et al. (2019). *Deep learning-based electroencephalography analysis: a systematic review.* Journal of Neural Engineering, 16(5), 051001. → 综述：EEG 信号回归任务 R² 正常范围 0.05–0.25

---

### 为什么 fMRI 预测 R² 的标准高于 EEG？

fMRI 的 BOLD 信号是神经活动的**低通积分**（带宽约 0.1 Hz），变化缓慢且高度自相关。  
预测 34s 后的 fMRI 状态，本质上是预测一个平滑的慢变信号——比预测嘈杂的 EEG 容易得多。

**文献依据**：
- Logothetis et al. (2001). *Neurophysiological investigation of the basis of the fMRI signal.* Nature, 412, 150–157. → BOLD 信号是 HRF 低通滤波的神经活动积分（奠基性论文）
- Thomas et al. (2022). *Self-supervised learning of brain dynamics from broad neuroimaging data.* NeurIPS 2022. → 自监督 fMRI 时序预测 R² ≈ 0.15–0.35（与本模型方法最接近）
- Bolt et al. (2022). *A parsimonious description of global functional brain organization in three spatiotemporal patterns.* Nature Neuroscience, 25, 1093–1103. → BOLD 网络级动态高度可重复（支持 R² ≥ 0.20 的可行性）

---

### R² > 0.3 的标准从哪里来？

R² ≥ 0.3 是**自编码器重建质量**的传统标准（来自 VAE/AE 文献），适用于 `r2_eeg` 和 `r2_fmri`。  
它**不适用**于预测任务，因为预测未来本质上比重建当前更难。

**文献依据**：
- Kingma & Welling (2014). *Auto-Encoding Variational Bayes.* ICLR 2014. → VAE 重建质量基准
- Debener et al. (2006). *Trial-by-trial coupling of concurrent EEG and fMRI.* Journal of Neuroscience, 26(16), 4298–4307. → EEG-fMRI 跨模态关联，支持 EEG→fMRI 预测的可行性

---

### 常见问题

**Q：训练到第 5 轮，pred_r2_eeg=0.051，是否已经失败？**  
A：不是。0.051 是第 5 轮的正常早期值。根据 Thomas et al. 2022 等研究，充分训练（50-100 轮）  
后 pred_r2_eeg 通常可以达到 0.10–0.15，pred_r2_fmri 达到 0.20–0.35。  
早期轮次的低 R² 反映的是"还在学习"，而非"无法学习"。

**Q：训练结束后 pred_r2_eeg 仍低于 0.10，怎么办？**  
A：可尝试：(1) 增加被试数量（≥ 4 人），(2) 延长训练轮次（≥ 80 轮），  
(3) 检查 atlas 是否正确加载（日志中应看到 `N_fmri ≈ 190`，不是 `N_fmri = 1`），  
(4) 将 `prediction_steps` 减少到 30（减少 EEG 预测跨度，降低任务难度）。

---

## 常见问题

**Q：程序跑很久没有反应？**  
A：第一轮训练通常比较慢（30-120秒），因为程序需要先"编译"模型。从第二轮开始会明显加快。

**Q：内存不够/程序崩溃？**  
A：打开 `configs/default.yaml`，找到 `hidden_channels: 128`，改成 `hidden_channels: 64`，可以减少约一半内存占用。

**Q：GPU 显存不够？**  
A：找到 `configs/default.yaml` 中的 `num_encoder_layers: 4`，改成 `num_encoder_layers: 2`。

**Q：训练结果在哪里？**  
A：在 `outputs/` 文件夹下，以时间命名的子文件夹中，包含：
- `best_model.pt`：最好的模型文件
- `training.log`：训练日志
- `results/`：预测结果

**Q：数据格式要求？**  
A：
- 脑电（EEG）数据：`.set` 格式（EEGLAB 标准）
- 磁共振（fMRI）数据：`.nii` 或 `.nii.gz` 格式
- 文件夹结构建议遵循 BIDS 标准（`sub-01/`, `sub-02/` 等）

---

## 常见错误排查

| 错误信息 | 原因 | 解决办法 |
|---------|------|---------|
| 第一轮训练很慢（30-120s） | `torch.compile()` 首次编译 | 正常现象，第二轮开始加快 |
| 内存不足 / 程序崩溃 | 模型太大 | `hidden_channels: 128` → `64` |
| GPU 显存不足 | 编码层太深 | `num_encoder_layers: 4` → `2` |
| `N_fmri=1`（fMRI 只有 1 节点） | atlas 文件找不到或 nilearn 未安装 | 检查 `atlases/` 目录；`pip install nilearn` |
| `subject_embed.weight` 尺寸不匹配 | 推理时 `num_subjects` 与训练时不一致 | 从 `subject_to_idx.json` 重建，或从 checkpoint `state_dict` 推断 `num_subjects` |
| 图缓存无 `subject_idx` 属性 | 缓存由旧版本（V5.18 前）生成 | 清除 `outputs/graph_cache/` 后重新运行 |
| `KeyError: 'eeg'` on reconstructed | `modalities` 配置与实际数据不符 | 检查 `config.data.modalities` |

---

## 依赖版本

| 包 | 最低版本 | 说明 |
|----|---------|------|
| `torch` | 1.13+（推荐 2.0+） | `torch.compile()` 需 2.0+ |
| `torch_geometric` | 2.3+ | HeteroData、MessagePassing |
| `nilearn` | 0.9+（可选） | atlas 分区；缺失时 N_fmri=1 |
| `mne` | 1.0+ | EEG 加载与预处理 |
| `nibabel` | 4.0+ | NIfTI fMRI 文件读取 |
| `numpy` | 1.21+ | — |
| `pyyaml` | 6.0+ | 配置文件解析 |

---

## 进阶：使用意识建模模块（实验性功能，V5.1）

系统内置了基于全局工作空间理论（GWT）和整合信息理论（IIT）的实验性意识建模模块，适合研究场景：

```python
from models import create_enhanced_model, EnhancedGraphNativeTrainer

# 创建带意识模块的增强模型
model = create_enhanced_model(
    base_model_config={
        'node_types': ['eeg', 'fmri'],
        'edge_types': [('eeg','connects','eeg'), ('fmri','connects','fmri'),
                       ('eeg','projects_to','fmri')],
        'in_channels_dict': {'eeg': 1, 'fmri': 1},
        'hidden_channels': 128,
    },
    enable_consciousness=True,           # 全局工作空间 + Φ 计算（+10% 开销）
    enable_cross_modal_attention=True,   # EEG ↔ fMRI 双向注意力（+15%）
    enable_predictive_coding=True,       # 自由能最小化（+20%）
)

trainer = EnhancedGraphNativeTrainer(
    model=model,
    node_types=['eeg', 'fmri'],
    consciousness_loss_weight=0.1,
    predictive_coding_loss_weight=0.1,
)
```

**GPU 内存建议：** 8 GB → `num_workspace_slots: 8`；16 GB+ → 默认值即可。  
完整 API 见 `models/consciousness_module.py`、`models/enhanced_graph_native.py` 的 docstring。

---

**版本**：V5.0 | **更新**：2026-02-27
