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
