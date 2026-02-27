# TwinBrain V5 — 接口参考手册（API Reference）

> **适用对象**：与 TwinBrain 训练管线集成的前端 Agent、下游分析脚本、推理服务或自动化工作流。
> 本文件描述训练完成后产出的所有持久化文件的**命名规则、目录结构、内容格式与读写接口**。

---

## 目录

1. [输出目录结构总览](#1-输出目录结构总览)
2. [图缓存（Graph Cache）](#2-图缓存-graph-cache)
   - [2.1 命名规则](#21-命名规则)
   - [2.2 内容格式](#22-内容格式)
   - [2.3 读取接口（推荐）](#23-读取接口推荐)
   - [2.4 直接 PyG 读取](#24-直接-pyg-读取)
   - [2.5 跨模态边的特殊处理](#25-跨模态边的特殊处理)
   - [2.6 缓存失效条件](#26-缓存失效条件)
3. [训练检查点（Checkpoints）](#3-训练检查点-checkpoints)
   - [3.1 文件类型](#31-文件类型)
   - [3.2 内容格式](#32-内容格式)
   - [3.3 加载接口](#33-加载接口)
   - [3.4 断点续训](#34-断点续训)
4. [被试索引映射（subject_to_idx.json）](#4-被试索引映射-subject_to_idxjson)
5. [训练历史（Training History）](#5-训练历史-training-history)
6. [训练曲线图（可视化输出）](#6-训练曲线图可视化输出)
7. [配置快照（config.yaml）](#7-配置快照-configyaml)
8. [训练日志（training.log）](#8-训练日志-traininglog)
9. [CLI 接口](#9-cli-接口)
10. [配置文件关键参数速查](#10-配置文件关键参数速查)
11. [常见 Agent 使用模式](#11-常见-agent-使用模式)

---

## 1. 输出目录结构总览

训练一次后，`outputs/` 目录下产生如下结构：

```
outputs/
├── graph_cache/                        ← 图缓存（跨训练共享）
│   ├── sub-01_GRADON_a1b2c3d4.pt      ← (被试, 任务) 的完整 run 图
│   ├── sub-01_GRADOFF_e5f6g7h8.pt
│   └── sub-02_rest_00000000.pt
│
└── twinbrain_v5_20260227_123456/       ← 单次训练的输出目录
    │                                      (格式: {experiment_name}_{timestamp})
    ├── config.yaml                     ← 本次训练所用完整配置快照
    ├── training.log                    ← 全量训练日志
    ├── subject_to_idx.json             ← 被试 ID → 嵌入整数索引映射
    ├── best_model.pt                   ← 验证集损失最低的检查点
    ├── swa_model.pt                    ← SWA 权重（仅 use_swa: true 时生成）
    ├── checkpoint_epoch_10.pt          ← 定期保存点（每 save_frequency epoch）
    ├── checkpoint_epoch_20.pt
    ├── training_loss_curve.png         ← 训练/验证损失曲线图
    ├── training_r2_curve.png           ← 验证 R² 曲线图（含各模态）
    ├── checkpoints/                    ← （预留子目录，目前未使用）
    ├── logs/                           ← （预留子目录）
    └── results/                        ← （预留子目录）
```

> **重要**：`graph_cache/` 路径由 `data.cache.dir` 配置（默认 `outputs/graph_cache`，相对于 `main.py` 所在目录解析），与训练输出目录相互独立，可跨多次训练实验共用。

---

## 2. 图缓存（Graph Cache）

### 2.1 命名规则

**文件名格式**：

```
{subject_id}_{task}_{config_hash}.pt
```

| 字段 | 来源 | 示例 |
|------|------|------|
| `subject_id` | BIDS 目录名（`sub-XX`） | `sub-01` |
| `task` | BIDS 任务名；若无任务则固定为 `notask` | `GRADON`、`rest`、`notask` |
| `config_hash` | 图相关配置参数的 MD5 前 8 位十六进制 | `a1b2c3d4` |

**完整示例**：

```
sub-01_GRADON_a1b2c3d4.pt
sub-01_notask_ff12ab34.pt
sub-02_rest_00abcdef.pt
```

**哈希包含的配置参数**（任一改变 → 哈希变化 → 旧缓存自动失效）：

| 参数路径 | 说明 |
|---------|------|
| `graph.*` | 所有图构建参数（阈值、K 近邻、自环、有向性等） |
| `data.atlas.*` | Atlas 文件名和标签文件 |
| `training.max_seq_len` | 序列截断（仅 windowed_sampling 关闭时） |
| `data.modalities` | 模态列表（排序后） |
| `windowed_sampling.enabled` | 窗口模式开关 |
| `data.dti_structural_edges` | DTI 边开关 |
| `data.fmri_condition_bounds` | fMRI 条件时间段截取 |
| `graph.eeg_connectivity_method` | `correlation` 或 `coherence` |

**不进入哈希的参数**（改变后缓存仍有效）：

| 参数路径 | 原因 |
|---------|------|
| `graph.k_cross_modal` | 跨模态边在每次加载时动态重建，不存入缓存 |
| `output.*` | 输出配置不影响图数据 |
| `training.*`（除 max_seq_len） | 训练超参不影响图拓扑 |

---

### 2.2 内容格式

缓存文件是 PyG `HeteroData` 对象（`torch.save` 序列化），**仅包含节点特征和同模态边**：

#### 节点类型：`'eeg'`（EEG 电极节点）

| 属性 | 类型/形状 | 说明 |
|------|---------|------|
| `x` | `torch.Tensor [N_eeg, T_eeg, 1]` float32 | **主要时序数据**。z-score 标准化后的 EEG 信号幅度。每节点是一个电极，每时间步是一个采样点。最后维度 C=1（标量特征）。 |
| `num_nodes` | `int` | 电极数 N_eeg（通常 32–64） |
| `pos` | `torch.Tensor [N_eeg, 3]` float32，可选 | 电极 3D 坐标（MNE head 坐标系，单位 mm）。仅在原始数据提供位置信息时存在。 |
| `sampling_rate` | `float` | EEG 采样率（Hz），通常 250.0 |
| `labels` | `list[str]`，可选 | 电极通道名（如 `['Fp1', 'Fp2', ...]`） |

#### 节点类型：`'fmri'`（fMRI ROI 节点）

| 属性 | 类型/形状 | 说明 |
|------|---------|------|
| `x` | `torch.Tensor [N_fmri, T_fmri, 1]` float32 | **主要时序数据**。z-score 标准化后的 BOLD 信号。每节点是一个脑图谱 ROI，每时间步是一个 TR。最后维度 C=1。 |
| `num_nodes` | `int` | ROI 数 N_fmri（Schaefer200 通常 ~190，因 EPI 覆盖而异） |
| `pos` | `torch.Tensor [N_fmri, 3]` float32，可选 | ROI 质心 MNI 坐标（mm） |
| `sampling_rate` | `float` | fMRI 等效采样率（Hz） = 1/TR，通常约 0.5（TR≈2s） |

#### 边类型：`('eeg', 'connects', 'eeg')`（EEG 功能连通性边）

| 属性 | 类型/形状 | 说明 |
|------|---------|------|
| `edge_index` | `torch.Tensor [2, E_eeg]` int64 | 稀疏边索引，`edge_index[0]` = 源节点，`edge_index[1]` = 目标节点 |
| `edge_attr` | `torch.Tensor [E_eeg, 1]` float32 | 边权重 = 绝对 Pearson 相关系数 \|r\| 或 wideband coherence，取决于 `eeg_connectivity_method` |

#### 边类型：`('fmri', 'connects', 'fmri')`（fMRI 功能连通性边）

| 属性 | 类型/形状 | 说明 |
|------|---------|------|
| `edge_index` | `torch.Tensor [2, E_fmri]` int64 | 同上 |
| `edge_attr` | `torch.Tensor [E_fmri, 1]` float32 | 边权重 = 绝对 Pearson 相关系数 \|r\| |

#### 边类型：`('fmri', 'structural', 'fmri')`（DTI 结构连通性边，可选）

仅在 `data.dti_structural_edges: true` 且被试目录下存在 DTI 连通性矩阵文件时存在。

| 属性 | 类型/形状 | 说明 |
|------|---------|------|
| `edge_index` | `torch.Tensor [2, E_dti]` int64 | DTI 白质纤维束连通性边 |
| `edge_attr` | `torch.Tensor [E_dti, 1]` float32 | 纤维束强度（log1p 归一化 streamline count 或 FA 加权） |

> ⚠️ **不缓存**的边类型：`('eeg', 'projects_to', 'fmri')` — 跨模态边在每次加载时从节点特征动态重建（见 2.5 节）。

---

### 2.3 读取接口（推荐）

使用 `load_subject_graph_from_cache()` 获取 numpy 数组，**无需了解 PyG**：

```python
from utils.helpers import load_subject_graph_from_cache

data = load_subject_graph_from_cache(
    'outputs/graph_cache/sub-01_GRADON_a1b2c3d4.pt'
)

# ── 时序数据 ──────────────────────────────────────────────────
eeg  = data['eeg_timeseries']    # np.ndarray [N_eeg, T_eeg]  float32
fmri = data['fmri_timeseries']   # np.ndarray [N_fmri, T_fmri] float32

# ── 维度信息 ──────────────────────────────────────────────────
n_eeg  = data['eeg_n_channels']   # int, e.g. 63
n_fmri = data['fmri_n_rois']      # int, e.g. 190
fs_eeg = data['eeg_sampling_rate']  # float, e.g. 250.0
fs_fmri= data['fmri_sampling_rate'] # float, e.g. 0.5

# ── 空间坐标（如可用）────────────────────────────────────────
eeg_pos  = data['eeg_pos']   # np.ndarray [N_eeg, 3] or None
fmri_pos = data['fmri_pos']  # np.ndarray [N_fmri, 3] or None

# ── 图拓扑 ────────────────────────────────────────────────────
eeg_ei   = data['eeg_edge_index']  # np.ndarray [2, E_eeg]  int64  or None
eeg_ea   = data['eeg_edge_attr']   # np.ndarray [E_eeg]     float32 or None
fmri_ei  = data['fmri_edge_index'] # np.ndarray [2, E_fmri] int64  or None
fmri_ea  = data['fmri_edge_attr']  # np.ndarray [E_fmri]    float32 or None

# ── 元信息 ────────────────────────────────────────────────────
print(data['node_types'])   # list[str], e.g. ['eeg', 'fmri']
print(data['edge_types'])   # list[tuple], e.g. [('eeg','connects','eeg'), ...]
```

**返回字典键完整说明**：

| 键 | 类型 | 形状/值 | 说明 |
|----|------|---------|------|
| `eeg_timeseries` | `np.ndarray` \| `None` | `[N_eeg, T_eeg]` float32 | EEG 时序（z-scored） |
| `fmri_timeseries` | `np.ndarray` \| `None` | `[N_fmri, T_fmri]` float32 | fMRI BOLD 时序（z-scored） |
| `eeg_n_channels` | `int` \| `None` | e.g. 63 | EEG 电极数 |
| `fmri_n_rois` | `int` \| `None` | e.g. 190 | fMRI ROI 数 |
| `eeg_sampling_rate` | `float` \| `None` | e.g. 250.0 | EEG 采样率（Hz） |
| `fmri_sampling_rate` | `float` \| `None` | e.g. 0.5 | fMRI 等效采样率（Hz） |
| `eeg_pos` | `np.ndarray` \| `None` | `[N_eeg, 3]` float32 | 电极 3D 坐标（mm，head 坐标系） |
| `fmri_pos` | `np.ndarray` \| `None` | `[N_fmri, 3]` float32 | ROI 质心 MNI 坐标（mm） |
| `eeg_edge_index` | `np.ndarray` \| `None` | `[2, E_eeg]` int64 | EEG 连通性边（COO 格式） |
| `eeg_edge_attr` | `np.ndarray` \| `None` | `[E_eeg]` float32 | EEG 边权重（\|r\|，已去末维度） |
| `fmri_edge_index` | `np.ndarray` \| `None` | `[2, E_fmri]` int64 | fMRI 连通性边（COO 格式） |
| `fmri_edge_attr` | `np.ndarray` \| `None` | `[E_fmri]` float32 | fMRI 边权重（\|r\|，已去末维度） |
| `node_types` | `list[str]` | e.g. `['eeg', 'fmri']` | 缓存中存在的节点类型 |
| `edge_types` | `list[tuple]` | e.g. `[('eeg','connects','eeg'), ...]` | 缓存中存在的边类型 |

**异常处理**：

```python
try:
    data = load_subject_graph_from_cache(path)
except FileNotFoundError as e:
    # 缓存文件不存在：路径有误，或尚未运行训练管线
    print(e)
```

---

### 2.4 直接 PyG 读取

当需要操作原始 `HeteroData`（如重建跨模态边、传入 GNN）时：

```python
import torch
from torch_geometric.data import HeteroData

graph: HeteroData = torch.load(
    'outputs/graph_cache/sub-01_GRADON_a1b2c3d4.pt',
    map_location='cpu',
    weights_only=False,   # 必须 False：HeteroData 不是纯 tensor
)

# 访问 EEG 节点特征
eeg_x = graph['eeg'].x           # Tensor [N_eeg, T_eeg, 1]
eeg_x_np = eeg_x.squeeze(-1).numpy()  # → [N_eeg, T_eeg]

# 访问 fMRI 节点特征
fmri_x = graph['fmri'].x         # Tensor [N_fmri, T_fmri, 1]

# 访问 EEG 功能连通性
ei = graph['eeg', 'connects', 'eeg'].edge_index   # [2, E_eeg]
ea = graph['eeg', 'connects', 'eeg'].edge_attr    # [E_eeg, 1]

# 枚举所有内容
print(graph.node_types)   # ['eeg', 'fmri']
print(graph.edge_types)   # [('eeg','connects','eeg'), ('fmri','connects','fmri')]
```

---

### 2.5 跨模态边的特殊处理

跨模态边 `('eeg', 'projects_to', 'fmri')` **不持久化到缓存**，每次从节点特征动态重建：

```python
from models.graph_native_mapper import GraphNativeBrainMapper

# 初始化 Mapper（device 可用 'cpu'，仅用于重建边）
mapper = GraphNativeBrainMapper(device='cpu')

# 从已加载的缓存图重建跨模态边
cross_ei, cross_ea = mapper.create_simple_cross_modal_edges(
    graph,
    k_cross_modal=5,   # 每个 EEG 电极保留相关性最高的 k 个 fMRI ROI
)
# cross_ei: Tensor [2, N_eeg*k]  — EEG→fMRI 边索引（src=EEG，dst=fMRI）
# cross_ea: Tensor [N_eeg*k, 1]  — 边权重 = |Pearson r|（EEG-fMRI 时序相关性）

# 写回图
graph['eeg', 'projects_to', 'fmri'].edge_index = cross_ei
graph['eeg', 'projects_to', 'fmri'].edge_attr  = cross_ea
```

**设计原因**：跨模态边是节点特征的函数（仅需已缓存的 `x` 张量），存入缓存会导致修改 `k_cross_modal` 时旧缓存全部失效。动态重建代价 O(N_eeg × N_fmri × T)，CPU 上通常 < 100ms。

---

### 2.6 缓存失效条件

| 触发条件 | 行为 |
|---------|------|
| 修改 `graph.*` 中任何参数 | 哈希变化 → 旧文件名不匹配 → 重建 |
| 修改 `data.atlas.*` | 同上 |
| 修改 `graph.eeg_connectivity_method` | 同上 |
| 修改 `data.dti_structural_edges` | 同上 |
| 修改 `data.fmri_condition_bounds` | 同上 |
| 修改 `graph.k_cross_modal` | **不触发**（跨模态边动态重建） |
| 修改训练超参（lr、epoch 等） | **不触发** |
| 手动删除 `.pt` 文件 | 下次运行时重建 |

---

## 3. 训练检查点（Checkpoints）

### 3.1 文件类型

训练过程中产生三类检查点文件：

| 文件名 | 触发条件 | 说明 |
|--------|---------|------|
| `best_model.pt` | 每次验证损失刷新最低值时覆盖写入 | **最重要**：验证集最优权重，推理时优先使用 |
| `checkpoint_epoch_{N}.pt` | 每 `training.save_frequency` 个 epoch | 定期保存，供断点续训或对比分析 |
| `swa_model.pt` | 主训练结束后的 SWA 阶段（仅 `use_swa: true`） | SWA 平均权重，可能有更好的 OOD 泛化 |

### 3.2 内容格式

所有检查点均为 `dict`，通过 `torch.load(..., weights_only=False)` 读取：

```python
{
    'epoch': int,                        # 保存时的 epoch 编号（1-based）

    'model_state_dict': OrderedDict,     # torch.nn.Module.state_dict()
                                         # GraphNativeBrainModel 所有可学习参数

    'optimizer_state_dict': dict,        # AdamW optimizer 状态（动量、二阶矩等）

    'history': {                         # 训练历史（见第5节）
        'train_loss':    [float, ...],   # 每 epoch 平均训练损失
        'val_loss':      [float, ...],   # 每次验证的验证损失
        'val_r2_eeg':    [float, ...],   # 每次验证的 EEG R²（可选）
        'val_r2_fmri':   [float, ...],   # 每次验证的 fMRI R²（可选）
    },

    # 以下键为可选（取决于配置）：
    'loss_balancer_state': dict,         # 自适应损失平衡器状态（use_adaptive_loss: true）
    'scheduler_state_dict': dict,        # LR 调度器状态（use_scheduler: true）
}
```

> **注意**：`swa_model.pt` 仅包含 `model.state_dict()`（原始 dict 格式），不含 optimizer 等状态，用 `model.load_state_dict(torch.load('swa_model.pt'))` 直接加载。

### 3.3 加载接口

#### 方式一：通过 `GraphNativeTrainer`（完整状态恢复，推荐用于断点续训）

```python
from models.graph_native_system import GraphNativeBrainModel, GraphNativeTrainer

# 1. 重建 model 和 trainer（与原始训练时参数相同）
model   = GraphNativeBrainModel(...)
trainer = GraphNativeTrainer(model, node_types=['eeg', 'fmri'], ...)

# 2. 加载检查点（恢复 model + optimizer + scheduler + loss_balancer + history）
saved_epoch = trainer.load_checkpoint('outputs/.../best_model.pt')
# saved_epoch: int — 检查点保存时的 epoch

# 3. 从 saved_epoch + 1 继续训练
for epoch in range(saved_epoch + 1, total_epochs + 1):
    trainer.train_epoch(train_graphs, epoch=epoch)
```

#### 方式二：通过 `utils.helpers.load_checkpoint`（仅恢复 model 权重）

```python
from utils.helpers import load_checkpoint

epoch = load_checkpoint(
    checkpoint_path='outputs/.../best_model.pt',
    model=model,
    optimizer=None,   # 传入 optimizer 则同时恢复 optimizer 状态
    device='cpu',
)
```

#### 方式三：推理时纯权重加载

```python
import torch

ckpt = torch.load('outputs/.../best_model.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
```

#### 方式四：通过 CLI 断点续训（见 9. CLI 接口）

```bash
python main.py --resume outputs/twinbrain_v5_20260227_123456/best_model.pt
```

### 3.4 断点续训

CLI 的 `--resume` 参数自动处理以下逻辑：

1. 加载 `model_state_dict` + `optimizer_state_dict` + `scheduler_state_dict` + `loss_balancer_state`
2. 从 `checkpoint['epoch'] + 1` 继续训练循环（跳过已完成的 epoch）
3. 若检查点不含 `scheduler_state_dict`（旧版本保存的检查点），LR 从 epoch 0 重新开始（会打印 warning）
4. 加载失败时降级为从 epoch 1 重新开始训练（打印 warning）

---

## 4. 被试索引映射（subject_to_idx.json）

**路径**：`{output_dir}/subject_to_idx.json`

**用途**：将被试 ID（字符串）映射到 `nn.Embedding` 的整数索引，用于个性化数字孪生（被试特异性嵌入）。推理时必须使用训练时相同的映射文件，否则嵌入含义错乱。

**格式**（JSON）：

```json
{
    "sub-01": 0,
    "sub-02": 1,
    "sub-03": 2
}
```

**读取示例**：

```python
import json

with open('outputs/twinbrain_v5_xxx/subject_to_idx.json') as f:
    subject_to_idx = json.load(f)

# 推理时查询被试索引
subject_idx = subject_to_idx['sub-01']   # int, e.g. 0
subject_idx_tensor = torch.tensor(subject_idx, dtype=torch.long)

# 传入 model.forward()
graph.subject_idx = subject_idx_tensor
reconstructed, prediction = model(graph)
```

**不变性保证**：同一次训练中，`subject_to_idx` 由文件系统发现的被试 ID 排序后确定性推导（`sorted(all_subject_ids)`），与数据加载顺序无关。不同次训练中，若被试集合不变，映射相同。

---

## 5. 训练历史（Training History）

**访问方式**：`trainer.history`（内存对象）或检查点文件中的 `'history'` 键。

**结构**：

```python
history = {
    'train_loss':    [float, ...],    # 每 epoch 一个值（共 num_epochs 个）
    'val_loss':      [float, ...],    # 每次验证一个值（共 num_epochs/val_frequency 个）
    'val_r2_eeg':    [float, ...],    # 每次验证一个 EEG R² 值（仅当 eeg 存在时）
    'val_r2_fmri':   [float, ...],    # 每次验证一个 fMRI R² 值（仅当 fmri 存在时）
    # 若有更多模态，例如 'val_r2_dti': [...]
}
```

| 键 | 频率 | 说明 |
|----|------|------|
| `train_loss` | 每 epoch | 当前 epoch 全部训练样本的平均总损失 |
| `val_loss` | 每 `val_frequency` epoch | 验证集平均损失（重建 + 预测损失之和） |
| `val_r2_eeg` | 每 `val_frequency` epoch | EEG 重建的决定系数。≥0.3 为有效；<0 为失效 |
| `val_r2_fmri` | 每 `val_frequency` epoch | fMRI 重建的决定系数。≥0.3 为有效；<0 为失效 |

**R² 解读**：

| R² 范围 | 含义 | 评级 |
|---------|------|------|
| ≥ 0.3 | 重建质量良好，模型解释超过 30% 的信号方差 | ✅ 良好 |
| 0 ~ 0.3 | 有一定重建能力，但质量有限 | ⚠️ 有限 |
| < 0 | 重建效果比"恒预测均值"更差，模型失效 | ⛔ 不可信 |

**在 Agent 中使用示例**：

```python
import torch

ckpt = torch.load('outputs/.../best_model.pt', map_location='cpu', weights_only=False)
history = ckpt['history']

best_val_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
best_r2_eeg    = history['val_r2_eeg'][history['val_loss'].index(min(history['val_loss']))]
best_r2_fmri   = history['val_r2_fmri'][history['val_loss'].index(min(history['val_loss']))]

print(f"Best epoch: {best_val_epoch}")
print(f"Best val R²: EEG={best_r2_eeg:.3f}, fMRI={best_r2_fmri:.3f}")
```

---

## 6. 训练曲线图（可视化输出）

**路径**：

| 文件 | 说明 |
|------|------|
| `{output_dir}/training_loss_curve.png` | 训练/验证损失随 epoch 的折线图，标注最低验证损失点 |
| `{output_dir}/training_r2_curve.png` | 各模态验证 R² 随验证轮次的折线图，含 R²=0.3（良好）和 R²=0（基线）参考线 |

**程序化调用**（在 Agent 内生成/更新图）：

```python
from utils.visualization import plot_training_curves

plot_training_curves(
    history=trainer.history,
    output_dir='outputs/my_experiment',
    best_epoch=42,             # 可选：标注最佳 epoch
    best_r2_dict={             # 可选：在 x 轴标注最佳 R²
        'r2_eeg': 0.41,
        'r2_fmri': 0.28,
    },
)
# 生成: outputs/my_experiment/training_loss_curve.png
#       outputs/my_experiment/training_r2_curve.png
```

**依赖**：matplotlib。若未安装，函数静默跳过（不抛出异常）。

---

## 7. 配置快照（config.yaml）

**路径**：`{output_dir}/config.yaml`

每次训练开始时，完整配置（含所有已解析的默认值）被写入此文件，格式与 `configs/default.yaml` 相同（YAML）。

**用途**：
- 复现实验（将此文件直接作为 `--config` 传入）
- Agent 读取以了解本次训练的具体超参配置
- 与缓存哈希参数对应（可手动验证哈希值）

```python
import yaml

with open('outputs/twinbrain_v5_xxx/config.yaml') as f:
    config = yaml.safe_load(f)

graph_config     = config['graph']
training_config  = config['training']
output_dir       = config['output']['output_dir']
```

---

## 8. 训练日志（training.log）

**路径**：`{output_dir}/training.log`

**格式**：

```
2026-02-27 18:12:34 - twinbrain_v5 - INFO - 步骤 4/4: 训练模型
2026-02-27 18:12:35 - twinbrain_v5 - INFO - 早停设置: 每 5 epoch 验证一次 | 连续 20 次验证无改善触发早停 | 等效 100 epoch 的实际耐心值
2026-02-27 18:13:01 - twinbrain_v5 - INFO - Epoch [1/100] | train_loss=0.4521 | time=25.3s
2026-02-27 18:13:25 - twinbrain_v5 - INFO - [Epoch 5] val_loss=0.3812 | r2_eeg=0.12 | r2_fmri=0.08 ← 最佳模型
2026-02-27 18:18:45 - twinbrain_v5 - WARNING - ⛔ r2_fmri=-0.02 < 0: 模型重建效果差于均值基线预测
```

**关键日志模式**（Agent 可解析）：

| 模式 | 含义 |
|------|------|
| `Epoch [N/M]` | 当前/总 epoch，含训练损失和耗时 |
| `val_loss=X.XXXX` | 验证损失 |
| `r2_eeg=X.XX` / `r2_fmri=X.XX` | 各模态 R² |
| `← 最佳模型` 或 `★ 最佳` | 已保存 best_model.pt |
| `⛔ r2_{modality}<0` | 该模态重建失效警告 |
| `过拟合风险` | val/train 损失比率 > 3 |
| `📊 训练可信度摘要` | 训练结束后的综合评估摘要 |
| `🔄 断点续训` | 已从检查点加载，从 epoch N 继续 |

---

## 9. CLI 接口

**入口点**：`python main.py`

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config PATH` | `str` | `configs/default.yaml` | 配置文件路径 |
| `--seed INT` | `int` | `42` | 全局随机种子（影响数据划分和模型初始化） |
| `--resume PATH` | `str` | `None` | 从指定检查点恢复训练（传入检查点 `.pt` 文件路径） |

### 使用示例

```bash
# 基本训练
python main.py

# 指定配置文件
python main.py --config configs/my_experiment.yaml

# 断点续训（从 best_model.pt 恢复）
python main.py --resume outputs/twinbrain_v5_20260227_123456/best_model.pt

# 断点续训 + 自定义配置
python main.py --config configs/my_experiment.yaml \
               --resume outputs/twinbrain_v5_20260227_123456/best_model.pt \
               --seed 123
```

### `--resume` 行为说明

1. 加载完整训练状态（model + optimizer + scheduler + loss_balancer）
2. 从 `saved_epoch + 1` 继续训练
3. 检查点路径不存在 → 警告 + 从 epoch 1 重新开始
4. 加载失败 → 警告 + 从 epoch 1 重新开始

---

## 10. 配置文件关键参数速查

下表仅列出与 API 接口直接相关的参数，完整说明见 `configs/default.yaml`。

### 缓存相关

| 参数路径 | 类型 | 默认值 | 说明 |
|---------|------|--------|------|
| `data.cache.enabled` | `bool` | `true` | 是否启用图缓存 |
| `data.cache.dir` | `str` | `"outputs/graph_cache"` | 缓存目录（相对于 main.py） |
| `graph.k_cross_modal` | `int` | `5` | 跨模态边 top-k，修改不需清缓存 |
| `graph.eeg_connectivity_method` | `str` | `"correlation"` | `"correlation"` 或 `"coherence"`，修改会使缓存失效 |

### 检查点相关

| 参数路径 | 类型 | 默认值 | 说明 |
|---------|------|--------|------|
| `training.save_frequency` | `int` | `10` | 每 N epoch 保存一次定期检查点 |
| `training.val_frequency` | `int` | `5` | 每 N epoch 验证一次（决定 best_model 更新频率） |
| `training.early_stopping_patience` | `int` | `20` | 连续多少次验证无改善触发早停 |
| `training.use_swa` | `bool` | `false` | 是否运行 SWA 阶段（生成 swa_model.pt） |
| `training.swa_epochs` | `int` | `10` | SWA 额外训练轮数 |

### 输出目录相关

| 参数路径 | 类型 | 默认值 | 说明 |
|---------|------|--------|------|
| `output.output_dir` | `str` | `"outputs"` | 训练输出根目录 |
| `output.experiment_name` | `str` | `"twinbrain_v5"` | 实验名前缀（拼接时间戳构成子目录名） |
| `output.log_level` | `str` | `"INFO"` | 日志详细程度：`DEBUG`/`INFO`/`WARNING` |

---

## 11. 常见 Agent 使用模式

### 模式 A：加载缓存数据用于独立分析

```python
from utils.helpers import load_subject_graph_from_cache
import numpy as np

# 遍历所有缓存文件
from pathlib import Path
cache_dir = Path('outputs/graph_cache')
for cache_file in sorted(cache_dir.glob('*.pt')):
    # 解析文件名: sub-01_GRADON_a1b2c3d4.pt
    parts = cache_file.stem.split('_')
    subject_id = parts[0]
    task = parts[1]

    data = load_subject_graph_from_cache(cache_file)
    if data['eeg_timeseries'] is None or data['fmri_timeseries'] is None:
        continue   # 单模态图，跳过
    eeg  = data['eeg_timeseries']   # [N_eeg, T]
    fmri = data['fmri_timeseries']  # [N_fmri, T]
    print(f"{subject_id}/{task}: EEG={eeg.shape}, fMRI={fmri.shape}")
```

### 模式 B：读取训练结果评估模型质量

```python
import torch, json

# 加载最佳检查点
ckpt = torch.load('outputs/twinbrain_v5_xxx/best_model.pt',
                  map_location='cpu', weights_only=False)

best_val_loss = min(ckpt['history']['val_loss'])
best_idx      = ckpt['history']['val_loss'].index(best_val_loss)

result = {
    'best_epoch':   best_idx * VAL_FREQUENCY + 1,   # VAL_FREQUENCY from config
    'best_val_loss': best_val_loss,
    'best_r2_eeg':  ckpt['history'].get('val_r2_eeg',  [None])[best_idx],
    'best_r2_fmri': ckpt['history'].get('val_r2_fmri', [None])[best_idx],
}
print(result)
```

### 模式 C：推理 — 加载模型重建脑信号

```python
import torch, yaml
from models.graph_native_system import GraphNativeBrainModel
from utils.helpers import load_subject_graph_from_cache
from models.graph_native_mapper import GraphNativeBrainMapper

# 1. 加载配置和被试索引
with open('outputs/twinbrain_v5_xxx/config.yaml') as f:
    config = yaml.safe_load(f)
import json
with open('outputs/twinbrain_v5_xxx/subject_to_idx.json') as f:
    subject_to_idx = json.load(f)

# 2. 初始化并加载模型
model = GraphNativeBrainModel(...)  # 使用与训练相同的参数
ckpt  = torch.load('outputs/twinbrain_v5_xxx/best_model.pt',
                   map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# 3. 加载缓存图并重建跨模态边
graph = torch.load('outputs/graph_cache/sub-01_GRADON_a1b2c3d4.pt',
                   map_location='cpu', weights_only=False)
mapper = GraphNativeBrainMapper(device='cpu')
cross_ei, cross_ea = mapper.create_simple_cross_modal_edges(graph, k_cross_modal=5)
graph['eeg', 'projects_to', 'fmri'].edge_index = cross_ei
graph['eeg', 'projects_to', 'fmri'].edge_attr  = cross_ea

# 4. 设置被试嵌入索引
graph.subject_idx = torch.tensor(subject_to_idx['sub-01'], dtype=torch.long)

# 5. 推理
with torch.no_grad():
    reconstructed, prediction = model(graph)

recon_eeg  = reconstructed['eeg']   # Tensor [N_eeg, T, 1]
recon_fmri = reconstructed['fmri']  # Tensor [N_fmri, T, 1]
```

### 模式 D：断点续训自动化

```python
from pathlib import Path
import subprocess

output_dir = Path('outputs/twinbrain_v5_20260227_123456')
best_model = output_dir / 'best_model.pt'

if best_model.exists():
    # 从最佳检查点恢复训练
    subprocess.run([
        'python', 'main.py',
        '--config', str(output_dir / 'config.yaml'),
        '--resume', str(best_model),
    ], check=True)
else:
    # 首次训练
    subprocess.run(['python', 'main.py'], check=True)
```

---

*本文件由 TwinBrain V5.30 自动生成。如发现不一致之处，以源代码（`main.py`、`utils/helpers.py`、`models/graph_native_system.py`）为准。*
