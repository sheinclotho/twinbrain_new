# TwinBrain V5 — 项目规范说明

> **受众**：另一个 AI Agent，目标是能够大体复现本项目。  
> **版本**：V5.7 | **状态**：生产就绪 | **更新**：2026-02-23

---

## 一、项目目标

构建一个**图原生数字孪生脑（Graph-Native Digital Twin Brain）**训练系统，将 EEG（脑电图）和 fMRI（功能磁共振成像）数据联合建模，保持大脑天然的图拓扑结构，实现：
1. 多模态脑信号的时空特征编码
2. 跨模态信息融合（EEG ↔ fMRI）
3. 未来时间步预测（数字孪生核心）

**核心设计原则**：
- **图是第一性**：大脑 = 异构图，全程保持图结构，不做序列转换
- **时空不可分**：空间（图拓扑）和时间（信号序列）耦合建模
- **能量自适应**：自动平衡 EEG（低能量）和 fMRI（高能量）的梯度差异

---

## 二、系统架构

### 2.1 数据流

```
原始数据 (BIDS格式)
    ↓
BrainDataLoader (data/loaders.py)          — 统一加载 EEG/fMRI
    | _discover_tasks()                    — 自动发现每个被试的所有任务
    | load_all_subjects(tasks=None)        — 多任务加载，每(被试,任务)→一个样本
    ↓
EEG预处理 / fMRI预处理                       — 滤波、配准、标准化
    ↓
GraphNativeBrainMapper (graph_native_mapper.py)
    ├── build_graph_structure()             — K近邻 + 小世界图
    ├── create_cross_modal_edges()          — 跨模态边（距离阈值）
    └── HeteroData 输出
         ├── fmri 节点: [N_fmri, T, 1]
         ├── eeg  节点: [N_eeg,  T, 1]
         └── 边: fmri↔fmri, eeg↔eeg, fmri↔eeg
    ↓
图缓存 (main.py _graph_cache_key)          — 保存/加载 .pt 文件，跳过重复预处理
    ↓
GraphNativeBrainModel (graph_native_system.py)
    ├── GraphNativeEncoder                  — ST-GCN 时空编码
    ├── GraphNativeDecoder                  — 信号重建
    └── EnhancedMultiStepPredictor          — 未来预测
    ↓
GraphNativeTrainer                         — 训练循环 + 优化
```

### 2.2 训练样本设计

每个 **(被试, 任务)** 组合 → 一个图样本 → 加入训练列表。

- 多被试混训：捕捉跨被试的群体级脑结构共性（population-level patterns）。
- 多任务（静息 + 工作记忆等）：捕捉被试内跨认知状态的脑动态变化。
- 样本 = 一张 `HeteroData` 异质图，包含该被试在该任务下的完整 EEG + fMRI 节点特征。

### 2.3 图缓存

图构建完成后自动保存为 `.pt` 文件：
- **路径**：`{cache_dir}/{subject_id}_{task}_{config_hash}.pt`
- **config_hash**：atlas、图参数（k近邻、阈值等）、max_seq_len 的 MD5 短哈希；参数变更时自动失效。
- **好处**：再次运行时直接加载，跳过预处理和图构建，节省数分钟到数十分钟。

### 2.4 核心模型组件

#### GraphNativeEncoder (`models/graph_native_encoder.py`)
```
Input: HeteroData (node features [N, T, C_in])
  → Input Projection: Linear per node type → [N, T, H]
  → N × SpatialTemporalGraphConv (ST-GCN):
      ├── Temporal Conv1d: [N, C, T] → [N, H, T] → [N, T, H]
      ├── Spatial MessagePassing per timestep (with grad checkpoint)
      └── Residual + LayerNorm
  → TemporalAttention (Flash Attention):
      ├── Multi-head self-attention over T
      └── Residual
Output: HeteroData (node features [N, T, H])
```

#### SpatialTemporalGraphConv
- 核心创新：将时间卷积（Conv1d）+ 空间消息传递（MessagePassing）合并为一个操作
- 注意力机制：`alpha = att_src(x_j) + att_dst(x_i)` → softmax → 加权消息
- 谱归一化（Spectral Norm）：应用于所有线性层以稳定训练
- **梯度检查点**（Gradient Checkpointing）：在时间步循环内逐步释放中间激活，解决长序列 MemoryError

#### GraphNativeTrainer (`models/graph_native_system.py`)
- AMP（自动混合精度）：`torch.cuda.amp.autocast`
- torch.compile()：PyTorch 2.0+ 图编译
- 自适应损失平衡：`AdaptiveLossBalancer`（处理 EEG/fMRI 能量差异）
- EEG 增强：`EEGChannelHandler`（注意力 + dropout + 正则化）
- 余弦退火 LR Scheduler

### 2.3 图构建参数（`configs/default.yaml` → `graph` 节）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `k_nearest_fmri` | 20 | fMRI K近邻（小世界网络） |
| `k_nearest_eeg` | 10 | EEG K近邻 |
| `threshold_fmri` | 0.3 | fMRI 连接阈值 |
| `threshold_eeg` | 0.2 | EEG 连接阈值 |
| `cross_modal_distance_threshold` | 30.0 mm | 跨模态边距离阈值 |

---

## 三、项目结构

```
twinbrain_new/
├── AGENTS.md                    # AI Agent 指令 + 错误记录（必读）
├── SPEC.md                      # 本文件：项目规范
├── USERGUIDE.md                 # 用户使用说明
├── CHANGELOG.md                 # 更新日志
├── main.py                      # 主程序入口
├── requirements.txt             # 依赖列表
├── configs/
│   └── default.yaml             # 全局配置（数据路径在此修改）
├── data/
│   ├── loaders.py               # 统一数据加载器
│   ├── eeg_preprocessor.py      # EEG 预处理
│   └── fmri_preprocessor.py     # fMRI 预处理
├── models/
│   ├── graph_native_mapper.py   # 图映射器（构建异构图）
│   ├── graph_native_encoder.py  # ST-GCN 编码器
│   ├── graph_native_system.py   # 完整模型 + 训练器
│   ├── adaptive_loss_balancer.py
│   ├── eeg_channel_handler.py
│   └── advanced_prediction.py
├── utils/
│   └── helpers.py
└── atlases/                     # 脑图谱文件（Schaefer200）
```

---

## 四、关键依赖

```
torch >= 2.0        # torch.compile() 支持
torch-geometric     # 图神经网络
nibabel             # NIfTI 文件读写
mne                 # EEG 数据处理
pyyaml              # 配置文件
numpy, scipy        # 数值计算
```

---

## 五、复现步骤（给 Agent）

1. 安装依赖：`pip install -r requirements.txt`
2. 修改 `configs/default.yaml` 中 `data.root_dir` 为数据路径
3. 数据需满足 BIDS 格式，包含 EEG（`.set`）和/或 fMRI（`.nii.gz`）
4. 运行 `python main.py`
5. 如遇内存不足，确认 `training.use_gradient_checkpointing: true`（已为默认值）
6. 如需降低显存，将 `model.hidden_channels` 从 128 降至 64

---

## 六、设计决策记录

| 决策 | 理由 |
|------|------|
| HeteroData 而非 homogeneous graph | EEG 和 fMRI 节点特性不同，需要类型区分 |
| ST-GCN 而非 Transformer | 显式保持图拓扑，Transformer 缺乏空间归纳偏置 |
| 时间步循环而非批量时间卷积 | 灵活支持变长序列；已用 gradient checkpointing 解决内存问题 |
| Spectral Norm | 防止深层 GNN 梯度爆炸，但增加每次前向传播的计算量 |
| Huber Loss（默认） | 对 EEG/fMRI 噪声信号比 MSE 更鲁棒 |
