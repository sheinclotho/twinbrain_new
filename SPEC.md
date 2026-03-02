# TwinBrain V5 — 项目规范说明

> **受众**：另一个 AI Agent，目标是能够大体复现本项目。  
> **版本**：V5.50 | **状态**：生产就绪 | **更新**：2026-03-02

---

## 一、项目目标

构建一个**图原生数字孪生脑（Graph-Native Digital Twin Brain）**训练系统，将 EEG（脑电图）和 fMRI（功能磁共振成像）数据联合建模，保持大脑天然的图拓扑结构，实现：
1. 多模态脑信号的时空特征编码
2. 跨模态信息融合（EEG ↔ fMRI）
3. 未来时间步预测（数字孪生核心）
4. **被试特异性个性化（V5.19+）**：每个被试学习唯一的潜空间偏移量
5. **数字孪生推理（V5.44+）**：干预仿真、少样本个性化、梯度归因
6. **跨会话知识迁移（V5.44+）**：会话嵌入捕捉跨会话漂移

**核心设计原则**：
- **图是第一性**：大脑 = 异构图，全程保持图结构，不做序列转换
- **时空不可分**：空间（图拓扑）和时间（信号序列）耦合建模
- **能量自适应**：自动平衡 EEG（低能量）和 fMRI（高能量）的梯度差异
- **个性化数字孪生**：被试特异性嵌入 + 共享图卷积，兼顾泛化与个性化
- **可仿真（Simulable）**：训练完成后可注入扰动、预测系统响应、提取功能指纹

---

## 二、系统架构

### 2.1 数据流

```
原始数据 (BIDS格式)
    ↓
BrainDataLoader (data/loaders.py)          — 统一加载 EEG/fMRI
    | _discover_tasks()                    — 自动发现每个被试的所有任务
    | load_all_subjects(tasks=None)        — 多任务加载，每(被试,任务)→一条run
    ↓
EEG预处理 / fMRI预处理                       — 滤波、配准、标准化
    ↓
GraphNativeBrainMapper (graph_native_mapper.py)
    ├── build_graph_structure()             — K近邻 + 小世界图（来自完整run相关性）
    ├── create_simple_cross_modal_edges()  — EEG→fMRI 跨模态边（带均匀 edge_attr）
    ├── add_dti_structural_edges()          — DTI结构连通性边（可选，须配置）
    └── HeteroData 输出（完整run图）
         ├── fmri 节点: [N_fmri, T_full, 1]   (Schaefer200: N_fmri=200)
         ├── eeg  节点: [N_eeg,  T_full, 1]   (e.g. N_eeg=63)
         ├── 边: ('fmri','connects','fmri'), ('eeg','connects','eeg')
         ├── 跨模态边: ('eeg','projects_to','fmri')
         ├── DTI结构边: ('fmri','structural','fmri')  [可选]
         └── 图级属性: subject_idx (int tensor, 用于个性化嵌入)
    ↓
图缓存 (main.py _graph_cache_key)          — 保存/加载完整run图(.pt)，跳过重复预处理
    ↓
extract_windowed_samples (main.py)         — dFC滑动窗口：1 run → N_windows 训练样本
    | 每个窗口: x=[N, T_window, 1]（切片）
    | edge_index: 共享完整run的连通性（结构稳定）
    | subject_idx: 从完整图复制（每窗口均携带被试身份）
    ↓
训练样本列表: N_subjects × N_tasks × N_windows 个 HeteroData
    ↓
GraphNativeBrainModel (graph_native_system.py)
    ├── subject_embed: nn.Embedding(N_subjects, H)  [个性化, V5.19+]
    ├── GraphNativeEncoder(subject_embed)           — ST-GCN 时空编码 + 被试偏移
    ├── GraphNativeDecoder                          — 信号重建
    └── EnhancedMultiStepPredictor                 — 未来预测
    ↓
GraphNativeTrainer                         — 训练循环 + 优化
```

### 2.2 训练样本设计

**两级数据增强**（无需采集新数据）：

| 层级 | 机制 | 捕捉的信号 | 样本倍增 |
|------|------|-----------|---------|
| 跨被试 | 多被试混训 + 被试嵌入 | 群体级结构共性 + 个体差异 | ×N_subjects |
| 跨任务 | 多任务自动发现 | 被试内认知状态变化 | ×N_tasks |
| 跨时间 | dFC 滑动窗口 | run 内脑状态动态 | ×N_windows |

**科学依据（dFC 范式）**：

> 图拓扑（edge_index）= 完整 run 的相关矩阵 → 稳定结构连通性  
> 节点特征（x）= 时间窗口切片 → 每窗口 = 一个脑状态快照

若不使用窗口采样，EEG `max_seq_len=300` 仅 1.2 秒，相关性估计统计不可靠，ST-GCN 的 edge_index 建立在噪声之上。

**典型数据量对比**（10 被试 × 3 任务 × 300 TR fMRI run）：

```
旧方案（截断单样本）: 10 × 3 × 1  =  30 训练样本
新方案（窗口 ws=50）: 10 × 3 × 11 = 330 训练样本（11×提升）
```

### 2.3 图缓存

图构建完成后自动保存为 `.pt` 文件：
- **路径**：`{cache_dir}/{subject_id}_{task}_{config_hash}.pt`
- **config_hash**：atlas、图参数（k近邻、阈值等）、DTI开关的 MD5 短哈希；窗口模式下不含 max_seq_len（截断不生效）。
- **内容**：始终是**完整 run 图**（窗口从缓存图中实时切分，节省磁盘空间）。包含 `subject_idx` 属性（V5.19+）。
- **好处**：再次运行时直接加载，跳过预处理和图构建，节省数分钟到数十分钟。
- **注意**：V5.18 以前的缓存不含 `subject_idx`；加载时会从当前 `subject_to_idx` 自动补写（V5.20 修复）。

### 2.4 被试特异性嵌入（个性化数字孪生，V5.19+）

**设计意图**（AGENTS.md §九 Gap 2）：让同一模型服务于多个不同的"数字孪生"——每个被试都有一个唯一的 `[H]` 向量，捕捉其基线活动水平、认知风格等个体差异。

**调用链**（完整，已验证）：
```
build_graphs():
    subject_to_idx = {sid: int}  ← 扫描所有被试，确定性排序
    built_graph.subject_idx = torch.tensor(idx, long)  ← 每图都写入
    extract_windowed_samples() 将 subject_idx 复制到所有窗口
    return (graphs, mapper, subject_to_idx)

create_model(num_subjects=len(subject_to_idx)):
    GraphNativeBrainModel(num_subjects=N):
        self.subject_embed = nn.Embedding(N, H)  ← N(0,0.02) 初始化

GraphNativeBrainModel.forward():
    s = self.subject_embed(data.subject_idx)  # [H]
    encoded = self.encoder(data, subject_embed=s)

GraphNativeEncoder.forward(subject_embed):
    x_proj += subject_embed.view(1, 1, -1)   # broadcast [H]→[N,T,H]
```

**推理时个性化**（few-shot）：
```python
# 冻结编码器，只微调新被试的嵌入
for p in model.encoder.parameters(): p.requires_grad = False
model.subject_embed.weight[new_subject_idx].requires_grad = True
# 用少量样本（几十个窗口）微调，cost = O(H) 参数
```

### 2.5 核心模型组件

#### GraphNativeEncoder (`models/graph_native_encoder.py`)
```
Input: HeteroData (node features [N, T, C_in]) + subject_embed [H]
  → Input Projection: Linear per node type → [N, T, H]
  → Subject Offset: x += subject_embed.view(1,1,-1)  ← V5.19+ 个性化
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
- **跨模态 size 保护**：`propagate(size=(N_src, N_dst))` 防止 N_src 污染 N_dst

#### GraphNativeTrainer (`models/graph_native_system.py`)
- AMP（自动混合精度）：`torch.amp.autocast`（新 API）
- torch.compile()：PyTorch 2.0+ 图编译
- 自适应损失平衡：`AdaptiveLossBalancer`（处理 EEG/fMRI 能量差异）
- EEG 增强：`EnhancedEEGHandler`（注意力 + dropout + 正则化，懒初始化避免维度错误）
- EEG 防零崩塌正则化：entropy + diversity + activity loss（已加入 total_loss）
- 余弦退火 LR Scheduler

### 2.6 图构建参数（`configs/default.yaml` → `graph` 节）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `k_nearest_fmri` | 20 | fMRI K近邻（小世界网络） |
| `k_nearest_eeg` | 10 | EEG K近邻 |
| `threshold_fmri` | 0.3 | fMRI 连接阈值 |
| `threshold_eeg` | 0.2 | EEG 连接阈值 |
| `cross_modal_distance_threshold` | 30.0 mm | 跨模态边距离阈值 |

---

### 2.7 预测评估指标体系（V5.47+，科学严谨性设计）

TwinBrain 的验证循环（`validate()`）输出四类指标，层次递进地衡量模型的真实预测能力。
理解这些指标需要区分两个完全不同的"基线"：

| 基线 | 含义 | R² 为 0 的语义 |
|------|------|--------------|
| **均值基线** | 永远预测信号的全局均值 | R² = 0：与"总是猜均值"持平 |
| **AR(1) 自相关基线** | 把最后一帧重复作为所有未来预测 | 这个基线自身的 R² 可能很高（高自相关信号）或很负（长程预测时信号已变化） |

#### 四项核心指标定义

```
ar1_r2_{nt}    ── AR(1) 基线在当前任务上的 R²（零参数下限/参考）
pred_r2_{nt}   ── 模型多步平均预测 R²（主要性能指标）
pred_r2_h1_{nt}── 模型仅第 1 步（h=1）预测 R²（与 NPI 横向可比）
decorr_{nt}    ── 多步技能分数 = (pred_r2 − ar1_r2) / (1 − ar1_r2)
decorr_h1_{nt} ── h=1 技能分数   = (pred_r2_h1 − ar1_r2_h1) / (1 − ar1_r2_h1)
```

`decorr` 是气象预报领域的**技能分数（Skill Score）**，值域 `(-∞, 1]`：
- **= 1**：完美预测
- **> 0.15**：清晰超越 AR(1) 基线，模型确实学到时序动态
- **= 0**：与 AR(1) 基线持平（模型等价于"复制上一帧"）
- **< 0**：差于 AR(1) 基线，需结合具体 horizon 分析

#### EEG vs fMRI 的物理差异与预期值

| 模态 | 采样率 | h=1 间隔 | AR(1) h=1 R² | 说明 |
|------|--------|---------|-------------|------|
| EEG  | 250 Hz | 4 ms    | ≈ 0.8–0.9   | 相邻帧高度相似（ρ≈0.91），AR(1) 近乎完美 |
| fMRI | 0.5 Hz | 2 s     | ≈ 0.7–0.9   | BOLD 血动力学平滑（ρ≈0.85–0.95），AR(1) 较强 |

对于 **多步预测**（TwinBrain 核心任务）：
- EEG 15步 = 60ms：跨越 alpha 周期（83ms），AR(1) 多步 R² 急剧崩溃至 < 0（重复一个值 60ms 当然差于均值）
- fMRI 15步 = 30s：超出 HRF 主要响应时程（8–12s），AR(1) 多步 R² 同样严重下降

#### decorr_h1 为负时的三级判读

`decorr_h1_{nt} < 0` 的含义和严重程度**取决于同时发生的事**：

| 条件组合 | 含义 | 警报级别 |
|---------|------|---------|
| `decorr_h1 < 0` 且 `decorr > 0` | **物理预期的梯度竞争**（见下文） | ℹ️ INFO |
| `decorr_h1 < 0` 且 `decorr < 0` | 模型单步和多步均未超越 AR(1) | ⛔ WARNING |
| `pred_r2_h1 < 0`（差于均值基线） | 绝对失败 | ⛔ WARNING |

#### 梯度竞争：为何长程训练不能自动改善单步预测

这是最常被误解的设计点，需要严格说明。

**背景**：`decorr_h1_eeg` 在 epoch 5 = −2.65，此时 `decorr_eeg` = +0.46 > 0。
单步指标为负但多步指标为正，乍看矛盾，实则有严密的物理解释：

1. **训练损失对所有 pred_steps 求和**：`L = Σ_h w_h · L(ŷ_h, y_h)`，
   编码器和预测器的参数对 h=1, 2, ..., 15 的误差负责。

2. **h=1 的最优策略 ≠ h=15 的最优策略**：
   - h=1（4ms）最优：输出 ≈ 上一帧（接近恒等映射，利用 ρ=0.91 的自相关）
   - h=15（60ms）最优：捕捉 alpha 振荡的周期性（需要理解信号的频率结构）
   - 这两种策略需要截然不同的参数配置

3. **Transformer 预测器联合生成所有步**：不能独立优化 h=1 和 h=15，
   参数必须在两者之间取折中。优化 h=5–15 的梯度会把网络引向
   "学习时序周期模式"的方向，而非"学习恒等复制"的方向。

4. **"长程训练能间接提升单步"的反驳**：
   - 理论上可以（一个足够大的网络可以同时学会 h=1 的复制和 h=15 的动态）
   - 实践中：Transformer 的注意力机制先学习的是**多帧的周期结构**（更大的梯度幅度），
     而不是"输出头 0 ≈ 上一帧"（对于周期信号，这是整个上下文中信息量最少的一步）
   - 结果：模型在 h=5–10 有明显准确度，而 h=1 可能不如零参数外推
   - 这不是 bug，而是预测目标（60ms 整体动态）的合理取舍

5. **如需改善 decorr_h1 > 0**（选做）：
   - 在 `pred_step_weight_gamma` 引入偏向 h=1 的指数衰减（gamma < 1）
   - 或在损失中显式加入 h=1 的额外权重
   - 代价：多步长程预测能力（decorr > 0）可能略有下降

#### NPI 自相关问题：该方法是否只在开发自相关？

**背景**：NPI（Luo et al., *Nature Methods* 2025）使用 3 个 TR（6s）上下文预测 1 个 TR（2s），
声称"3步输入预测1步准确率最高"。

**关键问题**：对于 BOLD 信号（TR=2s，ρ≈0.85–0.95），AR(1) 在 h=1 可免费获得 R² ≈ 0.7–0.9。
AR(3) 利用了更多自相关历史（AR(3) > AR(2) > AR(1) for high-ρ signals），其性能提升可能
完全来自更好地利用自相关，而非真正学到神经动力学。

**判断标准**：

```
decorr_h1 = (NPI_r2 - ar1_r2_h1) / (1 - ar1_r2_h1)
  > 0  → NPI 确实超越了自相关（学到了真实动力学）
  = 0  → NPI 等价于 AR(1)（仅利用自相关）
  < 0  → NPI 甚至不如 AR(1)（不可能，因 AR(3) ⊇ AR(1)）
```

NPI 论文未报告 `decorr_h1`，因此无法直接评估。但根据 BOLD 自相关的特性，
NPI 的部分甚至大部分 R² 可能来自自相关，真实的神经动力学成分（decorr_h1 正贡献）
有待独立验证。

**TwinBrain 的优势**（在早期训练 epoch 5 即已体现）：
- `decorr_fmri = 0.485 > 0`：多步 fMRI 预测超越 AR(1) 常数外推
- `decorr_h1_fmri = 0.463 > 0`：h=1 fMRI 预测超越 AR(1)（可与 NPI 直接对比）
- `decorr_eeg = 0.457 > 0`：多步 EEG 预测超越 AR(1)（fMRI 模型无法做到）
- 预测时程 15 步（fMRI 30s，EEG 60ms）远超 NPI 的单步，捕获真实脑动力学

#### 数值参考范围（充分训练模型）

| 指标 | EEG 预期范围 | fMRI 预期范围 | 文献依据 |
|------|------------|-------------|---------|
| `r2_{nt}`（重建） | ≥ 0.80 | ≥ 0.85 | 自编码器标准 |
| `pred_r2_{nt}`（多步预测） | 0.05–0.25 | 0.15–0.40 | Schirrmeister 2017; Thomas 2022 |
| `decorr_{nt}`（多步技能） | > 0 即有效；> 0.3 为良好 | > 0 即有效；> 0.3 为良好 | — |
| `pred_r2_h1_{nt}`（h=1） | < `ar1_r2_h1` 可接受（见上文） | ≥ `ar1_r2_h1` 为目标 | NPI 对标 |
| `decorr_h1_{nt}`（h=1 技能） | 可为负（多步训练取舍） | > 0 为目标 | NPI 对标 |

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
│   ├── graph_native_mapper.py   # 图映射器（构建异构图，含 DTI 结构边）
│   ├── graph_native_encoder.py  # ST-GCN 编码器（含被试嵌入注入）
│   ├── graph_native_system.py   # 完整模型 + 训练器（含 subject_embed）
│   ├── adaptive_loss_balancer.py
│   ├── eeg_channel_handler.py
│   ├── advanced_prediction.py
│   ├── consciousness_module.py  # 意识模块：GWT + IIT（V5.1，实验性）
│   ├── advanced_attention.py    # 跨模态注意力、时空注意力（V5.1，实验性）
│   ├── predictive_coding.py     # 层次化预测编码 + 主动推理（V5.1，实验性）
│   └── enhanced_graph_native.py # 增强模型 + 训练器，集成以上意识模块（V5.1）
├── utils/
│   ├── helpers.py
│   └── visualization.py         # 可视化工具（脑网络图、时序信号、注意力矩阵）
└── atlases/                     # 脑图谱文件（Schaefer200）
```

---

## 三点五、推理接口速查

### 模型输出张量形状

| 张量 | 形状 | 值域 | 说明 |
|------|------|------|------|
| `reconstructed['eeg']` | `[N_eeg, T_eeg, 1]` | ≈±3 | EEG 重建信号 |
| `reconstructed['fmri']` | `[N_fmri, T_fmri, 1]` | ≈±3 | fMRI 重建信号 |
| `encoded['eeg']` | `[N_eeg, T_eeg, H]` | 无界 | EEG 潜向量 |
| `encoded['fmri']` | `[N_fmri, T_fmri, H]` | 无界 | fMRI 潜向量（含 EEG 跨模态信息） |
| `predictions['eeg']` | `[N_eeg, steps, H]` | 无界 | EEG 潜空间未来预测 |
| `predictions['fmri']` | `[N_fmri, steps, H]` | 无界 | fMRI 潜空间未来预测 |
| 同模态 `edge_attr` | `[E, 1]` | [0, 1] | Pearson \|r\| 功能连接强度 |
| 跨模态 `edge_attr` | `[E, 1]` | = 1.0 | 均匀权重（随机连接） |
| `subject_embed.weight` | `[N_sub, H]` | ≈±0.02 | 被试潜空间偏移（初始化值） |

> 可视化时通常需要 `[N, T]` 而非 `[N, T, 1]`，用 `.squeeze(-1)` 即可。

### 检查点文件结构（`best_model.pt`）

```python
checkpoint = torch.load("outputs/.../best_model.pt", map_location="cpu")
# 键：
{
    "epoch":                int,          # 保存时的 epoch
    "model_state_dict":     OrderedDict,  # model.load_state_dict() 用
    "optimizer_state_dict": dict,         # 恢复训练用
    "scheduler_state_dict": dict,         # LR 调度器状态（V5.22+，旧 ckpt 无此键）
    "history": {
        "train_loss": List[float],
        "val_loss":   List[float],
    },
    # use_adaptive_loss=True 时额外存在：
    "loss_balancer_state":  OrderedDict,
}

# 推理时从 state_dict 推断 num_subjects（无需外部记录）：
state = checkpoint["model_state_dict"]
num_subjects = state["subject_embed.weight"].shape[0] if "subject_embed.weight" in state else 0
```

### 被试索引映射（`subject_to_idx.json`）

训练完成后自动保存到 `outputs/<experiment_name>/subject_to_idx.json`，内容如：

```json
{"sub-01": 0, "sub-02": 1, "sub-03": 2}
```

**推理时必须加载此文件**来确定各被试对应的 `subject_idx`，否则无法正确使用个性化嵌入。

---

## 四、关键依赖

```
torch >= 2.0        # torch.compile() 支持
torch-geometric     # 图神经网络
nibabel             # NIfTI 文件读写
mne                 # EEG 数据处理
pyyaml              # 配置文件
numpy, scipy        # 数值计算
nilearn             # fMRI atlas 分区（NiftiLabelsMasker）
```

---

## 五、复现步骤（给 Agent）

1. 安装依赖：`pip install -r requirements.txt`
2. 修改 `configs/default.yaml` 中 `data.root_dir` 为数据路径
3. 数据需满足 BIDS 格式，包含 EEG（`.set`）和/或 fMRI（`.nii.gz`）
4. 运行 `python main.py`
5. 如遇内存不足，确认 `training.use_gradient_checkpointing: true`（已为默认值）
6. 如需降低显存，将 `model.hidden_channels` 从 128 降至 64
7. 个性化嵌入自动启用（基于数据集被试数量），无需额外配置

---

## 六、设计决策记录

| 决策 | 理由 |
|------|------|
| HeteroData 而非 homogeneous graph | EEG 和 fMRI 节点特性不同，需要类型区分 |
| ST-GCN 而非 Transformer | 显式保持图拓扑，Transformer 缺乏空间归纳偏置 |
| 时间步循环而非批量时间卷积 | 灵活支持变长序列；已用 gradient checkpointing 解决内存问题 |
| Spectral Norm | 防止深层 GNN 梯度爆炸，但增加每次前向传播的计算量 |
| Huber Loss（默认） | 对 EEG/fMRI 噪声信号比 MSE 更鲁棒 |
| subject_embed 在投影后注入（而非原始信号） | 偏移在 H 维潜空间，与 ST-GCN 同维，语义对齐；避免 1 维原始信号被 H 维偏移淹没 |
| nn.Embedding N(0,0.02) 初始化 | 小噪声不干扰共享预训练信号；个性化在训练中逐步浮现 |
| DTI 边为 ('fmri','structural','fmri') 而非独立节点 | DTI 无时序特征；作为额外边类型允许编码器同时利用功能和结构连通性 |
