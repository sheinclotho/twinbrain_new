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

`GraphNativeTrainer.validate()`（`models/graph_native_system.py`）输出九项指标，覆盖重建质量、预测能力和与 AR(1) 基线的相对技能。本节给出每项指标的**精确计算公式**及简要说明，所有公式均与代码实现一一对应。

两个不同的基线：

| 基线 | 定义 | 特点 |
|------|------|------|
| **均值基线**（R²=0 参考点） | 永远预测信号的全局均值 | R²=0 表示模型等价于"总是猜均值" |
| **AR(1) 自相关基线** | 把上下文最后一帧复制到所有未来步 | 对高自相关信号（如 BOLD）h=1 极强（免费 R²≈0.7–0.9），对多步预测通常为负 |

---

#### 2.7.0 前置：上下文切分与最小序列约束

**代码位置**：`GraphNativeBrainModel._PRED_CONTEXT_RATIO = 2/3`，`_PRED_MIN_SEQ_LEN = 4`

设某模态的时序长度（latent 时间步数）为 $T$。验证时执行以下切分：

$$
T_{\text{ctx}} = \left\lfloor T \times \frac{2}{3} \right\rfloor, \qquad T_{\text{fut}} = T - T_{\text{ctx}}
$$

仅当 $T \geq 4$（`_PRED_MIN_SEQ_LEN`）时才参与预测评估，否则跳过该模态。

**有效预测步数**（避免浪费算力预测超出实际信号长度的未来）：

$$
S = \min(\text{prediction\_steps},\; \max(1, T_{\text{fut}}))
$$

---

#### 2.7.1 全局均值单遍算法（R² 计算基础）

**代码位置**：`GraphNativeTrainer._r2_from_accum(ss_res, ss_raw, ss_sum, ss_cnt)`

所有 R² 指标共用同一套单遍在线累加器（避免"逐样本均值"导致的乐观偏差）。对全量样本用代数恒等式 $SS_{\text{tot}} = \sum y_i^2 - n\bar{y}^2$ 代替两次扫描：

$$
\bar{y} = \frac{\sum y_i}{n}, \qquad SS_{\text{tot}} = \sum y_i^2 - n\,\bar{y}^2
$$

$$
\boxed{R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}, \qquad
SS_{\text{res}} = \sum(y_i - \hat{y}_i)^2}
$$

当 $SS_{\text{tot}} \leq 10^{-12}$（信号为常数）时返回 $R^2 = 0$。取值域 $(-\infty, 1]$；$R^2=1$ 完美，$R^2=0$ 等于均值预测，$R^2<0$ 差于均值。

---

#### 2.7.2 重建 R²（`r2_{nt}`）

**代码位置**：`validate()` → 重建 R² 累加器，键名 `r2_{node_type}`

衡量编码器–解码器的信号重建质量。对验证集每个样本，将重建信号 $\hat{\mathbf{X}}$ 与原始信号 $\mathbf{X}$ 对齐到 $T_{\min} = \min(T, T')$ 后，累加残差：

$$
SS_{\text{res}}^{\text{recon}} \mathrel{+}= \sum_{n,t,c} \bigl(X_{n,t,c} - \hat{X}_{n,t,c}\bigr)^2, \qquad
\text{r2}_{nt} = 1 - \frac{SS_{\text{res}}^{\text{recon}}}{SS_{\text{tot}}^{\text{recon}}}
$$

---

#### 2.7.3 信号空间预测 R²（`pred_r2_{nt}`）★ 主要指标

**代码位置**：`validate()` → 信号空间预测 R² 累加器，键名 `pred_r2_{node_type}`

最重要的数字孪生能力指标：在完全不知道未来的前提下，预测未来信号的准确程度。**因果保证**：验证时对前 $T_{\text{ctx}}$ 步重新独立编码（新的 forward pass，无梯度），避免双向注意力对未来信息的访问。随后调用 `predictor.predict_next(h_ctx, num_steps=S)` 自回归生成预测，经 `GraphPredictionPropagator` 图传播后解码，与真实未来段对比：

$$
SS_{\text{res}}^{\text{pred}} \mathrel{+}= \sum_{n,h,c} \bigl(Y_{n,h,c} - \hat{Y}_{n,h,c}\bigr)^2, \quad h = 1, \ldots, S
$$

$$
\text{pred\_r2}_{nt} = 1 - \frac{SS_{\text{res}}^{\text{pred}}}{SS_{\text{tot}}^{\text{pred}}}
$$

---

#### 2.7.4 AR(1) 多步基线 R²（`ar1_r2_{nt}`）

**代码位置**：`validate()` → `ar1_ss_res` 累加器，键名 `ar1_r2_{node_type}`

零参数基线：对所有 $h = 1, \ldots, S$ 步一律预测上下文最后一帧（常数外推），与 `pred_r2` 共享 $SS_{\text{tot}}^{\text{pred}}$ 统计量：

$$
\hat{Y}_{n,h,c}^{\text{AR(1)}} = X_{n,\, T_{\text{ctx}}-1,\, c} \quad \forall h, \qquad
\text{ar1\_r2}_{nt} = 1 - \frac{\sum_{n,h,c}(Y_{n,h,c} - X_{n,T_{\text{ctx}}-1,c})^2}{SS_{\text{tot}}^{\text{pred}}}
$$

`ar1_r2` 对多步预测通常为负：对多步预测（如 15 步），AR(1) 基线（重复最后一帧）的累积误差通常超过均值基线，导致 `ar1_r2 < 0`，这是信号时序变化的预期结果，不是 bug。

---

#### 2.7.5 多步去相关技能分数（`decorr_{nt}`）

**代码位置**：`validate()` 末尾，`decorr = (pred_r2_val - ar1_r2) / max(1e-3, 1.0 - ar1_r2)`，键名 `decorr_{node_type}`

气象预报领域的**技能分数（Skill Score）**，衡量模型相对于 AR(1) 基线的额外预测价值：

$$
\boxed{\text{decorr}_{nt} = \frac{\text{pred\_r2}_{nt} - \text{ar1\_r2}_{nt}}{\max(10^{-3},\; 1 - \text{ar1\_r2}_{nt})}}
$$

值域 $(-\infty, 1]$，代码用 $\max(10^{-3}, \cdot)$ 防除零：

| 值 | 语义 |
|----|------|
| = 1 | 完美预测 |
| > 0.15 | 清晰超越 AR(1) 基线，模型确实学到时序动态 |
| = 0 | 与 AR(1) 基线持平（等价于"复制上一帧"） |
| < 0 | 差于 AR(1) 基线；当 `ar1_r2 < 0`（长程预测）时 decorr < 0 含义需结合 horizon 分析 |

---

#### 2.7.6 h=1 预测 R²（`pred_r2_h1_{nt}`）

**代码位置**：`validate()` → `pred_h1_ss_*` 累加器，键名 `pred_r2_h1_{node_type}`

与 `pred_r2` 完全相同的流程，但**只取第 1 步**（$h = 1$）进行评估：

$$
SS_{\text{res}}^{\text{pred,h1}} \mathrel{+}= \sum_{n,c} \bigl(Y_{n,1,c} - \hat{Y}_{n,1,c}\bigr)^2
$$

$$
\text{pred\_r2\_h1}_{nt} = 1 - \frac{SS_{\text{res}}^{\text{pred,h1}}}{SS_{\text{tot}}^{\text{pred,h1}}}
$$

其中 $SS_{\text{tot}}^{\text{pred,h1}}$ 使用 **仅 h=1 目标** 的独立统计量（$\sum y_{h=1}^2$，$\sum y_{h=1}$，$n_{h=1}$），确保与 NPI 的 3→1 预测直接可比。

---

#### 2.7.7 h=1 AR(1) 基线 R²（`ar1_r2_h1_{nt}`）★ NPI 可比

**代码位置**：`validate()` → `ar1_h1_ss_res` 累加器，键名 `ar1_r2_h1_{node_type}`

仅对下一步（$h = 1$）的 AR(1) 常数外推：

$$
\hat{Y}_{n,1,c}^{\text{AR(1)}} = X_{n,\, T_{\text{ctx}}-1,\, c}
$$

$$
SS_{\text{res}}^{\text{AR(1),h1}} \mathrel{+}= \sum_{n,c} \bigl(Y_{n,1,c} - X_{n,\, T_{\text{ctx}}-1,\, c}\bigr)^2
$$

与 `pred_r2_h1` **共享** h=1 目标的统计量（$\sum y_{h=1}^2$，$\sum y_{h=1}$，$n_{h=1}$）：

$$
\text{ar1\_r2\_h1}_{nt} = 1 - \frac{SS_{\text{res}}^{\text{AR(1),h1}}}{SS_{\text{tot}}^{\text{pred,h1}}}
$$

对于高自相关信号（BOLD ρ ≈ 0.85–0.95）：$\text{ar1\_r2\_h1} \approx 2\rho - 1 \approx 0.7\text{–}0.9$，
这是任何预测器在 h=1 可"免费"获得的 R²，反映数据自相关而非模型能力。

---

#### 2.7.8 h=1 去相关技能分数（`decorr_h1_{nt}`）★ NPI 可比

**代码位置**：`validate()` 末尾，键名 `decorr_h1_{node_type}`

$$
\boxed{\text{decorr\_h1}_{nt} = \frac{\text{pred\_r2\_h1}_{nt} - \text{ar1\_r2\_h1}_{nt}}{\max(10^{-3},\; 1 - \text{ar1\_r2\_h1}_{nt})}}
$$

与 NPI（Luo et al., *Nature Methods* 2025）的 3→1 预测直接可比：`decorr_h1 > 0` 表示 TwinBrain 单步预测超越纯自相关；`decorr_h1 < 0` 且 `decorr > 0` 是物理预期的梯度竞争（见 §2.7.11），非模型失败。

---

#### 2.7.9 指标间关系一览

```
原始信号 y[N, T, C]
├── 重建路径：encoder → decoder → ŷ_recon
│   └── r2_{nt}  = 1 - SS_res(y, ŷ_recon) / SS_tot(y)
│
└── 预测路径（causal re-encode → predictor → propagator → decoder）
    │   目标：y_future = y[:, T_ctx:T_ctx+S, :]   (S = min(pred_steps, T_fut))
    │   AR(1)：ŷ_AR1[n,h,c] = y[n, T_ctx-1, c]   (常数外推)
    │
    ├── pred_r2_{nt}      = 1 - SS_res(y_future, ŷ_pred)     / SS_tot(y_future)
    ├── ar1_r2_{nt}       = 1 - SS_res(y_future, ŷ_AR1)      / SS_tot(y_future)
    ├── decorr_{nt}       = (pred_r2 - ar1_r2) / max(1e-3, 1 - ar1_r2)
    │
    ├── pred_r2_h1_{nt}   = 1 - SS_res(y_future[:,0,:], ŷ_pred[:,0,:])  / SS_tot(y_future[:,0,:])
    ├── ar1_r2_h1_{nt}    = 1 - SS_res(y_future[:,0,:], ŷ_AR1[:,0,:])  / SS_tot(y_future[:,0,:])
    └── decorr_h1_{nt}    = (pred_r2_h1 - ar1_r2_h1) / max(1e-3, 1 - ar1_r2_h1)
```

所有 $SS_{\text{tot}}$ 均用 §2.7.1 的单遍全局均值算法计算。

---

#### 2.7.10 训练损失与评估指标的对应关系

`compute_loss()` 计算以下三类损失驱动 `pred_r2` 的提升（不直接出现在验证报告中）：

**① 潜空间预测损失 `pred_{nt}`**（快速收敛锚）

在 latent 空间对未来 $S$ 步做步权重加权的 Huber/MSE 监督。$\gamma>0$ 时远步权重更大（偏向长程），$\gamma=0$ 时等权：

$$
L_{\text{pred}} = \frac{1}{S} \sum_{h=1}^{S} w_h \cdot \ell\bigl(\hat{z}_h,\; z_h^{\text{fut}}\bigr), \qquad
w_h = \frac{\exp\!\left(\gamma \cdot \frac{h-1}{S-1}\right)}{\frac{1}{S}\sum_{h'}\exp\!\left(\gamma \cdot \frac{h'-1}{S-1}\right)}
$$

**② 信号空间预测损失 `pred_sig_{nt}`**（直接对齐 `pred_r2` 指标）

在原始信号空间监督，同时优化幅度（Huber）和时序形状（Pearson 相关），确保解码器在预测潜向量上也能泛化：

$$
L_{\text{pred\_sig}} = \ell\bigl(\hat{Y}_{:S},\; Y_{\text{fut},:S}\bigr) + 0.5 \cdot \bigl(1 - r(\hat{Y}_{:S},\; Y_{\text{fut},:S})\bigr)
$$

$$
r(\hat{Y}, Y) = \frac{\sum_{n,h,c}(\hat{Y}_{n,h,c} - \bar{\hat{Y}})( Y_{n,h,c} - \bar{Y})}{\sqrt{\sum_{n,h,c}(\hat{Y}_{n,h,c}-\bar{\hat{Y}})^2 \cdot \sum_{n,h,c}(Y_{n,h,c}-\bar{Y})^2}}
$$

**③ InfoNCE 对比预测损失 `pred_nce_{nt}`**（防均值崩塌，V5.47+，默认开启）

对 batch 内所有其他 (节点, 步) 对作为负样本，用余弦相似度做对比学习，防止预测器退化为预测常数均值：

$$
L_{\text{NCE}} = -\frac{1}{N \cdot S} \sum_{n,h} \log \frac{\exp(\text{sim}(\hat{z}_{n,h},\; z_{n,h}^+) / \tau)}{\sum_{n',h'} \exp(\text{sim}(\hat{z}_{n,h},\; z_{n',h'}) / \tau)}
$$

$\text{sim}(\cdot,\cdot)$ 为余弦相似度，$\tau$ 为温度参数（`info_nce_temperature`，默认 1.0）。

---

#### 2.7.11 EEG vs fMRI 的物理差异与预期值

| 模态 | 采样率 | h=1 间隔 | `ar1_r2_h1` | `decorr_h1` 目标 |
|------|--------|---------|-------------|-----------------|
| EEG  | 250 Hz | 4 ms    | ≈ 0.8–0.9   | 多步训练时可为负（梯度竞争，见下） |
| fMRI | 0.5 Hz | 2 s     | ≈ 0.7–0.9   | > 0 为目标 |

多步预测（15 步 = EEG 60ms / fMRI 30s）会把参数推向"学习时序周期模式"，而非"复制上一帧"。两种策略梯度方向不同，导致多步训练时 `decorr_h1_eeg < 0` 而 `decorr_eeg > 0` — 这是**物理预期的取舍**，不是模型失败。判读规则：

| 条件组合 | 含义 | 严重程度 |
|---------|------|---------|
| `decorr_h1 < 0` 且 `decorr > 0` | 梯度竞争，多步有效 | ℹ️ INFO |
| `decorr_h1 < 0` 且 `decorr < 0` | 单步和多步均差于 AR(1) | ⛔ WARNING |
| `pred_r2 < 0`（差于均值基线） | 绝对失败 | ⛔ WARNING |

**NPI 对比说明**：NPI（Luo et al., *Nature Methods* 2025）做 3→1 预测，未报告 `decorr_h1`。对于高自相关 BOLD（ρ≈0.85–0.95），AR(1) 单步可免费获得 R²≈0.7–0.9，NPI 部分性能增益可能来自自相关而非真实神经动力学。`decorr_h1` 是量化这一分离的客观指标。

---

#### 2.7.12 数值参考范围（充分训练模型）

| 指标 | EEG 预期范围 | fMRI 预期范围 | 文献依据 |
|------|------------|-------------|---------|
| `r2_{nt}`（重建） | ≥ 0.80 | ≥ 0.85 | 自编码器标准 |
| `pred_r2_{nt}`（多步预测） | 0.05–0.25 | 0.15–0.40 | Schirrmeister 2017; Thomas 2022 |
| `decorr_{nt}`（多步技能） | > 0 即有效；> 0.3 为良好 | > 0 即有效；> 0.3 为良好 | — |
| `pred_r2_h1_{nt}`（h=1） | < `ar1_r2_h1` 可接受（见上文） | ≥ `ar1_r2_h1` 为目标 | NPI 对标 |
| `decorr_h1_{nt}`（h=1 技能） | 可为负（多步训练取舍） | > 0 为目标 | NPI 对标 |

> **可视化工具**：训练结束后，运行 `python plot_log.py outputs/<run_dir>/` 可从 `training.log` 自动解析以上指标并生成 2×2 训练曲线图（`training_curves.png`）。详见 `plot_log.py` 文件头注释。

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
