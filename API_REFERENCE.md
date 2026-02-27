# TwinBrain V5 — 接口参考文档（API Reference）

> **本文档面向前端 / 外部智能体**，描述 TwinBrain V5 系统的所有可调用接口。  
> 读者无需阅读源代码，按本文档即可完整复现缓存读取、模型加载、推理调用、输出解析。

---

## 目录

1. [缓存图文件（Graph Cache）](#1-缓存图文件graph-cache)
2. [模型检查点文件（Checkpoint）](#2-模型检查点文件checkpoint)
3. [模型构造参数（Model Construction）](#3-模型构造参数model-construction)
4. [模型输入格式（Model Input）](#4-模型输入格式model-input)
5. [模型输出格式（Model Output）](#5-模型输出格式model-output)
6. [模型调用示例（End-to-End Usage）](#6-模型调用示例end-to-end-usage)
7. [训练器接口（Trainer API）](#7-训练器接口trainer-api)
8. [数据流全链路（Data Pipeline）](#8-数据流全链路data-pipeline)
9. [配置文件参数速查（Config Quick Reference）](#9-配置文件参数速查config-quick-reference)
10. [前端可视化数据接口（Frontend Visualization Guide）](#10-前端可视化数据接口frontend-visualization-guide)

---

## 1. 缓存图文件（Graph Cache）

### 1.1 文件位置与后缀

| 属性 | 值 |
|------|-----|
| 默认目录 | `outputs/graph_cache/`（由 `config.data.cache.dir` 控制） |
| 文件后缀 | `.pt`（PyTorch 序列化格式） |
| 单文件代表 | 一个「被试 × 任务」的**完整 run 异质图** |

### 1.2 文件命名规则

```
{subject_id}_{task_str}_{params_hash}.pt
```

| 字段 | 说明 | 示例 |
|------|------|------|
| `subject_id` | 被试目录名（BIDS 惯例为 `sub-01`） | `sub-01` |
| `task_str` | 任务名；若无任务则为 `notask` | `rest` / `wm` / `notask` |
| `params_hash` | 图构建参数（atlas、threshold、k、max_seq_len 等）的 MD5 前 8 位；参数变更后缓存自动失效 | `a3f7c912` |

**完整示例：**
```
sub-01_rest_a3f7c912.pt
sub-02_notask_a3f7c912.pt
```

### 1.3 排序约定

缓存文件**无固定排序**。`build_graphs()` 按 `data_loader.load_all_subjects()` 返回的顺序遍历被试，生成/读取缓存。  
若需要确定性顺序，调用方应自行对文件名排序：

```python
import glob, re
cache_dir = "outputs/graph_cache"
files = sorted(glob.glob(f"{cache_dir}/*.pt"))  # 字典序
```

### 1.4 缓存文件内容（HeteroData 对象）

每个 `.pt` 文件用 `torch.load` 反序列化后得到一个 `torch_geometric.data.HeteroData` 对象。

#### 加载方法

```python
import torch
graph = torch.load("outputs/graph_cache/sub-01_rest_a3f7c912.pt",
                   map_location="cpu",
                   weights_only=False)   # weights_only=False 必须，因为包含 HeteroData 对象
```

#### 节点类型（`graph.node_types`）

典型值：`['eeg', 'fmri']`（取决于 `config.data.modalities`）

#### 各节点类型的属性

**EEG 节点（`graph['eeg']`）**

| 属性 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `x` | `torch.Tensor float32` | `[N_eeg, T_eeg, 1]` | EEG 电极时序信号；N_eeg = 电极数（典型 32–64），T_eeg = 时间点数，最后一维 = 1（信号通道数） |
| `num_nodes` | `int` | — | N_eeg，与 `x.shape[0]` 一致 |
| `pos` | `torch.Tensor float32` \| `None` | `[N_eeg, 3]` | 电极三维坐标（MNE head 坐标系，单位 mm）；无坐标时为 `None` |
| `labels` | `List[str]` \| *(属性可能不存在)* | — | 电极通道名称（如 `['Fp1','Fp2',...]`）；用于可解释性分析和可视化轴标签。**建议用 `getattr(graph['eeg'], 'labels', None)` 访问**，因为老版本缓存可能无此属性。 |
| `sampling_rate` | `float` | — | 采样率（Hz），典型值 250.0 |
| `temporal_length` | `int` | — | 完整 run 的时间点数（窗口样本中应读 `x.shape[1]`） |

**fMRI 节点（`graph['fmri']`）**

| 属性 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `x` | `torch.Tensor float32` | `[N_fmri, T_fmri, 1]` | fMRI ROI 时序信号；N_fmri = ROI 数（Schaefer200 图谱时典型值 185–200，单节点回退时为 1），最后一维 = 1 |
| `num_nodes` | `int` | — | N_fmri |
| `pos` | `torch.Tensor float32` \| `None` | `[N_fmri, 3]` | ROI 质心 MNI 坐标（mm）；从 atlas 推导 |
| `labels` | `List[str]` \| *(属性可能不存在)* | — | ROI 名称（如 `['7Networks_LH_Vis_1',...]`）；用于可视化脑区标注。**建议用 `getattr(graph['fmri'], 'labels', None)` 访问**，因为老版本缓存可能无此属性。 |
| `sampling_rate` | `float` | — | 采样率（Hz），典型值 0.5（即 TR = 2 s） |
| `temporal_length` | `int` | — | 完整 run 的时间点数（窗口样本中应读 `x.shape[1]`） |

**图级属性（`graph` 对象本身，不属于任何节点类型）**

| 属性 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `subject_idx` | `torch.Tensor int64` | `[]`（0-D 标量） | 被试在数据集中的整数索引；由 `build_graphs()` 返回的 `subject_to_idx` 字典推导。模型 `forward()` 用此索引查询 `nn.Embedding` 实现个性化被试嵌入。**窗口样本会从完整图自动继承此属性。** |

> **注意**：`subject_idx` 仅在 `num_subjects > 0`（模型启用被试嵌入）时对推理有意义。
> 老版本缓存（V5.18 以前）可能不含此属性；重建缓存即可获得。

#### 边类型（`graph.edge_types`）

| 边类型元组 | 始终存在 | 说明 |
|-----------|---------|------|
| `('eeg', 'connects', 'eeg')` | ✅ | EEG 内部边（电极间 Pearson \|r\| 功能连接，k-NN + 阈值过滤） |
| `('fmri', 'connects', 'fmri')` | ✅ | fMRI 内部边（ROI 间功能连接，同上） |
| `('eeg', 'projects_to', 'fmri')` | 仅双模态 | 跨模态边：EEG → fMRI（两个模态均存在时自动建立） |
| `('fmri', 'structural', 'fmri')` | 可选 | DTI 结构连通性边（`dti_structural_edges: true` 且被试目录下存在连通矩阵时才存在） |

**各边属性**

| 属性 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `edge_index` | `torch.Tensor int64` | `[2, E]` | 边索引；`edge_index[0]` = 源节点索引，`edge_index[1]` = 目标节点索引 |
| `edge_attr` | `torch.Tensor float32` | `[E, 1]` | 边权重。**同模态边**（`connects`、`structural`）：Pearson \|r\| 相关系数（0–1）。**跨模态边**（`projects_to`）：每条边权重 **精确等于 1.0**（无坐标配准时使用随机连接，统一赋值 1.0）。 |

#### 不变量（Invariants，违反即数据错误）

```
N_eeg < N_fmri                    # EEG 电极数 < fMRI ROI 数（仅当 atlas 分区成功时成立；
                                   # atlas 失败回退到单节点时 N_fmri=1，此不变量被违反，
                                   # graph_cache_mapper 会打印 WARNING，模型仍可运行但无空间信息）
graph['eeg'].x.shape[-1] == 1     # 最后一维始终为 1
graph['fmri'].x.shape[-1] == 1    # 最后一维始终为 1
graph.subject_idx.ndim == 0       # 0 维标量，dtype=int64
```

---

## 2. 模型检查点文件（Checkpoint）

### 2.1 文件位置与命名

| 文件名 | 触发时机 | 说明 |
|--------|---------|------|
| `outputs/<experiment_name>/best_model.pt` | 验证 loss 改善时自动保存 | 最优模型，推理首选 |
| `outputs/<experiment_name>/checkpoint_epoch_{N}.pt` | 每 `save_frequency`（默认 10）个 epoch | 定期快照，用于恢复训练 |

### 2.2 检查点文件内容

```python
checkpoint = torch.load("best_model.pt", map_location="cpu")
```

```python
# checkpoint 是一个 dict，包含以下键：
{
    "epoch": int,                        # 保存时的 epoch 编号
    "model_state_dict": OrderedDict,     # 模型权重（用于 model.load_state_dict()）
    "optimizer_state_dict": dict,        # AdamW 优化器状态（用于恢复训练）
    "history": {
        "train_loss": List[float],       # 每 epoch 的训练平均 loss
        "val_loss": List[float],         # 每次验证的验证 loss
    },
    # 仅当 use_adaptive_loss=True 时存在：
    "loss_balancer_state": OrderedDict,  # AdaptiveLossBalancer 权重状态
}
```

### 2.3 纯推理加载（最小代码）

```python
import torch
from models.graph_native_system import GraphNativeBrainModel

# ── 0. 读取训练时保存的 config（与训练参数保持一致）───────────────────
import yaml
with open("outputs/twinbrain_v5_xxx/config.yaml") as f:
    saved_cfg = yaml.safe_load(f)

# ── 1. 重建模型（参数必须与训练时一致，见第 3 节）───────────────────
# num_subjects 必须与训练时完全一致（从保存的 config 或 checkpoint 读取）
# 若训练时启用了被试嵌入，此处必须传入正确的 num_subjects，否则权重无法加载
num_subjects = saved_cfg.get('num_subjects_at_train', 0)  # 见附录 C

model = GraphNativeBrainModel(
    node_types=['eeg', 'fmri'],
    edge_types=[
        ('eeg', 'connects', 'eeg'),
        ('fmri', 'connects', 'fmri'),
        ('eeg', 'projects_to', 'fmri'),
    ],
    in_channels_dict={'eeg': 1, 'fmri': 1},
    hidden_channels=128,
    num_encoder_layers=4,
    num_decoder_layers=3,
    use_prediction=True,
    prediction_steps=10,
    num_subjects=num_subjects,  # 若训练时为 0，此处也传 0
)

# ── 2. 加载权重 ──────────────────────────────────────────────────
checkpoint = torch.load("outputs/twinbrain_v5_xxx/best_model.pt",
                        map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

---

## 3. 模型构造参数（Model Construction）

### `GraphNativeBrainModel.__init__`

```python
GraphNativeBrainModel(
    node_types:               List[str],           # 节点类型列表，如 ['eeg', 'fmri']
    edge_types:               List[Tuple[str,str,str]],  # 边类型三元组列表
    in_channels_dict:         Dict[str, int],      # 各模态输入通道数，固定为 {'eeg':1, 'fmri':1}
    hidden_channels:          int   = 128,         # 编码器隐层维度 H
    num_encoder_layers:       int   = 4,           # ST-GCN 编码器层数
    num_decoder_layers:       int   = 3,           # 解码器卷积层数
    use_prediction:           bool  = True,        # 是否启用潜空间预测头
    prediction_steps:         int   = 10,          # 预测步数
    dropout:                  float = 0.1,         # Dropout 比率
    loss_type:                str   = 'mse',       # 损失函数 'mse'|'huber'|'smooth_l1'
    use_gradient_checkpointing: bool = False,      # 梯度检查点（节省显存）
    predictor_config:         Optional[dict] = None,  # 见下方子参数
    use_dynamic_graph:        bool  = False,       # 动态图结构学习（DynamicGraphConstructor）
    k_dynamic_neighbors:      int   = 10,          # 动态图 k 近邻数
    num_subjects:             int   = 0,           # 被试总数；>0 时启用被试特异性嵌入（个性化）
)
```

**`predictor_config` 子参数**（对应 `config.v5_optimization.advanced_prediction`）：

| 键 | 类型 | 默认值 | 说明 |
|----|------|--------|------|
| `use_hierarchical` | bool | True | 多尺度层级预测（粗→细） |
| `use_transformer` | bool | True | 使用 Transformer（否则 GRU） |
| `use_uncertainty` | bool | True | 输出预测均值 + 方差 |
| `num_scales` | int | 3 | 时间尺度数（层级模式） |
| `num_windows` | int | 3 | 每次 forward 采样窗口数 |
| `sampling_strategy` | str | `"uniform"` | 窗口采样策略：`uniform`/`random`/`adaptive` |

**`num_subjects` 参数说明（被试特异性嵌入）：**

| 值 | 行为 |
|----|------|
| `0`（默认） | 禁用个性化；所有被试共享同一模型权重 |
| `> 0` | 创建 `nn.Embedding(num_subjects, H)`，为每个被试学习一个 `[H]` 维潜空间偏移，在编码器输入投影后施加到所有节点特征 |

- **训练时**：由 `build_graphs()` 返回的 `subject_to_idx` 字典长度决定，自动传入 `create_model(num_subjects=len(subject_to_idx))`
- **推理时**：必须与训练时的 `num_subjects` 完全一致，否则 `Embedding` 维度不匹配无法加载权重
- **个性化推理**：可对新被试 fine-tune 仅该 Embedding（frozen encoder），实现少样本个性化
- **图缓存中的对应字段**：`graph.subject_idx`（0-D int64 张量），模型 `forward()` 自动读取

**与配置文件 `default.yaml` 的对应关系**

```yaml
# default.yaml 对应字段 → GraphNativeBrainModel 参数
model.hidden_channels      → hidden_channels
model.num_encoder_layers   → num_encoder_layers
model.num_decoder_layers   → num_decoder_layers
model.use_prediction       → use_prediction
model.prediction_steps     → prediction_steps
model.dropout              → dropout
model.loss_type            → loss_type
model.use_dynamic_graph    → use_dynamic_graph
model.k_dynamic_neighbors  → k_dynamic_neighbors
training.use_gradient_checkpointing → use_gradient_checkpointing
v5_optimization.advanced_prediction → predictor_config
# num_subjects 由 build_graphs() 在运行时从数据集中自动推导，无 YAML 配置项
# 推理时须从训练的 config.yaml 中读取实际被试数（见第 6.5 节）
```

---

## 4. 模型输入格式（Model Input）

### 4.1 输入类型

模型的 `forward()` 接受一个 **`torch_geometric.data.HeteroData`** 对象，结构与缓存文件中的图完全相同（见第 1.4 节）。

**关键约束：**

```python
# 每个节点类型的特征张量必须满足：
graph['eeg'].x.shape   == [N_eeg, T_eeg, 1]    # 3 维，最后维 = 1
graph['fmri'].x.shape  == [N_fmri, T_fmri, 1]  # 3 维，最后维 = 1

# 不允许出现 NaN 或 Inf：
torch.isnan(graph['eeg'].x).any()   == False
torch.isinf(graph['fmri'].x).any()  == False
```

### 4.2 将缓存图送入模型

```python
# 直接加载缓存图后移至 GPU 即可用作输入
graph = torch.load("outputs/graph_cache/sub-01_rest_a3f7c912.pt",
                   map_location="cpu", weights_only=False)
graph = graph.to("cuda")   # 移至目标设备

with torch.no_grad():
    reconstructed, predictions = model(graph)
```

### 4.3 时间窗口切片输入（dFC 模式）

若使用窗口采样，每个窗口仍是一个 `HeteroData`，但 `x` 已切片为 `[N, window_size, 1]`，`edge_index` 与完整图共享（不复制）。

---

## 5. 模型输出格式（Model Output）

### 5.1 `forward()` 签名

```python
def forward(
    self,
    data:              HeteroData,
    return_prediction: bool = False,    # 是否返回未来预测（推理时使用）
    return_encoded:    bool = False,    # 是否返回潜空间编码（训练/分析时使用）
) -> Tuple
```

### 5.2 返回值结构（三种模式）

#### 模式 A：默认（`return_encoded=False`）

```python
reconstructed, predictions = model(data)
# reconstructed: Dict[str, Tensor]  — 信号重建结果
# predictions:   None               — return_prediction=False 时为 None
```

#### 模式 B：含预测（`return_prediction=True`）

```python
reconstructed, predictions = model(data, return_prediction=True)
# reconstructed: Dict[str, Tensor]
# predictions:   Dict[str, Tensor]  — 各模态未来预测
```

#### 模式 C：含编码（`return_encoded=True`）

```python
reconstructed, predictions, encoded_dict = model(data, return_encoded=True)
# reconstructed:  Dict[str, Tensor]
# predictions:    None（除非同时 return_prediction=True）
# encoded_dict:   Dict[str, Tensor]  — 潜空间表征
```

### 5.3 各返回张量的详细格式

#### `reconstructed`（信号重建）

```python
reconstructed: Dict[str, torch.Tensor]
# Key   = node_type ('eeg', 'fmri')
# Value = Tensor[N, T, 1]   — 与输入 x 形状相同（T' 可能因 decoder 略有差异，系统自动截对齐）

reconstructed['eeg']   # shape: [N_eeg,  T_eeg,  1]
reconstructed['fmri']  # shape: [N_fmri, T_fmri, 1]
```

#### `predictions`（未来预测，`return_prediction=True`）

```python
predictions: Dict[str, torch.Tensor]
# Key   = node_type
# Value = Tensor[N, prediction_steps, H]  — 潜空间预测，非信号空间

predictions['eeg']   # shape: [N_eeg,  prediction_steps, H]  H = hidden_channels
predictions['fmri']  # shape: [N_fmri, prediction_steps, H]
```

> **注意**：`predictions` 输出在**潜空间**（维度 H），不是原始信号空间（维度 1）。  
> 若要获取信号空间的未来预测，需额外通过 decoder 解码，或使用 `encoded` + decoder 的组合（当前版本未封装此接口）。

#### `encoded_dict`（潜空间编码，`return_encoded=True`）

```python
encoded_dict: Dict[str, torch.Tensor]
# Key   = node_type
# Value = Tensor[N, T, H]  — 编码器输出，包含跨模态信息

encoded_dict['eeg']   # shape: [N_eeg,  T_eeg,  H]
encoded_dict['fmri']  # shape: [N_fmri, T_fmri, H]
```

> `encoded_dict['fmri']` 已隐含 EEG→fMRI 跨模态消息（通过 ST-GCN 跨模态边传入）。

### 5.4 损失计算输出（`compute_loss`）

```python
losses: Dict[str, torch.Tensor] = model.compute_loss(
    data,
    reconstructed,
    encoded=encoded_dict,   # 传入时启用潜空间预测损失
)

# losses 的键：
{
    'recon_eeg':  Tensor(scalar),  # EEG 重建损失（Huber/MSE）
    'recon_fmri': Tensor(scalar),  # fMRI 重建损失
    'pred_eeg':   Tensor(scalar),  # EEG 潜空间预测损失（仅当 encoded 非 None 且 use_prediction=True）
    'pred_fmri':  Tensor(scalar),  # fMRI 潜空间预测损失
}
```

---

## 6. 模型调用示例（End-to-End Usage）

### 6.1 纯推理（信号重建）

```python
import torch
from models.graph_native_system import GraphNativeBrainModel

# ── 1. 构建模型（与训练时参数一致）───────────────────────────────────
model = GraphNativeBrainModel(
    node_types=['eeg', 'fmri'],
    edge_types=[
        ('eeg', 'connects', 'eeg'),
        ('fmri', 'connects', 'fmri'),
        ('eeg', 'projects_to', 'fmri'),
    ],
    in_channels_dict={'eeg': 1, 'fmri': 1},
    hidden_channels=128,
    num_encoder_layers=4,
    num_decoder_layers=3,
    use_prediction=True,
    prediction_steps=10,
    loss_type='huber',
)

# ── 2. 加载检查点 ──────────────────────────────────────────────────
ckpt = torch.load("outputs/twinbrain_v5_xxx/best_model.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ── 3. 加载缓存图 ──────────────────────────────────────────────────
graph = torch.load("outputs/graph_cache/sub-01_rest_a3f7c912.pt",
                   map_location="cpu", weights_only=False)

# ── 4. 推理 ───────────────────────────────────────────────────────
with torch.no_grad():
    reconstructed, _ = model(graph)

# ── 5. 读取重建结果 ────────────────────────────────────────────────
eeg_recon  = reconstructed['eeg']   # [N_eeg,  T_eeg,  1]
fmri_recon = reconstructed['fmri']  # [N_fmri, T_fmri, 1]

print(f"EEG 重建形状:  {eeg_recon.shape}")
print(f"fMRI 重建形状: {fmri_recon.shape}")
```

### 6.2 推理（含未来预测）

```python
with torch.no_grad():
    reconstructed, predictions = model(graph, return_prediction=True)

# 预测结果在潜空间
eeg_pred  = predictions['eeg']   # [N_eeg,  prediction_steps, H]
fmri_pred = predictions['fmri']  # [N_fmri, prediction_steps, H]

# 每一行 = 一个节点（电极/ROI）的未来 prediction_steps 步潜向量序列
```

### 6.3 提取潜空间编码（用于下游分析）

```python
with torch.no_grad():
    _, _, encoded = model(graph, return_encoded=True)

# encoded['fmri'] 含有 EEG 跨模态信息，可直接用于分类/聚类
fmri_latent = encoded['fmri']  # [N_fmri, T_fmri, H=128]

# 示例：提取每个 ROI 的时序均值特征用于静息态指纹分析
roi_fingerprint = fmri_latent.mean(dim=1)  # [N_fmri, H]
```

### 6.4 批量推理（多个缓存图）

```python
import glob, torch
from pathlib import Path

cache_dir = Path("outputs/graph_cache")
cache_files = sorted(cache_dir.glob("*.pt"))  # 字典序

results = []
model.eval()
with torch.no_grad():
    for pt_file in cache_files:
        graph = torch.load(pt_file, map_location="cpu", weights_only=False)
        reconstructed, _ = model(graph)
        results.append({
            "file": pt_file.name,
            "subject_idx": graph.subject_idx.item() if hasattr(graph, 'subject_idx') else None,
            "eeg_labels":  getattr(graph['eeg'], 'labels', None),   # List[str] 或 None
            "fmri_labels": getattr(graph['fmri'], 'labels', None),  # List[str] 或 None
            "eeg_recon":   reconstructed['eeg'].numpy(),   # [N_eeg, T, 1]
            "fmri_recon":  reconstructed['fmri'].numpy(),  # [N_fmri, T, 1]
        })
```

### 6.5 推理时确定 `num_subjects`（含被试嵌入的模型）

若训练时启用了被试特异性嵌入（`num_subjects > 0`），推理时必须传入相同的 `num_subjects`。
推荐从训练输出目录读取保存的 checkpoint（其中嵌入了模型 state_dict 的形状信息）：

```python
import torch

ckpt = torch.load("outputs/twinbrain_v5_xxx/best_model.pt", map_location="cpu")

# 方法 A：从 state_dict 推断（最可靠，不依赖外部记录）
state = ckpt["model_state_dict"]
if "subject_embed.weight" in state:
    num_subjects = state["subject_embed.weight"].shape[0]
    hidden_channels = state["subject_embed.weight"].shape[1]
    print(f"被试嵌入启用: {num_subjects} 个被试, H={hidden_channels}")
else:
    num_subjects = 0
    print("被试嵌入未启用")

# 方法 B：从 history 中读取（需要训练时显式保存）
# num_subjects = ckpt.get("num_subjects", 0)

# 然后用正确的 num_subjects 构建模型
model = GraphNativeBrainModel(..., num_subjects=num_subjects)
model.load_state_dict(ckpt["model_state_dict"])
```

---

## 7. 训练器接口（Trainer API）

### 7.1 `GraphNativeTrainer.__init__`

```python
GraphNativeTrainer(
    model:                    GraphNativeBrainModel,
    node_types:               List[str],
    learning_rate:            float = 1e-4,
    weight_decay:             float = 1e-5,
    use_adaptive_loss:        bool  = True,        # GradNorm 自适应损失平衡
    use_eeg_enhancement:      bool  = True,        # EEG 静默通道增强
    use_amp:                  bool  = True,        # 混合精度（AMP）
    use_gradient_checkpointing: bool = False,
    use_scheduler:            bool  = True,        # LR 调度器
    scheduler_type:           str   = 'cosine',    # 'cosine'|'onecycle'|'plateau'
    use_torch_compile:        bool  = True,
    compile_mode:             str   = 'reduce-overhead',
    device:                   str   = 'cuda',
    optimization_config:      Optional[dict] = None,  # config['v5_optimization']
)
```

### 7.2 核心方法

| 方法 | 签名 | 返回值 | 说明 |
|------|------|--------|------|
| `train_step` | `(data: HeteroData) -> Dict[str, float]` | `{'recon_eeg': float, 'recon_fmri': float, 'pred_eeg': float, 'pred_fmri': float, 'total': float}` | 单步训练 |
| `train_epoch` | `(data_list, epoch, total_epochs) -> float` | 平均 loss（float） | 一个 epoch |
| `validate` | `(data_list: List[HeteroData]) -> float` | 平均验证 loss | 验证集评估 |
| `save_checkpoint` | `(path: Path, epoch: int)` | — | 原子保存检查点 |
| `load_checkpoint` | `(path: Path) -> int` | 已恢复的 epoch 编号 | 加载检查点（含 optimizer 状态） |

### 7.3 `train_step` 输出详解

```python
loss_dict = trainer.train_step(data)
# {
#   'recon_eeg':  0.042,   # EEG 重建损失（当前 batch 标量值）
#   'recon_fmri': 0.031,   # fMRI 重建损失
#   'pred_eeg':   0.018,   # EEG 潜空间预测损失（use_prediction=True 时）
#   'pred_fmri':  0.025,   # fMRI 潜空间预测损失
#   'eeg_reg':    0.003,   # EEG 防零崩塌正则化（use_eeg_enhancement=True 时）
#   'total':      0.119,   # 加权总损失
# }
```

---

## 8. 数据流全链路（Data Pipeline）

```
原始数据文件（.set / .nii.gz）
         ↓  BrainDataLoader.load_all_subjects()
subject_data dict:
  {'subject_id': 'sub-01',
   'eeg':  {'data': ndarray[N_ch, T], 'ch_names': [...], 'sfreq': 250.0, 'ch_pos': ...},
   'fmri': {'data': ndarray[X,Y,Z,T] or [N_roi,T], 'img': NIfTI obj}}
         ↓  GraphNativeBrainMapper
         ↓    map_fmri_to_graph()   → HeteroData['fmri'].x  [N_fmri, T, 1]
         ↓    map_eeg_to_graph()    → HeteroData['eeg'].x   [N_eeg, T, 1]
         ↓    create_simple_cross_modal_edges()              (edge_attr=均匀1.0)
         ↓    add_dti_structural_edges()  (可选，仅当 dti_structural_edges=true)
         ↓    built_graph.subject_idx = subject_to_idx[subject_id]
完整 run 异质图（HeteroData）
         ↓  torch.save(graph, cache_path)   ← 缓存 .pt 文件（含 subject_idx）
         ↓  extract_windowed_samples()       ← dFC 窗口切片（可选）
              窗口样本继承: num_nodes, pos, sampling_rate, labels, subject_idx
训练样本列表 List[HeteroData]
         ↓  GraphNativeBrainModel.forward()
         ↓    Encoder（ST-GCN × num_encoder_layers）
         ↓      subject_embed 查询（若 num_subjects>0）
         ↓    Decoder（Conv1d × num_decoder_layers）
         ↓    Predictor（EnhancedMultiStepPredictor，可选）
输出:
  reconstructed   Dict[str, Tensor[N, T, 1]]
  predictions     Dict[str, Tensor[N, steps, H]]  （可选）
  encoded_dict    Dict[str, Tensor[N, T, H]]       （可选）
```

### `build_graphs()` 返回值

```python
graphs, mapper, subject_to_idx = build_graphs(config, logger)
# graphs:          List[HeteroData]          — 所有被试-任务图（或窗口样本）
# mapper:          GraphNativeBrainMapper    — 映射器实例（含 atlas 信息）
# subject_to_idx:  Dict[str, int]            — {subject_id_str → int_idx}
#                  例: {'sub-01': 0, 'sub-02': 1, ...}
#                  传入 create_model(num_subjects=len(subject_to_idx)) 以启用个性化嵌入
```

---

## 9. 配置文件参数速查（Config Quick Reference）

配置文件默认路径：`configs/default.yaml`  
训练时自动保存到：`outputs/<experiment_name>/config.yaml`

### 影响缓存命名（修改即失效）的参数

```yaml
graph:
  threshold_fmri: 0.3
  threshold_eeg:  0.2
  k_nearest_fmri: 20
  k_nearest_eeg:  10
  add_self_loops: true
  make_undirected: true

data:
  atlas:
    name: "schaefer200"
    file: "atlases/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii"  # 注意：1mm 非压缩格式
  modalities: ["eeg", "fmri"]
  dti_structural_edges: false  # 切换此值会使旧缓存失效并触发重建

training:
  max_seq_len: 300   # windowed_sampling.enabled=true 时设为 null

windowed_sampling:
  enabled: false
```

### 影响模型结构（加载时必须与训练一致）的参数

```yaml
model:
  hidden_channels: 128
  num_encoder_layers: 4
  num_decoder_layers: 3
  use_prediction: true
  prediction_steps: 10
  dropout: 0.1
  loss_type: "huber"
  use_dynamic_graph: false
  k_dynamic_neighbors: 10
```

### 图缓存快速开关

```yaml
data:
  cache:
    enabled: true          # false = 每次重新构建，不读写缓存
    dir: "outputs/graph_cache"
```

---

## 附录 A：常见错误与排查

| 错误信息 | 原因 | 解决 |
|---------|------|------|
| `RuntimeError: Node count mismatch for 'fmri'` | 跨模态边的 `size` 参数缺失导致 N_eeg 污染 N_fmri | 使用正确的训练版本，缓存重建 |
| `RuntimeError: Trying to backward through the graph a second time` | 跨 epoch 共享了带 grad_fn 的张量 | 确保 `train_step` 的 finally 块正常执行 |
| `weights_only=False` 报警告 | PyTorch 2.6+ 安全警告 | 保留 `weights_only=False`，HeteroData 非纯权重对象，必须此参数 |
| 加载缓存后 `N_fmri=1` | atlas 文件路径错误或 nilearn 未安装 | 检查 `atlases/` 目录，`pip install nilearn` |
| `KeyError: 'eeg'` on `reconstructed` | 数据中仅有单模态 | 确保 `config.data.modalities` 与数据集匹配 |
| `size mismatch for subject_embed.weight` | 推理时 `num_subjects` 与训练时不一致 | 按第 6.5 节从 state_dict 推断正确的 `num_subjects` |
| `graphs[0]` 无 `subject_idx` 属性 | 图缓存由 V5.18 或更早版本生成 | 清除 `outputs/graph_cache/` 后重新运行 |

---

## 附录 B：版本与依赖

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| `torch` | 1.13+ | 推荐 2.0+（支持 `torch.compile`） |
| `torch_geometric` | 2.3+ | HeteroData、MessagePassing |
| `nilearn` | 0.9+ | atlas 分区（可选，无则 N_fmri=1） |
| `mne` | 1.0+ | EEG 加载与预处理 |
| `nibabel` | 4.0+ | NIfTI fMRI 文件读取 |
| `numpy` | 1.21+ | — |
| `yaml` (`pyyaml`) | 6.0+ | 配置文件解析 |

---

## 附录 C：训练产物文件列表

训练完成后，`outputs/<experiment_name>/` 目录下的文件一览：

| 文件 | 说明 |
|------|------|
| `best_model.pt` | 验证 loss 最优时保存的检查点（推理首选） |
| `checkpoint_epoch_N.pt` | 每 `save_frequency` 个 epoch 保存的快照 |
| `config.yaml` | 训练时使用的完整配置（含数据路径、模型参数、超参数） |
| `training.log` | 完整训练日志（含每 epoch loss、配置核对表） |

图缓存（所有训练复用）：

| 文件 | 说明 |
|------|------|
| `outputs/graph_cache/{sub}_{task}_{hash}.pt` | 预处理后的异质图（含 subject_idx） |

---

## 10. 前端可视化数据接口（Frontend Visualization Guide）

本节面向前端智能体，描述如何将 TwinBrain 的训练产物（checkpoint + graph cache）转换为可视化所需的数据格式。

### 10.1 数据提取工作流

```
训练产物
  ├── best_model.pt               ← 模型权重（第 2 节）
  ├── config.yaml                 ← 训练配置（第 9 节）
  └── graph_cache/*.pt            ← 预处理图（第 1 节）
          ↓ 加载 + 推理
可视化数据
  ├── 脑网络图（节点 + 边）
  ├── 时序信号（原始 vs 重建）
  ├── 潜空间表征（降维后 2D/3D）
  ├── 功能连接矩阵
  └── 被试差异热图
```

### 10.2 脑网络图可视化数据

从缓存图直接提取节点坐标和边：

```python
import torch
import numpy as np

graph = torch.load("outputs/graph_cache/sub-01_rest_xxxx.pt",
                   map_location="cpu", weights_only=False)

# ── 辅助函数（在 Section 10 中全局复用）──────────────────────────────
def get_labels(node_store, n_nodes):
    """安全读取节点标签；老版本缓存可能无此属性。"""
    labels = getattr(node_store, 'labels', None)
    return list(labels) if labels is not None else [f"node_{i}" for i in range(n_nodes)]

# ── EEG 电极网络 ──────────────────────────────────────────────────────
N_eeg = graph['eeg'].num_nodes
eeg_nodes = {
    "positions": graph['eeg'].pos.numpy() if graph['eeg'].pos is not None else None,
                 # [N_eeg, 3]  MNE head 坐标 (mm)
    "labels":    get_labels(graph['eeg'], N_eeg),  # ['Fp1', 'Fp2', ...]
    "signals":   graph['eeg'].x.squeeze(-1).numpy(),  # [N_eeg, T]  去掉最后维度的1
}

eeg_edges = {
    "edge_index": graph['eeg', 'connects', 'eeg'].edge_index.numpy(),       # [2, E]
    "edge_weight": graph['eeg', 'connects', 'eeg'].edge_attr.squeeze(-1).numpy(),  # [E]  Pearson |r|
}

# ── fMRI ROI 网络 ─────────────────────────────────────────────────────
N_fmri = graph['fmri'].num_nodes
fmri_nodes = {
    "positions": graph['fmri'].pos.numpy() if graph['fmri'].pos is not None else None,
                 # [N_fmri, 3]  MNI 坐标 (mm)
    "labels":    get_labels(graph['fmri'], N_fmri),  # ['7Networks_LH_Vis_1', ...]
    "signals":   graph['fmri'].x.squeeze(-1).numpy(),  # [N_fmri, T]
}

fmri_edges = {
    "edge_index": graph['fmri', 'connects', 'fmri'].edge_index.numpy(),      # [2, E]
    "edge_weight": graph['fmri', 'connects', 'fmri'].edge_attr.squeeze(-1).numpy(),  # [E]
}

# ── 跨模态边（EEG → fMRI）─────────────────────────────────────────────
if ('eeg', 'projects_to', 'fmri') in graph.edge_types:
    cross_edges = {
        "edge_index": graph['eeg', 'projects_to', 'fmri'].edge_index.numpy(),       # [2, E]
        "edge_weight": graph['eeg', 'projects_to', 'fmri'].edge_attr.squeeze(-1).numpy(),  # [E] 均匀1.0
    }
```

### 10.3 时序信号可视化（原始 vs 重建）

```python
import torch
from models.graph_native_system import GraphNativeBrainModel

# ── 加载模型（见第 2.3 节 + 第 6.5 节）─────────────────────────────────
ckpt = torch.load("outputs/twinbrain_v5_xxx/best_model.pt", map_location="cpu")
state = ckpt["model_state_dict"]
num_subjects = state["subject_embed.weight"].shape[0] if "subject_embed.weight" in state else 0

model = GraphNativeBrainModel(
    node_types=['eeg', 'fmri'],
    edge_types=[('eeg','connects','eeg'), ('fmri','connects','fmri'), ('eeg','projects_to','fmri')],
    in_channels_dict={'eeg': 1, 'fmri': 1},
    hidden_channels=128, num_encoder_layers=4, num_decoder_layers=3,
    use_prediction=True, prediction_steps=10,
    num_subjects=num_subjects,
)
model.load_state_dict(state)
model.eval()

# ── 推理 ──────────────────────────────────────────────────────────────
graph = torch.load("outputs/graph_cache/sub-01_rest_xxxx.pt",
                   map_location="cpu", weights_only=False)

with torch.no_grad():
    reconstructed, _ = model(graph)

# ── 构建可视化数据 ────────────────────────────────────────────────────
# EEG 时序（形状: [N_eeg, T]）
eeg_original      = graph['eeg'].x.squeeze(-1).numpy()           # 原始信号
eeg_reconstructed = reconstructed['eeg'].squeeze(-1).numpy()    # 重建信号
eeg_labels        = get_labels(graph['eeg'], graph['eeg'].num_nodes)

# fMRI 时序（形状: [N_fmri, T]）
fmri_original      = graph['fmri'].x.squeeze(-1).numpy()
fmri_reconstructed = reconstructed['fmri'].squeeze(-1).numpy()
fmri_labels        = get_labels(graph['fmri'], graph['fmri'].num_nodes)

# 采样率（用于生成时间轴）
eeg_sr  = getattr(graph['eeg'],  'sampling_rate', 250.0)    # Hz
fmri_sr = getattr(graph['fmri'], 'sampling_rate', 0.5)      # Hz
T_eeg   = eeg_original.shape[1]
T_fmri  = fmri_original.shape[1]
eeg_time_axis  = np.arange(T_eeg)  / eeg_sr   # 秒
fmri_time_axis = np.arange(T_fmri) / fmri_sr  # 秒

# 重建误差（逐节点 RMSE，用于热图着色）
eeg_rmse  = np.sqrt(((eeg_original  - eeg_reconstructed)  ** 2).mean(axis=1))  # [N_eeg]
fmri_rmse = np.sqrt(((fmri_original - fmri_reconstructed) ** 2).mean(axis=1))  # [N_fmri]
```

### 10.4 潜空间降维可视化（UMAP / PCA）

```python
import torch
from sklearn.decomposition import PCA   # 或 import umap

with torch.no_grad():
    _, _, encoded = model(graph, return_encoded=True)

# fMRI 潜向量: [N_fmri, T, H] → 每 ROI 取时序均值 → [N_fmri, H]
fmri_latent = encoded['fmri'].mean(dim=1).numpy()  # [N_fmri, H=128]

# 降至 2D/3D 用于散点图
pca = PCA(n_components=2)
fmri_2d = pca.fit_transform(fmri_latent)  # [N_fmri, 2]

# 结合节点坐标和标签
visualization_data = {
    "nodes_2d":      fmri_2d.tolist(),          # [[x,y], ...]
    "node_labels":   get_labels(graph['fmri'], graph['fmri'].num_nodes),
    "node_positions_mni": graph['fmri'].pos.numpy().tolist() if graph['fmri'].pos is not None else None,
}
```

### 10.5 功能连接矩阵（Functional Connectivity Matrix）

```python
# 从 edge_index + edge_attr 重建 N×N 功能连接矩阵
def edge_to_matrix(edge_index, edge_attr, n_nodes):
    """[2, E], [E, 1] → [N, N] dense matrix
    
    注意：intra-modal edges 由 make_undirected=True 双向存储（每条无向边有两条有向边），
    因此 mat[src,dst] 和 mat[dst,src] 都会被填入，结果矩阵是对称的。
    """
    mat = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    src = edge_index[0]
    dst = edge_index[1]
    w   = edge_attr.squeeze(-1)
    mat[src, dst] = w  # 双向边已包含 (i,j) 和 (j,i)，矩阵自然对称
    return mat

fmri_ei = graph['fmri', 'connects', 'fmri'].edge_index.numpy()
fmri_ea = graph['fmri', 'connects', 'fmri'].edge_attr.numpy()
N_fmri  = graph['fmri'].num_nodes

fc_matrix = edge_to_matrix(fmri_ei, fmri_ea, N_fmri)
# fc_matrix: [N_fmri, N_fmri]  值域 [0, 1]（Pearson |r|），对称矩阵（make_undirected=True）
# 可直接传入热图组件（如 Plotly Heatmap / D3.js 矩阵）

fmri_labels = get_labels(graph['fmri'], N_fmri)  # Schaefer200 网络层级着色参考
```

### 10.6 被试差异可视化（Subject Embedding）

`subject_to_idx` 字典（`{subject_id_str → int_idx}`）由训练时 `build_graphs()` 生成。推理时需要此字典将嵌入索引映射回被试 ID。**必须在训练结束后持久化保存**，否则无法在推理时重建映射：

```python
# ── 训练阶段：保存 subject_to_idx ────────────────────────────────────
import json
graphs, mapper, subject_to_idx = build_graphs(config, logger)
with open(f"outputs/{experiment_name}/subject_to_idx.json", "w") as f:
    json.dump(subject_to_idx, f, indent=2)
# 内容示例: {"sub-01": 0, "sub-02": 1, "sub-03": 2}

# ── 推理阶段：加载 subject_to_idx ────────────────────────────────────
with open("outputs/twinbrain_v5_xxx/subject_to_idx.json") as f:
    subject_to_idx = json.load(f)
# subject_to_idx: {'sub-01': 0, 'sub-02': 1, ...}
```

```python
# ── 可视化被试嵌入 ───────────────────────────────────────────────────
from sklearn.decomposition import PCA

if num_subjects > 0:
    all_embeddings = model.subject_embed.weight.detach().numpy()  # [num_subjects, H]

    # 降维至 2D 展示被试间差异
    pca = PCA(n_components=2)
    subjects_2d = pca.fit_transform(all_embeddings)  # [num_subjects, 2]

    # 按 idx 排序还原 subject_id 列表
    subject_ids = sorted(subject_to_idx.keys(), key=lambda k: subject_to_idx[k])
    subject_viz = {
        "subject_ids":   subject_ids,                # ['sub-01', 'sub-02', ...]
        "embeddings_2d": subjects_2d.tolist(),        # [[x,y], ...]
    }
```

### 10.7 数据导出为 JSON（REST API 接口示例）

将所有可视化数据序列化为 JSON，供前端直接消费：

```python
import json, numpy as np

class NumpyEncoder(json.JSONEncoder):
    """将 numpy array 序列化为 list"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

payload = {
    # ── 图结构 ──────────────────────────────────────────
    "eeg_nodes": {
        "count":     int(graph['eeg'].num_nodes),
        "labels":    get_labels(graph['eeg'], graph['eeg'].num_nodes),
        "positions": graph['eeg'].pos.numpy() if graph['eeg'].pos is not None else None,
    },
    "fmri_nodes": {
        "count":     int(graph['fmri'].num_nodes),
        "labels":    get_labels(graph['fmri'], graph['fmri'].num_nodes),
        "positions": graph['fmri'].pos.numpy() if graph['fmri'].pos is not None else None,
    },
    "eeg_edges": {
        "edge_index":  graph['eeg','connects','eeg'].edge_index.numpy(),
        "edge_weight": graph['eeg','connects','eeg'].edge_attr.squeeze(-1).numpy(),
    },
    "fmri_edges": {
        "edge_index":  graph['fmri','connects','fmri'].edge_index.numpy(),
        "edge_weight": graph['fmri','connects','fmri'].edge_attr.squeeze(-1).numpy(),
        "fc_matrix":   fc_matrix,   # [N_fmri, N_fmri]
    },
    # ── 时序信号 ─────────────────────────────────────────
    "eeg_timeseries": {
        "sampling_rate_hz": float(getattr(graph['eeg'], 'sampling_rate', 250.0)),
        "original":         eeg_original,            # [N_eeg, T]
        "reconstructed":    eeg_reconstructed,       # [N_eeg, T]
        "rmse_per_node":    eeg_rmse,                # [N_eeg]
    },
    "fmri_timeseries": {
        "sampling_rate_hz": float(getattr(graph['fmri'], 'sampling_rate', 0.5)),
        "original":         fmri_original,           # [N_fmri, T]
        "reconstructed":    fmri_reconstructed,      # [N_fmri, T]
        "rmse_per_node":    fmri_rmse,               # [N_fmri]
    },
    # ── 元数据 ───────────────────────────────────────────
    "subject_idx": int(graph.subject_idx.item()) if hasattr(graph, 'subject_idx') else None,
    "training_history": ckpt.get("history", {}),    # {"train_loss": [...], "val_loss": [...]} 或 {}
}

with open("visualization_data.json", "w") as f:
    json.dump(payload, f, cls=NumpyEncoder)
```

### 10.8 关键数组形状速查表（前端对接参考）

| 数组 | 形状 | 值域 | 用途 |
|------|------|------|------|
| `graph['eeg'].x` | `[N_eeg, T_eeg, 1]` | 标准化后约 ±3 | EEG 原始时序 |
| `graph['fmri'].x` | `[N_fmri, T_fmri, 1]` | 标准化后约 ±3 | fMRI 原始时序 |
| `reconstructed['eeg']` | `[N_eeg, T_eeg, 1]` | 同上 | 模型重建的 EEG |
| `reconstructed['fmri']` | `[N_fmri, T_fmri, 1]` | 同上 | 模型重建的 fMRI |
| `encoded['eeg']` | `[N_eeg, T_eeg, H]` | 无界（潜空间） | EEG 潜向量 |
| `encoded['fmri']` | `[N_fmri, T_fmri, H]` | 无界（潜空间） | fMRI 潜向量（含 EEG 跨模态信息） |
| `predictions['eeg']` | `[N_eeg, steps, H]` | 无界 | EEG 潜空间未来预测 |
| `predictions['fmri']` | `[N_fmri, steps, H]` | 无界 | fMRI 潜空间未来预测 |
| `edge_attr`（同模态） | `[E, 1]` | [0, 1] | Pearson \|r\| 功能连接强度 |
| `edge_attr`（跨模态） | `[E, 1]` | = 1.0 （每条边精确为 1.0）| 均匀权重（随机连接，无坐标配准） |
| `subject_embed.weight` | `[N_sub, H]` | 约 ±0.02（初始化） | 被试潜空间偏移 |
| FC matrix | `[N_fmri, N_fmri]` | [0, 1] | 功能连接矩阵（从稀疏边重建） |

> **去除最后维度的 1**：可视化时通常需要 `[N, T]` 而非 `[N, T, 1]`，用 `.squeeze(-1)` 或 `[..., 0]`。
