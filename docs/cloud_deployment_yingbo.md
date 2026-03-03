# TwinBrain V5 — 英博云部署指南

> 从零开始，到开机即训练的完整步骤手册。  
> 适用平台：**英博云**（[cloud.yingboai.com](https://cloud.yingboai.com)）

---

## 目录

1. [前提准备](#1-前提准备)  
2. [创建 GPU 集群 / 实例](#2-创建-gpu-集群--实例)  
3. [配置 Docker 镜像](#3-配置-docker-镜像)  
4. [挂载本地训练数据](#4-挂载本地训练数据)  
5. [推荐：本地预处理生成缓存图后上传（跳过云端数据预处理）](#5-推荐本地预处理生成缓存图后上传)  
6. [克隆代码并安装依赖](#6-克隆代码并安装依赖)  
7. [选择训练配置文件](#7-选择训练配置文件)  
8. [启动训练](#8-启动训练)  
9. [监控与查看结果](#9-监控与查看结果)  
10. [常见问题排查](#10-常见问题排查)  
11. [停止实例与费用管理](#11-停止实例与费用管理)  

---

## 1. 前提准备

在正式操作之前，确保你已准备好以下内容：

| 项目 | 说明 |
|------|------|
| 英博云账号 | 完成实名认证，账户余额充足（建议先充值 ≥ 200 元用于测试） |
| 训练数据 | EEG（`.set`）+ fMRI（`.nii.gz`），已按 BIDS 目录组织（见下方说明） |
| GitHub 访问 | 用于克隆 TwinBrain 代码仓库（或提前打包好的 `.zip`） |
| SSH 客户端 | Windows 推荐 MobaXterm / PuTTY；macOS/Linux 使用系统自带终端 |

### BIDS 数据目录结构（必须提前确认）

```
my_data/
├── sub-01/
│   ├── eeg/
│   │   └── sub-01_task-rest_eeg.set
│   └── func/
│       └── sub-01_task-rest_bold.nii.gz
├── sub-02/
│   ├── eeg/
│   │   └── sub-02_task-rest_eeg.set
│   └── func/
│       └── sub-02_task-rest_bold.nii.gz
└── ...
```

> **提示**：任务名（`task-rest`）需要在 EEG 和 fMRI 文件名中保持一致，
> 否则跨模态边无法建立。详见 `configs/default.yaml` 中的 `fmri_task_mapping` 注释。

---

## 2. 创建 GPU 集群 / 实例

### 2.1 登录控制台

打开 [cloud.yingboai.com](https://cloud.yingboai.com)，用账号密码或手机号登录。

### 2.2 进入算力市场

左侧导航栏 → **算力市场** → **GPU 云服务器**。

### 2.3 选择 GPU 规格

根据你的被试数量和预算选择：

| 场景 | 推荐 GPU | 显存 | 配置文件 | 预估费用 |
|------|----------|------|---------|---------|
| 快速测试（1-4 被试） | RTX 3080 / 4080 | 10-16 GB | `config_16gb.yaml` | 约 2-4 元/时 |
| 标准研究（4-16 被试） | RTX 3090 / 4090 | 24 GB | `config_32gb.yaml` | 约 4-6 元/时 |
| 大规模研究（16+ 被试） | A100 40G / 80G | 40-80 GB | `config_32gb.yaml`（可进一步放大） | 约 8-15 元/时 |
| 8 GB 限制（预算紧张） | RTX 3070 / 4060 Ti | 8 GB | `default.yaml` | 约 1-2 元/时 |

### 2.4 配置实例参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| **镜像** | PyTorch 2.1 + CUDA 12.1 | 见第 3 节详细说明 |
| **CPU 核心数** | 8-16 核 | 数据预处理（fMRI atlas 分区）为 CPU 密集型 |
| **内存** | 32-64 GB | fMRI `.nii.gz` 解压后可达 1-5 GB，需预留足够内存 |
| **系统盘** | 50 GB SSD | 存放代码 + 依赖 + 输出结果 |
| **数据盘** | 按需（100-500 GB） | 挂载训练数据（见第 4 节） |
| **计费方式** | 按量付费（调试阶段）；包天/包月（长期训练） | |

### 2.5 设置 SSH 密钥（强烈推荐）

在"登录凭证"处选择 **SSH 密钥**，上传你的公钥（`~/.ssh/id_rsa.pub`），避免密码登录的安全风险。

如果还没有 SSH 密钥对，在本地终端运行：

```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
# 按 Enter 使用默认路径和空密码
cat ~/.ssh/id_rsa.pub   # 复制输出内容，粘贴到英博云控制台
```

### 2.6 创建并启动实例

点击 **立即购买** → 确认配置信息 → **确认下单**。

实例创建通常需要 1-3 分钟，状态变为"运行中"后记录实例的 **公网 IP**。

---

## 3. 配置 Docker 镜像

英博云支持选择预置镜像或自定义镜像。TwinBrain V5 推荐使用**官方预置镜像**，
无需自行构建 Docker，只需在实例启动后执行依赖安装脚本。

### 3.1 推荐预置镜像

在创建实例时选择：

```
PyTorch → 2.1.0 → Python 3.10 → CUDA 12.1 → Ubuntu 20.04
```

> 英博云镜像名称示例（实际以控制台显示为准）：
> `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel`

此镜像已预装：
- CUDA 12.1 驱动
- PyTorch 2.1.0
- Python 3.10
- pip、conda

### 3.2 验证镜像可用性（SSH 登录后执行）

```bash
# 连接实例（将 <IP> 替换为实例公网 IP）
ssh root@<IP>

# 验证 GPU 可见
nvidia-smi

# 验证 PyTorch + CUDA
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

期望输出类似：
```
2.1.0  True  NVIDIA RTX 4090
```

### 3.3 可选：使用自定义镜像（高级用法）

如果你需要预装所有 TwinBrain 依赖（适合多次复用同一环境），可以将配置好的实例制作成快照并保存为自定义镜像：

1. 按本指南第 5 节完整安装所有依赖
2. 控制台 → **实例** → **操作** → **创建镜像**
3. 下次创建实例时选择此自定义镜像，跳过第 5 节的安装步骤

---

## 4. 挂载本地训练数据

训练数据（通常在你的本地电脑或机构服务器上）需要传输到英博云实例。
英博云提供两种主流方式，按数据量选择：

---

### 方式 A：英博云对象存储（推荐，数据量 > 10 GB）

#### Step 1：上传数据到对象存储（Bucket）

在英博云控制台找到 **对象存储（OSS）**：

1. 创建 Bucket：**新建** → 填写名称（如 `twinbrain-data`）→ 选择与实例**相同的地域** → 确认
2. 进入 Bucket → **上传文件** → 将本地 `my_data/` 目录整体上传

   或使用命令行工具（速度更快，适合大文件）。下载英博云 CLI 工具后：
   ```bash
   # 安装英博云 CLI（以 ossutil 为例，参考官方文档安装）
   # 上传整个数据目录（-r 递归，-u 增量更新）
   ossutil cp -r ./my_data oss://twinbrain-data/my_data/ -u
   ```

#### Step 2：在实例上挂载 Bucket

```bash
# 在实例上安装 ossfs 挂载工具
apt-get install -y fuse

# 下载 ossfs
# ⚠ 请到英博云控制台 → 文档 → 对象存储 → ossfs 工具下载，复制官方提供的实际 URL
# 下方 URL 为示例格式，不保证有效，必须替换为官方地址后再执行
OSSFS_DEB="ossfs_1.91.3_ubuntu20.04_amd64.deb"
wget -O /tmp/${OSSFS_DEB} "https://<英博云ossfs下载地址>/${OSSFS_DEB}"
dpkg -i /tmp/${OSSFS_DEB}

# 创建挂载点
mkdir -p /data

# 配置访问凭证（替换 <AccessKeyID> 和 <AccessKeySecret>）
echo "twinbrain-data:<AccessKeyID>:<AccessKeySecret>" > /etc/passwd-ossfs
chmod 640 /etc/passwd-ossfs

# 挂载（替换 <endpoint> 为所在地域 endpoint）
ossfs twinbrain-data /data -o url=http://oss-cn-<endpoint>.yingboai.com \
      -o allow_other -o umask=0022

# 验证
ls /data/my_data/sub-01/
```

#### Step 3：设置开机自动挂载（可选）

```bash
# 添加到 /etc/fstab（实例重启后自动挂载）
echo "twinbrain-data /data fuse.ossfs \
_netdev,allow_other,umask=0022,url=http://oss-cn-<endpoint>.yingboai.com 0 0" \
>> /etc/fstab
```

---

### 方式 B：SCP/SFTP 直接传输（适合数据量 < 10 GB）

在**本地终端**执行（将 `<IP>` 替换为实例 IP）：

```bash
# 传输整个数据目录（-r 递归，-C 压缩传输，-P 指定端口）
scp -r -C ./my_data root@<IP>:/data/my_data

# 或使用 rsync（支持断点续传，推荐）
rsync -avzP ./my_data/ root@<IP>:/data/my_data/
```

---

### 方式 C：英博云 NFS 共享存储（多实例场景）

若你需要多台 GPU 实例同时访问同一份数据（分布式训练），在控制台创建 **文件存储（NFS）**，
然后在每台实例上挂载：

```bash
# 安装 NFS 客户端
apt-get install -y nfs-common

# 创建挂载点
mkdir -p /data

# 挂载（替换 <nfs-server-ip> 和路径为控制台显示的值）
mount -t nfs <nfs-server-ip>:/exported/path /data

# 验证
df -h /data
```

---

### 数据路径确认

无论使用哪种方式，确认数据挂载正确后，目录结构应如下：

```
/data/my_data/
├── sub-01/
│   ├── eeg/
│   └── func/
└── sub-02/
    ├── eeg/
    └── func/
```

记住这个路径：`/data/my_data`，后续会用到。

---

## 5. 推荐：本地预处理生成缓存图后上传

> **这是云端训练最高效的方式。**  
> 数据预处理（atlas 分区、连通性估计）在本地一次性完成，上传轻量的 `.pt` 缓存图即可，
> 云端实例开机后直接跳过所有耗时的预处理步骤，立刻开始训练。

### 5.1 为什么推荐这种方式？

| 方式 | 云端启动训练耗时 | 需上传文件 | 需要云端安装 nilearn/MNE | 备注 |
|------|----------------|-----------|------------------------|------|
| 上传原始数据（EEG + fMRI） | 每次重启都需数分钟至数十分钟 | 原始文件（可达数十 GB） | ✅ 必须 | 标准方式 |
| **上传预处理缓存图** | **几乎为零（直接读取 .pt）** | **仅 .pt 文件（通常 < 1 GB）** | **❌ 不需要** | **推荐方式** |

### 5.2 在本地生成缓存图

确保本地已安装完整依赖（包括 nilearn、MNE），然后在本地运行一次训练：

```bash
# 在本地执行（使用 CPU 即可，只需运行到缓存生成完毕后 Ctrl+C 停止）
cd /path/to/twinbrain_new
python main.py --config configs/default.yaml
```

程序会：
1. 读取原始 EEG/fMRI 数据（仅一次）
2. 完成 atlas 分区、连通性估计
3. 将预处理结果保存为 `.pt` 缓存图（位于 `outputs/graph_cache/`）
4. 开始训练（此时可以 Ctrl+C 停止，缓存已保存）

### 5.3 确认缓存图文件

```bash
ls outputs/graph_cache/
# 期望输出（文件名格式：{被试ID}_{任务}_{8位哈希}.pt）：
# sub-01_rest_a1b2c3d4.pt
# sub-02_rest_a1b2c3d4.pt
# sub-03_GRADON_a1b2c3d4.pt
# ...

# 确认每个被试都有对应的缓存文件
ls outputs/graph_cache/ | wc -l  # 文件数应 = 被试数 × 任务数
```

> **重要**：同一套数据在相同配置下只需生成一次缓存图。
> 之后修改训练超参数（学习率、epoch 数、模型大小等）不会影响缓存图，无需重新生成。
> 只有修改以下配置才需要重新生成（会自动产生新哈希）：
> `atlas`、`graph` 拓扑参数、`eeg_connectivity_method`、`dti_structural_edges`。

### 5.4 将缓存图上传到云端实例

```bash
# 在本地终端执行（将 <IP> 替换为云端实例 IP）

# 方式 A：SCP 直接传输（简单，适合 < 5 GB）
scp -r outputs/graph_cache/ root@<IP>:/root/twinbrain_new/outputs/graph_cache/

# 方式 B：rsync（推荐，支持断点续传、增量更新）
rsync -avzP outputs/graph_cache/ root@<IP>:/root/twinbrain_new/outputs/graph_cache/

# 方式 C：先上传到 OSS，再在云端实例上下载（适合数据量大或多实例）
ossutil cp -r outputs/graph_cache oss://twinbrain-data/graph_cache/ -u
# 在云端实例上：
ossutil cp -r oss://twinbrain-data/graph_cache/ /root/twinbrain_new/outputs/graph_cache/ -u
```

### 5.5 云端配置：无需原始数据即可训练

TwinBrain V5 内置**缓存专用模式**：当 `root_dir` 下没有 `sub-*` 原始数据目录，
但 `cache.dir` 下已有 `.pt` 缓存文件时，程序会自动从缓存文件名解析被试和任务信息，
**完全跳过原始数据加载**，直接进入训练。

在云端实例上，只需确保以下配置正确（`root_dir` 可以指向一个不存在或空的目录）：

```yaml
# configs/default.yaml（云端版本）
data:
  root_dir: "/root/twinbrain_new/outputs/graph_cache"  # 可任意填写，不需要包含原始数据
  
  cache:
    enabled: true                                        # ← 必须为 true
    dir: "outputs/graph_cache"                          # ← 缓存图所在目录（与上传路径一致）
```

> 程序启动时会打印：  
> `[缓存专用模式] 原始数据目录 (...) 中未发现被试文件夹，从图缓存目录发现 N 个 (被试, 任务) 组合，将直接使用缓存图训练（无需原始 EEG/fMRI 文件）。`  
> 这是正常的，说明缓存专用模式已成功启用。

### 5.6 快速验证缓存专用模式

```bash
cd /root/twinbrain_new

# 快速验证：只跑 1 个 epoch，确认能读取缓存图并启动训练
python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('configs/default.yaml'))
cfg['training']['epochs'] = 1
yaml.dump(cfg, open('/tmp/test_cfg.yaml', 'w'), allow_unicode=True)
" && python3 main.py --config /tmp/test_cfg.yaml

# 期望输出包含：
# [缓存专用模式] 原始数据目录 (...) 中未发现被试文件夹 ...
# Epoch 1 | train_loss=X.XXXX ...
```

---

## 6. 克隆代码并安装依赖

### 6.1 克隆代码

```bash
# SSH 登录实例
ssh root@<IP>

# 切换到工作目录
cd /root

# 克隆 TwinBrain 仓库（替换为实际仓库地址）
git clone https://github.com/sheinclotho/twinbrain_new.git
cd twinbrain_new
```

### 6.2 安装 PyTorch Geometric（关键依赖，顺序不可乱）

PyTorch Geometric（PyG）需要与 PyTorch 版本严格匹配，必须先安装 PyTorch 再安装 PyG：

```bash
# 1. 确认当前 PyTorch + CUDA 版本
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"
# 示例输出：2.1.0  12.1

# 2. 安装 PyG 依赖（根据输出的版本号选择对应命令）
# === PyTorch 2.1.0 + CUDA 12.1 ===
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-geometric==2.3.1

# === PyTorch 2.0.x + CUDA 11.8（如果镜像版本不同，换成对应参数）===
# pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
# pip install torch-geometric==2.3.1
```

> **注意**：`torch-scatter` 和 `torch-sparse` 的安装地址中的版本号
> 必须与 `torch.__version__` 和 `torch.version.cuda` 完全一致，
> 否则会出现 `GLIBCXX` 或 `symbol not found` 错误。

### 6.3 安装其余依赖

```bash
pip install -r requirements.txt
```

### 6.4 安装 nilearn（fMRI atlas 分区，必需）

```bash
pip install nilearn

# 验证 atlas 可用
python3 -c "from nilearn import datasets; print('nilearn OK')"
```

### 6.5 安装 MNE（EEG 数据加载，必需）

```bash
pip install mne

# 验证
python3 -c "import mne; print('MNE', mne.__version__)"
```

### 6.6 完整一键安装脚本

以上步骤可合并为一个 shell 脚本，方便下次复用：

```bash
cat << 'EOF' > /root/setup_twinbrain.sh
#!/bin/bash
set -e

echo "=== TwinBrain V5 环境安装 ==="

# 检测 PyTorch 版本
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda.replace('.',''))")
echo "检测到：PyTorch $TORCH_VER, CUDA $CUDA_VER"

# 安装 PyG（根据版本自动选择）
pip install torch-scatter torch-sparse \
    -f "https://data.pyg.org/whl/torch-${TORCH_VER}+cu${CUDA_VER}.html"
pip install torch-geometric==2.3.1

# 安装其余依赖
cd /root/twinbrain_new
pip install -r requirements.txt
pip install nilearn mne

echo "=== 安装完成 ==="
python3 -c "
import torch
import torch_geometric
import nilearn
import mne
print(f'torch={torch.__version__}, pyg={torch_geometric.__version__}')
print(f'nilearn={nilearn.__version__}, mne={mne.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
EOF

chmod +x /root/setup_twinbrain.sh
bash /root/setup_twinbrain.sh
```

---

## 7. 选择训练配置文件

### 7.1 修改数据路径

打开配置文件，将 `root_dir` 改为你挂载的数据路径：

```bash
# 以 default.yaml 为例（适用于 8 GB GPU）
sed -i 's|root_dir: "F:/twinbrain_v3/test_file3"|root_dir: "/data/my_data"|g' \
    configs/default.yaml

# 验证修改
grep "root_dir" configs/default.yaml
```

### 7.2 按 GPU 显存选择配置文件

| GPU 显存 | 配置文件 | 说明 |
|---------|---------|------|
| 8 GB | `configs/default.yaml` | 默认配置，梯度检查点 + 分块已为 8GB 优化 |
| 16 GB | `configs/config_16gb.yaml` | 关闭梯度检查点，速度更快 |
| 24-32 GB | `configs/config_32gb.yaml` | 更大模型（H=256），更长预测视界 |
| 40+ GB (A100) | `configs/config_32gb.yaml` + 手动放大 | 可将 `hidden_channels` 改为 512 |

### 7.3 A100 / H100 进一步放大参数

```bash
# 创建 A100 专用配置（在 config_32gb.yaml 基础上覆盖）
cat << 'EOF' > configs/config_a100.yaml
# TwinBrain V5 — A100 80G 配置
model:
  hidden_channels: 512
  num_encoder_layers: 6
  prediction_steps: 50

training:
  use_gradient_checkpointing: false
  gradient_accumulation_steps: 1
  learning_rate: 0.00005

windowed_sampling:
  fmri_window_size: 150
  eeg_window_size: 1500

device:
  type: "cuda"
  gpu_memory: "80GB"
  use_amp: true
  use_torch_compile: false
EOF
```

### 7.4 编辑 root_dir（各配置文件均需修改）

如果你使用的是非 `default.yaml` 配置文件，由于它们是覆盖式配置，
`root_dir` 在 `default.yaml` 中定义，修改一次即可：

```bash
# 永久修改默认配置中的路径（推荐）
python3 -c "
import yaml, pathlib

cfg_path = 'configs/default.yaml'
cfg = yaml.safe_load(open(cfg_path))
cfg['data']['root_dir'] = '/data/my_data'
yaml.dump(cfg, open(cfg_path, 'w'), allow_unicode=True, default_flow_style=False)
print('root_dir 已更新为 /data/my_data')
"
```

---

## 8. 启动训练

### 8.1 首次启动（前台运行，验证环境）

```bash
cd /root/twinbrain_new

# 8 GB GPU（默认配置）
python3 main.py --config configs/default.yaml

# 16 GB GPU
python3 main.py --config configs/config_16gb.yaml

# 32 GB GPU
python3 main.py --config configs/config_32gb.yaml
```

**观察前几分钟的日志**，确认以下关键信息：

```
# 正常启动日志示例
[INFO] N_eeg = 63         ← EEG 节点数（通常 32-128）
[INFO] N_fmri = 190       ← fMRI ROI 数（应为 ~190，而非 1！）
[INFO] 图缓存未命中，正在构建...
[INFO] 开始训练，共 100 epochs
[INFO] Epoch 1 | train_loss=2.4513 | val_loss=2.8901 | r2_eeg=0.12 | r2_fmri=0.09
```

> ⚠️ **如果 `N_fmri = 1`**：说明 atlas 文件加载失败，检查 `atlases/` 目录是否存在。
>
> ```bash
> ls atlases/  # 应有 Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii 等文件
> ```

### 8.2 后台运行（长期训练推荐）

使用 `nohup` 或 `tmux`，防止 SSH 断开后训练中断：

#### 方式 A：tmux（推荐，可随时重连查看日志）

```bash
# 安装 tmux
apt-get install -y tmux

# 创建新会话
tmux new -s twinbrain

# 在 tmux 内启动训练
cd /root/twinbrain_new
python3 main.py --config configs/config_32gb.yaml 2>&1 | tee outputs/train.log

# 断开 tmux（训练继续在后台运行）：按 Ctrl+B，然后按 D

# 下次 SSH 登录后重连会话
tmux attach -t twinbrain
```

#### 方式 B：nohup（简单快捷）

```bash
cd /root/twinbrain_new
nohup python3 main.py --config configs/config_32gb.yaml \
    > outputs/train.log 2>&1 &

echo "训练 PID: $!"   # 记录进程号，用于后续停止

# 实时查看日志
tail -f outputs/train.log
```

### 8.3 从断点恢复训练

如果训练被中断（SSH 断开、实例重启），使用 `--resume` 参数继续：

```bash
# 自动从最新 checkpoint 继续
python3 main.py --config configs/config_32gb.yaml --resume

# 指定特定 checkpoint 路径继续
python3 main.py --config configs/config_32gb.yaml \
    --resume outputs/twinbrain_v5_20260301_120000/best_model.pt
```

---

## 9. 监控与查看结果

### 9.1 实时日志

```bash
# 查看最新日志
tail -f outputs/train.log

# 查看带时间戳的日志（搜索关键指标）
grep -E "Epoch|r2_eeg|r2_fmri|pred_r2|best" outputs/train.log | tail -50
```

### 9.2 TensorBoard 可视化

TwinBrain 训练过程会自动保存 TensorBoard 日志到 `outputs/` 目录。

```bash
# 在实例上启动 TensorBoard（端口 6006）
tensorboard --logdir=outputs/ --port=6006 --host=0.0.0.0 &

# 在英博云控制台 → 实例 → 安全组/防火墙，开放 6006 端口（入站规则）
# 然后在本地浏览器访问：http://<实例公网IP>:6006
```

> 也可以在本地通过 SSH 隧道访问（无需开放防火墙端口，更安全）：
> ```bash
> # 在本地终端运行
> ssh -L 6006:localhost:6006 root@<IP>
> # 然后访问 http://localhost:6006
> ```

### 9.3 训练结果文件

所有结果保存在 `outputs/twinbrain_v5_<时间戳>/` 目录下：

```
outputs/twinbrain_v5_20260301_120000/
├── best_model.pt          ← 验证集损失最低的模型权重
├── config.yaml            ← 本次训练使用的完整配置（含自动合并结果）
├── training.log           ← 完整训练日志
├── subject_to_idx.json    ← 被试编号映射（推理时必需）
├── results/
│   ├── predictions.npy    ← 预测结果（信号空间）
│   └── r2_scores.json     ← 各模态 R² 汇总
└── graph_cache/           ← 预处理后的图缓存（下次加速）
```

### 9.4 将结果下载到本地

```bash
# 从本地终端执行
scp -r root@<IP>:/root/twinbrain_new/outputs/twinbrain_v5_20260301_120000 ./

# 或使用 rsync（大文件推荐）
rsync -avzP root@<IP>:/root/twinbrain_new/outputs/ ./outputs/
```

### 9.5 关键训练指标速查

| 指标 | 研究可用标准 | 早期训练（< 20 epoch）正常范围 |
|------|------------|-------------------------------|
| `r2_eeg` | ≥ 0.30 | 0.05 – 0.30 |
| `r2_fmri` | ≥ 0.30 | 0.05 – 0.30 |
| `pred_r2_fmri` | ≥ 0.20（2-8 被试） | 0.05 – 0.15 |
| `pred_r2_eeg` | ≥ 0.10（物理上限约 0.20） | 0.01 – 0.08 |

> **R² < 0 是异常信号**：训练日志会自动打印 ⛔ 警告，常见原因是数据路径错误
> 或 atlas 加载失败（此时 N_fmri=1）。

---

## 10. 常见问题排查

### Q1：`N_fmri = 1`，只有 1 个 fMRI 节点

**原因**：atlas 文件未找到或 `nilearn` 未安装。

```bash
# 检查 atlas 文件
ls -la /root/twinbrain_new/atlases/

# 重新安装 nilearn
pip install nilearn

# 如果 atlas 文件丢失，重新下载
python3 -c "
from nilearn import datasets
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200)
print('Atlas 路径:', atlas.maps)
"
```

---

### Q2：CUDA OOM（显存不足）

```bash
# 查看 GPU 使用情况
watch -n 1 nvidia-smi

# 方案 1：切换到更保守的配置
python3 main.py --config configs/default.yaml  # 适合 8GB GPU

# 方案 2：减小隐层维度（修改 config.yaml）
sed -i 's/hidden_channels: 256/hidden_channels: 128/' configs/config_32gb.yaml

# 方案 3：减小窗口大小
# 在 configs/default.yaml 中将 fmri_window_size: 50 改为 fmri_window_size: 30
```

---

### Q3：`torch-scatter` 或 `torch-sparse` 安装失败

```bash
# 明确指定版本和下载源
TORCH=$(python3 -c "import torch; print(torch.__version__)")
# 只取 CUDA 主次版本号（如 12.1.0 → cu121，12.1 → cu121）
CUDA_SHORT=$(python3 -c "import torch; parts=torch.version.cuda.split('.')[:2]; print('cu'+''.join(parts))")
echo "检测到：PyTorch $TORCH  $CUDA_SHORT"

pip install --no-cache-dir torch-scatter \
    -f "https://data.pyg.org/whl/torch-${TORCH}+${CUDA_SHORT}.html"
pip install --no-cache-dir torch-sparse \
    -f "https://data.pyg.org/whl/torch-${TORCH}+${CUDA_SHORT}.html"
```

---

### Q4：SSH 断开后训练停止

```bash
# 确认使用 tmux（不受 SSH 断开影响）
tmux ls                  # 查看现有会话
tmux attach -t twinbrain # 重连
```

---

### Q5：数据挂载后文件读取很慢（OSS 网络延迟）

英博云 OSS 挂载适合读取大文件（`.nii.gz`），但频繁随机读写会较慢。
建议**预先将数据复制到本地 SSD**：

```bash
# 将数据从 OSS 挂载点复制到实例本地 SSD（系统盘或数据盘）
cp -r /data/my_data /root/my_data_local/

# 修改配置
python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/default.yaml'))
cfg['data']['root_dir'] = '/root/my_data_local'
yaml.dump(cfg, open('configs/default.yaml', 'w'), allow_unicode=True)
print('已切换到本地路径')
"
```

---

### Q6：图缓存（graph_cache）已存在但想重建

```bash
# 删除缓存，下次训练时自动重建
rm -rf /root/twinbrain_new/outputs/graph_cache/

# 或只删除特定被试的缓存
ls /root/twinbrain_new/outputs/graph_cache/
```

---

### Q7：训练日志中出现 `pred_r2 < 0` 警告

这是 ⛔ 信号，模型的预测质量比均值基线更差。常见原因：

1. **数据量不足**：建议至少 4 个被试
2. **训练轮数太少**：继续训练（`--resume`），至少 30-50 epoch 后再判断
3. **N_fmri = 1**：atlas 加载失败，见 Q1
4. **学习率过大**：将 `learning_rate` 从 `0.001` 降低至 `0.0003`

---

## 11. 停止实例与费用管理

### 10.1 保存工作并释放实例

```bash
# 训练完成后，先将结果同步到 OSS（防止实例删除后结果丢失）
ossutil cp -r /root/twinbrain_new/outputs oss://twinbrain-data/outputs/ -u

# 或直接 scp 到本地
# （在本地执行）scp -r root@<IP>:/root/twinbrain_new/outputs/ ./
```

### 10.2 按量计费：释放实例

若选择**按量付费**，训练完成后务必释放实例以停止计费：

控制台 → **实例** → 找到对应实例 → **操作** → **释放实例**

> ⚠️ 释放后实例和数据盘数据**不可恢复**，请务必提前下载结果。

### 10.3 包月/包天计费：关机但保留实例

若需要保留环境（如后续继续训练），可以选择**关机**而非释放：

控制台 → **实例** → **操作** → **停止**（关机后不产生 GPU 计费，但仍产生存储费用）

下次训练时重新启动实例，所有文件保持不变，直接跳到第 7 节。

---

## 快速参考：完整命令序列

以下提供两种常见工作流的精简命令序列。

### 工作流 A：仅上传缓存图（推荐，最快开始训练）

> **前提**：已在本地运行过 `python main.py` 生成了 `outputs/graph_cache/*.pt` 文件。

```bash
# ① 登录
ssh root@<IP>

# ② 克隆代码并安装依赖
git clone https://github.com/sheinclotho/twinbrain_new.git && cd twinbrain_new
bash /root/setup_twinbrain.sh  # 约 5-10 分钟（首次）

# ③ 上传缓存图（在本地执行）
rsync -avzP outputs/graph_cache/ root@<IP>:/root/twinbrain_new/outputs/graph_cache/

# ④ 确认缓存专用模式配置（cache.enabled=true 且 cache.dir 指向正确位置）
grep -A5 "cache:" /root/twinbrain_new/configs/default.yaml
# 确认输出包含: enabled: true

# ⑤ 启动训练（tmux 保护，防断线）
tmux new -s twinbrain
python3 main.py --config configs/config_32gb.yaml 2>&1 | tee outputs/train.log
# Ctrl+B 然后 D 退出 tmux，训练继续后台运行

# ⑥ 查看进度
tmux attach -t twinbrain   # 重连查看实时日志
# 日志中会出现：[缓存专用模式] 从图缓存目录发现 N 个 (被试, 任务) 组合 ...
```

### 工作流 B：上传原始数据（标准方式）

```bash
# ① 登录
ssh root@<IP>

# ② 克隆代码
git clone https://github.com/sheinclotho/twinbrain_new.git && cd twinbrain_new

# ③ 挂载数据（OSS 方式，参见第 4 节配置凭证）
mkdir -p /data
ossfs twinbrain-data /data -o url=http://oss-cn-<endpoint>.yingboai.com -o allow_other

# ④ 安装依赖（第一次运行时需要 5-10 分钟）
bash /root/setup_twinbrain.sh

# ⑤ 设置数据路径
python3 -c "
import yaml
cfg = yaml.safe_load(open('configs/default.yaml'))
cfg['data']['root_dir'] = '/data/my_data'
yaml.dump(cfg, open('configs/default.yaml', 'w'), allow_unicode=True)
"

# ⑥ 启动训练（tmux 保护，防断线）
tmux new -s twinbrain
python3 main.py --config configs/config_32gb.yaml 2>&1 | tee outputs/train.log
# Ctrl+B 然后 D 退出 tmux，训练继续后台运行

# ⑦ 查看进度
tmux attach -t twinbrain   # 重连查看实时日志
# 或
tail -f outputs/train.log
```

---

**版本**：V5 Cloud Deploy Guide v1.1 | **适用于**：英博云（cloud.yingboai.com）| **更新**：2026-03-03
