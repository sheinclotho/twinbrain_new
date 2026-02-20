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

## 进阶：调整参数

如果你对机器学习有一定了解，可以在 `configs/default.yaml` 中调整：

| 参数 | 位置 | 说明 |
|------|------|------|
| `num_epochs: 100` | training | 训练轮数，越多越慢但效果可能更好 |
| `learning_rate: 0.0001` | training | 学习率，一般不需要修改 |
| `hidden_channels: 128` | model | 模型大小，内存不足时改为 64 |
| `max_subjects: 0` | data | 0 表示用全部数据；改为 10 可快速测试 |

---

**版本**：V5.0 | **更新**：2026-02-20
