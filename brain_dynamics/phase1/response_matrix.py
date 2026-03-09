"""
phase1/response_matrix.py — 响应矩阵计算（可配置刺激节点）
============================================================

科学含义：
  "响应矩阵" R 是一个 [N × N_stim] 矩阵，其中 R[:, s] 代表
  "对节点 s 施加单位扰动后，所有节点的响应模式"。

  这是数字孪生脑的核心功能：通过计算响应矩阵，我们可以预测
  "刺激某个脑区后，其他脑区会如何响应"，等同于 TMS-fMRI 实验的数字仿真。

可配置参数（dynamics.yaml 中的 phase1.response_matrix.stimulation）：
  mode:
    "all"     — 刺激所有 N 个节点（最全面，计算 O(N) 次仿真）
    "sampled" — 随机采样 n_nodes 个节点（默认，快速探索）
    "indices" — 刺激指定节点列表
    "hubs"    — 自动选择度中心度最高的 n_nodes 个枢纽节点

  n_nodes: 当 mode="sampled" 或 mode="hubs" 时的刺激节点数量

注意：
  此模块之前版本将 n_nodes 硬编码为 10，无法从配置文件调整。
  V1.0 修复：完全由 dynamics.yaml 中的 stimulation 配置块控制。

References:
  - Deco et al. (2013). Resting brains never rest. J. Neurosci., 33:9499.
  - Huang et al. (2019). Measuring and interpreting neuronal oscillations.
    Neuron, 102:1157.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Import font configuration helper from spectral dynamics
try:
    from brain_dynamics.spectral_dynamics.e1_spectral_analysis import (
        _configure_matplotlib_fonts,
    )
except ImportError:
    def _configure_matplotlib_fonts(use_latex_math: bool = True) -> None:  # type: ignore[misc]
        pass

# ─────────────────────────────────────────────────────────────────────────────
# 刺激节点选择
# ─────────────────────────────────────────────────────────────────────────────

def select_stimulation_nodes(
    fc: np.ndarray,
    mode: str,
    n_nodes: int = 10,
    node_indices: Optional[List[int]] = None,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """根据配置选择要刺激的节点集合。

    Args:
        fc: [N, N] 功能连接矩阵（用于 "hubs" 模式）。
        mode: 刺激模式："all" | "sampled" | "indices" | "hubs"。
        n_nodes: 当 mode 为 "sampled" 或 "hubs" 时的节点数量。
        node_indices: 当 mode="indices" 时，指定的节点索引列表。
        seed: 随机种子（用于 "sampled" 模式的可复现性）。

    Returns:
        stim_nodes: 排序后的节点索引数组（1D，dtype=int）。

    Raises:
        ValueError: 参数组合不合法（如 mode="indices" 但未提供 node_indices）。
    """
    N = fc.shape[0]

    if mode == "all":
        stim_nodes = np.arange(N)
        logger.info(f"刺激模式: all — 刺激全部 {N} 个节点")

    elif mode == "sampled":
        if n_nodes <= 0:
            raise ValueError(f"n_nodes 必须为正整数，当前值: {n_nodes}")
        if n_nodes > N:
            logger.warning(
                f"n_nodes={n_nodes} 超过总节点数 N={N}，将改为刺激全部节点。"
            )
            n_nodes = N
        rng = np.random.default_rng(seed)
        stim_nodes = np.sort(rng.choice(N, size=n_nodes, replace=False))
        logger.info(
            f"刺激模式: sampled — 随机采样 {n_nodes} 个节点（seed={seed}）: "
            f"{stim_nodes[:5].tolist()}{'...' if n_nodes > 5 else ''}"
        )

    elif mode == "indices":
        if node_indices is None or len(node_indices) == 0:
            raise ValueError(
                "mode='indices' 时必须提供非空的 node_indices 列表。"
            )
        stim_nodes = np.sort(np.asarray(node_indices, dtype=int))
        out_of_range = stim_nodes[stim_nodes >= N]
        if len(out_of_range) > 0:
            raise ValueError(
                f"node_indices 中存在超出范围的索引: {out_of_range.tolist()}。"
                f"有效范围: [0, {N-1}]。"
            )
        logger.info(f"刺激模式: indices — 刺激 {len(stim_nodes)} 个指定节点")

    elif mode == "hubs":
        if n_nodes <= 0:
            raise ValueError(f"n_nodes 必须为正整数，当前值: {n_nodes}")
        if n_nodes > N:
            n_nodes = N
        # 度中心度 = 各节点的连接强度总和（绝对值）
        degree = np.sum(np.abs(fc), axis=1)
        hub_indices = np.argsort(degree)[::-1][:n_nodes]
        stim_nodes = np.sort(hub_indices)
        logger.info(
            f"刺激模式: hubs — 选择度中心度最高的 {n_nodes} 个枢纽节点: "
            f"{stim_nodes[:5].tolist()}{'...' if n_nodes > 5 else ''}"
        )

    else:
        raise ValueError(
            f"未知的刺激模式: '{mode}'。支持: 'all', 'sampled', 'indices', 'hubs'。"
        )

    return stim_nodes


# ─────────────────────────────────────────────────────────────────────────────
# 响应矩阵计算（基于 FC 线性传播）
# ─────────────────────────────────────────────────────────────────────────────

def compute_response_matrix_linear(
    fc: np.ndarray,
    stim_nodes: np.ndarray,
    perturbation_amplitude: float = 1.0,
    alpha: float = 0.2,
) -> np.ndarray:
    """使用 FC 线性传播模型计算响应矩阵。

    模型：response = (I - α × W)⁻¹ × δ
    其中 W 为 FC 矩阵（行归一化），α 为传播衰减系数，δ 为单位扰动向量。

    物理直觉：
      每个脑区通过功能连接向邻近脑区传播信号。
      (I - αW)⁻¹ 是矩阵的 "Green's function"，捕捉所有阶次的间接影响。
      这等同于几何级数 I + αW + α²W² + ... （当 α‖W‖ < 1 时收敛）。

    Args:
        fc: [N, N] 功能连接矩阵。
        stim_nodes: 要刺激的节点索引数组。
        perturbation_amplitude: 扰动幅度（标准差单位）。
        alpha: 传播衰减系数。建议范围 [0.05, 0.3]。

    Returns:
        response_matrix: [N, N_stim] 响应矩阵。
                         response_matrix[:, k] = 对第 stim_nodes[k] 个节点施加扰动后的全脑响应。

    Scientific Notes:
        线性传播是一阶近似（忽略神经元的非线性和 HRF 卷积）。
        对于 Pearson FC（值域 [-1, 1]），α ≤ 0.3 保证矩阵级数收敛。
        Ref: Honey et al. (2009), PNAS 106(6):2035.
    """
    N = fc.shape[0]
    N_stim = len(stim_nodes)

    # 行归一化 FC（避免强连接节点的数值不稳定）
    row_sum = np.abs(fc).sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    W = fc / row_sum  # 行归一化 FC

    # 检查稳定性：α × 谱半径 < 1
    spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
    if alpha * spectral_radius >= 1.0:
        alpha_safe = 0.9 / (spectral_radius + 1e-8)
        logger.warning(
            f"传播系数 α={alpha:.3f} × 谱半径={spectral_radius:.3f} ≥ 1，"
            f"矩阵级数不收敛（(I-αW) 奇异）。"
            f"自动调整 α={alpha_safe:.4f} 以保证稳定性。"
            f"若要保持 α={alpha:.3f}，请在配置中设置 linear_propagation_alpha ≤ {alpha_safe:.4f}。"
        )
        alpha = alpha_safe

    # 计算 (I - αW)⁻¹
    M = np.eye(N) - alpha * W
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        logger.warning("(I - αW) 矩阵奇异，使用伪逆。")
        M_inv = np.linalg.pinv(M)

    # 构造扰动向量并计算响应
    response_matrix = np.zeros((N, N_stim), dtype=np.float32)
    for k, node_idx in enumerate(stim_nodes):
        delta = np.zeros(N)
        delta[node_idx] = perturbation_amplitude
        response = M_inv @ delta
        response_matrix[:, k] = response.astype(np.float32)

    # 归一化：减去自发响应（本底），只保留差值
    # baseline ≈ M_inv @ 0 = 0，这里的 response 已经是净响应
    return response_matrix


# ─────────────────────────────────────────────────────────────────────────────
# 响应矩阵可视化
# ─────────────────────────────────────────────────────────────────────────────

def plot_response_matrix(
    response_matrix: np.ndarray,
    stim_nodes: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    roi_labels: Optional[List[str]] = None,
    dpi: int = 120,
) -> plt.Figure:
    """绘制响应矩阵热图。

    Args:
        response_matrix: [N, N_stim] 响应矩阵。
        stim_nodes: 刺激节点索引数组（用于标注 x 轴）。
        output_path: 保存路径。
        roi_labels: ROI 标签列表（可选）。
        dpi: 图像 DPI。
    """
    N, N_stim = response_matrix.shape
    fig_width = max(8, min(N_stim * 0.4, 20))
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    vmax = np.percentile(np.abs(response_matrix), 95)
    im = ax.imshow(
        response_matrix,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="响应强度")

    ax.set_title(
        f"响应矩阵 (刺激 {N_stim} 个节点)",
        fontsize=12,
    )
    ax.set_xlabel("刺激节点索引", fontsize=11)
    ax.set_ylabel("目标节点索引", fontsize=11)

    # X 轴标注刺激节点编号
    tick_step = max(1, N_stim // 20)
    x_ticks = np.arange(0, N_stim, tick_step)
    ax.set_xticks(x_ticks)
    x_labels = [str(stim_nodes[i]) for i in x_ticks]
    ax.set_xticklabels(x_labels, fontsize=8, rotation=45)

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"保存: {output_path}")

    return fig


def plot_hub_influence(
    response_matrix: np.ndarray,
    stim_nodes: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    roi_labels: Optional[List[str]] = None,
    dpi: int = 120,
) -> plt.Figure:
    """绘制每个刺激节点的全脑影响力排行图。

    Args:
        response_matrix: [N, N_stim] 响应矩阵。
        stim_nodes: 刺激节点索引数组。
        output_path: 保存路径。
        roi_labels: ROI 标签列表（可选）。
        dpi: 图像 DPI。
    """
    # 影响力 = 各刺激节点响应的 L2 范数
    influence = np.linalg.norm(response_matrix, axis=0)  # [N_stim]
    sorted_idx = np.argsort(influence)[::-1]

    N_stim = len(stim_nodes)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(N_stim), influence[sorted_idx], color="#4CAF50", alpha=0.8)

    x_labels = [
        roi_labels[stim_nodes[i]] if roi_labels else str(stim_nodes[i])
        for i in sorted_idx
    ]
    tick_step = max(1, N_stim // 20)
    ax.set_xticks(range(0, N_stim, tick_step))
    ax.set_xticklabels(x_labels[::tick_step], rotation=45, fontsize=8)
    ax.set_xlabel("刺激节点", fontsize=11)
    ax.set_ylabel(r"影响力 $\|\mathbf{r}\|_2$", fontsize=11)
    ax.set_title("各刺激节点的全脑影响力排序", fontsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"保存: {output_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 主分析函数
# ─────────────────────────────────────────────────────────────────────────────

def run_response_matrix_analysis(
    fc: np.ndarray,
    config: Dict,
    output_dir: Union[str, Path],
    roi_labels: Optional[List[str]] = None,
) -> Dict:
    """运行完整的响应矩阵分析流程。

    Args:
        fc: [N, N] 功能连接矩阵。
        config: dynamics.yaml 中的 phase1 配置块。
        output_dir: 输出根目录。
        roi_labels: ROI 标签列表（可选）。

    Returns:
        results: 包含 response_matrix、stim_nodes 的字典。
    """
    output_dir = Path(output_dir) / "structure"
    output_dir.mkdir(parents=True, exist_ok=True)

    rm_cfg = config.get("response_matrix", {})
    if not rm_cfg.get("enabled", True):
        return {}

    stim_cfg = rm_cfg.get("stimulation", {})
    out_cfg = config.get("output", {})
    dpi = out_cfg.get("dpi", 120)
    fmt = out_cfg.get("figure_format", "png")

    # 1. 选择刺激节点
    stim_nodes = select_stimulation_nodes(
        fc=fc,
        mode=stim_cfg.get("mode", "sampled"),
        n_nodes=stim_cfg.get("n_nodes", 10),
        node_indices=stim_cfg.get("node_indices"),
        seed=stim_cfg.get("seed", 42),
    )

    # 2. 计算响应矩阵
    logger.info(
        f"计算响应矩阵（刺激 {len(stim_nodes)} 个节点）..."
    )
    response_matrix = compute_response_matrix_linear(
        fc=fc,
        stim_nodes=stim_nodes,
        perturbation_amplitude=rm_cfg.get("perturbation_amplitude", 1.0),
        alpha=rm_cfg.get("linear_propagation_alpha", 0.2),
    )

    # 3. 保存
    rm_path = output_dir / "response_matrix.npy"
    np.save(rm_path, response_matrix)
    logger.info(f"响应矩阵已保存: {rm_path}，形状: {response_matrix.shape}")

    # 4. 可视化
    plot_response_matrix(
        response_matrix=response_matrix,
        stim_nodes=stim_nodes,
        output_path=output_dir / f"response_matrix.{fmt}",
        roi_labels=roi_labels,
        dpi=dpi,
    )

    plot_hub_influence(
        response_matrix=response_matrix,
        stim_nodes=stim_nodes,
        output_path=output_dir / f"hub_influence.{fmt}",
        roi_labels=roi_labels,
        dpi=dpi,
    )

    return {
        "response_matrix": response_matrix,
        "stim_nodes": stim_nodes,
    }
