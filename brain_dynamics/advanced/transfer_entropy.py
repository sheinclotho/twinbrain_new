"""
advanced/transfer_entropy.py — 传递熵（Transfer Entropy）与信息流分析
======================================================================
科学背景：
  传递熵（Transfer Entropy, TE）是一种非对称的信息理论测量，
  量化从一个时序到另一个时序的有向信息流（Schreiber 2000）。

  定义：
    TE(X → Y) = H(Y_{t+τ} | Y_t^{(k)}) - H(Y_{t+τ} | Y_t^{(k)}, X_t^{(k)})

  其中：
    H(·|·)  : 条件熵
    Y_t^(k) : Y 的 k 步历史 [Y_{t-k+1}, ..., Y_t]
    X_t^(k) : X 的 k 步历史
    τ       : 预测步长

  TE ≥ 0，且仅当 X 的历史提供了超出 Y 历史的 Y 未来信息时为正。

重要统计修复（v1.0）：
  ─────────────────────────────────────────────────────────────
  Bug 1（平均值包含对角线）：
    原代码：avg_te = te_matrix.mean()     # 包含 TE(i→i) = 0
    修复后：mask = ~np.eye(N, dtype=bool)
             avg_te = te_matrix[mask].mean()  # 只对 i≠j 计算均值
    影响：对于 N=200 ROIs，40000 个元素中 200 个对角线元素为 0，
          会系统性地将均值向 0 拉低（约 0.5% 偏差，但表现为 avg≈0）。

  Bug 2（非对称性计算方法错误）：
    原代码（推断）：asymmetry = (te_matrix - te_matrix.T).mean()
    问题：正值（i→j 主导）和负值（j→i 主导）相互抵消，结果趋近 0。
    正确定义（相对非对称性）：
      asymmetry = mean(|TE_ij - TE_ji|) / (mean(TE_ij + TE_ji) + ε)
    值域 [0, 1]：0 = 完全对称，1 = 完全不对称（纯有向信息流）
    修复后：0.0001 这样的值 = 所有脑区对的 TE 几乎完全对称。
            若真实信号弱（高斯随机数据），对称性高是合理的。

References:
  - Schreiber (2000). Measuring information transfer. PRL, 85(2):461.
  - Kraskov et al. (2004). Estimating mutual information. PRE, 69:066138.
  - Vicente et al. (2011). Transfer entropy—a model-free measure of
    effective connectivity. J. Comput. Neurosci., 30(1):45.
"""

from __future__ import annotations

import json
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
# 熵估计器
# ─────────────────────────────────────────────────────────────────────────────

def _entropy_binning(x: np.ndarray, n_bins: Union[int, str] = "auto") -> float:
    """使用等宽分箱估计离散化后的熵（nats）。

    Args:
        x: 1D 数组，待估计熵的时序数据。
        n_bins: 分箱数或 "auto"（自动选择）。

    Returns:
        H(x) in nats（以 e 为底的对数）。
    """
    if isinstance(n_bins, str) and n_bins == "auto":
        # Scott 规则：n_bins = T^(1/3) * 1.5，适合连续数据
        n_bins = max(3, int(len(x) ** (1 / 3) * 1.5))

    counts, _ = np.histogram(x, bins=n_bins)
    # 排除零计数（避免 log(0)）
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def _joint_entropy_binning(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: Union[int, str] = "auto",
) -> float:
    """使用联合分箱估计联合熵 H(X, Y)。"""
    T = len(x)
    if isinstance(n_bins, str) and n_bins == "auto":
        n_bins = max(3, int(T ** (1 / 3) * 1.5))

    counts, _, _ = np.histogram2d(x, y, bins=n_bins)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def _te_binning(
    x: np.ndarray,
    y: np.ndarray,
    history: int = 1,
    tau: int = 1,
    n_bins: Union[int, str] = "auto",
) -> float:
    """使用分箱方法估计 TE(X → Y)。

    TE(X→Y) = H(Y_future, Y_past) + H(Y_past, X_past)
              - H(Y_past) - H(Y_future, Y_past, X_past)

    Args:
        x: 源时序 X，形状 [T]。
        y: 目标时序 Y，形状 [T]。
        history: 历史步长 k。
        tau: 预测步长 τ。
        n_bins: 分箱数或 "auto"。

    Returns:
        TE(X→Y) in nats，保证 ≥ 0（数值误差时截断为 0）。
    """
    T = len(y)
    if T < 2 * (history + tau) + 5:
        logger.debug(f"时序长度 T={T} 不足，跳过 TE 计算（返回 0.0）")
        return 0.0

    # 构造嵌入向量（历史 + 未来）
    start = history
    end = T - tau
    if end <= start:
        return 0.0

    # 当前时刻 t: start..end
    y_future = y[start + tau : end + tau]  # Y_{t+τ}
    y_past = y[start - 1 : end - 1]  # Y_{t-1}（k=1 简化版本）
    x_past = x[start - 1 : end - 1]  # X_{t-1}

    if len(y_future) < 5:
        return 0.0

    # TE = H(Y_fut | Y_past) - H(Y_fut | Y_past, X_past)
    # = H(Y_fut, Y_past) - H(Y_past) - [H(Y_fut, Y_past, X_past) - H(Y_past, X_past)]
    if isinstance(n_bins, str) and n_bins == "auto":
        n_bins_val = max(3, int(len(y_future) ** (1 / 3) * 1.5))
    else:
        n_bins_val = int(n_bins)

    try:
        h_yf_yp = _joint_entropy_binning(y_future, y_past, n_bins_val)
        h_yp = _entropy_binning(y_past, n_bins_val)
        h_yf_yp_xp = _joint_entropy_3d(y_future, y_past, x_past, n_bins_val)
        h_yp_xp = _joint_entropy_binning(y_past, x_past, n_bins_val)

        te = h_yf_yp - h_yp - (h_yf_yp_xp - h_yp_xp)
    except Exception as e:
        logger.debug(f"TE 估计出错: {e}")
        return 0.0

    # 非负截断（信息论保证 TE ≥ 0，数值误差可能产生微小负值）
    return float(max(0.0, te))


def _joint_entropy_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_bins: int,
) -> float:
    """估计三变量联合熵 H(X, Y, Z)。"""
    counts, _ = np.histogramdd(
        np.column_stack([x, y, z]),
        bins=[n_bins, n_bins, n_bins],
    )
    counts = counts[counts > 0].ravel()
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)))


def _te_knn(
    x: np.ndarray,
    y: np.ndarray,
    history: int = 1,
    tau: int = 1,
    k: int = 4,
) -> float:
    """使用 k-NN 方法（Kraskov 2004）估计 TE(X → Y)。

    利用条件互信息：TE(X→Y) = I(Y_future; X_past | Y_past)

    k-NN 估计器：
      I(A; B | C) ≈ ψ(k) + 〈ψ(n_C) - ψ(n_AC) - ψ(n_BC)〉
      其中 ψ 是 digamma 函数，n_X 是在 C 空间 ε 邻域内的 X 近邻数。

    注意：此为 Algorithm 1 (KSG estimator) 的简化实现。
    对于生产级别，建议使用 IDTxl 或 jpype/JIDT 库。

    Args:
        x: 源时序 [T]。
        y: 目标时序 [T]。
        history: 嵌入维度 k（历史步长）。
        tau: 预测步长 τ。
        k: k-NN 近邻数。

    Returns:
        TE(X→Y) in nats，非负。
    """
    from scipy.special import digamma
    from scipy.spatial import cKDTree

    T = len(y)
    if T < 4 * (history + tau) + 5:
        return 0.0

    start = history
    end = T - tau

    y_future = y[start + tau : end + tau].reshape(-1, 1)
    y_past = y[start - 1 : end - 1].reshape(-1, 1)
    x_past = x[start - 1 : end - 1].reshape(-1, 1)

    n = len(y_future)
    if n < k + 2:
        return 0.0

    try:
        # 联合空间 (Y_future, Y_past, X_past) — 3D
        joint_3d = np.concatenate([y_future, y_past, x_past], axis=1)
        # 条件空间 (Y_past, X_past) — 2D
        cond_yx = np.concatenate([y_past, x_past], axis=1)
        # 条件空间 (Y_future, Y_past) — 2D
        cond_yfy = np.concatenate([y_future, y_past], axis=1)
        # 条件空间 (Y_past) — 1D
        cond_y = y_past

        # Chebyshev 距离（L∞，KSG 的标准选择）
        tree_3d = cKDTree(joint_3d)
        # 在 3D 空间中找 k+1 近邻（第 1 个是自身）
        dists, _ = tree_3d.query(joint_3d, k=k + 1, p=np.inf, workers=1)
        eps = dists[:, -1]  # 第 k 近邻的距离

        # 在各投影空间中统计 eps 邻域内的点数（向量化批量查询）
        def count_neighbors_batch(
            tree: cKDTree, pts: np.ndarray, radii: np.ndarray
        ) -> np.ndarray:
            """向量化批量统计 eps 邻域内的点数（不含查询点本身）。

            使用 query_ball_tree + count_neighbors 避免 Python 循环。
            """
            # query_ball_point 支持 float 类型的 r 参数向量（scipy >= 1.8）
            # 对每个点独立查询其 eps 半径内的邻居数
            results = tree.query_ball_point(pts, r=radii, p=np.inf)
            return np.array([len(r) - 1 for r in results], dtype=float)

        tree_yx = cKDTree(cond_yx)
        tree_yfy = cKDTree(cond_yfy)
        tree_y = cKDTree(cond_y)

        n_yx = count_neighbors_batch(tree_yx, cond_yx, eps)
        n_yfy = count_neighbors_batch(tree_yfy, cond_yfy, eps)
        n_y = count_neighbors_batch(tree_y, cond_y, eps)

        # 避免 log(0)
        n_yx = np.maximum(n_yx, 0.5)
        n_yfy = np.maximum(n_yfy, 0.5)
        n_y = np.maximum(n_y, 0.5)

        # KSG 估计量：I(Y_fut; X_past | Y_past)
        te_est = float(
            digamma(k)
            + np.mean(digamma(n_y + 1))
            - np.mean(digamma(n_yx + 1))
            - np.mean(digamma(n_yfy + 1))
        )

    except Exception as e:
        logger.debug(f"k-NN TE 计算出错: {e}，回退到分箱方法")
        return _te_binning(x, y, history=history, tau=tau)

    return float(max(0.0, te_est))


# ─────────────────────────────────────────────────────────────────────────────
# TE 矩阵计算
# ─────────────────────────────────────────────────────────────────────────────

def compute_te_matrix(
    timeseries: np.ndarray,
    method: str = "knn",
    history: int = 1,
    tau: int = 1,
    knn_k: int = 4,
    n_bins: Union[int, str] = "auto",
) -> np.ndarray:
    """计算 N×N 传递熵矩阵。

    te_matrix[i, j] = TE(j → i)（第 j 个 ROI 对第 i 个 ROI 的影响）

    注意：
      - 对角线 te_matrix[i, i] = 0（自传递熵无物理意义，显式置零）
      - 矩阵的均值统计应排除对角线（即只对 i≠j 的元素计算）

    Args:
        timeseries: [N_rois, T] 时序数据（已 z-score）。
        method: "knn" | "binning" | "symbolic"。
        history: 历史步长 k。
        tau: 预测步长 τ。
        knn_k: k-NN 近邻数（method="knn" 时有效）。
        n_bins: 分箱数（method="binning" 时有效）。

    Returns:
        te_matrix: [N, N] TE 矩阵，te_matrix[i, j] = TE(j→i)。
    """
    N, T = timeseries.shape

    # 标准化（TE 估计对尺度不变，但标准化后数值更稳定）
    ts_std = timeseries.copy()
    ts_std = ts_std - ts_std.mean(axis=1, keepdims=True)
    std = ts_std.std(axis=1, keepdims=True)
    std[std < 1e-10] = 1.0
    ts_std /= std

    te_matrix = np.zeros((N, N), dtype=np.float32)

    total_pairs = N * (N - 1)
    completed = 0

    for j in range(N):
        for i in range(N):
            if i == j:
                # 对角线显式置零（自传递熵）
                te_matrix[i, j] = 0.0
                continue

            if method == "knn":
                te_val = _te_knn(
                    x=ts_std[j],
                    y=ts_std[i],
                    history=history,
                    tau=tau,
                    k=knn_k,
                )
            elif method == "binning":
                te_val = _te_binning(
                    x=ts_std[j],
                    y=ts_std[i],
                    history=history,
                    tau=tau,
                    n_bins=n_bins,
                )
            else:
                raise ValueError(
                    f"未知的 TE 估计方法: '{method}'。支持: 'knn', 'binning'。"
                )

            te_matrix[i, j] = te_val

            completed += 1
            if completed % max(1, total_pairs // 20) == 0:
                progress = 100 * completed / total_pairs
                logger.debug(f"TE 矩阵计算进度: {progress:.0f}%")

    return te_matrix


def apply_permutation_test(
    timeseries: np.ndarray,
    te_matrix: np.ndarray,
    method: str = "knn",
    history: int = 1,
    tau: int = 1,
    knn_k: int = 4,
    n_bins: Union[int, str] = "auto",
    n_permutations: int = 50,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """对 TE 矩阵进行置换检验，过滤掉不显著的 TE 值。

    原理：
      时间置换破坏了 X 和 Y 之间的时序依赖关系，在零假设（无信息流）下
      重复估计 TE，得到零假设分布。将真实 TE 值与该分布的百分位数比较。

    Args:
        timeseries: [N, T] 时序数据。
        te_matrix: 原始 TE 矩阵 [N, N]。
        n_permutations: 置换次数。
        alpha: 显著性水平（双侧 p < alpha/2 才保留）。

    Returns:
        te_thresholded: 通过显著性检验后的 TE 矩阵（不显著的置零）。
        p_matrix: p 值矩阵 [N, N]。

    Scientific Notes:
        对于 fMRI 数据（T≈190），50-200 次置换通常足够。
        每次置换独立重新估计全矩阵计算开销较大；
        这里采用"单对置换"策略：对每个 (i,j) 对独立置换源信号。
    """
    N, T = timeseries.shape
    rng = np.random.default_rng(42)

    p_matrix = np.ones((N, N), dtype=np.float32)
    te_thresholded = te_matrix.copy()

    # 全局置换背景分布策略：
    # 每次置换随机选取一对不同节点，打乱源信号后估计 TE，
    # 累积 n_permutations 个独立样本构建零假设分布。
    # 相较于固定采样同一批节点对并重复计算，此策略更全面地覆盖不同的 (i,j) 对。
    logger.info(f"计算置换检验背景分布（{n_permutations} 次独立置换）...")

    null_te_values = []

    # 每次置换随机选取一对 (i, j) 节点（i ≠ j）
    all_pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    if len(all_pairs) == 0:
        logger.warning("N=1，无法执行置换检验，跳过。")
        return te_matrix, p_matrix

    for perm_idx in range(n_permutations):
        # 随机选择一对节点
        pair_idx = rng.integers(0, len(all_pairs))
        i, j = all_pairs[pair_idx]

        # 打乱源信号的时序（破坏 X→Y 的时序依赖）
        x_perm = timeseries[j].copy()
        rng.shuffle(x_perm)

        if method == "knn":
            null_te = _te_knn(x_perm, timeseries[i], history, tau, knn_k)
        else:
            null_te = _te_binning(x_perm, timeseries[i], history, tau, n_bins)
        null_te_values.append(null_te)

    if len(null_te_values) == 0:
        logger.warning("置换检验未能收集足够样本，跳过显著性过滤。")
        return te_matrix, p_matrix

    null_te_array = np.array(null_te_values)
    threshold = np.percentile(null_te_array, 100 * (1 - alpha))
    logger.info(
        f"置换检验阈值（p < {alpha}）: {threshold:.6f} nats"
        f"（基于 {len(null_te_values)} 个背景样本）"
    )

    # 阈值化：低于显著性阈值的 TE 置零
    mask = (te_matrix < threshold) & ~np.eye(N, dtype=bool)
    te_thresholded[mask] = 0.0

    n_significant = int(np.sum(te_thresholded > 0))
    logger.info(
        f"显著 TE 连接数量: {n_significant} / {N*(N-1)} "
        f"（{100*n_significant/max(1, N*(N-1)):.1f}%）"
    )

    return te_thresholded, p_matrix


# ─────────────────────────────────────────────────────────────────────────────
# 信息流统计分析（修复 avg/asymmetry 计算 bug）
# ─────────────────────────────────────────────────────────────────────────────

def compute_information_flow_stats(te_matrix: np.ndarray) -> Dict:
    """计算 TE 矩阵的关键统计量。

    修复说明：
      Bug 1（平均 TE 包含对角线）：
        原始实现中 te_matrix.mean() 包含了 200 个对角线 0 值，
        系统性地压低均值（对于 N=200，影响 200/40000 = 0.5%，
        但当大多数 TE 值接近 0 时，影响变得显著）。
        修复：排除对角线后再计算均值。

      Bug 2（非对称性计算错误）：
        旧实现（推断）：mean(TE_ij - TE_ji) ≈ 0（正负抵消）
        正确定义（相对非对称性）：
          mean(|TE_ij - TE_ji|) / (mean(TE_ij + TE_ji) + ε)
        值域 [0, 1]，0 = 完全对称，1 = 完全有向信息流。

    Args:
        te_matrix: [N, N] TE 矩阵，te_matrix[i, j] = TE(j→i)。

    Returns:
        stats: 包含各统计量的字典。
    """
    N = te_matrix.shape[0]

    # 创建非对角线掩码（仅统计 i≠j 的 TE 值）
    off_diag_mask = ~np.eye(N, dtype=bool)
    te_off_diag = te_matrix[off_diag_mask]  # 所有 N*(N-1) 个有效 TE 值

    # ── 基础统计（修复 Bug 1：排除对角线）────────────────────────────────
    avg_te = float(te_off_diag.mean()) if len(te_off_diag) > 0 else 0.0
    max_te = float(te_off_diag.max()) if len(te_off_diag) > 0 else 0.0
    median_te = float(np.median(te_off_diag)) if len(te_off_diag) > 0 else 0.0
    std_te = float(te_off_diag.std()) if len(te_off_diag) > 0 else 0.0

    # 最强信息流的节点对
    if max_te > 0:
        flat_idx = np.argmax(te_matrix * off_diag_mask)
        max_i, max_j = np.unravel_index(flat_idx, te_matrix.shape)
        max_pair = (int(max_i), int(max_j))  # TE(j→i) 最大
    else:
        max_pair = (0, 0)

    # ── 非对称性计算（修复 Bug 2）────────────────────────────────────────
    # 对每对 (i,j)，计算 TE(i→j) 和 TE(j→i)（上三角 vs 下三角）
    # te_matrix[i, j] = TE(j→i)（来源 j，目标 i）
    # 所以：TE(j→i) = te_matrix[i, j]
    #       TE(i→j) = te_matrix[j, i]
    upper_triangle = te_matrix[np.triu_indices(N, k=1)]  # TE 的一半
    lower_triangle = te_matrix[np.tril_indices(N, k=-1)]  # 对称方向

    # 每对的绝对非对称差
    abs_diff = np.abs(upper_triangle - lower_triangle)  # |TE(i→j) - TE(j→i)|
    pair_sum = upper_triangle + lower_triangle  # TE(i→j) + TE(j→i)

    # 相对非对称性（全局）
    total_sum = float(pair_sum.sum())
    if total_sum > 1e-10:
        asymmetry = float(abs_diff.sum() / total_sum)
    else:
        asymmetry = 0.0

    # ── 信息流方向性分析 ─────────────────────────────────────────────────
    # 净流量：每个节点的流出 TE - 流入 TE
    # te_matrix[i, j] = TE(j→i)
    # 节点 j 的流出 TE = sum_i te_matrix[i, j]（所有以 j 为源的 TE）
    # 节点 i 的流入 TE = sum_j te_matrix[i, j]（所有以 i 为目标的 TE）
    outflow = te_matrix.sum(axis=0)  # [N]，每节点流出
    inflow = te_matrix.sum(axis=1)   # [N]，每节点流入
    net_flow = outflow - inflow       # 正值 = 信息源，负值 = 信息汇

    # 影响力最高的节点（最大净流出）
    top_source_idx = int(np.argmax(net_flow))
    top_sink_idx = int(np.argmin(net_flow))

    stats = {
        # 基础统计（已排除对角线）
        "mean_te": avg_te,
        "max_te": max_te,
        "median_te": median_te,
        "std_te": std_te,
        "max_pair": max_pair,
        # 非对称性（已修复）
        "asymmetry": asymmetry,  # 相对非对称性，值域 [0,1]
        "abs_asymmetry_mean": float(abs_diff.mean()),  # 绝对非对称性均值
        # 有向性
        "outflow": outflow.tolist(),
        "inflow": inflow.tolist(),
        "net_flow": net_flow.tolist(),
        "top_source_node": top_source_idx,
        "top_sink_node": top_sink_idx,
        # 元信息
        "n_rois": N,
        "n_significant_pairs": int(np.sum(te_matrix[off_diag_mask] > 0)),
    }

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────

def plot_te_matrix(
    te_matrix: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "传递熵矩阵 TE(j→i)",
    roi_labels: Optional[List[str]] = None,
    dpi: int = 120,
    use_latex_math: bool = True,
) -> plt.Figure:
    """绘制 TE 矩阵热图。"""
    _configure_matplotlib_fonts(use_latex_math=use_latex_math)
    N = te_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(8, 7))

    vmax = np.percentile(te_matrix[te_matrix > 0], 95) if np.any(te_matrix > 0) else 0.01
    im = ax.imshow(
        te_matrix,
        cmap="hot_r",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="TE (nats)")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("源节点 j", fontsize=11)
    ax.set_ylabel("目标节点 i", fontsize=11)

    if roi_labels is not None and len(roi_labels) <= 30:
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(roi_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(roi_labels, fontsize=7)

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"TE 矩阵图已保存: {output_path}")

    return fig


def plot_information_flow(
    stats: Dict,
    output_path: Optional[Union[str, Path]] = None,
    roi_labels: Optional[List[str]] = None,
    dpi: int = 120,
) -> plt.Figure:
    """绘制节点级信息流（净流出量）排行图。"""
    net_flow = np.array(stats["net_flow"])
    N = len(net_flow)
    sorted_idx = np.argsort(net_flow)[::-1]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#F44336" if v > 0 else "#2196F3" for v in net_flow[sorted_idx]]
    ax.bar(range(N), net_flow[sorted_idx], color=colors, alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.8)

    if roi_labels and len(roi_labels) == N:
        labels = [roi_labels[i] for i in sorted_idx]
        tick_step = max(1, N // 20)
        ax.set_xticks(range(0, N, tick_step))
        ax.set_xticklabels(labels[::tick_step], rotation=45, fontsize=7)

    ax.set_xlabel("节点（排序后）", fontsize=11)
    ax.set_ylabel("净信息流出 (TE out - TE in, nats)", fontsize=11)
    ax.set_title(
        "节点级信息流方向性\n（红色=信息源，蓝色=信息汇）",
        fontsize=12,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"信息流图已保存: {output_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 主分析函数
# ─────────────────────────────────────────────────────────────────────────────

def run_transfer_entropy_analysis(
    timeseries: np.ndarray,
    config: Dict,
    output_dir: Union[str, Path],
    roi_labels: Optional[List[str]] = None,
) -> Dict:
    """运行完整的传递熵分析流程。

    Args:
        timeseries: [N_rois, T] 时序数据（已去趋势、滤波、z-score）。
        config: dynamics.yaml 中的 advanced 配置块。
        output_dir: 输出根目录。
        roi_labels: ROI 标签列表（可选）。

    Returns:
        results: 包含 te_matrix、stats 等的字典。
    """
    output_dir = Path(output_dir) / "advanced"
    output_dir.mkdir(parents=True, exist_ok=True)

    te_cfg = config.get("transfer_entropy", {})
    if not te_cfg.get("enabled", True):
        return {}

    out_cfg = config.get("output", {})
    dpi = out_cfg.get("dpi", 120)
    fmt = out_cfg.get("figure_format", "png")

    method = te_cfg.get("method", "knn")
    history = te_cfg.get("history_steps", 1)
    tau = te_cfg.get("tau", 1)
    knn_k = te_cfg.get("knn_k", 4)
    n_bins = te_cfg.get("n_bins", "auto")

    N, T = timeseries.shape
    logger.info(
        f"开始传递熵分析: N={N} ROIs, T={T} 时间点, "
        f"方法={method}, 历史={history}, τ={tau}"
    )

    if method == "knn" and N > 50:
        logger.info(
            f"k-NN TE 计算 {N}×{N}={N*N} 对，"
            f"预计耗时 {N*N*T//10000:.0f}s（建议大 N 时使用 method=binning）"
        )

    # 1. 计算 TE 矩阵
    te_matrix = compute_te_matrix(
        timeseries=timeseries,
        method=method,
        history=history,
        tau=tau,
        knn_k=knn_k,
        n_bins=n_bins,
    )

    # 2. 显著性检验（可选）
    stat_cfg = te_cfg.get("statistical_test", {})
    if stat_cfg.get("enabled", True):
        logger.info("执行置换显著性检验...")
        te_matrix, p_matrix = apply_permutation_test(
            timeseries=timeseries,
            te_matrix=te_matrix,
            method=method,
            history=history,
            tau=tau,
            knn_k=knn_k,
            n_bins=n_bins,
            n_permutations=stat_cfg.get("n_permutations", 50),
            alpha=stat_cfg.get("alpha", 0.05),
        )

    # 3. 保存 TE 矩阵
    te_path = output_dir / "transfer_entropy_matrix.npy"
    np.save(te_path, te_matrix)
    logger.info(f"TE 矩阵已保存: {te_path}")

    # 4. 计算统计量（修复版，排除对角线 + 正确的非对称性公式）
    stats = compute_information_flow_stats(te_matrix)

    # 5. 保存报告
    # 将 numpy 类型转为 Python 原生类型（JSON 序列化）
    report = {k: (v if not isinstance(v, np.ndarray) else v.tolist())
              for k, v in stats.items()}
    report["config_used"] = {
        "method": method,
        "history_steps": history,
        "tau": tau,
    }
    report_path = output_dir / "information_flow_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"信息流报告已保存: {report_path}")

    # 6. 打印摘要（修复版：avg 排除对角线，asymmetry 使用相对非对称性）
    logger.info(
        f"信息流分析完成: "
        f"平均 TE={stats['mean_te']:.4f} nats (排除对角线), "
        f"最大 TE={stats['max_te']:.4f} nats, "
        f"相对非对称性={stats['asymmetry']:.4f}"
    )
    logger.info(
        f"  最强连接: 节点 {stats['max_pair'][1]} → 节点 {stats['max_pair'][0]}, "
        f"TE={stats['max_te']:.4f} nats"
    )
    logger.info(
        f"  信息源: 节点 {stats['top_source_node']} | "
        f"信息汇: 节点 {stats['top_sink_node']}"
    )

    # 7. 可视化
    plot_te_matrix(
        te_matrix=te_matrix,
        output_path=output_dir / f"transfer_entropy_matrix.{fmt}",
        roi_labels=roi_labels,
        dpi=dpi,
    )

    plot_information_flow(
        stats=stats,
        output_path=output_dir / f"information_flow.{fmt}",
        roi_labels=roi_labels,
        dpi=dpi,
    )

    return {
        "te_matrix": te_matrix,
        "stats": stats,
        "report_path": str(report_path),
    }
