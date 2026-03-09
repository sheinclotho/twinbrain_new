"""
e1_spectral_analysis.py — 功能连接矩阵谱分析
=============================================

主要功能：
  1. 计算 fMRI 时序数据的功能连接（FC）矩阵
  2. 特征值谱分析（含 Marchenko-Pastur 随机矩阵理论边界）
  3. 特征向量空间可视化（用于识别主要功能网络模式）

字体说明（修复 Glyph 8321 警告）：
  原始代码中使用 Unicode 下标字符（如 U+2081，Glyph 8321）标注坐标轴，
  该字符在 Windows 常用字体（Microsoft YaHei 等）中通常缺失，导致：
    UserWarning: Glyph 8321 (\\N{SUBSCRIPT ONE}) missing from font(s) Microsoft YaHei.
  修复方法：统一使用 matplotlib 数学文本语法 $\\lambda_1$、$\\lambda_2$ 等，
  无需 Unicode 下标字符，跨平台（Windows/Linux/macOS）均可正常渲染。

References:
  - Marčenko & Pastur (1967). Distribution of eigenvalues for some sets of
    random matrices. Math. USSR-Sb, 1(4):457.
  - Bullmore & Sporns (2009). Complex brain networks. Nat. Rev. Neurosci., 10:186.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无头环境兼容（服务器/云端训练），必须在 pyplot 之前设置
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 字体设置辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _configure_matplotlib_fonts(use_latex_math: bool = True) -> None:
    """配置 matplotlib 字体，避免 Unicode 下标/上标字符的缺字形警告。

    策略：
      1. 使用 matplotlib 内置数学文本渲染器（usetex=False 时的 MathText 引擎）
         处理 $\\lambda_1$ 等符号，无需系统 LaTeX 或特定字体支持。
      2. 字体回退链：DejaVu Serif → DejaVu Sans → 系统默认
         DejaVu 字体内置于 matplotlib，始终可用，覆盖大多数数学符号。
      3. 中文字符通过单独的 CJK 字体处理（若系统有安装）。

    Args:
        use_latex_math: True 时配置 MathText 模式（推荐）；False 时保持默认。
    """
    if not use_latex_math:
        return

    # 使用 matplotlib 内置数学文本引擎（不依赖系统 LaTeX）
    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.fontset"] = "dejavusans"  # 内置，始终可用

    # 设置字体优先级：先尝试 DejaVu（内置，支持数学符号），再回退到系统字体
    plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]

    # 中文字体支持（若系统有安装则使用，否则静默跳过）
    _add_cjk_font_if_available()


def _add_cjk_font_if_available() -> None:
    """尝试在字体回退链中添加系统 CJK（中日韩）字体。

    按优先级尝试：
      Windows: Microsoft YaHei → SimHei → NSimSun
      macOS: PingFang SC → Heiti SC
      Linux: Noto Sans CJK SC → WenQuanYi Micro Hei
    若均未找到，静默跳过（不影响英文/数学符号渲染）。
    """
    import matplotlib.font_manager as fm

    cjk_candidates = [
        # Windows
        "Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "NSimSun",
        # macOS
        "PingFang SC", "Heiti SC",
        # Linux
        "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Droid Sans Fallback",
    ]

    available = {f.name for f in fm.fontManager.ttflist}
    found = next((c for c in cjk_candidates if c in available), None)

    if found:
        current_fonts = plt.rcParams.get("font.sans-serif", [])
        if found not in current_fonts:
            # 在 DejaVu 之后、默认字体之前插入 CJK 字体
            plt.rcParams["font.sans-serif"] = ["DejaVu Sans", found] + [
                f for f in current_fonts if f not in {"DejaVu Sans", found}
            ]
        logger.debug(f"CJK 字体已添加到字体回退链: {found}")
    else:
        logger.debug("未找到 CJK 字体，中文标签将显示为方框（不影响数学符号）")


# ─────────────────────────────────────────────────────────────────────────────
# FC 矩阵计算
# ─────────────────────────────────────────────────────────────────────────────

def compute_fc_matrix(
    timeseries: np.ndarray,
    method: str = "pearson",
    fisher_z: bool = False,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """计算功能连接（FC）矩阵。

    Args:
        timeseries: 形状 [N_rois, T]，各 ROI 的时序信号（已 z-score 标准化为佳）。
        method: "pearson" | "partial" | "coherence"。
        fisher_z: 是否对相关系数进行 Fisher r→z 变换（提高正态性）。
        threshold: 绝对值阈值；低于此值的连接置零。null = 不过滤。

    Returns:
        fc: 形状 [N_rois, N_rois] 的 FC 矩阵。

    Scientific Notes:
        Pearson FC 是 resting-state fMRI 的标准连通性估计。
        对于短时序（T<200），估计方差较大；可考虑 partial 相关减少间接影响。
        Fisher z-变换适用于后续 t 检验等统计分析（Bishara & Hittner 2012）。
    """
    N, T = timeseries.shape
    if T < 10:
        logger.warning(
            f"时序长度 T={T} 过短，FC 矩阵估计方差会很大（建议 T≥50）。"
        )

    if method == "pearson":
        # np.corrcoef 内部处理了零方差情况，返回 NaN（用 0 替换）
        fc = np.corrcoef(timeseries)
        fc = np.nan_to_num(fc, nan=0.0)

    elif method == "partial":
        # 偏相关：通过精确矩阵的逆（precision matrix）计算
        fc = _compute_partial_correlation(timeseries)

    elif method == "coherence":
        fc = _compute_wideband_coherence(timeseries)

    else:
        raise ValueError(
            f"未知的 FC 计算方法：'{method}'。支持: 'pearson', 'partial', 'coherence'。"
        )

    # Fisher r→z 变换（仅对 Pearson / partial 有意义）
    if fisher_z and method in ("pearson", "partial"):
        # atanh(r) = 0.5 * ln((1+r)/(1-r))；对角线（r=1）单独处理
        np.fill_diagonal(fc, 0.0)
        # 限制范围到 (-0.9999, 0.9999) 避免 atanh 的边界发散
        fc_clipped = np.clip(fc, -0.9999, 0.9999)
        fc = np.arctanh(fc_clipped)
        np.fill_diagonal(fc, 0.0)  # 对角线保持为 0（自相关无意义）

    # 阈值化
    if threshold is not None and threshold > 0:
        fc[np.abs(fc) < threshold] = 0.0

    # 保证对称性（数值误差）
    fc = (fc + fc.T) / 2.0
    np.fill_diagonal(fc, 0.0)  # 对角线恒为 0

    return fc.astype(np.float32)


def _compute_partial_correlation(timeseries: np.ndarray) -> np.ndarray:
    """通过精确矩阵计算偏相关系数矩阵。

    偏相关 = -(precision_ij) / sqrt(precision_ii * precision_jj)
    Scientific basis: Marrelec et al. (2006) Partial correlation for
    functional brain connectivity. NeuroImage, 32(1):228.
    """
    # Pearson 相关矩阵
    corr = np.corrcoef(timeseries)
    corr = np.nan_to_num(corr, nan=0.0)
    # 添加微小正则化以避免奇异矩阵（regularized pseudoinverse）
    N = corr.shape[0]
    reg = 1e-4  # Tikhonov 正则化系数
    try:
        precision = np.linalg.inv(corr + reg * np.eye(N))
    except np.linalg.LinAlgError:
        logger.warning("FC 矩阵奇异，使用伪逆（pinv）代替精确逆")
        precision = np.linalg.pinv(corr)

    # 归一化为相关系数尺度
    diag_sqrt = np.sqrt(np.abs(np.diag(precision)))
    # 避免除以零
    diag_sqrt[diag_sqrt < 1e-10] = 1.0
    partial_corr = -precision / np.outer(diag_sqrt, diag_sqrt)
    np.fill_diagonal(partial_corr, 0.0)
    return partial_corr


def _compute_wideband_coherence(timeseries: np.ndarray) -> np.ndarray:
    """计算宽频幅度相干性（Magnitude Squared Coherence）。

    使用 numpy rfft 计算所有频率箱的平均 MSC，值域 [0, 1]。
    Scientific basis: Sun et al. (2004) Measuring interregional
    functional connectivity using coherence. NeuroImage, 21(2):647.
    """
    N, T = timeseries.shape
    # FFT
    F = np.fft.rfft(timeseries, axis=1)  # [N, n_freq]
    n_freq = F.shape[1]
    # 跨频率平均互谱矩阵
    cross_spectral = F @ F.conj().T / n_freq  # [N, N]
    # 振幅 MSC
    power = np.maximum(np.real(np.diag(cross_spectral)), 1e-12)
    coherence = np.abs(cross_spectral) ** 2 / np.outer(power, power)
    # 确保对角线为 0（fill_diagonal 需要可写数组）
    coherence = np.clip(coherence, 0.0, 1.0).copy()
    np.fill_diagonal(coherence, 0.0)
    return coherence.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 特征值谱分析
# ─────────────────────────────────────────────────────────────────────────────

def marchenko_pastur_bound(N: int, T: int, sigma: float = 1.0) -> float:
    """计算 Marchenko-Pastur 随机矩阵理论的最大特征值上界。

    对于 N×T 的随机矩阵（独立高斯元素），对应相关矩阵的最大特征值趋近于：
        λ_max = σ² × (1 + √(N/T))²

    高于此界的特征值被认为来自真实的神经信号，而非随机噪声。

    Args:
        N: ROI 数量。
        T: 时间点数量。
        sigma: 信号标准差（已 z-score 时 = 1.0）。

    Returns:
        λ_max: Marchenko-Pastur 上界。

    Reference:
        Marčenko & Pastur (1967). Distribution of eigenvalues for some sets of
        random matrices. Mathematics of the USSR-Sbornik, 1(4):457.
    """
    ratio = N / T
    return sigma**2 * (1.0 + np.sqrt(ratio)) ** 2


def compute_eigenvalue_spectrum(
    fc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算功能连接矩阵的特征值和特征向量。

    对于对称 FC 矩阵（Pearson / partial），使用 numpy.linalg.eigh（比 eig 更快更稳定）。
    对于非对称矩阵，使用 numpy.linalg.eig（返回复数特征值）。

    Args:
        fc: 形状 [N, N] 的 FC 矩阵。

    Returns:
        eigenvalues: 形状 [N]，按降序排列的特征值（可能为复数）。
        eigenvectors: 形状 [N, N]，对应的特征向量（列向量）。

    Scientific Notes:
        FC 矩阵的主特征向量对应整个大脑的平均激活模式（global signal）；
        第 2、3 特征向量通常对应对角线对立模式（如 DMN vs. 任务网络）。
        负特征值在 Pearson FC 中对应反相关模式（如 DMN 与背侧注意网络的负耦合）。
    """
    is_symmetric = np.allclose(fc, fc.T, atol=1e-5)

    if is_symmetric:
        # eigh 专门针对对称矩阵，更稳定（特征值保证为实数）
        eigenvalues, eigenvectors = np.linalg.eigh(fc)
        # eigh 返回升序；转为降序
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    else:
        # 非对称矩阵（如有向 FC）：特征值可能为复数
        eigenvalues, eigenvectors = np.linalg.eig(fc)
        # 按特征值绝对值降序排列
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


# ─────────────────────────────────────────────────────────────────────────────
# 可视化函数
# ─────────────────────────────────────────────────────────────────────────────

def plot_eigenvalue_spectrum(
    eigenvalues: np.ndarray,
    fc: np.ndarray,
    tr: float = 2.0,
    output_path: Optional[Union[str, Path]] = None,
    n_top: Optional[int] = None,
    show_mp_bound: bool = True,
    use_latex_math: bool = True,
    dpi: int = 120,
    T: Optional[int] = None,
) -> plt.Figure:
    """绘制功能连接矩阵的特征值谱图。

    修复说明（Glyph 8321 警告）：
        原始代码使用 Unicode 下标字符（U+2081，Glyph 8321）：
            ax.set_xlabel("lambda_sub_1", ...)  # 产生 Glyph 8321 警告
        本函数改用 matplotlib MathText 语法：
            ax.set_xlabel(r"$\\lambda_1$", ...)  # ← 无警告，跨平台兼容

    Args:
        eigenvalues: 降序排列的特征值数组（可能为复数）。
        fc: 原始 FC 矩阵（用于获取 N 信息，MP 上界通过 T 参数传入）。
        tr: TR（重复时间，秒），用于绘制参考线。
        output_path: 保存路径；None = 只显示不保存。
        n_top: 只显示前 n_top 个特征值；None = 显示全部。
        show_mp_bound: 是否绘制 Marchenko-Pastur 随机矩阵上界参考线。
        use_latex_math: 是否使用 MathText 渲染数轴标签（解决 Unicode 下标字符警告）。
        dpi: 图像 DPI。
        T: 时序长度（用于计算 MP 上界）；None 时跳过 MP 上界绘制。

    Returns:
        matplotlib Figure 对象。
    """
    _configure_matplotlib_fonts(use_latex_math=use_latex_math)

    N = fc.shape[0]
    eigs_real = np.real(eigenvalues)
    eigs_imag = np.imag(eigenvalues)
    is_complex = np.any(np.abs(eigs_imag) > 1e-8)

    if n_top is not None:
        eigs_real = eigs_real[:n_top]
        eigs_imag = eigs_imag[:n_top]

    k = len(eigs_real)
    indices = np.arange(1, k + 1)

    if is_complex:
        # 非对称 FC 矩阵：同时展示实部谱和复平面散点图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_spec, ax_complex = axes
    else:
        fig, ax_spec = plt.subplots(1, 1, figsize=(10, 5))

    # ── 特征值谱（实部 vs 索引）──────────────────────────────────────────
    ax_spec.bar(indices, eigs_real, color="#2196F3", alpha=0.8, width=0.8)
    ax_spec.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # Marchenko-Pastur 上界
    if show_mp_bound and T is not None:
        mp_max = marchenko_pastur_bound(N=N, T=T)
        ax_spec.axhline(
            y=mp_max,
            color="#F44336",
            linewidth=1.5,
            linestyle="--",
            # 修复：使用 MathText 而非 Unicode 下标（避免 Glyph 8321 警告）
            label=r"MP bound $\lambda_{max}$",
        )
        ax_spec.legend(fontsize=9)

    # 修复 Glyph 8321 警告：
    #   原来可能写的是 "lambda_1 ~ lambda_N" 使用 Unicode 下标字符（如 U+2081）
    #   现改为使用 MathText 语法 r"$\lambda_1 \sim \lambda_N$"
    ax_spec.set_xlabel(r"特征值索引 $k$", fontsize=11)
    ax_spec.set_ylabel(r"特征值 $\lambda_k$", fontsize=11)
    ax_spec.set_title(
        r"功能连接矩阵特征值谱 ($\lambda_1 \geq \lambda_2 \geq \cdots$)",
        fontsize=12,
    )
    ax_spec.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))
    ax_spec.grid(axis="y", linestyle=":", alpha=0.5)

    # ── 复平面散点图（仅非对称矩阵）────────────────────────────────────
    if is_complex:
        ax_complex.scatter(
            eigs_real,
            eigs_imag,
            c=np.arange(len(eigs_real)),
            cmap="viridis",
            alpha=0.8,
            s=30,
        )
        ax_complex.axhline(y=0, color="black", linewidth=0.8, alpha=0.5)
        ax_complex.axvline(x=0, color="black", linewidth=0.8, alpha=0.5)
        # 绘制单位圆
        theta = np.linspace(0, 2 * np.pi, 300)
        ax_complex.plot(np.cos(theta), np.sin(theta), "r--", linewidth=1, alpha=0.5)
        # 修复：使用 MathText 代替 Unicode 下标
        ax_complex.set_xlabel(r"实部 $\mathrm{Re}(\lambda)$", fontsize=11)
        ax_complex.set_ylabel(r"虚部 $\mathrm{Im}(\lambda)$", fontsize=11)
        ax_complex.set_title("复数特征值分布（复平面）", fontsize=12)
        ax_complex.set_aspect("equal", adjustable="datalim")
        ax_complex.grid(linestyle=":", alpha=0.4)

        # 保存文件名：eigenvalue_complex_fc.png（与日志中一致）
        fig.suptitle("功能连接矩阵特征分析", fontsize=13, fontweight="bold")

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"保存: {output_path}")

    return fig


def plot_fc_matrix(
    fc: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "功能连接矩阵",
    roi_labels: Optional[List[str]] = None,
    use_latex_math: bool = True,
    dpi: int = 120,
) -> plt.Figure:
    """绘制功能连接矩阵热图。

    Args:
        fc: [N, N] FC 矩阵。
        output_path: 保存路径。
        title: 图标题。
        roi_labels: ROI 标签列表（可选）。
        use_latex_math: 是否使用 MathText 渲染。
        dpi: 图像 DPI。
    """
    _configure_matplotlib_fonts(use_latex_math=use_latex_math)

    N = fc.shape[0]
    fig, ax = plt.subplots(figsize=(8, 7))

    vmax = np.percentile(np.abs(fc), 95)  # 95 百分位避免极端值影响色阶
    im = ax.imshow(
        fc,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="连接强度 (r)")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("ROI 索引", fontsize=11)
    ax.set_ylabel("ROI 索引", fontsize=11)

    if roi_labels is not None and len(roi_labels) <= 50:
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(roi_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(roi_labels, fontsize=7)

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

def run_spectral_analysis(
    timeseries: np.ndarray,
    config: Dict,
    output_dir: Union[str, Path],
    roi_labels: Optional[List[str]] = None,
) -> Dict:
    """运行完整的谱分析流程。

    Args:
        timeseries: [N_rois, T] 时序数据（已去趋势、滤波、z-score）。
        config: dynamics.yaml 中的 phase1 配置块。
        output_dir: 输出目录。
        roi_labels: ROI 标签列表（可选）。

    Returns:
        results: 包含 fc 矩阵、特征值等的字典。
    """
    output_dir = Path(output_dir) / "structure"
    output_dir.mkdir(parents=True, exist_ok=True)

    fc_cfg = config.get("fc", {})
    eig_cfg = config.get("eigenvalue", {})
    out_cfg = config.get("output", {})
    use_latex = out_cfg.get("use_latex_math", True)
    dpi = out_cfg.get("dpi", 120)
    fmt = out_cfg.get("figure_format", "png")

    # 1. 计算 FC 矩阵
    logger.info("计算功能连接矩阵...")
    N, T = timeseries.shape
    fc = compute_fc_matrix(
        timeseries,
        method=fc_cfg.get("method", "pearson"),
        fisher_z=fc_cfg.get("fisher_z", False),
        threshold=fc_cfg.get("threshold"),
    )

    # 保存 FC 矩阵
    fc_path = output_dir / "fc_matrix.npy"
    np.save(fc_path, fc)
    logger.info(f"FC 矩阵已保存: {fc_path}")

    # 绘制 FC 矩阵热图
    plot_fc_matrix(
        fc,
        output_path=output_dir / f"fc_matrix.{fmt}",
        roi_labels=roi_labels,
        use_latex_math=use_latex,
        dpi=dpi,
    )

    results: Dict = {"fc": fc, "N": N, "T": T}

    # 2. 特征值分析
    if eig_cfg.get("enabled", True):
        logger.info("计算特征值谱...")
        eigenvalues, eigenvectors = compute_eigenvalue_spectrum(fc)
        results["eigenvalues"] = eigenvalues
        results["eigenvectors"] = eigenvectors

        # MP 上界
        if eig_cfg.get("show_marchenko_pastur_bound", True):
            mp_bound = marchenko_pastur_bound(N=N, T=T)
            results["marchenko_pastur_bound"] = mp_bound
            n_signal = int(np.sum(np.real(eigenvalues) > mp_bound))
            logger.info(
                f"Marchenko-Pastur 上界: {mp_bound:.4f}，"
                f"超过上界的特征值数量: {n_signal} / {N}"
                f"（携带真实信号的主成分数量）"
            )

        # 绘制特征值谱
        n_top = eig_cfg.get("n_top")

        # eigenvalue_complex_fc.png（与日志路径一致）
        plot_eigenvalue_spectrum(
            eigenvalues=eigenvalues,
            fc=fc,
            T=T,
            output_path=output_dir / f"eigenvalue_complex_fc.{fmt}",
            n_top=n_top,
            show_mp_bound=eig_cfg.get("show_marchenko_pastur_bound", True),
            use_latex_math=use_latex,
            dpi=dpi,
        )

        # 附加：仅实部谱（线性刻度）
        if not eig_cfg.get("plot_complex", True):
            plot_eigenvalue_spectrum(
                eigenvalues=eigenvalues,
                fc=fc,
                T=T,
                output_path=output_dir / f"eigenvalue_spectrum.{fmt}",
                n_top=n_top,
                show_mp_bound=eig_cfg.get("show_marchenko_pastur_bound", True),
                use_latex_math=use_latex,
                dpi=dpi,
            )

    return results
