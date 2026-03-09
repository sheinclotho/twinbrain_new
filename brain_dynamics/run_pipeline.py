"""
run_pipeline.py — 脑动力学分析流水线主入口
============================================

用法示例：
  # 使用默认配置，从 npy 文件分析
  python brain_dynamics/run_pipeline.py \
      --config brain_dynamics/config/dynamics.yaml \
      --timeseries data/fmri_rois.npy

  # 覆盖输出目录
  python brain_dynamics/run_pipeline.py \
      --config brain_dynamics/config/dynamics.yaml \
      --timeseries data/fmri_rois.npy \
      --output-dir my_outputs/dynamics

流水线阶段：
  Phase 1 (结构分析):
    1. 功能连接（FC）矩阵计算
    2. 特征值谱分析（含 MP 上界）
    3. 响应矩阵计算（可配置刺激节点）
  Advanced (高级分析):
    4. 传递熵矩阵计算（k-NN 估计器）
    5. 信息流统计与报告
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml

# 确保模块路径正确
sys.path.insert(0, str(Path(__file__).parent.parent))

from brain_dynamics.spectral_dynamics.e1_spectral_analysis import run_spectral_analysis
from brain_dynamics.phase1.response_matrix import run_response_matrix_analysis
from brain_dynamics.advanced.transfer_entropy import run_transfer_entropy_analysis

# ─────────────────────────────────────────────────────────────────────────────
# 日志配置
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(level: str = "INFO") -> logging.Logger:
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt, datefmt=datefmt)
    return logging.getLogger("dynamics_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# 配置加载
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> Dict:
    """加载并返回 YAML 配置。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_cli_overrides(cfg: Dict, args: argparse.Namespace) -> Dict:
    """将命令行参数合并到配置中（CLI 参数优先级高于配置文件）。"""
    if args.timeseries:
        cfg.setdefault("input", {})["timeseries_path"] = args.timeseries
    if args.output_dir:
        cfg.setdefault("output", {})["output_dir"] = args.output_dir
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def load_timeseries(path: str, logger: logging.Logger) -> np.ndarray:
    """从文件加载 fMRI 时序数据。

    支持格式：
      .npy  — numpy 数组，形状 [N_rois, T]
      .npz  — numpy 压缩包，key "timeseries" 或第一个 key
      .csv  — 逗号分隔，行=ROI，列=时间点

    Args:
        path: 文件路径。
        logger: 日志对象。

    Returns:
        timeseries: [N_rois, T] numpy 数组（float32）。
    """
    path = Path(path)
    logger.info(f"加载时序数据: {path}")

    if not path.exists():
        raise FileNotFoundError(f"时序文件不存在: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        ts = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        keys = list(data.keys())
        key = "timeseries" if "timeseries" in keys else keys[0]
        ts = data[key]
        logger.info(f"从 .npz 加载 key='{key}'")
    elif suffix in (".csv", ".tsv"):
        sep = "," if suffix == ".csv" else "\t"
        ts = np.loadtxt(path, delimiter=sep)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}。支持: .npy, .npz, .csv, .tsv")

    if ts.ndim != 2:
        raise ValueError(
            f"时序数据形状错误: {ts.shape}，应为 [N_rois, T]（2D 矩阵）。"
        )

    # 确保 N < T（ROI 数量通常远小于时间点数量）
    if ts.shape[0] > ts.shape[1]:
        logger.warning(
            f"时序数据形状 {ts.shape} 中 N_rois > T，"
            f"自动转置为 [N_rois={ts.shape[1]}, T={ts.shape[0]}]。"
            f"若这不是期望的，请检查输入文件的行/列方向。"
        )
        ts = ts.T

    N, T = ts.shape
    logger.info(f"时序数据: N_rois={N}, T={T}")
    return ts.astype(np.float32)


def preprocess_timeseries(
    ts: np.ndarray,
    logger: logging.Logger,
) -> np.ndarray:
    """基础预处理：去均值 + z-score 标准化。

    注意：实际神经影像分析通常需要更完整的预处理流程（高通滤波、去漂移等）。
    此处仅提供基础的数值标准化，以确保 FC 和 TE 估计的稳定性。

    Args:
        ts: [N, T] 时序数据。
        logger: 日志对象。

    Returns:
        ts_z: z-score 标准化后的时序数据。
    """
    # 去均值
    ts = ts - ts.mean(axis=1, keepdims=True)

    # z-score（逐 ROI 标准化）
    std = ts.std(axis=1, keepdims=True)
    # 对方差为零的 ROI 发出警告并置零
    zero_std_mask = std.ravel() < 1e-10
    if zero_std_mask.any():
        n_zero = zero_std_mask.sum()
        logger.warning(
            f"{n_zero} 个 ROI 的时序方差接近零（信号缺失），已置零处理。"
        )
        std[std < 1e-10] = 1.0

    ts_z = ts / std
    logger.info("时序预处理完成（z-score 标准化）")
    return ts_z


# ─────────────────────────────────────────────────────────────────────────────
# 主流水线
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    cfg: Dict,
    logger: logging.Logger,
) -> Dict:
    """运行完整的脑动力学分析流水线。

    Args:
        cfg: 完整配置字典。
        logger: 日志对象。

    Returns:
        all_results: 各阶段分析结果的字典。
    """
    input_cfg = cfg.get("input", {})
    output_cfg = cfg.get("output", {})
    phase1_cfg = cfg.get("phase1", {})
    advanced_cfg = cfg.get("advanced", {})

    output_dir = Path(output_cfg.get("output_dir", "outputs/dynamics_pipeline"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 将 output 配置下沉到 phase1/advanced 中（子模块需要）
    phase1_cfg["output"] = output_cfg
    advanced_cfg["output"] = output_cfg

    all_results: Dict = {}

    # ── 加载数据 ──────────────────────────────────────────────────────────
    ts_path = input_cfg.get("timeseries_path")
    if ts_path is None:
        raise ValueError(
            "未指定时序数据路径。请在 dynamics.yaml 中设置 input.timeseries_path，"
            "或通过 --timeseries 命令行参数指定。"
        )

    timeseries = load_timeseries(ts_path, logger)
    timeseries = preprocess_timeseries(timeseries, logger)

    # 加载 ROI 标签（可选）
    roi_labels = None
    roi_labels_path = input_cfg.get("roi_labels_path")
    if roi_labels_path and Path(roi_labels_path).exists():
        with open(roi_labels_path, encoding="utf-8") as f:
            roi_data = json.load(f)
        if isinstance(roi_data, list):
            roi_labels = roi_data
        elif isinstance(roi_data, dict):
            roi_labels = list(roi_data.values())
        logger.info(f"ROI 标签加载完成: {len(roi_labels)} 个")

    # ── Phase 1：结构分析 ─────────────────────────────────────────────────
    if phase1_cfg.get("enabled", True):
        logger.info("─── Phase 1: 结构分析 ───────────────────────────")

        # 谱分析（特征值）
        spectral_results = run_spectral_analysis(
            timeseries=timeseries,
            config=phase1_cfg,
            output_dir=output_dir,
            roi_labels=roi_labels,
        )
        all_results["spectral"] = spectral_results
        fc = spectral_results.get("fc")

        # 响应矩阵
        if fc is not None:
            rm_results = run_response_matrix_analysis(
                fc=fc,
                config=phase1_cfg,
                output_dir=output_dir,
                roi_labels=roi_labels,
            )
            all_results["response_matrix"] = rm_results

        logger.info("─── Phase 1 完成 ────────────────────────────────")

    # ── Advanced：高级分析 ────────────────────────────────────────────────
    if advanced_cfg.get("enabled", True):
        logger.info("─── Advanced: 高级分析 ──────────────────────────")

        te_cfg = advanced_cfg.get("transfer_entropy", {})
        if te_cfg.get("enabled", True):
            logger.info("  Information flow: started.")
            te_results = run_transfer_entropy_analysis(
                timeseries=timeseries,
                config=advanced_cfg,
                output_dir=output_dir,
                roi_labels=roi_labels,
            )
            all_results["transfer_entropy"] = te_results
            logger.info("  Information flow: done.")

        logger.info("─── Advanced 完成 ───────────────────────────────")

    logger.info(f"流水线完成！结果保存于: {output_dir}")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TwinBrain 脑动力学分析流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="brain_dynamics/config/dynamics.yaml",
        help="配置文件路径（默认: brain_dynamics/config/dynamics.yaml）",
    )
    parser.add_argument(
        "--timeseries",
        type=str,
        default=None,
        help="fMRI 时序数据路径（.npy / .npz / .csv），覆盖配置文件中的路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        help="输出目录，覆盖配置文件中的路径",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认: INFO）",
    )
    args = parser.parse_args()

    logger = _setup_logging(args.log_level)

    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args)

    run_pipeline(cfg=cfg, logger=logger)


if __name__ == "__main__":
    main()
