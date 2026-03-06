#!/usr/bin/env python3
"""
TwinBrain — 平均图生成脚本
===========================

将文件夹内所有图缓存文件 (*.pt, HeteroData) 合并为平均图。

工作流程:
1. 扫描指定文件夹，读取所有 *.pt 文件中的同模态连通性边。
2. 检查所有图的每种边类型的 edge_index 完全一致；若不一致则报错退出。
3. 对各边类型的 edge_attr (边权) 在所有图上取均值
   （可选：先做 Fisher z 变换，均值后再反变换回相关系数域）。
4. 保存 mean_graph.pt，包含 edge_index 与平均后的 edge_attr，
   并保留原有图节点类型结构（EEG、fMRI）不变。
5. 打印每种边类型的统计信息（最小值、最大值、均值）。
6. 可选：生成边权分布直方图（需要 matplotlib）。

使用方法::

    python compute_mean_graph.py <cache_dir> [OPTIONS]

选项::

    --output PATH        输出文件路径（默认: <cache_dir>/mean_graph.pt）
    --pattern GLOB       文件匹配模式（默认: *.pt，排除 mean_graph.pt）
    --fisher-z           使用 Fisher z 变换对相关系数进行平均
    --plot               保存边权分布直方图（需要 matplotlib）
    --plot-output PATH   图片保存路径（默认: <output_dir>/mean_graph_dist.png）

    --log-level LEVEL    日志级别（默认: INFO）

示例::

    # 基本用法
    python compute_mean_graph.py outputs/graph_cache

    # 使用 Fisher z 变换并绘图
    python compute_mean_graph.py outputs/graph_cache --fisher-z --plot

    # 指定输出路径
    python compute_mean_graph.py outputs/graph_cache --output results/mean_graph.pt
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from torch_geometric.data import HeteroData


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def load_graphs(files: List[Path]) -> List[HeteroData]:
    """Load a list of HeteroData graph cache files from disk.

    Args:
        files: List of .pt file paths to load.

    Returns:
        List of HeteroData objects.

    Raises:
        RuntimeError: If a file cannot be loaded or does not contain a
            HeteroData object.
    """
    graphs: List[HeteroData] = []
    for path in files:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"无法加载文件 {path}: {exc}") from exc

        if not isinstance(obj, HeteroData):
            raise RuntimeError(
                f"文件 {path} 不是 HeteroData 对象 (类型: {type(obj).__name__})，"
                "请确认该文件夹中仅包含图缓存文件。"
            )
        graphs.append(obj)
    return graphs


def _edge_index_key(edge_index: torch.Tensor) -> Tuple:
    """Convert an edge_index tensor to a hashable tuple for equality checks."""
    return tuple(edge_index.cpu().numpy().reshape(-1).tolist())


def check_edge_index_consistency(
    graphs: List[HeteroData],
    files: List[Path],
) -> None:
    """Verify that every graph has the same edge types and identical edge_index
    for each edge type.

    Args:
        graphs: List of loaded HeteroData objects.
        files: Corresponding file paths (for error messages).

    Raises:
        ValueError: If any inconsistency is found.
    """
    if len(graphs) == 0:
        raise ValueError("未找到任何图文件。")

    ref_graph = graphs[0]
    ref_file = files[0]

    # Reference edge types (sorted for determinism)
    ref_edge_types = sorted(ref_graph.edge_types)
    logger.debug("参考图 edge_types: %s", ref_edge_types)

    # Store reference edge_index keys per edge type
    ref_ei_keys: Dict[Tuple, Tuple] = {}
    for et in ref_edge_types:
        if not hasattr(ref_graph[et], "edge_index") or ref_graph[et].edge_index is None:
            logger.warning("参考图 %s 的边类型 %s 缺少 edge_index，跳过。", ref_file.name, et)
            continue
        ref_ei_keys[et] = _edge_index_key(ref_graph[et].edge_index)

    # Check each subsequent graph
    for g, f in zip(graphs[1:], files[1:]):
        g_edge_types = sorted(g.edge_types)

        if g_edge_types != ref_edge_types:
            raise ValueError(
                f"边类型不一致！\n"
                f"  参考文件 {ref_file.name}: {ref_edge_types}\n"
                f"  当前文件 {f.name}: {g_edge_types}"
            )

        for et in ref_edge_types:
            if et not in ref_ei_keys:
                continue
            if not hasattr(g[et], "edge_index") or g[et].edge_index is None:
                raise ValueError(
                    f"文件 {f.name} 的边类型 {et} 缺少 edge_index，"
                    f"但参考文件 {ref_file.name} 中存在。"
                )
            g_key = _edge_index_key(g[et].edge_index)
            if g_key != ref_ei_keys[et]:
                raise ValueError(
                    f"edge_index 不一致！边类型 {et}\n"
                    f"  参考文件: {ref_file.name} "
                    f"(shape={ref_graph[et].edge_index.shape})\n"
                    f"  不匹配文件: {f.name} "
                    f"(shape={g[et].edge_index.shape})\n"
                    "所有图的 edge_index 必须完全相同才能计算平均图。\n"
                    "提示：不同被试/条件可能具有不同的图拓扑，请只传入拓扑相同的图。"
                )

    logger.info("✔ 所有 %d 个图的 edge_index 完全一致。", len(graphs))


def _fisher_z(r: torch.Tensor) -> torch.Tensor:
    """Apply Fisher z-transform: z = atanh(r).  Clamps r to (-1+eps, 1-eps)."""
    eps = 1e-7
    r_clamped = torch.clamp(r, -1.0 + eps, 1.0 - eps)
    return torch.atanh(r_clamped)


def _fisher_z_inv(z: torch.Tensor) -> torch.Tensor:
    """Inverse Fisher z-transform: r = tanh(z)."""
    return torch.tanh(z)


def compute_mean_edge_attrs(
    graphs: List[HeteroData],
    use_fisher_z: bool = False,
) -> Dict[Tuple, torch.Tensor]:
    """Compute the mean edge_attr for each edge type across all graphs.

    Args:
        graphs: List of HeteroData objects with consistent edge_index.
        use_fisher_z: If True, apply Fisher z-transform before averaging and
            inverse-transform afterwards.  Use this when edge_attr values are
            Pearson correlation coefficients.

    Returns:
        Dict mapping each edge type tuple to the averaged edge_attr tensor
        with the same shape as the original per-graph edge_attr.
    """
    ref_graph = graphs[0]
    mean_attrs: Dict[Tuple, torch.Tensor] = {}

    for et in ref_graph.edge_types:
        if not hasattr(ref_graph[et], "edge_attr") or ref_graph[et].edge_attr is None:
            logger.debug("边类型 %s 不含 edge_attr，跳过平均。", et)
            continue

        # Stack all edge_attr tensors: [n_graphs, E, feat_dim]
        attrs = []
        for g in graphs:
            if not hasattr(g[et], "edge_attr") or g[et].edge_attr is None:
                raise ValueError(
                    f"图文件集合中部分图的边类型 {et} 存在 edge_attr，"
                    "而其他图缺少 edge_attr。请确保所有图来自同一训练配置。"
                )
            attrs.append(g[et].edge_attr.float())

        stacked = torch.stack(attrs, dim=0)  # [n_graphs, E, feat_dim]

        if use_fisher_z:
            stacked = _fisher_z(stacked)

        mean_attr = stacked.mean(dim=0)  # [E, feat_dim]

        if use_fisher_z:
            mean_attr = _fisher_z_inv(mean_attr)

        mean_attrs[et] = mean_attr

    return mean_attrs


def build_mean_graph(
    graphs: List[HeteroData],
    mean_attrs: Dict[Tuple, torch.Tensor],
) -> HeteroData:
    """Construct a new HeteroData containing edge_index and averaged edge_attr.

    Node-type metadata (num_nodes) is copied from the reference graph so that
    the output file is self-consistent.  Temporal node features (x) are NOT
    included because they are subject-specific signals.

    Args:
        graphs: Original graph list (first element used as reference topology).
        mean_attrs: Mapping from edge type to averaged edge_attr tensor.

    Returns:
        A new HeteroData with graph topology and averaged edge weights.
    """
    ref = graphs[0]
    mean_graph = HeteroData()

    # Copy node-type metadata (num_nodes only; no x to avoid subject-specific data)
    for nt in ref.node_types:
        if hasattr(ref[nt], "num_nodes") and ref[nt].num_nodes is not None:
            mean_graph[nt].num_nodes = ref[nt].num_nodes

    # Copy edge topology + averaged edge attributes
    for et in ref.edge_types:
        if not hasattr(ref[et], "edge_index") or ref[et].edge_index is None:
            continue
        mean_graph[et].edge_index = ref[et].edge_index.clone()
        if et in mean_attrs:
            mean_graph[et].edge_attr = mean_attrs[et]

    return mean_graph


def print_statistics(
    mean_attrs: Dict[Tuple, torch.Tensor],
    n_graphs: int,
    use_fisher_z: bool,
) -> None:
    """Print summary statistics for each edge type's averaged edge_attr."""
    print()
    print("=" * 60)
    print(f"  平均图统计信息  ({n_graphs} 个图)")
    print("=" * 60)
    if use_fisher_z:
        print("  ※ 已使用 Fisher z 变换（相关系数模式）")
    print()

    for et, attr in sorted(mean_attrs.items(), key=lambda x: str(x[0])):
        flat = attr.squeeze().float()
        print(f"  边类型: {et}")
        print(f"    边数量  : {flat.numel()}")
        print(f"    最小值  : {flat.min().item():.6f}")
        print(f"    最大值  : {flat.max().item():.6f}")
        print(f"    均值    : {flat.mean().item():.6f}")
        print(f"    标准差  : {flat.std().item():.6f}")
        print()

    print("=" * 60)


def plot_distribution(
    mean_attrs: Dict[Tuple, torch.Tensor],
    output_path: Path,
    n_graphs: int,
    use_fisher_z: bool,
) -> None:
    """Save a histogram of the averaged edge-weight distribution.

    Args:
        mean_attrs: Mapping from edge type to averaged edge_attr tensor.
        output_path: Where to save the PNG file.
        n_graphs: Number of graphs used for the average (for title).
        use_fisher_z: Whether Fisher z-transform was applied (for annotation).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 未安装，跳过可视化。安装方法: pip install matplotlib")
        return

    n_types = len(mean_attrs)
    if n_types == 0:
        logger.warning("没有可用的 edge_attr，跳过可视化。")
        return

    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 4), squeeze=False)

    for idx, (et, attr) in enumerate(
        sorted(mean_attrs.items(), key=lambda x: str(x[0]))
    ):
        ax = axes[0][idx]
        flat = attr.squeeze().float().numpy()
        ax.hist(flat, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
        ax.set_xlabel("平均边权值")
        ax.set_ylabel("边数量")
        src, rel, dst = et
        ax.set_title(f"{src} → {dst}\n({rel})")
        ax.grid(True, alpha=0.3)

        # Annotate with stats
        textstr = (
            f"n={flat.size}\n"
            f"min={flat.min():.3f}\n"
            f"max={flat.max():.3f}\n"
            f"mean={flat.mean():.3f}"
        )
        ax.text(
            0.97, 0.97, textstr,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4),
        )

    fisher_note = "（Fisher z 变换）" if use_fisher_z else ""
    fig.suptitle(
        f"TwinBrain 平均边权分布{fisher_note}  —  {n_graphs} 个图",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("📊 边权分布图已保存至: %s", output_path)


# ---------------------------------------------------------------------------
# Public API (callable from other modules / tests)
# ---------------------------------------------------------------------------


def compute_mean_graph(
    cache_dir: Path,
    output_path: Optional[Path] = None,
    pattern: str = "*.pt",
    use_fisher_z: bool = False,
    plot: bool = False,
    plot_output: Optional[Path] = None,
) -> HeteroData:
    """Compute and save the mean graph from a folder of HeteroData cache files.

    This is the primary entry-point callable from other modules.

    Args:
        cache_dir: Directory containing ``*.pt`` graph cache files.
        output_path: Where to save ``mean_graph.pt``.  Defaults to
            ``cache_dir / mean_graph.pt``.
        pattern: Glob pattern used to discover files inside *cache_dir*.
            Defaults to ``"*.pt"``.  The file ``mean_graph.pt`` is always
            excluded to avoid circular accumulation.
        use_fisher_z: Average Pearson correlation weights in z-space (apply
            ``atanh`` before averaging, ``tanh`` afterwards).
        plot: If *True*, save an edge-weight distribution histogram.
        plot_output: Path for the histogram PNG.  Defaults to
            ``output_path.parent / mean_graph_dist.png``.

    Returns:
        The constructed mean HeteroData object (also persisted to disk).

    Raises:
        FileNotFoundError: If *cache_dir* does not exist or no matching files
            are found.
        ValueError: If the graphs have inconsistent edge types or edge_index.
        RuntimeError: If a file cannot be loaded.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"缓存文件夹不存在: {cache_dir}")

    # Discover .pt files, always exclude the output file itself
    _output = output_path if output_path is not None else cache_dir / "mean_graph.pt"
    output_path = Path(_output)

    files = sorted(
        p for p in cache_dir.glob(pattern)
        if p.name != output_path.name
    )

    if len(files) == 0:
        raise FileNotFoundError(
            f"在 {cache_dir} 中未找到与模式 '{pattern}' 匹配的图文件。"
        )

    logger.info("发现 %d 个图文件:", len(files))
    for f in files:
        logger.info("  %s", f.name)

    # Load
    graphs = load_graphs(files)
    logger.info("已加载 %d 个图。", len(graphs))

    # Consistency check
    check_edge_index_consistency(graphs, files)

    # Average
    mean_attrs = compute_mean_edge_attrs(graphs, use_fisher_z=use_fisher_z)

    # Build output graph
    mean_g = build_mean_graph(graphs, mean_attrs)

    # Statistics
    print_statistics(mean_attrs, n_graphs=len(graphs), use_fisher_z=use_fisher_z)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mean_g, output_path)
    logger.info("✔ mean_graph.pt 已保存至: %s", output_path)

    # Optional visualization
    if plot:
        _plot_out = plot_output if plot_output is not None else (
            output_path.parent / "mean_graph_dist.png"
        )
        plot_distribution(
            mean_attrs,
            output_path=Path(_plot_out),
            n_graphs=len(graphs),
            use_fisher_z=use_fisher_z,
        )

    return mean_g


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compute_mean_graph",
        description=(
            "TwinBrain 平均图生成脚本：\n"
            "将文件夹内所有图缓存文件 (*.pt) 合并为平均图 mean_graph.pt。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python compute_mean_graph.py outputs/graph_cache\n"
            "  python compute_mean_graph.py outputs/graph_cache --fisher-z --plot\n"
            "  python compute_mean_graph.py outputs/graph_cache \\\n"
            "      --output results/mean_graph.pt --log-level DEBUG\n"
        ),
    )
    p.add_argument(
        "cache_dir",
        type=str,
        help="包含图缓存文件 (*.pt) 的文件夹路径。",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（默认: <cache_dir>/mean_graph.pt）。",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="*.pt",
        help="文件匹配模式（默认: *.pt）。",
    )
    p.add_argument(
        "--fisher-z",
        action="store_true",
        help="使用 Fisher z 变换对相关系数进行平均（推荐用于 Pearson 边权）。",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="保存边权分布直方图（需要 matplotlib）。",
    )
    p.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="直方图保存路径（默认: <output_dir>/mean_graph_dist.png）。",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认: INFO）。",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry-point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, args.log_level),
        stream=sys.stdout,
    )

    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output) if args.output else None
    plot_output = Path(args.plot_output) if args.plot_output else None

    try:
        compute_mean_graph(
            cache_dir=cache_dir,
            output_path=output_path,
            pattern=args.pattern,
            use_fisher_z=args.fisher_z,
            plot=args.plot,
            plot_output=plot_output,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("❌ %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
