"""
val.py — TwinBrain V5 独立验证脚本
====================================

在无需重新训练的情况下，对已保存的检查点在图缓存数据上运行完整验证，
并输出与训练日志完全一致的指标报告。

使用方法
--------

1. 最简用法（自动从检查点目录推断 config.yaml 和缓存目录）::

    python val.py --checkpoint outputs/twinbrain_v5_20240304/best_model.pt

2. 指定缓存目录::

    python val.py --checkpoint best_model.pt --cache-dir outputs/graph_cache

3. 指定配置文件::

    python val.py --checkpoint best_model.pt --config configs/my_config.yaml

4. 将所有缓存图都纳入验证（不做训练/验证集划分）::

    python val.py --checkpoint best_model.pt --all

5. 仅验证指定被试/任务（按文件名前缀过滤）::

    python val.py --checkpoint best_model.pt --subject sub-029

配置推断顺序
-----------
1. ``--config`` 显式指定的路径；
2. 检查点同目录下的 ``config.yaml``（训练时 ``save_config()`` 自动保存）；
3. ``configs/default.yaml``。

缓存目录推断顺序
---------------
1. ``--cache-dir`` 显式指定的路径；
2. config 中 ``data.cache.dir``（相对路径相对项目根目录解析）；
3. ``outputs/graph_cache``（默认值）。

输出
----
* 控制台：与训练循环一致的指标表格（val_loss、pred_r2、r2、decorr、ar1_r2 等）。
* JSON 文件（可选，``--output`` 指定路径）：完整指标 dict，便于后续比较。
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch_geometric.data import HeteroData

# ── 确保项目根目录在 sys.path 上 ──────────────────────────────────────────────
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# NOTE: We do NOT import from main.py at module level.
# main.py → data/loaders.py → mne (optional heavy dependency).
# All main.py imports are deferred to the functions that need them so that
# "import val" or "from val import _helper" works in mne-free environments
# (e.g. CI runners).  The imports execute on first function call.
from models.graph_native_mapper import GraphNativeBrainMapper
from models.graph_native_system import GraphNativeTrainer
from utils.helpers import set_seed, setup_logging

# Module-level logger; replaced by setup_logging() in run_val().
logger = logging.getLogger("twinbrain_val")


# ---------------------------------------------------------------------------
# Constants (inlined from main.py to avoid the module-level mne import)
# ---------------------------------------------------------------------------

# Format: {subject_id}_{task_str}_{8-hex-char hash}.pt
# Non-greedy group 2 ensures the last underscore-separated segment is
# captured as the hash even when task_str itself contains underscores.
_CACHE_FILENAME_RE = re.compile(r"^(sub-[^_]+)_(.+?)_([0-9a-f]{8})\.pt$")

_DEFAULT_CACHE_DIR = "outputs/graph_cache"


# ---------------------------------------------------------------------------
# Config loader (lazy-imports main.load_config)
# ---------------------------------------------------------------------------


def _load_config(config_path: Optional[str] = None) -> dict:
    """Load a TwinBrain YAML config, merging onto default.yaml when needed.

    This is a thin wrapper around ``main.load_config`` that is only called
    at runtime (inside functions), so importing ``val.py`` does NOT trigger
    the ``main.py`` → ``data/loaders.py`` → ``mne`` import chain.
    """
    from main import load_config as _main_load_config  # lazy import
    return _main_load_config(config_path)


# ---------------------------------------------------------------------------
# Graph-cache loading
# ---------------------------------------------------------------------------


def _load_graphs_from_cache(
    cache_dir: Path,
    config: dict,
    mapper: GraphNativeBrainMapper,
    subject_filter: Optional[str] = None,
) -> List[HeteroData]:
    """Load all `.pt` graph-cache files from *cache_dir*.

    Mirrors the cache-hit path in ``main.py::build_graphs()``:

    1. Load raw ``HeteroData`` from disk.
    2. Re-attach runtime metadata (``subject_idx``, ``run_idx``, ``task_id``,
       ``subject_id_str``).
    3. Re-create cross-modal edges from node features (they are deliberately
       not stored in cache files — see AGENTS.md V5.30 note).
    4. Apply ``extract_windowed_samples`` if windowed_sampling is enabled.

    Parameters
    ----------
    cache_dir:
        Directory containing ``*.pt`` graph cache files.
    config:
        Loaded TwinBrain config dict.
    mapper:
        Initialised ``GraphNativeBrainMapper`` (used for cross-modal edges).
    subject_filter:
        If set (e.g. ``"sub-029"``), only load files whose filename starts
        with this prefix.

    Returns
    -------
    List of ``HeteroData`` windows ready for ``GraphNativeTrainer.validate()``.
    """
    from main import extract_windowed_samples  # lazy import (avoids mne)

    pt_files = sorted(cache_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(
            f"No .pt graph-cache files found in {cache_dir}.\n"
            "  • Run training with data.cache.enabled=true first, OR\n"
            "  • Specify a different directory with --cache-dir."
        )

    if subject_filter:
        pt_files = [f for f in pt_files if f.name.startswith(subject_filter)]
        if not pt_files:
            raise FileNotFoundError(
                f"No .pt files matching subject prefix '{subject_filter}' "
                f"in {cache_dir}."
            )

    w_cfg: dict = config.get("windowed_sampling", {})
    k_cross_modal: int = config.get("graph", {}).get("k_cross_modal", 5)
    disable_cross: bool = config.get("ablation", {}).get(
        "disable_cross_modal_edges", False
    )

    # Build subject_to_idx deterministically from file names
    all_subject_ids = sorted(
        {
            _m.group(1)
            for f in pt_files
            if (_m := _CACHE_FILENAME_RE.match(f.name)) is not None
        }
    )
    subject_to_idx: Dict[str, int] = {
        sid: i for i, sid in enumerate(all_subject_ids)
    }

    graphs: List[HeteroData] = []
    run_counter = 0

    for pt in pt_files:
        _m = _CACHE_FILENAME_RE.match(pt.name)
        if _m is None:
            logger.debug(f"跳过文件（名称格式不匹配）: {pt.name}")
            continue

        subject_id, task_str, _hash = _m.group(1), _m.group(2), _m.group(3)
        task: Optional[str] = None if task_str == "notask" else task_str

        try:
            full_graph: HeteroData = torch.load(
                pt, map_location="cpu", weights_only=False
            )
        except Exception as exc:
            logger.warning(f"加载缓存失败 ({pt.name}): {exc}，跳过")
            continue

        # ── 跨模态边：每次加载时重建（不存储在缓存文件中）─────────────────
        if (
            not disable_cross
            and "fmri" in full_graph.node_types
            and "eeg" in full_graph.node_types
        ):
            cross = mapper.create_simple_cross_modal_edges(
                full_graph, k_cross_modal=k_cross_modal
            )
            if cross is not None:
                full_graph["eeg", "projects_to", "fmri"].edge_index = cross[0]
                full_graph["eeg", "projects_to", "fmri"].edge_attr = cross[1]

        # ── 运行时元数据 ───────────────────────────────────────────────────
        full_graph.subject_idx = torch.tensor(
            subject_to_idx.get(subject_id, 0), dtype=torch.long
        )
        full_graph.run_idx = torch.tensor(run_counter, dtype=torch.long)
        full_graph.task_id = task or ""
        full_graph.subject_id_str = subject_id

        windows = extract_windowed_samples(full_graph, w_cfg, logger)
        graphs.extend(windows)
        run_counter += 1

        logger.debug(
            f"已加载: {pt.name}  → {len(windows)} 个窗口"
            f"  [节点: {list(full_graph.node_types)}, "
            f"边: {[str(et) for et in full_graph.edge_types]}]"
        )

    logger.info(
        f"📂 从 {cache_dir} 加载了 {run_counter} 条 run "
        f"（{len(graphs)} 个窗口）"
    )
    return graphs


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _infer_embedding_sizes(state_dict: dict) -> Dict[str, int]:
    """Infer num_subjects / num_runs from checkpoint state_dict shapes.

    Parameters
    ----------
    state_dict:
        ``model_state_dict`` from a TwinBrain checkpoint.

    Returns
    -------
    dict with optional keys ``num_subjects`` and ``num_runs``.
    """
    result: Dict[str, int] = {}
    if "subject_embed.weight" in state_dict:
        result["num_subjects"] = int(state_dict["subject_embed.weight"].shape[0])
    if "run_embed.weight" in state_dict:
        result["num_runs"] = int(state_dict["run_embed.weight"].shape[0])
    return result


def _load_model_from_checkpoint(
    checkpoint_path: Path,
    config: dict,
    device: str,
) -> tuple:
    """Load model architecture from config and weights from checkpoint.

    Returns
    -------
    (model, epoch)
        model: Loaded ``GraphNativeBrainModel`` in eval mode on *device*.
        epoch: The epoch number saved in the checkpoint (int or None).
    """
    from main import create_model  # lazy import (avoids mne at module level)

    logger.info(f"📥 加载检查点: {checkpoint_path}")
    # weights_only=False is required because checkpoints may contain Python
    # objects (e.g. history dicts, HeteroData fragments).  Only load
    # checkpoints from trusted sources — arbitrary pickle payloads can
    # execute code on deserialization.
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    state_dict = checkpoint["model_state_dict"]
    epoch: Optional[int] = checkpoint.get("epoch")

    # Auto-infer embedding table sizes from weight shapes so the model
    # architecture exactly matches what was saved — no config edits needed.
    embed_sizes = _infer_embedding_sizes(state_dict)
    num_subjects = embed_sizes.get("num_subjects", 0)
    num_runs = embed_sizes.get("num_runs", 0)

    if num_subjects > 0:
        logger.info(f"  从检查点推断 num_subjects={num_subjects}")
    if num_runs > 0:
        logger.info(f"  从检查点推断 num_runs={num_runs}")

    _dummy_logger = logging.getLogger("twinbrain_val.create_model")
    model = create_model(
        config, _dummy_logger, num_subjects=num_subjects, num_runs=num_runs
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"  检查点中缺少 {len(missing)} 个权重键（可能是架构版本差异）")
        for k in missing[:5]:
            logger.debug(f"    缺失: {k}")
    if unexpected:
        logger.warning(
            f"  检查点中有 {len(unexpected)} 个未知权重键（可能是旧版本检查点）"
        )
        for k in unexpected[:5]:
            logger.debug(f"    多余: {k}")

    model.to(device)
    model.eval()
    logger.info(
        f"  ✅ 模型权重加载完成 (epoch={epoch}, "
        f"num_subjects={num_subjects}, num_runs={num_runs})"
    )
    return model, epoch


# ---------------------------------------------------------------------------
# Config / cache-dir resolution
# ---------------------------------------------------------------------------


def _resolve_config(
    config_arg: Optional[str], checkpoint_path: Path
) -> dict:
    """Find and load the best-matching config for a checkpoint.

    Priority:
    1. Explicit ``--config`` argument.
    2. ``config.yaml`` in the same directory as the checkpoint
       (saved by ``main.py`` via ``save_config()``).
    3. Default ``configs/default.yaml``.
    """
    if config_arg is not None:
        cfg_path = Path(config_arg)
        if not cfg_path.exists():
            raise FileNotFoundError(f"指定的配置文件不存在: {cfg_path}")
        logger.info(f"📄 使用配置文件: {cfg_path}")
        return _load_config(str(cfg_path))

    sibling = checkpoint_path.parent / "config.yaml"
    if sibling.exists():
        logger.info(f"📄 从检查点目录读取配置: {sibling}")
        return _load_config(str(sibling))

    default_cfg = _HERE / "configs" / "default.yaml"
    logger.info(f"📄 未找到指定配置，使用默认配置: {default_cfg}")
    return _load_config(str(default_cfg))


def _resolve_cache_dir(cache_dir_arg: Optional[str], config: dict) -> Path:
    """Resolve the graph-cache directory.

    Priority:
    1. Explicit ``--cache-dir`` argument.
    2. ``data.cache.dir`` from config (resolved relative to project root).
    3. ``outputs/graph_cache`` (hard-coded default).
    """
    if cache_dir_arg is not None:
        return Path(cache_dir_arg)

    cfg_dir = (
        config.get("data", {}).get("cache", {}).get("dir", _DEFAULT_CACHE_DIR)
    )
    p = Path(cfg_dir)
    if not p.is_absolute():
        p = _HERE / p
    return p


# ---------------------------------------------------------------------------
# Metric reporting
# ---------------------------------------------------------------------------


def _format_metrics(val_loss: float, r2_dict: dict) -> str:
    """Single-line metric summary (mirrors training-loop format)."""
    pred_items = {
        k: v for k, v in r2_dict.items()
        if k.startswith("pred_r2_") and "_h1_" not in k
    }
    recon_items = {k: v for k, v in r2_dict.items() if k.startswith("r2_")}
    parts = [
        "  ".join(f"{k}={v:.3f}" for k, v in sorted(pred_items.items())),
        "  ".join(f"{k}={v:.3f}" for k, v in sorted(recon_items.items())),
    ]
    return f"val_loss={val_loss:.4f}  " + "  ".join(filter(None, parts))


def _print_full_report(val_loss: float, r2_dict: dict) -> None:
    """Detailed multi-section metric report (mirrors training-loop diagnostics)."""
    sep = "=" * 70
    logger.info(sep)
    logger.info("📊 验证结果报告")
    logger.info(sep)
    logger.info(f"  总体验证损失:  {val_loss:.4f}")
    logger.info("")

    groups = {
        "预测 R²（★ 主要指标，信号空间）": {
            k: v for k, v in r2_dict.items()
            if k.startswith("pred_r2_") and "_h1_" not in k
        },
        "重建 R²（encoder-decoder 能力）": {
            k: v for k, v in r2_dict.items() if k.startswith("r2_")
        },
        "h=1 预测 R²（单步，NPI 可比）": {
            k: v for k, v in r2_dict.items() if k.startswith("pred_r2_h1_")
        },
        "AR(1) 基线 R²（h=1）": {
            k: v for k, v in r2_dict.items() if k.startswith("ar1_r2_h1_")
        },
        "去相关 skill score（h=1，NPI 可比）": {
            k: v for k, v in r2_dict.items() if k.startswith("decorr_h1_")
        },
        "多步去相关 skill score": {
            k: v for k, v in r2_dict.items()
            if k.startswith("decorr_") and "_h1_" not in k
        },
        "AR(1) 多步基线 R²": {
            k: v for k, v in r2_dict.items()
            if k.startswith("ar1_r2_") and "_h1_" not in k
        },
    }

    for title, items in groups.items():
        if not items:
            continue
        logger.info(f"  【{title}】")
        for k, v in sorted(items.items()):
            flag = "✅" if v >= 0 else "⛔"
            logger.info(f"    {flag}  {k:40s} = {v:+.4f}")
        logger.info("")

    logger.info(sep)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def run_val(
    checkpoint: str,
    config_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
    use_all: bool = False,
    subject_filter: Optional[str] = None,
    output_json: Optional[str] = None,
    seed: int = 42,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> Dict:
    """Run validation programmatically (also callable as a library function).

    Parameters
    ----------
    checkpoint:
        Path to a TwinBrain ``.pt`` checkpoint file.
    config_path:
        Optional path to a YAML config.  Inferred from checkpoint directory
        if omitted.
    cache_dir:
        Optional path to the graph-cache directory.  Inferred from config
        if omitted.
    device:
        ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect from config).
    use_all:
        If ``True``, validate on ALL cached graphs (skip train/val split).
    subject_filter:
        If set, only load files whose filename starts with this prefix.
    output_json:
        If set, write the metrics dict to this path as JSON.
    seed:
        Random seed — ensures the val split matches the training split.
    log_level:
        Logging verbosity.
    log_file:
        Optional path to a log file.

    Returns
    -------
    dict
        Keys: ``checkpoint``, ``saved_epoch``, ``n_val_samples``,
        ``val_loss`` (float), ``r2_dict`` (dict of metric name → float).
    """
    global logger
    logger = setup_logging(
        log_file=Path(log_file) if log_file else None, level=log_level
    )
    set_seed(seed)

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    # ── 1. Config ──────────────────────────────────────────────────────────
    config = _resolve_config(config_path, checkpoint_path)

    # ── 2. Device ──────────────────────────────────────────────────────────
    if device is None:
        device = config.get("device", {}).get("type", "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("⚠️  CUDA 不可用，回退到 CPU")
            device = "cpu"
    logger.info(f"🖥️  设备: {device}")

    # ── 3. Cache dir ───────────────────────────────────────────────────────
    resolved_cache_dir = _resolve_cache_dir(cache_dir, config)
    if not resolved_cache_dir.exists():
        raise FileNotFoundError(
            f"图缓存目录不存在: {resolved_cache_dir}\n"
            "  请先运行训练并启用 data.cache.enabled=true，\n"
            f"  或使用 --cache-dir 指定正确路径。"
        )
    logger.info(f"📁 图缓存目录: {resolved_cache_dir}")

    # ── 4. Load graphs ─────────────────────────────────────────────────────
    mapper = GraphNativeBrainMapper(device="cpu")
    graphs = _load_graphs_from_cache(
        resolved_cache_dir, config, mapper, subject_filter=subject_filter
    )
    if not graphs:
        raise ValueError("缓存目录中未找到有效的图数据。")

    # ── 5. Train / val split (mirrors main.py train_model) ────────────────
    if not use_all:
        import random as _rnd
        from collections import defaultdict as _ddict

        windowed = config.get("windowed_sampling", {}).get("enabled", False)
        rng = _rnd.Random(seed)

        if windowed and hasattr(graphs[0], "run_idx"):
            run_groups: dict = _ddict(list)
            for g in graphs:
                run_groups[g.run_idx.item()].append(g)
            run_keys = sorted(run_groups.keys())
            rng.shuffle(run_keys)
            min_val = max(1, len(run_keys) // 10)
            val_keys = run_keys[len(run_keys) - min_val:]
            val_graphs = [g for k in val_keys for g in run_groups[k]]
            logger.info(
                f"run-level 验证集: {len(val_graphs)} 个窗口"
                f"（{len(val_keys)} 个 run）/ 全集 {len(graphs)} 个窗口"
                f"  [提示: --all 可验证全部数据]"
            )
        else:
            shuffled = graphs.copy()
            rng.shuffle(shuffled)
            min_val = max(1, len(shuffled) // 10)
            val_graphs = shuffled[len(shuffled) - min_val:]
            logger.info(
                f"样本级验证集: {len(val_graphs)} / {len(graphs)} 个样本"
                f"  [提示: --all 可验证全部数据]"
            )
    else:
        val_graphs = graphs
        logger.info(f"验证全部 {len(val_graphs)} 个样本（--all 模式）")

    # ── 6. Build model & load checkpoint ──────────────────────────────────
    model, saved_epoch = _load_model_from_checkpoint(
        checkpoint_path, config, device
    )
    if saved_epoch is not None:
        logger.info(f"  检查点保存于 epoch {saved_epoch}")

    # ── 7. Minimal inference-only trainer ──────────────────────────────────
    # use_adaptive_loss=False: validation does not need the loss-balancer state
    # that was saved in the training checkpoint.
    trainer = GraphNativeTrainer(
        model=model,
        node_types=config["data"]["modalities"],
        learning_rate=1e-4,        # irrelevant for validation
        use_adaptive_loss=False,
        use_eeg_enhancement=False,
        use_amp=(
            config.get("device", {}).get("use_amp", False)
            and "cuda" in str(device)
        ),
        use_torch_compile=False,   # skip compile overhead
        device=device,
    )

    # ── 8. GPU pre-load (if available) ────────────────────────────────────
    if "cuda" in str(device):
        logger.info(f"📦 预加载 {len(val_graphs)} 个验证样本到 {device}...")
        val_graphs = [g.to(trainer.device) for g in val_graphs]

    # ── 9. Validate ────────────────────────────────────────────────────────
    logger.info(f"🔍 开始验证 ({len(val_graphs)} 个样本)...")
    val_loss, r2_dict = trainer.validate(val_graphs)

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    # ── 10. Report ─────────────────────────────────────────────────────────
    logger.info(f"✅ {_format_metrics(val_loss, r2_dict)}")
    _print_full_report(val_loss, r2_dict)

    result = {
        "checkpoint": str(checkpoint_path),
        "saved_epoch": saved_epoch,
        "n_val_samples": len(val_graphs),
        "val_loss": val_loss,
        "r2_dict": r2_dict,
    }

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as _f:
            json.dump(result, _f, ensure_ascii=False, indent=2)
        logger.info(f"💾 指标已保存: {out_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="val.py",
        description=(
            "TwinBrain V5 独立验证脚本 — 加载检查点和图缓存，"
            "输出完整验证指标报告。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", required=True, metavar="PATH",
        help="检查点文件路径，例如 outputs/twinbrain_v5_xxx/best_model.pt",
    )
    p.add_argument(
        "--config", default=None, metavar="PATH",
        help=(
            "配置文件路径。若不指定，自动查找检查点同目录的 config.yaml，"
            "否则使用 configs/default.yaml。"
        ),
    )
    p.add_argument(
        "--cache-dir", default=None, dest="cache_dir", metavar="DIR",
        help=f"图缓存目录（含 .pt 文件）。默认从 config 读取或使用 {_DEFAULT_CACHE_DIR}。",
    )
    p.add_argument(
        "--device", default=None, choices=["cpu", "cuda"],
        help="计算设备（默认从 config.device.type 读取）。",
    )
    p.add_argument(
        "--all", action="store_true", dest="use_all",
        help="验证全部缓存图（不做训练/验证集划分）。",
    )
    p.add_argument(
        "--subject", default=None, metavar="PREFIX",
        help="只加载文件名以此前缀开头的缓存文件，例如 sub-029。",
    )
    p.add_argument(
        "--output", default=None, metavar="JSON_PATH",
        help="将指标结果保存为 JSON 文件（可选）。",
    )
    p.add_argument("--seed", type=int, default=42, help="随机种子。")
    p.add_argument(
        "--log-level", default="INFO", dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument("--log-file", default=None, dest="log_file", metavar="PATH")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_val(
        checkpoint=args.checkpoint,
        config_path=args.config,
        cache_dir=args.cache_dir,
        device=args.device,
        use_all=args.use_all,
        subject_filter=args.subject,
        output_json=args.output,
        seed=args.seed,
        log_level=args.log_level,
        log_file=args.log_file,
    )
