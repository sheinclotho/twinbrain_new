"""
TwinBrain V5 主程序
==================

图原生数字孪生脑训练系统

使用方法:
    python main.py --config configs/default.yaml
    
或直接运行:
    python main.py  # 使用默认配置
"""

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Optional
import yaml
import torch
import numpy as np
from torch_geometric.data import HeteroData

# Reduce CUDA memory fragmentation (recommended when reserved >> allocated).
# Set before any CUDA allocations; setdefault preserves user overrides.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from data.loaders import BrainDataLoader
from models.graph_native_mapper import GraphNativeBrainMapper
from models.graph_native_system import GraphNativeBrainModel, GraphNativeTrainer
from utils.helpers import setup_logging, set_seed, save_config, create_output_dir
from utils.visualization import plot_training_curves


def truncate_timeseries(ts: np.ndarray, max_len: int) -> np.ndarray:
    """Truncate timeseries [..., T] to at most max_len timepoints.

    Prevents CUDA OOM caused by very long EEG/fMRI sequences creating
    multi-GB [N, T, hidden] tensors inside the ST-GCN encoder.
    """
    if ts.shape[-1] > max_len:
        return ts[..., :max_len]
    return ts


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "default.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


# ── 时间窗口默认值（神经影像经验值，可通过配置覆盖）──────────────
# fMRI: 50 TRs × TR≈2s = 100s — 覆盖一个完整慢波脑状态周期（Hutchison 2013）
# EEG: 500 pts ÷ 250Hz = 2s — 覆盖 alpha (8-12 Hz) + beta (13-30 Hz) 主要节律
_DEFAULT_FMRI_WINDOW_SIZE = 50
_DEFAULT_EEG_WINDOW_SIZE = 500


def _graph_cache_key(subject_id: str, task: Optional[str], config: dict) -> str:
    """为图缓存生成稳定的文件名。

    文件名内嵌图相关配置参数的 MD5 短哈希（8位），修改 atlas、图拓扑参数或
    max_seq_len 后，旧缓存文件名将不再匹配，系统自动重建。

    当时间窗口采样（windowed_sampling）启用时，缓存存储的是完整 run 的图（
    用全序列计算连通性），对应的缓存键不含 max_seq_len（不截断）。
    """
    w_enabled = config.get('windowed_sampling', {}).get('enabled', False)
    relevant = {
        'graph': config.get('graph', {}),
        'atlas': config['data'].get('atlas', {}),
        # 只有在 windowed_sampling 关闭时才截断，此时 max_seq_len 影响连通性估计
        'max_seq_len': None if w_enabled else config['training'].get('max_seq_len'),
        'modalities': sorted(config['data'].get('modalities', [])),
        'windowed': w_enabled,
        # DTI 开关影响图结构（是否含 ('fmri','structural','fmri') 边）；
        # 修改此选项必须使旧缓存失效，否则切换 DTI 后仍加载无 DTI 边的旧图。
        'dti_structural_edges': config['data'].get('dti_structural_edges', False),
        # fMRI 条件时间段截取：修改此选项改变 fMRI 节点特征的时间维度和连通性估计，
        # 必须使旧缓存失效，否则使用错误时间段的图参与训练。
        'fmri_condition_bounds': config['data'].get('fmri_condition_bounds'),
        # EEG 连通性方法：'correlation' vs 'coherence' 产生不同 edge_index/edge_attr，
        # 切换方法必须使旧缓存失效，否则 coherence 模式会使用 correlation 权重的旧图。
        'eeg_connectivity_method': config['graph'].get('eeg_connectivity_method', 'correlation'),
        # 注意：k_cross_modal 不纳入缓存键。
        # 跨模态边在每次加载缓存时从节点特征动态重建（代价低，仅矩阵乘法），
        # 因此修改 k_cross_modal 无需重建缓存。
    }
    params_hash = hashlib.md5(
        json.dumps(relevant, sort_keys=True).encode()
    ).hexdigest()[:8]
    task_str = task if task else 'notask'
    return f"{subject_id}_{task_str}_{params_hash}.pt"


def extract_windowed_samples(
    full_graph: HeteroData,
    w_cfg: dict,
    logger: logging.Logger,
) -> List[HeteroData]:
    """将一条完整扫描的图切分为多个重叠时间窗口样本（动态功能连接，dFC）。

    设计理念（参见 Hutchison 2013; Chang & Glover 2010）：
    - 图拓扑（edge_index）= 完整 run 的相关性 → 稳定的结构连通性估计
    - 节点特征（x）= 时间窗口切片 → 每个窗口代表一次脑状态快照
    - 多个重叠窗口 = 多个训练样本，且每样本 T = window_size << T_full → 无 OOM

    与朴素截断（max_seq_len）的关键区别：
    - 截断：丢弃 run 末尾数据，且仅产生 1 个训练样本
    - 窗口：覆盖完整 run，产生 N_windows 个样本，每样本均由完整连通性支撑

    Args:
        full_graph: 完整 run 构建的异质图（edge_index 来自全序列相关性估计）
        w_cfg:      windowed_sampling 配置字典
        logger:     日志记录器

    Returns:
        HeteroData 列表；关闭时返回 [full_graph]（与旧行为兼容）
    """
    if not w_cfg.get('enabled', False):
        return [full_graph]

    node_types = full_graph.node_types
    T_per_type = {nt: full_graph[nt].x.shape[1] for nt in node_types}

    # 各模态的窗口大小（单位：该模态的时间步数）
    window_sizes: dict = {}
    for nt in node_types:
        ws = w_cfg.get(f'{nt}_window_size')
        if ws is None:
            # 神经影像经验默认值：fMRI 50 TRs ≈ 100s（一个脑状态周期）；
            # EEG 500 pts = 2s（覆盖 alpha/beta/gamma 主要节律）
            ws = _DEFAULT_FMRI_WINDOW_SIZE if nt == 'fmri' else _DEFAULT_EEG_WINDOW_SIZE
        window_sizes[nt] = int(ws)

    stride_fraction = w_cfg.get('stride_fraction', 0.5)

    # 以 fMRI 作为参考模态（时间步最少，避免分数窗口）
    # 若无 fMRI 则取节点数第一项
    ref_type = 'fmri' if 'fmri' in node_types else node_types[0]

    # ── 跨模态时间对齐（可选）────────────────────────────────────────
    # 默认（cross_modal_align=False）：各模态使用各自的自然时间尺度。
    #   fMRI 50 TRs ≈ 100s（慢血动力学），EEG 500 pts = 2s（快神经振荡）。
    #   适用于：各模态预测自身未来（intra-modal prediction，默认场景）。
    #
    # cross_modal_align=True：强制所有模态窗口覆盖相同的实际时长。
    #   ws_eeg = round(ws_fmri × T_eeg / T_fmri)
    #   适用于：跨模态预测（EEG→fMRI、fMRI→EEG）。
    #   ⚠ 注意：对齐后 EEG 窗口约 12500 pts（500s at 250Hz），
    #            可能导致 CUDA OOM。确保 VRAM 足够后再启用。
    if w_cfg.get('cross_modal_align', False) and ref_type == 'fmri' and 'eeg' in node_types:
        T_fmri_ref = T_per_type['fmri']
        if T_fmri_ref > 0:
            T_eeg = T_per_type['eeg']
            ws_fmri = window_sizes['fmri']
            window_sizes['eeg'] = round(ws_fmri * (T_eeg / T_fmri_ref))
            logger.debug(
                f"跨模态时间对齐已启用: EEG 窗口调整为 {window_sizes['eeg']} pts"
                f" (与 fMRI {ws_fmri} TRs 覆盖相同实际时长)"
            )
    ws_ref = window_sizes[ref_type]
    T_ref = T_per_type[ref_type]
    stride = max(1, int(ws_ref * stride_fraction))

    if ws_ref >= T_ref:
        # 窗口覆盖完整序列：无法再分割，退化为原始单样本
        logger.debug(
            f"窗口大小 ({ref_type}: {ws_ref}) ≥ 序列长度 ({T_ref})，"
            f" 窗口采样退化为单样本。若需多窗口，请减小 window_size 或"
            f" 增大序列（设 max_seq_len: null）。"
        )
        return [full_graph]

    window_starts = list(range(0, T_ref - ws_ref + 1, stride))

    windows: List[HeteroData] = []
    for t_start_ref in window_starts:
        win = HeteroData()

        # 共享图拓扑（所有窗口使用相同的 edge_index，来自全序列连通性估计）
        for edge_type in full_graph.edge_types:
            win[edge_type].edge_index = full_graph[edge_type].edge_index
            if hasattr(full_graph[edge_type], 'edge_attr'):
                win[edge_type].edge_attr = full_graph[edge_type].edge_attr

        # 按比例对齐各模态的窗口切片
        for nt in node_types:
            T_nt = T_per_type[nt]
            ws_nt = window_sizes[nt]
            # 根据参考模态时间步比例，等比例定位该模态的起始点
            # 使用 int() 而非 round()：数组索引用整数截断语义更可预期
            t_start_nt = int(t_start_ref * (T_nt / T_ref))
            t_end_nt = t_start_nt + ws_nt

            x_full = full_graph[nt].x  # [N, T, C]
            if t_end_nt > T_nt:
                # 末尾窗口越界：用零填充保持固定 T=ws_nt
                x_slice = x_full[:, t_start_nt:, :]
                pad_len = ws_nt - x_slice.shape[1]
                pad = torch.zeros(
                    x_slice.shape[0], pad_len, x_slice.shape[2],
                    dtype=x_slice.dtype,
                )
                x_slice = torch.cat([x_slice, pad], dim=1)
            else:
                x_slice = x_full[:, t_start_nt:t_end_nt, :]

            win[nt].x = x_slice
            # 复制「窗口静态」属性：在同一 run 的所有窗口间不随时间变化。
            # - num_nodes: 节点数，用于 PyG 内部图构建
            # - pos: 空间坐标（EEG 电极坐标 / fMRI ROI 质心），用于可视化和距离加权边
            # - sampling_rate: 采样率，用于日志摘要（不随窗口变化）
            # - labels: ROI/通道名称，用于可解释性分析（例如 visualization.py）
            # 有意不复制的属性（非窗口静态或当前未读取）：
            # - atlas_mapping: EEG→atlas ROI 映射，SET 后从未被 READ，暂不传播
            # - temporal_length: 全序列长度，窗口中应使用 x.shape[1] 动态获取
            for attr in ('num_nodes', 'pos', 'sampling_rate', 'labels'):
                if hasattr(full_graph[nt], attr):
                    setattr(win[nt], attr, getattr(full_graph[nt], attr))

        windows.append(win)

    # 复制图级属性（不属于任何节点类型，但对整个样本共享的元数据）
    # 当前图级属性：
    #   subject_idx:    被试整数索引，供 subject embedding（个性化）使用（AGENTS.md §九 Gap 2）
    #   run_idx:        扫描 run 的全局整数索引，供 train_model run-level 划分
    #   task_id:        任务名字符串（如 'GRADON'），供日志摘要按任务统计
    #   subject_id_str: 被试 ID 字符串（如 'sub-01'），供日志摘要按被试统计
    for attr in ('subject_idx', 'run_idx', 'task_id', 'subject_id_str'):
        if hasattr(full_graph, attr):
            for win in windows:
                setattr(win, attr, getattr(full_graph, attr))

    return windows


def build_graphs(config: dict, logger: logging.Logger):
    """加载数据并构建图结构（含缓存感知的按需数据加载）。

    将原先拆成两步的「准备数据」+「构建图」合并为一步：
    对每个 (被试, 任务) 组合先检查图缓存，命中则直接加载缓存图（完全跳过
    EEG/fMRI 原始数据的读取与预处理），未命中才调用 BrainDataLoader 加载
    原始数据、执行预处理并构建图后写入缓存。

    这解决了之前「即使图缓存已存在，每次运行仍会从头预处理所有原始数据」的问题。
    """
    logger.info("=" * 60)
    logger.info("步骤 1-2/4: 加载数据 & 构建图结构")
    logger.info("=" * 60)

    # ── 初始化数据加载器 ───────────────────────────────────────────
    fmri_task_mapping = config['data'].get('fmri_task_mapping') or {}
    if fmri_task_mapping:
        logger.info(f"fMRI 任务映射: {fmri_task_mapping}")
        logger.info(
            "  说明：配置了映射后，任务发现仅扫描 EEG 文件，"
            "fMRI 文件由映射关系确定（避免 fMRI-only 任务产生无 EEG 的单模态图）。"
        )
    data_loader = BrainDataLoader(
        data_root=config['data']['root_dir'],
        modalities=config['data']['modalities'],
        fmri_task_mapping=fmri_task_mapping if fmri_task_mapping else None,
    )
    
    _MIN_VOLUMES = 10  # Shared threshold for minimum valid fMRI timepoints

    def process_fmri_timeseries(fmri_data, min_volumes=_MIN_VOLUMES):
        """Extract and normalize fMRI timeseries.

        Handles all common input shapes:
        - 4-D [X, Y, Z, T]: raw volumetric fMRI → averaged to [1, T]
        - 3-D [N_rois, T, ?] or [X, Y, T]: reshaped → averaged to [1, T]
        - 2-D [N_rois, T] or [T, N_rois]: pre-parcellated ROI data → ALL ROIs
          kept as separate nodes [N_rois, T] (no averaging).

        Returns (timeseries [N_rois, T], error_or_None).
        """
        if fmri_data.ndim == 4:
            n_volumes = fmri_data.shape[-1]
            if n_volumes < min_volumes:
                return None, f"Too few volumes: {n_volumes} < {min_volumes}"
            # Average all in-mask voxels — single timeseries
            fmri_ts = fmri_data.reshape(-1, n_volumes).mean(axis=0)
            fmri_ts = (fmri_ts - fmri_ts.mean()) / (fmri_ts.std() + 1e-8)
            return fmri_ts.reshape(1, -1), None

        elif fmri_data.ndim == 3:
            n_volumes = fmri_data.shape[-1]
            if n_volumes < min_volumes:
                return None, f"Too few volumes: {n_volumes} < {min_volumes}"
            fmri_ts = fmri_data.reshape(-1, n_volumes).mean(axis=0)
            fmri_ts = (fmri_ts - fmri_ts.mean()) / (fmri_ts.std() + 1e-8)
            return fmri_ts.reshape(1, -1), None

        elif fmri_data.ndim == 2:
            # Already ROI timeseries — preserve all ROIs as separate graph nodes.
            # Ensure layout is [N_rois, T].
            if fmri_data.shape[0] > fmri_data.shape[1]:
                fmri_data = fmri_data.T
            N_rois, T = fmri_data.shape
            if T < min_volumes:
                return None, f"Too few timepoints: {T} < {min_volumes}"
            # Normalise each ROI independently
            mean = fmri_data.mean(axis=1, keepdims=True)
            std = fmri_data.std(axis=1, keepdims=True) + 1e-8
            return (fmri_data - mean) / std, None

        else:
            return None, f"Unsupported fMRI shape: {fmri_data.shape}"

    def _parcellate_fmri_with_atlas(fmri_img, atlas_path: Path) -> Optional[np.ndarray]:
        """Apply atlas parcellation to extract per-ROI timeseries.

        Uses nilearn NiftiLabelsMasker which handles resampling automatically.
        Returns [N_rois, T] float32 array, or None on failure.

        Why this matters: without parcellation fMRI is collapsed to a single node
        ([1, T]), making graph convolution meaningless.  With the Schaefer200 atlas
        we get ~190-200 anatomically meaningful nodes — the actual design intent.
        The exact count depends on how many atlas parcels overlap with the
        subject's fMRI brain mask after resampling; see logging below.
        """
        # Infer the expected number of parcels from the atlas file name.
        # Pattern: "schaeferNNN" → NNN parcels; "aal116" → 116; etc.
        # Falls back to None (unknown) for non-standard names.
        import re as _re
        _m = _re.search(r'_?(\d{2,4})_?[Pp]arcels', atlas_path.name)
        if _m is None:
            _m = _re.search(r'[Ss]chaefer(\d+)', atlas_path.name)
        expected_n_rois: Optional[int] = int(_m.group(1)) if _m else None
        try:
            try:
                from nilearn.maskers import NiftiLabelsMasker  # nilearn >= 0.10
            except ImportError:
                from nilearn.input_data import NiftiLabelsMasker  # nilearn < 0.10
            masker = NiftiLabelsMasker(
                labels_img=str(atlas_path),
                standardize=True,
                detrend=True,
            )
            roi_ts = masker.fit_transform(fmri_img)  # [T, N_rois]
            n_rois = roi_ts.shape[1]
            # NiftiLabelsMasker excludes atlas parcels that have no valid voxels
            # in the fMRI brain mask after resampling the atlas to the EPI
            # resolution.  For Schaefer200 it is normal to get 190-199 ROIs:
            # the excluded parcels are typically temporal-pole and orbitofrontal
            # regions prone to EPI signal dropout, or very small parcels that
            # lose all voxels when resampled from 1 mm to ~3 mm.
            # This is expected behaviour — the actual ROI count varies by subject
            # and acquisition and does NOT indicate a problem.
            if expected_n_rois is not None and n_rois < expected_n_rois:
                logger.info(
                    f"Atlas parcellation ({atlas_path.name}): "
                    f"{n_rois}/{expected_n_rois} ROIs extracted. "
                    f"({expected_n_rois - n_rois} parcels excluded — no overlap with "
                    f"subject brain mask after resampling; expected for EPI data.)"
                )
            else:
                logger.info(
                    f"Atlas parcellation ({atlas_path.name}): {n_rois} ROIs extracted."
                    + (" (all parcels have valid voxel coverage)" if expected_n_rois and n_rois == expected_n_rois else "")
                )
            if roi_ts.shape[0] < _MIN_VOLUMES:
                logger.warning(
                    f"Atlas parcellation produced only {roi_ts.shape[0]} timepoints; skipping."
                )
                return None
            if roi_ts.shape[1] < 2:
                logger.warning(
                    f"Atlas parcellation produced only {roi_ts.shape[1]} ROIs; skipping."
                )
                return None
            return roi_ts.T.astype(np.float32)  # [N_rois, T]
        except Exception as e:
            logger.warning(f"Atlas parcellation failed ({e}); falling back to single-node fMRI.")
            return None

    # 初始化图映射器
    mapper = GraphNativeBrainMapper(
        atlas_name=config['data']['atlas']['name'],
        add_self_loops=config['graph']['add_self_loops'],
        make_undirected=config['graph']['make_undirected'],
        k_nearest_fmri=config['graph'].get('k_nearest_fmri', 20),
        k_nearest_eeg=config['graph'].get('k_nearest_eeg', 10),
        threshold_fmri=config['graph'].get('threshold_fmri', 0.3),
        threshold_eeg=config['graph'].get('threshold_eeg', 0.2),
        device=config['device']['type'],
        eeg_connectivity_method=config['graph'].get('eeg_connectivity_method', 'correlation'),
    )

    # Resolve atlas file path once (relative to project root)
    atlas_file = Path(__file__).parent / config['data']['atlas']['file']
    if atlas_file.exists():
        logger.info(
            f"Atlas parcellation enabled: {atlas_file.name} → up to 200 fMRI ROI nodes"
        )
    else:
        logger.warning(
            f"Atlas file not found at {atlas_file}; fMRI will use single-node fallback."
        )
    
    # ── 图缓存设置 ──────────────────────────────────────────────
    cache_cfg = config['data'].get('cache', {})
    cache_enabled = cache_cfg.get('enabled', False)
    cache_dir: Optional[Path] = None
    if cache_enabled:
        # 相对路径相对于项目根目录（即 main.py 所在目录）解析，
        # 而非运行时工作目录（CWD）——这保证了无论用户从哪个目录启动
        # 脚本，缓存路径始终固定一致，不会因 CWD 不同导致「找不到缓存」。
        _cache_dir_cfg = cache_cfg.get('dir', 'outputs/graph_cache')
        _cache_dir_path = Path(_cache_dir_cfg)
        cache_dir = (
            _cache_dir_path
            if _cache_dir_path.is_absolute()
            else Path(__file__).parent / _cache_dir_path
        )
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"图缓存已启用: {cache_dir}")
        except OSError as e:
            logger.warning(f"无法创建缓存目录 {cache_dir}: {e}，缓存已禁用")
            cache_dir = None

    # ── 时间窗口采样配置 ────────────────────────────────────────
    w_cfg = config.get('windowed_sampling', {})
    windowed = w_cfg.get('enabled', False)

    # 当时间窗口采样开启时，连通性由完整 run 估计（不截断）；
    # 截断仅在单样本训练模式（windowed=False）下保留，用于防 OOM。
    max_seq_len = config['training'].get('max_seq_len', None)
    if windowed:
        if max_seq_len is not None:
            logger.info(
                f"时间窗口采样已启用 (fMRI_ws={w_cfg.get('fmri_window_size', 50)}, "
                f"EEG_ws={w_cfg.get('eeg_window_size', 500)}, "
                f"stride={w_cfg.get('stride_fraction', 0.5)}×ws)。"
                f" 图构建将使用完整序列以获得可靠连通性估计"
                f"（max_seq_len={max_seq_len} 仅在单样本模式下生效）。"
                f" 建议设 max_seq_len: null 以完全利用全序列。"
            )
        else:
            logger.info(
                f"时间窗口采样已启用 (fMRI_ws={w_cfg.get('fmri_window_size', 50)}, "
                f"EEG_ws={w_cfg.get('eeg_window_size', 500)}, "
                f"stride={w_cfg.get('stride_fraction', 0.5)}×ws)。"
                f" 图构建将使用完整序列。"
            )
    else:
        if max_seq_len is not None:
            logger.info(f"序列截断已启用: max_seq_len={max_seq_len} (防止 CUDA OOM)")

    # ── 被试/任务发现（纯文件扫描，无数据加载）──────────────────────
    # 在任何数据加载之前先枚举所有 (被试, 任务) 对，用于：
    # 1. 建立全局 subject_to_idx（subject embedding 索引必须在全数据集中唯一且稳定）
    # 2. 逐对检查缓存，命中则跳过原始数据加载，大幅缩短二次运行启动时间
    tasks_cfg = config['data'].get('tasks')
    if tasks_cfg is None:
        legacy_task = config['data'].get('task')
        if legacy_task is not None:
            tasks_cfg = [legacy_task]
            logger.info(
                f"使用旧版 'task: {legacy_task}' 配置。"
                f" 建议迁移到 'tasks: [{legacy_task}]'。"
            )
        # tasks_cfg 仍为 None → 自动发现所有任务
    elif isinstance(tasks_cfg, str):
        tasks_cfg = [tasks_cfg]

    if tasks_cfg is None:
        logger.info("tasks: null → 自动发现每个被试的所有任务")
    else:
        logger.info(f"将加载以下任务: {tasks_cfg}")

    # 0 和 null 均表示"不限制被试数"（与 load_all_subjects 行为一致）。
    # 使用显式 None 检查而非 or None，避免意外将合法的 0 解释为"不限制"的语义混淆。
    _max_sub = config['data'].get('max_subjects')
    max_subjects_cfg = int(_max_sub) if _max_sub else None  # 0 / null → None

    subject_dirs = sorted(data_loader.data_root.glob("sub-*"))
    if max_subjects_cfg:
        subject_dirs = subject_dirs[:max_subjects_cfg]

    # 第一遍：仅扫描文件名，不加载任何原始数据
    all_pairs: List[tuple] = []  # [(subject_id, task), ...]
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        if tasks_cfg is None:
            subject_tasks = data_loader._discover_tasks(subject_id)
        elif len(tasks_cfg) == 0:
            # 空列表 = 不过滤任务，直接加载首个匹配文件（与 load_all_subjects 行为一致）
            subject_tasks = [None]
        else:
            subject_tasks = list(tasks_cfg)
        for t in subject_tasks:
            all_pairs.append((subject_id, t))

    if not all_pairs:
        raise ValueError(
            "未发现任何被试-任务组合，请检查数据路径配置。"
            f" data.root_dir={config['data']['root_dir']!r}"
        )

    logger.info(f"发现 {len(all_pairs)} 个被试-任务组合")

    # 被试 ID → 整数索引映射（用于 subject embedding，AGENTS.md §九 Gap 2）
    # 由文件系统发现结果确定性推导，与数据加载顺序无关，保证全局唯一。
    all_subject_ids = sorted(set(sid for sid, _ in all_pairs))
    subject_to_idx = {sid: i for i, sid in enumerate(all_subject_ids)}
    logger.info(f"被试索引映射已建立: {len(subject_to_idx)} 个被试")

    graphs: List[HeteroData] = []
    n_cached = 0
    n_windows_total = 0
    run_idx_counter = 0  # 每个 (subject, task) run 的全局整数索引，用于 run-level 训练/验证集划分
    for subject_id, task in all_pairs:
        # 计算一次缓存 key，供本次迭代的"读"和"写"共用，避免重复计算。
        cache_key = _graph_cache_key(subject_id, task, config) if cache_dir is not None else None

        if cache_dir is not None and cache_key is not None:
            cache_path = cache_dir / cache_key
            if cache_path.exists():
                try:
                    full_graph = torch.load(cache_path, map_location='cpu', weights_only=False)
                    # ── 跨模态边：在每次加载时从节点特征重新计算 ────────────
                    # 设计原则：跨模态边不是"原始数据衍生物"（需要原始EEG/fMRI），
                    # 而是节点特征的函数（仅需图内的 x 张量，代价 O(N_eeg×N_fmri×T)）。
                    # 将其存入缓存会带来两个问题：
                    #   1. k_cross_modal 变化时旧缓存失效，需要访问原始数据重建；
                    #   2. 用户调整跨模态连接策略后，看不到任何效果（仍用旧边）。
                    # 每次加载时重建可以保证：无论缓存来自哪个版本，
                    # 跨模态边始终由当前配置（k_cross_modal, eeg_connectivity_method）决定。
                    if 'fmri' in full_graph.node_types and 'eeg' in full_graph.node_types:
                        _cross = mapper.create_simple_cross_modal_edges(
                            full_graph,
                            k_cross_modal=config['graph'].get('k_cross_modal', 5),
                        )
                        if _cross is not None:
                            full_graph['eeg', 'projects_to', 'fmri'].edge_index = _cross[0]
                            full_graph['eeg', 'projects_to', 'fmri'].edge_attr = _cross[1]
                    # 被试索引（AGENTS.md §九 Gap 2）：
                    # 缓存图可能由 V5.18 或更早版本保存，不含 subject_idx。
                    # 使用当前运行时的 subject_to_idx 补写，保证嵌入对缓存图同样生效。
                    # 即使图由 V5.19 保存（已含 subject_idx），此处覆写也无害——
                    # subject_to_idx 由当前数据集的 subject_id 确定性推导，与保存时一致。
                    if subject_id not in subject_to_idx:
                        logger.warning(
                            f"subject_id='{subject_id}' not found in subject_to_idx "
                            f"(cache load path). Falling back to index 0."
                        )
                    full_graph.subject_idx = torch.tensor(
                        subject_to_idx.get(subject_id, 0), dtype=torch.long
                    )
                    # run_idx 供 run-level 训练/验证集划分使用。
                    # 与 subject_idx 同理：缓存图可能不含此属性，此处补写无害。
                    full_graph.run_idx = torch.tensor(run_idx_counter, dtype=torch.long)
                    # task_id / subject_id_str：补写任务和被试字符串元数据，
                    # 供 log_training_summary 展示数据组成。
                    full_graph.task_id = task or ''
                    full_graph.subject_id_str = subject_id
                    win_samples = extract_windowed_samples(full_graph, w_cfg, logger)
                    graphs.extend(win_samples)
                    n_windows_total += len(win_samples)
                    n_cached += 1
                    run_idx_counter += 1
                    logger.debug(
                        f"从缓存加载图: {cache_key} "
                        f"[节点类型: {list(full_graph.node_types)}, "
                        f"边类型: {[str(et) for et in full_graph.edge_types]}]"
                        + (f" → {len(win_samples)} 个窗口" if windowed else "")
                    )
                    continue
                except Exception as e:
                    logger.warning(f"缓存加载失败 ({cache_key}): {e}，重新构建")

        # ── 缓存未命中：加载原始数据并构建图 ─────────────────────
        # 只有在缓存中找不到预构建图时，才执行耗时的 EEG/fMRI 原始数据加载与预处理。
        logger.debug(f"缓存未命中，加载原始数据: {subject_id}/{task}")
        subject_data = data_loader.load_subject(subject_id, task)
        if not subject_data:
            logger.warning(f"跳过被试 {subject_id}/{task}: 数据加载失败")
            continue
        # 确保 task 字段一致：load_subject() 不设置 'task'，此处补写。
        # （旧版 load_all_subjects 在 subject_data 加入 all_data 前才添加 task 字段，
        #  此处维持相同语义。）
        subject_data['task'] = task
        if 'fmri' not in subject_data and 'eeg' not in subject_data:
            logger.warning(f"跳过被试 {subject_id}/{task}: 无有效模态数据")
            continue

        graph_list = []

        # fMRI图
        if 'fmri' in subject_data:
            fmri_data = subject_data['fmri']['data']
            fmri_img = subject_data['fmri'].get('img')   # preprocessed NIfTI object

            # Prefer atlas parcellation: gives [N_rois, T] (e.g. 200 ROI nodes).
            # Fallback to spatial average → [1, T] when atlas unavailable.
            fmri_ts = None
            if atlas_file.exists() and fmri_img is not None:
                fmri_ts = _parcellate_fmri_with_atlas(fmri_img, atlas_file)

            if fmri_ts is None:
                fmri_ts, error = process_fmri_timeseries(fmri_data)
                if error:
                    logger.warning(f"fMRI processing failed: {error}, skipping subject")
                    continue
            
            # ── 条件特异性时间段截取 ────────────────────────────────────────────
            # 适用场景：多个 EEG 条件（GRADON/GRADOFF）共享同一 fMRI 运行（如 task-CB），
            # fMRI 文件按条件顺序包含完整会话，例如：
            #   t=0..150  TRs → GRADON 条件
            #   t=150..300 TRs → GRADOFF 条件
            #
            # 若不截取，会产生两类错误：
            #   1. 非窗口模式：两个条件的 max_seq_len 截断都从 t=0 开始，
            #      GRADOFF 错误地使用 GRADON 时段的 fMRI 数据。
            #   2. 窗口模式：滑动窗口跨越条件边界（如 t=275-325 同时包含
            #      GRADON 末尾和 GRADOFF 开头），预测损失在边界处学习
            #      虚假的时序演化，模型被迫"预测"实验条件切换而非神经动态。
            #
            # 配置方式（单位：TR）：
            #   fmri_condition_bounds:
            #     GRADON: [0, 150]
            #     GRADOFF: [150, 300]
            #
            # 应用时机：在连通性估计 (mapper.map_fmri_to_graph) 之前截取，
            # 保证条件特异性连通性矩阵和窗口均不跨越条件边界。
            condition_bounds_cfg = config['data'].get('fmri_condition_bounds') or {}
            if task in condition_bounds_cfg:
                bounds = condition_bounds_cfg[task]
                # Validate format early: must be a sequence of at least 2 numbers.
                if not (hasattr(bounds, '__len__') and len(bounds) >= 2):
                    raise ValueError(
                        f"fmri_condition_bounds['{task}'] must be [start_TR, end_TR], "
                        f"got: {bounds!r}. Example: {{'GRADON': [0, 150]}}"
                    )
                start_tr, end_tr = int(bounds[0]), int(bounds[1])
                T_full = fmri_ts.shape[1]
                if end_tr > T_full:
                    logger.warning(
                        f"fMRI 条件时间段越界: task={task}, "
                        f"配置 end_tr={end_tr} > T_full={T_full}。"
                        f" 将截断至 {T_full}。"
                    )
                    end_tr = T_full
                if start_tr >= end_tr:
                    logger.warning(
                        f"fMRI 条件时间段无效: task={task}, [{start_tr}, {end_tr}) 为空区间，"
                        f"跳过边界截取（可能导致跨条件时序学习）。"
                    )
                else:
                    fmri_ts = fmri_ts[:, start_tr:end_tr]
                    logger.info(
                        f"fMRI 条件时间段截取: task={task}, "
                        f"TRs [{start_tr}, {end_tr}) → {fmri_ts.shape[1]} TPs"
                        f"（防止跨条件边界的时序学习）"
                    )

            # 截断仅在单样本训练模式下启用（防 CUDA OOM）。
            # 窗口模式下不截断，以使连通性估计来自完整（条件特异性）序列。
            if not windowed and max_seq_len is not None:
                fmri_ts = truncate_timeseries(fmri_ts, max_seq_len)
            
            logger.debug(f"fMRI timeseries shape: {fmri_ts.shape} → {fmri_ts.shape[0]} nodes")
            fmri_graph = mapper.map_fmri_to_graph(
                timeseries=fmri_ts,
                connectivity_matrix=None,  # 自动计算
            )
            graph_list.append(('fmri', fmri_graph))
        
        # EEG图
        if 'eeg' in subject_data:
            eeg_data = subject_data['eeg']['data']  # [n_channels, n_times]
            eeg_ch_names = subject_data['eeg']['ch_names']
            # 从 loader 获取真实采样率和电极坐标（由 EEGPreprocessor 通过
            # standard_1020 montage 设置，单位 mm）
            eeg_sfreq = subject_data['eeg'].get('sfreq', 250.0)
            eeg_ch_pos = subject_data['eeg'].get('ch_pos', None)  # [N_ch, 3] mm 或 None
            
            # Validate EEG data
            if eeg_data.shape[0] < 8:
                logger.warning(f"EEG has too few channels: {eeg_data.shape[0]}, skipping")
                continue
            if eeg_data.shape[1] < 100:
                logger.warning(f"EEG has too few timepoints: {eeg_data.shape[1]}, skipping")
                continue
            if np.isnan(eeg_data).any() or np.isinf(eeg_data).any():
                logger.warning("EEG contains NaN or Inf values, skipping")
                continue
            
            # 截断仅在单样本训练模式下启用（防 CUDA OOM）。
            # 窗口模式下不截断，以使连通性估计来自完整 run。
            if not windowed and max_seq_len is not None:
                eeg_data = truncate_timeseries(eeg_data, max_seq_len)
            
            eeg_graph = mapper.map_eeg_to_graph(
                timeseries=eeg_data,
                channel_names=eeg_ch_names,
                channel_positions=eeg_ch_pos,
                sampling_rate=eeg_sfreq,
            )
            graph_list.append(('eeg', eeg_graph))
        
        # 合并图 - FIX: Properly merge multi-modal graphs
        if len(graph_list) > 0:
            if len(graph_list) == 1:
                # Single modality: use as-is
                built_graph = graph_list[0][1]
            else:
                # Multi-modal: merge into heterograph
                built_graph = HeteroData()
                for modality, graph in graph_list:
                    # Copy node features, structure, AND metadata
                    for key in graph.node_types:
                        built_graph[key].x = graph[key].x
                        if hasattr(graph[key], 'num_nodes'):
                            built_graph[key].num_nodes = graph[key].num_nodes
                        if hasattr(graph[key], 'pos'):
                            built_graph[key].pos = graph[key].pos
                        # sampling_rate used by log_training_summary; omitting it
                        # causes silent fallback to wrong defaults (250 Hz / 0.5 Hz)
                        if hasattr(graph[key], 'sampling_rate'):
                            built_graph[key].sampling_rate = graph[key].sampling_rate
                    
                    # Copy edge structure
                    for edge_type in graph.edge_types:
                        built_graph[edge_type].edge_index = graph[edge_type].edge_index
                        if hasattr(graph[edge_type], 'edge_attr'):
                            built_graph[edge_type].edge_attr = graph[edge_type].edge_attr
                
                # 跨模态边：EEG → fMRI
                # 设计理念：EEG 电极（较少节点）向 fMRI ROI（较多节点）投射信号。
                # create_simple_cross_modal_edges 返回 (edge_index, edge_attr)，
                # 其中 edge_attr 为均匀权重（1.0），保持与同模态边一致的加权语义。
                if 'fmri' in built_graph.node_types and 'eeg' in built_graph.node_types:
                    cross_result = mapper.create_simple_cross_modal_edges(
                        built_graph,
                        k_cross_modal=config['graph'].get('k_cross_modal', 5),
                    )
                    if cross_result is not None:
                        cross_edges, cross_weights = cross_result
                        built_graph['eeg', 'projects_to', 'fmri'].edge_index = cross_edges
                        built_graph['eeg', 'projects_to', 'fmri'].edge_attr = cross_weights

            # DTI 结构连通性边（可选）
            # 当被试有预计算 DTI 连通性矩阵且配置启用时，
            # 在 fMRI 节点上新增 ('fmri','structural','fmri') 边类型。
            # 此举使编码器能同时利用 fMRI 功能连通性（相关性边）
            # 和 DTI 结构连通性（白质纤维边），体现异质图"多边类型"的核心价值。
            dti_cfg = config['data'].get('dti_structural_edges', False)
            dti_subject = subject_data.get('dti')
            if dti_cfg and dti_subject is not None and 'fmri' in built_graph.node_types:
                mapper.add_dti_structural_edges(
                    built_graph,
                    dti_subject['connectivity'],
                )

            # 被试索引（AGENTS.md §九 Gap 2：个性化 subject embedding）
            # graph-level 属性：每个图（全序列图和窗口样本）均携带，
            # 供 GraphNativeBrainModel.forward() 查询 subject embedding。
            # subject_id 必须在 subject_to_idx 中（上方预扫描保证）；
            # 若 KeyError 发生，说明 subject_id 在预扫描和此处使用之间不一致（逻辑 bug）。
            if subject_id not in subject_to_idx:
                logger.warning(
                    f"subject_id='{subject_id}' not found in subject_to_idx. "
                    f"This indicates a logic bug — subject_id changed between pre-scan and use. "
                    f"Falling back to index 0; this subject's embedding will be shared with another."
                )
            built_graph.subject_idx = torch.tensor(
                subject_to_idx.get(subject_id, 0), dtype=torch.long
            )
            # run_idx: 全局 run 顺序索引，供 run-level 训练/验证集划分。
            # 窗口模式下，同一 run 的所有窗口共享相同 run_idx，
            # 训练时以 run 为单位划分，防止重叠窗口同时出现在训练集和验证集中。
            built_graph.run_idx = torch.tensor(run_idx_counter, dtype=torch.long)
            # task_id: 任务名称字符串，供日志摘要和每任务统计使用。
            # 注意：此处存储字符串而非整数，仅用于日志和调试，不参与模型计算。
            built_graph.task_id = task or ''
            # subject_id_str: 被试 ID 字符串（与 subject_idx 整数一一对应）。
            # 单独保留字符串形式，供日志摘要按被试统计样本数时使用。
            built_graph.subject_id_str = subject_id

            # ── 保存到缓存（始终保存完整 run 图） ──────────────────
            # 缓存内容：
            #   - eeg 节点：x [N_eeg, T, 1]、num_nodes、pos、sampling_rate
            #   - fmri 节点：x [N_fmri, T, 1]、num_nodes、pos、sampling_rate
            #   - ('eeg','connects','eeg') 同模态边（来自 EEG 连通性估计）
            #   - ('fmri','connects','fmri') 同模态边（来自 fMRI 连通性估计）
            #   ⚠ 跨模态边 ('eeg','projects_to','fmri') 不持久化：
            #     每次加载时从节点特征动态重建，使 k_cross_modal 修改立即生效
            #     而无需清空缓存或访问原始数据。
            if cache_dir is not None and cache_key is not None:
                try:
                    cache_path = cache_dir / cache_key
                    # 构建只含同模态数据的缓存图（跨模态边在加载时重建）
                    _cache_graph = HeteroData()
                    for _nt in built_graph.node_types:
                        _cache_graph[_nt].x = built_graph[_nt].x
                        for _attr in ('num_nodes', 'pos', 'sampling_rate', 'labels'):
                            if hasattr(built_graph[_nt], _attr):
                                setattr(_cache_graph[_nt], _attr, getattr(built_graph[_nt], _attr))
                    for _et in built_graph.edge_types:
                        if _et == ('eeg', 'projects_to', 'fmri'):
                            continue  # 跨模态边：不缓存，每次加载时重建
                        _cache_graph[_et].edge_index = built_graph[_et].edge_index
                        if hasattr(built_graph[_et], 'edge_attr'):
                            _cache_graph[_et].edge_attr = built_graph[_et].edge_attr
                    torch.save(_cache_graph, cache_path)
                    logger.debug(
                        f"图已缓存: {cache_key} "
                        f"[节点类型: {list(_cache_graph.node_types)}, "
                        f"同模态边类型: {[str(et) for et in _cache_graph.edge_types]}]"
                    )
                except Exception as e:
                    logger.warning(f"缓存保存失败 ({subject_id}/{task}): {e}")

            # ── 加入训练列表 ─────────────────────────────────────
            # 窗口模式：切分为多个短窗口样本；单样本模式：直接加入完整图。
            if windowed:
                win_samples = extract_windowed_samples(built_graph, w_cfg, logger)
                graphs.extend(win_samples)
                n_windows_total += len(win_samples)
                logger.debug(
                    f"  {subject_id}/{task}: {len(win_samples)} 个时间窗口样本"
                )
            else:
                graphs.append(built_graph)
            run_idx_counter += 1

    if len(graphs) == 0:
        raise ValueError("No valid graphs constructed. Check data quality and preprocessing.")

    # ── 汇总日志 ────────────────────────────────────────────────
    n_runs = len(all_pairs)
    if windowed:
        avg_win = n_windows_total / max(n_runs, 1)
        logger.info(
            f"图构建完成: {n_runs} 条 run → {n_windows_total} 个时间窗口训练样本"
            f" (平均 {avg_win:.1f} 个窗口/run)"
            + (f"，其中 {n_cached} 条 run 来自缓存" if n_cached else "")
        )
    elif n_cached > 0:
        logger.info(
            f"图构建完成: {len(graphs)} 个图"
            f" (其中 {n_cached} 个来自缓存，{len(graphs) - n_cached} 个新建)"
        )
    else:
        logger.info(f"成功构建 {len(graphs)} 个图")

    return graphs, mapper, subject_to_idx


def create_model(config: dict, logger: logging.Logger, num_subjects: int = 0):
    """创建模型

    Args:
        config: 配置字典
        logger: 日志记录器
        num_subjects: 数据集中的总被试数（由 build_graphs 返回的 subject_to_idx 推导）。
            > 0 时在模型中创建 nn.Embedding(num_subjects, hidden_channels) 实现
            个性化被试嵌入（AGENTS.md §九 Gap 2）。
            0 = 禁用（默认，兼容旧行为）。
    """
    logger.info("=" * 60)
    logger.info("步骤 3/4: 创建模型")
    logger.info("=" * 60)
    
    # 确定节点和边类型
    node_types = config['data']['modalities']
    edge_types = []

    for modality in node_types:
        edge_types.append((modality, 'connects', modality))

    # 跨模态边：设计理念是 EEG → fMRI
    # EEG 电极（通常 32–64 通道）节点数 < fMRI ROI（如 Schaefer200 的 200 个），
    # 因此由 EEG 向 fMRI 投射消息符合"少节点向多节点传播"的图卷积语义。
    # 使用模态名而非位置索引，保证不受 config['data']['modalities'] 顺序影响。
    if 'eeg' in node_types and 'fmri' in node_types:
        edge_types.append(('eeg', 'projects_to', 'fmri'))
    elif len(node_types) > 1:
        # 非 EEG/fMRI 模态组合的通用回退
        edge_types.append((node_types[0], 'projects_to', node_types[1]))

    # DTI 结构连通性边（可选）
    # 当 dti_structural_edges: true 时，编码器预先注册 ('fmri','structural','fmri')
    # 边类型。若某个图实际不含该边（DTI 文件缺失），编码器会安静跳过
    # （GraphNativeEncoder.forward 中 `if edge_type in edge_index_dict` 保护）。
    # 这是异质图扩展性的核心体现：模型知道该边类型，但不强依赖其存在。
    if config['data'].get('dti_structural_edges', False) and 'fmri' in node_types:
        edge_types.append(('fmri', 'structural', 'fmri'))
        logger.info("DTI结构连通性边已启用: ('fmri','structural','fmri') 已加入边类型列表")

    # 输入通道
    in_channels_dict = {modality: 1 for modality in node_types}
    
    # 创建模型
    model = GraphNativeBrainModel(
        node_types=node_types,
        edge_types=edge_types,
        in_channels_dict=in_channels_dict,
        hidden_channels=config['model']['hidden_channels'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        use_prediction=config['model']['use_prediction'],
        prediction_steps=config['model']['prediction_steps'],
        dropout=config['model']['dropout'],
        loss_type=config['model'].get('loss_type', 'mse'),
        use_gradient_checkpointing=config['training'].get('use_gradient_checkpointing', False),
        predictor_config=config.get('v5_optimization', {}).get('advanced_prediction'),
        use_dynamic_graph=config['model'].get('use_dynamic_graph', False),
        k_dynamic_neighbors=config['model'].get('k_dynamic_neighbors', 10),
        num_subjects=num_subjects,
    )

    if num_subjects > 0:
        logger.info(
            f"被试特异性嵌入已启用: {num_subjects} 个被试 × "
            f"{config['model']['hidden_channels']} 维 (AGENTS.md §九 Gap 2)"
        )
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def log_training_summary(
    config: dict,
    graphs: list,
    model,
    logger: logging.Logger,
) -> None:
    """启动训练前打印一次人类可读的配置核对表。

    目的：让任何人（不需要了解代码细节）在看到日志的第一眼就能
    确认数据处理方式是否符合预期，避免"默默地用了错误参数训练完
    才发现"的情况。

    信息来源：
    - 优先读取 graphs[0] 中的实际运行时值（节点数、序列长度等），
      而不是 config 中的期望值——两者可能因数据质量问题而不同。
    - 读取 config 获取模型结构和训练超参数。
    """
    sep = "=" * 60

    logger.info(sep)
    logger.info("📋 训练配置核对表 (Training Configuration Summary)")
    logger.info(sep)

    # ── 从第一个图提取运行时实际值 ──────────────────────────────
    g = graphs[0] if graphs else None
    modalities = config['data'].get('modalities', [])

    logger.info("【数据】")

    if g is not None:
        # EEG
        if 'eeg' in getattr(g, 'node_types', []):
            eeg_x = g['eeg'].x  # [N, T, C]
            N_eeg, T_eeg = eeg_x.shape[0], eeg_x.shape[1]
            has_eeg_pos = (
                hasattr(g['eeg'], 'pos')
                and g['eeg'].pos is not None
                and g['eeg'].pos.shape[0] > 0
            )
            # sampling_rate is always written by map_eeg_to_graph; the fallback
            # here matches that function's default (250 Hz) for robustness.
            eeg_sr = getattr(g['eeg'], 'sampling_rate', 250.0)
            logger.info(
                f"  EEG  : {N_eeg} 个电极通道 | "
                f"采样率: {eeg_sr:.1f} Hz | "
                f"序列长度: {T_eeg} 个时间点"
            )
            pos_note = (
                "已找到 (来自 MNE standard_1020 montage，单位 mm)"
                if has_eeg_pos
                else "未找到 → 将使用随机跨模态连接（非距离加权）"
            )
            logger.info(f"         电极坐标: {pos_note}")

        # fMRI
        if 'fmri' in getattr(g, 'node_types', []):
            fmri_x = g['fmri'].x  # [N, T, C]
            N_fmri, T_fmri = fmri_x.shape[0], fmri_x.shape[1]
            # sampling_rate is always written by map_fmri_to_graph; the fallback
            # here matches that function's default (0.5 Hz = TR 2 s) for robustness.
            fmri_sr = getattr(g['fmri'], 'sampling_rate', 0.5)
            tr_sec = 1.0 / fmri_sr if fmri_sr > 0 else float('nan')
            atlas_used = N_fmri > 1
            atlas_note = (
                f"已启用 ({config['data']['atlas']['name']}, {N_fmri} 个 ROI 节点)"
                if atlas_used
                else f"未启用 → 单节点回退 (N_fmri={N_fmri}，空间信息已丢失)"
            )
            logger.info(
                f"  fMRI : {N_fmri} 个 ROI 节点 | "
                f"采样率: {fmri_sr:.3g} Hz (TR≈{tr_sec:.1f}s) | "
                f"序列长度: {T_fmri} 个时间点"
            )
            logger.info(f"         图谱分区: {atlas_note}")
            if not atlas_used:
                logger.info(
                    f"  ⚠️   fMRI 只有 {N_fmri} 个节点！图卷积无法提取空间信息。"
                    f" 请检查 atlas 文件路径是否正确、nilearn 是否已安装。"
                )

        # 跨模态边
        if (
            'eeg' in getattr(g, 'node_types', [])
            and 'fmri' in getattr(g, 'node_types', [])
        ):
            cross_edge_type = ('eeg', 'projects_to', 'fmri')
            if cross_edge_type in getattr(g, 'edge_types', []):
                n_cross = g[cross_edge_type].edge_index.shape[1]
                logger.info(
                    f"  跨模态边 (EEG→fMRI): {n_cross} 条"
                    f" | 方向正确 (N_eeg={N_eeg} < N_fmri={N_fmri})"
                    if N_eeg < N_fmri
                    else
                    f"  跨模态边 (EEG→fMRI): {n_cross} 条"
                    f"  ⚠️  N_eeg({N_eeg}) ≥ N_fmri({N_fmri}), 请检查图谱加载"
                )
            else:
                logger.info("  跨模态边 (EEG→fMRI): 未建立")
    else:
        logger.info("  (无图数据可供分析)")

    max_seq = config['training'].get('max_seq_len')
    if max_seq:
        logger.info(f"  序列截断 max_seq_len: {max_seq} (防止 CUDA OOM)")
    else:
        logger.info("  序列截断: 未启用 (若序列过长可能 OOM，建议设置 max_seq_len)")

    # ── 训练数据组成摘要 ──────────────────────────────────────────
    # 显示每个被试、每个任务的样本数，让用户一眼看出数据是否均衡、
    # 是否有被试/任务缺失，以及跨任务/被试训练的实际规模。
    logger.info("【训练数据组成】")
    total_samples = len(graphs)
    logger.info(f"  总样本数: {total_samples}")

    # 按任务统计
    # 检查 graphs[0] 即可：build_graphs() 为所有图统一写入 task_id，
    # extract_windowed_samples() 将其传播到所有窗口样本，所以要么所有图都有，要么都没有。
    if graphs and hasattr(graphs[0], 'task_id'):
        task_counts = Counter(getattr(g, 'task_id', '') for g in graphs)
        logger.info(f"  按任务分布 ({len(task_counts)} 个任务):")
        for task_name, cnt in sorted(task_counts.items()):
            pct = cnt / total_samples * 100
            logger.info(f"    {task_name or '(无任务名)'}: {cnt} 个样本 ({pct:.1f}%)")
    else:
        logger.info("  任务分布: 不可用（task_id 未存储，请清除缓存重建图）")

    # 按被试统计（同上，检查 graphs[0] 即可）
    if graphs and hasattr(graphs[0], 'subject_id_str'):
        subj_counts = Counter(getattr(g, 'subject_id_str', '') for g in graphs)
        logger.info(f"  按被试分布 ({len(subj_counts)} 个被试):")
        for subj_id, cnt in sorted(subj_counts.items()):
            pct = cnt / total_samples * 100
            logger.info(f"    {subj_id}: {cnt} 个样本 ({pct:.1f}%)")
    else:
        logger.info("  被试分布: 不可用（subject_id_str 未存储，请清除缓存重建图）")

    # 数据独立性说明（关键：帮助用户理解"多被试多任务合成列表"的训练语义）
    logger.info(
        "  ✅ 数据独立性确认: 每个样本均来自独立的 (被试, 任务) 组合，"
        "不同被试/任务的图在内存中相互独立，梯度更新以单样本（batch_size=1）进行，"
        "不存在跨被试或跨任务的节点/特征混合。"
    )
    logger.info(
        "  ✅ 逐 Epoch 打乱: train_epoch 每轮以 epoch 编号为种子打乱样本顺序，"
        "确保 SGD 不会系统性地偏向列表末尾的被试/任务。"
    )

    # ── 模型 ────────────────────────────────────────────────────
    logger.info("【模型】")
    logger.info(
        f"  隐层维度: {config['model']['hidden_channels']} | "
        f"编码层数: {config['model']['num_encoder_layers']} | "
        f"解码层数: {config['model']['num_decoder_layers']}"
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  总参数量: {total_params:,}")
    logger.info(f"  损失函数: {config['model'].get('loss_type', 'mse')}")

    # 被试特异性嵌入（AGENTS.md §九 Gap 2 个性化数字孪生）
    # 直接从模型属性读取运行时值，而非 config，以覆盖 num_subjects=0 等默认情形。
    num_subjects_rt = getattr(model, 'num_subjects', 0)
    if num_subjects_rt > 0:
        H = config['model']['hidden_channels']
        embed_params = num_subjects_rt * H
        logger.info(
            f"  被试特异性嵌入: ✅ 已启用 | "
            f"{num_subjects_rt} 个被试 × {H} 维 = {embed_params:,} 个个性化参数"
        )
        logger.info(
            f"  个性化原理: 每个被试学习一个 [H={H}] 潜空间偏移，"
            f"施加于所有节点特征（编码器输入投影之后）"
        )
        # Verify that graphs carry subject_idx so embedding is actually activated
        has_sidx = graphs and hasattr(graphs[0], 'subject_idx')
        if not has_sidx:
            logger.warning(
                "  ⚠️  graphs[0] 缺少 subject_idx 属性！"
                " 被试嵌入在 forward() 中将被跳过。"
                " 这通常意味着图缓存来自 V5.18 或更早版本，"
                " 请清除缓存目录后重新运行以重建含 subject_idx 的图。"
            )
    else:
        logger.info("  被试特异性嵌入: ❌ 未启用 (num_subjects=0)")

    # ── 训练 ────────────────────────────────────────────────────
    logger.info("【训练】")
    device = config['device']['type']
    use_amp = config['device'].get('use_amp', True)
    use_gc = config['training'].get('use_gradient_checkpointing', False)
    use_al = config['training'].get('use_adaptive_loss', True)
    lr = config['training']['learning_rate']
    logger.info(
        f"  设备: {device} | "
        f"混合精度(AMP): {'是' if use_amp else '否'} | "
        f"梯度检查点: {'是' if use_gc else '否'}"
    )
    logger.info(
        f"  学习率: {lr} | "
        f"自适应损失权重: {'是' if use_al else '否'}"
    )

    logger.info(sep)
    logger.info("⚠️  请核对以上参数是否与您的实验预期一致，再继续训练。")
    logger.info(sep)


def train_model(model, graphs, config: dict, logger: logging.Logger,
                resume_checkpoint: Optional[str] = None):
    """训练模型"""
    logger.info("=" * 60)
    logger.info("步骤 4/4: 训练模型")
    logger.info("=" * 60)
    
    # 划分训练/验证集
    if len(graphs) < 2:
        logger.error(f"❌ 数据不足: 需要至少2个样本进行训练，但只有 {len(graphs)} 个样本")
        logger.error("提示: 请增加数据量或调整 max_subjects 配置")
        raise ValueError(f"需要至少2个样本进行训练,但只有 {len(graphs)} 个。请检查数据配置。")
    
    rng = random.Random(42)
    windowed = config.get('windowed_sampling', {}).get('enabled', False)

    if windowed and graphs and hasattr(graphs[0], 'run_idx'):
        # ── Run-level 划分（窗口模式）────────────────────────────────
        # 窗口模式下，同一 run 产生多个 50% 重叠窗口。
        # 若以窗口为单位随机划分，来自同一 run 的重叠窗口会同时出现在训练集和
        # 验证集中（数据泄漏），导致验证损失虚低、无法反映真实泛化性能。
        # 以 run 为单位划分确保：某个 run 的所有窗口只进入训练集或只进入验证集。
        run_groups: dict = defaultdict(list)
        for g in graphs:
            run_groups[g.run_idx.item()].append(g)
        
        run_keys = sorted(run_groups.keys())
        rng.shuffle(run_keys)
        
        min_val_runs = max(1, len(run_keys) // 10)
        n_train_runs = len(run_keys) - min_val_runs
        if n_train_runs < 1:
            n_train_runs = 1
            min_val_runs = len(run_keys) - 1
        
        train_run_keys = run_keys[:n_train_runs]
        val_run_keys = run_keys[n_train_runs:]
        train_graphs = [g for k in train_run_keys for g in run_groups[k]]
        val_graphs = [g for k in val_run_keys for g in run_groups[k]]
        logger.info(
            f"训练集: {len(train_graphs)} 个窗口 (来自 {len(train_run_keys)} 个 run) | "
            f"验证集: {len(val_graphs)} 个窗口 (来自 {len(val_run_keys)} 个 run) "
            f"[run-level 划分，防止重叠窗口数据泄漏，seed=42]"
        )
    else:
        # ── 样本级划分（单样本模式或无 run_idx）────────────────────
        # 打乱后再划分，避免以下偏差：
        # 1. 被试按字母顺序排列时最后几个被试全部只出现在验证集
        # 2. 多任务场景下某类任务全集中在一端
        shuffled = graphs.copy()
        rng.shuffle(shuffled)
        
        min_val_samples = max(1, len(shuffled) // 10)
        n_train = len(shuffled) - min_val_samples
        if n_train < 1:
            n_train = 1
        train_graphs = shuffled[:n_train]
        val_graphs = shuffled[n_train:]
        logger.info(
            f"训练集: {len(train_graphs)} 个样本 | 验证集: {len(val_graphs)} 个样本 "
            f"(seed=42 随机打乱, 结果可复现)"
        )
    
    if len(train_graphs) < 5:
        logger.warning("⚠️ 训练样本较少，模型可能过拟合。建议使用更多数据。")
    
    # 创建训练器
    logger.info("正在初始化训练器...")
    if config['device'].get('use_torch_compile', True):
        logger.info("⚙️ torch.compile() 已启用，首次训练可能需要额外时间进行模型编译...")
    
    trainer = GraphNativeTrainer(
        model=model,
        node_types=config['data']['modalities'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        use_adaptive_loss=config['training']['use_adaptive_loss'],
        use_eeg_enhancement=config['training']['use_eeg_enhancement'],
        use_amp=config['device'].get('use_amp', True),
        use_gradient_checkpointing=config['training'].get('use_gradient_checkpointing', False),
        use_scheduler=config['training'].get('use_scheduler', True),
        scheduler_type=config['training'].get('scheduler_type', 'cosine'),
        use_torch_compile=config['device'].get('use_torch_compile', True),
        compile_mode=config['device'].get('compile_mode', 'reduce-overhead'),
        device=config['device']['type'],
        optimization_config=config.get('v5_optimization'),
        max_grad_norm=config['training'].get('max_grad_norm', 1.0),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        augmentation_config=config['training'].get('augmentation'),
    )
    logger.info("✅ 训练器初始化完成")

    # ── 断点续训（--resume）─────────────────────────────────────────────────
    # 若提供了检查点路径，加载已保存的 model/optimizer/scheduler/loss_balancer 状态，
    # 并从 checkpoint_epoch + 1 继续训练，而非从 epoch 1 重头开始。
    _start_epoch = 1
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        if resume_path.exists():
            try:
                saved_epoch = trainer.load_checkpoint(resume_path)
                # load_checkpoint returns the saved epoch number (int).
                # Guard against None in case an old checkpoint had no 'epoch' key.
                _start_epoch = (int(saved_epoch) + 1) if saved_epoch is not None else 1
                logger.info(
                    f"🔄 断点续训: 从 epoch {_start_epoch} 继续"
                    f" (已加载检查点 {resume_path})"
                )
            except Exception as _re:
                logger.warning(
                    f"⚠️ 检查点加载失败 ({resume_path}): {_re}。"
                    f" 将从 epoch 1 重新开始训练。"
                )
        else:
            logger.warning(
                f"⚠️ 检查点路径不存在: {resume_path}。"
                f" 将从 epoch 1 重新开始训练。"
            )

    logger.info("=" * 60)
    logger.info("开始训练循环")
    logger.info("=" * 60)
    
    # 训练循环
    best_val_loss = float('inf')
    best_epoch = 0
    best_r2_dict: dict = {}   # R² at best-val-loss epoch, used in credibility summary
    patience_counter = 0
    no_improvement_warning_shown = False
    epoch_times = []
    output_dir = Path(config['output']['output_dir'])
    best_checkpoint_path = output_dir / "best_model.pt"

    # ── 早停配置说明 ──────────────────────────────────────────────────────
    # patience_counter 以"验证次数"为单位递增（非 epoch 数），
    # 因此实际等效 epoch 耐心值 = early_stopping_patience × val_frequency。
    _val_freq = config['training']['val_frequency']
    _early_patience = config['training']['early_stopping_patience']
    _effective_epoch_patience = _val_freq * _early_patience
    logger.info(
        f"早停设置: 每 {_val_freq} epoch 验证一次 | "
        f"连续 {_early_patience} 次验证无改善触发早停 | "
        f"等效 {_effective_epoch_patience} epoch 的实际耐心值"
    )

    for epoch in range(_start_epoch, config['training']['num_epochs'] + 1):
        epoch_start_time = time.time()
        
        # 训练
        train_loss = trainer.train_epoch(train_graphs, epoch=epoch, total_epochs=config['training']['num_epochs'])
        
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Estimate remaining time (after first few epochs)
        if len(epoch_times) >= 3:
            avg_epoch_time = sum(epoch_times[-5:]) / len(epoch_times[-5:])  # Use last 5 epochs
            remaining_epochs = config['training']['num_epochs'] - epoch
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_minutes = eta_seconds / 60
            if eta_minutes < 60:
                eta_str = f"{eta_minutes:.1f} 分钟"
            else:
                eta_str = f"{eta_minutes/60:.1f} 小时"
        else:
            eta_str = "计算中..."
        
        # Memory monitoring every 10 epochs
        if epoch % 10 == 0 and torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            reserved_gb = torch.cuda.memory_reserved() / 1e9
            logger.info(f"  💾 GPU Memory: allocated={allocated_gb:.2f} GB, reserved={reserved_gb:.2f} GB")
        
        # Check for NaN loss
        if np.isnan(train_loss) or np.isinf(train_loss):
            logger.error(f"❌ Training loss is NaN/Inf at epoch {epoch}. Stopping training.")
            raise ValueError("Training diverged: loss is NaN or Inf")
        
        # 验证
        if epoch % config['training']['val_frequency'] == 0:
            val_loss, r2_dict = trainer.validate(val_graphs)
            
            # Step scheduler based on validation loss (for ReduceLROnPlateau)
            trainer.step_scheduler_on_validation(val_loss)
            
            # Check for NaN validation loss
            if np.isnan(val_loss) or np.isinf(val_loss):
                logger.error(f"❌ Validation loss is NaN/Inf at epoch {epoch}. Stopping training.")
                raise ValueError("Validation diverged: loss is NaN or Inf")
            
            # Format R² values for logging (show all modalities)
            r2_str = "  ".join(f"{k}={v:.3f}" for k, v in sorted(r2_dict.items()))
            logger.info(
                f"✓ Epoch {epoch}/{config['training']['num_epochs']}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"{r2_str}, "
                f"time={epoch_time:.1f}s, ETA={eta_str}"
            )

            # ── R² < 0 警报：模型差于均值基线时明确告警 ─────────────────
            # R² < 0 表示模型重建误差 > 信号总方差，即比"永远预测均值"还差。
            # 这是模型科学失效的明确信号，非专业用户无法从裸数字判断。
            for _r2k, _r2v in r2_dict.items():
                if _r2v < 0.0:
                    logger.warning(
                        f"  ⛔ {_r2k}={_r2v:.3f} < 0: 模型重建效果差于均值基线预测，"
                        "当前模型尚未从数据中学到有效信号。"
                        " 请检查数据质量、atlas 加载、或降低学习率后重试。"
                    )

            # ── 过拟合检测：训练/验证损失比超阈值时警告 ─────────────────
            if train_loss > 0 and val_loss > 0:
                _overfit_ratio = val_loss / train_loss
                if _overfit_ratio > 3.0:
                    logger.warning(
                        f"  ⚠️ 过拟合风险: val_loss/train_loss={_overfit_ratio:.1f}× > 3.0"
                        "，训练损失远低于验证损失。"
                        " 建议: 增大 weight_decay、减少 num_encoder_layers 或增加训练数据。"
                    )
            
            # Warn if no improvement after many epochs
            if epoch >= 50 and best_val_loss == float('inf') and not no_improvement_warning_shown:
                logger.warning("⚠️ No improvement in validation loss after 50 epochs. Check data quality and hyperparameters.")
                no_improvement_warning_shown = True
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                improvement = (best_val_loss - val_loss) / best_val_loss * 100 if best_val_loss != float('inf') else 100
                best_val_loss = val_loss
                best_epoch = epoch
                best_r2_dict = r2_dict.copy()   # 同步记录该 epoch 的 R²
                patience_counter = 0
                
                # 保存检查点
                trainer.save_checkpoint(best_checkpoint_path, epoch)
                _r2_at_best = "  ".join(f"{k}={v:.3f}" for k, v in sorted(best_r2_dict.items()))
                if improvement != 100:
                    logger.info(f"  🎯 保存最佳模型: val_loss={val_loss:.4f}, {_r2_at_best} (提升 {improvement:.1f}%)")
                else:
                    logger.info(f"  🎯 保存最佳模型: val_loss={val_loss:.4f}, {_r2_at_best}")
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= config['training']['early_stopping_patience']:
                logger.info(
                    f"⏹️ 早停触发: 连续 {patience_counter} 次验证"
                    f" (约 {patience_counter * _val_freq} epoch) 无改进"
                )
                break
        else:
            logger.info(
                f"✓ Epoch {epoch}/{config['training']['num_epochs']}: "
                f"train_loss={train_loss:.4f}, time={epoch_time:.1f}s, ETA={eta_str}"
            )
        
        # 定期保存检查点
        if epoch % config['training']['save_frequency'] == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            trainer.save_checkpoint(checkpoint_path, epoch)
    
    logger.info("训练完成!")
    logger.info(f"最佳验证损失: {best_val_loss:.4f}")

    # ── 训练可信度摘要 ─────────────────────────────────────────────────────
    # 向非专业用户提供一次性、人类可读的科学可信度评估：
    # 1. 最佳模型的 R² 指标及其含义
    # 2. 是否存在过拟合风险
    # 3. 结论：模型是否可信用于后续分析
    _sep = "=" * 60
    logger.info(_sep)
    logger.info("📊 训练可信度摘要 (Training Credibility Summary)")
    logger.info(_sep)
    logger.info(f"  最佳模型: Epoch {best_epoch}, val_loss={best_val_loss:.4f}")
    if best_r2_dict:
        _all_trustworthy = True
        for _r2k, _r2v in sorted(best_r2_dict.items()):
            if _r2v >= 0.3:
                _sym, _rating = "✅", "良好重建能力 (R² ≥ 0.3，达到神经影像研究可用标准)"
            elif 0.1 <= _r2v < 0.3:
                _sym, _rating = "⚠️", "有限重建能力 (0.1 ≤ R² < 0.3，建议增加数据量或调整模型)"
                _all_trustworthy = False
            elif 0.0 <= _r2v < 0.1:
                _sym, _rating = "⚠️", "极弱重建能力 (0 ≤ R² < 0.1，模型几乎未学到有效信号)"
                _all_trustworthy = False
            else:
                _sym, _rating = "⛔", "模型不可信：差于均值基线 (R² < 0，请检查数据或重新训练)"
                _all_trustworthy = False
            logger.info(f"  {_sym} {_r2k}={_r2v:.3f} — {_rating}")
        if _all_trustworthy:
            logger.info("  ✅ 结论：最佳模型在所有模态上达到可信重建水平 (R² ≥ 0.3)")
        else:
            logger.warning(
                "  ⚠️ 结论：部分或全部模态的 R² 低于可信阈值 (R² < 0.3)。"
                " 当前最佳模型的重建能力有限，建议检查数据质量、atlas 配置"
                " 或增加训练数据后重训。"
            )
    else:
        logger.warning("  R² 未计算（训练中从未执行验证，请检查 val_frequency 配置）")
    logger.info(_sep)

    # ── 训练曲线可视化 ─────────────────────────────────────────────────────
    # 训练结束后自动将 loss + R² 历史绘制为 PNG 图像，
    # 保存到输出目录，方便非专业用户直观判断训练质量。
    try:
        plot_training_curves(
            history=trainer.history,
            output_dir=output_dir,
            best_epoch=best_epoch,
            best_r2_dict=best_r2_dict if best_r2_dict else None,
        )
        logger.info(
            f"📈 训练曲线图已保存: "
            f"training_loss_curve.png, training_r2_curve.png"
        )
    except Exception as _viz_err:
        logger.debug(f"训练曲线绘制跳过: {_viz_err}")

    # ── 恢复最佳模型（Optimization 4）────────────────────────────────────
    # 训练结束（含早停）后，trainer.model 处于最后一个 epoch 的状态。
    # 自动加载 best_model.pt，确保后续推理/评估使用验证集最优权重，
    # 而非训练轨迹末端可能已过拟合的权重。
    # 这是所有现代训练框架（Keras ModelCheckpoint、PyTorch-Lightning）的标准行为。
    if best_checkpoint_path.exists() and best_val_loss < float('inf'):
        try:
            trainer.load_checkpoint(best_checkpoint_path)
            _r2_loaded = "  ".join(f"{k}={v:.3f}" for k, v in sorted(best_r2_dict.items())) if best_r2_dict else "N/A"
            logger.info(
                f"✅ 已自动恢复最佳模型 "
                f"(epoch={best_epoch}, val_loss={best_val_loss:.4f}, {_r2_loaded})"
            )
        except Exception as _e:
            logger.warning(
                f"⚠️ 最佳模型自动恢复失败: {_e}。"
                f" 当前模型为最后一个 epoch 的状态。"
                f" 可手动调用 trainer.load_checkpoint('{best_checkpoint_path}')。",
                exc_info=True,
            )

    # ── SWA 阶段（Optimization 2，可选）────────────────────────────────────
    # 在主训练（含最佳模型恢复）之后，以固定低 LR 继续训练若干 epoch，
    # 对途中权重快照取平均，找到比 SGD 终点更平坦的极小值。
    # 参考：Izmailov et al. (2018) "Averaging Weights Leads to Wider Optima"
    if config['training'].get('use_swa', False):
        try:
            from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
            _swa_available = True
        except ImportError:
            logger.warning(
                "⚠️ torch.optim.swa_utils 不可用（需要 PyTorch >= 1.6）。"
                " 跳过 SWA 阶段。"
            )
            _swa_available = False

        if _swa_available:
            swa_epochs = int(config['training'].get('swa_epochs', 10))
            swa_lr_ratio = float(config['training'].get('swa_lr_ratio', 0.05))
            swa_lr = config['training']['learning_rate'] * swa_lr_ratio

            logger.info("=" * 60)
            logger.info(f"📊 开始 SWA 阶段: {swa_epochs} epochs, LR={swa_lr:.2e}")
            logger.info("=" * 60)

            # 1. Build averaged model wrapper (wraps trainer.model in-place for inference)
            _orig_model = trainer.model  # keep reference for attribute access + compute_loss
            swa_model = AveragedModel(_orig_model)

            # 2. Replace scheduler with constant SWALR for SWA phase
            swa_scheduler = SWALR(
                trainer.optimizer,
                swa_lr=swa_lr,
                anneal_epochs=max(1, swa_epochs // 3),  # smooth transition
                anneal_strategy='cos',
            )

            for swa_ep in range(1, swa_epochs + 1):
                swa_train_loss = trainer.train_epoch(
                    train_graphs,
                    epoch=config['training']['num_epochs'] + swa_ep,
                    total_epochs=config['training']['num_epochs'] + swa_epochs,
                )
                swa_model.update_parameters(trainer.model)
                swa_scheduler.step()
                logger.info(
                    f"  SWA Epoch {swa_ep}/{swa_epochs}: "
                    f"train_loss={swa_train_loss:.4f}"
                )

            # 3. Update BatchNorm running statistics.
            #    The SWA-averaged weights never processed any real batch; BN layers
            #    (present in GraphNativeDecoder) still have stale running_mean/var
            #    from the original training run.  We use PyTorch's built-in
            #    update_bn() which resets running stats and re-estimates them via a
            #    cumulative moving average over the entire training set.
            #    update_bn() sets the model to train() mode internally and uses
            #    the iterable's items as positional arguments to swa_model.forward().
            #    GraphNativeBrainModel.forward(data) accepts a single HeteroData
            #    positional argument with all other params defaulting, so the
            #    plain list iteration works correctly.
            logger.info("  更新 SWA 模型 BatchNorm 统计量...")
            try:
                update_bn(train_graphs, swa_model, device=trainer.device)
                logger.info("  ✅ BatchNorm 统计量已更新")
            except Exception as _bn_err:
                logger.warning(
                    f"  BatchNorm 更新遇到问题: {_bn_err}。"
                    f" SWA 模型 BN 统计量可能不准确。",
                    exc_info=True,
                )

            # 4. Validate SWA model directly (without swapping trainer.model
            #    to avoid AveragedModel attribute-proxy issues).
            swa_model.eval()
            swa_total_loss = 0.0
            swa_ss_res: Dict[str, float] = {}
            swa_ss_tot: Dict[str, float] = {}
            with torch.no_grad():
                for val_data in val_graphs:
                    val_data = val_data.to(trainer.device)
                    # AveragedModel.forward() proxies to the wrapped module's forward.
                    # Use the original model for attribute checks like use_prediction.
                    if _orig_model.use_prediction:
                        recon_swa, _, enc_swa = swa_model(
                            val_data, return_prediction=False, return_encoded=True
                        )
                    else:
                        recon_swa, _ = swa_model(val_data, return_prediction=False)
                        enc_swa = None
                    # compute_loss is defined on the underlying module
                    swa_losses = _orig_model.compute_loss(val_data, recon_swa, encoded=enc_swa)
                    swa_total_loss += sum(swa_losses.values()).item()
                    # R² per modality
                    for nt in trainer.node_types:
                        if nt in val_data.node_types and nt in recon_swa:
                            tgt = val_data[nt].x
                            rec = recon_swa[nt]
                            T_min = min(tgt.shape[1], rec.shape[1])
                            tgt = tgt[:, :T_min, :]
                            rec = rec[:, :T_min, :]
                            if rec.shape[0] != tgt.shape[0]:
                                continue
                            tgt_mean = tgt.mean()
                            swa_ss_res[nt] = swa_ss_res.get(nt, 0.0) + ((tgt - rec) ** 2).sum().item()
                            swa_ss_tot[nt] = swa_ss_tot.get(nt, 0.0) + ((tgt - tgt_mean) ** 2).sum().item()
            swa_val_loss = swa_total_loss / max(len(val_graphs), 1)
            swa_r2_dict = {
                f'r2_{nt}': (1.0 - swa_ss_res.get(nt, 0.0) / max(swa_ss_tot.get(nt, 1e-12), 1e-12))
                for nt in trainer.node_types
            }

            swa_r2_str = "  ".join(f"{k}={v:.3f}" for k, v in sorted(swa_r2_dict.items()))
            logger.info(
                f"✅ SWA 完成: val_loss={swa_val_loss:.4f} "
                f"(主训练最佳: {best_val_loss:.4f})  {swa_r2_str}"
            )
            if swa_val_loss < best_val_loss:
                logger.info(
                    f"  🎯 SWA 验证损失优于主训练最佳 "
                    f"({swa_val_loss:.4f} < {best_val_loss:.4f})，"
                    f" 保存为 swa_model.pt 并推荐用于推理。"
                )
            else:
                logger.info(
                    f"  ℹ️  SWA 验证损失 ({swa_val_loss:.4f}) 未优于主训练最佳 "
                    f"({best_val_loss:.4f})。两种模型均已保存，可按需选择。"
                )

            # 5. Save SWA model — always save regardless of comparison
            #    (SWA's generalization benefit often shows on held-out test sets)
            swa_checkpoint_path = output_dir / "swa_model.pt"
            try:
                torch.save(swa_model.state_dict(), swa_checkpoint_path)
                logger.info(f"  💾 SWA 模型已保存: {swa_checkpoint_path}")
            except Exception as _save_err:
                logger.warning(f"  SWA 模型保存失败: {_save_err}")


def main():
    """主函数"""
    # 解析参数
    parser = argparse.ArgumentParser(description='TwinBrain V5 训练系统')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径 (default: configs/default.yaml)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (default: 42)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        metavar='CHECKPOINT',
        help='从指定检查点文件恢复训练 (例如: outputs/twinbrain_v5_xxx/best_model.pt)。'
             '加载后从保存的 epoch+1 继续，保留 model/optimizer/scheduler 状态。'
    )
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建输出目录
    output_dir = create_output_dir(
        config['output']['output_dir'],
        config['output']['experiment_name']
    )
    config['output']['output_dir'] = str(output_dir)
    
    # 设置日志
    logger = setup_logging(
        output_dir / "training.log",
        level=config['output']['log_level']
    )
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 保存配置
    save_config(config, output_dir / "config.yaml")
    
    # 打印配置
    logger.info("=" * 60)
    logger.info("TwinBrain V5 - 图原生数字孪生脑训练系统")
    logger.info("=" * 60)
    logger.info(f"配置文件: {args.config or 'configs/default.yaml'}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"设备: {config['device']['type']}")
    logger.info(f"随机种子: {args.seed}")
    if args.resume:
        logger.info(f"断点续训: {args.resume}")
    logger.info("=" * 60)
    
    try:
        # 步骤1-2: 加载数据 & 构建图（缓存感知，命中时跳过原始数据加载）
        # subject_to_idx: {subject_id_str → int_idx}，传给 create_model 以创建正确大小的 Embedding
        graphs, mapper, subject_to_idx = build_graphs(config, logger)

        # 持久化 subject_to_idx 映射，确保推理时能将被试 ID 还原到 Embedding 索引。
        # 不保存此文件则无法在训练后推理时恢复正确的 subject_idx，
        # 导致被试特异性嵌入无法使用（详见 SPEC.md §2.4）。
        if subject_to_idx:
            sidx_path = Path(config['output']['output_dir']) / "subject_to_idx.json"
            try:
                with open(sidx_path, "w", encoding="utf-8") as _f:
                    json.dump(subject_to_idx, _f, ensure_ascii=False, indent=2)
                logger.info(f"被试索引映射已保存: {sidx_path}")
            except OSError as _e:
                logger.warning(f"保存 subject_to_idx 失败 ({sidx_path}): {_e}")

        # 步骤3: 创建模型
        model = create_model(config, logger, num_subjects=len(subject_to_idx))
        
        # 启动前打印一次人类可读的配置核对表，方便快速验证参数
        log_training_summary(config, graphs, model, logger)
        
        # 步骤4: 训练
        train_model(model, graphs, config, logger, resume_checkpoint=args.resume)
        
        logger.info("=" * 60)
        logger.info("✅ 所有任务完成!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ 运行失败: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
