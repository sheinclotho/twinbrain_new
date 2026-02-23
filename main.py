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
import sys
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


def prepare_data(config: dict, logger: logging.Logger):
    """准备训练数据"""
    logger.info("=" * 60)
    logger.info("步骤 1/4: 加载数据")
    logger.info("=" * 60)
    
    # 初始化数据加载器
    data_loader = BrainDataLoader(
        data_root=config['data']['root_dir'],
        modalities=config['data']['modalities'],
    )
    
    # 解析任务列表配置
    # 优先使用 tasks（列表），兼容旧版 task（单字符串）
    tasks = config['data'].get('tasks')
    if tasks is None:
        legacy_task = config['data'].get('task')
        if legacy_task is not None:
            tasks = [legacy_task]
            logger.info(
                f"使用旧版 'task: {legacy_task}' 配置。"
                f" 建议迁移到 'tasks: [{legacy_task}]'。"
            )
        # tasks 仍为 None → 自动发现所有任务
    elif isinstance(tasks, str):
        tasks = [tasks]

    if tasks is None:
        logger.info("tasks: null → 自动发现每个被试的所有任务")
    else:
        logger.info(f"将加载以下任务: {tasks}")
    
    # 加载所有被试（可跨多任务）
    all_data = data_loader.load_all_subjects(
        tasks=tasks,
        max_subjects=config['data'].get('max_subjects'),
    )
    
    if not all_data:
        raise ValueError("未加载到任何数据，请检查数据路径配置")
    
    logger.info(f"成功加载 {len(all_data)} 个被试-任务组合")
    
    return all_data


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
            # 复制静态属性（节点数、空间坐标、采样率）
            for attr in ('num_nodes', 'pos', 'sampling_rate'):
                if hasattr(full_graph[nt], attr):
                    setattr(win[nt], attr, getattr(full_graph[nt], attr))

        windows.append(win)

    return windows


def build_graphs(all_data, config: dict, logger: logging.Logger):
    """构建图结构"""
    logger.info("=" * 60)
    logger.info("步骤 2/4: 构建图结构")
    logger.info("=" * 60)
    
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
        we get 200 anatomically meaningful nodes — the actual design intent.
        """
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
        cache_dir = Path(cache_cfg.get('dir', 'outputs/graph_cache'))
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

    graphs: List[HeteroData] = []
    n_cached = 0
    n_windows_total = 0
    for subject_data in all_data:
        subject_id = subject_data.get('subject_id', 'unknown')
        task = subject_data.get('task')

        # ── 尝试从缓存加载 ──────────────────────────────────────
        if cache_dir is not None:
            cache_key = _graph_cache_key(subject_id, task, config)
            cache_path = cache_dir / cache_key
            if cache_path.exists():
                try:
                    full_graph = torch.load(cache_path, map_location='cpu', weights_only=False)
                    win_samples = extract_windowed_samples(full_graph, w_cfg, logger)
                    graphs.extend(win_samples)
                    n_windows_total += len(win_samples)
                    n_cached += 1
                    logger.debug(
                        f"从缓存加载图: {cache_key}"
                        + (f" → {len(win_samples)} 个窗口" if windowed else "")
                    )
                    continue
                except Exception as e:
                    logger.warning(f"缓存加载失败 ({cache_key}): {e}，重新构建")

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
            
            # 截断仅在单样本训练模式下启用（防 CUDA OOM）。
            # 窗口模式下不截断，以使连通性估计来自完整 run。
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
                    # Copy node features and structure
                    for key in graph.node_types:
                        built_graph[key].x = graph[key].x
                        if hasattr(graph[key], 'num_nodes'):
                            built_graph[key].num_nodes = graph[key].num_nodes
                        if hasattr(graph[key], 'pos'):
                            built_graph[key].pos = graph[key].pos
                    
                    # Copy edge structure
                    for edge_type in graph.edge_types:
                        built_graph[edge_type].edge_index = graph[edge_type].edge_index
                        if hasattr(graph[edge_type], 'edge_attr'):
                            built_graph[edge_type].edge_attr = graph[edge_type].edge_attr
                
                # 跨模态边：EEG → fMRI
                # 设计理念：EEG 电极（较少节点）向 fMRI ROI（较多节点）投射信号。
                # create_simple_cross_modal_edges 会验证 N_eeg < N_fmri 并在违反时给出警告。
                if 'fmri' in built_graph.node_types and 'eeg' in built_graph.node_types:
                    cross_edges = mapper.create_simple_cross_modal_edges(built_graph)
                    if cross_edges is not None:
                        built_graph['eeg', 'projects_to', 'fmri'].edge_index = cross_edges
            
            # ── 保存到缓存（始终保存完整 run 图） ──────────────────
            if cache_dir is not None:
                try:
                    cache_key = _graph_cache_key(subject_id, task, config)
                    cache_path = cache_dir / cache_key
                    torch.save(built_graph, cache_path)
                    logger.debug(f"图已缓存: {cache_key}")
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

    if len(graphs) == 0:
        raise ValueError("No valid graphs constructed. Check data quality and preprocessing.")

    # ── 汇总日志 ────────────────────────────────────────────────
    n_runs = len(all_data)
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

    return graphs, mapper


def create_model(config: dict, logger: logging.Logger):
    """创建模型"""
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


def train_model(model, graphs, config: dict, logger: logging.Logger):
    """训练模型"""
    logger.info("=" * 60)
    logger.info("步骤 4/4: 训练模型")
    logger.info("=" * 60)
    
    # 划分训练/验证集 - Ensure both train and validation have samples
    if len(graphs) < 2:
        logger.error(f"❌ 数据不足: 需要至少2个样本进行训练，但只有 {len(graphs)} 个样本")
        logger.error("提示: 请增加数据量或调整 max_subjects 配置")
        raise ValueError(f"需要至少2个样本进行训练,但只有 {len(graphs)} 个。请检查数据配置。")
    
    # Use at least 10% or 1 sample for validation, ensure both train and val have at least 1
    min_val_samples = max(1, len(graphs) // 10)
    n_train = len(graphs) - min_val_samples
    
    # Safety check: ensure both sets have at least 1 sample
    if n_train < 1:
        n_train = 1
        min_val_samples = len(graphs) - 1
    
    train_graphs = graphs[:n_train]
    val_graphs = graphs[n_train:]
    
    logger.info(f"训练集: {len(train_graphs)} 个样本")
    logger.info(f"验证集: {len(val_graphs)} 个样本")
    
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
    )
    logger.info("✅ 训练器初始化完成")
    logger.info("=" * 60)
    logger.info("开始训练循环")
    logger.info("=" * 60)
    
    # 训练循环
    import time
    best_val_loss = float('inf')
    patience_counter = 0
    no_improvement_warning_shown = False
    epoch_times = []
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
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
            val_loss = trainer.validate(val_graphs)
            
            # Step scheduler based on validation loss (for ReduceLROnPlateau)
            trainer.step_scheduler_on_validation(val_loss)
            
            # Check for NaN validation loss
            if np.isnan(val_loss) or np.isinf(val_loss):
                logger.error(f"❌ Validation loss is NaN/Inf at epoch {epoch}. Stopping training.")
                raise ValueError("Validation diverged: loss is NaN or Inf")
            
            logger.info(
                f"✓ Epoch {epoch}/{config['training']['num_epochs']}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"time={epoch_time:.1f}s, ETA={eta_str}"
            )
            
            # Warn if no improvement after many epochs
            if epoch >= 50 and best_val_loss == float('inf') and not no_improvement_warning_shown:
                logger.warning("⚠️ No improvement in validation loss after 50 epochs. Check data quality and hyperparameters.")
                no_improvement_warning_shown = True
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                improvement = (best_val_loss - val_loss) / best_val_loss * 100 if best_val_loss != float('inf') else 100
                best_val_loss = val_loss
                patience_counter = 0
                
                # 保存检查点
                output_dir = Path(config['output']['output_dir'])
                checkpoint_path = output_dir / "best_model.pt"
                trainer.save_checkpoint(checkpoint_path, epoch)
                if improvement != 100:
                    logger.info(f"  🎯 保存最佳模型: val_loss={val_loss:.4f} (提升 {improvement:.1f}%)")
                else:
                    logger.info(f"  🎯 保存最佳模型: val_loss={val_loss:.4f}")
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= config['training']['early_stopping_patience']:
                logger.info(f"⏹️ 早停触发: {patience_counter} 个epoch无改进")
                break
        else:
            logger.info(
                f"✓ Epoch {epoch}/{config['training']['num_epochs']}: "
                f"train_loss={train_loss:.4f}, time={epoch_time:.1f}s, ETA={eta_str}"
            )
        
        # 定期保存检查点
        if epoch % config['training']['save_frequency'] == 0:
            output_dir = Path(config['output']['output_dir'])
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            trainer.save_checkpoint(checkpoint_path, epoch)
    
    logger.info("训练完成!")
    logger.info(f"最佳验证损失: {best_val_loss:.4f}")


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
    logger.info("=" * 60)
    
    try:
        # 步骤1: 准备数据
        all_data = prepare_data(config, logger)
        
        # 步骤2: 构建图
        graphs, mapper = build_graphs(all_data, config, logger)
        
        # 步骤3: 创建模型
        model = create_model(config, logger)
        
        # 启动前打印一次人类可读的配置核对表，方便快速验证参数
        log_training_summary(config, graphs, model, logger)
        
        # 步骤4: 训练
        train_model(model, graphs, config, logger)
        
        logger.info("=" * 60)
        logger.info("✅ 所有任务完成!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ 运行失败: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
