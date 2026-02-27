"""
工具函数
=======

通用辅助函数
"""

import logging
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import yaml


def setup_logging(
    log_file: Path = None,
    level: str = 'INFO',
) -> logging.Logger:
    """
    设置日志
    
    Args:
        log_file: 日志文件路径
        level: 日志级别
        
    Returns:
        logger对象
    """
    # 创建logger
    logger = logging.getLogger('twinbrain_v5')
    logger.setLevel(getattr(logging, level))
    
    # 清除已有的handlers
    logger.handlers = []
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int = 42):
    """
    设置随机种子
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_output_dir(
    base_dir: str,
    experiment_name: str,
) -> Path:
    """
    创建输出目录
    
    Args:
        base_dir: 基础目录
        experiment_name: 实验名称
        
    Returns:
        输出目录路径
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    (output_dir / 'results').mkdir(exist_ok=True)
    
    return output_dir


def save_config(config: dict, save_path: Path):
    """
    保存配置文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: str = 'cpu',
) -> int:
    """
    加载检查点
    
    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器 (可选)
        device: 设备
        
    Returns:
        epoch: 当前epoch
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    
    return epoch


def count_parameters(model: torch.nn.Module) -> int:
    """
    统计模型参数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_subject_graph_from_cache(cache_path: str) -> dict:
    """从缓存文件加载被试脑图数据（供下游分析使用）。

    训练流程将每个 (被试, 任务) 的 EEG 和 fMRI 时序特征缓存为 PyG
    HeteroData 异质图文件（*.pt）。此函数提供一个简单的字典接口，
    让下游代码无需了解 PyG 即可直接读取 EEG 和 fMRI 的时序数据和图拓扑。

    缓存文件结构（训练管线自动生成）：
    - 节点类型 'eeg': 电极通道图节点
        x          : [N_eeg, T_eeg, 1] float32 — 标准化后的 EEG 时序信号
        num_nodes  : int — 电极数（N_eeg）
        pos        : [N_eeg, 3] float32（可选）— 电极 3D 坐标（mm）
        sampling_rate: float — 采样率（Hz），通常 250.0
    - 节点类型 'fmri': 脑图谱 ROI 图节点
        x          : [N_fmri, T_fmri, 1] float32 — 标准化后的 BOLD 时序信号
        num_nodes  : int — ROI 数（N_fmri）
        pos        : [N_fmri, 3] float32（可选）— ROI 质心 MNI 坐标（mm）
        sampling_rate: float — TR 的倒数（Hz），通常约 0.5 Hz（TR ≈ 2s）
    - 边类型 ('eeg','connects','eeg'): EEG 同模态功能连通性边
        edge_index : [2, E_eeg] int64
        edge_attr  : [E_eeg, 1] float32 — |Pearson r| 或 coherence
    - 边类型 ('fmri','connects','fmri'): fMRI 同模态功能连通性边
        edge_index : [2, E_fmri] int64
        edge_attr  : [E_fmri, 1] float32 — |Pearson r|

    注意：跨模态边 ('eeg','projects_to','fmri') 不存入缓存文件，
    由训练管线在每次加载时动态重建（见 main.py build_graphs）。
    下游如需跨模态边，可直接调用
    ``GraphNativeBrainMapper.create_simple_cross_modal_edges(graph)``。

    Args:
        cache_path: 缓存 .pt 文件路径（绝对路径或相对路径）。

    Returns:
        dict，包含以下键（取决于缓存内容）：
        - 'eeg_timeseries'  : np.ndarray [N_eeg, T_eeg] or None
        - 'fmri_timeseries' : np.ndarray [N_fmri, T_fmri] or None
        - 'eeg_n_channels'  : int or None
        - 'fmri_n_rois'     : int or None
        - 'eeg_sampling_rate': float or None
        - 'fmri_sampling_rate': float or None
        - 'eeg_pos'         : np.ndarray [N_eeg, 3] or None
        - 'fmri_pos'        : np.ndarray [N_fmri, 3] or None
        - 'eeg_edge_index'  : np.ndarray [2, E_eeg] or None
        - 'eeg_edge_attr'   : np.ndarray [E_eeg] or None
        - 'fmri_edge_index' : np.ndarray [2, E_fmri] or None
        - 'fmri_edge_attr'  : np.ndarray [E_fmri] or None
        - 'node_types'      : list[str]  — e.g. ['eeg', 'fmri']
        - 'edge_types'      : list[tuple] — e.g. [('eeg','connects','eeg'), ...]

    Example::

        data = load_subject_graph_from_cache('outputs/graph_cache/sub-01_GRADON_abcd1234.pt')
        eeg  = data['eeg_timeseries']   # ndarray [N_eeg, T_eeg]
        fmri = data['fmri_timeseries']  # ndarray [N_fmri, T_fmri]
        print(f"EEG: {eeg.shape}, fMRI: {fmri.shape}")
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"缓存文件不存在: {cache_path}\n"
            f"请先运行训练管线生成缓存，或检查路径是否正确。"
        )

    graph = torch.load(cache_path, map_location='cpu', weights_only=False)

    result: dict = {
        'node_types': list(graph.node_types),
        'edge_types': list(graph.edge_types),
        'eeg_timeseries': None,
        'fmri_timeseries': None,
        'eeg_n_channels': None,
        'fmri_n_rois': None,
        'eeg_sampling_rate': None,
        'fmri_sampling_rate': None,
        'eeg_pos': None,
        'fmri_pos': None,
        'eeg_edge_index': None,
        'eeg_edge_attr': None,
        'fmri_edge_index': None,
        'fmri_edge_attr': None,
    }

    # Extract EEG data
    if 'eeg' in graph.node_types and hasattr(graph['eeg'], 'x') and graph['eeg'].x is not None:
        eeg_x = graph['eeg'].x   # [N_eeg, T, 1]
        result['eeg_timeseries'] = eeg_x.squeeze(-1).numpy()  # [N_eeg, T]
        result['eeg_n_channels'] = eeg_x.shape[0]
        result['eeg_sampling_rate'] = getattr(graph['eeg'], 'sampling_rate', None)
        if hasattr(graph['eeg'], 'pos') and graph['eeg'].pos is not None:
            result['eeg_pos'] = graph['eeg'].pos.numpy()

    # Extract fMRI data
    if 'fmri' in graph.node_types and hasattr(graph['fmri'], 'x') and graph['fmri'].x is not None:
        fmri_x = graph['fmri'].x  # [N_fmri, T, 1]
        result['fmri_timeseries'] = fmri_x.squeeze(-1).numpy()  # [N_fmri, T]
        result['fmri_n_rois'] = fmri_x.shape[0]
        result['fmri_sampling_rate'] = getattr(graph['fmri'], 'sampling_rate', None)
        if hasattr(graph['fmri'], 'pos') and graph['fmri'].pos is not None:
            result['fmri_pos'] = graph['fmri'].pos.numpy()

    # Extract EEG connectivity edges
    _eeg_et = ('eeg', 'connects', 'eeg')
    if _eeg_et in graph.edge_types:
        result['eeg_edge_index'] = graph[_eeg_et].edge_index.numpy()
        if hasattr(graph[_eeg_et], 'edge_attr') and graph[_eeg_et].edge_attr is not None:
            result['eeg_edge_attr'] = graph[_eeg_et].edge_attr.squeeze(-1).numpy()

    # Extract fMRI connectivity edges
    _fmri_et = ('fmri', 'connects', 'fmri')
    if _fmri_et in graph.edge_types:
        result['fmri_edge_index'] = graph[_fmri_et].edge_index.numpy()
        if hasattr(graph[_fmri_et], 'edge_attr') and graph[_fmri_et].edge_attr is not None:
            result['fmri_edge_attr'] = graph[_fmri_et].edge_attr.squeeze(-1).numpy()

    return result
