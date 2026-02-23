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
import logging
import os
import sys
from pathlib import Path
from typing import Optional
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
    
    # 加载所有被试
    all_data = data_loader.load_all_subjects(
        task=config['data'].get('task'),
        max_subjects=config['data'].get('max_subjects'),
    )
    
    if not all_data:
        raise ValueError("未加载到任何数据，请检查数据路径配置")
    
    logger.info(f"成功加载 {len(all_data)} 个被试数据")
    
    return all_data


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
            from nilearn.input_data import NiftiLabelsMasker
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
    
    # 为每个被试构建图
    max_seq_len = config['training'].get('max_seq_len', None)
    if max_seq_len is not None:
        logger.info(f"Sequence truncation enabled: max_seq_len={max_seq_len} (prevents CUDA OOM)")
    graphs = []
    for subject_data in all_data:
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
            
            # Truncate to max_seq_len to prevent CUDA OOM with long sequences
            if max_seq_len is not None:
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
            
            # Truncate to max_seq_len to prevent CUDA OOM with long sequences
            if max_seq_len is not None:
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
                graphs.append(graph_list[0][1])
            else:
                # Multi-modal: merge into heterograph
                merged_graph = HeteroData()
                for modality, graph in graph_list:
                    # Copy node features and structure
                    for key in graph.node_types:
                        merged_graph[key].x = graph[key].x
                        if hasattr(graph[key], 'num_nodes'):
                            merged_graph[key].num_nodes = graph[key].num_nodes
                        if hasattr(graph[key], 'pos'):
                            merged_graph[key].pos = graph[key].pos
                    
                    # Copy edge structure
                    for edge_type in graph.edge_types:
                        merged_graph[edge_type].edge_index = graph[edge_type].edge_index
                        if hasattr(graph[edge_type], 'edge_attr'):
                            merged_graph[edge_type].edge_attr = graph[edge_type].edge_attr
                
                # 跨模态边：EEG → fMRI
                # 设计理念：EEG 电极（较少节点）向 fMRI ROI（较多节点）投射信号。
                # create_simple_cross_modal_edges 会验证 N_eeg < N_fmri 并在违反时给出警告。
                if 'fmri' in merged_graph.node_types and 'eeg' in merged_graph.node_types:
                    cross_edges = mapper.create_simple_cross_modal_edges(merged_graph)
                    if cross_edges is not None:
                        merged_graph['eeg', 'projects_to', 'fmri'].edge_index = cross_edges
                
                graphs.append(merged_graph)
    
    if len(graphs) == 0:
        raise ValueError("No valid graphs constructed. Check data quality and preprocessing.")
    
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
