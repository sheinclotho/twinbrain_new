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
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
from torch_geometric.data import HeteroData

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from data.loaders import BrainDataLoader
from models.graph_native_mapper import GraphNativeBrainMapper
from models.graph_native_system import GraphNativeBrainModel, GraphNativeTrainer
from utils.helpers import setup_logging, set_seed, save_config, create_output_dir


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
    
    def process_fmri_timeseries(fmri_data, min_volumes=10):
        """Helper to extract and normalize fMRI timeseries."""
        if fmri_data.ndim == 4:  # 4D fMRI
            n_volumes = fmri_data.shape[-1]
        elif fmri_data.ndim == 3:  # 3D (already ROI timeseries)
            n_volumes = fmri_data.shape[-1]
        else:
            return None, f"Unsupported fMRI shape: {fmri_data.shape}"
        
        if n_volumes < min_volumes:
            return None, f"Too few volumes: {n_volumes} < {min_volumes}"
        
        # Average and normalize
        fmri_ts = fmri_data.reshape(-1, n_volumes).mean(axis=0)
        fmri_ts = (fmri_ts - fmri_ts.mean()) / (fmri_ts.std() + 1e-8)
        return fmri_ts.reshape(1, -1), None
    
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
    
    # 为每个被试构建图
    graphs = []
    for subject_data in all_data:
        graph_list = []
        
        # fMRI图
        if 'fmri' in subject_data:
            fmri_data = subject_data['fmri']['data']
            fmri_ts, error = process_fmri_timeseries(fmri_data)
            
            if error:
                logger.warning(f"fMRI processing failed: {error}, skipping")
                continue
            
            fmri_graph = mapper.map_fmri_to_graph(
                timeseries=fmri_ts,
                connectivity_matrix=None,  # 自动计算
            )
            graph_list.append(('fmri', fmri_graph))
        
        # EEG图
        if 'eeg' in subject_data:
            eeg_data = subject_data['eeg']['data']  # [n_channels, n_times]
            eeg_ch_names = subject_data['eeg']['ch_names']
            
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
            
            eeg_graph = mapper.map_eeg_to_graph(
                timeseries=eeg_data,
                channel_names=eeg_ch_names,
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
                
                # Add cross-modal edges if we have both modalities
                if 'fmri' in merged_graph.node_types and 'eeg' in merged_graph.node_types:
                    # Create simple cross-modal connections (can be improved with atlas mapping)
                    # Returns edges from EEG to fMRI [eeg_idx, fmri_idx]
                    cross_edges = mapper.create_simple_cross_modal_edges(merged_graph)
                    if cross_edges is not None:
                        # Edge direction: EEG -> fMRI
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
    
    # 跨模态边
    if len(node_types) > 1:
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
