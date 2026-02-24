"""
Complete Graph-Native Training System
=====================================

Full reimagination of TwinBrain training pipeline:
1. Graph-native from data to prediction
2. No unnecessary conversions
3. Spatial-temporal modeling on graphs
4. Efficient and interpretable

This is a COMPLETE standalone system, not just optimization modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

# Import AMP components if available (for mixed precision training)
AMP_AVAILABLE = False
USE_NEW_AMP_API = False
try:
    # Try new API: autocast from torch.amp (requires device_type parameter)
    from torch.amp import autocast
    # GradScaler might still be in torch.cuda.amp in some PyTorch versions
    try:
        from torch.amp import GradScaler
    except ImportError:
        from torch.cuda.amp import GradScaler
    AMP_AVAILABLE = True
    USE_NEW_AMP_API = True
except ImportError:
    try:
        # Fallback to old API if new one not available
        from torch.cuda.amp import autocast, GradScaler
        AMP_AVAILABLE = True
        USE_NEW_AMP_API = False
    except ImportError:
        AMP_AVAILABLE = False
        autocast = None
        GradScaler = None

from .graph_native_mapper import GraphNativeBrainMapper, TemporalGraphFeatureExtractor
from .graph_native_encoder import GraphNativeEncoder, SpatialTemporalGraphConv
from .adaptive_loss_balancer import AdaptiveLossBalancer
from .eeg_channel_handler import EnhancedEEGHandler
from .advanced_prediction import EnhancedMultiStepPredictor

logger = logging.getLogger(__name__)


class GraphNativeDecoder(nn.Module):
    """
    Graph-native decoder that reconstructs temporal signals.
    
    Decodes from latent graph representations back to temporal signals
    WITHOUT breaking graph structure.
    """
    
    def __init__(
        self,
        node_types: List[str],
        hidden_channels: int,
        out_channels_dict: Dict[str, int],
        num_layers: int = 3,
        temporal_upsample: Optional[int] = None,
    ):
        """
        Initialize decoder.
        
        Args:
            node_types: Node types to decode
            hidden_channels: Hidden dimension from encoder
            out_channels_dict: Output channels per node type
            num_layers: Number of decoding layers
            temporal_upsample: Upsample factor for temporal dimension
        """
        super().__init__()
        
        self.node_types = node_types
        self.hidden_channels = hidden_channels
        self.out_channels_dict = out_channels_dict
        self.temporal_upsample = temporal_upsample
        
        # Temporal deconvolution per node type
        self.temporal_deconv = nn.ModuleDict()
        
        for node_type in node_types:
            layers = []
            current_dim = hidden_channels
            
            # Stack of convolution layers
            for i in range(num_layers):
                out_dim = hidden_channels // (2 ** i) if i < num_layers - 1 else out_channels_dict[node_type]
                use_stride2 = bool(temporal_upsample and i == 0)
                
                if use_stride2:
                    # Temporal upsampling: ConvTranspose1d doubles T (stride=2 is correct here)
                    layers.append(nn.ConvTranspose1d(
                        current_dim, out_dim, kernel_size=4, stride=2, padding=1,
                    ))
                else:
                    # Feature transform at same temporal resolution.
                    # Conv1d(kernel_size=3, padding=1) preserves T exactly.
                    # ConvTranspose1d(kernel_size=4, stride=1, padding=1) silently adds 1
                    # to T per layer — causing shape mismatch in compute_loss.
                    layers.append(nn.Conv1d(
                        current_dim, out_dim, kernel_size=3, padding=1,
                    ))
                
                if i < num_layers - 1:
                    layers.append(nn.BatchNorm1d(out_dim))
                    layers.append(nn.ReLU())
                
                current_dim = out_dim
            
            self.temporal_deconv[node_type] = nn.Sequential(*layers)
    
    def forward(self, encoded_data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Decode to temporal signals.
        
        Args:
            encoded_data: Encoded HeteroData with features [N, T, H]
            
        Returns:
            Reconstructed signals per node type [N, T', C_out]
        """
        reconstructed = {}
        
        for node_type in self.node_types:
            if node_type in encoded_data.node_types:
                x = encoded_data[node_type].x  # [N, T, H]
                N, T, H = x.shape
                
                # Reshape for conv: [N, T, H] -> [N, H, T]
                x = x.permute(0, 2, 1)
                
                # Apply temporal deconvolution
                x_recon = self.temporal_deconv[node_type](x)  # [N, C_out, T']
                
                # Reshape back: [N, C_out, T'] -> [N, T', C_out]
                x_recon = x_recon.permute(0, 2, 1)
                
                reconstructed[node_type] = x_recon
        
        return reconstructed


class GraphNativeBrainModel(nn.Module):
    """
    Complete graph-native brain model.
    
    End-to-end architecture:
    1. Mapper: Build graph from brain data
    2. Encoder: Spatial-temporal encoding on graph
    3. Predictor: Future prediction on graph
    4. Decoder: Reconstruct signals
    
    NO sequence conversions - pure graph operations throughout.
    """
    
    # 潜空间预测切分比例：前 CONTEXT_RATIO 作为 context，余下部分为 future target。
    # context ≥ 2/3 确保 StratifiedWindowSampler 能在 context 内取到至少 1 个完整窗口。
    _PRED_CONTEXT_RATIO: float = 2 / 3
    # 潜空间预测所需最小序列长度（= 保证 T_ctx ≥ 1 且 T_fut ≥ 1）
    _PRED_MIN_SEQ_LEN: int = 4
    
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        in_channels_dict: Dict[str, int],
        hidden_channels: int = 128,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 3,
        use_prediction: bool = True,
        prediction_steps: int = 10,
        dropout: float = 0.1,
        loss_type: str = 'mse',
        use_gradient_checkpointing: bool = False,
        predictor_config: Optional[Dict] = None,
    ):
        """
        Initialize complete model.
        
        Args:
            node_types: Node types (e.g., ['fmri', 'eeg'])
            edge_types: Edge types
            in_channels_dict: Input channels per modality
            hidden_channels: Hidden dimension
            num_encoder_layers: Encoder depth
            num_decoder_layers: Decoder depth
            use_prediction: Enable future prediction
            prediction_steps: Steps to predict ahead
            dropout: Dropout rate
            loss_type: Loss function type ('mse', 'huber', 'smooth_l1')
            use_gradient_checkpointing: Free intermediate activations per timestep
                to avoid MemoryError on long sequences (trades memory for compute)
            predictor_config: Optional dict from config['v5_optimization']['advanced_prediction'].
                Keys: use_hierarchical, use_transformer, use_uncertainty, num_scales,
                num_windows, sampling_strategy.  Defaults used when None.
        """
        super().__init__()
        
        self.node_types = node_types
        self.hidden_channels = hidden_channels
        self.use_prediction = use_prediction
        self.loss_type = loss_type
        
        # Encoder: Graph-native spatial-temporal encoding
        self.encoder = GraphNativeEncoder(
            node_types=node_types,
            edge_types=edge_types,
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
            num_layers=num_encoder_layers,
            use_gradient_checkpointing=use_gradient_checkpointing,
            dropout=dropout,
        )
        
        # Decoder: Reconstruct temporal signals
        self.decoder = GraphNativeDecoder(
            node_types=node_types,
            hidden_channels=hidden_channels,
            out_channels_dict=in_channels_dict,
            num_layers=num_decoder_layers,
        )
        
        # Predictor: Future prediction (optional)
        if use_prediction:
            pred_cfg = predictor_config or {}
            self.predictor = EnhancedMultiStepPredictor(
                input_dim=hidden_channels,
                hidden_dim=hidden_channels * 2,
                prediction_steps=prediction_steps,
                use_hierarchical=pred_cfg.get('use_hierarchical', True),
                use_transformer=pred_cfg.get('use_transformer', True),
                use_uncertainty=pred_cfg.get('use_uncertainty', True),
                num_scales=pred_cfg.get('num_scales', 3),
                num_windows=pred_cfg.get('num_windows', 3),
                sampling_strategy=pred_cfg.get('sampling_strategy', 'uniform'),
            )
    
    def forward(
        self,
        data: HeteroData,
        return_prediction: bool = False,
        return_encoded: bool = False,
    ) -> Tuple:
        """
        Forward pass with input validation.
        
        Args:
            data: Input HeteroData with temporal features
            return_prediction: Whether to return future predictions
            return_encoded: Whether to return latent encoded representations
                {node_type: h[N, T, H]} — needed by compute_loss for
                the latent-space self-supervised prediction loss.
            
        Returns:
            When return_encoded=False (default):
                (reconstructed, predictions) — 2-tuple
            When return_encoded=True:
                (reconstructed, predictions, encoded_dict) — 3-tuple
            reconstructed: {node_type: tensor[N, T, C]}
            predictions: {node_type: tensor[N, steps, H]} or None
            encoded_dict: {node_type: tensor[N, T, H]}
        """
        # Input validation (use explicit checks, not assertions)
        for node_type in self.node_types:
            if node_type in data.node_types and hasattr(data[node_type], 'x'):
                x = data[node_type].x
                if x.ndim != 3:
                    raise ValueError(f"Expected [N, T, C] for {node_type}, got {x.shape}")
                if torch.isnan(x).any():
                    raise ValueError(f"NaN detected in {node_type} input")
                if torch.isinf(x).any():
                    raise ValueError(f"Inf detected in {node_type} input")
        
        # 1. Encode: Graph-native spatial-temporal encoding
        encoded_data = self.encoder(data)
        
        # 2. Decode: Reconstruct signals
        reconstructed = self.decoder(encoded_data)
        
        # 3. Predict: Future steps (optional)
        predictions = None
        if return_prediction and self.use_prediction:
            predictions = {}
            for node_type in self.node_types:
                if node_type in encoded_data.node_types:
                    h = encoded_data[node_type].x  # [N, T, H]

                    # Treat N nodes as the batch dimension.
                    # EnhancedMultiStepPredictor.forward expects [batch, seq_len, dim].
                    # h.unsqueeze(0) would produce [1, N, T, H] (4-D) → window sampler
                    # would fail to unpack 3 dims.  Use h directly so nodes = batch.
                    # Returns (predictions, targets, uncertainties):
                    #   predictions: [num_windows, N, prediction_steps, H]
                    pred_windows, _, _ = self.predictor(h, return_uncertainty=False)

                    # Average across sampled windows → [N, prediction_steps, H]
                    predictions[node_type] = pred_windows.mean(dim=0)
        
        # 4. Optionally return latent encoded dict for compute_loss prediction loss
        if return_encoded:
            encoded_dict = {
                nt: encoded_data[nt].x
                for nt in self.node_types
                if nt in encoded_data.node_types
            }
            return reconstructed, predictions, encoded_dict

        return reconstructed, predictions
    
    def compute_loss(
        self,
        data: HeteroData,
        reconstructed: Dict[str, torch.Tensor],
        predictions: Optional[Dict[str, torch.Tensor]] = None,
        encoded: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            data: Original data
            reconstructed: Reconstructed signals
            predictions: Unused (kept for API compatibility)
            encoded: Latent representations {node_type: h[N, T, H]}.
                When provided together with use_prediction=True, a
                self-supervised prediction loss is computed entirely in
                latent space (first 2/3 → predict last 1/3), giving the
                predictor a real training signal for the first time.
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Reconstruction loss per modality
        for node_type in self.node_types:
            if node_type in data.node_types and node_type in reconstructed:
                target = data[node_type].x  # [N, T, C]
                recon = reconstructed[node_type]  # [N, T', C_out]
                
                # Align temporal dimensions: the decoder may produce T' ≠ T when
                # temporal_upsample is set, or (defensively) if any upstream change
                # shifts T.  Truncate to the shorter of the two.
                T_min = min(target.shape[1], recon.shape[1])
                if target.shape[1] != recon.shape[1]:
                    logger.warning(
                        f"Decoder output T={recon.shape[1]} ≠ target T={target.shape[1]} "
                        f"for {node_type}; truncating to T={T_min}."
                    )
                    recon = recon[:, :T_min, :]
                    target = target[:, :T_min, :]
                
                # Guard against N-mismatch: this signals the cross-modal encoder bug
                # (propagate() called without size=(N_src, N_dst)).  Raise clearly
                # rather than letting broadcasting silently corrupt gradients.
                if recon.shape[0] != target.shape[0]:
                    raise RuntimeError(
                        f"Node count mismatch for '{node_type}': "
                        f"recon has {recon.shape[0]} nodes but target has {target.shape[0]}. "
                        f"This usually means propagate() was called without size=(N_src, N_dst) "
                        f"for a cross-modal edge, causing N_src to bleed into N_dst."
                    )
                
                # Choose loss function
                if self.loss_type == 'huber':
                    recon_loss = F.huber_loss(recon, target, delta=1.0)
                elif self.loss_type == 'smooth_l1':
                    recon_loss = F.smooth_l1_loss(recon, target)
                else:
                    recon_loss = F.mse_loss(recon, target)
                
                losses[f'recon_{node_type}'] = recon_loss
        
        # ── 潜空间自监督预测损失 ─────────────────────────────────────────
        # 将编码器潜空间序列切分为 context（前 2/3）→ 预测 future（后 1/3）。
        # 与旧代码（仅推理时运行预测头）的关键区别：
        #   旧：预测头参数从不接收梯度信号 → 实为死代码
        #   新：context/future 均在潜空间 H，可直接 MSE/Huber 比较，预测头真正被训练
        # 注：该 loss 隐式包含跨模态信息——编码器的 ST-GCN 跨模态边使 fMRI 潜向量中
        #     已包含 EEG 信息（反之亦然），故"预测 fMRI 潜向量未来"等价于用混合了
        #     EEG 信息的表征预测混合了 EEG 信息的未来表征。
        if encoded is not None and self.use_prediction:
            for node_type in self.node_types:
                if node_type not in encoded:
                    continue
                h = encoded[node_type]  # [N, T, H]
                T = h.shape[1]
                if T < self._PRED_MIN_SEQ_LEN:
                    # 序列过短，无法切分 (需 ≥ _PRED_MIN_SEQ_LEN 才能得到非空 context 和 future)
                    continue
                T_ctx = int(T * self._PRED_CONTEXT_RATIO)
                context = h[:, :T_ctx, :]           # [N, T_ctx, H]
                future_target = h[:, T_ctx:, :]     # [N, T_fut, H]

                # EnhancedMultiStepPredictor 期望输入 [batch, seq_len, H]；
                # 以 N 节点为 batch 维度（与 forward() 一致）
                # 返回 (pred_windows[num_windows, N, pred_steps, H], targets, unc)
                pred_windows, _, _ = self.predictor(context, return_uncertainty=False)
                pred_mean = pred_windows.mean(dim=0)  # [N, pred_steps, H]

                aligned_steps = min(pred_mean.shape[1], future_target.shape[1])
                if aligned_steps > 0:
                    if self.loss_type == 'huber':
                        pred_loss = F.huber_loss(
                            pred_mean[:, :aligned_steps, :],
                            future_target[:, :aligned_steps, :],
                            delta=1.0,
                        )
                    else:
                        pred_loss = F.mse_loss(
                            pred_mean[:, :aligned_steps, :],
                            future_target[:, :aligned_steps, :],
                        )
                    losses[f'pred_{node_type}'] = pred_loss
        
        return losses


class GraphNativeTrainer:
    """
    Complete training system for graph-native brain model.
    
    Integrates:
    - Graph-native model
    - Adaptive loss balancing
    - EEG channel enhancement
    - Advanced prediction
    
    This is a STANDALONE system, not dependent on old trainer.
    """
    
    def __init__(
        self,
        model: GraphNativeBrainModel,
        node_types: List[str],
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        use_adaptive_loss: bool = True,
        use_eeg_enhancement: bool = True,
        use_amp: bool = True,
        use_gradient_checkpointing: bool = False,
        use_scheduler: bool = True,
        scheduler_type: str = 'cosine',
        use_torch_compile: bool = True,
        compile_mode: str = 'reduce-overhead',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        optimization_config: Optional[Dict] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: GraphNativeBrainModel
            node_types: Node types in data
            learning_rate: Learning rate
            weight_decay: Weight decay
            use_adaptive_loss: Use adaptive loss balancing
            use_eeg_enhancement: Use EEG channel enhancement
            use_amp: Use automatic mixed precision (AMP) for 2-3x speedup
            use_gradient_checkpointing: Use gradient checkpointing to save memory
            use_scheduler: Use learning rate scheduling (10-20% faster convergence)
            scheduler_type: Type of scheduler ('cosine', 'onecycle', 'plateau')
            use_torch_compile: Use torch.compile() for 20-40% speedup (PyTorch 2.0+)
            compile_mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
            device: Device to train on
            optimization_config: Optional dict from config['v5_optimization'].
                Contains sub-dicts 'adaptive_loss' and 'eeg_enhancement' with
                fine-grained hyperparameters.  Defaults used when None.
        """
        self._optimization_config = optimization_config or {}
        self.model = model.to(device)
        self.device = device
        self.node_types = node_types
        
        # Verify CUDA availability
        if device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning(f"CUDA requested but not available. Falling back to CPU.")
            self.device = 'cpu'
            self.model = self.model.to('cpu')
        
        # torch.compile() for PyTorch 2.0+ (20-40% speedup)
        # NOTE: torch.compile() uses the 'inductor' backend which requires Triton.
        # Triton is unavailable on Windows and some other platforms. The compilation
        # is *lazy* – torch.compile() itself doesn't raise; the crash happens at the
        # first forward pass. We therefore probe Triton availability upfront and skip
        # compilation gracefully rather than relying on the try/except below.
        if use_torch_compile and hasattr(torch, 'compile'):
            _triton_ok = False
            try:
                import triton  # noqa: F401
                _triton_ok = True
            except ImportError:
                pass

            if not _triton_ok:
                logger.warning(
                    "torch.compile() skipped: Triton is not installed or not supported "
                    "on this platform (e.g. Windows). Running in eager mode."
                )
            else:
                logger.info(f"Enabling torch.compile() with mode={compile_mode}")
                try:
                    self.model = torch.compile(
                        self.model,
                        mode=compile_mode,
                        fullgraph=False  # Allow graph breaks for flexibility
                    )
                    logger.info("torch.compile() enabled successfully")
                except Exception as e:
                    logger.warning(f"torch.compile() failed, continuing without it: {e}")
        elif use_torch_compile:
            logger.warning("torch.compile() requested but not available (requires PyTorch >= 2.0)")
        
        # Mixed precision training
        self.use_amp = use_amp and device != 'cpu' and AMP_AVAILABLE
        if self.use_amp:
            # Use new API if available (torch.amp.GradScaler), otherwise old API (torch.cuda.amp.GradScaler)
            if USE_NEW_AMP_API:
                # New API expects device type as string (e.g., 'cuda', not 'cuda:0')
                # Extract device type: handle both string and torch.device inputs
                device_type = getattr(self.device, 'type', str(self.device).split(':')[0])
                self.device_type = device_type  # Store for use with autocast()
                self.scaler = GradScaler(device=device_type)
            else:
                # Old API doesn't take device parameter
                self.device_type = 'cuda'  # Old API only supports CUDA
                self.scaler = GradScaler()
            logger.info("Mixed precision training (AMP) enabled")
        elif use_amp and not AMP_AVAILABLE:
            logger.warning("AMP requested but not available. Training without mixed precision.")
        
        # Gradient checkpointing
        # NOTE: checkpointing is applied inside SpatialTemporalGraphConv's
        # temporal loop (via use_gradient_checkpointing on the model).
        # The trainer just logs its status here.
        if use_gradient_checkpointing:
            logger.info("Gradient checkpointing enabled (applied per ST-GCN timestep)")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        self.use_scheduler = use_scheduler
        self.scheduler = None
        if use_scheduler:
            if scheduler_type == 'cosine':
                # Linear warm-up for the first `warmup_epochs` epochs, then
                # CosineAnnealingWarmRestarts.  Without warm-up, the full LR
                # is applied from epoch 1, which often causes large gradient
                # steps on a freshly initialised model — particularly harmful
                # when training on small neuroimaging datasets (N < 100 samples).
                #
                # SequentialLR chains two schedulers: LinearLR ramps from
                # start_factor × lr up to lr over warmup_epochs steps, then
                # hands off to CosineAnnealingWarmRestarts.
                warmup_epochs = self._optimization_config.get(
                    'warmup_epochs',
                    3,  # safe minimum; v5_optimization.warmup_epochs in default.yaml is 5
                )
                warmup_sched = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.1,   # start at 10% of target LR
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                cosine_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=10,    # restart every 10 epochs
                    T_mult=2,  # double period after each restart
                    eta_min=learning_rate * 0.01,
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_sched, cosine_sched],
                    milestones=[warmup_epochs],
                )
                logger.info(
                    f"LR scheduler: Linear warmup ({warmup_epochs} epochs) "
                    f"→ CosineAnnealingWarmRestarts(T_0=10, T_mult=2)"
                )
            elif scheduler_type == 'onecycle':
                # OneCycle (will need total_steps, set in train_epoch)
                self.scheduler_type = 'onecycle'
                self.scheduler = None  # Will be created when we know total steps
                logger.info(f"Learning rate scheduler: OneCycle (will be initialized with total steps)")
            elif scheduler_type == 'plateau':
                # Reduce on plateau
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=0.5,
                    patience=5,
                    verbose=True
                )
                logger.info(f"Learning rate scheduler enabled: ReduceLROnPlateau")
            else:
                logger.warning(f"Unknown scheduler type: {scheduler_type}. No scheduler will be used.")
                self.use_scheduler = False
        
        # Adaptive loss balancing
        self.use_adaptive_loss = use_adaptive_loss
        if use_adaptive_loss:
            # Create task names from node types
            task_names = []
            for node_type in node_types:
                task_names.append(f'recon_{node_type}')
                if model.use_prediction:
                    task_names.append(f'pred_{node_type}')
            
            al_cfg = self._optimization_config.get('adaptive_loss', {})
            self.loss_balancer = AdaptiveLossBalancer(
                task_names=task_names,
                modality_names=node_types,
                alpha=al_cfg.get('alpha', 1.5),
                update_frequency=al_cfg.get('update_frequency', 10),
                learning_rate=al_cfg.get('learning_rate', 0.025),
                warmup_epochs=al_cfg.get('warmup_epochs', 5),
                modality_energy_ratios=al_cfg.get('modality_energy_ratios', {'eeg': 0.02, 'fmri': 1.0}),
            )
        
        # EEG enhancement
        # DESIGN NOTE: EEG graph data has shape [N_eeg, T, 1] — each graph node is
        # one electrode, with 1-dimensional feature (signal amplitude).
        # EnhancedEEGHandler was designed for [batch, time, channels] where 'channels'
        # are individual EEG electrodes.  The correct mapping is:
        #   graph format: [N_eeg, T, 1]  →  handler format: [1, T, N_eeg]
        # N_eeg is only known at data-loading time (varies per dataset/subject), so
        # we use lazy initialisation: the handler is created on the first training step.
        self.use_eeg_enhancement = use_eeg_enhancement and 'eeg' in node_types
        self.eeg_handler = None          # created lazily in train_step()
        self._eeg_handler_cfg = self._optimization_config.get('eeg_enhancement', {})
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }

    def _ensure_eeg_handler(self, n_eeg_channels: int) -> None:
        """Lazily initialise EnhancedEEGHandler once N_eeg is known from real data.

        Called from train_step() on the first forward pass so that the handler
        is built with the correct num_channels = N_eeg (number of electrodes),
        not the graph feature dimension (always 1).
        """
        if self.eeg_handler is not None:
            return  # already initialised
        cfg = self._eeg_handler_cfg
        try:
            self.eeg_handler = EnhancedEEGHandler(
                num_channels=n_eeg_channels,
                enable_monitoring=cfg.get('enable_monitoring', True),
                enable_dropout=cfg.get('enable_dropout', True),
                enable_attention=cfg.get('enable_attention', True),
                enable_regularization=cfg.get('enable_regularization', True),
                dropout_rate=cfg.get('dropout_rate', 0.1),
                attention_hidden_dim=cfg.get('attention_hidden_dim', 64),
                entropy_weight=cfg.get('entropy_weight', 0.01),
                diversity_weight=cfg.get('diversity_weight', 0.01),
                activity_weight=cfg.get('activity_weight', 0.01),
            ).to(self.device)
            logger.info(f"EEG handler initialised for {n_eeg_channels} channels.")
        except Exception as e:
            logger.warning(f"Failed to initialise EEG handler: {e}. Disabling EEG enhancement.")
            self.use_eeg_enhancement = False
    
    @staticmethod
    def _graph_to_handler_format(eeg_x: torch.Tensor) -> torch.Tensor:
        """Reshape EEG graph tensor for EnhancedEEGHandler.

        EnhancedEEGHandler expects ``[batch, time, channels]`` where *channels*
        are individual EEG electrodes.  Graph node features are stored as
        ``[N_eeg, T, 1]``.  This helper converts between the two formats.

        ``[N_eeg, T, 1]`` → ``[1, T, N_eeg]``
        """
        return eeg_x.squeeze(-1).permute(1, 0).unsqueeze(0)

    @staticmethod
    def _handler_to_graph_format(eeg_x: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`_graph_to_handler_format`.

        ``[1, T, N_eeg]`` → ``[N_eeg, T, 1]``
        """
        return eeg_x.squeeze(0).permute(1, 0).unsqueeze(-1)

    def train_step(self, data: HeteroData) -> Dict[str, float]:
        """
        Single training step with optional mixed precision.
        
        Args:
            data: Input HeteroData
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        data = data.to(self.device)
        
        # Apply EEG enhancement if enabled.
        # Graph data shape: [N_eeg, T, 1] (N_eeg nodes, each with a 1-dim feature).
        # EnhancedEEGHandler expects [batch, time, channels] where 'channels' are
        # individual EEG electrodes.  We therefore reshape:
        #   graph format: [N_eeg, T, 1]  →  handler format: [1, T, N_eeg]
        # After processing we reshape back.
        # IMPORTANT: Save original eeg_x and restore it in a finally block.
        # eeg_x_enhanced is part of the current computation graph (requires_grad=True).
        # If we leave data['eeg'].x pointing to it, the next epoch's forward pass
        # will build a new graph ON TOP of last epoch's freed graph, triggering:
        # "Trying to backward through the graph a second time".
        # Using try-finally guarantees restoration even if an exception is raised
        # anywhere inside the forward/backward path.
        eeg_info: dict = {}  # 初始为空；仅当 EEG handler 激活时（下方 if 块）被填充
        original_eeg_x = None
        if self.use_eeg_enhancement and 'eeg' in data.node_types:
            original_eeg_x = data['eeg'].x  # [N_eeg, T, 1]
            N_eeg = original_eeg_x.shape[0]
            # Lazy-initialise handler with true electrode count (N_eeg).
            # Previous approach used in_features=1 (graph feature dim), which
            # made all channel-specific processing trivially useless.
            self._ensure_eeg_handler(N_eeg)
            if self.use_eeg_enhancement and self.eeg_handler is not None:
                eeg_x_t, eeg_info = self.eeg_handler(
                    self._graph_to_handler_format(original_eeg_x), training=True
                )
                eeg_x_enhanced = self._handler_to_graph_format(eeg_x_t)
                data['eeg'].x = eeg_x_enhanced
        
        try:
            # Forward and backward pass with optional mixed precision
            if self.use_amp:
                # Use appropriate autocast context manager based on API version
                if USE_NEW_AMP_API:
                    # New API: torch.amp.autocast() requires device_type
                    amp_context = autocast(device_type=self.device_type)
                else:
                    # Old API: torch.cuda.amp.autocast() doesn't require device_type
                    amp_context = autocast()
                
                with amp_context:
                    # Forward pass.
                    # When use_prediction=True, retrieve encoded latent representations
                    # so compute_loss can train the predictor in latent space.
                    if self.model.use_prediction:
                        reconstructed, _, encoded = self.model(
                            data, return_prediction=False, return_encoded=True
                        )
                    else:
                        reconstructed, _ = self.model(data, return_prediction=False)
                        encoded = None
                    
                    # Compute losses (reconstruction + optional latent prediction)
                    losses = self.model.compute_loss(data, reconstructed, encoded=encoded)
                    
                    # Adaptive loss balancing
                    if self.use_adaptive_loss:
                        total_loss, weights = self.loss_balancer(losses)
                    else:
                        total_loss = sum(losses.values())
                    
                    # ── EEG 防零崩塌正则化 ───────────────────────────────
                    # eeg_handler 计算的熵+多样性+活动损失之前从未加入总损失
                    # （eeg_info 被静默丢弃）。此处补全，确保其梯度信号生效。
                    # 注：权重已在 AntiCollapseRegularizer 初始化时配置
                    # (entropy_weight, diversity_weight, activity_weight)，默认 0.01。
                    eeg_reg = eeg_info.get('regularization_loss')
                    if eeg_reg is not None:
                        total_loss = total_loss + eeg_reg
                        losses['eeg_reg'] = eeg_reg
                
                # Backward pass with gradient scaling
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training without AMP
                if self.model.use_prediction:
                    reconstructed, _, encoded = self.model(
                        data, return_prediction=False, return_encoded=True
                    )
                else:
                    reconstructed, _ = self.model(data, return_prediction=False)
                    encoded = None
                
                # Compute losses (reconstruction + optional latent prediction)
                losses = self.model.compute_loss(data, reconstructed, encoded=encoded)
                
                # Adaptive loss balancing
                if self.use_adaptive_loss:
                    total_loss, weights = self.loss_balancer(losses)
                else:
                    total_loss = sum(losses.values())
                
                # EEG 防零崩塌正则化（同 AMP 路径）
                eeg_reg = eeg_info.get('regularization_loss')
                if eeg_reg is not None:
                    total_loss = total_loss + eeg_reg
                    losses['eeg_reg'] = eeg_reg
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Update loss balancer with detached scalar values.
            # backward() has already freed the computation graph by this point;
            # update_weights() uses .item() internally, so passing detached losses
            # makes the post-backward contract explicit and avoids any accidental
            # graph access that would raise "backward through the graph a second time".
            if self.use_adaptive_loss:
                detached_losses = {k: v.detach() for k, v in losses.items()}
                self.loss_balancer.update_weights(detached_losses, self.model)
            
            # Return loss values
            loss_dict = {k: v.item() for k, v in losses.items()}
            loss_dict['total'] = total_loss.item()
            
            return loss_dict
        
        finally:
            # Always restore original EEG data so subsequent epochs start from the
            # raw (detached) tensor rather than this step's enhanced (grad-bearing)
            # tensor — regardless of whether an exception was raised.
            if original_eeg_x is not None:
                data['eeg'].x = original_eeg_x
    
    def train_epoch(self, data_list: List[HeteroData], epoch: int = None, total_epochs: int = None) -> float:
        """
        Train for one epoch with learning rate scheduling and progress logging.
        
        Args:
            data_list: List of training data
            epoch: Current epoch number (for logging)
            total_epochs: Total number of epochs (for logging)
            
        Returns:
            Average loss for epoch
        """
        if len(data_list) == 0:
            raise ValueError("Cannot train on empty data_list")
        
        total_loss = 0.0
        num_batches = len(data_list)
        
        # Advance epoch counter in the adaptive loss balancer so that the warmup
        # period expires correctly and weight adaptation becomes active.
        if self.use_adaptive_loss:
            self.loss_balancer.set_epoch(epoch or 0)
        
        # Log start of epoch
        if epoch is not None:
            if epoch == 1:
                logger.info("🚀 开始训练... (首个epoch可能因模型编译而较慢)")
            elif epoch <= 3:
                logger.info(f"📊 Epoch {epoch}/{total_epochs or '?'} 训练中...")
        
        for i, data in enumerate(data_list):
            loss_dict = self.train_step(data)
            total_loss += loss_dict['total']
            
            # Log progress for longer training runs (every 10 batches or at 25%, 50%, 75%)
            if num_batches > 10 and i > 0 and (i % 10 == 0 or i == num_batches // 4 or i == num_batches // 2 or i == 3 * num_batches // 4):
                progress_pct = (i + 1) / num_batches * 100
                avg_loss_so_far = total_loss / (i + 1)
                logger.info(f"  进度: {i+1}/{num_batches} batches ({progress_pct:.0f}%) - 当前平均loss: {avg_loss_so_far:.4f}")
        
        avg_loss = total_loss / len(data_list)
        self.history['train_loss'].append(avg_loss)
        
        # Release fragmented GPU memory blocks once per epoch.
        # This frees reserved-but-unallocated blocks back to the allocator so
        # they can be reused, reducing fragmentation OOM across epochs.
        # Called per-epoch (not per-step) to avoid repeated sync overhead.
        device_type = getattr(self.device, 'type', str(self.device).split(':')[0])
        if device_type == 'cuda':
            torch.cuda.empty_cache()
        
        # Step scheduler (if not ReduceLROnPlateau)
        if self.use_scheduler and self.scheduler is not None:
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
        
        return avg_loss
    
    def step_scheduler_on_validation(self, val_loss: float):
        """
        Step scheduler based on validation loss (for ReduceLROnPlateau).
        
        Args:
            val_loss: Validation loss
        """
        if self.use_scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
    
    @torch.no_grad()
    def validate(self, data_list: List[HeteroData]) -> float:
        """
        Validation pass.
        
        Computes the same loss terms as training (reconstruction + prediction)
        so that early stopping is driven by the actual optimisation objective.
        Previously this only computed reconstruction loss, making val_loss
        artificially lower than train_loss regardless of overfitting.
        
        Args:
            data_list: List of validation data
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        for data in data_list:
            data = data.to(self.device)
            
            # Request encoded representations when prediction is enabled so that
            # compute_loss() can include the latent-space prediction loss —
            # the same terms that are optimised during training.
            if self.model.use_prediction:
                reconstructed, _, encoded = self.model(
                    data, return_prediction=False, return_encoded=True
                )
            else:
                reconstructed, _ = self.model(data, return_prediction=False)
                encoded = None
            
            losses = self.model.compute_loss(data, reconstructed, encoded=encoded)
            total_loss += sum(losses.values()).item()
        
        avg_loss = total_loss / len(data_list)
        self.history['val_loss'].append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, path: Path, epoch: int):
        """Save training checkpoint with atomic write for safety."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }
        
        if self.use_adaptive_loss:
            checkpoint['loss_balancer_state'] = self.loss_balancer.state_dict()
        
        # Atomic save: write to temp file then rename
        temp_path = path.parent / f"{path.name}.tmp"
        
        try:
            torch.save(checkpoint, temp_path)
            # Atomic rename (on most filesystems)
            temp_path.replace(path)
            logger.info(f"Saved checkpoint to {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temp file: {cleanup_error}")
            raise RuntimeError(f"Checkpoint save failed: {e}")
    
    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        if self.use_adaptive_loss and 'loss_balancer_state' in checkpoint:
            self.loss_balancer.load_state_dict(checkpoint['loss_balancer_state'])
        
        logger.info(f"Loaded checkpoint from {path}")
        
        return checkpoint['epoch']
