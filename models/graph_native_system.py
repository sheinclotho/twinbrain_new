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
            self.predictor = EnhancedMultiStepPredictor(
                input_dim=hidden_channels,
                hidden_dim=hidden_channels * 2,
                prediction_steps=prediction_steps,
                use_hierarchical=True,
                use_transformer=True,
            )
    
    def forward(
        self,
        data: HeteroData,
        return_prediction: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass with input validation.
        
        Args:
            data: Input HeteroData with temporal features
            return_prediction: Whether to return future predictions
            
        Returns:
            reconstructed: Reconstructed signals per modality
            predictions: Future predictions (if return_prediction=True)
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
        
        return reconstructed, predictions
    
    def compute_loss(
        self,
        data: HeteroData,
        reconstructed: Dict[str, torch.Tensor],
        predictions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            data: Original data
            reconstructed: Reconstructed signals
            predictions: Predicted future signals (currently unused in loss)
            
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
                
                # Choose loss function
                if self.loss_type == 'huber':
                    recon_loss = F.huber_loss(recon, target, delta=1.0)
                elif self.loss_type == 'smooth_l1':
                    recon_loss = F.smooth_l1_loss(recon, target)
                else:
                    recon_loss = F.mse_loss(recon, target)
                
                losses[f'recon_{node_type}'] = recon_loss
        
        # Prediction loss: pred is in latent space H while data[node_type].x is in
        # original space C.  Comparing them directly is undefined and produces
        # meaningless gradients.  A proper prediction loss requires encoding the
        # future window and comparing in latent space — implement as future work.
        # For now, the predictor is used for inference only.
        
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
        """
        self.model = model.to(device)
        self.device = device
        self.node_types = node_types
        
        # Verify CUDA availability
        if device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning(f"CUDA requested but not available. Falling back to CPU.")
            self.device = 'cpu'
            self.model = self.model.to('cpu')
        
        # torch.compile() for PyTorch 2.0+ (20-40% speedup)
        if use_torch_compile and hasattr(torch, 'compile'):
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
                # Cosine annealing with warm restarts
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=10,  # Restart every 10 epochs
                    T_mult=2,  # Double period after each restart
                    eta_min=learning_rate * 0.01
                )
                logger.info(f"Learning rate scheduler enabled: CosineAnnealingWarmRestarts")
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
            
            self.loss_balancer = AdaptiveLossBalancer(
                task_names=task_names,
                modality_names=node_types,
                modality_energy_ratios={'eeg': 0.02, 'fmri': 1.0},
            )
        
        # EEG enhancement
        self.use_eeg_enhancement = use_eeg_enhancement
        if use_eeg_enhancement and 'eeg' in node_types:
            try:
                # Get EEG channel count from model (with safety checks)
                if hasattr(model.encoder, 'input_proj') and 'eeg' in model.encoder.input_proj:
                    eeg_channels = model.encoder.input_proj['eeg'].in_features
                    self.eeg_handler = EnhancedEEGHandler(
                        num_channels=eeg_channels,
                        enable_monitoring=True,
                        enable_attention=True,
                        enable_regularization=True,
                    ).to(self.device)  # Move to device to prevent device mismatch errors
                else:
                    logger.warning("EEG enhancement requested but no EEG encoder found. Disabling.")
                    self.use_eeg_enhancement = False
            except (AttributeError, KeyError) as e:
                logger.warning(f"Failed to initialize EEG handler: {e}. Disabling EEG enhancement.")
                self.use_eeg_enhancement = False
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }
    
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
        
        # Apply EEG enhancement if enabled
        if self.use_eeg_enhancement and 'eeg' in data.node_types:
            eeg_x = data['eeg'].x
            eeg_x_enhanced, eeg_info = self.eeg_handler(eeg_x, training=True)
            data['eeg'].x = eeg_x_enhanced
        
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
                # Forward pass: skip prediction during training — the predictor
                # operates in latent space and has no training loss (see compute_loss).
                # This also saves the compute of running the full prediction head.
                reconstructed, _ = self.model(data, return_prediction=False)
                
                # Compute losses
                losses = self.model.compute_loss(data, reconstructed, None)
                
                # Adaptive loss balancing
                if self.use_adaptive_loss:
                    total_loss, weights = self.loss_balancer(losses)
                else:
                    total_loss = sum(losses.values())
            
            # Backward pass with gradient scaling
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training without AMP
            reconstructed, _ = self.model(data, return_prediction=False)
            
            # Compute losses
            losses = self.model.compute_loss(data, reconstructed, None)
            
            # Adaptive loss balancing
            if self.use_adaptive_loss:
                total_loss, weights = self.loss_balancer(losses)
            else:
                total_loss = sum(losses.values())
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        # Update loss balancer
        if self.use_adaptive_loss:
            self.loss_balancer.update_weights(losses, self.model)
        
        # Return loss values
        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict['total'] = total_loss.item()
        
        return loss_dict
    
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
        
        Args:
            data_list: List of validation data
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        for data in data_list:
            data = data.to(self.device)
            
            # Forward pass: skip prediction (compute_loss doesn't use it)
            reconstructed, _ = self.model(data, return_prediction=False)
            
            # Compute losses
            losses = self.model.compute_loss(data, reconstructed, None)
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
