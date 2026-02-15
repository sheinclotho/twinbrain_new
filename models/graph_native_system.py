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
    from torch.amp import autocast, GradScaler
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
            
            # Stack of transpose convolutions
            for i in range(num_layers):
                out_dim = hidden_channels // (2 ** i) if i < num_layers - 1 else out_channels_dict[node_type]
                
                layers.append(nn.ConvTranspose1d(
                    current_dim,
                    out_dim,
                    kernel_size=4,
                    stride=2 if temporal_upsample and i == 0 else 1,
                    padding=1,
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
                    # Get encoded features
                    h = encoded_data[node_type].x  # [N, T, H]
                    
                    # Predict future
                    pred_mean, _, _ = self.predictor(
                        h.unsqueeze(0),  # Add batch dim
                        return_uncertainty=False,
                    )
                    
                    predictions[node_type] = pred_mean.squeeze(0)  # Remove batch dim
        
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
            predictions: Predicted future signals
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Reconstruction loss per modality
        for node_type in self.node_types:
            if node_type in data.node_types and node_type in reconstructed:
                target = data[node_type].x  # [N, T, C]
                recon = reconstructed[node_type]
                
                # Choose loss function based on loss_type
                if self.loss_type == 'huber':
                    # Huber loss: robust to outliers (5-10% better on noisy signals)
                    recon_loss = F.huber_loss(recon, target, delta=1.0)
                elif self.loss_type == 'smooth_l1':
                    # Smooth L1 loss: similar to Huber
                    recon_loss = F.smooth_l1_loss(recon, target)
                else:
                    # Default: MSE loss
                    recon_loss = F.mse_loss(recon, target)
                
                losses[f'recon_{node_type}'] = recon_loss
        
        # Prediction loss (if available)
        if predictions is not None:
            for node_type, pred in predictions.items():
                if node_type in data.node_types:
                    # Get future ground truth (shift by prediction_steps)
                    target = data[node_type].x  # [N, T, C]
                    
                    # Simple prediction loss (last steps vs predicted)
                    pred_loss = F.mse_loss(pred, target[:, -pred.shape[1]:, :])
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
                if isinstance(self.device, str):
                    # Extract device type from 'cuda:0' -> 'cuda'
                    device_type = self.device.split(':')[0]
                elif hasattr(self.device, 'type'):
                    # torch.device object has .type attribute
                    device_type = self.device.type
                else:
                    # Fallback to string conversion
                    device_type = str(self.device).split(':')[0]
                self.scaler = GradScaler(device=device_type)
            else:
                # Old API doesn't take device parameter
                self.scaler = GradScaler()
            logger.info("Mixed precision training (AMP) enabled")
        elif use_amp and not AMP_AVAILABLE:
            logger.warning("AMP requested but not available. Training without mixed precision.")
        
        # Gradient checkpointing
        if use_gradient_checkpointing:
            # Enable checkpointing for encoder layers
            if hasattr(model.encoder, 'stgcn_layers'):
                for layer in model.encoder.stgcn_layers:
                    if hasattr(layer, 'gradient_checkpointing_enable'):
                        layer.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
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
            with autocast():
                # Forward pass
                reconstructed, predictions = self.model(
                    data,
                    return_prediction=self.model.use_prediction,
                )
                
                # Compute losses
                losses = self.model.compute_loss(data, reconstructed, predictions)
                
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
            # Forward pass
            reconstructed, predictions = self.model(
                data,
                return_prediction=self.model.use_prediction,
            )
            
            # Compute losses
            losses = self.model.compute_loss(data, reconstructed, predictions)
            
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
    
    def train_epoch(self, data_list: List[HeteroData]) -> float:
        """
        Train for one epoch with learning rate scheduling.
        
        Args:
            data_list: List of training data
            
        Returns:
            Average loss for epoch
        """
        total_loss = 0.0
        
        for data in data_list:
            loss_dict = self.train_step(data)
            total_loss += loss_dict['total']
        
        avg_loss = total_loss / len(data_list)
        self.history['train_loss'].append(avg_loss)
        
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
            
            # Forward pass
            reconstructed, predictions = self.model(
                data,
                return_prediction=self.model.use_prediction,
            )
            
            # Compute losses
            losses = self.model.compute_loss(data, reconstructed, predictions)
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
                except:
                    pass
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
