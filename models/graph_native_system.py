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
import random
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

from .graph_native_mapper import GraphNativeBrainMapper
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


class GraphPredictionPropagator(nn.Module):
    """
    System-level graph propagation for predictions.

    After the per-node temporal predictor generates initial predictions
    ``{node_type: [N, pred_steps, H]}``, this module propagates them
    through the brain's connectivity graph so that a change in one brain
    region's predicted activity influences all its connected neighbours.

    Neuroscientific motivation
    --------------------------
    The brain operates as a **coupled dynamical system**.  Stimulating
    region A (or altering its activity) does not affect A in isolation —
    the perturbation propagates via white-matter tracts (structural
    connectivity) and correlated activity (functional connectivity) to
    regions B, C, D, …  Without this module the predictor treats every
    node independently, so stimulating a single ROI only alters that
    ROI's trajectory.  With this module the prediction becomes a
    system-level forecast: changed activity at A ripples through the
    graph to its neighbours in proportion to their edge weights.

    Architecture
    ------------
    ``num_prop_layers`` rounds of ST-GCN message passing applied to the
    prediction tensor ``[N, pred_steps, H]``, treating ``pred_steps`` as
    the temporal dimension (analogous to ``T`` in the encoder).  The
    same edge connectivity used during encoding is reused here, so the
    propagator is consistent with the learned representations.
    Residual connections + LayerNorm keep gradients healthy.
    """

    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        hidden_channels: int,
        num_prop_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            node_types: Node types present in the graph.
            edge_types: Edge types (same list as the encoder).
            hidden_channels: Feature dimension H (must match predictor output).
            num_prop_layers: Number of graph-propagation rounds.  2 is
                sufficient to reach 2-hop neighbours (e.g. A→B→C), which
                covers the typical cortical relay distance.
            dropout: Dropout rate.
        """
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types

        # Stack of ST-GCN layers (temporal_kernel_size=1 ≡ 1×1-conv so
        # consecutive prediction steps are NOT mixed — each step is
        # propagated independently, preserving temporal structure).
        self.prop_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(num_prop_layers):
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict['__'.join(edge_type)] = SpatialTemporalGraphConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    temporal_kernel_size=1,
                    use_attention=True,
                    use_spectral_norm=True,
                    dropout=dropout,
                )
            self.prop_layers.append(nn.ModuleDict(conv_dict))
            self.layer_norms.append(nn.ModuleDict({
                nt: nn.LayerNorm(hidden_channels)
                for nt in node_types
            }))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        pred_dict: Dict[str, torch.Tensor],
        data: HeteroData,
    ) -> Dict[str, torch.Tensor]:
        """
        Propagate predictions through graph connectivity.

        Args:
            pred_dict: Initial per-node predictions
                ``{node_type: Tensor[N, pred_steps, H]}``.
            data: HeteroData providing ``edge_index`` (and optionally
                ``edge_attr``) for each edge type.  The node features are
                NOT read from here — only the topology is used.

        Returns:
            Propagated predictions ``{node_type: Tensor[N, pred_steps, H]}``.
            Nodes with no inbound edges in a given layer pass through
            unchanged (same as the encoder residual convention).
        """
        x_dict = {k: v for k, v in pred_dict.items()}

        # Build edge index / attr dicts once (same across layers)
        edge_index_dict: Dict[Tuple, torch.Tensor] = {}
        edge_attr_dict: Dict[Tuple, torch.Tensor] = {}
        for edge_type in self.edge_types:
            if edge_type in data.edge_types:
                edge_index_dict[edge_type] = data[edge_type].edge_index
                if hasattr(data[edge_type], 'edge_attr'):
                    edge_attr_dict[edge_type] = data[edge_type].edge_attr

        for conv_dict, layer_norm_dict in zip(self.prop_layers, self.layer_norms):
            x_dict_new: Dict[str, torch.Tensor] = {}

            for node_type, x in x_dict.items():
                messages = []

                for edge_type in self.edge_types:
                    src, _, dst = edge_type
                    if dst != node_type:
                        continue
                    if edge_type not in edge_index_dict:
                        continue
                    if src not in x_dict:
                        continue

                    x_src = x_dict[src]
                    edge_index = edge_index_dict[edge_type]
                    edge_attr = edge_attr_dict.get(edge_type, None)

                    conv = conv_dict['__'.join(edge_type)]
                    N_src, N_dst = x_src.shape[0], x.shape[0]
                    msg = conv(x_src, edge_index, edge_attr, size=(N_src, N_dst))

                    # Align pred_steps dimension if source and destination
                    # modalities have different prediction lengths.
                    T_dst = x.shape[1]
                    if msg.shape[1] != T_dst:
                        msg = F.interpolate(
                            msg.permute(0, 2, 1),   # [N, H, T_src]
                            size=T_dst,
                            mode='linear',
                            align_corners=False,
                        ).permute(0, 2, 1)           # [N, T_dst, H]

                    messages.append(msg)

                # Residual aggregation — same convention as the encoder:
                # average incoming messages and add as a residual.
                if messages:
                    x_new = x + self.dropout(sum(messages) / len(messages))
                else:
                    x_new = x  # passthrough when no inbound edges

                # Per-timestep LayerNorm
                N, T, H = x_new.shape
                x_new = layer_norm_dict[node_type](x_new.view(N * T, H)).view(N, T, H)
                x_dict_new[node_type] = x_new

            x_dict = x_dict_new

        return x_dict


class GraphNativeBrainModel(nn.Module):
    """
    Complete graph-native brain model.

    End-to-end architecture:
    1. Mapper: Build graph from brain data
    2. Encoder: Spatial-temporal encoding on graph
    3. Predictor: Per-node future prediction (temporal)
    4. Propagator: System-level graph propagation of predictions
    5. Decoder: Reconstruct signals

    NO sequence conversions - pure graph operations throughout.

    System-level prediction
    -----------------------
    Step 3 runs ``EnhancedMultiStepPredictor`` independently on each
    node's latent time series.  Step 4 (``GraphPredictionPropagator``)
    then propagates those predictions through the brain's connectivity
    graph.  This means stimulating a single brain region affects all
    its neighbours in proportion to their edge weights, producing a
    scientifically meaningful whole-brain forecast rather than an
    isolated single-region forecast.
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
        use_dynamic_graph: bool = False,
        k_dynamic_neighbors: int = 10,
        num_subjects: int = 0,
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
            use_dynamic_graph: Enable self-iterating graph structure learning
                (DynamicGraphConstructor per ST-GCN layer, intra-modal edges only).
            k_dynamic_neighbors: k-nearest neighbours for the dynamic adjacency.
            num_subjects: Total number of subjects in the dataset.
                > 0 creates nn.Embedding(num_subjects, hidden_channels) for
                per-subject personalization (AGENTS.md §九 Gap 2).
                Each subject learns a unique latent offset added to all node
                features after input projection, enabling the model to capture
                individual brain differences without a separate model per subject.
                At inference time, fine-tune only the subject embedding (frozen
                encoder) for few-shot personalization.
                0 = disabled (default, backward-compatible).
        """
        super().__init__()
        
        self.node_types = node_types
        self.hidden_channels = hidden_channels
        self.use_prediction = use_prediction
        self.loss_type = loss_type
        self.num_subjects = num_subjects

        # 被试特异性嵌入 (AGENTS.md §九 Gap 2)
        # num_subjects > 0: each subject gets a learnable [H] offset added to
        # all node features after input projection, capturing individual differences.
        # Initialized with small Gaussian noise (std=0.02) to avoid disrupting
        # the shared pre-training signal in early epochs.
        if num_subjects > 0:
            self.subject_embed = nn.Embedding(num_subjects, hidden_channels)
            nn.init.normal_(self.subject_embed.weight, std=0.02)
        
        # Encoder: Graph-native spatial-temporal encoding
        self.encoder = GraphNativeEncoder(
            node_types=node_types,
            edge_types=edge_types,
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
            num_layers=num_encoder_layers,
            use_gradient_checkpointing=use_gradient_checkpointing,
            dropout=dropout,
            use_dynamic_graph=use_dynamic_graph,
            k_dynamic_neighbors=k_dynamic_neighbors,
        )
        
        # Decoder: Reconstruct temporal signals
        self.decoder = GraphNativeDecoder(
            node_types=node_types,
            hidden_channels=hidden_channels,
            out_channels_dict=in_channels_dict,
            num_layers=num_decoder_layers,
        )
        
        # Predictor: per-node temporal future prediction (optional)
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

            # Prediction propagator: system-level graph diffusion of predictions.
            # After EnhancedMultiStepPredictor generates independent per-node
            # trajectories, GraphPredictionPropagator runs num_prop_layers rounds
            # of graph message-passing so that stimulating one brain region
            # propagates its predicted activity to connected regions.
            # num_prop_layers=2 lets signals travel ≥2 hops (A→B→C), covering
            # the typical cortical relay distance.
            self.prediction_propagator = GraphPredictionPropagator(
                node_types=node_types,
                edge_types=edge_types,
                hidden_channels=hidden_channels,
                num_prop_layers=pred_cfg.get('num_prop_layers', 2),
                dropout=dropout,
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
        # Subject embedding (AGENTS.md §九 Gap 2):
        # If num_subjects > 0 and data carries a subject_idx scalar, look up the
        # per-subject [H] offset and pass it to the encoder.  The encoder adds it
        # to all node features after input projection, capturing individual brain
        # differences while sharing the rest of the model weights.
        subject_embed = None
        if self.num_subjects > 0 and hasattr(data, 'subject_idx') and data.subject_idx is not None:
            s_idx = data.subject_idx
            if not isinstance(s_idx, torch.Tensor):
                s_idx = torch.tensor(s_idx, dtype=torch.long)
            # Validate range; out-of-range usually means a stale cache was loaded
            # after num_subjects changed — warn instead of silently remapping.
            if s_idx.item() < 0 or s_idx.item() >= self.num_subjects:
                logger.warning(
                    f"subject_idx={s_idx.item()} out of range [0, {self.num_subjects-1}]. "
                    f"This likely means a cached graph was built with a different "
                    f"num_subjects.  Clearing the graph cache and re-running will fix this. "
                    f"Falling back to subject 0 for this sample."
                )
                s_idx = s_idx.clamp(0, self.num_subjects - 1)
            subject_embed = self.subject_embed(s_idx)  # [H]
        encoded_data = self.encoder(data, subject_embed=subject_embed)
        
        # 2. Decode: Reconstruct signals
        reconstructed = self.decoder(encoded_data)
        
        # 3. Predict: Future steps (optional)
        predictions = None
        if return_prediction and self.use_prediction:
            # Step 3a — Per-node temporal prediction.
            # EnhancedMultiStepPredictor treats N nodes as the batch dimension
            # (h: [N, T, H]).  Each node's future trajectory is predicted
            # independently from its own latent history.
            predictions = {}
            for node_type in self.node_types:
                if node_type in encoded_data.node_types:
                    h = encoded_data[node_type].x  # [N, T, H]
                    # Returns (pred_windows, targets, uncertainties):
                    #   pred_windows: [num_windows, N, prediction_steps, H]
                    pred_windows, _, _ = self.predictor(h, return_uncertainty=False)
                    # Average across sampled windows → [N, prediction_steps, H]
                    predictions[node_type] = pred_windows.mean(dim=0)

            # Step 3b — System-level graph propagation.
            # The per-node predictions above are independent.  The propagator
            # runs graph message-passing on {node_type: [N, pred_steps, H]} so
            # that stimulating one brain region influences its connected
            # neighbours — the brain as a coupled dynamical system.
            if predictions:
                predictions = self.prediction_propagator(predictions, data)

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
        
        # ── 潜空间自监督预测损失（系统级）──────────────────────────────────
        # 流程：
        #   1. 对每个模态，将编码器潜空间序列切分为
        #      context（前 2/3）→ per-node 预测 future（后 1/3）
        #   2. 对所有模态的初步预测应用 GraphPredictionPropagator，
        #      让预测在连通脑区间传播（系统级预测）
        #   3. 计算传播后的预测 vs. future_target 的损失
        #
        # 关键区别（与节点独立预测）：
        #   旧：每节点独立预测，刺激脑区 A 仅影响 A 的预测轨迹
        #   新：传播后，A 的预测变化通过图拓扑扩散至相邻脑区 B、C…
        #       等同于"大脑作为耦合动力学系统"的建模原则
        #
        # 注：编码器的 ST-GCN 跨模态边已使 fMRI 潜向量包含 EEG 信息，
        #     故系统级预测损失隐式覆盖了跨模态预测目标。
        if encoded is not None and self.use_prediction:
            # Step 1: Per-node temporal predictions for all modalities
            pred_means: Dict[str, torch.Tensor] = {}
            future_targets: Dict[str, torch.Tensor] = {}

            for node_type in self.node_types:
                if node_type not in encoded:
                    continue
                h = encoded[node_type]  # [N, T, H]
                T = h.shape[1]
                if T < self._PRED_MIN_SEQ_LEN:
                    # 序列过短，跳过（需 ≥ _PRED_MIN_SEQ_LEN 才能切分 context/future）
                    continue
                T_ctx = int(T * self._PRED_CONTEXT_RATIO)
                context = h[:, :T_ctx, :]       # [N, T_ctx, H]
                future_target = h[:, T_ctx:, :] # [N, T_fut, H]

                # EnhancedMultiStepPredictor: nodes treated as batch dim
                # Returns (pred_windows[W, N, pred_steps, H], targets, unc)
                pred_windows, _, _ = self.predictor(context, return_uncertainty=False)
                pred_means[node_type] = pred_windows.mean(dim=0)  # [N, pred_steps, H]
                future_targets[node_type] = future_target

            # Step 2: System-level graph propagation of predictions.
            # Allows the predicted activity change at one brain region to
            # propagate to its neighbours, producing a whole-brain forecast.
            if pred_means:
                pred_means = self.prediction_propagator(pred_means, data)

            # Step 3: Prediction loss per modality
            for node_type, pred_mean in pred_means.items():
                future_target = future_targets[node_type]
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
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        augmentation_config: Optional[Dict] = None,
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
            max_grad_norm: Max gradient norm for clipping (config['training']['max_grad_norm']).
                Hardcoding this to 1.0 previously caused the config value to be silently ignored.
            gradient_accumulation_steps: Accumulate gradients over this many steps before
                calling optimizer.step().  Default 1 = standard single-step update (backward
                compatible).  Set to 4 for an effective batch size of 4 with batch_size=1
                data, stabilising gradient estimates on small datasets without extra memory.
                The loss is scaled by 1/gradient_accumulation_steps before backward() to
                keep gradient magnitudes consistent regardless of the accumulation count.
            augmentation_config: Optional dict from config['training']['augmentation'].
                Supported keys:
                  enabled (bool): master switch (default False — backward compatible).
                  noise_std (float): std of Gaussian noise added to node features
                      (default 0.01; relative to z-scored signals so 1% amplitude).
                  scale_range ([min, max]): random amplitude scaling per sample
                      (default [0.9, 1.1]; None to disable).
        """
        self._optimization_config = optimization_config or {}
        self.model = model.to(device)
        self.device = device
        self.node_types = node_types
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))

        # Temporal augmentation config (applied only during training, not validation)
        _aug = augmentation_config or {}
        self._aug_enabled    = bool(_aug.get('enabled', False))
        self._aug_noise_std  = float(_aug.get('noise_std', 0.01))
        _sr = _aug.get('scale_range', [0.9, 1.1])
        self._aug_scale_min  = float(_sr[0]) if _sr else 1.0
        self._aug_scale_max  = float(_sr[1]) if _sr else 1.0
        if self._aug_enabled:
            logger.info(
                f"时序数据增强已启用: "
                f"noise_std={self._aug_noise_std}, "
                f"scale_range=[{self._aug_scale_min}, {self._aug_scale_max}]"
            )
        
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
        self._eeg_n_channels: Optional[int] = None   # recorded in _ensure_eeg_handler; used to detect channel-count mismatch across subjects
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
            self._eeg_n_channels = n_eeg_channels
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

    def train_step(
        self,
        data: HeteroData,
        do_zero_grad: bool = True,
        do_optimizer_step: bool = True,
        loss_scale: float = 1.0,
    ) -> Dict[str, float]:
        """
        Single training step with optional mixed precision.
        
        Args:
            data: Input HeteroData
            do_zero_grad: Whether to zero gradients before this step.
                Set False when accumulating gradients across multiple steps.
            do_optimizer_step: Whether to call optimizer.step() after backward.
                Set False for all but the last step in gradient accumulation.
            loss_scale: Multiply total_loss by this factor before backward().
                Use 1/gradient_accumulation_steps to normalise gradient magnitude
                when accumulating, keeping effective gradient scale constant
                regardless of accumulation count.
            
        Returns:
            Dictionary of loss values
        """
        self.model.train()
        if do_zero_grad:
            self.optimizer.zero_grad()
        
        # Move data to device
        data = data.to(self.device)

        # ── 时序数据增强（仅训练模式）────────────────────────────────────
        # 在每个训练步随机向节点特征添加小量高斯噪声和/或随机幅度缩放，
        # 提升模型对信号噪声和个体差异的鲁棒性（类似神经影像信号增强文献中
        # 的 jitter 和 scaling 方法）。验证时不增强，保证评估一致性。
        # 实现为纯 in-place-free 加法/乘法，不影响 autograd 图。
        if self._aug_enabled:
            for _nt in data.node_types:
                if hasattr(data[_nt], 'x') and data[_nt].x is not None:
                    _x = data[_nt].x
                    if self._aug_noise_std > 0:
                        _x = _x + torch.randn_like(_x) * self._aug_noise_std
                    if self._aug_scale_min != 1.0 or self._aug_scale_max != 1.0:
                        _scale = (
                            torch.empty(1, device=_x.device, dtype=_x.dtype)
                            .uniform_(self._aug_scale_min, self._aug_scale_max)
                        )
                        _x = _x * _scale
                    data[_nt].x = _x
        
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
            N_eeg = data['eeg'].x.shape[0]
            # Guard: if the handler was already initialised for a different channel
            # count (e.g. this subject has 64 channels but another had 63), skip
            # enhancement rather than feeding a wrong-shape tensor.  This silently
            # fails in the original code and produces garbage gradients or shape
            # errors that are hard to diagnose.
            # original_eeg_x is only set here (when we're about to modify), keeping
            # the finally-block restoration logic clean: non-None ↔ modified.
            if self.eeg_handler is not None and self._eeg_n_channels != N_eeg:
                logger.debug(
                    f"EEG channel count mismatch: handler built for "
                    f"{self._eeg_n_channels} channels, this sample has {N_eeg}. "
                    f"Skipping EEG enhancement for this sample."
                )
                # No modification made — original_eeg_x stays None, nothing to restore
            else:
                # Lazy-initialise handler with true electrode count (N_eeg).
                # Previous approach used in_features=1 (graph feature dim), which
                # made all channel-specific processing trivially useless.
                self._ensure_eeg_handler(N_eeg)
                if self.use_eeg_enhancement and self.eeg_handler is not None:
                    original_eeg_x = data['eeg'].x  # save before modifying
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
                
                # Backward pass with gradient scaling.
                # Apply loss_scale (= 1/gradient_accumulation_steps) to normalise
                # gradient magnitude when accumulating across multiple steps.
                self.scaler.scale(total_loss * loss_scale).backward()
                if do_optimizer_step:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
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
                
                # Backward pass.
                # Apply loss_scale (= 1/gradient_accumulation_steps) to normalise
                # gradient magnitude when accumulating across multiple steps.
                (total_loss * loss_scale).backward()
                if do_optimizer_step:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                    self.optimizer.step()
            
            # Update loss balancer with detached scalar values.
            # backward() has already freed the computation graph by this point;
            # update_weights() uses .item() internally, so passing detached losses
            # makes the post-backward contract explicit and avoids any accidental
            # graph access that would raise "backward through the graph a second time".
            if self.use_adaptive_loss:
                detached_losses = {k: v.detach() for k, v in losses.items()}
                self.loss_balancer.update_weights(detached_losses)
            
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
        
        # ── 逐 epoch 打乱训练样本 ────────────────────────────────────────────
        # 使用 epoch 作为随机种子：每个 epoch 的顺序不同（保证 SGD 随机性），
        # 但相同 epoch 编号时顺序可复现（方便调试）。
        #
        # 为什么必须打乱：
        # 1. `train_model` 在训练开始时只打乱一次；之后每个 epoch 都使用相同顺序。
        # 2. 不打乱时，排在列表末尾的被试/任务每个 epoch 总是得到最后一次权重更新，
        #    模型会对它们产生隐式偏好（optimizer 的动量使最后几步梯度影响最大）。
        # 3. 窗口模式下，同一 run 的 11 个窗口若连续处理，局部 loss 景观过于平滑，
        #    学习曲线呈"锯齿"型而非平稳下降。
        epoch_data = data_list.copy()   # 浅拷贝：不修改原始列表，仅改变本 epoch 的遍历顺序
        random.Random(epoch or 0).shuffle(epoch_data)
        total_loss = 0.0
        num_batches = len(epoch_data)
        ga = self.gradient_accumulation_steps  # shorthand
        
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
        
        for i, data in enumerate(epoch_data):
            # ── 梯度累积控制 ────────────────────────────────────────────────────
            # 每 ga 步执行一次 optimizer.step()，等效 batch size = ga × 1。
            # loss 除以 ga 以保持梯度期望不随 ga 变化（梯度期望 = Σ∇L_i / ga）。
            # do_zero_grad=True 仅在每组累积的第一步清零，避免丢失已累积梯度。
            is_accum_boundary = (i + 1) % ga == 0 or i == num_batches - 1
            loss_dict = self.train_step(
                data,
                do_zero_grad=(i % ga == 0),
                do_optimizer_step=is_accum_boundary,
                loss_scale=1.0 / ga,
            )
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
    def validate(self, data_list: List[HeteroData]) -> Tuple[float, Dict[str, float]]:
        """
        Validation pass.

        Computes the same loss terms as training (reconstruction + prediction)
        so that early stopping is driven by the actual optimisation objective.

        Metrics computed
        ----------------
        1. Reconstruction R² (r2_<node_type>):
           How well the model reconstructs the input signal.
           R² = 1 − SS_res/SS_tot.  R² = 1 → perfect; R² = 0 → mean baseline;
           R² < 0 → worse than mean baseline (failure).

        2. Signal-space prediction R² (pred_r2_<node_type>):
           **Primary quality indicator** — how well the model predicts future
           brain signals from past signals.
           Procedure: encode full sequence → split context (first 2/3) / future
           (last 1/3) → run predictor on context latent → decode predicted
           latent back to signal space → compare against actual future signal.
           This is the real "digital-twin" capability metric.

        Args:
            data_list: List of validation data

        Returns:
            Tuple of:
                avg_loss: Average validation loss (scalar)
                r2_dict: Per-modality metrics dict, keys:
                    'r2_<nt>'      — reconstruction R²
                    'pred_r2_<nt>' — signal-space prediction R² (★ primary)
        """
        self.model.eval()
        total_loss = 0.0
        # Accumulators for reconstruction R²
        ss_res: Dict[str, float] = {nt: 0.0 for nt in self.node_types}
        ss_tot: Dict[str, float] = {nt: 0.0 for nt in self.node_types}
        # Accumulators for signal-space prediction R²
        pred_ss_res: Dict[str, float] = {nt: 0.0 for nt in self.node_types}
        pred_ss_tot: Dict[str, float] = {nt: 0.0 for nt in self.node_types}

        with torch.no_grad():
            for data in data_list:
                data = data.to(self.device)

                # Forward pass — always request encoded representations so that:
                # a) compute_loss() can include the latent prediction loss, and
                # b) we can compute signal-space prediction R² below.
                if self.model.use_prediction:
                    reconstructed, _, encoded = self.model(
                        data, return_prediction=False, return_encoded=True
                    )
                else:
                    reconstructed, _ = self.model(data, return_prediction=False)
                    encoded = None

                losses = self.model.compute_loss(data, reconstructed, encoded=encoded)
                total_loss += sum(losses.values()).item()

                # ── Reconstruction R² per modality ──────────────────────────
                for node_type in self.node_types:
                    if node_type in data.node_types and node_type in reconstructed:
                        target = data[node_type].x       # [N, T, C]
                        recon  = reconstructed[node_type]  # [N, T', C]
                        T_min  = min(target.shape[1], recon.shape[1])
                        target = target[:, :T_min, :]
                        recon  = recon[:, :T_min, :]
                        if recon.shape[0] != target.shape[0]:
                            continue
                        target_mean = target.mean()
                        ss_res[node_type] += ((target - recon) ** 2).sum().item()
                        ss_tot[node_type] += ((target - target_mean) ** 2).sum().item()

                # ── Signal-space prediction R² per modality ─────────────────
                # The predictor operates in latent space; we decode its output
                # back to signal space so we can measure the genuinely useful
                # "given the past, how well can you predict the future signal?"
                # capability — the primary metric for a digital-twin brain model.
                if self.model.use_prediction and encoded is not None:
                    pred_latents: Dict[str, torch.Tensor] = {}
                    pred_T_ctx: Dict[str, int] = {}

                    for node_type in self.node_types:
                        if node_type not in encoded:
                            continue
                        h = encoded[node_type]  # [N, T, H]
                        T = h.shape[1]
                        if T < self.model._PRED_MIN_SEQ_LEN:
                            continue
                        T_ctx = int(T * self.model._PRED_CONTEXT_RATIO)
                        context = h[:, :T_ctx, :]
                        # Use same predictor as compute_loss()
                        pred_windows, _, _ = self.model.predictor(
                            context, return_uncertainty=False
                        )
                        pred_latents[node_type] = pred_windows.mean(dim=0)  # [N, pred_steps, H]
                        pred_T_ctx[node_type] = T_ctx

                    # System-level graph propagation of predictions
                    if pred_latents:
                        pred_latents = self.model.prediction_propagator(pred_latents, data)

                    if pred_latents:
                        # Decode predicted latents to signal space.
                        # Build a temporary HeteroData where each node type's
                        # .x is the predicted latent [N, pred_steps, H].
                        pred_enc = data.clone()
                        for nt, pred_lat in pred_latents.items():
                            pred_enc[nt].x = pred_lat
                        pred_signals = self.model.decoder(pred_enc)  # {nt: [N, pred_steps', C]}

                        for node_type, pred_sig in pred_signals.items():
                            if node_type not in data.node_types:
                                continue
                            T_ctx = pred_T_ctx.get(node_type)
                            if T_ctx is None:
                                continue
                            future_sig = data[node_type].x[:, T_ctx:, :]  # [N, T_fut, C]
                            n_steps = min(pred_sig.shape[1], future_sig.shape[1])
                            if n_steps < 1:
                                continue
                            if pred_sig.shape[0] != future_sig.shape[0]:
                                continue
                            pred_aligned   = pred_sig[:, :n_steps, :]
                            future_aligned = future_sig[:, :n_steps, :]
                            future_mean = future_aligned.mean()
                            pred_ss_res[node_type] += ((future_aligned - pred_aligned) ** 2).sum().item()
                            pred_ss_tot[node_type] += ((future_aligned - future_mean) ** 2).sum().item()

        avg_loss = total_loss / len(data_list)
        self.history['val_loss'].append(avg_loss)

        # ── Assemble R² dict ────────────────────────────────────────────────
        r2_dict: Dict[str, float] = {}

        # Reconstruction R²
        for node_type in self.node_types:
            tot = ss_tot[node_type]
            r2 = 1.0 - ss_res[node_type] / tot if tot > 1e-12 else 0.0
            r2_dict[f'r2_{node_type}'] = r2
            key = f'val_r2_{node_type}'
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(r2)

        # Signal-space prediction R² (★ primary metric)
        if self.model.use_prediction:
            for node_type in self.node_types:
                tot = pred_ss_tot[node_type]
                pred_r2 = 1.0 - pred_ss_res[node_type] / tot if tot > 1e-12 else 0.0
                r2_dict[f'pred_r2_{node_type}'] = pred_r2
                key = f'val_pred_r2_{node_type}'
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(pred_r2)

        return avg_loss, r2_dict
    
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
        
        # Save scheduler state so that LR schedule resumes correctly after loading.
        # Without this, load_checkpoint() would restart the LR from the initial
        # value + warmup phase, disrupting cosine-annealing restarts.
        if self.use_scheduler and self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
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
        # weights_only=False is required for loading HeteroData and custom objects
        # stored in checkpoints.  Omitting it triggers a FutureWarning in recent
        # PyTorch versions and will become an error in a future release.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        if self.use_adaptive_loss and 'loss_balancer_state' in checkpoint:
            self.loss_balancer.load_state_dict(checkpoint['loss_balancer_state'])
        
        # Restore scheduler state to continue LR annealing from where it left off.
        # Old checkpoints (saved before this fix) will not have 'scheduler_state_dict';
        # in that case the scheduler restarts from epoch 0 (pre-fix behaviour).
        if self.use_scheduler and self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        elif self.use_scheduler and self.scheduler is not None:
            logger.warning(
                "Checkpoint does not contain scheduler_state_dict "
                "(checkpoint was saved before V5.22). "
                "LR schedule will restart from epoch 0."
            )
        
        logger.info(f"Loaded checkpoint from {path}")
        
        return checkpoint['epoch']
