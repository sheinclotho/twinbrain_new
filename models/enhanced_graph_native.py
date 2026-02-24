"""
Enhanced Graph-Native Brain Model with Consciousness
====================================================

Integrates consciousness theories and predictive coding into the graph-native architecture:
1. Global Workspace Theory (information integration)
2. Integrated Information Theory (Φ computation)
3. Predictive Coding (prediction error minimization)
4. Cross-modal attention (EEG-fMRI fusion)
5. Hierarchical representations

This is an enhanced version that adds consciousness modeling on top of
the existing graph-native architecture.
"""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple
import logging

from .graph_native_system import GraphNativeBrainModel, GraphNativeTrainer
from .consciousness_module import ConsciousnessModule
from .advanced_attention import CrossModalAttention, SpatialTemporalAttention
from .predictive_coding import HierarchicalPredictiveCoding, compute_free_energy_loss

# AMP imports mirrored from graph_native_system
try:
    from torch.amp import autocast, GradScaler
    USE_NEW_AMP_API = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    USE_NEW_AMP_API = False

logger = logging.getLogger(__name__)


class ConsciousGraphNativeBrainModel(nn.Module):
    """
    Graph-Native Brain Model with Consciousness.
    
    Extends the base GraphNativeBrainModel with:
    1. Consciousness module (GWT + IIT)
    2. Cross-modal attention
    3. Predictive coding hierarchy
    4. Enhanced interpretability
    """
    
    def __init__(
        self,
        base_model: GraphNativeBrainModel,
        enable_consciousness: bool = True,
        enable_cross_modal_attention: bool = True,
        enable_predictive_coding: bool = True,
        consciousness_config: Optional[Dict] = None,
        attention_config: Optional[Dict] = None,
        predictive_coding_config: Optional[Dict] = None,
    ):
        """
        Initialize conscious graph-native brain model.
        
        Args:
            base_model: Base GraphNativeBrainModel
            enable_consciousness: Enable consciousness module
            enable_cross_modal_attention: Enable cross-modal attention
            enable_predictive_coding: Enable predictive coding
            consciousness_config: Config for consciousness module
            attention_config: Config for attention module
            predictive_coding_config: Config for predictive coding
        """
        super().__init__()
        
        self.base_model = base_model
        self.enable_consciousness = enable_consciousness
        self.enable_cross_modal_attention = enable_cross_modal_attention
        self.enable_predictive_coding = enable_predictive_coding
        
        # Get hidden dimension from base model
        hidden_channels = base_model.hidden_channels
        
        # Consciousness module (Global Workspace + IIT)
        if enable_consciousness:
            consciousness_config = consciousness_config or {}
            self.consciousness_module = ConsciousnessModule(
                hidden_channels=hidden_channels,
                num_heads=consciousness_config.get('num_heads', 8),
                workspace_dim=consciousness_config.get('workspace_dim', 256),
                num_workspace_slots=consciousness_config.get('num_workspace_slots', 16),
                num_partitions=consciousness_config.get('num_partitions', 4),
                num_consciousness_states=consciousness_config.get('num_states', 7),
                dropout=consciousness_config.get('dropout', 0.1),
            )
            logger.info("✓ Consciousness module initialized (GWT + IIT)")
        
        # Cross-modal attention (EEG ↔ fMRI)
        if enable_cross_modal_attention:
            attention_config = attention_config or {}
            self.cross_modal_attention = CrossModalAttention(
                eeg_channels=hidden_channels,
                fmri_channels=hidden_channels,
                hidden_dim=attention_config.get('hidden_dim', 256),
                num_heads=attention_config.get('num_heads', 8),
                dropout=attention_config.get('dropout', 0.1),
            )
            
            # Spatial-temporal attention for enhanced temporal modeling
            self.spatial_temporal_attention = SpatialTemporalAttention(
                channels=hidden_channels,
                num_heads=attention_config.get('num_heads', 8),
                dropout=attention_config.get('dropout', 0.1),
            )
            logger.info("✓ Cross-modal and spatial-temporal attention initialized")
        
        # Predictive coding hierarchy
        if enable_predictive_coding:
            predictive_coding_config = predictive_coding_config or {}
            
            # Determine input dimension (flatten temporal features)
            # Assuming features are [batch, num_nodes, time, channels]
            # We'll flatten to [batch, num_nodes * time * channels] for predictive coding
            
            # For now, use a simpler approach: apply predictive coding to aggregated features
            self.predictive_coding = HierarchicalPredictiveCoding(
                input_dim=hidden_channels,
                hidden_dims=predictive_coding_config.get('hidden_dims', [256, 512, 1024]),
                num_iterations=predictive_coding_config.get('num_iterations', 5),
                use_precision=predictive_coding_config.get('use_precision', True),
                learning_rate=predictive_coding_config.get('learning_rate', 0.1),
            )
            logger.info("✓ Predictive coding hierarchy initialized")
    
    # ── 与 GraphNativeBrainModel 的 API 兼容性属性 ──────────────
    # GraphNativeTrainer.train_step() 通过这些属性控制前向传播路径。
    # 必须代理到 base_model，否则父类 train_step() 中 AttributeError。

    @property
    def use_prediction(self) -> bool:
        return self.base_model.use_prediction

    @property
    def loss_type(self) -> str:
        return self.base_model.loss_type

    def compute_loss(self, data, reconstructed, predictions=None, encoded=None):
        """代理到 base_model.compute_loss（重建损失 + 潜空间预测损失）。"""
        return self.base_model.compute_loss(
            data, reconstructed, predictions=predictions, encoded=encoded
        )

    def forward(
        self,
        data: HeteroData,
        return_prediction: bool = False,
        return_encoded: bool = False,
        return_consciousness_metrics: bool = False,
    ) -> Tuple:
        """
        Forward pass with consciousness and predictive coding.

        API 与 GraphNativeBrainModel.forward() 完全兼容：
        - 默认 (return_encoded=False, return_consciousness_metrics=False):
            returns (reconstructed, predictions) — 2-tuple
        - return_encoded=True:
            returns (reconstructed, predictions, encoded_dict) — 3-tuple
        - return_consciousness_metrics=True:
            returns (reconstructed, predictions, info) — 3-tuple
        - return_encoded=True AND return_consciousness_metrics=True:
            returns (reconstructed, predictions, encoded_dict, info) — 4-tuple
        
        Args:
            data: Input heterogeneous graph
            return_prediction: Pass to base_model
            return_encoded: If True, 3rd element is the latent encoded dict
            return_consciousness_metrics: If True, appends info dict at the end
        
        Returns:
            See above; shape depends on flags.
        """
        # 1. Base model forward pass
        # Always retrieve encoded representations so that:
        #   a) cross-modal attention uses latent space (not signal-space recon)
        #   b) compute_loss can train the predictor in latent space
        base_fwd = self.base_model(
            data, return_prediction=return_prediction, return_encoded=True
        )
        reconstructions, predictions, encoded = base_fwd

        info = {}

        # 2. Cross-modal attention using ENCODED latent features (NOT reconstructions).
        # BUG in original: used reconstructions [N, T, 1] → CrossModalAttention received
        # wrong channel dimension (1 instead of hidden_channels).  Fixed: use encoded [N, T, H].
        if (
            self.enable_cross_modal_attention
            and 'eeg' in data.node_types
            and 'fmri' in data.node_types
        ):
            eeg_feat = encoded.get('eeg')    # [N_eeg, T, H]
            fmri_feat = encoded.get('fmri')  # [N_fmri, T, H]

            if eeg_feat is not None and fmri_feat is not None:
                # Aggregate temporal dim and add batch dim: [N, T, H] → [1, N, H]
                eeg_agg = eeg_feat.mean(dim=1).unsqueeze(0)    # [1, N_eeg, H]
                fmri_agg = fmri_feat.mean(dim=1).unsqueeze(0)  # [1, N_fmri, H]

                # SpatialTemporalAttention expects [batch, nodes, T, H];
                # we provide [1, N, 1, H] by unsqueezing the (missing) T dim.
                eeg_4d = eeg_agg.unsqueeze(2)    # [1, N_eeg, 1, H]
                fmri_4d = fmri_agg.unsqueeze(2)  # [1, N_fmri, 1, H]

                eeg_attended, eeg_st_info = self.spatial_temporal_attention(eeg_4d)
                fmri_attended, fmri_st_info = self.spatial_temporal_attention(fmri_4d)
                info['eeg_st_attention'] = eeg_st_info
                info['fmri_st_attention'] = fmri_st_info

                # Remove dummy T dim before cross-modal attention
                eeg_agg2 = eeg_attended.squeeze(2)   # [1, N_eeg, H]
                fmri_agg2 = fmri_attended.squeeze(2)  # [1, N_fmri, H]

                eeg_enhanced, fmri_enhanced, cm_info = self.cross_modal_attention(
                    eeg_features=eeg_agg2,
                    fmri_features=fmri_agg2,
                )
                info['cross_modal_attention'] = cm_info
                reconstructions['eeg_enhanced'] = eeg_enhanced
                reconstructions['fmri_enhanced'] = fmri_enhanced

        # 3. Consciousness module
        if self.enable_consciousness and return_consciousness_metrics:
            all_features = []
            for modality in ['eeg', 'fmri']:
                feat = encoded.get(modality)
                if feat is not None:  # [N, T, H]
                    feat_agg = feat.mean(dim=1).unsqueeze(0)  # [1, N, H]
                    all_features.append(feat_agg)

            if all_features:
                combined_features = torch.cat(all_features, dim=1)  # [1, total_N, H]

                if ('fmri', 'connects', 'fmri') in data.edge_types:
                    edge_index = data['fmri', 'connects', 'fmri'].edge_index
                else:
                    num_nodes = combined_features.shape[1]
                    edge_index = torch.combinations(
                        torch.arange(num_nodes), r=2
                    ).T
                    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
                    edge_index = edge_index.to(combined_features.device)

                conscious_features, consciousness_info = self.consciousness_module(
                    x=combined_features,
                    edge_index=edge_index,
                )
                info['consciousness'] = consciousness_info
                reconstructions['conscious_representation'] = conscious_features

        # 4. Predictive coding
        if self.enable_predictive_coding:
            for modality in ['eeg', 'fmri']:
                feat = encoded.get(modality)
                if feat is not None:  # [N, T, H]
                    feat_agg = feat.mean(dim=0).unsqueeze(0)  # [1, H] (global avg)
                    pc_state, pc_predictions, pc_info = self.predictive_coding(feat_agg)
                    info[f'{modality}_predictive_coding'] = pc_info
                    reconstructions[f'{modality}_pc_state'] = pc_state

        # Return format mirrors GraphNativeBrainModel.forward()
        if return_encoded and return_consciousness_metrics:
            return reconstructions, predictions, encoded, info
        elif return_encoded:
            return reconstructions, predictions, encoded
        elif return_consciousness_metrics:
            return reconstructions, predictions, info
        else:
            return reconstructions, predictions


class EnhancedGraphNativeTrainer(GraphNativeTrainer):
    """
    Enhanced trainer with consciousness and free energy losses.
    
    Extends the base trainer with:
    1. Free energy loss (predictive coding)
    2. Consciousness-aware loss weighting
    3. Additional metrics logging
    """
    
    def __init__(
        self,
        model: ConsciousGraphNativeBrainModel,
        node_types: List[str],
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        use_adaptive_loss: bool = True,
        use_eeg_enhancement: bool = True,
        consciousness_loss_weight: float = 0.1,
        predictive_coding_loss_weight: float = 0.1,
        **kwargs
    ):
        """
        Initialize enhanced trainer.
        
        Args:
            model: Conscious graph-native brain model
            node_types: List of node types
            learning_rate: Learning rate
            weight_decay: Weight decay
            use_adaptive_loss: Use adaptive loss balancing
            use_eeg_enhancement: Use EEG enhancement
            consciousness_loss_weight: Weight for consciousness loss
            predictive_coding_loss_weight: Weight for predictive coding loss
            **kwargs: Additional arguments for base trainer
        """
        # Initialize base trainer with the base model so that the EEG handler
        # can introspect model.encoder.input_proj['eeg'] correctly.
        super().__init__(
            model=model.base_model,
            node_types=node_types,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            use_adaptive_loss=use_adaptive_loss,
            use_eeg_enhancement=use_eeg_enhancement,
            **kwargs
        )
        
        # Replace model with the full enhanced model.
        self.model = model
        
        # BUG FIX: re-create optimizer with FULL model parameters.
        # super().__init__ created the optimizer with base_model.parameters() only.
        # Consciousness module, cross-modal attention, and predictive coding parameters
        # were therefore excluded from the optimizer — they received gradients but those
        # gradients were never applied (parameter update step was a no-op for them).
        # `learning_rate` and `weight_decay` are explicit parameters of this __init__,
        # so they are in scope here (not shadowed by **kwargs).
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Additional loss weights
        self.consciousness_loss_weight = consciousness_loss_weight
        self.predictive_coding_loss_weight = predictive_coding_loss_weight
        
        logger.info(
            f"Enhanced trainer initialized with consciousness_loss_weight={consciousness_loss_weight}, "
            f"predictive_coding_loss_weight={predictive_coding_loss_weight}"
        )
    
    def compute_additional_losses(
        self,
        info: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute additional losses from consciousness and predictive coding.
        
        Args:
            info: Info dict from model forward pass
        
        Returns:
            Dict of additional losses
        """
        additional_losses = {}
        
        # Consciousness loss: encourage high Φ (integrated information)
        if 'consciousness' in info:
            consciousness_info = info['consciousness']
            
            if 'phi' in consciousness_info:
                phi = consciousness_info['phi']
                # Encourage high Φ (negative loss)
                consciousness_loss = -phi.mean() * self.consciousness_loss_weight
                additional_losses['consciousness'] = consciousness_loss
        
        # Predictive coding loss: minimize free energy
        for key in info:
            if 'predictive_coding' in key:
                pc_info = info[key]
                if 'total_free_energy' in pc_info:
                    free_energy = pc_info['total_free_energy']
                    pc_loss = free_energy * self.predictive_coding_loss_weight
                    additional_losses[f'{key}_free_energy'] = pc_loss
        
        return additional_losses

    def train_step(self, data: HeteroData) -> Dict[str, float]:
        """
        Single training step for the enhanced model.

        Extends the base train_step by:
        1. Requesting consciousness metrics from forward()
        2. Adding consciousness and free energy losses to total_loss

        This override is necessary because:
        - ConsciousGraphNativeBrainModel.forward() can return a 4-tuple
          (recon, preds, encoded, info) when both return_encoded and
          return_consciousness_metrics are True.
        - compute_additional_losses() must be called WITHIN the same
          forward/backward pass — calling it after super().train_step()
          (which has already called backward()) would operate on a freed
          computation graph.
        """
        self.model.train()
        self.optimizer.zero_grad()
        data = data.to(self.device)

        eeg_info: dict = {}
        original_eeg_x = None
        if self.use_eeg_enhancement and 'eeg' in data.node_types:
            original_eeg_x = data['eeg'].x
            eeg_x_enhanced, eeg_info = self.eeg_handler(original_eeg_x, training=True)
            data['eeg'].x = eeg_x_enhanced

        try:
            # ── forward (inside AMP autocast when enabled) ──────────
            if self.use_amp:
                if USE_NEW_AMP_API:
                    _amp_ctx = autocast(device_type=self.device_type)
                else:
                    _amp_ctx = autocast()
            else:
                from contextlib import nullcontext
                _amp_ctx = nullcontext()

            with _amp_ctx:
                if self.model.use_prediction:
                    reconstructed, _predictions, encoded, info = self.model(
                        data,
                        return_prediction=False,
                        return_encoded=True,
                        return_consciousness_metrics=True,
                    )
                else:
                    reconstructed, _predictions, info = self.model(
                        data,
                        return_prediction=False,
                        return_consciousness_metrics=True,
                    )
                    encoded = None

                # ── losses ──────────────────────────────────────────────
                losses = self.model.compute_loss(data, reconstructed, encoded=encoded)

                if self.use_adaptive_loss:
                    total_loss, _ = self.loss_balancer(losses)
                else:
                    total_loss = sum(losses.values())

                # EEG anti-collapse regularization
                eeg_reg = eeg_info.get('regularization_loss')
                if eeg_reg is not None:
                    total_loss = total_loss + eeg_reg
                    losses['eeg_reg'] = eeg_reg

                # Consciousness + free-energy losses
                additional_losses = self.compute_additional_losses(info)
                for name, loss in additional_losses.items():
                    total_loss = total_loss + loss
                    losses[name] = loss

            # ── backward ────────────────────────────────────────────
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            if self.use_adaptive_loss:
                detached = {k: v.detach() for k, v in losses.items()}
                self.loss_balancer.update_weights(detached, self.model)

            loss_dict = {k: v.item() for k, v in losses.items()}
            loss_dict['total'] = total_loss.item()
            return loss_dict

        finally:
            if original_eeg_x is not None:
                data['eeg'].x = original_eeg_x


def create_enhanced_model(
    base_model_config: Dict,
    enable_consciousness: bool = True,
    enable_cross_modal_attention: bool = True,
    enable_predictive_coding: bool = True,
) -> ConsciousGraphNativeBrainModel:
    """
    Factory function to create enhanced model.
    
    Args:
        base_model_config: Config dict for base model
        enable_consciousness: Enable consciousness module
        enable_cross_modal_attention: Enable cross-modal attention
        enable_predictive_coding: Enable predictive coding
    
    Returns:
        Enhanced conscious graph-native brain model
    """
    # Create base model
    base_model = GraphNativeBrainModel(**base_model_config)
    
    # Wrap with consciousness and predictive coding
    enhanced_model = ConsciousGraphNativeBrainModel(
        base_model=base_model,
        enable_consciousness=enable_consciousness,
        enable_cross_modal_attention=enable_cross_modal_attention,
        enable_predictive_coding=enable_predictive_coding,
    )
    
    logger.info("Enhanced conscious brain model created successfully")
    
    return enhanced_model
