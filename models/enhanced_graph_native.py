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
    
    def forward(
        self,
        data: HeteroData,
        return_consciousness_metrics: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass with consciousness and predictive coding.
        
        Args:
            data: Input heterogeneous graph
            return_consciousness_metrics: Whether to return consciousness metrics
        
        Returns:
            Tuple of:
                - Reconstructions dict
                - Predictions dict
                - Info dict with consciousness and attention metrics
        """
        # 1. Base model forward pass
        reconstructions, predictions = self.base_model(data)
        
        info = {}
        
        # 2. Apply cross-modal attention if both modalities present
        if self.enable_cross_modal_attention and 'eeg' in data.node_types and 'fmri' in data.node_types:
            # Get encoded features from base model
            # Note: This requires access to intermediate representations
            # For now, we'll work with reconstructions as a proxy
            
            eeg_features = reconstructions.get('eeg')
            fmri_features = reconstructions.get('fmri')
            
            if eeg_features is not None and fmri_features is not None:
                # Ensure features have batch dimension
                if len(eeg_features.shape) == 3:  # [nodes, time, channels]
                    eeg_features = eeg_features.unsqueeze(0)  # [1, nodes, time, channels]
                if len(fmri_features.shape) == 3:
                    fmri_features = fmri_features.unsqueeze(0)
                
                # Apply spatial-temporal attention first
                if len(eeg_features.shape) == 4:  # [batch, nodes, time, channels]
                    eeg_attended, eeg_st_info = self.spatial_temporal_attention(eeg_features)
                    fmri_attended, fmri_st_info = self.spatial_temporal_attention(fmri_features)
                    
                    info['eeg_st_attention'] = eeg_st_info
                    info['fmri_st_attention'] = fmri_st_info
                    
                    # Aggregate time dimension for cross-modal attention
                    eeg_agg = eeg_attended.mean(dim=2)  # [batch, nodes, channels]
                    fmri_agg = fmri_attended.mean(dim=2)
                else:
                    eeg_agg = eeg_features
                    fmri_agg = fmri_features
                
                # Apply cross-modal attention
                eeg_enhanced, fmri_enhanced, cm_info = self.cross_modal_attention(
                    eeg_features=eeg_agg,
                    fmri_features=fmri_agg,
                )
                
                info['cross_modal_attention'] = cm_info
                
                # Update reconstructions with enhanced features
                reconstructions['eeg_enhanced'] = eeg_enhanced
                reconstructions['fmri_enhanced'] = fmri_enhanced
        
        # 3. Apply consciousness module
        if self.enable_consciousness and return_consciousness_metrics:
            # Aggregate features across modalities for consciousness computation
            all_features = []
            
            for modality in ['eeg', 'fmri']:
                if modality in reconstructions:
                    feat = reconstructions[modality]
                    
                    # Ensure shape [batch, num_nodes, channels]
                    if len(feat.shape) == 3:  # [nodes, time, channels]
                        feat = feat.unsqueeze(0).mean(dim=2)  # [1, nodes, channels]
                    elif len(feat.shape) == 4:  # [batch, nodes, time, channels]
                        feat = feat.mean(dim=2)  # [batch, nodes, channels]
                    
                    all_features.append(feat)
            
            if len(all_features) > 0:
                # Concatenate features from different modalities
                combined_features = torch.cat(all_features, dim=1)  # [batch, total_nodes, channels]
                
                # Get edge index (use fMRI edges as default)
                if ('fmri', 'connects', 'fmri') in data.edge_types:
                    edge_index = data['fmri', 'connects', 'fmri'].edge_index
                else:
                    # Create a simple fully connected graph
                    num_nodes = combined_features.shape[1]
                    edge_index = torch.combinations(torch.arange(num_nodes), r=2).T
                    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
                    edge_index = edge_index.to(combined_features.device)
                
                # Apply consciousness module
                conscious_features, consciousness_info = self.consciousness_module(
                    x=combined_features,
                    edge_index=edge_index,
                )
                
                info['consciousness'] = consciousness_info
                reconstructions['conscious_representation'] = conscious_features
                
                logger.debug(
                    f"Consciousness metrics - Φ: {consciousness_info['phi'].mean():.4f}, "
                    f"Level: {consciousness_info['consciousness_level'].mean():.4f}"
                )
        
        # 4. Apply predictive coding
        if self.enable_predictive_coding:
            # Apply to aggregated features
            for modality in ['eeg', 'fmri']:
                if modality in reconstructions:
                    feat = reconstructions[modality]
                    
                    # Aggregate to [batch, channels]
                    if len(feat.shape) == 3:  # [nodes, time, channels]
                        feat_agg = feat.mean(dim=(0, 1)).unsqueeze(0)  # [1, channels]
                    elif len(feat.shape) == 4:  # [batch, nodes, time, channels]
                        feat_agg = feat.mean(dim=(1, 2))  # [batch, channels]
                    else:
                        feat_agg = feat.mean(dim=1)  # [batch, channels]
                    
                    # Apply predictive coding
                    pc_state, pc_predictions, pc_info = self.predictive_coding(feat_agg)
                    
                    info[f'{modality}_predictive_coding'] = pc_info
                    reconstructions[f'{modality}_pc_state'] = pc_state
        
        return reconstructions, predictions, info


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
        # Initialize base trainer with the base model
        super().__init__(
            model=model.base_model,
            node_types=node_types,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            use_adaptive_loss=use_adaptive_loss,
            use_eeg_enhancement=use_eeg_enhancement,
            **kwargs
        )
        
        # Replace model with conscious model
        self.model = model
        
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
