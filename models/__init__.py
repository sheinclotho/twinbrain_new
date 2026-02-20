"""
TwinBrain Models Module
=======================

Exports all model components including:
- Base graph-native models
- Consciousness modules (GWT, IIT)
- Advanced attention mechanisms
- Predictive coding
- Enhanced models
"""

# Base models
from .graph_native_system import (
    GraphNativeBrainModel,
    GraphNativeTrainer,
    GraphNativeDecoder,
)
from .graph_native_encoder import (
    GraphNativeEncoder,
    SpatialTemporalGraphConv,
)
from .graph_native_mapper import (
    GraphNativeBrainMapper,
    TemporalGraphFeatureExtractor,
)

# Optimization modules
from .adaptive_loss_balancer import AdaptiveLossBalancer
from .eeg_channel_handler import EnhancedEEGHandler
from .advanced_prediction import EnhancedMultiStepPredictor

# Consciousness modules (new)
from .consciousness_module import (
    ConsciousnessModule,
    GlobalWorkspaceIntegrator,
    IntegratedInformationCalculator,
    ConsciousnessStateClassifier,
    CONSCIOUSNESS_STATES,
)

# Advanced attention (new)
from .advanced_attention import (
    CrossModalAttention,
    SpatialTemporalAttention,
    GraphAttentionWithEdges,
    HierarchicalAttention,
    ContrastiveAttention,
)

# Predictive coding (new)
from .predictive_coding import (
    PredictiveCodingLayer,
    HierarchicalPredictiveCoding,
    ActiveInference,
    PredictiveBrainModel,
    compute_free_energy_loss,
)

# Enhanced models (new)
from .enhanced_graph_native import (
    ConsciousGraphNativeBrainModel,
    EnhancedGraphNativeTrainer,
    create_enhanced_model,
)

__all__ = [
    # Base models
    'GraphNativeBrainModel',
    'GraphNativeTrainer',
    'GraphNativeDecoder',
    'GraphNativeEncoder',
    'SpatialTemporalGraphConv',
    'GraphNativeBrainMapper',
    'TemporalGraphFeatureExtractor',
    # Optimization
    'AdaptiveLossBalancer',
    'EnhancedEEGHandler',
    'EnhancedMultiStepPredictor',
    # Consciousness
    'ConsciousnessModule',
    'GlobalWorkspaceIntegrator',
    'IntegratedInformationCalculator',
    'ConsciousnessStateClassifier',
    'CONSCIOUSNESS_STATES',
    # Attention
    'CrossModalAttention',
    'SpatialTemporalAttention',
    'GraphAttentionWithEdges',
    'HierarchicalAttention',
    'ContrastiveAttention',
    # Predictive coding
    'PredictiveCodingLayer',
    'HierarchicalPredictiveCoding',
    'ActiveInference',
    'PredictiveBrainModel',
    'compute_free_energy_loss',
    # Enhanced models
    'ConsciousGraphNativeBrainModel',
    'EnhancedGraphNativeTrainer',
    'create_enhanced_model',
]
