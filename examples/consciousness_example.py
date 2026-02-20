"""
Complete Example: Using Consciousness-Aware TwinBrain
=====================================================

This script demonstrates how to:
1. Create an enhanced model with consciousness modules
2. Train with consciousness and predictive coding losses
3. Evaluate consciousness metrics
4. Visualize results

Usage:
    python examples/consciousness_example.py
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import yaml
import logging
from typing import Dict, List

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    create_enhanced_model,
    EnhancedGraphNativeTrainer,
    CONSCIOUSNESS_STATES,
)
from utils.visualization import ConsciousnessVisualizer
from torch_geometric.data import HeteroData

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_dummy_brain_graph(
    num_eeg_nodes: int = 64,
    num_fmri_nodes: int = 200,
    time_steps: int = 100,
    device: str = 'cpu',
) -> HeteroData:
    """
    Create a dummy brain graph for demonstration.
    
    In real usage, this would come from actual EEG/fMRI data.
    """
    data = HeteroData()
    
    # EEG nodes and features
    data['eeg'].x = torch.randn(num_eeg_nodes, time_steps, 1).to(device)
    data['eeg'].pos = torch.randn(num_eeg_nodes, 3).to(device)  # 3D positions
    data['eeg'].num_nodes = num_eeg_nodes
    
    # fMRI nodes and features
    data['fmri'].x = torch.randn(num_fmri_nodes, time_steps, 1).to(device)
    data['fmri'].pos = torch.randn(num_fmri_nodes, 3).to(device)
    data['fmri'].num_nodes = num_fmri_nodes
    
    # EEG connectivity (k-nearest neighbors)
    k_eeg = 10
    eeg_edges = []
    for i in range(num_eeg_nodes):
        distances = torch.cdist(data['eeg'].pos[i:i+1], data['eeg'].pos)
        nearest = torch.topk(distances, k=k_eeg+1, largest=False).indices[0, 1:]  # Exclude self
        for j in nearest:
            eeg_edges.append([i, j.item()])
    
    data['eeg', 'connects', 'eeg'].edge_index = torch.tensor(eeg_edges, dtype=torch.long).T.to(device)
    data['eeg', 'connects', 'eeg'].edge_attr = torch.rand(len(eeg_edges), 1).to(device)
    
    # fMRI connectivity
    k_fmri = 20
    fmri_edges = []
    for i in range(num_fmri_nodes):
        distances = torch.cdist(data['fmri'].pos[i:i+1], data['fmri'].pos)
        nearest = torch.topk(distances, k=k_fmri+1, largest=False).indices[0, 1:]
        for j in nearest:
            fmri_edges.append([i, j.item()])
    
    data['fmri', 'connects', 'fmri'].edge_index = torch.tensor(fmri_edges, dtype=torch.long).T.to(device)
    data['fmri', 'connects', 'fmri'].edge_attr = torch.rand(len(fmri_edges), 1).to(device)
    
    # Cross-modal connections (EEG → fMRI)
    cross_edges = []
    num_cross = min(num_eeg_nodes, num_fmri_nodes) // 2
    for i in range(num_cross):
        eeg_idx = i
        # Find nearest fMRI node
        distances = torch.cdist(data['eeg'].pos[eeg_idx:eeg_idx+1], data['fmri'].pos)
        fmri_idx = distances.argmin().item()
        cross_edges.append([eeg_idx, fmri_idx])
    
    data['eeg', 'projects_to', 'fmri'].edge_index = torch.tensor(cross_edges, dtype=torch.long).T.to(device)
    data['eeg', 'projects_to', 'fmri'].edge_attr = torch.rand(len(cross_edges), 1).to(device)
    
    return data


def main():
    """Main demonstration function."""
    
    logger.info("=" * 80)
    logger.info("TwinBrain Consciousness-Aware Model Example")
    logger.info("=" * 80)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    output_dir = Path('outputs/consciousness_example')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create enhanced model
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Creating Enhanced Model with Consciousness")
    logger.info("=" * 80)
    
    base_model_config = {
        'node_types': ['eeg', 'fmri'],
        'edge_types': [
            ('eeg', 'connects', 'eeg'),
            ('fmri', 'connects', 'fmri'),
            ('eeg', 'projects_to', 'fmri'),
        ],
        'in_channels_dict': {'eeg': 1, 'fmri': 1},
        'hidden_channels': 64,  # Smaller for demo
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'use_prediction': True,
        'prediction_steps': 10,
        'dropout': 0.1,
    }
    
    enhanced_model = create_enhanced_model(
        base_model_config=base_model_config,
        enable_consciousness=True,
        enable_cross_modal_attention=True,
        enable_predictive_coding=True,
    )
    
    enhanced_model = enhanced_model.to(device)
    
    logger.info(f"✓ Model created with {sum(p.numel() for p in enhanced_model.parameters()):,} parameters")
    
    # Step 2: Create training data
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Creating Training Data")
    logger.info("=" * 80)
    
    num_samples = 10
    train_graphs = [create_dummy_brain_graph(device=device) for _ in range(num_samples)]
    
    logger.info(f"✓ Created {len(train_graphs)} training samples")
    logger.info(f"  - EEG nodes: {train_graphs[0]['eeg'].num_nodes}")
    logger.info(f"  - fMRI nodes: {train_graphs[0]['fmri'].num_nodes}")
    logger.info(f"  - Time steps: {train_graphs[0]['eeg'].x.shape[1]}")
    
    # Step 3: Forward pass and inspect consciousness metrics
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Forward Pass and Consciousness Metrics")
    logger.info("=" * 80)
    
    enhanced_model.eval()
    with torch.no_grad():
        data = train_graphs[0]
        reconstructions, predictions, info = enhanced_model(
            data,
            return_consciousness_metrics=True,
        )
    
    # Display consciousness metrics
    if 'consciousness' in info:
        consciousness_info = info['consciousness']
        
        logger.info("\nConsciousness Metrics:")
        logger.info(f"  - Integrated Information Φ: {consciousness_info['phi'].mean():.4f}")
        logger.info(f"  - Consciousness Level: {consciousness_info['consciousness_level'].mean():.2%}")
        
        state_logits = consciousness_info['state_logits']
        predicted_state_idx = state_logits.argmax(dim=-1).item()
        predicted_state = CONSCIOUSNESS_STATES[predicted_state_idx]
        confidence = torch.softmax(state_logits, dim=-1)[0, predicted_state_idx].item()
        
        logger.info(f"  - Predicted State: {predicted_state} (confidence: {confidence:.2%})")
        
        # Show top 3 states
        probs = torch.softmax(state_logits, dim=-1)[0]
        top3_idx = torch.topk(probs, k=3).indices
        logger.info("\n  Top 3 Consciousness States:")
        for i, idx in enumerate(top3_idx):
            state_name = CONSCIOUSNESS_STATES[idx.item()]
            prob = probs[idx].item()
            logger.info(f"    {i+1}. {state_name}: {prob:.2%}")
    
    # Display attention metrics
    if 'cross_modal_attention' in info:
        cm_info = info['cross_modal_attention']
        logger.info("\nCross-Modal Attention:")
        logger.info(f"  - EEG→fMRI attention: {cm_info['eeg_to_fmri_weights'].mean():.4f}")
        logger.info(f"  - fMRI→EEG attention: {cm_info['fmri_to_eeg_weights'].mean():.4f}")
    
    # Display predictive coding metrics
    for key in info:
        if 'predictive_coding' in key:
            pc_info = info[key]
            logger.info(f"\nPredictive Coding ({key}):")
            logger.info(f"  - Total Free Energy: {pc_info['total_free_energy']:.4f}")
            logger.info(f"  - Prediction Errors: {len(pc_info['all_errors'])} layers")
    
    # Step 4: Visualization
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Creating Visualizations")
    logger.info("=" * 80)
    
    viz = ConsciousnessVisualizer(output_dir=output_dir)
    
    # Plot global workspace
    if 'consciousness' in info:
        viz.plot_global_workspace(
            info['consciousness'],
            save_as='global_workspace.png'
        )
        logger.info("✓ Global workspace visualization saved")
    
    # Simulate Φ over time (in practice, this comes from multiple forward passes)
    phi_history = []
    for _ in range(100):
        with torch.no_grad():
            _, _, info_t = enhanced_model(data, return_consciousness_metrics=True)
            if 'consciousness' in info_t:
                phi_history.append(info_t['consciousness']['phi'].mean().item())
    
    viz.plot_phi_timeseries(phi_history, save_as='phi_timeseries.png')
    logger.info("✓ Φ timeseries visualization saved")
    
    # Plot cross-modal attention
    if 'cross_modal_attention' in info:
        cm_info = info['cross_modal_attention']
        viz.plot_cross_modal_attention(
            eeg_to_fmri=cm_info['eeg_to_fmri_weights'][0],
            fmri_to_eeg=cm_info['fmri_to_eeg_weights'][0],
            save_as='cross_modal_attention.png'
        )
        logger.info("✓ Cross-modal attention visualization saved")
    
    # Step 5: Training (optional demo)
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: Training Demo (5 epochs)")
    logger.info("=" * 80)
    
    trainer = EnhancedGraphNativeTrainer(
        model=enhanced_model,
        node_types=['eeg', 'fmri'],
        learning_rate=1e-3,
        weight_decay=1e-5,
        use_adaptive_loss=True,
        consciousness_loss_weight=0.1,
        predictive_coding_loss_weight=0.1,
        device=device,
    )
    
    # Train for a few epochs
    for epoch in range(1, 6):
        train_loss = trainer.train_epoch(train_graphs, epoch=epoch, total_epochs=5)
        logger.info(f"Epoch {epoch}/5: train_loss={train_loss:.4f}")
    
    # Step 6: Summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    
    logger.info("\n✓ Successfully demonstrated:")
    logger.info("  1. Enhanced model creation with consciousness modules")
    logger.info("  2. Forward pass with consciousness metrics")
    logger.info("  3. Visualization of consciousness indicators")
    logger.info("  4. Training with consciousness-aware losses")
    
    logger.info(f"\n✓ Outputs saved to: {output_dir}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Next Steps:")
    logger.info("=" * 80)
    logger.info("1. Replace dummy data with real EEG/fMRI data")
    logger.info("2. Train on larger datasets for longer")
    logger.info("3. Validate consciousness metrics against ground truth")
    logger.info("4. Experiment with different architectures")
    logger.info("5. Analyze attention patterns for scientific insights")
    
    logger.info("\n" + "=" * 80)
    logger.info("Example completed successfully!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
