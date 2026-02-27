"""
Visualization Tools for Consciousness-Aware Brain Model
=======================================================

Tools for visualizing:
1. Global workspace integration and broadcasting
2. Integrated information (Φ) over time
3. Cross-modal attention weights
4. Spatial-temporal attention patterns
5. Predictive coding hierarchies
6. Consciousness state transitions

Usage:
    from utils.visualization import ConsciousnessVisualizer
    
    viz = ConsciousnessVisualizer()
    viz.plot_global_workspace(info)
    viz.plot_phi_timeseries(phi_history)
    viz.plot_cross_modal_attention(attention_weights)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class ConsciousnessVisualizer:
    """
    Visualization tools for consciousness modeling components.
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        style: str = 'seaborn',
        dpi: int = 150,
    ):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save figures (None = don't save)
            style: Matplotlib style
            dpi: Figure DPI
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use(style)
        self.dpi = dpi
        
        # Color schemes
        self.colors = {
            'integration': 'viridis',
            'broadcasting': 'plasma',
            'attention': 'coolwarm',
            'phi': 'RdYlGn',
            'states': 'tab10',
        }
    
    def _save_or_show(self, filename: Optional[str] = None):
        """Save figure or show it."""
        if filename and self.output_dir:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
        else:
            plt.show()
    
    def plot_global_workspace(
        self,
        info: Dict[str, torch.Tensor],
        sample_idx: int = 0,
        region_names: Optional[List[str]] = None,
        save_as: Optional[str] = None,
    ):
        """
        Plot global workspace integration and broadcasting.
        
        Args:
            info: Info dict from consciousness module
            sample_idx: Which sample in batch to plot
            region_names: Names of brain regions
            save_as: Filename to save (optional)
        """
        integration_weights = info['integration_weights'][sample_idx].detach().cpu().numpy()
        broadcast_weights = info['broadcast_weights'][sample_idx].detach().cpu().numpy()
        competition_probs = info['competition_probs'][sample_idx].detach().cpu().numpy()
        
        fig = plt.figure(figsize=(16, 5))
        
        # Integration weights
        ax1 = plt.subplot(1, 3, 1)
        im1 = ax1.imshow(integration_weights, cmap=self.colors['integration'], aspect='auto')
        ax1.set_title('Integration: Brain Regions → Workspace', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Brain Regions')
        ax1.set_ylabel('Workspace Slots')
        plt.colorbar(im1, ax=ax1, label='Attention Weight')
        
        if region_names:
            ax1.set_xticks(range(len(region_names)))
            ax1.set_xticklabels(region_names, rotation=90, fontsize=8)
        
        # Broadcasting weights
        ax2 = plt.subplot(1, 3, 2)
        im2 = ax2.imshow(broadcast_weights, cmap=self.colors['broadcasting'], aspect='auto')
        ax2.set_title('Broadcasting: Workspace → Brain Regions', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Workspace Slots')
        ax2.set_ylabel('Brain Regions')
        plt.colorbar(im2, ax=ax2, label='Attention Weight')
        
        if region_names:
            ax2.set_yticks(range(len(region_names)))
            ax2.set_yticklabels(region_names, fontsize=8)
        
        # Competition probabilities
        ax3 = plt.subplot(1, 3, 3)
        slots = np.arange(len(competition_probs))
        bars = ax3.bar(slots, competition_probs, color='steelblue', alpha=0.7)
        
        # Highlight winning slots
        top_3_idx = np.argsort(competition_probs)[-3:]
        for idx in top_3_idx:
            bars[idx].set_color('orange')
        
        ax3.set_title('Competition: Slot Selection', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Workspace Slot')
        ax3.set_ylabel('Selection Probability')
        ax3.axhline(y=1.0/len(competition_probs), color='r', linestyle='--', 
                   label='Uniform (random)', alpha=0.5)
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        plt.tight_layout()
        self._save_or_show(save_as or 'global_workspace.png')
    
    def plot_phi_timeseries(
        self,
        phi_history: List[float],
        timestamps: Optional[np.ndarray] = None,
        consciousness_thresholds: Optional[Dict[str, float]] = None,
        save_as: Optional[str] = None,
    ):
        """
        Plot integrated information Φ over time.
        
        Args:
            phi_history: List of Φ values over time
            timestamps: Time points (seconds)
            consciousness_thresholds: Dict of state thresholds
            save_as: Filename to save
        """
        phi_array = np.array(phi_history)
        
        if timestamps is None:
            timestamps = np.arange(len(phi_array))
        
        if consciousness_thresholds is None:
            consciousness_thresholds = {
                'High (Wakefulness)': 0.6,
                'Medium (REM Sleep)': 0.4,
                'Low (Deep Sleep)': 0.2,
                'Very Low (Anesthesia)': 0.1,
            }
        
        plt.figure(figsize=(14, 6))
        
        # Plot Φ
        plt.plot(timestamps, phi_array, linewidth=2, color='navy', label='Φ (Integrated Information)')
        plt.fill_between(timestamps, 0, phi_array, alpha=0.3, color='skyblue')
        
        # Add threshold lines
        colors = ['green', 'yellow', 'orange', 'red']
        for (label, threshold), color in zip(consciousness_thresholds.items(), colors):
            plt.axhline(y=threshold, color=color, linestyle='--', alpha=0.7, label=label)
        
        # Styling
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Integrated Information Φ', fontsize=12)
        plt.title('Consciousness Level Over Time', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', framealpha=0.9)
        plt.grid(alpha=0.3)
        
        # Stats
        stats_text = f'Mean: {phi_array.mean():.3f}\nStd: {phi_array.std():.3f}\nMax: {phi_array.max():.3f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        self._save_or_show(save_as or 'phi_timeseries.png')
    
    def plot_cross_modal_attention(
        self,
        eeg_to_fmri: torch.Tensor,
        fmri_to_eeg: torch.Tensor,
        eeg_names: Optional[List[str]] = None,
        fmri_names: Optional[List[str]] = None,
        save_as: Optional[str] = None,
    ):
        """
        Plot cross-modal attention matrices.
        
        Args:
            eeg_to_fmri: EEG → fMRI attention [n_eeg, n_fmri]
            fmri_to_eeg: fMRI → EEG attention [n_fmri, n_eeg]
            eeg_names: EEG channel names
            fmri_names: fMRI ROI names
            save_as: Filename to save
        """
        eeg_to_fmri_np = eeg_to_fmri.detach().cpu().numpy()
        fmri_to_eeg_np = fmri_to_eeg.detach().cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # EEG → fMRI
        im1 = axes[0].imshow(eeg_to_fmri_np, cmap=self.colors['attention'], aspect='auto')
        axes[0].set_title('EEG Queries fMRI\n(High Temporal → High Spatial)', 
                         fontsize=12, fontweight='bold')
        axes[0].set_xlabel('fMRI ROIs')
        axes[0].set_ylabel('EEG Channels')
        plt.colorbar(im1, ax=axes[0], label='Attention Weight')
        
        if eeg_names:
            axes[0].set_yticks(range(len(eeg_names)))
            axes[0].set_yticklabels(eeg_names, fontsize=7)
        if fmri_names:
            axes[0].set_xticks(range(len(fmri_names)))
            axes[0].set_xticklabels(fmri_names, rotation=90, fontsize=7)
        
        # fMRI → EEG
        im2 = axes[1].imshow(fmri_to_eeg_np, cmap=self.colors['attention'], aspect='auto')
        axes[1].set_title('fMRI Queries EEG\n(High Spatial → High Temporal)', 
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('EEG Channels')
        axes[1].set_ylabel('fMRI ROIs')
        plt.colorbar(im2, ax=axes[1], label='Attention Weight')
        
        if fmri_names:
            axes[1].set_yticks(range(len(fmri_names)))
            axes[1].set_yticklabels(fmri_names, fontsize=7)
        if eeg_names:
            axes[1].set_xticks(range(len(eeg_names)))
            axes[1].set_xticklabels(eeg_names, rotation=90, fontsize=7)
        
        plt.tight_layout()
        self._save_or_show(save_as or 'cross_modal_attention.png')
    
    def plot_consciousness_state_trajectory(
        self,
        state_history: List[int],
        state_names: List[str],
        phi_history: Optional[List[float]] = None,
        timestamps: Optional[np.ndarray] = None,
        save_as: Optional[str] = None,
    ):
        """
        Plot consciousness state transitions over time.
        
        Args:
            state_history: List of state indices over time
            state_names: Names of consciousness states
            phi_history: Optional Φ values
            timestamps: Time points
            save_as: Filename to save
        """
        state_array = np.array(state_history)
        
        if timestamps is None:
            timestamps = np.arange(len(state_array))
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # State trajectory
        ax1 = axes[0]
        
        # Color-coded segments
        colors = plt.cm.get_cmap(self.colors['states'], len(state_names))
        for i in range(len(timestamps) - 1):
            state = state_array[i]
            ax1.plot(timestamps[i:i+2], [state, state], 
                    linewidth=3, color=colors(state))
        
        ax1.set_ylabel('Consciousness State', fontsize=12)
        ax1.set_yticks(range(len(state_names)))
        ax1.set_yticklabels(state_names)
        ax1.set_title('Consciousness State Trajectory', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3, axis='x')
        
        # Φ trajectory (if provided)
        if phi_history is not None:
            ax2 = axes[1]
            phi_array = np.array(phi_history)
            
            ax2.plot(timestamps, phi_array, linewidth=2, color='navy')
            ax2.fill_between(timestamps, 0, phi_array, alpha=0.3, color='skyblue')
            ax2.set_xlabel('Time (s)', fontsize=12)
            ax2.set_ylabel('Φ', fontsize=12)
            ax2.set_title('Integrated Information Φ', fontsize=12, fontweight='bold')
            ax2.grid(alpha=0.3)
        else:
            axes[1].set_visible(False)
        
        plt.tight_layout()
        self._save_or_show(save_as or 'consciousness_trajectory.png')
    
    def plot_spatial_temporal_attention(
        self,
        spatial_weights: torch.Tensor,
        temporal_weights: torch.Tensor,
        region_names: Optional[List[str]] = None,
        save_as: Optional[str] = None,
    ):
        """
        Plot spatial and temporal attention patterns.
        
        Args:
            spatial_weights: Spatial attention [num_nodes, num_nodes]
            temporal_weights: Temporal attention [time_steps, time_steps]
            region_names: Brain region names
            save_as: Filename to save
        """
        spatial_np = spatial_weights.detach().cpu().numpy()
        temporal_np = temporal_weights.detach().cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Spatial attention
        im1 = axes[0].imshow(spatial_np, cmap=self.colors['attention'])
        axes[0].set_title('Spatial Attention\n(Brain Region Interactions)', 
                         fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Brain Regions')
        axes[0].set_ylabel('Brain Regions')
        plt.colorbar(im1, ax=axes[0], label='Attention Weight')
        
        if region_names:
            axes[0].set_xticks(range(len(region_names)))
            axes[0].set_xticklabels(region_names, rotation=90, fontsize=7)
            axes[0].set_yticks(range(len(region_names)))
            axes[0].set_yticklabels(region_names, fontsize=7)
        
        # Temporal attention
        im2 = axes[1].imshow(temporal_np, cmap=self.colors['attention'])
        axes[1].set_title('Temporal Attention\n(Time Point Dependencies)', 
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Time Steps')
        axes[1].set_ylabel('Time Steps')
        plt.colorbar(im2, ax=axes[1], label='Attention Weight')
        
        plt.tight_layout()
        self._save_or_show(save_as or 'spatial_temporal_attention.png')
    
    def plot_predictive_coding_hierarchy(
        self,
        predictions: List[torch.Tensor],
        errors: List[torch.Tensor],
        layer_names: Optional[List[str]] = None,
        save_as: Optional[str] = None,
    ):
        """
        Plot predictive coding hierarchy.
        
        Args:
            predictions: Predictions at each layer
            errors: Prediction errors at each layer
            layer_names: Names of layers
            save_as: Filename to save
        """
        num_layers = len(predictions)
        
        if layer_names is None:
            layer_names = [f'Layer {i+1}' for i in range(num_layers)]
        
        fig, axes = plt.subplots(2, num_layers, figsize=(4*num_layers, 8))
        
        # Predictions
        for i, (pred, name) in enumerate(zip(predictions, layer_names)):
            ax = axes[0, i] if num_layers > 1 else axes[0]
            pred_np = pred.detach().cpu().numpy().squeeze()
            
            if len(pred_np.shape) == 1:
                ax.plot(pred_np)
            else:
                im = ax.imshow(pred_np, cmap='viridis', aspect='auto')
                plt.colorbar(im, ax=ax)
            
            ax.set_title(f'{name}\nPrediction', fontsize=10)
        
        # Errors
        for i, (err, name) in enumerate(zip(errors, layer_names)):
            ax = axes[1, i] if num_layers > 1 else axes[1]
            err_np = err.detach().cpu().numpy().squeeze()
            
            if len(err_np.shape) == 1:
                ax.plot(err_np, color='red')
            else:
                im = ax.imshow(err_np, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                plt.colorbar(im, ax=ax)
            
            ax.set_title(f'{name}\nPrediction Error', fontsize=10)
        
        plt.suptitle('Predictive Coding Hierarchy', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_or_show(save_as or 'predictive_coding_hierarchy.png')


def plot_training_curves(
    history: Dict[str, list],
    output_dir,
    best_epoch: Optional[int] = None,
    best_r2_dict: Optional[Dict[str, float]] = None,
) -> None:
    """Save training-loss and R² curves after training completes.

    Generates two PNG files in ``output_dir``:
    - ``training_loss_curve.png``: train_loss + val_loss vs epoch
    - ``training_r2_curve.png``: val R² per modality vs validation index
      (only when R² history is present in ``history``)

    Gracefully skips if matplotlib is unavailable or if ``history`` is empty.

    Args:
        history: Trainer history dict with keys such as 'train_loss',
            'val_loss', and optionally 'val_r2_eeg', 'val_r2_fmri', etc.
        output_dir: Directory (str or Path) where PNGs are saved.
        best_epoch: Epoch index (1-based) of the best model, for annotation.
        best_r2_dict: R² values at the best epoch, for annotation.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')   # non-interactive backend; safe on headless servers
        import matplotlib.pyplot as _plt
    except ImportError:
        return  # matplotlib unavailable — skip silently

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loss = history.get('train_loss', [])
    val_loss   = history.get('val_loss',   [])

    # ── Loss curve ─────────────────────────────────────────────────────────
    if train_loss or val_loss:
        fig, ax = _plt.subplots(figsize=(8, 4))
        if train_loss:
            ax.plot(range(1, len(train_loss) + 1), train_loss, label='Train Loss',
                    color='steelblue', linewidth=1.5)
        if val_loss:
            # val_loss is recorded only at validation epochs; x-coordinates are
            # inferred from val_frequency but we don't have it here, so use
            # sequential indices and label the axis accordingly.
            ax.plot(range(1, len(val_loss) + 1), val_loss, label='Val Loss',
                    color='tomato', linewidth=1.5, marker='o', markersize=3)
            # Annotate best validation point
            best_val_idx = int(min(range(len(val_loss)), key=lambda i: val_loss[i]))
            ax.annotate(
                f'best\n{val_loss[best_val_idx]:.4f}',
                xy=(best_val_idx + 1, val_loss[best_val_idx]),
                xytext=(best_val_idx + 1 + max(1, len(val_loss) // 20), val_loss[best_val_idx]),
                fontsize=7,
                arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                color='gray',
            )
        ax.set_xlabel('Epoch / Validation Index')
        ax.set_ylabel('Loss')
        ax.set_title('TwinBrain V5 — Training Loss Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        loss_path = output_dir / 'training_loss_curve.png'
        fig.savefig(loss_path, dpi=120, bbox_inches='tight')
        _plt.close(fig)

    # ── R² curve ───────────────────────────────────────────────────────────
    # Collect all val_r2_* keys present in history
    r2_keys = sorted(k for k in history if k.startswith('val_r2_') and history[k])
    if r2_keys:
        fig, ax = _plt.subplots(figsize=(8, 4))
        colors = ['steelblue', 'tomato', 'forestgreen', 'darkorange', 'purple']
        for idx, key in enumerate(r2_keys):
            vals = history[key]
            label = key.replace('val_r2_', 'R² ')
            ax.plot(range(1, len(vals) + 1), vals,
                    label=label, color=colors[idx % len(colors)],
                    linewidth=1.5, marker='o', markersize=3)
        # Reference lines
        ax.axhline(0.3, color='green',  linestyle='--', linewidth=0.8, alpha=0.6, label='R²=0.3 (good)')
        ax.axhline(0.0, color='orange', linestyle='--', linewidth=0.8, alpha=0.6, label='R²=0 (baseline)')
        ax.set_xlabel('Validation Index')
        ax.set_ylabel('R² (coefficient of determination)')
        ax.set_title('TwinBrain V5 — Validation R² Curve')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        # Annotate best R² values if provided
        if best_r2_dict:
            note = '  '.join(f'{k}={v:.3f}' for k, v in sorted(best_r2_dict.items()))
            ax.set_xlabel(f'Validation Index\n[Best model: {note}]', fontsize=8)
        fig.tight_layout()
        r2_path = output_dir / 'training_r2_curve.png'
        fig.savefig(r2_path, dpi=120, bbox_inches='tight')
        _plt.close(fig)


def create_sample_visualizations(output_dir: str = 'visualization_examples'):
    """
    Create sample visualizations with dummy data.
    
    Args:
        output_dir: Directory to save examples
    """
    print("Creating sample visualizations...")
    
    viz = ConsciousnessVisualizer(output_dir=output_dir)
    
    # 1. Global workspace (dummy data)
    dummy_gwt_info = {
        'integration_weights': torch.rand(1, 16, 50),
        'broadcast_weights': torch.rand(1, 50, 16),
        'competition_probs': torch.softmax(torch.randn(1, 16), dim=-1),
    }
    viz.plot_global_workspace(dummy_gwt_info, save_as='example_global_workspace.png')
    
    # 2. Φ timeseries
    phi_history = 0.3 + 0.3 * np.sin(np.linspace(0, 4*np.pi, 300)) + 0.05 * np.random.randn(300)
    phi_history = np.clip(phi_history, 0, 1)
    viz.plot_phi_timeseries(phi_history, save_as='example_phi_timeseries.png')
    
    # 3. Cross-modal attention
    eeg_to_fmri = torch.softmax(torch.randn(64, 200), dim=-1)
    fmri_to_eeg = torch.softmax(torch.randn(200, 64), dim=-1)
    viz.plot_cross_modal_attention(eeg_to_fmri, fmri_to_eeg, save_as='example_cross_modal.png')
    
    # 4. State trajectory
    state_names = ['Wakefulness', 'REM', 'NREM', 'Anesthesia', 'Coma']
    state_history = np.random.choice(len(state_names), size=200)
    viz.plot_consciousness_state_trajectory(
        state_history, state_names, phi_history, save_as='example_state_trajectory.png'
    )
    
    print(f"✓ Sample visualizations saved to: {output_dir}")


if __name__ == '__main__':
    create_sample_visualizations()
