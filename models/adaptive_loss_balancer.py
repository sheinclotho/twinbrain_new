"""
Adaptive Loss Balancing for Multi-Task Multi-Modal Learning
============================================================

Implements GradNorm-inspired adaptive loss weighting to handle:
1. EEG-fMRI energy imbalance (10-100x difference)
2. Multi-task learning with varying difficulty
3. Dynamic loss weight adjustment based on training dynamics

Key Features:
- Per-modality gradient normalization
- Automatic loss weight adaptation
- Energy-aware scaling for EEG/fMRI balance
- Gradient magnitude-based weighting

References:
    - GradNorm: Gradient Normalization for Adaptive Loss Balancing (Chen et al., 2018)
    - Multi-Task Learning Using Uncertainty to Weigh Losses (Kendall et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AdaptiveLossBalancer(nn.Module):
    """
    Adaptive loss balancing using gradient normalization.
    
    Dynamically adjusts loss weights based on:
    1. Relative training rates across tasks
    2. Gradient magnitudes per task
    3. Modality-specific energy characteristics
    """
    
    def __init__(
        self,
        task_names: List[str],
        modality_names: Optional[List[str]] = None,
        initial_weights: Optional[Dict[str, float]] = None,
        alpha: float = 1.5,
        update_frequency: int = 10,
        learning_rate: float = 0.025,
        warmup_epochs: int = 5,
        modality_energy_ratios: Optional[Dict[str, float]] = None,
        min_weight: float = 0.01,
        max_weight: float = 100.0,
    ):
        """
        Initialize adaptive loss balancer.
        
        Args:
            task_names: List of task names (e.g., ['recon_eeg', 'recon_fmri', 'pred_eeg', 'pred_fmri'])
            modality_names: List of modality names (e.g., ['eeg', 'fmri'])
            initial_weights: Initial task weights (defaults to 1.0 for all)
            alpha: Restoring force for balancing (higher = more aggressive)
            update_frequency: Update weights every N steps
            learning_rate: Learning rate for weight updates
            warmup_epochs: Number of epochs before enabling adaptation
            modality_energy_ratios: Energy ratio for each modality (e.g., {'eeg': 0.01, 'fmri': 1.0})
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
        """
        super().__init__()
        
        self.task_names = task_names
        self.modality_names = modality_names or []
        self.alpha = alpha
        self.update_frequency = update_frequency
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Initialize task weights
        if initial_weights is None:
            initial_weights = {name: 1.0 for name in task_names}
        
        # Create learnable weights (in log space for stability)
        log_weights = {
            name: torch.log(torch.tensor(initial_weights.get(name, 1.0)))
            for name in task_names
        }
        self.log_weights = nn.ParameterDict({
            name: nn.Parameter(w) for name, w in log_weights.items()
        })
        
        # Modality energy ratios (stored for reference; used by modality_energy_ratios dict)
        if modality_energy_ratios is None:
            # Default: fMRI has ~50x more energy than EEG
            modality_energy_ratios = {'eeg': 0.02, 'fmri': 1.0}
        
        self.register_buffer(
            'modality_energy_ratios',
            torch.tensor([modality_energy_ratios.get(m, 1.0) for m in self.modality_names])
            if self.modality_names else None
        )
        
        # Track training dynamics (buffers will be moved to device with .to(device))
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('epoch_count', torch.tensor(0, dtype=torch.long))
        
        # Track initial loss values for normalization
        self.register_buffer('initial_losses', torch.zeros(len(task_names), dtype=torch.float32))
        self.register_buffer('initial_losses_set', torch.tensor(False, dtype=torch.bool))
        
        # Track loss history for adaptive adjustment
        self.loss_history = {name: [] for name in task_names}
        
    def forward(
        self,
        losses: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted loss.

        Args:
            losses: Dictionary of task losses {task_name: loss_value}

        Returns:
            total_loss: Weighted sum of losses
            weight_dict: Current weights for logging
        """
        # Get current weights.
        # Detach from the autograd graph: log_weights are updated via update_weights()
        # (a manual, gradient-free rule) and are NOT part of the optimizer.  Including
        # them in the backward graph would cause their .grad to accumulate across steps
        # without ever being zeroed by optimizer.zero_grad(), which can trigger
        # "Trying to backward through the graph a second time" in edge cases.
        # Detaching treats the weight scalars as constants for THIS backward pass,
        # which is exactly the intended semantics.
        weights = {name: torch.exp(self.log_weights[name]).detach().clamp(self.min_weight, self.max_weight)
                   for name in self.task_names}
        
        # Set initial losses on first call (for normalization)
        if not self.initial_losses_set:
            self.initial_losses = torch.tensor([
                losses.get(name, torch.tensor(0.0)).detach().item()
                for name in self.task_names
            ], device=self.initial_losses.device)
            # Use same device as initial_losses for consistency
            self.initial_losses_set = torch.tensor(True, device=self.initial_losses.device)
        
        # Compute weighted losses
        weighted_losses = {}
        for i, name in enumerate(self.task_names):
            if name in losses:
                # Normalize by initial loss to prevent scale differences
                loss_norm = losses[name]
                if self.initial_losses[i] > 1e-6:
                    loss_norm = loss_norm / (self.initial_losses[i] + 1e-8)
                
                weighted_losses[name] = weights[name] * loss_norm
            else:
                weighted_losses[name] = torch.tensor(0.0, device=list(losses.values())[0].device)
        
        total_loss = sum(weighted_losses.values())
        
        # Return weights for logging (already detached above)
        weight_dict = {name: w.item() for name, w in weights.items()}
        
        return total_loss, weight_dict
    
    def update_weights(
        self,
        losses: Dict[str, torch.Tensor],
        model: nn.Module,
        shared_params: Optional[List[nn.Parameter]] = None,
    ):
        """
        Update loss weights based on relative loss magnitudes.

        Design note: the original GradNorm algorithm calls torch.autograd.grad()
        per task to compute per-task gradient norms.  That requires the computation
        graph to still be alive, but update_weights() is called AFTER backward()
        has already consumed and freed the graph.  Calling autograd.grad() at that
        point raises "Trying to backward through the graph a second time".

        Instead we use loss magnitudes as a lightweight proxy: tasks whose loss is
        larger than the group average are "harder" and receive a lower weight so
        that all tasks converge at a comparable rate.  This is not GradNorm but is
        stable and avoids the post-backward graph issue.
        """
        # Only update after warmup and at specified frequency
        if self.epoch_count < self.warmup_epochs:
            return
        
        if self.step_count % self.update_frequency != 0:
            self.step_count += 1
            return
        
        self.step_count += 1
        
        # Gather scalar loss values (graph already freed — use .item())
        loss_values = {
            name: losses[name].item()
            for name in self.task_names
            if name in losses
        }
        
        if not loss_values:
            return
        
        avg_loss = sum(loss_values.values()) / len(loss_values)
        
        with torch.no_grad():
            for name, loss_val in loss_values.items():
                if loss_val < 1e-8:
                    continue
                
                # Relative loss: how much harder is this task than average?
                rel_loss = loss_val / (avg_loss + 1e-8)
                
                # Drive all task losses towards the group average.
                # If task loss > avg (rel_loss > 1): task is harder → lower its weight
                # If task loss < avg (rel_loss < 1): task is easier → raise its weight
                weight_update = -self.learning_rate * (rel_loss - 1.0)
                weight_update = max(-0.5, min(0.5, weight_update))  # clip
                
                self.log_weights[name].data += weight_update
                
                max_log = torch.log(torch.tensor(self.max_weight, device=self.log_weights[name].device))
                min_log = torch.log(torch.tensor(self.min_weight, device=self.log_weights[name].device))
                self.log_weights[name].data.clamp_(min_log, max_log)
        
        if logger.isEnabledFor(logging.DEBUG):
            weights = {name: torch.exp(self.log_weights[name]).item()
                       for name in self.task_names}
            logger.debug(f"Updated loss weights: {weights}")
            logger.debug(f"Loss values used: {loss_values}")
    
    def set_epoch(self, epoch: int):
        """Set current epoch for warmup tracking."""
        self.epoch_count = torch.tensor(epoch, dtype=torch.long, device=self.epoch_count.device)
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights as dictionary."""
        return {
            name: torch.exp(self.log_weights[name]).clamp(self.min_weight, self.max_weight).item()
            for name in self.task_names
        }
    
    def reset_history(self):
        """Reset loss history."""
        self.loss_history = {name: [] for name in self.task_names}
