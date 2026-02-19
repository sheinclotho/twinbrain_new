"""
Predictive Coding and Free Energy Principle Implementation
===========================================================

Implements predictive coding framework inspired by:
- Free Energy Principle (Friston, 2010)
- Predictive Processing (Clark, 2013)
- Active Inference (Friston et al., 2016)

The brain as a prediction machine that minimizes surprise.

Key concepts:
1. Prediction: Top-down generative model
2. Prediction Error: Bottom-up signal
3. Precision: Confidence in predictions
4. Free Energy: Surprise + Complexity
5. Active Inference: Action selection to minimize free energy

References:
- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science.
- Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex.
- Friston, K. et al. (2016). Active inference and learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class PredictiveCodingLayer(nn.Module):
    """
    Single layer of predictive coding hierarchy.
    
    Each layer:
    1. Receives prediction from above (top-down)
    2. Receives input from below (bottom-up)
    3. Computes prediction error
    4. Updates representation to minimize error
    5. Sends prediction down and error up
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_precision: bool = True,
        num_iterations: int = 5,
        learning_rate: float = 0.1,
    ):
        """
        Initialize predictive coding layer.
        
        Args:
            input_dim: Dimension of input from below
            hidden_dim: Hidden dimension for internal representation
            output_dim: Dimension of prediction sent down
            use_precision: Whether to learn precision (confidence) weights
            num_iterations: Number of iterative inference steps
            learning_rate: Learning rate for internal inference
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_precision = use_precision
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        
        # Top-down prediction pathway (generative model)
        self.prediction_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Bottom-up error pathway (recognition model)
        self.error_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Lateral connections for context
        self.lateral_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Precision (inverse variance) for weighting errors
        if use_precision:
            self.precision_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Softplus(),  # Ensure positive precision
            )
        
        # Learnable initial state
        self.initial_state = nn.Parameter(torch.randn(1, hidden_dim))
    
    def forward(
        self,
        bottom_up_input: torch.Tensor,
        top_down_prediction: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with iterative inference.
        
        Args:
            bottom_up_input: Input from layer below [batch, input_dim]
            top_down_prediction: Prediction from layer above [batch, hidden_dim] (optional)
            return_trajectory: Whether to return full inference trajectory
        
        Returns:
            Tuple of:
                - Internal representation [batch, hidden_dim]
                - Prediction sent down [batch, output_dim]
                - Info dict with errors and precision
        """
        batch_size = bottom_up_input.shape[0]
        
        # Initialize internal representation
        if top_down_prediction is not None:
            state = top_down_prediction
        else:
            state = self.initial_state.expand(batch_size, -1)
        
        # Compute precision (confidence) if enabled
        if self.use_precision:
            precision = self.precision_net(bottom_up_input)  # [B, output_dim]
        else:
            precision = torch.ones(batch_size, self.output_dim, device=bottom_up_input.device)
        
        # Iterative inference to minimize prediction error
        errors = []
        free_energies = []
        states = [state] if return_trajectory else []
        
        for iteration in range(self.num_iterations):
            # Generate prediction from current state
            prediction = self.prediction_net(state)  # [B, output_dim]
            
            # Compute prediction error (bottom-up input - prediction)
            error = bottom_up_input - prediction  # [B, output_dim] if output_dim == input_dim
            
            # Weight error by precision
            weighted_error = error * precision
            
            # Compute free energy (prediction error + complexity)
            free_energy = (weighted_error ** 2).sum(dim=-1).mean()  # [scalar]
            
            errors.append(error)
            free_energies.append(free_energy)
            
            # Update state to minimize prediction error
            # Gradient descent on free energy
            error_signal = self.error_net(weighted_error if weighted_error.shape[-1] == self.input_dim else error)
            lateral_signal = self.lateral_net(state)
            
            state = state - self.learning_rate * (state - error_signal - lateral_signal)
            
            if return_trajectory:
                states.append(state)
        
        # Final prediction
        final_prediction = self.prediction_net(state)
        final_error = bottom_up_input - final_prediction
        
        info = {
            'prediction': final_prediction,
            'error': final_error,
            'precision': precision,
            'free_energy': free_energies[-1],
            'errors_trajectory': torch.stack(errors) if errors else None,
            'free_energies_trajectory': torch.stack(free_energies) if free_energies else None,
            'states_trajectory': torch.stack(states) if return_trajectory else None,
        }
        
        return state, final_prediction, info


class HierarchicalPredictiveCoding(nn.Module):
    """
    Multi-layer hierarchical predictive coding network.
    
    Implements a hierarchy of predictive coding layers:
    - Lower layers: Fast, detailed predictions (e.g., sensory)
    - Higher layers: Slow, abstract predictions (e.g., conceptual)
    
    Information flows:
    - Bottom-up: Prediction errors
    - Top-down: Predictions
    - Lateral: Context
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_iterations: int = 5,
        use_precision: bool = True,
        learning_rate: float = 0.1,
    ):
        """
        Initialize hierarchical predictive coding network.
        
        Args:
            input_dim: Dimension of sensory input
            hidden_dims: Dimensions of hidden layers (bottom to top)
            num_iterations: Number of inference iterations per layer
            use_precision: Whether to use precision weighting
            learning_rate: Learning rate for inference
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        
        # Build hierarchy of predictive coding layers
        self.layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            layer_hidden_dim = hidden_dims[i]
            layer_output_dim = layer_input_dim  # Predict input to this layer
            
            layer = PredictiveCodingLayer(
                input_dim=layer_input_dim,
                hidden_dim=layer_hidden_dim,
                output_dim=layer_output_dim,
                use_precision=use_precision,
                num_iterations=num_iterations,
                learning_rate=learning_rate,
            )
            
            self.layers.append(layer)
    
    def forward(
        self,
        sensory_input: torch.Tensor,
        return_all_layers: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass through hierarchy.
        
        Args:
            sensory_input: Bottom-up sensory input [batch, input_dim]
            return_all_layers: Whether to return representations from all layers
        
        Returns:
            Tuple of:
                - Top-level representation [batch, hidden_dims[-1]]
                - List of predictions at each layer
                - Info dict with errors and free energies
        """
        batch_size = sensory_input.shape[0]
        
        # Storage for layer outputs
        representations = []
        predictions = []
        errors = []
        free_energies = []
        
        # Bottom-up pass (compute errors)
        current_input = sensory_input
        
        for i, layer in enumerate(self.layers):
            # Infer representation at this layer
            representation, prediction, info = layer(
                bottom_up_input=current_input,
                top_down_prediction=None,
            )
            
            representations.append(representation)
            predictions.append(prediction)
            errors.append(info['error'])
            free_energies.append(info['free_energy'])
            
            # Use representation as input to next layer
            current_input = representation
        
        # Top-level representation
        top_representation = representations[-1]
        
        # Aggregate info
        info = {
            'all_representations': representations if return_all_layers else None,
            'all_predictions': predictions,
            'all_errors': errors,
            'all_free_energies': free_energies,
            'total_free_energy': sum(free_energies),
        }
        
        return top_representation, predictions, info


class ActiveInference(nn.Module):
    """
    Active Inference for action selection.
    
    Selects actions that minimize expected free energy:
    1. Epistemic value: Actions that reduce uncertainty
    2. Pragmatic value: Actions that achieve goals
    
    The agent acts to confirm its predictions (minimize surprise).
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        planning_horizon: int = 5,
    ):
        """
        Initialize active inference module.
        
        Args:
            state_dim: State representation dimension
            action_dim: Action space dimension
            hidden_dim: Hidden dimension for networks
            planning_horizon: Number of steps to plan ahead
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.planning_horizon = planning_horizon
        
        # Generative model: predicts next state given current state and action
        self.transition_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        
        # Uncertainty estimator
        self.uncertainty_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus(),  # Positive uncertainty
        )
        
        # Goal/preference network
        self.preference_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Policy network (action selection)
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def compute_expected_free_energy(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute expected free energy of a state-action pair.
        
        EFE = Epistemic value + Pragmatic value
            = Expected information gain + Expected reward
        
        Args:
            state: Current state [batch, state_dim]
            action: Proposed action [batch, action_dim]
        
        Returns:
            Expected free energy [batch]
        """
        # Predict next state
        state_action = torch.cat([state, action], dim=-1)
        next_state_pred = self.transition_model(state_action)
        
        # Estimate uncertainty (epistemic value)
        uncertainty = self.uncertainty_net(state)
        epistemic_value = -uncertainty.sum(dim=-1)  # More uncertainty = less value
        
        # Estimate goal alignment (pragmatic value)
        preference = self.preference_net(next_state_pred).squeeze(-1)
        pragmatic_value = preference
        
        # Expected free energy (lower is better)
        efe = -(epistemic_value + pragmatic_value)
        
        return efe
    
    def select_action(
        self,
        state: torch.Tensor,
        num_samples: int = 10,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Select action that minimizes expected free energy.
        
        Args:
            state: Current state [batch, state_dim]
            num_samples: Number of action samples to evaluate
        
        Returns:
            Tuple of:
                - Selected action [batch, action_dim]
                - Info dict with EFE values
        """
        batch_size = state.shape[0]
        
        # Sample actions from policy
        action_dist = self.policy_net(state)  # [B, action_dim]
        action_mean = action_dist
        
        # Sample multiple actions
        action_std = torch.ones_like(action_mean) * 0.1
        actions = torch.randn(num_samples, batch_size, self.action_dim, device=state.device)
        actions = actions * action_std.unsqueeze(0) + action_mean.unsqueeze(0)
        
        # Evaluate expected free energy for each action
        efes = []
        for i in range(num_samples):
            efe = self.compute_expected_free_energy(state, actions[i])
            efes.append(efe)
        
        efes = torch.stack(efes)  # [num_samples, batch]
        
        # Select action with minimum EFE
        best_action_idx = efes.argmin(dim=0)  # [batch]
        best_action = actions[best_action_idx, torch.arange(batch_size)]
        
        info = {
            'efes': efes,
            'best_efe': efes.min(dim=0)[0],
            'action_mean': action_mean,
        }
        
        return best_action, info
    
    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass: select action via active inference.
        
        Args:
            state: Current state [batch, state_dim]
        
        Returns:
            Tuple of:
                - Selected action [batch, action_dim]
                - Info dict
        """
        return self.select_action(state)


class PredictiveBrainModel(nn.Module):
    """
    Complete predictive brain model integrating:
    1. Hierarchical Predictive Coding
    2. Active Inference
    3. Free Energy Minimization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        action_dim: int,
        num_iterations: int = 5,
        use_precision: bool = True,
    ):
        """
        Initialize predictive brain model.
        
        Args:
            input_dim: Sensory input dimension
            hidden_dims: Hierarchy of hidden dimensions
            action_dim: Action space dimension
            num_iterations: Inference iterations
            use_precision: Use precision weighting
        """
        super().__init__()
        
        # Predictive coding hierarchy (perception)
        self.perception = HierarchicalPredictiveCoding(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_iterations=num_iterations,
            use_precision=use_precision,
        )
        
        # Active inference (action)
        self.action_selection = ActiveInference(
            state_dim=hidden_dims[-1],
            action_dim=action_dim,
            planning_horizon=5,
        )
    
    def forward(
        self,
        sensory_input: torch.Tensor,
        select_action: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass: perceive and optionally act.
        
        Args:
            sensory_input: Sensory input [batch, input_dim]
            select_action: Whether to select an action
        
        Returns:
            Tuple of:
                - State representation [batch, hidden_dims[-1]]
                - Action (if select_action=True) [batch, action_dim]
                - Info dict
        """
        # Perception: hierarchical predictive coding
        state, predictions, perception_info = self.perception(sensory_input)
        
        action = None
        action_info = {}
        
        # Action: active inference (optional)
        if select_action:
            action, action_info = self.action_selection(state)
        
        info = {
            **perception_info,
            **action_info,
            'state': state,
        }
        
        return state, action, info


def compute_free_energy_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    precision: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute free energy loss (variational free energy).
    
    Free Energy = Expected log likelihood + KL divergence
    
    Simplified as: Precision-weighted prediction error
    
    Args:
        predictions: Model predictions [batch, ...]
        targets: Target values [batch, ...]
        precision: Precision weights [batch, ...] (optional)
    
    Returns:
        Free energy loss [scalar]
    """
    # Prediction error
    error = targets - predictions
    
    # Precision weighting
    if precision is not None:
        weighted_error = error * precision
    else:
        weighted_error = error
    
    # Free energy (sum of squared errors)
    free_energy = (weighted_error ** 2).sum() / predictions.shape[0]
    
    return free_energy
