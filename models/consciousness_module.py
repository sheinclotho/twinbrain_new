"""
Consciousness Modeling Module
==============================

Implements consciousness theories for digital twin brain:
1. Global Workspace Theory (GWT) - Information integration and broadcasting
2. Integrated Information Theory (IIT) - Φ (phi) computation
3. Attention Schema Theory - Meta-awareness modeling
4. Dynamic Core Hypothesis - Thalamocortical integration

References:
- Baars, B. J. (1988). A cognitive theory of consciousness.
- Tononi, G. (2004). An information integration theory of consciousness.
- Dehaene, S., & Changeux, J. P. (2011). Experimental and theoretical approaches to conscious processing.
- Graziano, M. S. (2013). Consciousness and the social brain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from typing import Dict, List, Optional, Tuple
import math


class GlobalWorkspaceIntegrator(nn.Module):
    """
    Global Workspace Theory (GWT) implementation.
    
    Models consciousness as a "global workspace" where information from
    specialized processors is integrated and broadcast to the entire system.
    
    Key components:
    1. Local processors (specialized brain regions)
    2. Global workspace (integration hub)
    3. Broadcasting mechanism (attention-based)
    4. Competition for access to workspace
    """
    
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int = 8,
        workspace_dim: int = 256,
        num_workspace_slots: int = 16,
        dropout: float = 0.1,
    ):
        """
        Initialize Global Workspace Integrator.
        
        Args:
            hidden_channels: Dimension of input features
            num_heads: Number of attention heads for multi-head attention
            workspace_dim: Dimension of global workspace
            num_workspace_slots: Number of information slots in workspace
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.workspace_dim = workspace_dim
        self.num_workspace_slots = num_workspace_slots
        self.num_heads = num_heads
        
        # Project local features to workspace dimension
        self.local_to_workspace = nn.Linear(hidden_channels, workspace_dim)
        
        # Learnable workspace slots (similar to memory slots in transformers)
        self.workspace_slots = nn.Parameter(torch.randn(num_workspace_slots, workspace_dim))
        
        # Multi-head attention for integration
        self.integration_attention = nn.MultiheadAttention(
            embed_dim=workspace_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Broadcasting mechanism
        self.broadcast_attention = nn.MultiheadAttention(
            embed_dim=workspace_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Competition mechanism (winner-take-all with softmax)
        self.competition_gate = nn.Sequential(
            nn.Linear(workspace_dim, workspace_dim // 2),
            nn.ReLU(),
            nn.Linear(workspace_dim // 2, 1),
        )
        
        # Output projection
        self.workspace_to_output = nn.Linear(workspace_dim, hidden_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(workspace_dim)
        self.layer_norm2 = nn.LayerNorm(workspace_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through global workspace.
        
        Args:
            x: Input features [batch, num_nodes, hidden_channels]
            mask: Optional attention mask [batch, num_nodes]
        
        Returns:
            Tuple of:
                - Integrated features [batch, num_nodes, hidden_channels]
                - Info dict with workspace metrics
        """
        batch_size, num_nodes, _ = x.shape
        
        # 1. Project local features to workspace dimension
        x_workspace = self.local_to_workspace(x)  # [B, N, workspace_dim]
        
        # 2. Expand workspace slots for batch
        workspace_slots = self.workspace_slots.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [B, num_slots, workspace_dim]
        
        # 3. Integration: Local features compete for workspace slots
        # Use cross-attention: queries=slots, keys/values=local features
        integrated, integration_weights = self.integration_attention(
            query=workspace_slots,
            key=x_workspace,
            value=x_workspace,
            key_padding_mask=mask,
            need_weights=True,
        )  # [B, num_slots, workspace_dim], [B, num_slots, N]
        
        integrated = self.layer_norm1(integrated + workspace_slots)
        integrated = self.dropout(integrated)
        
        # 4. Competition: Select most salient information
        competition_scores = self.competition_gate(integrated).squeeze(-1)  # [B, num_slots]
        competition_probs = F.softmax(competition_scores, dim=-1)  # [B, num_slots]
        
        # Weight workspace slots by competition scores
        integrated_weighted = integrated * competition_probs.unsqueeze(-1)  # [B, num_slots, workspace_dim]
        
        # 5. Broadcasting: Workspace broadcasts to all local processors
        # Use cross-attention: queries=local features, keys/values=workspace
        broadcast, broadcast_weights = self.broadcast_attention(
            query=x_workspace,
            key=integrated_weighted,
            value=integrated_weighted,
            need_weights=True,
        )  # [B, N, workspace_dim], [B, N, num_slots]
        
        broadcast = self.layer_norm2(broadcast + x_workspace)
        broadcast = self.dropout(broadcast)
        
        # 6. Project back to original dimension
        output = self.workspace_to_output(broadcast)  # [B, N, hidden_channels]
        
        # Compute consciousness metrics
        info = {
            'integration_weights': integration_weights,  # Which local features accessed workspace
            'broadcast_weights': broadcast_weights,      # How workspace influenced local features
            'competition_probs': competition_probs,      # Which slots won competition
            'workspace_content': integrated_weighted,    # Content of global workspace
        }
        
        return output, info


class IntegratedInformationCalculator(nn.Module):
    """
    Integrated Information Theory (IIT) Φ (Phi) calculator.
    
    Computes integrated information as a measure of consciousness.
    Φ measures how much a system is "more than the sum of its parts".
    
    Simplified approximation for computational efficiency:
    - Uses graph connectivity and information flow
    - Approximates system partitioning with spectral clustering
    - Computes effective information across partitions
    """
    
    def __init__(
        self,
        hidden_channels: int,
        num_partitions: int = 4,
    ):
        """
        Initialize IIT Phi calculator.
        
        Args:
            hidden_channels: Feature dimension
            num_partitions: Number of partitions for MIP computation
        """
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_partitions = num_partitions
        
        # Networks for computing effective information
        self.cause_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
        )
        
        self.effect_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
        )
    
    def compute_effective_information(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute effective information in the system.
        
        Args:
            x: Node features [num_nodes, hidden_channels]
            edge_index: Graph edges [2, num_edges]
        
        Returns:
            Effective information score [scalar]
        """
        num_nodes = x.shape[0]
        
        # Compute cause repertoire (past states)
        causes = self.cause_network(x)  # [N, C]
        
        # Compute effect repertoire (future states)
        effects = self.effect_network(x)  # [N, C]
        
        # Measure information flow across edges
        src, dst = edge_index[0], edge_index[1]
        
        # Cause-effect relationships across edges
        cause_src = causes[src]  # [E, C]
        effect_dst = effects[dst]  # [E, C]
        
        # Effective information = mutual information between causes and effects
        # Approximated by cosine similarity (normalized correlation)
        ei = F.cosine_similarity(cause_src, effect_dst, dim=-1)  # [E]
        
        # Average over all edges
        effective_info = ei.mean()
        
        return effective_info
    
    def compute_phi(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute integrated information Φ (phi).
        
        Φ = EI(whole) - max_partition(EI(partition))
        
        Args:
            x: Node features [num_nodes, hidden_channels]
            edge_index: Graph edges [2, num_edges]
        
        Returns:
            Tuple of:
                - Phi value [scalar]
                - Info dict with detailed metrics
        """
        # Compute effective information of whole system
        ei_whole = self.compute_effective_information(x, edge_index)
        
        # Approximate minimum information partition (MIP)
        # For efficiency, we use a simple random partitioning approximation
        # In practice, exhaustive search is computationally prohibitive
        
        num_nodes = x.shape[0]
        partition_size = num_nodes // self.num_partitions
        
        min_partition_ei = float('inf')
        
        # Try a few random partitions
        for _ in range(5):
            # Random partition
            perm = torch.randperm(num_nodes)
            partition_eis = []
            
            for i in range(self.num_partitions):
                start = i * partition_size
                end = start + partition_size if i < self.num_partitions - 1 else num_nodes
                partition_nodes = perm[start:end]
                
                # Get edges within this partition
                mask = torch.isin(edge_index[0], partition_nodes) & torch.isin(edge_index[1], partition_nodes)
                partition_edges = edge_index[:, mask]
                
                if partition_edges.shape[1] > 0:
                    # Remap node indices to local partition
                    node_mapping = {node.item(): i for i, node in enumerate(partition_nodes)}
                    partition_edges_remapped = torch.tensor([
                        [node_mapping[edge_index[0, i].item()], node_mapping[edge_index[1, i].item()]]
                        for i in range(partition_edges.shape[1])
                        if edge_index[0, i].item() in node_mapping and edge_index[1, i].item() in node_mapping
                    ], dtype=torch.long).T
                    
                    if partition_edges_remapped.shape[1] > 0:
                        partition_ei = self.compute_effective_information(
                            x[partition_nodes], partition_edges_remapped
                        )
                        partition_eis.append(partition_ei)
            
            if len(partition_eis) > 0:
                avg_partition_ei = torch.stack(partition_eis).mean()
                min_partition_ei = min(min_partition_ei, avg_partition_ei.item())
        
        # Φ = information lost by partitioning
        if min_partition_ei == float('inf'):
            phi = ei_whole
        else:
            phi = ei_whole - min_partition_ei
        
        # Ensure phi is non-negative
        phi = torch.clamp(phi, min=0.0)
        
        info = {
            'phi': phi,
            'ei_whole': ei_whole,
            'min_partition_ei': torch.tensor(min_partition_ei),
        }
        
        return phi, info


class ConsciousnessStateClassifier(nn.Module):
    """
    Classifies consciousness states based on brain activity.
    
    States:
    - Wakefulness
    - REM sleep
    - Non-REM sleep
    - Anesthesia
    - Coma
    - Vegetative state
    - Minimally conscious state
    """
    
    def __init__(
        self,
        hidden_channels: int,
        num_states: int = 7,
        dropout: float = 0.1,
    ):
        """
        Initialize consciousness state classifier.
        
        Args:
            hidden_channels: Feature dimension
            num_states: Number of consciousness states
            dropout: Dropout rate
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_states),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify consciousness state.
        
        Args:
            x: Aggregated brain features [batch, hidden_channels]
        
        Returns:
            State logits [batch, num_states]
        """
        return self.classifier(x)


class ConsciousnessModule(nn.Module):
    """
    Complete consciousness modeling module.
    
    Integrates:
    1. Global Workspace Theory (information integration)
    2. Integrated Information Theory (Φ computation)
    3. Consciousness state classification
    """
    
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int = 8,
        workspace_dim: int = 256,
        num_workspace_slots: int = 16,
        num_partitions: int = 4,
        num_consciousness_states: int = 7,
        dropout: float = 0.1,
    ):
        """
        Initialize consciousness module.
        
        Args:
            hidden_channels: Feature dimension
            num_heads: Number of attention heads
            workspace_dim: Global workspace dimension
            num_workspace_slots: Number of workspace slots
            num_partitions: Number of partitions for Φ computation
            num_consciousness_states: Number of consciousness states
            dropout: Dropout rate
        """
        super().__init__()
        
        self.gwt = GlobalWorkspaceIntegrator(
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            workspace_dim=workspace_dim,
            num_workspace_slots=num_workspace_slots,
            dropout=dropout,
        )
        
        self.iit = IntegratedInformationCalculator(
            hidden_channels=hidden_channels,
            num_partitions=num_partitions,
        )
        
        self.state_classifier = ConsciousnessStateClassifier(
            hidden_channels=hidden_channels,
            num_states=num_consciousness_states,
            dropout=dropout,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through consciousness module.
        
        Args:
            x: Node features [batch, num_nodes, hidden_channels]
            edge_index: Graph edges [2, num_edges]
            mask: Optional attention mask [batch, num_nodes]
        
        Returns:
            Tuple of:
                - Integrated features [batch, num_nodes, hidden_channels]
                - Info dict with consciousness metrics
        """
        batch_size, num_nodes, hidden_channels = x.shape
        
        # 1. Global workspace integration
        x_integrated, gwt_info = self.gwt(x, mask)
        
        # 2. Compute Φ (phi) for each sample in batch
        phi_values = []
        for i in range(batch_size):
            phi, iit_info = self.iit.compute_phi(x[i], edge_index)
            phi_values.append(phi)
        
        phi_batch = torch.stack(phi_values)  # [batch]
        
        # 3. Aggregate features for state classification
        x_aggregated = x_integrated.mean(dim=1)  # [batch, hidden_channels]
        
        # 4. Classify consciousness state
        state_logits = self.state_classifier(x_aggregated)  # [batch, num_states]
        
        # Combine all metrics
        info = {
            **gwt_info,
            'phi': phi_batch,
            'state_logits': state_logits,
            'consciousness_level': phi_batch / (phi_batch.max() + 1e-8),  # Normalized [0, 1]
        }
        
        return x_integrated, info


# Consciousness state names for interpretation
CONSCIOUSNESS_STATES = [
    'wakefulness',
    'rem_sleep',
    'nrem_sleep',
    'anesthesia',
    'coma',
    'vegetative_state',
    'minimally_conscious',
]
