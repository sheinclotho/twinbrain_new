"""
Advanced Attention Mechanisms for Brain Modeling
=================================================

Implements state-of-the-art attention mechanisms for multimodal brain data:
1. Cross-Modal Attention (EEG ↔ fMRI)
2. Spatial-Temporal Attention
3. Graph Attention with edge features
4. Hierarchical Attention
5. Self-supervised contrastive attention

References:
- Vaswani et al. (2017). Attention is All You Need.
- Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers.
- Velickovic et al. (2018). Graph Attention Networks.
- Lee et al. (2019). Set Transformer.
- Chen et al. (2020). A Simple Framework for Contrastive Learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from typing import Dict, List, Optional, Tuple
import math


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention between EEG and fMRI.
    
    Enables bidirectional information flow:
    - EEG → fMRI: High temporal resolution informs spatial patterns
    - fMRI → EEG: High spatial resolution informs temporal dynamics
    
    Uses cross-attention mechanism where one modality queries the other.
    """
    
    def __init__(
        self,
        eeg_channels: int,
        fmri_channels: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize cross-modal attention.
        
        Args:
            eeg_channels: EEG feature dimension
            fmri_channels: fMRI feature dimension
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.eeg_channels = eeg_channels
        self.fmri_channels = fmri_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # EEG → fMRI attention
        self.eeg_to_fmri_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # fMRI → EEG attention
        self.fmri_to_eeg_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Projections
        self.eeg_proj = nn.Linear(eeg_channels, hidden_dim)
        self.fmri_proj = nn.Linear(fmri_channels, hidden_dim)
        
        # Output projections
        self.eeg_out_proj = nn.Linear(hidden_dim, eeg_channels)
        self.fmri_out_proj = nn.Linear(hidden_dim, fmri_channels)
        
        # Layer normalization
        self.eeg_norm1 = nn.LayerNorm(hidden_dim)
        self.eeg_norm2 = nn.LayerNorm(eeg_channels)
        self.fmri_norm1 = nn.LayerNorm(hidden_dim)
        self.fmri_norm2 = nn.LayerNorm(fmri_channels)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        eeg_features: torch.Tensor,
        fmri_features: torch.Tensor,
        eeg_mask: Optional[torch.Tensor] = None,
        fmri_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply cross-modal attention.
        
        Args:
            eeg_features: EEG features [batch, n_eeg, eeg_channels]
            fmri_features: fMRI features [batch, n_fmri, fmri_channels]
            eeg_mask: Optional mask for EEG [batch, n_eeg]
            fmri_mask: Optional mask for fMRI [batch, n_fmri]
        
        Returns:
            Tuple of:
                - Enhanced EEG features [batch, n_eeg, eeg_channels]
                - Enhanced fMRI features [batch, n_fmri, fmri_channels]
                - Info dict with attention weights
        """
        # Project to common hidden dimension
        eeg_hidden = self.eeg_proj(eeg_features)  # [B, N_eeg, H]
        fmri_hidden = self.fmri_proj(fmri_features)  # [B, N_fmri, H]
        
        # EEG queries fMRI (EEG learns from fMRI's spatial patterns)
        eeg_enhanced, eeg_to_fmri_weights = self.eeg_to_fmri_attention(
            query=eeg_hidden,
            key=fmri_hidden,
            value=fmri_hidden,
            key_padding_mask=fmri_mask,
            need_weights=True,
        )  # [B, N_eeg, H], [B, N_eeg, N_fmri]
        
        eeg_enhanced = self.eeg_norm1(eeg_enhanced + eeg_hidden)
        eeg_enhanced = self.dropout(eeg_enhanced)
        
        # fMRI queries EEG (fMRI learns from EEG's temporal dynamics)
        fmri_enhanced, fmri_to_eeg_weights = self.fmri_to_eeg_attention(
            query=fmri_hidden,
            key=eeg_hidden,
            value=eeg_hidden,
            key_padding_mask=eeg_mask,
            need_weights=True,
        )  # [B, N_fmri, H], [B, N_fmri, N_eeg]
        
        fmri_enhanced = self.fmri_norm1(fmri_enhanced + fmri_hidden)
        fmri_enhanced = self.dropout(fmri_enhanced)
        
        # Project back to original dimensions
        eeg_out = self.eeg_out_proj(eeg_enhanced)  # [B, N_eeg, C_eeg]
        fmri_out = self.fmri_out_proj(fmri_enhanced)  # [B, N_fmri, C_fmri]
        
        # Residual connections
        eeg_out = self.eeg_norm2(eeg_out + eeg_features)
        fmri_out = self.fmri_norm2(fmri_out + fmri_features)
        
        info = {
            'eeg_to_fmri_weights': eeg_to_fmri_weights,
            'fmri_to_eeg_weights': fmri_to_eeg_weights,
        }
        
        return eeg_out, fmri_out, info


class SpatialTemporalAttention(nn.Module):
    """
    Joint spatial-temporal attention for brain graphs.
    
    Learns to attend to important:
    - Spatial locations (brain regions)
    - Temporal moments (time points)
    
    Uses factorized attention for efficiency.
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize spatial-temporal attention.
        
        Args:
            channels: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        
        # Spatial attention (across nodes)
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Temporal attention (across time)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer normalization
        self.spatial_norm = nn.LayerNorm(channels)
        self.temporal_norm = nn.LayerNorm(channels)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply spatial-temporal attention.
        
        Args:
            x: Input features [batch, num_nodes, time_steps, channels]
            spatial_mask: Optional spatial mask [batch, num_nodes]
            temporal_mask: Optional temporal mask [batch, time_steps]
        
        Returns:
            Tuple of:
                - Attended features [batch, num_nodes, time_steps, channels]
                - Info dict with attention weights
        """
        batch_size, num_nodes, time_steps, channels = x.shape
        
        # 1. Spatial attention (attend over nodes at each time step)
        # Reshape to [B * T, N, C]
        x_spatial = x.permute(0, 2, 1, 3).reshape(batch_size * time_steps, num_nodes, channels)
        
        x_spatial_att, spatial_weights = self.spatial_attention(
            query=x_spatial,
            key=x_spatial,
            value=x_spatial,
            key_padding_mask=spatial_mask.repeat(time_steps, 1) if spatial_mask is not None else None,
            need_weights=True,
        )
        
        x_spatial_att = self.spatial_norm(x_spatial_att + x_spatial)
        x_spatial_att = self.dropout(x_spatial_att)
        
        # Reshape back to [B, N, T, C]
        x_spatial_att = x_spatial_att.reshape(batch_size, time_steps, num_nodes, channels).permute(0, 2, 1, 3)
        
        # 2. Temporal attention (attend over time steps at each node)
        # Reshape to [B * N, T, C]
        x_temporal = x_spatial_att.permute(0, 1, 2, 3).reshape(batch_size * num_nodes, time_steps, channels)
        
        x_temporal_att, temporal_weights = self.temporal_attention(
            query=x_temporal,
            key=x_temporal,
            value=x_temporal,
            key_padding_mask=temporal_mask.repeat(num_nodes, 1) if temporal_mask is not None else None,
            need_weights=True,
        )
        
        x_temporal_att = self.temporal_norm(x_temporal_att + x_temporal)
        x_temporal_att = self.dropout(x_temporal_att)
        
        # Reshape back to [B, N, T, C]
        x_out = x_temporal_att.reshape(batch_size, num_nodes, time_steps, channels)
        
        info = {
            'spatial_weights': spatial_weights.reshape(batch_size, time_steps, num_nodes, num_nodes).mean(dim=1),
            'temporal_weights': temporal_weights.reshape(batch_size, num_nodes, time_steps, time_steps).mean(dim=1),
        }
        
        return x_out, info


class GraphAttentionWithEdges(MessagePassing):
    """
    Graph Attention Network (GAT) with edge features.
    
    Extends standard GAT to incorporate edge attributes in attention computation.
    Useful for brain graphs where edge weights represent connectivity strength.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int = 1,
        num_heads: int = 1,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ):
        """
        Initialize graph attention with edge features.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            edge_dim: Edge feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            negative_slope: LeakyReLU negative slope
        """
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        
        # Linear transformations
        self.lin = nn.Linear(in_channels, num_heads * out_channels, bias=False)
        self.lin_edge = nn.Linear(edge_dim, num_heads * out_channels, bias=False)
        
        # Attention parameters
        self.att_src = nn.Parameter(torch.Tensor(1, num_heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, num_heads, out_channels))
        self.att_edge = nn.Parameter(torch.Tensor(1, num_heads, out_channels))
        
        self.bias = nn.Parameter(torch.Tensor(num_heads * out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.att_edge)
        nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
        
        Returns:
            Updated node features [num_nodes, num_heads * out_channels]
        """
        # Transform node features
        x = self.lin(x).view(-1, self.num_heads, self.out_channels)
        
        # Transform edge features if provided
        if edge_attr is not None:
            edge_attr = self.lin_edge(edge_attr).view(-1, self.num_heads, self.out_channels)
        else:
            edge_attr = torch.zeros(
                edge_index.shape[1], self.num_heads, self.out_channels,
                device=x.device, dtype=x.dtype
            )
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Add bias and reshape
        out = out.view(-1, self.num_heads * self.out_channels) + self.bias
        
        return out
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> torch.Tensor:
        """
        Compute messages with attention.
        
        Args:
            x_i: Target node features [num_edges, num_heads, out_channels]
            x_j: Source node features [num_edges, num_heads, out_channels]
            edge_attr: Edge features [num_edges, num_heads, out_channels]
            index: Target node indices
            ptr: Batch pointer (for batched graphs)
            size_i: Number of target nodes
        
        Returns:
            Weighted messages [num_edges, num_heads, out_channels]
        """
        # Compute attention coefficients
        # alpha = (x_i || x_j || edge_attr) · att
        alpha = (x_i * self.att_dst).sum(dim=-1)  # [E, H]
        alpha += (x_j * self.att_src).sum(dim=-1)  # [E, H]
        alpha += (edge_attr * self.att_edge).sum(dim=-1)  # [E, H]
        
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Weight messages by attention
        return x_j * alpha.unsqueeze(-1)


class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention for multi-scale brain representations.
    
    Attends at multiple levels:
    1. Local (within brain region)
    2. Regional (across neighboring regions)
    3. Global (across entire brain)
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize hierarchical attention.
        
        Args:
            channels: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Local attention (self-attention)
        self.local_attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Regional attention (neighborhood aggregation)
        self.regional_attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Global attention (full brain)
        self.global_attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Hierarchical combination
        self.combination = nn.Sequential(
            nn.Linear(channels * 3, channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )
        
        self.layer_norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        local_mask: Optional[torch.Tensor] = None,
        regional_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply hierarchical attention.
        
        Args:
            x: Input features [batch, num_nodes, channels]
            local_mask: Local attention mask [batch, num_nodes]
            regional_mask: Regional attention mask [batch, num_nodes]
        
        Returns:
            Tuple of:
                - Hierarchically attended features [batch, num_nodes, channels]
                - Info dict with attention weights
        """
        # Local attention (node attends to itself)
        local_out, local_weights = self.local_attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=local_mask,
            need_weights=True,
        )
        
        # Regional attention (node attends to neighbors)
        regional_out, regional_weights = self.regional_attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=regional_mask,
            need_weights=True,
        )
        
        # Global attention (node attends to all nodes)
        global_out, global_weights = self.global_attention(
            query=x,
            key=x,
            value=x,
            need_weights=True,
        )
        
        # Combine hierarchical representations
        combined = torch.cat([local_out, regional_out, global_out], dim=-1)
        combined = self.combination(combined)
        
        # Residual and normalization
        out = self.layer_norm(combined + x)
        out = self.dropout(out)
        
        info = {
            'local_weights': local_weights,
            'regional_weights': regional_weights,
            'global_weights': global_weights,
        }
        
        return out, info


class ContrastiveAttention(nn.Module):
    """
    Self-supervised contrastive attention learning.
    
    Learns to align similar brain states and separate dissimilar ones.
    Uses InfoNCE loss for contrastive learning.
    """
    
    def __init__(
        self,
        channels: int,
        projection_dim: int = 128,
        temperature: float = 0.07,
    ):
        """
        Initialize contrastive attention.
        
        Args:
            channels: Feature dimension
            projection_dim: Projection dimension for contrastive learning
            temperature: Temperature for softmax in InfoNCE loss
        """
        super().__init__()
        
        self.channels = channels
        self.projection_dim = projection_dim
        self.temperature = temperature
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, projection_dim),
        )
        
        # Attention weights for positive/negative pairs
        self.attention = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, 1),
        )
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute contrastive loss and attention.
        
        Args:
            x1: First view [batch, num_nodes, channels]
            x2: Second view [batch, num_nodes, channels]
        
        Returns:
            Tuple of:
                - Contrastive loss [scalar]
                - Info dict with similarity matrix
        """
        batch_size = x1.shape[0]
        
        # Aggregate features (mean pooling)
        x1_agg = x1.mean(dim=1)  # [B, C]
        x2_agg = x2.mean(dim=1)  # [B, C]
        
        # Project to contrastive space
        z1 = self.projection(x1_agg)  # [B, projection_dim]
        z2 = self.projection(x2_agg)  # [B, projection_dim]
        
        # Normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(z1, z2.T) / self.temperature  # [B, B]
        
        # InfoNCE loss (positive pairs on diagonal)
        labels = torch.arange(batch_size, device=x1.device)
        loss = F.cross_entropy(similarity, labels)
        
        # Compute attention weights
        attention_weights1 = self.attention(x1).squeeze(-1)  # [B, N]
        attention_weights2 = self.attention(x2).squeeze(-1)  # [B, N]
        
        attention_weights1 = F.softmax(attention_weights1, dim=-1)
        attention_weights2 = F.softmax(attention_weights2, dim=-1)
        
        info = {
            'contrastive_loss': loss,
            'similarity_matrix': similarity,
            'attention_weights1': attention_weights1,
            'attention_weights2': attention_weights2,
        }
        
        return loss, info
