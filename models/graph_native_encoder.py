"""
Graph-Native Spatial-Temporal Encoder
======================================

Encoder that works DIRECTLY on graph structure with temporal features.
No graph -> sequence -> graph conversions.

Key innovations:
1. Spatial-Temporal Graph Convolution (ST-GCN)
2. Temporal attention within graph nodes
3. Preserves small-world structure throughout encoding
4. Memory-efficient chunked processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from torch_geometric.nn import MessagePassing, HeteroConv, GCNConv, GATConv
from torch_geometric.data import HeteroData
from typing import Dict, Optional, Tuple, List
import math


class SpatialTemporalGraphConv(MessagePassing):
    """
    Spatial-Temporal Graph Convolution.
    
    Combines:
    1. Spatial message passing (along graph edges)
    2. Temporal convolution (along time axis)
    3. Attention mechanism for adaptive aggregation
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel_size: int = 3,
        use_attention: bool = True,
        use_spectral_norm: bool = True,
        use_gradient_checkpointing: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__(aggr='add')  # Sum aggregation
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.use_spectral_norm = use_spectral_norm
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Temporal convolution (processes time dimension)
        self.temporal_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=temporal_kernel_size,
            padding=temporal_kernel_size // 2,
        )
        
        # Apply spectral normalization to linear layers for training stability
        if use_spectral_norm:
            from torch.nn.utils.parametrizations import spectral_norm
            # Spatial projection (for message passing)
            self.lin_msg = spectral_norm(nn.Linear(out_channels, out_channels))
            self.lin_self = spectral_norm(nn.Linear(in_channels, out_channels))
            
            # Attention mechanism
            if use_attention:
                self.att_src = spectral_norm(nn.Linear(out_channels, 1))
                self.att_dst = spectral_norm(nn.Linear(out_channels, 1))
        else:
            # Spatial projection (for message passing)
            self.lin_msg = nn.Linear(out_channels, out_channels)
            self.lin_self = nn.Linear(in_channels, out_channels)
            
            # Attention mechanism
            if use_attention:
                self.att_src = nn.Linear(out_channels, 1)
                self.att_dst = nn.Linear(out_channels, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        # Xavier initialization for weights
        # Note: spectral_norm wraps the module, so we access .weight directly
        if hasattr(self.lin_msg, 'weight'):
            nn.init.xavier_uniform_(self.lin_msg.weight)
        if hasattr(self.lin_self, 'weight'):
            nn.init.xavier_uniform_(self.lin_self.weight)
        if self.use_attention:
            if hasattr(self.att_src, 'weight'):
                nn.init.xavier_uniform_(self.att_src.weight)
            if hasattr(self.att_dst, 'weight'):
                nn.init.xavier_uniform_(self.att_dst.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [N, T, C_in]
            edge_index: Graph edges [2, E]
            edge_attr: Edge weights [E, 1] (optional)
            
        Returns:
            out: Updated features [N, T, C_out]
        """
        N, T, C_in = x.shape
        
        # 1. Temporal convolution (process time dimension)
        # [N, T, C_in] -> [N, C_in, T]
        x_t = x.permute(0, 2, 1)
        x_t = self.temporal_conv(x_t)  # [N, C_out, T]
        x_t = x_t.permute(0, 2, 1)  # [N, T, C_out]
        
        # 2. Spatial message passing (along graph)
        # Process each timestep independently.
        # Gradient checkpointing frees intermediate activations (attention,
        # messages, etc.) between timesteps, trading recomputation for memory.
        # Without it, all T propagation graphs stay in memory for backprop,
        # causing MemoryError on long sequences.
        out_list = []
        for t in range(T):
            x_t_slice = x_t[:, t, :].contiguous()   # [N, C_out]
            x_orig_slice = x[:, t, :].contiguous()  # [N, C_in]
            
            if self.use_gradient_checkpointing and self.training:
                # Wrap propagate in gradient checkpoint to release intermediate
                # activations immediately; they are recomputed during backward.
                # Pass edge_index and edge_attr explicitly (use_reentrant=False
                # supports non-tensor arguments such as None edge_attr).
                def _propagate(xt_s, xo_s, ei, ea):
                    return self.propagate(ei, x=xt_s, x_self=xo_s, edge_attr=ea)
                out_t = gradient_checkpoint(
                    _propagate, x_t_slice, x_orig_slice, edge_index, edge_attr,
                    use_reentrant=False,
                )
            else:
                out_t = self.propagate(
                    edge_index,
                    x=x_t_slice,
                    x_self=x_orig_slice,
                    edge_attr=edge_attr,
                )
            out_list.append(out_t)
        
        # Stack timesteps
        out = torch.stack(out_list, dim=1)  # [N, T, C_out]
        
        return self.dropout(out)
    
    def message(
        self,
        x_j: torch.Tensor,
        x_i: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> torch.Tensor:
        """
        Construct messages from neighbors.
        
        Args:
            x_j: Source node features [E, C_out]
            x_i: Target node features [E, C_out]
            edge_attr: Edge attributes [E, 1]
            
        Returns:
            messages: [E, C_out]
        """
        # Project neighbor features
        msg = self.lin_msg(x_j)  # [E, C_out]
        
        # Apply edge weights if available
        if edge_attr is not None:
            msg = msg * edge_attr
        
        # Attention mechanism
        if self.use_attention:
            alpha = self.att_src(x_j) + self.att_dst(x_i)  # [E, 1]
            alpha = F.leaky_relu(alpha, 0.2)
            alpha = torch.softmax(alpha, dim=0)  # Normalize per target node
            msg = msg * alpha
        
        return msg
    
    def update(self, aggr_out: torch.Tensor, x_self: torch.Tensor) -> torch.Tensor:
        """
        Update node features with aggregated messages.
        
        Args:
            aggr_out: Aggregated messages [N_dst, C_out]
            x_self: Original node features [N_src, C_in]
            
        Returns:
            Updated features [N_dst, C_out]
        """
        # For cross-modal edges N_src != N_dst.  x_self comes from the SOURCE
        # node tensor, so adding lin_self(x_self=[N_src, C_in]) to
        # aggr_out=[N_dst, C_out] would broadcast the batch dimension and
        # silently produce shape [N_src, C_out] instead of [N_dst, C_out].
        # Skip the self-connection entirely for cross-modal edges; it is only
        # meaningful for same-modal (intra-type) edges where N_src == N_dst.
        if aggr_out.shape[0] != x_self.shape[0]:
            return aggr_out
        
        # Same-modal edge: add self-connection (residual-like)
        return aggr_out + self.lin_self(x_self)


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for graph nodes.
    
    Learns to weight different time points based on their importance.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply temporal attention with Flash Attention optimization.
        
        Uses PyTorch's scaled_dot_product_attention (2.0+) for 2-4x speedup
        and 50% memory reduction. Falls back to standard attention for older versions.
        
        Args:
            x: Node features [N, T, H]
            mask: Attention mask [T, T] (optional)
            
        Returns:
            Attended features [N, T, H]
        """
        N, T, H = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # [N, T, H]
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head: [N, T, H] -> [N, num_heads, T, head_dim]
        Q = Q.view(N, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Try Flash Attention (PyTorch 2.0+), fallback to standard attention
        if hasattr(F, 'scaled_dot_product_attention'):
            # Use Flash Attention - automatically uses optimal kernel based on hardware
            attended = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            # Standard attention fallback for PyTorch < 2.0
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attended = torch.matmul(attn_weights, V)
        
        # Reshape back: [N, num_heads, T, head_dim] -> [N, T, H]
        attended = attended.transpose(1, 2).contiguous().view(N, T, H)
        
        # Output projection
        out = self.out_proj(attended)
        
        return self.dropout(out)


class GraphNativeEncoder(nn.Module):
    """
    Complete graph-native encoder.
    
    Encodes temporal signals on graph structure WITHOUT
    breaking graph into sequences.
    
    Architecture:
    1. Input: HeteroData with temporal node features
    2. Stack of ST-GCN layers (spatial-temporal convolution)
    3. Temporal attention for long-range modeling
    4. Output: Encoded graph with rich spatio-temporal features
    """
    
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        in_channels_dict: Dict[str, int],
        hidden_channels: int = 128,
        num_layers: int = 4,
        temporal_kernel_size: int = 3,
        use_temporal_attention: bool = True,
        attention_heads: int = 4,
        use_gradient_checkpointing: bool = False,
        dropout: float = 0.1,
    ):
        """
        Initialize graph-native encoder.
        
        Args:
            node_types: List of node types (e.g., ['fmri', 'eeg'])
            edge_types: List of edge types (e.g., [('fmri', 'connects', 'fmri')])
            in_channels_dict: Input channels per node type
            hidden_channels: Hidden feature dimension
            num_layers: Number of ST-GCN layers
            temporal_kernel_size: Kernel size for temporal conv
            use_temporal_attention: Use temporal attention mechanism
            attention_heads: Number of attention heads
            use_gradient_checkpointing: Free intermediate activations per timestep
                to avoid MemoryError on long sequences (trades memory for compute)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.use_temporal_attention = use_temporal_attention
        
        # Input projection per node type
        self.input_proj = nn.ModuleDict({
            node_type: nn.Linear(in_channels_dict[node_type], hidden_channels)
            for node_type in node_types
        })
        
        # Stack of ST-GCN layers (heterogeneous)
        self.stgcn_layers = nn.ModuleList()
        for _ in range(num_layers):
            # Create heterogeneous convolution
            conv_dict = {}
            for edge_type in edge_types:
                src, rel, dst = edge_type
                conv_dict[edge_type] = SpatialTemporalGraphConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    temporal_kernel_size=temporal_kernel_size,
                    use_attention=True,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    dropout=dropout,
                )
            
            hetero_conv = HeteroConv(conv_dict, aggr='sum')
            self.stgcn_layers.append(hetero_conv)
        
        # Layer normalization per node type
        self.layer_norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_channels)
                for node_type in node_types
            })
            for _ in range(num_layers)
        ])
        
        # Temporal attention (optional)
        if use_temporal_attention:
            self.temporal_attention = nn.ModuleDict({
                node_type: TemporalAttention(
                    hidden_channels,
                    num_heads=attention_heads,
                    dropout=dropout,
                )
                for node_type in node_types
            })
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data: HeteroData) -> HeteroData:
        """
        Encode graph with temporal features.
        
        Args:
            data: HeteroData with temporal node features
                  Expected: data[node_type].x = [N, T, C_in]
                  
        Returns:
            Encoded HeteroData with features [N, T, H]
        """
        # 1. Input projection
        x_dict = {}
        for node_type in self.node_types:
            if node_type in data.node_types:
                x = data[node_type].x  # [N, T, C_in]
                N, T, C_in = x.shape
                
                # Project each timestep
                x_proj = self.input_proj[node_type](x)  # [N, T, H]
                x_dict[node_type] = x_proj
        
        # 2. Stack of ST-GCN layers
        for layer_idx, (stgcn, layer_norm) in enumerate(zip(self.stgcn_layers, self.layer_norms)):
            # Prepare edge indices and attributes
            edge_index_dict = {}
            edge_attr_dict = {}
            
            for edge_type in self.edge_types:
                if edge_type in data.edge_types:
                    edge_index_dict[edge_type] = data[edge_type].edge_index
                    if hasattr(data[edge_type], 'edge_attr'):
                        edge_attr_dict[edge_type] = data[edge_type].edge_attr
            
            # Apply ST-GCN (heterogeneous convolution)
            x_dict_new = {}
            for node_type, x in x_dict.items():
                # Collect messages from all edge types involving this node type
                messages = []
                
                for edge_type in self.edge_types:
                    src, rel, dst = edge_type
                    if dst == node_type and edge_type in edge_index_dict:
                        # Apply convolution for this edge type
                        if src in x_dict:
                            edge_index = edge_index_dict[edge_type]
                            edge_attr = edge_attr_dict.get(edge_type, None)
                            
                            # Get source features
                            x_src = x_dict[src]
                            
                            # Apply ST-GCN
                            conv = stgcn.convs[edge_type]
                            msg = conv(x_src, edge_index, edge_attr)
                            
                            # Cross-modal edges may have different source T than
                            # destination T (e.g. EEG T=190 vs fMRI T=300).
                            # Resample temporally so all messages share dst's T.
                            T_dst = x.shape[1]
                            if msg.shape[1] != T_dst:
                                msg = F.interpolate(
                                    msg.permute(0, 2, 1),  # [N, H, T_src]
                                    size=T_dst,
                                    mode='linear',
                                    align_corners=False,
                                ).permute(0, 2, 1)  # [N, T_dst, H]
                            
                            messages.append(msg)
                
                # Aggregate messages
                if messages:
                    x_new = sum(messages) / len(messages)
                else:
                    x_new = x
                
                # Residual connection
                x_new = x + self.dropout(x_new)
                
                # Layer normalization (per timestep)
                N, T, H = x_new.shape
                x_new_flat = x_new.view(N * T, H)
                x_new_flat = layer_norm[node_type](x_new_flat)
                x_new = x_new_flat.view(N, T, H)
                
                x_dict_new[node_type] = x_new
            
            x_dict = x_dict_new
        
        # 3. Temporal attention (optional)
        if self.use_temporal_attention:
            for node_type, x in x_dict.items():
                x_attended = self.temporal_attention[node_type](x)
                # Residual connection
                x_dict[node_type] = x + self.dropout(x_attended)
        
        # 4. Update data with encoded features
        encoded_data = data.clone()
        for node_type, x in x_dict.items():
            encoded_data[node_type].x = x
        
        return encoded_data
    
    def get_temporal_pooling(
        self,
        data: HeteroData,
        method: str = 'mean',
    ) -> Dict[str, torch.Tensor]:
        """
        Pool temporal dimension to get static node features.
        
        Useful for tasks that need per-node representations.
        
        Args:
            data: Encoded HeteroData
            method: Pooling method ('mean', 'max', 'last')
            
        Returns:
            Dict of pooled features per node type [N, H]
        """
        pooled_dict = {}
        
        for node_type in self.node_types:
            if node_type in data.node_types:
                x = data[node_type].x  # [N, T, H]
                
                if method == 'mean':
                    pooled = x.mean(dim=1)
                elif method == 'max':
                    pooled = x.max(dim=1)[0]
                elif method == 'last':
                    pooled = x[:, -1, :]
                else:
                    raise ValueError(f"Unknown pooling method: {method}")
                
                pooled_dict[node_type] = pooled
        
        return pooled_dict
