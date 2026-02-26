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
from torch_geometric.nn import MessagePassing
from torch_geometric.data import HeteroData
from typing import Dict, Optional, Tuple, List
import math


class DynamicGraphConstructor(nn.Module):
    """
    自迭代图结构学习器 — Self-iterating Graph Structure Learning.

    用户的"自迭代图结构，模拟复杂系统的自演化"在机器学习文献中对应
    Graph Structure Learning (GSL) 或自适应图卷积网络 (AGCRN, Bai et al. 2020)。

    核心思路：图拓扑不再固定在数据预处理阶段，而是在每个 ST-GCN 层内
    根据当前节点特征动态计算，与预先估计的静态拓扑混合使用。这与神经科学
    中动态功能连接（Dynamic Functional Connectivity, dFC; Hutchison 2013）
    的概念完全对应：大脑的功能连接随认知状态实时重构，而非固定不变。

    算法步骤：
    1. 均值池化时间维度：x[N, T, H] → x_agg[N, H]
    2. 投影 + L2 归一化：e[N, H//2]（无偏置，避免常数偏移影响相似性计算）
    3. 余弦相似度矩阵：sim[N, N] = e @ e.T
    4. Top-k 稀疏化：每节点保留 k 个最强连接（去除自环）
    5. 可学习混合权重 α（sigmoid 约束到 [0,1]）：
       combined = (1-α) × fixed_edges + α × dynamic_edges

    可学习参数：
    - node_proj (H→H//2)：节点嵌入投影
    - mix_logit (scalar)：控制动态 vs 固定拓扑的混合比例

    参考文献：
    - Bai et al. (2020). Adaptive Graph Convolutional Recurrent Network.
    - Hutchison et al. (2013). Dynamic functional connectivity. NeuroImage.
    - Cao et al. (2020). Spectral Temporal Graph Neural Network. NeurIPS.
    """

    def __init__(
        self,
        hidden_channels: int,
        k_neighbors: int = 10,
        mix_alpha: float = 0.3,
    ):
        """
        Args:
            hidden_channels: 节点特征维度（ST-GCN 输出）。
            k_neighbors: 动态图每节点保留的 k 近邻数。
                建议：fMRI 10（200 节点图）；EEG 5（63 节点图）。
            mix_alpha: 初始混合比例（0 = 全静态，1 = 全动态）。
                设为 0.3：以静态拓扑为主（来自全 run 相关估计，统计可靠），
                动态分量 30% 允许模型捕捉认知状态依赖的连接。
        """
        super().__init__()
        self.k = k_neighbors
        # 可学习混合参数：sigmoid(mix_logit) = alpha
        # logit(0.3) ≈ -0.847
        self.mix_logit = nn.Parameter(
            torch.tensor(math.log(mix_alpha / (1.0 - mix_alpha + 1e-8)))
        )
        # 节点嵌入投影（无偏置，保证 L2 归一化后余弦相似度的纯方向语义）
        self.node_proj = nn.Linear(hidden_channels, hidden_channels // 2, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        fixed_edge_index: torch.Tensor,
        fixed_edge_attr: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算混合（动态 + 静态）图拓扑。

        Args:
            x: 当前层节点特征 [N, T, H]
            fixed_edge_index: 预计算静态边 [2, E_fixed]
            fixed_edge_attr: 静态边权重 [E_fixed, 1]，可为 None

        Returns:
            combined_edge_index: [2, E_fixed + E_dyn]
            combined_edge_attr: [E_fixed + E_dyn, 1]
        """
        N = x.shape[0]

        # 1. 时间维度均值池化（跨 T 聚合，得到节点级全局表征）
        x_agg = x.mean(dim=1)  # [N, H]

        # 2. 投影 + L2 归一化（保证余弦相似度等价于点积）
        e = F.normalize(self.node_proj(x_agg), dim=-1)  # [N, H//2]

        # 3. 余弦相似度矩阵 [N, N]（GPU 矩阵乘法，O(N²·H//2)）
        sim = torch.mm(e, e.T)

        # 4. Top-k 稀疏化：每节点保留 k 个最强连接
        #    k+1 是为了包含自环（自身余弦相似度 = 1.0，永远最大），再去掉
        k = min(self.k, N - 1)
        if k < 1:
            # N=1 的退化情况：无法构建任何动态边，直接返回静态拓扑
            fa = fixed_edge_attr if fixed_edge_attr is not None else \
                torch.ones(fixed_edge_index.shape[1], 1, device=x.device)
            return fixed_edge_index, fa

        topk_vals, topk_idx = torch.topk(sim, k + 1, dim=1)  # [N, k+1]
        topk_vals = topk_vals[:, 1:]  # [N, k]，去掉自环
        topk_idx  = topk_idx[:, 1:]   # [N, k]

        # 构建稀疏 edge_index
        src = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, k).reshape(-1)
        dst = topk_idx.reshape(-1)
        dyn_edge_index = torch.stack([src, dst], dim=0)   # [2, N*k]
        dyn_edge_attr  = topk_vals.reshape(-1, 1)         # [N*k, 1]

        # 5. 可学习混合：alpha 控制动态比例
        alpha = torch.sigmoid(self.mix_logit)

        if fixed_edge_attr is None:
            fixed_edge_attr = torch.ones(
                fixed_edge_index.shape[1], 1, device=x.device
            )

        combined_edge_index = torch.cat([fixed_edge_index, dyn_edge_index], dim=1)
        combined_edge_attr  = torch.cat([
            fixed_edge_attr * (1.0 - alpha),
            dyn_edge_attr   * alpha,
        ], dim=0)

        return combined_edge_index, combined_edge_attr


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
        size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [N_src, T, C_in]
            edge_index: Graph edges [2, E]
            edge_attr: Edge weights [E, 1] (optional)
            size: (N_src, N_dst) for cross-modal edges where N_src != N_dst.
                  Must be provided for cross-modal edges so that propagate()
                  allocates the correct number of destination-node slots.
                  Without it, PyG defaults to (N_src, N_src) and aggregation
                  silently produces [N_src, H] instead of [N_dst, H].
            
        Returns:
            out: Updated features [N_dst, T, C_out]  (N_dst = size[1] if given)
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
                # `size` is a plain immutable tuple (or None) captured by the
                # closure — compatible with gradient_checkpoint's non-tensor
                # argument handling under use_reentrant=False.
                def _propagate(xt_s, xo_s, ei, ea):
                    # For bipartite (cross-modal) edges where N_src != N_dst,
                    # PyG requires x to be a (x_src, x_dst) tuple so it can
                    # validate each tensor's size independently.  Provide a
                    # zero-filled destination placeholder; att_dst is inactive
                    # for cross-modal (source-only attention applies).
                    if size is not None and size[0] != size[1]:
                        x_in = (xt_s, xt_s.new_zeros(size[1], xt_s.shape[-1]))
                    else:
                        x_in = xt_s
                    return self.propagate(ei, x=x_in, x_self=xo_s, edge_attr=ea, size=size)
                out_t = gradient_checkpoint(
                    _propagate, x_t_slice, x_orig_slice, edge_index, edge_attr,
                    use_reentrant=False,
                )
            else:
                # Same bipartite-tuple fix for the non-checkpoint path.
                if size is not None and size[0] != size[1]:
                    x_in = (x_t_slice, x_t_slice.new_zeros(size[1], x_t_slice.shape[-1]))
                else:
                    x_in = x_t_slice
                out_t = self.propagate(
                    edge_index,
                    x=x_in,
                    x_self=x_orig_slice,
                    edge_attr=edge_attr,
                    size=size,
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
        use_dynamic_graph: bool = False,
        k_dynamic_neighbors: int = 10,
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
            use_dynamic_graph: Enable self-iterating graph structure learning.
                At each ST-GCN layer, a DynamicGraphConstructor computes a soft
                adjacency from current node features (cosine similarity + top-k)
                and mixes it with the pre-computed fixed edges via a learnable α.
                Implements the "自迭代图结构" concept (AGCRN, Bai et al. 2020).
                Only applies to intra-modal edges (src == dst node type).
            k_dynamic_neighbors: Number of neighbors kept per node in the
                dynamically computed adjacency.
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
        # Use plain nn.ModuleDict (string keys: '__'.join(edge_type)) instead of
        # HeteroConv to avoid PyG's internal to_internal_key() transformations,
        # which can produce key mismatches when accessed with '__'.join(edge_type).
        # HeteroConv.forward() is never called here; we iterate edges manually.
        self.stgcn_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict['__'.join(edge_type)] = SpatialTemporalGraphConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    temporal_kernel_size=temporal_kernel_size,
                    use_attention=True,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    dropout=dropout,
                )
            
            self.stgcn_layers.append(nn.ModuleDict(conv_dict))
        
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

        # 自迭代图结构学习器（每层一个，仅同模态边有效）
        # Cross-modal edges skip dynamic topology: EEG and fMRI are different
        # node types with no natural intra-type similarity to exploit.
        self.use_dynamic_graph = use_dynamic_graph
        if use_dynamic_graph:
            intra_modal_ets = [et for et in edge_types if et[0] == et[2]]
            self.dynamic_constructors = nn.ModuleList([
                nn.ModuleDict({
                    '__'.join(et): DynamicGraphConstructor(
                        hidden_channels=hidden_channels,
                        k_neighbors=k_dynamic_neighbors,
                    )
                    for et in intra_modal_ets
                })
                for _ in range(num_layers)
            ])
    
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

                            # 自迭代图结构：仅对同模态边（src==dst）启用动态拓扑。
                            # 跨模态边（EEG→fMRI）跳过：两者节点类型不同，
                            # 不存在可用于推断连接的"节点间余弦相似性"语义。
                            ei = edge_index
                            ea = edge_attr
                            if (
                                self.use_dynamic_graph
                                and src == dst  # intra-modal only
                                and hasattr(self, 'dynamic_constructors')
                                and layer_idx < len(self.dynamic_constructors)
                                and '__'.join(edge_type) in self.dynamic_constructors[layer_idx]
                            ):
                                ei, ea = self.dynamic_constructors[layer_idx][
                                    '__'.join(edge_type)
                                ](x_src, edge_index, edge_attr)
                            
                            # Apply ST-GCN
                            # Pass size=(N_src, N_dst) explicitly so propagate()
                            # allocates N_dst destination slots.  Without this,
                            # PyG defaults to (N_src, N_src) and cross-modal
                            # aggregation silently produces [N_src, H] instead
                            # of [N_dst, H], corrupting fMRI node features.
                            # stgcn is a plain nn.ModuleDict keyed by
                            # '__'.join(edge_type) — direct string lookup.
                            conv = stgcn['__'.join(edge_type)]
                            N_src = x_src.shape[0]
                            N_dst = x.shape[0]
                            msg = conv(x_src, ei, ea, size=(N_src, N_dst))
                            
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
