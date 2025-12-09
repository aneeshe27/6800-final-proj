"""
Graph Neural Network modules for per-frame graph encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm
from torch_geometric.nn import global_mean_pool, global_max_pool


class PerFrameGNN(nn.Module):
    """
    Per-frame graph encoder that converts a scene graph G_t into an embedding z_t.
    
    Uses GCN layers with residual connections and dual pooling (mean + max).
    
    Architecture:
        Input -> Linear -> [GCN + Norm + ReLU + Dropout + Residual] x N -> Pool -> Output
    """
    
    def __init__(
        self,
        in_channels: int = 9,
        hidden_channels: int = 32,
        out_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.25,
    ):
        """
        Initialize the per-frame GNN.
        
        Args:
            in_channels: Number of input node features.
            hidden_channels: Hidden dimension size.
            out_channels: Output embedding dimension.
            num_layers: Number of GCN layers.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(GraphNorm(hidden_channels))
        
        self.dropout = dropout
        
        # Mean + max pooling -> 2 * hidden_channels
        self.output_proj = nn.Linear(hidden_channels * 2, out_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_channels].
            edge_index: Edge indices [2, num_edges].
            batch: Batch assignment for each node [num_nodes].
            
        Returns:
            Graph embeddings [batch_size, out_channels].
        """
        # Initial projection
        x = F.relu(self.input_proj(x))
        
        # GCN layers with residual connections
        for conv, norm in zip(self.convs, self.norms):
            x_res = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res  # Residual connection
        
        # Dual pooling
        z_mean = global_mean_pool(x, batch)
        z_max = global_max_pool(x, batch)
        z = torch.cat([z_mean, z_max], dim=1)
        
        return self.output_proj(z)

