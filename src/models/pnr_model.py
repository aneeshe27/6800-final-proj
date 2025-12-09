"""
Full PNR Temporal Model combining GNN and Transformer.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from typing import List, Tuple

from .gnn import PerFrameGNN
from .temporal import TemporalTransformer


class PNRTemporalModel(nn.Module):
    """
    Full PNR detection model.
    
    Pipeline:
        Graph sequence -> Per-frame GNN -> Temporal Transformer -> Distance Head -> PNR Prediction
    
    The model learns to predict the distance of each frame from the PNR frame,
    and the PNR is identified as the frame with the minimum predicted distance.
    """
    
    def __init__(
        self,
        node_feat_dim: int = 9,
        gnn_hidden: int = 32,
        embed_dim: int = 64,
        num_gnn_layers: int = 2,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.25,
        max_seq_len: int = 32,
    ):
        """
        Initialize the PNR model.
        
        Args:
            node_feat_dim: Number of node features in input graphs.
            gnn_hidden: Hidden dimension of GNN.
            embed_dim: Embedding dimension.
            num_gnn_layers: Number of GNN layers.
            num_transformer_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            max_seq_len: Maximum sequence length.
        """
        super().__init__()
        
        self.gnn = PerFrameGNN(
            in_channels=node_feat_dim,
            hidden_channels=gnn_hidden,
            out_channels=embed_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
        )
        
        self.temporal = TemporalTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        
        # Distance regression head (used for PNR classification via argmin)
        self.distance_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )
        
        self.embed_dim = embed_dim
    
    def encode_graphs(self, graphs_list: List[List]) -> List[torch.Tensor]:
        """
        Encode a list of graph sequences into embeddings.
        
        Args:
            graphs_list: List of graph sequences, where each sequence is
                         a list of PyG Data objects.
                         
        Returns:
            List of embedding tensors, each of shape [seq_len, embed_dim].
        """
        batch_embeddings = []
        
        for graphs in graphs_list:
            # Batch all graphs in the sequence
            batched = Batch.from_data_list(graphs)
            
            # Get embeddings for all graphs
            z = self.gnn(batched.x, batched.edge_index, batched.batch)
            batch_embeddings.append(z)
        
        return batch_embeddings
    
    def forward(
        self,
        graphs_list: List[List],
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            graphs_list: List of graph sequences.
            seq_lens: Tensor of sequence lengths [batch_size].
            
        Returns:
            Tuple of:
                - z_tilde: Contextualized embeddings [batch_size, max_len, embed_dim]
                - dist_pred: Distance predictions [batch_size, max_len]
                - mask: Padding mask [batch_size, max_len]
        """
        device = next(self.parameters()).device
        batch_size = len(graphs_list)
        max_len = max(seq_lens).item()
        
        # Encode all graphs
        z_list = self.encode_graphs(graphs_list)
        
        # Pad sequences
        z_padded = torch.zeros(batch_size, max_len, self.embed_dim, device=device)
        mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)
        
        for i, (z, length) in enumerate(zip(z_list, seq_lens)):
            z_padded[i, :length] = z
            mask[i, :length] = False
        
        # Apply temporal transformer
        z_tilde = self.temporal(z_padded, mask=mask)
        
        # Predict distances
        dist_pred = self.distance_head(z_tilde).squeeze(-1)
        
        return z_tilde, dist_pred, mask
    
    def predict_pnr(
        self,
        graphs_list: List[List],
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict PNR frame indices.
        
        Args:
            graphs_list: List of graph sequences.
            seq_lens: Tensor of sequence lengths.
            
        Returns:
            Predicted PNR indices [batch_size].
        """
        _, dist_pred, mask = self.forward(graphs_list, seq_lens)
        
        # Mask padded positions with large values
        dist_pred = dist_pred.clone()
        dist_pred[mask] = float('inf')
        
        # PNR is the frame with minimum predicted distance
        return dist_pred.argmin(dim=1)
    
    @property
    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

