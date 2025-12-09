"""
Temporal Transformer module for sequence modeling.
"""

import torch
import torch.nn as nn
from typing import Optional


class TemporalTransformer(nn.Module):
    """
    Temporal encoder that processes a sequence of frame embeddings.
    
    Converts (z_1, ..., z_T) into context-aware representations (z̃_1, ..., z̃_T).
    
    Uses learnable positional embeddings and standard Transformer encoder layers.
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.25,
        max_seq_len: int = 32,
    ):
        """
        Initialize the temporal transformer.
        
        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dropout: Dropout probability.
            max_seq_len: Maximum sequence length.
        """
        super().__init__()
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        z_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z_seq: Sequence of embeddings [batch_size, seq_len, embed_dim].
            mask: Padding mask [batch_size, seq_len]. True = padded.
            
        Returns:
            Context-aware embeddings [batch_size, seq_len, embed_dim].
        """
        B, T, D = z_seq.shape
        
        # Add positional embeddings
        z_seq = z_seq + self.pos_embedding[:, :T, :]
        
        # Apply transformer
        z_seq = self.transformer(z_seq, src_key_padding_mask=mask)
        
        return self.norm(z_seq)

