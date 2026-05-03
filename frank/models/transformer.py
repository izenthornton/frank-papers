"""Transformer baseline model."""
import math
import torch
import torch.nn as nn
from typing import Optional

from .base import ModelConfig, PositionalEncoding


class TransformerModel(nn.Module):
    """Transformer encoder model for sequence tasks.
    
    No persistent state between calls - processes full sequence each time.
    Target: ~500K parameters
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_seq_len: int = 200,
        config: Optional[ModelConfig] = None
    ):
        super().__init__()
        
        if config is None:
            config = ModelConfig()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = config.d_model
        
        self.embedding = nn.Embedding(input_dim, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, max_seq_len, config.dropout)

        # Use d_ff=200 (not config default 256) to hit ~500K params
        d_ff = 200
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        
        self.output_proj = nn.Linear(config.d_model, output_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len)
            mask: Optional padding mask
            
        Returns:
            Output logits of shape (batch, seq_len, output_dim)
        """
        embedded = self.embedding(x) * math.sqrt(self.d_model)
        embedded = self.pos_encoder(embedded)
        
        if mask is not None:
            src_key_padding_mask = ~mask.bool()
        else:
            src_key_padding_mask = None
        
        encoded = self.transformer(embedded, src_key_padding_mask=src_key_padding_mask)
        output = self.output_proj(encoded)
        
        return output
    
    def reset_state(self):
        """No state to reset for transformer."""
        pass
