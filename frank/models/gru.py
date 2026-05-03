"""GRU baseline model."""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .base import ModelConfig


class GRUModel(nn.Module):
    """GRU recurrent model for sequence tasks.
    
    Processes step-by-step with persistent state.
    Target: ~500K parameters
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: Optional[ModelConfig] = None
    ):
        super().__init__()
        
        if config is None:
            config = ModelConfig()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 214
        self.n_layers = 2
        
        self.embedding = nn.Embedding(input_dim, config.d_model)
        
        self.gru = nn.GRU(
            input_size=config.d_model,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=config.dropout if self.n_layers > 1 else 0
        )
        
        self.output_proj = nn.Linear(self.hidden_dim, output_dim)
        
        self.hidden_state: Optional[torch.Tensor] = None
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        return_hidden: bool = False
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len)
            hidden: Optional initial hidden state
            return_hidden: Whether to return hidden state
            
        Returns:
            Output logits of shape (batch, seq_len, output_dim)
        """
        batch_size = x.size(0)
        device = x.device
        
        if hidden is None:
            hidden = torch.zeros(
                self.n_layers, batch_size, self.hidden_dim, device=device
            )
        
        embedded = self.embedding(x)
        
        output, hidden = self.gru(embedded, hidden)
        
        self.hidden_state = hidden.detach()
        
        logits = self.output_proj(output)
        
        if return_hidden:
            return logits, hidden
        return logits
    
    def reset_state(self):
        """Reset the hidden state."""
        self.hidden_state = None
