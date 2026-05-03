"""Modular Recurrent baseline model."""
import torch
import torch.nn as nn
from typing import Optional, List

from .base import ModelConfig, RecurrentModule, LateralConnections


class ModularRecurrentModel(nn.Module):
    """Modular recurrent model with multiple time constants.
    
    Has 4 recurrent modules with different tau values and lateral connections.
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
        self.n_modules = config.n_modules
        self.module_hidden_dim = 209
        
        self.embedding = nn.Embedding(input_dim, config.d_model)
        
        self.recurrent_modules = nn.ModuleList([
            RecurrentModule(
                input_dim=config.d_model,
                hidden_dim=self.module_hidden_dim,
                tau_init=tau
            )
            for tau in config.tau_values
        ])
        
        self.lateral = LateralConnections(self.n_modules, self.module_hidden_dim)
        
        total_hidden = self.n_modules * self.module_hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(total_hidden, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, output_dim)
        )
        
        self.hidden_states: Optional[List[torch.Tensor]] = None
        
    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len)
            hidden_states: Optional initial hidden states for each module
            
        Returns:
            Output logits of shape (batch, seq_len, output_dim)
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        if hidden_states is None:
            hidden_states = [
                torch.zeros(batch_size, self.module_hidden_dim, device=device)
                for _ in range(self.n_modules)
            ]
        
        embedded = self.embedding(x)
        
        outputs = []
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            
            new_states = []
            for i, module in enumerate(self.recurrent_modules):
                h = module(x_t, hidden_states[i])
                new_states.append(h)
            
            stacked = torch.stack(new_states, dim=1)
            stacked = self.lateral(stacked)
            hidden_states = [stacked[:, i, :] for i in range(self.n_modules)]
            
            combined = torch.cat(hidden_states, dim=-1)
            output = self.output_proj(combined)
            outputs.append(output)
        
        self.hidden_states = [h.detach() for h in hidden_states]
        
        return torch.stack(outputs, dim=1)
    
    def reset_state(self):
        """Reset the hidden states."""
        self.hidden_states = None
        
    def get_tau_values(self) -> List[float]:
        """Get current tau values for all modules."""
        return [float(m.tau.item()) for m in self.recurrent_modules]
