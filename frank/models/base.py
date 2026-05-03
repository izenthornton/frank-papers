"""Base model utilities and configuration."""
from dataclasses import dataclass
from typing import Optional
import copy
import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Configuration for all models."""
    target_params: int = 500_000
    d_model: int = 128
    d_ff: int = 256
    n_heads: int = 4
    n_layers: int = 4
    hidden_dim: int = 256
    n_modules: int = 4
    tau_values: tuple = (5.0, 15.0, 30.0, 50.0)
    dropout: float = 0.1
    

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def apply_damage(
    model: nn.Module,
    damage_pct: float,
    seed: int
) -> nn.Module:
    """Apply random weight damage by zeroing out weights.
    
    Args:
        model: The model to damage
        damage_pct: Percentage of weights to zero (0-100)
        seed: Random seed for reproducibility
        
    Returns:
        A damaged copy of the model
    """
    torch.manual_seed(seed)
    damaged_model = copy.deepcopy(model)
    
    with torch.no_grad():
        for param in damaged_model.parameters():
            mask = torch.rand_like(param) > (damage_pct / 100.0)
            param.mul_(mask.float())
    
    return damaged_model


FRANK_COMPONENTS = {
    'reflex': 'reflex.',
    'brain': 'brain.',
    'memory': 'memory.',
    'anomaly_detector': 'anomaly_detector.',
}


def apply_targeted_damage(
    model: nn.Module,
    component: str,
    damage_pct: float,
    seed: int
) -> nn.Module:
    """Apply damage only to a specific named component of the model.
    
    Args:
        model: The model to damage (typically FrankModel)
        component: Component name ('reflex', 'brain', 'memory', 'anomaly_detector')
        damage_pct: Percentage of weights to zero (0-100)
        seed: Random seed for reproducibility
        
    Returns:
        A damaged copy of the model
    """
    prefix = FRANK_COMPONENTS.get(component)
    if prefix is None:
        raise ValueError(f"Unknown component: {component}. Choose from {list(FRANK_COMPONENTS.keys())}")

    torch.manual_seed(seed)
    damaged_model = copy.deepcopy(model)

    matched = 0
    with torch.no_grad():
        for name, param in damaged_model.named_parameters():
            if name.startswith(prefix):
                mask = torch.rand_like(param) > (damage_pct / 100.0)
                param.mul_(mask.float())
                matched += 1

    if matched == 0:
        print(f"Warning: No parameters matched component '{component}' (prefix='{prefix}')")

    return damaged_model


class PositionalEncoding(nn.Module):
    """Learned positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Input with positional encoding added
        """
        seq_len = x.size(1)
        max_pos = self.pos_embedding.num_embeddings
        positions = torch.arange(seq_len, device=x.device).clamp(max=max_pos - 1).unsqueeze(0)
        pos_enc = self.pos_embedding(positions)
        return self.dropout(x + pos_enc)


class RecurrentModule(nn.Module):
    """A single recurrent module with learnable time constant."""
    
    def __init__(self, input_dim: int, hidden_dim: int, tau_init: float = 10.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.log_tau = nn.Parameter(torch.log(torch.tensor(tau_init)))
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        
    @property
    def tau(self) -> torch.Tensor:
        return torch.exp(self.log_tau).clamp(min=1.0, max=100.0)
    
    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Single step forward.
        
        Args:
            x: Input of shape (batch, input_dim)
            h: Previous hidden state of shape (batch, hidden_dim)
            
        Returns:
            New hidden state
        """
        batch_size = x.size(0)
        device = x.device
        
        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        alpha = 1.0 / self.tau
        pre_activation = self.input_proj(x) + self.hidden_proj(h)
        new_h = (1 - alpha) * h + alpha * torch.tanh(pre_activation)
        
        return new_h


class LateralConnections(nn.Module):
    """Learnable lateral connections between modules."""
    
    def __init__(self, n_modules: int, hidden_dim: int):
        super().__init__()
        self.n_modules = n_modules
        self.hidden_dim = hidden_dim
        
        self.lateral_weights = nn.Parameter(
            torch.zeros(n_modules, n_modules) * 0.01
        )
        nn.init.xavier_uniform_(self.lateral_weights)
        
    def forward(self, module_states: torch.Tensor) -> torch.Tensor:
        """Apply lateral connections.
        
        Args:
            module_states: Tensor of shape (batch, n_modules, hidden_dim)
            
        Returns:
            States with lateral influences added
        """
        influence = torch.einsum('ij,bjd->bid', self.lateral_weights, module_states)
        return module_states + 0.1 * influence
