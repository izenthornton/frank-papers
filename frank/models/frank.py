"""Frank: The hybrid biologically-inspired architecture."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from .base import ModelConfig, RecurrentModule, LateralConnections


@dataclass
class FrankDiagnostics:
    """Diagnostics for Frank model behavior."""
    inhibit_rate: float = 0.0
    memory_injection_rate: float = 0.0
    mean_injection_magnitude: float = 0.0
    reflex_accuracy: float = 0.0
    brain_accuracy: float = 0.0
    tau_values: List[float] = None
    
    def __post_init__(self):
        if self.tau_values is None:
            self.tau_values = []


class Reflex(nn.Module):
    """Reflex pathway: stateless pattern matcher (~100K params)."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input of shape (batch, input_dim)
            
        Returns:
            Output logits of shape (batch, output_dim)
        """
        return self.mlp(x)


class Brain(nn.Module):
    """Brain: modular recurrent network (~320K params)."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_modules: int = 4,
        module_hidden_dim: int = 160,
        tau_values: Tuple[float, ...] = (5.0, 15.0, 30.0, 50.0)
    ):
        super().__init__()
        self.n_modules = n_modules
        self.module_hidden_dim = module_hidden_dim
        
        self.recurrent_modules = nn.ModuleList([
            RecurrentModule(input_dim, module_hidden_dim, tau)
            for tau in tau_values
        ])
        
        self.lateral = LateralConnections(n_modules, module_hidden_dim)
        
        total_hidden = n_modules * module_hidden_dim
        self.output_proj = nn.Linear(total_hidden, output_dim)
        self.inhibit_proj = nn.Linear(total_hidden, 1)
        
        self.hidden_states: Optional[List[torch.Tensor]] = None
        
    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]] = None,
        injection: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input of shape (batch, input_dim)
            hidden_states: Previous hidden states
            injection: Memory injection to add to first module
            
        Returns:
            Tuple of (output logits, inhibit signal, combined hidden state)
        """
        batch_size = x.size(0)
        device = x.device
        
        if hidden_states is None:
            hidden_states = [
                torch.zeros(batch_size, self.module_hidden_dim, device=device)
                for _ in range(self.n_modules)
            ]
        
        new_states = []
        for i, module in enumerate(self.recurrent_modules):
            h = module(x, hidden_states[i])
            if i == 0 and injection is not None:
                h = h + injection
            new_states.append(h)
        
        stacked = torch.stack(new_states, dim=1)
        stacked = self.lateral(stacked)
        self.hidden_states = [stacked[:, i, :] for i in range(self.n_modules)]
        
        combined = torch.cat(self.hidden_states, dim=-1)
        
        output = self.output_proj(combined)
        inhibit = torch.sigmoid(self.inhibit_proj(combined))
        
        return output, inhibit, combined
    
    def get_tau_values(self) -> List[float]:
        """Get current tau values."""
        return [float(m.tau.item()) for m in self.recurrent_modules]
    
    def reset_state(self):
        """Reset hidden states."""
        self.hidden_states = None


class ActiveMemory(nn.Module):
    """Active Memory: content-addressable memory that injects without query (~60K params)."""
    
    def __init__(
        self,
        brain_dim: int,
        n_slots: int = 64,
        key_dim: int = 640,
        value_dim: int = 64
    ):
        super().__init__()
        self.n_slots = n_slots
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        self.keys = nn.Parameter(torch.randn(n_slots, key_dim) * 0.1)
        self.values = nn.Parameter(torch.randn(n_slots, value_dim) * 0.1)
        
        self.relevance_net = nn.Sequential(
            nn.Linear(brain_dim, 128),
            nn.ReLU(),
            nn.Linear(128, key_dim)
        )
        
        self.output_proj = nn.Linear(value_dim, brain_dim // 4)
        
        self.last_injection_magnitude: float = 0.0
        self.last_did_inject: bool = False
        
    def forward(self, brain_state: torch.Tensor) -> torch.Tensor:
        """Compute memory injection based on brain state.
        
        Args:
            brain_state: Combined brain hidden state (batch, brain_dim)
            
        Returns:
            Injection to add to brain (batch, module_dim)
        """
        query = self.relevance_net(brain_state)
        
        scores = torch.matmul(query, self.keys.t()) / (self.key_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        
        retrieved = torch.matmul(attention, self.values)
        
        injection = self.output_proj(retrieved)
        
        self.last_injection_magnitude = float(injection.abs().mean().item())
        self.last_did_inject = self.last_injection_magnitude > 0.01
        
        return injection


class AnomalyDetector(nn.Module):
    """Anomaly Detector: learns when input is unusual (~20K params)."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score.
        
        Args:
            x: Input of shape (batch, input_dim)
            
        Returns:
            Anomaly score in [0, 1]
        """
        return self.mlp(x)


class FrankModel(nn.Module):
    """Frank: Hybrid architecture combining reflex, brain, and active memory.
    
    Components:
    - Reflex: Fast stateless pattern matcher (~100K params)
    - Brain: Modular recurrent network with time constants (~320K params)
    - Active Memory: Content-addressable memory with injection (~60K params)
    - Anomaly Detector: Detects unusual inputs (~20K params)
    
    Target: ~500K total parameters
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
        
        self.embedding = nn.Embedding(input_dim, config.d_model)
        
        self.reflex = Reflex(config.d_model, output_dim, hidden_dim=128)
        
        n_modules = 4
        module_hidden_dim = 180
        brain_dim = n_modules * module_hidden_dim
        
        self.brain = Brain(
            input_dim=config.d_model,
            output_dim=output_dim,
            n_modules=n_modules,
            module_hidden_dim=module_hidden_dim,
            tau_values=config.tau_values
        )
        
        self.memory = ActiveMemory(
            brain_dim=brain_dim,
            n_slots=64,
            key_dim=640,
            value_dim=64
        )
        
        self.anomaly_detector = AnomalyDetector(config.d_model, hidden_dim=64)
        
        self.diagnostics = FrankDiagnostics()
        self._inhibit_count = 0
        self._total_count = 0
        self._injection_count = 0
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable training."""
        for name, param in self.named_parameters():
            if 'embedding' in name:
                nn.init.uniform_(param, -0.1, 0.1)
            elif 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        if hasattr(self.brain, 'inhibit_proj'):
            nn.init.zeros_(self.brain.inhibit_proj.weight)
            nn.init.constant_(self.brain.inhibit_proj.bias, -1.0)
        
    def forward(
        self,
        x: torch.Tensor,
        collect_diagnostics: bool = False
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len)
            collect_diagnostics: Whether to collect diagnostic information
            
        Returns:
            Output logits of shape (batch, seq_len, output_dim)
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        embedded = self.embedding(x)
        
        outputs = []
        prev_brain_state = None
        
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            
            injection = None
            if prev_brain_state is not None:
                injection = self.memory(prev_brain_state)
            
            brain_out, inhibit, brain_state = self.brain(
                x_t,
                self.brain.hidden_states,
                injection=injection
            )
            prev_brain_state = brain_state
            
            reflex_out = self.reflex(x_t)
            
            blend = inhibit
            output = blend * brain_out + (1 - blend) * reflex_out
            
            outputs.append(output)
            
            if collect_diagnostics:
                self._total_count += batch_size
                self._inhibit_count += (inhibit > 0.5).sum().item()
                if self.memory.last_did_inject:
                    self._injection_count += batch_size
        
        return torch.stack(outputs, dim=1)
    
    def reset_state(self):
        """Reset all stateful components."""
        self.brain.reset_state()
        self._inhibit_count = 0
        self._total_count = 0
        self._injection_count = 0
        
    def get_diagnostics(self) -> FrankDiagnostics:
        """Get current diagnostics."""
        if self._total_count > 0:
            self.diagnostics.inhibit_rate = self._inhibit_count / self._total_count
            self.diagnostics.memory_injection_rate = self._injection_count / self._total_count
        self.diagnostics.mean_injection_magnitude = self.memory.last_injection_magnitude
        self.diagnostics.tau_values = self.brain.get_tau_values()
        return self.diagnostics
    
    def get_tau_values(self) -> List[float]:
        """Get current tau values from brain."""
        return self.brain.get_tau_values()


class FrankVariant(nn.Module):
    """Frank variant for ablation studies."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: Optional[ModelConfig] = None,
        use_memory: bool = True,
        use_reflex: bool = True,
        use_modularity: bool = True,
        veto_mode: str = 'normal'
    ):
        super().__init__()
        
        if config is None:
            config = ModelConfig()
        
        self.use_memory = use_memory
        self.use_reflex = use_reflex
        self.veto_mode = veto_mode
        
        self.embedding = nn.Embedding(input_dim, config.d_model)
        
        if use_reflex:
            self.reflex = Reflex(config.d_model, output_dim, hidden_dim=128)
        else:
            self.reflex = None
        
        n_modules = 4 if use_modularity else 1
        module_hidden_dim = 180 if use_modularity else 720
        brain_dim = n_modules * module_hidden_dim
        
        tau_values = config.tau_values if use_modularity else (15.0,)
        
        self.brain = Brain(
            input_dim=config.d_model,
            output_dim=output_dim,
            n_modules=n_modules,
            module_hidden_dim=module_hidden_dim,
            tau_values=tau_values
        )
        
        if use_memory:
            self.memory = ActiveMemory(
                brain_dim=brain_dim,
                n_slots=64,
                key_dim=640,
                value_dim=64
            )
        else:
            self.memory = None
        
        self.anomaly_detector = AnomalyDetector(config.d_model, hidden_dim=64)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size, seq_len = x.shape
        
        embedded = self.embedding(x)
        outputs = []
        
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            
            injection = None
            if self.memory is not None:
                if self.brain.hidden_states is not None:
                    brain_state = torch.cat(self.brain.hidden_states, dim=-1)
                    injection = self.memory(brain_state)
            
            brain_out, inhibit, _ = self.brain(
                x_t,
                self.brain.hidden_states,
                injection=injection
            )
            
            if self.reflex is None or self.veto_mode == 'always_brain':
                output = brain_out
            elif self.veto_mode == 'always_reflex':
                output = self.reflex(x_t)
            else:
                reflex_out = self.reflex(x_t)
                use_brain = (inhibit > 0.5).float()
                output = use_brain * brain_out + (1 - use_brain) * reflex_out
            
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)
    
    def reset_state(self):
        """Reset stateful components."""
        self.brain.reset_state()
