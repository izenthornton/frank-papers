"""Model factories and utilities."""
from typing import Optional
import torch.nn as nn

from .base import ModelConfig, count_parameters, apply_damage, apply_targeted_damage, FRANK_COMPONENTS
from .transformer import TransformerModel
from .gru import GRUModel
from .modular import ModularRecurrentModel
from .frank import FrankModel


def create_model(
    model_name: str,
    input_dim: int,
    output_dim: int,
    max_seq_len: int = 200,
    config: Optional[ModelConfig] = None
) -> nn.Module:
    """Create a model by name.

    The 'modular_memory', 'rims', and 'frank_no_laterals' variants used in the
    papers live in paper_experiments.py. Use that runner for the full set.

    Args:
        model_name: One of 'transformer', 'gru', 'modular', 'frank'
        input_dim: Input vocabulary size
        output_dim: Output vocabulary size
        max_seq_len: Maximum sequence length
        config: Optional model configuration

    Returns:
        The requested model
    """
    if config is None:
        config = ModelConfig()

    model_name = model_name.lower()

    if model_name == 'transformer':
        return TransformerModel(input_dim, output_dim, max_seq_len, config)
    elif model_name == 'gru':
        return GRUModel(input_dim, output_dim, config)
    elif model_name == 'modular':
        return ModularRecurrentModel(input_dim, output_dim, config)
    elif model_name == 'frank':
        return FrankModel(input_dim, output_dim, config)
    else:
        raise ValueError(f"Unknown model: {model_name}")


__all__ = [
    'ModelConfig',
    'count_parameters',
    'apply_damage',
    'apply_targeted_damage',
    'FRANK_COMPONENTS',
    'create_model',
    'TransformerModel',
    'GRUModel',
    'ModularRecurrentModel',
    'FrankModel',
]
