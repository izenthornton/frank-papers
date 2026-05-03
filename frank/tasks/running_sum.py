"""Running sum task."""
from typing import Tuple, Dict
import torch

from .base import BaseTask, TaskConfig


class RunningSumTask(BaseTask):
    """Running sum task.
    
    Input:  [+3, -1, +4, -2, +1]
    Target: [3,   2,  6,  4,  5]
    
    At each step, output the cumulative sum so far.
    Tests: Can it continuously update and maintain state?
    
    Input encoding: -9 to +9 encoded as 0-18
    Output encoding: -50 to +50 encoded as 0-100 (clamped)
    """
    
    INPUT_RANGE = 9
    OUTPUT_RANGE = 50
    INPUT_VOCAB = 19
    OUTPUT_VOCAB = 101
    
    def __init__(self, config: TaskConfig = None):
        if config is None:
            config = TaskConfig()
        config.test_long_min = 21
        config.test_long_max = 50
        super().__init__(config)
        
    @property
    def name(self) -> str:
        return "sum"
    
    @property
    def input_vocab_size(self) -> int:
        return self.INPUT_VOCAB
    
    @property
    def output_vocab_size(self) -> int:
        return self.OUTPUT_VOCAB
    
    @property
    def max_seq_len(self) -> int:
        return self.config.test_long_max
    
    def _encode_input(self, value: int) -> int:
        """Encode signed value to token."""
        return value + self.INPUT_RANGE
    
    def _decode_input(self, token: int) -> int:
        """Decode token to signed value."""
        return token - self.INPUT_RANGE
    
    def _encode_output(self, value: int) -> int:
        """Encode cumulative sum to token."""
        clamped = max(-self.OUTPUT_RANGE, min(self.OUTPUT_RANGE, value))
        return clamped + self.OUTPUT_RANGE
    
    def _decode_output(self, token: int) -> int:
        """Decode token to cumulative sum."""
        return token - self.OUTPUT_RANGE
    
    def generate_sample(
        self,
        seq_len: int,
        seed: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a running sum sample.
        
        Args:
            seq_len: Length of the operation sequence
            seed: Optional random seed
            
        Returns:
            Tuple of (input, target) tensors
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        values = torch.randint(-self.INPUT_RANGE, self.INPUT_RANGE + 1, (seq_len,))
        
        input_tokens = torch.tensor([self._encode_input(v.item()) for v in values])
        
        cumsum = 0
        target_tokens = []
        for v in values:
            cumsum += v.item()
            target_tokens.append(self._encode_output(cumsum))
        target_tokens = torch.tensor(target_tokens)
        
        return input_tokens, target_tokens
    
    def generate_dataset(
        self,
        size: int,
        min_len: int,
        max_len: int,
        seed: int = 42
    ):
        """Generate dataset with sequence lengths in [min_len, max_len]."""
        torch.manual_seed(seed)
        
        inputs_list = []
        targets_list = []
        lengths_list = []
        
        for i in range(size):
            seq_len = torch.randint(min_len, max_len + 1, (1,)).item()
            inp, tgt = self.generate_sample(seq_len, seed=seed + i)
            
            actual_len = len(inp)  # Save actual length before padding
            if actual_len < max_len:
                pad_len = max_len - actual_len
                inp = torch.cat([inp, torch.zeros(pad_len, dtype=inp.dtype)])
                tgt = torch.cat([tgt, torch.full((pad_len,), self.PAD_TOKEN, dtype=tgt.dtype)])
            
            inputs_list.append(inp[:max_len])
            targets_list.append(tgt[:max_len])
            lengths_list.append(actual_len)  # Use actual length, not padded
        
        from .base import TaskDataset
        return TaskDataset(
            torch.stack(inputs_list),
            torch.stack(targets_list),
            torch.tensor(lengths_list)
        )
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor
    ) -> Dict[str, float]:
        """Compute running sum metrics."""
        pred_tokens = predictions.argmax(dim=-1)
        batch_size = predictions.size(0)
        
        correct_tokens = 0
        total_tokens = 0
        correct_seqs = 0
        total_mae = 0.0
        
        per_position = {}
        
        for i in range(batch_size):
            length = int(lengths[i].item())
            
            pred_seq = pred_tokens[i, :length]
            target_seq = targets[i, :length]
            
            valid_mask = target_seq != -100
            if not valid_mask.any():
                continue
            
            pred_valid = pred_seq[valid_mask]
            target_valid = target_seq[valid_mask]
            
            matches = (pred_valid == target_valid)
            correct_tokens += matches.sum().item()
            total_tokens += len(target_valid)
            
            if matches.all():
                correct_seqs += 1
            
            pred_values = pred_valid.float() - self.OUTPUT_RANGE
            target_values = target_valid.float() - self.OUTPUT_RANGE
            total_mae += (pred_values - target_values).abs().mean().item()
            
            for pos_idx, match in enumerate(matches):
                if pos_idx not in per_position:
                    per_position[pos_idx] = {'correct': 0, 'total': 0}
                per_position[pos_idx]['total'] += 1
                if match:
                    per_position[pos_idx]['correct'] += 1
        
        metrics = {
            'token_accuracy': correct_tokens / max(total_tokens, 1),
            'sequence_accuracy': correct_seqs / max(batch_size, 1),
            'mae': total_mae / max(batch_size, 1)
        }
        
        for pos, counts in per_position.items():
            if counts['total'] > batch_size // 10:
                metrics[f'pos_{pos}_accuracy'] = counts['correct'] / counts['total']
        
        return metrics
