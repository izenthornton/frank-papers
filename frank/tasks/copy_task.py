"""Copy sequence task."""
from typing import Tuple, Dict
import torch

from .base import BaseTask, TaskConfig


class CopyTask(BaseTask):
    """Copy sequence task.
    
    Input:  [3, 1, 4, 1, 5, DELIM, 0, 0, 0, 0, 0]
    Target: [0, 0, 0, 0, 0, 0,     3, 1, 4, 1, 5]
    
    Model sees sequence, then must reproduce it after delimiter.
    Tests: Can it store and recall a sequence?
    """
    
    VOCAB_SIZE = 10
    DELIM_TOKEN = 10
    PAD_TOKEN = -100
    
    def __init__(self, config: TaskConfig = None):
        super().__init__(config)
        
    @property
    def name(self) -> str:
        return "copy"
    
    @property
    def input_vocab_size(self) -> int:
        return 12
    
    @property
    def output_vocab_size(self) -> int:
        return 12
    
    def generate_sample(
        self,
        seq_len: int,
        seed: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a copy task sample.
        
        Args:
            seq_len: Length of the sequence to copy
            seed: Optional random seed
            
        Returns:
            Tuple of (input, target) tensors
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        sequence = torch.randint(0, self.VOCAB_SIZE, (seq_len,))
        
        input_seq = torch.cat([
            sequence,
            torch.tensor([self.DELIM_TOKEN]),
            torch.zeros(seq_len, dtype=torch.long)
        ])
        
        target_seq = torch.cat([
            torch.full((seq_len + 1,), self.PAD_TOKEN, dtype=torch.long),
            sequence
        ])
        
        return input_seq, target_seq
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor
    ) -> Dict[str, float]:
        """Compute copy task metrics.
        
        Focuses on the output portion (after delimiter).
        """
        pred_tokens = predictions.argmax(dim=-1)
        batch_size = predictions.size(0)
        
        correct_tokens = 0
        total_tokens = 0
        correct_seqs = 0
        
        per_position = {}
        
        for i in range(batch_size):
            length = int(lengths[i].item())
            copy_len = (length - 1) // 2
            output_start = copy_len + 1
            
            pred_output = pred_tokens[i, output_start:length]
            target_output = targets[i, output_start:length]
            
            valid_mask = target_output != -100
            if not valid_mask.any():
                continue
            
            pred_valid = pred_output[valid_mask]
            target_valid = target_output[valid_mask]
            
            matches = (pred_valid == target_valid)
            correct_tokens += matches.sum().item()
            total_tokens += len(pred_valid)
            
            if matches.all():
                correct_seqs += 1
            
            for pos_idx, match in enumerate(matches):
                if pos_idx not in per_position:
                    per_position[pos_idx] = {'correct': 0, 'total': 0}
                per_position[pos_idx]['total'] += 1
                if match:
                    per_position[pos_idx]['correct'] += 1
        
        metrics = {
            'token_accuracy': correct_tokens / max(total_tokens, 1),
            'sequence_accuracy': correct_seqs / max(batch_size, 1)
        }
        
        for pos, counts in per_position.items():
            if counts['total'] > 0:
                metrics[f'pos_{pos}_accuracy'] = counts['correct'] / counts['total']
        
        return metrics
