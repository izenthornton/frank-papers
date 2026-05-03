"""Delayed recall task."""
from typing import Tuple, Dict
import torch

from .base import BaseTask, TaskConfig


class DelayedRecallTask(BaseTask):
    """Delayed recall task.
    
    Input:  [TARGET, noise, noise, noise, noise, RECALL]
    Target: [0,      0,     0,     0,     0,     TARGET]
    
    Model sees target token, then noise, must recall target at end.
    Tests: Can it maintain memory through interference?
    """
    
    VOCAB_SIZE = 10
    RECALL_TOKEN = 10
    PAD_TOKEN = -100
    
    def __init__(self, config: TaskConfig = None):
        if config is None:
            config = TaskConfig()
        config.test_long_min = 31
        config.test_long_max = 60
        super().__init__(config)
        
    @property
    def name(self) -> str:
        return "recall"
    
    @property
    def input_vocab_size(self) -> int:
        return 11
    
    @property
    def output_vocab_size(self) -> int:
        return 11
    
    @property
    def max_seq_len(self) -> int:
        return self.config.test_long_max + 2
    
    def generate_sample(
        self,
        delay_len: int,
        seed: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a delayed recall sample.
        
        Args:
            delay_len: Number of noise tokens between target and recall
            seed: Optional random seed
            
        Returns:
            Tuple of (input, target) tensors
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        target_token = torch.randint(0, self.VOCAB_SIZE, (1,)).item()
        
        noise = torch.randint(0, self.VOCAB_SIZE, (delay_len,))
        for i in range(delay_len):
            while noise[i].item() == target_token:
                noise[i] = torch.randint(0, self.VOCAB_SIZE, (1,)).item()
        
        input_seq = torch.cat([
            torch.tensor([target_token]),
            noise,
            torch.tensor([self.RECALL_TOKEN])
        ])
        
        target_seq = torch.cat([
            torch.full((delay_len + 1,), self.PAD_TOKEN, dtype=torch.long),
            torch.tensor([target_token])
        ])
        
        return input_seq, target_seq
    
    def generate_dataset(
        self,
        size: int,
        min_len: int,
        max_len: int,
        seed: int = 42
    ):
        """Generate dataset with delay lengths in [min_len, max_len]."""
        torch.manual_seed(seed)
        
        inputs_list = []
        targets_list = []
        lengths_list = []
        
        max_total_len = max_len + 2
        
        for i in range(size):
            delay_len = torch.randint(min_len, max_len + 1, (1,)).item()
            inp, tgt = self.generate_sample(delay_len, seed=seed + i)
            
            actual_len = len(inp)  # Save actual length before padding
            if actual_len < max_total_len:
                pad_len = max_total_len - actual_len
                inp = torch.cat([inp, torch.zeros(pad_len, dtype=inp.dtype)])
                tgt = torch.cat([tgt, torch.full((pad_len,), self.PAD_TOKEN, dtype=tgt.dtype)])
            
            inputs_list.append(inp[:max_total_len])
            targets_list.append(tgt[:max_total_len])
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
        """Compute delayed recall metrics.
        
        Focuses on the final token (recall position).
        """
        pred_tokens = predictions.argmax(dim=-1)
        batch_size = predictions.size(0)
        
        correct = 0
        by_delay = {}
        
        for i in range(batch_size):
            length = int(lengths[i].item())
            recall_pos = length - 1
            delay = length - 2
            
            pred = pred_tokens[i, recall_pos].item()
            target = targets[i, recall_pos].item()
            
            if target == -100:
                continue
            
            is_correct = (pred == target)
            if is_correct:
                correct += 1
            
            if delay not in by_delay:
                by_delay[delay] = {'correct': 0, 'total': 0}
            by_delay[delay]['total'] += 1
            if is_correct:
                by_delay[delay]['correct'] += 1
        
        metrics = {
            'recall_accuracy': correct / max(batch_size, 1),
            'token_accuracy': correct / max(batch_size, 1),
            'sequence_accuracy': correct / max(batch_size, 1)
        }
        
        for delay, counts in by_delay.items():
            if counts['total'] > 0:
                metrics[f'delay_{delay}_accuracy'] = counts['correct'] / counts['total']
        
        return metrics
