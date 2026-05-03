"""Chain rule task - modular arithmetic sequence transformation."""
from typing import Tuple, Dict
import torch

from .base import BaseTask, TaskConfig


class ChainRuleTask(BaseTask):
    """Chain rule task.
    
    Input:  [3, 5, 2, 7, 4]
    Rule:   output[0] = input[0]
            output[t] = (output[t-1] + input[t]) mod 10
    Target: [3, 8, 0, 7, 1]
    
    Tests: Can the model learn a deterministic transition rule
    and generalize it to longer sequences than seen during training?
    
    Recurrent architectures can learn the step function state' = (state + input) mod 10
    and apply it at any position. Transformers learn position-specific attention
    patterns that break on unseen positions.
    """
    
    VOCAB_SIZE = 10
    
    def __init__(self, config: TaskConfig = None):
        if config is None:
            config = TaskConfig()
        config.test_long_min = 21
        config.test_long_max = 50
        super().__init__(config)
        
    @property
    def name(self) -> str:
        return "chain"
    
    @property
    def input_vocab_size(self) -> int:
        return self.VOCAB_SIZE
    
    @property
    def output_vocab_size(self) -> int:
        return self.VOCAB_SIZE
    
    @property
    def max_seq_len(self) -> int:
        return self.config.test_long_max
    
    def generate_sample(
        self,
        seq_len: int,
        seed: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)
        
        input_tokens = torch.randint(0, self.VOCAB_SIZE, (seq_len,))
        
        target_tokens = torch.zeros(seq_len, dtype=torch.long)
        state = 0
        for t in range(seq_len):
            if t == 0:
                state = input_tokens[t].item()
            else:
                state = (state + input_tokens[t].item()) % self.VOCAB_SIZE
            target_tokens[t] = state
        
        return input_tokens, target_tokens
    
    def generate_dataset(
        self,
        size: int,
        min_len: int,
        max_len: int,
        seed: int = 42
    ):
        torch.manual_seed(seed)
        
        inputs_list = []
        targets_list = []
        lengths_list = []
        
        for i in range(size):
            seq_len = torch.randint(min_len, max_len + 1, (1,)).item()
            inp, tgt = self.generate_sample(seq_len, seed=seed + i)
            
            actual_len = len(inp)
            if actual_len < max_len:
                pad_len = max_len - actual_len
                inp = torch.cat([inp, torch.zeros(pad_len, dtype=inp.dtype)])
                tgt = torch.cat([tgt, torch.full((pad_len,), self.PAD_TOKEN, dtype=tgt.dtype)])
            
            inputs_list.append(inp[:max_len])
            targets_list.append(tgt[:max_len])
            lengths_list.append(actual_len)
        
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
        pred_tokens = predictions.argmax(dim=-1)
        batch_size = predictions.size(0)
        
        correct_tokens = 0
        total_tokens = 0
        correct_seqs = 0
        
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
            if counts['total'] > batch_size // 10:
                metrics[f'pos_{pos}_accuracy'] = counts['correct'] / counts['total']
        
        return metrics
