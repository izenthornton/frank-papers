"""Base task class and configuration."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class TaskConfig:
    """Configuration for tasks."""
    train_size: int = 100_000
    val_size: int = 10_000
    test_size: int = 10_000
    test_long_size: int = 10_000
    
    min_seq_len: int = 5
    max_seq_len: int = 20
    test_long_min: int = 21
    test_long_max: int = 40
    
    batch_size: int = 64
    num_workers: int = 0


class TaskDataset(Dataset):
    """Generic dataset for algorithmic tasks."""
    
    def __init__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor
    ):
        self.inputs = inputs
        self.targets = targets
        self.lengths = lengths
        
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return self.inputs[idx], self.targets[idx], self.lengths[idx].item()


class BaseTask(ABC):
    """Base class for all tasks."""
    
    PAD_TOKEN = -100
    
    def __init__(self, config: TaskConfig = None):
        if config is None:
            config = TaskConfig()
        self.config = config
        self._datasets: Dict[str, TaskDataset] = {}
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Task name."""
        pass
    
    @property
    @abstractmethod
    def input_vocab_size(self) -> int:
        """Size of input vocabulary."""
        pass
    
    @property
    @abstractmethod
    def output_vocab_size(self) -> int:
        """Size of output vocabulary."""
        pass
    
    @property
    def pad_token(self) -> int:
        """Token to use for padding (ignored in loss calculation)."""
        return self.PAD_TOKEN
    
    @property
    def max_seq_len(self) -> int:
        """Maximum sequence length (includes test_long for model sizing)."""
        return self.config.test_long_max * 2 + 2
    
    @abstractmethod
    def generate_sample(
        self,
        seq_len: int,
        seed: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a single sample.
        
        Args:
            seq_len: Sequence length
            seed: Optional random seed
            
        Returns:
            Tuple of (input, target) tensors
        """
        pass
    
    def generate_dataset(
        self,
        size: int,
        min_len: int,
        max_len: int,
        seed: int = 42
    ) -> TaskDataset:
        """Generate a dataset.
        
        Args:
            size: Number of samples
            min_len: Minimum sequence length
            max_len: Maximum sequence length
            seed: Random seed
            
        Returns:
            TaskDataset
        """
        torch.manual_seed(seed)
        
        inputs_list = []
        targets_list = []
        lengths_list = []
        
        max_total_len = max_len * 2 + 2
        
        for i in range(size):
            seq_len = torch.randint(min_len, max_len + 1, (1,)).item()
            inp, tgt = self.generate_sample(seq_len, seed=seed + i)
            
            actual_len = len(inp)  # Save actual length before padding
            
            if actual_len < max_total_len:
                pad_len = max_total_len - actual_len
                inp = torch.cat([inp, torch.zeros(pad_len, dtype=inp.dtype)])
                tgt = torch.cat([tgt, torch.full((pad_len,), self.PAD_TOKEN, dtype=tgt.dtype)])
            
            inputs_list.append(inp[:max_total_len])
            targets_list.append(tgt[:max_total_len])
            lengths_list.append(actual_len)
        
        return TaskDataset(
            torch.stack(inputs_list),
            torch.stack(targets_list),
            torch.tensor(lengths_list)
        )
    
    def get_dataloaders(
        self,
        seed: int = 42
    ) -> Dict[str, DataLoader]:
        """Get all dataloaders for this task.
        
        Args:
            seed: Random seed
            
        Returns:
            Dictionary with 'train', 'val', 'test', 'test_long' dataloaders
        """
        if not self._datasets:
            self._datasets['train'] = self.generate_dataset(
                self.config.train_size,
                self.config.min_seq_len,
                self.config.max_seq_len,
                seed=seed
            )
            self._datasets['val'] = self.generate_dataset(
                self.config.val_size,
                self.config.min_seq_len,
                self.config.max_seq_len,
                seed=seed + 100000
            )
            self._datasets['test'] = self.generate_dataset(
                self.config.test_size,
                self.config.min_seq_len,
                self.config.max_seq_len,
                seed=seed + 200000
            )
            self._datasets['test_long'] = self.generate_dataset(
                self.config.test_long_size,
                self.config.test_long_min,
                self.config.test_long_max,
                seed=seed + 300000
            )
        
        def collate_fn(batch):
            inputs, targets, lengths = zip(*batch)
            return (
                torch.stack(inputs),
                torch.stack(targets),
                torch.tensor(lengths)
            )
        
        return {
            split: DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=(split == 'train'),
                num_workers=self.config.num_workers,
                collate_fn=collate_fn
            )
            for split, dataset in self._datasets.items()
        }
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor
    ) -> Dict[str, float]:
        """Compute task-specific metrics.
        
        Args:
            predictions: Model predictions (batch, seq_len, vocab)
            targets: Ground truth (batch, seq_len)
            lengths: Sequence lengths
            
        Returns:
            Dictionary of metric names to values
        """
        pred_tokens = predictions.argmax(dim=-1)
        
        batch_size = predictions.size(0)
        correct_tokens = 0
        total_tokens = 0
        correct_seqs = 0
        
        for i in range(batch_size):
            length = int(lengths[i].item())
            pred_seq = pred_tokens[i, :length]
            target_seq = targets[i, :length]
            
            matches = (pred_seq == target_seq)
            correct_tokens += matches.sum().item()
            total_tokens += length
            
            if matches.all():
                correct_seqs += 1
        
        return {
            'token_accuracy': correct_tokens / max(total_tokens, 1),
            'sequence_accuracy': correct_seqs / max(batch_size, 1)
        }
