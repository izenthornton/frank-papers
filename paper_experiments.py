#!/usr/bin/env python3
"""
FRANK Papers Experiment Runner
==============================================================

Drop this file + the frank_comparison codebase on a cloud GPU and run.

Phases (can run in parallel on separate GPUs):
  Phase 1: Train 7 models x 10 seeds on ALL tasks (chain, copy, recall, sum)
  Phase 2: Extreme chain generalization (up to 100,000x)
  Phase 3: Lesion study on ALL tasks (global + targeted)
  Phase 4: Standard multi-seed generalization (2x-10x) on ALL tasks

Usage:
  python paper_experiments.py --phase 1 --quick                    # smoke test
  python paper_experiments.py --phase 1 --models frank,gru --gpu 0  # split work
  python paper_experiments.py --phase 2 --gpu 0                     # extreme eval
  python paper_experiments.py --seeds 42,123,456,789                # all phases, subset of seeds
  python paper_experiments.py --phase all                           # everything

Resume after crash: just re-run the same command. Completed work is skipped.
"""

import argparse
import copy
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader

# --- Add project to path -----------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from frank.models.base import (
    ModelConfig, RecurrentModule, LateralConnections, PositionalEncoding,
    count_parameters, apply_damage, apply_targeted_damage, FRANK_COMPONENTS
)
from frank.models.frank import FrankModel, ActiveMemory
from frank.models.gru import GRUModel
from frank.models.modular import ModularRecurrentModel
from frank.models.transformer import TransformerModel
from frank.tasks.chain_rule import ChainRuleTask
from frank.tasks.copy_task import CopyTask
from frank.tasks.delayed_recall import DelayedRecallTask
from frank.tasks.running_sum import RunningSumTask
from frank.tasks.base import TaskConfig, TaskDataset

# --- Constants ----------------------------------------------------------------
ALL_SEEDS = [42, 123, 456, 789, 1337, 2024, 3141, 4242, 5555, 6789]
ALL_MODELS = ['transformer', 'gru', 'modular', 'modular_memory', 'rims', 'frank', 'frank_no_laterals']
ALL_TASKS = ['chain', 'copy', 'recall', 'sum']
EXTREME_SCALES = [10, 100, 1000, 10000, 30000, 100000]
STANDARD_SCALES = [2, 3, 5, 10]
DAMAGE_LEVELS = [0, 10, 20, 30, 50]
TARGETED_DAMAGE_LEVELS = [0, 10, 20, 30, 50, 70]
TRAIN_MAX_LEN = 20  # trained on lengths 5-20


def create_task_by_name(name: str, config: TaskConfig = None):
    """Create a task by name."""
    if config is None:
        config = TaskConfig()
    if name == 'chain':
        return ChainRuleTask(config)
    elif name == 'copy':
        return CopyTask(config)
    elif name == 'recall':
        return DelayedRecallTask(config)
    elif name == 'sum':
        return RunningSumTask(config)
    else:
        raise ValueError(f"Unknown task: {name}")

RESULTS_DIR = PROJECT_ROOT / 'paper1_results'
CHECKPOINTS_DIR = PROJECT_ROOT / 'paper1_checkpoints'
EXTREME_DIR = RESULTS_DIR / 'extreme'

# --- Global interrupt flag ----------------------------------------------------
_interrupted = False

def _signal_handler(sig, frame):
    global _interrupted
    if _interrupted:
        print("\n*** Force quit ***")
        sys.exit(1)
    _interrupted = True
    print("\n*** Graceful shutdown requested - finishing current work... ***")

signal.signal(signal.SIGINT, _signal_handler)


# ===============================================================================
# PROGRESS TRACKING & ATOMIC I/O
# ===============================================================================

def save_json_atomic(data, path):
    """Write JSON atomically (write-to-temp-then-rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(str(tmp), str(path))


def load_json_safe(path):
    """Load JSON, returning empty dict if missing or corrupt."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


class ProgressTracker:
    """Tracks completed work for resume support."""

    def __init__(self):
        self.path = RESULTS_DIR / 'progress.json'
        self.data = load_json_safe(self.path)

    def is_done(self, phase: str, key: str) -> bool:
        return self.data.get(phase, {}).get(key) == 'completed'

    def mark_done(self, phase: str, key: str):
        if phase not in self.data:
            self.data[phase] = {}
        self.data[phase][key] = 'completed'
        save_json_atomic(self.data, self.path)

    def mark_in_progress(self, phase: str, key: str):
        if phase not in self.data:
            self.data[phase] = {}
        self.data[phase][key] = 'in_progress'
        save_json_atomic(self.data, self.path)


# ===============================================================================
# NEW MODEL IMPLEMENTATIONS
# ===============================================================================

class ModularMemoryModel(nn.Module):
    """Modular Recurrent + Active Memory (no reflex/anomaly/inhibition).

    Same as ModularRecurrentModel but with ActiveMemory injecting into module 0.
    Target: ~500K parameters.
    """

    def __init__(self, input_dim: int, output_dim: int, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()

        self.n_modules = config.n_modules
        self.module_hidden_dim = 140  # sized to hit ~500K with memory overhead
        total_hidden = self.n_modules * self.module_hidden_dim

        self.embedding = nn.Embedding(input_dim, config.d_model)

        self.recurrent_modules = nn.ModuleList([
            RecurrentModule(config.d_model, self.module_hidden_dim, tau)
            for tau in config.tau_values
        ])

        self.lateral = LateralConnections(self.n_modules, self.module_hidden_dim)

        self.memory = ActiveMemory(
            brain_dim=total_hidden,
            n_slots=64,
            key_dim=total_hidden,
            value_dim=64
        )
        self.memory_proj = nn.Linear(total_hidden // 4, self.module_hidden_dim)

        self.output_proj = nn.Sequential(
            nn.Linear(total_hidden, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, output_dim)
        )

        self.hidden_states: Optional[List[torch.Tensor]] = None
        self.prev_combined: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        device = x.device

        hidden_states = [
            torch.zeros(batch_size, self.module_hidden_dim, device=device)
            for _ in range(self.n_modules)
        ]

        embedded = self.embedding(x)
        outputs = []

        for t in range(seq_len):
            x_t = embedded[:, t, :]

            # Memory injection into module 0
            injection = None
            if self.prev_combined is not None:
                mem_out = self.memory(self.prev_combined)
                injection = self.memory_proj(mem_out)

            new_states = []
            for i, module in enumerate(self.recurrent_modules):
                h = module(x_t, hidden_states[i])
                if i == 0 and injection is not None:
                    h = h + injection
                new_states.append(h)

            stacked = torch.stack(new_states, dim=1)
            stacked = self.lateral(stacked)
            hidden_states = [stacked[:, i, :] for i in range(self.n_modules)]

            combined = torch.cat(hidden_states, dim=-1)
            self.prev_combined = combined.detach()

            output = self.output_proj(combined)
            outputs.append(output)

        self.hidden_states = [h.detach() for h in hidden_states]
        return torch.stack(outputs, dim=1)

    def reset_state(self):
        self.hidden_states = None
        self.prev_combined = None


class RIMsModel(nn.Module):
    """Recurrent Independent Mechanisms (Goyal et al. 2021).

    4 GRU-based modules with:
    - Input attention: top-K=2 modules receive input per step
    - Communication attention: multi-head attention between modules
    Target: ~500K parameters.
    """

    def __init__(self, input_dim: int, output_dim: int, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()

        self.n_modules = 4
        self.k_active = 2
        self.module_hidden_dim = 76  # sized to hit ~500K with 8 GRU cells + communication
        total_hidden = self.n_modules * self.module_hidden_dim

        self.embedding = nn.Embedding(input_dim, config.d_model)

        # Per-module GRU cells
        self.gru_cells = nn.ModuleList([
            nn.GRUCell(config.d_model, self.module_hidden_dim)
            for _ in range(self.n_modules)
        ])

        # Null-input GRU cells (for non-selected modules)
        self.null_gru_cells = nn.ModuleList([
            nn.GRUCell(config.d_model, self.module_hidden_dim)
            for _ in range(self.n_modules)
        ])

        # Input attention: scores which modules get input
        self.input_attention = nn.Sequential(
            nn.Linear(config.d_model + self.module_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Communication attention (2 heads)
        self.comm_n_heads = 2
        self.comm_head_dim = self.module_hidden_dim // self.comm_n_heads
        self.comm_q = nn.Linear(self.module_hidden_dim, self.module_hidden_dim)
        self.comm_k = nn.Linear(self.module_hidden_dim, self.module_hidden_dim)
        self.comm_v = nn.Linear(self.module_hidden_dim, self.module_hidden_dim)
        self.comm_out = nn.Linear(self.module_hidden_dim, self.module_hidden_dim)

        rims_d_ff = 274  # tuned to hit ~500K params with module_hidden_dim=76
        self.output_proj = nn.Sequential(
            nn.Linear(total_hidden, rims_d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(rims_d_ff, output_dim)
        )

        self.hidden_states: Optional[List[torch.Tensor]] = None

    def _input_attention_step(self, x_t, hidden_states):
        """Compute which modules get input (soft top-K)."""
        batch_size = x_t.size(0)
        scores = []
        for i in range(self.n_modules):
            inp = torch.cat([x_t, hidden_states[i]], dim=-1)
            score = self.input_attention(inp)  # (batch, 1)
            scores.append(score)
        scores = torch.cat(scores, dim=-1)  # (batch, n_modules)

        # Soft top-K: keep top-K scores, zero out rest, then softmax
        topk_vals, topk_idx = scores.topk(self.k_active, dim=-1)
        mask = torch.zeros_like(scores).scatter_(-1, topk_idx, 1.0)
        # Softmax over selected modules only
        masked_scores = scores * mask + (1 - mask) * (-1e9)
        attention = F.softmax(masked_scores, dim=-1)
        return attention, mask

    def _communication_step(self, hidden_states):
        """Multi-head attention communication between modules."""
        batch_size = hidden_states[0].size(0)
        # Stack: (batch, n_modules, hidden_dim)
        h = torch.stack(hidden_states, dim=1)

        Q = self.comm_q(h)  # (batch, n_modules, hidden_dim)
        K = self.comm_k(h)
        V = self.comm_v(h)

        # Reshape for multi-head
        Q = Q.view(batch_size, self.n_modules, self.comm_n_heads, self.comm_head_dim).transpose(1, 2)
        K = K.view(batch_size, self.n_modules, self.comm_n_heads, self.comm_head_dim).transpose(1, 2)
        V = V.view(batch_size, self.n_modules, self.comm_n_heads, self.comm_head_dim).transpose(1, 2)

        # Attention
        scale = self.comm_head_dim ** 0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, self.n_modules, -1)
        out = self.comm_out(out)

        # Residual connection
        new_states = []
        for i in range(self.n_modules):
            new_states.append(hidden_states[i] + out[:, i, :])
        return new_states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        device = x.device

        hidden_states = [
            torch.zeros(batch_size, self.module_hidden_dim, device=device)
            for _ in range(self.n_modules)
        ]

        embedded = self.embedding(x)
        outputs = []

        null_input = torch.zeros(batch_size, embedded.size(-1), device=device)

        for t in range(seq_len):
            x_t = embedded[:, t, :]

            # Input attention
            attention, mask = self._input_attention_step(x_t, hidden_states)

            # Update modules
            new_states = []
            for i in range(self.n_modules):
                # Selected modules get input, others get null update
                is_active = mask[:, i].unsqueeze(-1)  # (batch, 1)
                h_active = self.gru_cells[i](x_t, hidden_states[i])
                h_null = self.null_gru_cells[i](null_input, hidden_states[i])
                h_new = is_active * h_active + (1 - is_active) * h_null
                new_states.append(h_new)

            # Communication
            hidden_states = self._communication_step(new_states)

            combined = torch.cat(hidden_states, dim=-1)
            output = self.output_proj(combined)
            outputs.append(output)

        self.hidden_states = [h.detach() for h in hidden_states]
        return torch.stack(outputs, dim=1)

    def reset_state(self):
        self.hidden_states = None


class FrankNoLaterals(nn.Module):
    """Frank with lateral connections disabled (ablation control).

    Identical to FrankModel except lateral connections return input unchanged.
    """

    def __init__(self, input_dim: int, output_dim: int, config: Optional[ModelConfig] = None):
        super().__init__()
        # Build a normal Frank model
        self._frank = FrankModel(input_dim, output_dim, config)
        # Disable lateral connections: replace forward to return identity
        self._frank.brain.lateral = _IdentityLateral()

    def forward(self, x, **kwargs):
        return self._frank(x, **kwargs)

    def reset_state(self):
        self._frank.reset_state()

    def get_tau_values(self):
        return self._frank.get_tau_values()

    def named_parameters(self, *args, **kwargs):
        return self._frank.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self._frank.parameters(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self._frank.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self._frank.load_state_dict(*args, **kwargs)

    def to(self, *args, **kwargs):
        self._frank = self._frank.to(*args, **kwargs)
        return self

    def train(self, mode=True):
        self._frank.train(mode)
        return self

    def eval(self):
        self._frank.eval()
        return self


class _IdentityLateral(nn.Module):
    """Dummy lateral connection that returns input unchanged."""
    def forward(self, module_states):
        return module_states


# ===============================================================================
# MODEL FACTORY
# ===============================================================================

def create_model(name: str, input_dim: int = 10, output_dim: int = 10,
                 max_seq_len: int = 200) -> nn.Module:
    """Create a model by name."""
    config = ModelConfig()
    if name == 'transformer':
        return TransformerModel(input_dim, output_dim, max_seq_len=max_seq_len, config=config)
    elif name == 'gru':
        return GRUModel(input_dim, output_dim, config=config)
    elif name == 'modular':
        return ModularRecurrentModel(input_dim, output_dim, config=config)
    elif name == 'modular_memory':
        return ModularMemoryModel(input_dim, output_dim, config=config)
    elif name == 'rims':
        return RIMsModel(input_dim, output_dim, config=config)
    elif name == 'frank':
        return FrankModel(input_dim, output_dim, config=config)
    elif name == 'frank_no_laterals':
        return FrankNoLaterals(input_dim, output_dim, config=config)
    else:
        raise ValueError(f"Unknown model: {name}")


# ===============================================================================
# TRAINING INFRASTRUCTURE
# ===============================================================================

def train_model(model_name: str, seed: int, device: str, task_name: str = 'chain',
                quick: bool = False):
    """Train a single model on a task. Returns test metrics."""
    global _interrupted

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Config
    is_frank = model_name in ('frank', 'frank_no_laterals')
    max_epochs = 10 if quick else (200 if is_frank else 100)
    patience = 3 if quick else 7

    # Task
    task_config = TaskConfig()
    task_config.train_size = 1000 if quick else 100_000
    task_config.val_size = 200 if quick else 10_000
    task_config.test_size = 200 if quick else 5_000
    task_config.test_long_size = 200 if quick else 5_000
    task = create_task_by_name(task_name, task_config)
    dataloaders = task.get_dataloaders(seed=seed)

    # Model
    model = create_model(model_name, input_dim=task.input_vocab_size,
                         output_dim=task.output_vocab_size, max_seq_len=task.max_seq_len)
    model = model.to(device)
    n_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"Training {model_name} on {task_name} seed={seed} ({n_params:,} params)")
    print(f"{'='*60}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Schedulers
    total_steps = len(dataloaders['train']) * max_epochs
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0
    warmup_sched = LambdaLR(optimizer, lr_lambda)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6)

    # Checkpoint path
    ckpt_path = CHECKPOINTS_DIR / f"best_{model_name}_seed{seed}_{task_name}.pt"
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Check for resumable checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0

    resume_ckpt = CHECKPOINTS_DIR / f"resume_{model_name}_seed{seed}_{task_name}.pt"
    if resume_ckpt.exists():
        print(f"  Resuming from {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        patience_counter = ckpt.get('patience_counter', 0)
        global_step = ckpt.get('global_step', 0)
        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    best_epoch = start_epoch

    for epoch in range(start_epoch, max_epochs):
        if _interrupted:
            # Save resume checkpoint
            torch.save({
                'epoch': epoch - 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'global_step': global_step,
            }, resume_ckpt)
            print(f"  Saved resume checkpoint at epoch {epoch}")
            return None

        # Train epoch
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_tokens = 0

        for inputs, targets, lengths in dataloaders['train']:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if hasattr(model, 'reset_state'):
                model.reset_state()

            optimizer.zero_grad()
            outputs = model(inputs)

            batch_size, seq_len, vocab_size = outputs.shape
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if global_step < warmup_steps:
                warmup_sched.step()
            else:
                cosine_sched.step()
            global_step += 1

            epoch_loss += loss.item() * batch_size

            with torch.no_grad():
                preds = outputs.argmax(dim=-1)
                for i in range(batch_size):
                    length = int(lengths[i].item())
                    valid = targets[i, :length] != -100
                    if valid.any():
                        epoch_correct += (preds[i, :length][valid] == targets[i, :length][valid]).sum().item()
                        epoch_tokens += valid.sum().item()

        train_acc = epoch_correct / max(epoch_tokens, 1)

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_tokens = 0

        with torch.no_grad():
            for inputs, targets, lengths in dataloaders['val']:
                inputs = inputs.to(device)
                targets = targets.to(device)
                if hasattr(model, 'reset_state'):
                    model.reset_state()
                outputs = model(inputs)
                batch_size, seq_len, vocab_size = outputs.shape
                loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
                val_loss += loss.item() * batch_size
                preds = outputs.argmax(dim=-1)
                for i in range(batch_size):
                    length = int(lengths[i].item())
                    valid = targets[i, :length] != -100
                    if valid.any():
                        val_correct += (preds[i, :length][valid] == targets[i, :length][valid]).sum().item()
                        val_tokens += valid.sum().item()

        val_loss_avg = val_loss / max(len(dataloaders['val'].dataset), 1)
        val_acc = val_correct / max(val_tokens, 1)

        print(f"  Epoch {epoch+1}/{max_epochs}: train_acc={train_acc:.4f} val_loss={val_loss_avg:.4f} val_acc={val_acc:.4f} patience={patience_counter}/{patience}")

        # Checkpoint
        if val_loss_avg < best_val_loss - 1e-5:
            best_val_loss = val_loss_avg
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_accuracy': val_acc,
                'global_step': global_step,
                'patience_counter': patience_counter,
                'n_params': n_params,
            }, ckpt_path)
            # Also save resume checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'global_step': global_step,
            }, resume_ckpt)
        else:
            # Only count patience once model has actually learned (>90% val acc)
            if val_acc >= 0.90:
                patience_counter += 1

        if patience_counter >= patience and val_acc >= 0.90:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best checkpoint for eval
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    # Evaluate on test set
    test_metrics = evaluate_on_dataloader(model, dataloaders['test'], device)
    print(f"  Test: token_acc={test_metrics['token_accuracy']:.4f} seq_acc={test_metrics['sequence_accuracy']:.4f}")

    # Clean up resume checkpoint on successful completion
    if resume_ckpt.exists():
        resume_ckpt.unlink()

    return {
        'model': model_name,
        'seed': seed,
        'n_params': n_params,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test': test_metrics,
    }


def evaluate_on_dataloader(model, dataloader, device):
    """Evaluate model on a DataLoader. Returns metrics dict."""
    model.eval()
    total_correct = 0
    total_tokens = 0
    correct_seqs = 0
    total_seqs = 0
    per_position = {}

    with torch.no_grad():
        for inputs, targets, lengths in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if hasattr(model, 'reset_state'):
                model.reset_state()

            outputs = model(inputs)
            preds = outputs.argmax(dim=-1)

            for i in range(inputs.size(0)):
                length = int(lengths[i].item())
                pred_seq = preds[i, :length]
                target_seq = targets[i, :length]

                valid = target_seq != -100
                if not valid.any():
                    continue

                matches = (pred_seq[valid] == target_seq[valid])
                total_correct += matches.sum().item()
                total_tokens += valid.sum().item()

                if matches.all():
                    correct_seqs += 1
                total_seqs += 1

                # Per-position
                valid_idx = 0
                for pos in range(length):
                    if target_seq[pos] != -100:
                        if valid_idx not in per_position:
                            per_position[valid_idx] = {'correct': 0, 'total': 0}
                        per_position[valid_idx]['total'] += 1
                        if pred_seq[pos] == target_seq[pos]:
                            per_position[valid_idx]['correct'] += 1
                        valid_idx += 1

    metrics = {
        'token_accuracy': total_correct / max(total_tokens, 1),
        'sequence_accuracy': correct_seqs / max(total_seqs, 1),
    }
    for pos, counts in sorted(per_position.items()):
        if counts['total'] > 10:
            metrics[f'pos_{pos}_accuracy'] = counts['correct'] / counts['total']

    return metrics


# ===============================================================================
# EXTREME LENGTH EVALUATION (STREAMING)
# ===============================================================================

def eval_chunk_gru(model, input_chunk, hidden, device):
    """Forward a chunk of tokens through GRU. Returns (logits, hidden).
    input_chunk: (1, chunk_len) tensor of token indices.
    """
    embedded = model.embedding(input_chunk)  # (1, chunk_len, d_model)
    if hidden is None:
        hidden = torch.zeros(model.n_layers, 1, model.hidden_dim, device=device)
    output, hidden = model.gru(embedded, hidden)
    logits = model.output_proj(output)  # (1, chunk_len, vocab)
    return logits[0], hidden  # (chunk_len, vocab), hidden


def eval_chunk_frank(model, input_chunk, device):
    """Forward a chunk through Frank, maintaining hidden state across calls.
    input_chunk: (1, chunk_len) tensor. Returns logits: (chunk_len, vocab).
    """
    frank = model._frank if hasattr(model, '_frank') else model
    batch_size, chunk_len = input_chunk.shape

    embedded = frank.embedding(input_chunk)  # (1, chunk_len, d_model)

    # Ensure _prev_brain_state attribute exists
    if not hasattr(frank, '_prev_brain_state'):
        frank._prev_brain_state = None

    outputs = []
    for t in range(chunk_len):
        x_t = embedded[:, t, :]

        injection = None
        if frank._prev_brain_state is not None:
            injection = frank.memory(frank._prev_brain_state)

        brain_out, inhibit, brain_state = frank.brain(
            x_t, frank.brain.hidden_states, injection=injection
        )
        frank._prev_brain_state = brain_state

        reflex_out = frank.reflex(x_t)
        output = inhibit * brain_out + (1 - inhibit) * reflex_out
        outputs.append(output[0])  # (vocab,) - squeeze batch dim

    return torch.stack(outputs, dim=0)  # (chunk_len, vocab)


def eval_chunk_modular(model, input_chunk, device):
    """Forward a chunk through Modular/ModularMemory, maintaining state.
    input_chunk: (1, chunk_len). Returns logits: (chunk_len, vocab).
    """
    batch_size, chunk_len = input_chunk.shape

    embedded = model.embedding(input_chunk)

    if model.hidden_states is None:
        model.hidden_states = [
            torch.zeros(batch_size, model.module_hidden_dim, device=device)
            for _ in range(model.n_modules)
        ]

    outputs = []
    for t in range(chunk_len):
        x_t = embedded[:, t, :]

        # Memory injection for ModularMemory
        injection = None
        if hasattr(model, 'memory') and hasattr(model, 'prev_combined') and model.prev_combined is not None:
            mem_out = model.memory(model.prev_combined)
            injection = model.memory_proj(mem_out)

        new_states = []
        for i, module in enumerate(model.recurrent_modules):
            h = module(x_t, model.hidden_states[i])
            if i == 0 and injection is not None:
                h = h + injection
            new_states.append(h)

        stacked = torch.stack(new_states, dim=1)
        stacked = model.lateral(stacked)
        model.hidden_states = [stacked[:, j, :] for j in range(model.n_modules)]

        combined = torch.cat(model.hidden_states, dim=-1)
        if hasattr(model, 'prev_combined'):
            model.prev_combined = combined.detach()

        output = model.output_proj(combined)
        outputs.append(output[0])  # (vocab,)

    return torch.stack(outputs, dim=0)  # (chunk_len, vocab)


def eval_chunk_rims(model, input_chunk, device):
    """Forward a chunk through RIMs, maintaining state.
    input_chunk: (1, chunk_len). Returns logits: (chunk_len, vocab).
    """
    batch_size, chunk_len = input_chunk.shape

    embedded = model.embedding(input_chunk)

    if model.hidden_states is None:
        model.hidden_states = [
            torch.zeros(batch_size, model.module_hidden_dim, device=device)
            for _ in range(model.n_modules)
        ]

    null_input = torch.zeros(batch_size, embedded.size(-1), device=device)

    outputs = []
    for t in range(chunk_len):
        x_t = embedded[:, t, :]

        attention, mask = model._input_attention_step(x_t, model.hidden_states)

        new_states = []
        for i in range(model.n_modules):
            is_active = mask[:, i].unsqueeze(-1)
            h_active = model.gru_cells[i](x_t, model.hidden_states[i])
            h_null = model.null_gru_cells[i](null_input, model.hidden_states[i])
            h_new = is_active * h_active + (1 - is_active) * h_null
            new_states.append(h_new)

        model.hidden_states = model._communication_step(new_states)

        combined = torch.cat(model.hidden_states, dim=-1)
        output = model.output_proj(combined)
        outputs.append(output[0])  # (vocab,)

    return torch.stack(outputs, dim=0)  # (chunk_len, vocab)


def eval_extreme_chain(model, model_name, seq_len, seed, n_sequences=50, device='cuda',
                       progress=None, scale_label=''):
    """Evaluate chain accuracy at extreme length using chunked processing.

    Processes tokens in chunks (default 1000) for GPU efficiency.
    Saves per-sequence results for resume support.
    """
    global _interrupted

    CHUNK_SIZE = 1000  # Process 1000 tokens at a time

    # Check for partially completed sequences
    result_path = EXTREME_DIR / f"{model_name}_seed{seed}_{scale_label}.json"
    existing = load_json_safe(result_path)
    completed_seqs = existing.get('per_sequence', {})

    total_correct = 0
    total_tokens = 0
    seq_results = dict(completed_seqs)  # copy existing

    for seq_idx in range(n_sequences):
        if _interrupted:
            break

        seq_key = str(seq_idx)
        if seq_key in seq_results:
            # Already done - accumulate
            total_correct += seq_results[seq_key]['correct']
            total_tokens += seq_results[seq_key]['total']
            continue

        torch.manual_seed(seed * 10000 + seq_idx)
        model.eval()

        # Reset model state
        if hasattr(model, 'reset_state'):
            model.reset_state()
        if hasattr(model, '_frank'):
            model._frank._prev_brain_state = None
            model._frank.brain.reset_state()
        elif hasattr(model, '_prev_brain_state'):
            model._prev_brain_state = None

        seq_correct = 0
        seq_total = 0
        gru_hidden = None

        t0 = time.time()

        # Pre-generate ALL random inputs for this sequence (reproducible)
        all_inputs = torch.randint(0, 10, (seq_len,))

        # Compute ALL ground truth targets
        all_targets = torch.zeros(seq_len, dtype=torch.long)
        state = 0
        for t in range(seq_len):
            if t == 0:
                state = all_inputs[t].item()
            else:
                state = (state + all_inputs[t].item()) % 10
            all_targets[t] = state

        # Process in chunks
        with torch.no_grad():
            for chunk_start in range(0, seq_len, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, seq_len)
                chunk_inputs = all_inputs[chunk_start:chunk_end].unsqueeze(0).to(device)  # (1, chunk_len)
                chunk_targets = all_targets[chunk_start:chunk_end]

                if model_name == 'gru':
                    logits, gru_hidden = eval_chunk_gru(model, chunk_inputs, gru_hidden, device)
                elif model_name in ('frank', 'frank_no_laterals'):
                    logits = eval_chunk_frank(model, chunk_inputs, device)
                elif model_name in ('modular', 'modular_memory'):
                    logits = eval_chunk_modular(model, chunk_inputs, device)
                elif model_name == 'rims':
                    logits = eval_chunk_rims(model, chunk_inputs, device)
                else:
                    raise ValueError(f"Extreme eval not supported for {model_name}")

                preds = logits.argmax(dim=-1).cpu()  # (chunk_len,)
                chunk_correct = (preds == chunk_targets).sum().item()
                seq_correct += chunk_correct
                seq_total += (chunk_end - chunk_start)

                # Progress reporting every ~100K tokens
                if seq_total % 100000 < CHUNK_SIZE and seq_total > 0:
                    elapsed = time.time() - t0
                    tps = seq_total / max(elapsed, 0.001)
                    eta = (seq_len - seq_total) / max(tps, 1)
                    curr_acc = seq_correct / seq_total
                    print(f"    seq {seq_idx+1}/{n_sequences}: {seq_total:,}/{seq_len:,} tokens "
                          f"({tps:,.0f} tok/s, ETA {eta:.0f}s, acc={curr_acc:.6f})")

        elapsed = time.time() - t0
        seq_acc = seq_correct / seq_total

        seq_results[seq_key] = {
            'correct': seq_correct,
            'total': seq_total,
            'accuracy': seq_acc,
            'elapsed_seconds': elapsed,
        }
        total_correct += seq_correct
        total_tokens += seq_total

        # Save immediately
        save_data = {
            'model': model_name,
            'seed': seed,
            'scale': scale_label,
            'seq_len': seq_len,
            'n_sequences': n_sequences,
            'per_sequence': seq_results,
            'aggregate_accuracy': total_correct / max(total_tokens, 1),
            'sequences_completed': len(seq_results),
        }
        save_json_atomic(save_data, result_path)

        print(f"    seq {seq_idx+1}/{n_sequences}: acc={seq_acc:.6f} ({elapsed:.1f}s)")

    overall_acc = total_correct / max(total_tokens, 1)
    return overall_acc


# ===============================================================================
# STANDARD GENERALIZATION EVALUATION
# ===============================================================================

def eval_standard_generalization(model, model_name, scale, seed, device,
                                  task_name='chain', quick=False):
    """Evaluate on sequences at scalex training length for any task."""
    min_len = TRAIN_MAX_LEN * scale - (TRAIN_MAX_LEN - 5)  # rough
    max_len = TRAIN_MAX_LEN * scale
    n_samples = 200 if quick else 5000

    task = create_task_by_name(task_name, TaskConfig())
    dataset = task.generate_dataset(
        size=n_samples,
        min_len=min_len,
        max_len=max_len,
        seed=seed
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model.eval()
    metrics = evaluate_on_dataloader(model, loader, device)
    return metrics


# ===============================================================================
# PHASE RUNNERS
# ===============================================================================

def run_phase1(args):
    """Phase 1: Train all models on ALL tasks with 10 seeds."""
    global _interrupted
    progress = ProgressTracker()
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    models = args.models.split(',') if args.models else ALL_MODELS
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else ALL_SEEDS
    tasks = ALL_TASKS

    print(f"\n{'#'*60}")
    print(f"# PHASE 1: Training {len(models)} models x {len(tasks)} tasks x {len(seeds)} seeds")
    print(f"# Tasks: {tasks}")
    print(f"# Device: {device}")
    print(f"{'#'*60}")

    all_results = load_json_safe(RESULTS_DIR / 'phase1_training.json')

    for task_name in tasks:
        for model_name in models:
            for seed in seeds:
                if _interrupted:
                    print("\nInterrupted. Re-run to resume.")
                    return

                key = f"{task_name}_{model_name}_seed{seed}"
                if progress.is_done('phase1', key):
                    print(f"\n  [SKIP] {key} - already completed")
                    continue

                progress.mark_in_progress('phase1', key)
                result = train_model(model_name, seed, device, task_name=task_name,
                                     quick=args.quick)

                if result is not None:
                    all_results[key] = result
                    save_json_atomic(all_results, RESULTS_DIR / 'phase1_training.json')
                    progress.mark_done('phase1', key)
                    print(f"  [DONE] {key}")


def run_phase2(args):
    """Phase 2: Extreme chain generalization."""
    global _interrupted
    progress = ProgressTracker()
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    # Models to test at extreme scales
    models = ['frank', 'gru', 'frank_no_laterals']
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else ALL_SEEDS
    scales = [10, 100, 1000] if args.quick else EXTREME_SCALES
    n_seqs = 5 if args.quick else 10

    EXTREME_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# PHASE 2: Extreme Chain Generalization")
    print(f"# Models: {models}")
    print(f"# Scales: {scales}")
    print(f"# Seeds: {seeds}")
    print(f"# Sequences per (model, seed, scale): {n_seqs}")
    print(f"{'#'*60}")

    summary = load_json_safe(RESULTS_DIR / 'phase2_extreme_gen.json')

    for model_name in models:
        for seed in seeds:
            if _interrupted:
                print("\nInterrupted. Re-run to resume.")
                return

            # Load checkpoint
            ckpt_path = CHECKPOINTS_DIR / f"best_{model_name}_seed{seed}_chain.pt"
            if not ckpt_path.exists():
                print(f"\n  [SKIP] {model_name} seed={seed} - no checkpoint")
                continue

            model = create_model(model_name, 10, 10)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device)

            for scale in scales:
                if _interrupted:
                    print("\nInterrupted. Re-run to resume.")
                    return

                key = f"{model_name}_seed{seed}_{scale}x"
                if progress.is_done('phase2', key):
                    print(f"\n  [SKIP] {key} - already completed")
                    continue

                seq_len = TRAIN_MAX_LEN * scale
                print(f"\n  Evaluating {model_name} seed={seed} at {scale}x ({seq_len:,} tokens)...")

                acc = eval_extreme_chain(
                    model, model_name, seq_len, seed,
                    n_sequences=n_seqs, device=device,
                    progress=progress, scale_label=f"{scale}x"
                )

                if not _interrupted:
                    if model_name not in summary:
                        summary[model_name] = {}
                    if str(seed) not in summary[model_name]:
                        summary[model_name][str(seed)] = {}
                    summary[model_name][str(seed)][f"{scale}x"] = acc

                    save_json_atomic(summary, RESULTS_DIR / 'phase2_extreme_gen.json')
                    progress.mark_done('phase2', key)
                    print(f"  [DONE] {key}: accuracy={acc:.6f}")


def run_phase3(args):
    """Phase 3: Lesion study on ALL tasks (all seeds)."""
    global _interrupted
    progress = ProgressTracker()
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    models = ALL_MODELS
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else ALL_SEEDS
    tasks = ALL_TASKS
    n_patterns = 3 if args.quick else 10

    print(f"\n{'#'*60}")
    print(f"# PHASE 3: Lesion Study (all tasks)")
    print(f"# Tasks: {tasks}")
    print(f"# Seeds: {seeds}")
    print(f"{'#'*60}")

    for task_name in tasks:
        if _interrupted:
            break

        # Task for eval
        task_config = TaskConfig()
        if args.quick:
            task_config.test_size = 500
        task = create_task_by_name(task_name, task_config)
        dataloaders = task.get_dataloaders(seed=42)
        test_loader = dataloaders['test']

        # === Global Lesion (all models x all seeds) ===
        global_file = RESULTS_DIR / f'phase3_global_lesion_{task_name}.json'
        global_results = load_json_safe(global_file)

        for model_name in models:
            if _interrupted:
                break

            for seed in seeds:
                if _interrupted:
                    break

                key = f"global_{task_name}_{model_name}_seed{seed}"
                if progress.is_done('phase3', key):
                    print(f"\n  [SKIP] Global lesion {task_name}/{model_name} seed={seed} - already completed")
                    continue

                ckpt_path = CHECKPOINTS_DIR / f"best_{model_name}_seed{seed}_{task_name}.pt"
                if not ckpt_path.exists():
                    print(f"\n  [SKIP] {model_name} seed={seed} {task_name} - no checkpoint")
                    continue

                model = create_model(model_name, task.input_vocab_size,
                                     task.output_vocab_size, max_seq_len=task.max_seq_len)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])

                print(f"\n  Global lesion: {task_name}/{model_name} seed={seed}")
                seed_results = {}

                for damage_pct in DAMAGE_LEVELS:
                    scores = []
                    for pattern_seed in range(n_patterns):
                        damaged = apply_damage(model, damage_pct, pattern_seed)
                        damaged = damaged.to(device)
                        metrics = evaluate_on_dataloader(damaged, test_loader, device)
                        scores.append(metrics['token_accuracy'])

                    seed_results[str(damage_pct)] = {
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores)),
                        'scores': scores,
                    }
                    print(f"    {damage_pct}% damage: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

                if model_name not in global_results:
                    global_results[model_name] = {}
                global_results[model_name][str(seed)] = seed_results
                save_json_atomic(global_results, global_file)
                progress.mark_done('phase3', key)

        # === Targeted Lesion (Frank x all seeds, per task) ===
        targeted_file = RESULTS_DIR / f'phase3_targeted_lesion_{task_name}.json'
        targeted_results = load_json_safe(targeted_file)

        for seed in seeds:
            if _interrupted:
                break

            key = f"targeted_{task_name}_frank_seed{seed}"
            if progress.is_done('phase3', key):
                print(f"\n  [SKIP] Targeted lesion {task_name}/Frank seed={seed} - already completed")
                continue

            ckpt_path = CHECKPOINTS_DIR / f"best_frank_seed{seed}_{task_name}.pt"
            if not ckpt_path.exists():
                print(f"\n  [SKIP] Frank seed={seed} {task_name} - no checkpoint")
                continue

            model = create_model('frank', task.input_vocab_size,
                                 task.output_vocab_size, max_seq_len=task.max_seq_len)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])

            print(f"\n  Targeted lesion: {task_name}/Frank seed={seed}")
            seed_targeted = {}

            for component in FRANK_COMPONENTS:
                comp_results = {}
                for damage_pct in TARGETED_DAMAGE_LEVELS:
                    scores = []
                    for pattern_seed in range(n_patterns):
                        damaged = apply_targeted_damage(model, component, damage_pct, pattern_seed)
                        damaged = damaged.to(device)
                        metrics = evaluate_on_dataloader(damaged, test_loader, device)
                        scores.append(metrics['token_accuracy'])

                    comp_results[str(damage_pct)] = {
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores)),
                        'scores': scores,
                    }
                    print(f"    {component} {damage_pct}%: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

                seed_targeted[component] = comp_results

            targeted_results[str(seed)] = seed_targeted
            save_json_atomic(targeted_results, targeted_file)
            progress.mark_done('phase3', key)


def run_phase4(args):
    """Phase 4: Standard multi-seed generalization (2x-10x) on ALL tasks."""
    global _interrupted
    progress = ProgressTracker()
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    models = ALL_MODELS
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else ALL_SEEDS
    tasks = ALL_TASKS
    scales = STANDARD_SCALES

    print(f"\n{'#'*60}")
    print(f"# PHASE 4: Standard Generalization (2x-10x) on all tasks")
    print(f"# Tasks: {tasks}")
    print(f"{'#'*60}")

    for task_name in tasks:
        if _interrupted:
            break

        task = create_task_by_name(task_name, TaskConfig())
        results_file = RESULTS_DIR / f'phase4_standard_gen_{task_name}.json'
        results = load_json_safe(results_file)

        for model_name in models:
            for seed in seeds:
                for scale in scales:
                    if _interrupted:
                        print("\nInterrupted. Re-run to resume.")
                        return

                    key = f"{task_name}_{model_name}_seed{seed}_{scale}x"
                    if progress.is_done('phase4', key):
                        continue

                    ckpt_path = CHECKPOINTS_DIR / f"best_{model_name}_seed{seed}_{task_name}.pt"
                    if not ckpt_path.exists():
                        continue

                    model = create_model(model_name, task.input_vocab_size,
                                         task.output_vocab_size,
                                         max_seq_len=TRAIN_MAX_LEN * scale + 10)
                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

                    # Handle positional embedding size mismatch for transformer
                    state_dict = ckpt['model_state_dict']
                    if 'pos_encoder.pos_embedding.weight' in state_dict:
                        saved_pe = state_dict['pos_encoder.pos_embedding.weight']
                        model_pe = model.state_dict()['pos_encoder.pos_embedding.weight']
                        if saved_pe.shape[0] != model_pe.shape[0]:
                            new_pe = model_pe.clone()
                            min_len = min(saved_pe.shape[0], model_pe.shape[0])
                            new_pe[:min_len] = saved_pe[:min_len]
                            state_dict['pos_encoder.pos_embedding.weight'] = new_pe

                    model.load_state_dict(state_dict)
                    model = model.to(device)

                    metrics = eval_standard_generalization(
                        model, model_name, scale, seed, device,
                        task_name=task_name, quick=args.quick
                    )

                    if model_name not in results:
                        results[model_name] = {}
                    if str(seed) not in results[model_name]:
                        results[model_name][str(seed)] = {}
                    results[model_name][str(seed)][f"{scale}x"] = metrics['token_accuracy']

                    save_json_atomic(results, results_file)
                    progress.mark_done('phase4', key)

                    print(f"  {task_name}/{model_name} seed={seed} {scale}x: {metrics['token_accuracy']:.4f}")

        # Print summary table per task
        if not _interrupted:
            print(f"\n{'='*80}")
            print(f"STANDARD GENERALIZATION SUMMARY: {task_name.upper()} (mean across seeds)")
            print(f"{'='*80}")
            header = f"{'Model':20s}" + "".join(f"  {s}x" for s in scales)
            print(header)
            print("-" * len(header))

            for model_name in models:
                row = f"{model_name:20s}"
                for scale in scales:
                    accs = []
                    for seed in seeds:
                        if model_name in results and str(seed) in results[model_name]:
                            val = results[model_name][str(seed)].get(f"{scale}x")
                            if val is not None:
                                accs.append(val)
                    if accs:
                        row += f"  {np.mean(accs):.4f}"
                    else:
                        row += f"     --"
                print(row)


# ===============================================================================
# MAIN
# ===============================================================================

def main():
    parser = argparse.ArgumentParser(description="FRANK Papers Experiment Runner")
    parser.add_argument('--phase', type=str, default='all',
                        help='Phase to run: 1, 2, 3, 4, or all')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model names (phase 1)')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index to use')
    parser.add_argument('--quick', action='store_true',
                        help='Quick smoke test mode')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='Custom results directory (for parallel runs)')
    args = parser.parse_args()

    global RESULTS_DIR, CHECKPOINTS_DIR, EXTREME_DIR
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)
        CHECKPOINTS_DIR = RESULTS_DIR / 'checkpoints'
        EXTREME_DIR = RESULTS_DIR / 'extreme'

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"FRANK Papers Experiment Runner")
    print(f"  Phase: {args.phase}")
    print(f"  GPU: {args.gpu}")
    print(f"  Quick mode: {args.quick}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Checkpoints: {CHECKPOINTS_DIR}")

    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(args.gpu)}")
    else:
        print(f"  WARNING: No CUDA - running on CPU (this will be very slow)")

    phases = ['1', '2', '3', '4'] if args.phase == 'all' else [args.phase]

    for phase in phases:
        if _interrupted:
            break
        if phase == '1':
            run_phase1(args)
        elif phase == '2':
            run_phase2(args)
        elif phase == '3':
            run_phase3(args)
        elif phase == '4':
            run_phase4(args)
        else:
            print(f"Unknown phase: {phase}")

    if not _interrupted:
        print(f"\n{'='*60}")
        print("ALL DONE! Results saved to paper1_results/")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
