"""Integration tests for dataset, forward pass, and gradients.

These tests exercise the building blocks used by paper_experiments.py.
The full reproduction pipeline (training, lesion, generalization) lives in
paper_experiments.py itself; run `python paper_experiments.py --quick` for
an end-to-end smoke test.
"""
import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataset(unittest.TestCase):
    """Test dataset generation and loading."""

    def test_copy_task_dataset(self):
        """Test copy task generates valid data."""
        from frank.tasks import create_task, TaskConfig

        config = TaskConfig(train_size=100, val_size=20, test_size=20, test_long_size=20)
        task = create_task('copy', config)
        dataloaders = task.get_dataloaders(seed=42)

        self.assertIn('train', dataloaders)
        self.assertIn('val', dataloaders)
        self.assertIn('test', dataloaders)

        inputs, targets, lengths = next(iter(dataloaders['train']))
        self.assertEqual(inputs.dim(), 2)
        self.assertEqual(targets.dim(), 2)

        self.assertTrue(inputs.max() <= task.input_vocab_size)
        self.assertTrue((targets != -100).any())

    def test_delayed_recall_dataset(self):
        """Test delayed recall task generates valid data."""
        from frank.tasks import create_task, TaskConfig

        config = TaskConfig(train_size=100, val_size=20, test_size=20, test_long_size=20)
        task = create_task('recall', config)
        dataloaders = task.get_dataloaders(seed=42)

        inputs, targets, lengths = next(iter(dataloaders['train']))

        ignore_count = (targets == -100).sum().item()
        self.assertGreater(ignore_count, 0, "Should have ignore tokens for non-output positions")

    def test_running_sum_dataset(self):
        """Test running sum task generates valid data."""
        from frank.tasks import create_task, TaskConfig

        config = TaskConfig(train_size=100, val_size=20, test_size=20, test_long_size=20)
        task = create_task('sum', config)
        dataloaders = task.get_dataloaders(seed=42)

        inputs, targets, lengths = next(iter(dataloaders['train']))

        self.assertTrue(targets.max() <= task.output_vocab_size)


class TestForwardPass(unittest.TestCase):
    """Test forward pass for the four base models in frank/models/."""

    def setUp(self):
        self.batch_size = 4
        self.seq_len = 20
        self.input_dim = 12
        self.output_dim = 12
        self.max_seq_len = 50

    def _test_model(self, model_name: str):
        from frank.models import create_model

        model = create_model(
            model_name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            max_seq_len=self.max_seq_len
        )

        x = torch.randint(0, self.input_dim, (self.batch_size, self.seq_len))

        model.eval()
        with torch.no_grad():
            output = model(x)

        expected_shape = (self.batch_size, self.seq_len, self.output_dim)
        self.assertEqual(output.shape, expected_shape)

        self.assertFalse(torch.isnan(output).any(), f"{model_name} has NaN in output")
        self.assertFalse(torch.isinf(output).any(), f"{model_name} has Inf in output")

    def test_transformer_forward(self):
        self._test_model('transformer')

    def test_gru_forward(self):
        self._test_model('gru')

    def test_modular_forward(self):
        self._test_model('modular')

    def test_frank_forward(self):
        self._test_model('frank')


class TestGradients(unittest.TestCase):
    """Test gradient flow for the four base models in frank/models/."""

    def _test_gradient_flow(self, model_name: str):
        from frank.models import create_model

        model = create_model(
            model_name,
            input_dim=12,
            output_dim=12,
            max_seq_len=50
        )

        x = torch.randint(0, 12, (2, 10))
        target = torch.randint(0, 12, (2, 10))

        model.train()
        output = model(x)

        loss = nn.CrossEntropyLoss(ignore_index=-100)(
            output.reshape(-1, 12),
            target.reshape(-1)
        )

        loss.backward()

        has_gradient = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradient = True
                break

        self.assertTrue(has_gradient, f"{model_name} has no flowing gradients")

        for name, param in model.named_parameters():
            if param.grad is not None:
                self.assertFalse(
                    torch.isnan(param.grad).any(),
                    f"{model_name} has NaN gradients in {name}"
                )

    def test_transformer_gradients(self):
        self._test_gradient_flow('transformer')

    def test_gru_gradients(self):
        self._test_gradient_flow('gru')

    def test_modular_gradients(self):
        self._test_gradient_flow('modular')

    def test_frank_gradients(self):
        self._test_gradient_flow('frank')


class TestIgnoreIndex(unittest.TestCase):
    """Test that ignore_index=-100 works correctly."""

    def test_copy_task_ignore_positions(self):
        """Test that copy task uses -100 for ignore positions."""
        from frank.tasks import create_task, TaskConfig

        config = TaskConfig(train_size=100, val_size=20, test_size=20, test_long_size=20)
        task = create_task('copy', config)

        inp, tgt = task.generate_sample(seq_len=5, seed=42)

        ignore_mask = tgt == -100
        self.assertTrue(ignore_mask[:6].all(), "First positions should be ignored")
        self.assertFalse(ignore_mask[6:].any(), "Output positions should not be ignored")

    def test_loss_ignores_pad_tokens(self):
        """Test that CrossEntropyLoss ignores -100 tokens."""
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        logits = torch.randn(4, 10, 12)
        targets = torch.randint(0, 12, (4, 10))
        targets[:, :5] = -100

        loss = criterion(logits.reshape(-1, 12), targets.reshape(-1))

        self.assertFalse(torch.isnan(loss), "Loss should not be NaN")
        self.assertFalse(torch.isinf(loss), "Loss should not be Inf")


if __name__ == '__main__':
    unittest.main()
