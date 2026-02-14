"""Shape and gradient-flow tests for the InceptionTime + Attention model."""

from __future__ import annotations

import pytest
import torch

from arrhythmia.models.attention import TemporalAttention
from arrhythmia.models.inception_time import InceptionBlock
from arrhythmia.models.inception_time_attention import InceptionTimeAttention


# InceptionBlock
class TestInceptionBlock:
    def test_output_shape_first_block(self):
        block = InceptionBlock(in_channels=12, num_filters=32, use_residual=True)
        x = torch.randn(2, 12, 1000)
        out = block(x)
        assert out.shape == (2, 128, 1000)

    def test_output_shape_subsequent_block(self):
        block = InceptionBlock(in_channels=128, num_filters=32, use_residual=False)
        x = torch.randn(2, 128, 1000)
        out = block(x)
        assert out.shape == (2, 128, 1000)

    def test_temporal_dimension_preserved(self):
        """Output T must equal input T regardless of kernel sizes."""
        for T in [500, 1000]:
            block = InceptionBlock(in_channels=12, use_residual=True)
            x = torch.randn(1, 12, T)
            assert block(x).shape[-1] == T

    def test_gradient_flows(self):
        block = InceptionBlock(in_channels=12, use_residual=True)
        x = torch.randn(2, 12, 1000, requires_grad=True)
        loss = block(x).sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0


# TemporalAttention
class TestTemporalAttention:
    def test_context_shape(self):
        attn = TemporalAttention(hidden_size=128, attention_hidden=64)
        h = torch.randn(4, 128, 1000)
        context, alpha = attn(h)
        assert context.shape == (4, 128)
        assert alpha.shape == (4, 1000)

    def test_attention_weights_sum_to_one(self):
        attn = TemporalAttention(hidden_size=128, attention_hidden=64)
        h = torch.randn(4, 128, 1000)
        _, alpha = attn(h)
        sums = alpha.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)


# InceptionTimeAttention
class TestInceptionTimeAttention:
    @pytest.fixture
    def model(self):
        return InceptionTimeAttention(
            in_channels=12,
            num_classes=5,
            num_filters=32,
            bottleneck_size=32,
            num_blocks=3,
            attention_hidden=64,
        )

    def test_output_shapes(self, model):
        x = torch.randn(4, 12, 1000)
        logits, alpha = model(x)
        assert logits.shape == (4, 5)
        assert alpha.shape == (4, 1000)

    def test_predict_proba_range(self, model):
        x = torch.randn(4, 12, 1000)
        probs = model.predict_proba(x)
        assert probs.shape == (4, 5)
        assert probs.min().item() >= 0.0
        assert probs.max().item() <= 1.0

    def test_gradient_flows_end_to_end(self, model):
        x = torch.randn(2, 12, 1000)
        labels = torch.randint(0, 2, (2, 5)).float()
        logits, _ = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_parameter_count_reasonable(self, model):
        total = sum(p.numel() for p in model.parameters())
        # Should be in the 300k–600k range for the default config
        assert 200_000 < total < 800_000, f"Unexpected param count: {total}"

    def test_batch_size_one(self, model):
        """BatchNorm can behave differently with B=1 in eval mode."""
        model.eval()
        x = torch.randn(1, 12, 1000)
        with torch.no_grad():
            logits, alpha = model(x)
        assert logits.shape == (1, 5)
        assert alpha.shape == (1, 1000)
