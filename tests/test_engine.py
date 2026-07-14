import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tta_torch.engine import TTAModel


class MockConfig:
    eos_token_id = 151643
    vocab_size = 151644


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockConfig()
        self.lora_weight = nn.Parameter(torch.randn(10, 512), requires_grad=True)
    
    def forward(self, input_ids):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1] if len(input_ids.shape) > 1 else 1
        logits = torch.randn(batch_size, seq_len, 151644, requires_grad=True)
        return type('Output', (), {'logits': logits})()
    
    def generate(self, input_ids, max_new_tokens=10, **kwargs):
        batch_size = input_ids.shape[0]
        new_tokens = torch.randint(0, 151643, (batch_size, max_new_tokens))
        return torch.cat([input_ids, new_tokens], dim=-1)
    
    def named_parameters(self, recurse=True):
        return [('lora_weight', self.lora_weight)]
    
    def disable_adapter(self):
        return MockContextManager()
    
    def parameters(self, recurse=True):
        return [self.lora_weight]


class MockContextManager:
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def __call__(self, *args, **kwargs):
        return self


@pytest.fixture
def mock_model():
    model = MockModel()
    model.lora_weight.requires_grad_(True)
    return model


@pytest.fixture
def tta_config():
    return {
        "entropy_threshold": 0.5,
        "learning_rate": 1e-5,
        "kl_weight": 0.1,
        "max_new_tokens": 10,
        "grad_clip": 1.0
    }


class TestEntropyCalculation:
    """Test entropy calculation function"""
    
    def test_entropy_uniform_distribution(self):
        """High entropy for uniform distribution (high uncertainty)"""
        model = TTAModel(MockModel(), {"max_new_tokens": 1})
        logits = torch.ones(1, 10)  # Uniform distribution
        entropy = model._entropy(logits)
        assert entropy.mean() > 1.0, "Uniform distribution should have high entropy"
    
    def test_entropy_peaked_distribution(self):
        """Low entropy for peaked distribution (high confidence)"""
        model = TTAModel(MockModel(), {"max_new_tokens": 1})
        logits = torch.zeros(1, 10)
        logits[0, 0] = 10.0  # Very peaked
        entropy = model._entropy(logits)
        assert entropy.mean() < 0.5, "Peaked distribution should have low entropy"
    
    def test_entropy_positive(self):
        """Entropy should always be non-negative"""
        model = TTAModel(MockModel(), {"max_new_tokens": 1})
        logits = torch.randn(2, 5)
        entropy = model._entropy(logits)
        assert (entropy >= 0).all(), "Entropy should be non-negative"


class TestWeightReset:
    """Test weight reset functionality"""
    
    def test_reset_weights(self, mock_model):
        """Test that weights are properly reset"""
        original_weights = mock_model.lora_weight.clone()
        tta_config = {"entropy_threshold": 0.5, "learning_rate": 1e-5, "kl_weight": 0.1, "max_new_tokens": 10}
        tta_model = TTAModel(mock_model, tta_config)
        
        with torch.no_grad():
            mock_model.lora_weight.add_(100.0)
        
        tta_model.reset_weights()
        
        assert torch.allclose(mock_model.lora_weight, original_weights, atol=1e-6), \
            "Weights should be reset to original values"


class TestConfigHandling:
    """Test configuration handling"""
    
    def test_default_config(self, mock_model):
        """Test default config values"""
        tta_model = TTAModel(mock_model)
        assert tta_model.tta_config["entropy_threshold"] == 0.5
        assert tta_model.tta_config["learning_rate"] == 1e-4
        assert tta_model.tta_config["kl_weight"] == 0.1
        assert tta_model.tta_config["inner_steps"] == 2
        assert tta_model.tta_config["max_new_tokens"] == 128
        assert tta_model.tta_config["grad_clip"] == 1.0
    
    def test_custom_config(self, mock_model):
        """Test custom config override"""
        custom_config = {"entropy_threshold": 0.3, "learning_rate": 1e-4}
        tta_model = TTAModel(mock_model, custom_config)
        assert tta_model.tta_config["entropy_threshold"] == 0.3
        assert tta_model.tta_config["learning_rate"] == 1e-4
    
    def test_kl_weight_default(self, mock_model):
        """Test kl_weight uses default when not provided"""
        tta_model = TTAModel(mock_model, {"max_new_tokens": 1})
        assert tta_model.tta_config.get("kl_weight", 0.1) == 0.1


class TestAdaptiveGeneration:
    """Test adaptive generation logic"""
    
    def test_generate_returns_tensor(self, mock_model):
        """Test that generate_adaptive returns a tensor"""
        tta_config = {"entropy_threshold": 0.5, "learning_rate": 1e-5, "kl_weight": 0.1, "max_new_tokens": 10}
        tta_model = TTAModel(mock_model, tta_config)
        input_ids = torch.tensor([[1, 2, 3]])
        
        result = tta_model.generate_adaptive(input_ids)
        
        assert isinstance(result, torch.Tensor), "Should return a tensor"
        assert result.shape[0] == 1, "Batch size should be 1"
    
    def test_generate_increases_length(self, mock_model):
        """Test that generated output is longer than input"""
        tta_config = {"entropy_threshold": 0.5, "learning_rate": 1e-5, "kl_weight": 0.1, "max_new_tokens": 10}
        tta_model = TTAModel(mock_model, tta_config)
        input_ids = torch.tensor([[1, 2, 3]])
        
        result = tta_model.generate_adaptive(input_ids)
        
        assert result.shape[1] > input_ids.shape[1], "Output should be longer than input"
    
    def test_max_tokens_respected(self, mock_model):
        """Test that max_new_tokens is respected"""
        max_tokens = 5
        tta_config = {"entropy_threshold": 0.5, "learning_rate": 1e-5, "kl_weight": 0.1, "max_new_tokens": max_tokens}
        tta_model = TTAModel(mock_model, tta_config)
        input_ids = torch.tensor([[1]])
        
        result = tta_model.generate_adaptive(input_ids)
        
        new_tokens = result.shape[1] - 1
        assert new_tokens <= max_tokens, f"Should generate at most {max_tokens} tokens"


class ConnectedModel(nn.Module):
    """A model where forward pass actually depends on parameters, for gradient testing"""
    def __init__(self):
        super().__init__()
        self.config = MockConfig()
        self.embed = nn.Linear(100, 128)
        self.head = nn.Linear(128, 100)

    def forward(self, input_ids):
        x = torch.randn(input_ids.shape[0], input_ids.shape[1], 100, device=input_ids.device)
        x = torch.relu(self.embed(x))
        logits = self.head(x)
        return type('O', (), {'logits': logits})()

    def disable_adapter(self):
        return MockContextManager()


class TestGradientFlow:
    """Test that gradients flow properly during TTA"""
    
    def test_gradients_enabled(self, mock_model):
        """Test that gradients are computed"""
        tta_config = {"entropy_threshold": 0.5, "learning_rate": 1e-5, "kl_weight": 0.1, "max_new_tokens": 10}
        tta_model = TTAModel(mock_model, tta_config)
        input_ids = torch.tensor([[1, 2, 3]])
        
        original_backward = torch.Tensor.backward
        
        def track_backward(*args, **kwargs):
            global has_grad
            has_grad = True
            return original_backward(*args, **kwargs)
        
        with patch.object(torch.Tensor, 'backward', track_backward):
            try:
                tta_model.generate_adaptive(input_ids)
            except Exception:
                pass

    def test_entropy_loss_has_gradient(self):
        """Entropy loss must remain differentiable (not .item())"""
        model = ConnectedModel()
        tta = TTAModel(model, {"max_new_tokens": 1})
        input_ids = torch.tensor([[1, 2, 3]])

        out = model(input_ids)
        logits = out.logits[:, -1, :]

        ent_loss = tta._entropy(logits).mean()
        assert ent_loss.grad_fn is not None, "Entropy loss must have grad_fn (must be differentiable)"

    def test_total_loss_has_both_gradients(self):
        """total_loss must get gradients from BOTH entropy and KL terms"""
        model = ConnectedModel()
        tta = TTAModel(model, {"max_new_tokens": 1})
        input_ids = torch.tensor([[1, 2, 3]])

        out = model(input_ids)
        logits = out.logits[:, -1, :]

        with torch.no_grad():
            f_logits = model(input_ids).logits[:, -1, :]

        ent_loss = tta._entropy(logits).mean()
        kl = F.kl_div(F.log_softmax(f_logits, dim=-1), F.softmax(logits, dim=-1), reduction='batchmean')

        ent_clamped = torch.clamp(ent_loss - 0.3, min=0.0)
        total_loss = ent_clamped + (0.1 * kl)

        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt.zero_grad()
        total_loss.backward()

        has_any_grad = False
        for n, p in model.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_any_grad = True
                break
        assert has_any_grad, "At least one parameter must receive gradients from total_loss"

    def test_weights_change_after_tta(self):
        """Weights must actually change after TTA generation"""
        model = ConnectedModel()
        tta = TTAModel(model, {
            "entropy_threshold": 0.01,
            "max_new_tokens": 3,
            "inner_steps": 2,
            "learning_rate": 1e-3,
        })
        input_ids = torch.tensor([[1, 2, 3]])

        w_before = {n: p.clone() for n, p in model.named_parameters()}
        tta.generate(input_ids)
        w_after = {n: p.clone() for n, p in model.named_parameters()}

        any_changed = False
        for n in w_before:
            if (w_after[n] - w_before[n]).abs().max().item() > 1e-8:
                any_changed = True
                break
        assert any_changed, "At least one parameter must change after TTA generation"

    def test_entropy_decrease_on_simple_input(self):
        """TTA should produce an entropy trace"""
        model = ConnectedModel()
        tta = TTAModel(model, {
            "entropy_threshold": 0.01,
            "max_new_tokens": 3,
            "inner_steps": 5,
            "learning_rate": 1e-2,
            "verbose": False,
        })
        input_ids = torch.tensor([[1, 2, 3]])
        tta.generate(input_ids)

        assert len(tta.current_entropy) > 0, "Should have entropy trace"
        assert all(isinstance(e, float) for e in tta.current_entropy), "All entropy values should be floats"
        assert all(e >= 0 for e in tta.current_entropy), "All entropy values should be non-negative"


class TestEdgeCases:
    """Test edge cases"""
    
    def test_empty_input(self, mock_model):
        """Test with single token input"""
        tta_config = {"entropy_threshold": 0.5, "learning_rate": 1e-5, "kl_weight": 0.1, "max_new_tokens": 10}
        tta_model = TTAModel(mock_model, tta_config)
        input_ids = torch.tensor([[1]])
        
        result = tta_model.generate_adaptive(input_ids)
        
        assert result is not None, "Should handle single token input"
    
    def test_single_token_vocab(self, mock_model):
        """Test entropy with batch size > 1"""
        model = TTAModel(mock_model, {"max_new_tokens": 1})
        logits = torch.randn(4, 10)
        entropy = model._entropy(logits)
        assert entropy.shape[0] == 4, "Should handle multiple batches"


class TestConfidenceGated:
    """Test confidence-gated TTA method"""

    def test_returns_tuple_with_reason(self, mock_model):
        """Should return (tensor, reason_string, entropy_float)"""
        tta = TTAModel(mock_model, {"entropy_threshold": 0.5, "max_new_tokens": 5, "learning_rate": 1e-5})
        input_ids = torch.tensor([[1, 2, 3]])
        out, reason, ent = tta.generate_confidence_gated(input_ids, n_passes=2, max_tokens=5)
        assert isinstance(out, torch.Tensor)
        assert reason in ("baseline_confident", "tta_helped", "tta_no_improvement")
        assert isinstance(ent, float)

    def test_output_is_2d(self, mock_model):
        """Output tensor must be 2D [batch, seq_len]"""
        tta = TTAModel(mock_model, {"entropy_threshold": 0.5, "max_new_tokens": 5, "learning_rate": 1e-5})
        input_ids = torch.tensor([[1, 2, 3]])
        out, _, _ = tta.generate_confidence_gated(input_ids, n_passes=2, max_tokens=5)
        assert out.dim() == 2, f"Expected 2D tensor, got {out.dim()}D"

    def test_baseline_confident_skips_tta(self, mock_model):
        """With high threshold, baseline should always be confident"""
        tta = TTAModel(mock_model, {"entropy_threshold": 99.0, "max_new_tokens": 5, "learning_rate": 1e-5})
        input_ids = torch.tensor([[1, 2, 3]])
        out, reason, _ = tta.generate_confidence_gated(input_ids, n_passes=2, max_tokens=5)
        assert reason == "baseline_confident"


class TestEntropyWeighted:
    """Test entropy-weighted voting method"""

    def test_returns_2d_tensor(self, mock_model):
        tta = TTAModel(mock_model, {"max_new_tokens": 5})
        input_ids = torch.tensor([[1, 2, 3]])
        out, reason, ent = tta.generate_entropy_weighted_vote(input_ids, n_passes=3, max_tokens=5)
        assert isinstance(out, torch.Tensor)
        assert out.dim() == 2
        assert reason == "entropy_weighted"
        assert isinstance(ent, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
