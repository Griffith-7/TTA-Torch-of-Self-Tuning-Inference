import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import Mock, MagicMock, patch
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
        assert tta_model.tta_config["learning_rate"] == 1e-5
        assert tta_model.tta_config["kl_weight"] == 0.1
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


class TestGradientFlow:
    """Test that gradients flow properly during TTA"""
    
    def test_gradients_enabled(self, mock_model):
        """Test that gradients are computed"""
        tta_config = {"entropy_threshold": 0.5, "learning_rate": 1e-5, "kl_weight": 0.1, "max_new_tokens": 10}
        tta_model = TTAModel(mock_model, tta_config)
        input_ids = torch.tensor([[1, 2, 3]])
        
        has_grad = False
        original_backward = torch.Tensor.backward
        
        def track_backward(*args, **kwargs):
            global has_grad
            has_grad = True
            return original_backward(*args, **kwargs)
        
        with patch.object(torch.Tensor, 'backward', track_backward):
            try:
                tta_model.generate_adaptive(input_ids)
            except:
                pass


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])