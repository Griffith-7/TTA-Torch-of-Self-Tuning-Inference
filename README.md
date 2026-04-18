# TTA-Torch: Dynamic Test-Time Adaptation for LLMs

A PyTorch inference wrapper that performs **real-time weight updates** during generation. Instead of treating inference as a static forward pass, TTA-Torch treats it as a dynamic optimization problem — the model adjusts its own LoRA adapters mid-generation based on its confidence level.

## How It Works

In standard LLM inference, model weights are frozen. In TTA-Torch, the model runs a micro-optimization loop on every token:

1. **Predict**: Forward pass produces next-token logits
2. **Measure Confidence**: Calculate Shannon Entropy $H(P)$ over the distribution
3. **Evaluate Drift**: If entropy exceeds a threshold, compute:
   - **Learning signal**: minimize entropy (more confident)
   - **Stability signal**: KL-Divergence vs frozen base (prevent drift)
4. **Update**: Perform multiple AdamW steps on LoRA adapters, then re-sample

```python
from tta_torch import TTAModel, load_tta_model

model, tokenizer = load_tta_model("Qwen/Qwen2.5-0.5B-Instruct", lora_rank=4)
tta = TTAModel(model, {
    "entropy_threshold": 0.4,
    "learning_rate": 1e-4,
    "inner_steps": 2,
    "verbose": True
})
output = tta.generate(input_ids)
```

## Features

- **Entropy-Triggered Updates**: Only adapts when the model is "confused" (high entropy)
- **Multi-Step TTA**: `inner_steps` parameter for multiple updates per trigger
- **AdamW Optimizer**: Better gradient handling than SGD for small step sizes
- **Consistency Shield**: KL-Divergence regularization against frozen base
- **Temperature Sampling**: Supports greedy (argmax) or stochastic sampling
- **Best-of-N Selection**: Generate multiple outputs, pick lowest entropy
- **Weight Reset**: Full rollback to original weights between tasks
- **4GB VRAM Compatible**: Uses 4-bit QLoRA + adapter toggling

## What's New (v2.0)

| Change | Before | After |
|--------|--------|-------|
| Optimizer | SGD | AdamW |
| Learning rate | 1e-5 | 1e-4 |
| Inner steps | 1 | 2 |
| LoRA rank | 2 | 4 |
| LoRA targets | q/k/v/o | + gate/up/down (MLP) |
| Temperature | No | Yes |

## Benchmark

Run the GSM8K benchmark:

```bash
cd examples
python benchmark_gsm8k.py
```

Results are saved to `benchmark_results.json`.

### Sample Results (Qwen 0.5B)

```
BASELINE: 0/5 (0.0%)
TTA:      0/5 (0.0%)
```

Note: 0.5B model is too small for GSM8K math reasoning. Results are expected to improve with 1.5B+ models.

### Verified Entropy Changes

```
Token 18: entropy=0.7338 -> 0.0010 (change: +99.9%)
Token 26: entropy=0.5470 -> 0.0013 (change: +99.8%)
```

**What this proves:**
- Entropy now decreases per update (30-100%)
- Multi-step updates compound the effect
- LoRA adapters express meaningful weight changes

## Architecture

```
Input → [Forward Pass] → Logits
                       ↓
               [Entropy Check]
                ↓           ↓
          (Low: Skip)   (High: Adapt)
                       ↓
           [Loss = Entropy + KL * kl_weight]
                       ↓
           [Inner Loop: 2 AdamW Steps]
                       ↓
               [Re-Forward Pass]
                       ↓
               [Temperature Sampling]
                       ↓
                 [Sample Token]
```

## Technical Specifications

| Component | Value |
|-----------|-------|
| Base Model | Any HuggingFace CausalLM |
| Tested On | Qwen2.5-0.5B-Instruct |
| Quantization | 4-bit NF4 with double quantization |
| Adapter | LoRA (rank 4, targeting q/k/v/o + MLP) |
| Optimizer | AdamW (lr=1e-4, betas=(0.9, 0.999)) |
| Inner steps | 2 |
| Min VRAM | ~2GB (0.5B model) |
| Trainable Params | ~2.2M (0.44% of total) |

## Installation

```bash
pip install torch transformers peft bitsandbytes accelerate datasets
```

## Tests

```bash
pytest tests/test_engine.py -v
# 13/13 passed
```

## Limitations

- **Model ceiling**: A 0.5B model lacks reasoning capacity for GSM8K. Use 1.5B+ for meaningful results.
- **No correctness signal**: Entropy minimization optimizes for confidence, not accuracy. Without a verifier, TTA can become "more confidently wrong."
- **Speed**: ~10-30x slower depending on entropy threshold and inner_steps

## Known Issues

- Baseline accuracy on GSM8K is 0% for 0.5B models (expected - model is too small)
- TTA shows improvement on simple arithmetic but not yet on multi-step math

## References

- [Test-Time Learning for LLMs (ICML 2025)](https://github.com/Fhujinwu/TLM)
- [Transformer² (SakanaAI)](https://github.com/SakanaAI/self-adaptive-llms)
- [Tent: Fully Test-Time Adaptation](https://arxiv.org/abs/2006.10726)
- [LookSharp: Attention Entropy Minimization (2025)](https://arxiv.org/abs/2511.18925)

---
*Built by Griffith-7*