# TTA-Torch: Dynamic Test-Time Adaptation for LLMs

A PyTorch inference wrapper that performs **real-time weight updates** during generation. Instead of treating inference as a static forward pass, TTA-Torch treats it as a dynamic optimization problem — the model adjusts its own LoRA adapters mid-generation based on its confidence level.

## How It Works

In standard LLM inference, model weights are frozen. In TTA-Torch, the model runs a micro-optimization loop on every token:

1. **Predict**: Forward pass produces next-token logits
2. **Measure Confidence**: Calculate Shannon Entropy $H(P)$ over the distribution
3. **Evaluate Drift**: If entropy exceeds a threshold, compute KL-Divergence between the adaptive model and the frozen base model
4. **Update**: Perform a single SGD step on LoRA adapters, then re-sample from updated logits

```python
from tta_torch import TTAModel, load_tta_model

model, tokenizer = load_tta_model("Qwen/Qwen2.5-0.5B-Instruct", lora_rank=2)
tta = TTAModel(model, {"entropy_threshold": 0.3, "verbose": True})
output = tta.generate(input_ids)
```

## Features

- **Entropy-Triggered Updates**: Only adapts when the model is "confused" (high entropy), saving compute on confident tokens
- **Consistency Shield**: KL-Divergence regularization against the frozen base model prevents catastrophic drift
- **Best-of-N Selection**: Generate multiple outputs and return the one with lowest average entropy
- **Weight Reset**: Full rollback to original weights between tasks for clean benchmarking
- **4GB VRAM Compatible**: Runs on budget GPUs using 4-bit QLoRA + adapter toggling (no second model needed)

## Verified Evidence

Verbose entropy logging from a real run on RTX GPU with 4GB VRAM:

```
Loading Qwen/Qwen2.5-0.5B-Instruct...
trainable params: 270,336 || all params: 494,303,104 || trainable%: 0.0547

--- VERBOSE TTA GENERATION (Logic Puzzle) ---
  Token 0:  entropy=1.2194 > 0.3 -> TTA update
  Token 2:  entropy=1.2304 > 0.3 -> TTA update
  Token 6:  entropy=1.0896 > 0.3 -> TTA update
  Token 27: entropy=2.2730 > 0.3 -> TTA update
  Token 37: entropy=3.4443 > 0.3 -> TTA update

TTA Stats: {'updates': 15, 'generations': 0}
Entropy trace: [1.22, 0.03, 1.23, 0.77, 0.24, 0.49, 1.09, 0.34, ...]
```

**What this proves:**
- Weights genuinely update during inference (15 SGD steps in one generation)
- Entropy monitoring correctly identifies high-uncertainty tokens
- The full stack (model + gradients + optimizer) fits in under 4GB VRAM

**What this does NOT prove:**
- That TTA consistently produces *better* answers than standard inference
- That small models can match large model reasoning through TTA alone
- Statistical significance — formal benchmarks with ground-truth evaluation are needed

## Architecture

```
Input → [Forward Pass] → Logits
                           ↓
                   [Entropy Check]
                    ↓           ↓
              (Low: Skip)  (High: Adapt)
                                ↓
                        [KL-Div vs Frozen]
                                ↓
                        [SGD Step on LoRA]
                                ↓
                        [Re-Forward Pass]
                                ↓
                         [Sample Token]
```

## Technical Specifications

| Component | Value |
|-----------|-------|
| Base Model | Any HuggingFace CausalLM |
| Tested On | Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct |
| Quantization | 4-bit NF4 with double quantization |
| Adapter | LoRA (rank 2-4, targeting q/k/v/o projections) |
| Optimizer | SGD (lr=1e-5 to 3e-5) |
| Min VRAM | ~2GB (0.5B model) / ~3GB (1B model) |
| Trainable Params | 270K (0.05% of total) |

## Installation

```bash
pip install torch transformers peft bitsandbytes accelerate
```

## Tests

```bash
pytest tests/test_engine.py -v
# 13/13 passed
```

## Limitations

- **Model ceiling**: A 0.5B-1B model cannot learn facts it doesn't already know, regardless of TTA
- **Weak signal**: Single-step SGD with small learning rates produces very small weight changes per token
- **No verifier**: Without a reward model, there is no ground-truth signal — the model optimizes for consistency, not correctness
- **Speed**: ~2x slower than standard inference due to the frozen-model forward pass per token

## References

- [Test-Time Learning for LLMs (ICML 2025)](https://github.com/Fhujinwu/TLM)
- [Transformer² (SakanaAI)](https://github.com/SakanaAI/self-adaptive-llms)
- [Tent: Fully Test-Time Adaptation](https://arxiv.org/abs/2006.10726)

---
*Built by Griffith-7*
