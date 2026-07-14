TTA-Torch — Self-Tuning Inference
==================================

**TTA-Torch** is a PyTorch library for Test-Time Adaptation that adds self-tuning
inference to any HuggingFace-compatible transformer model.

Features
--------

- **Entropy-triggered updates** — automatically adapts when prediction uncertainty spikes.
- **Confidence-gated TTA** — applies adaptation only when confidence falls below a learned
  threshold, keeping stable predictions untouched.
- **Multi-pass selection** — generates multiple candidate outputs and selects the best one
  via self-consistency voting.
- **Self-consistency voting** — aggregates stochastic forward passes to choose the most
  consistent response.
- **4 GB VRAM compatible** — runs on consumer GPUs through 8-bit quantization and gradient
  checkpointing.
- **13+ architecture support via HuggingFace** — works with any model loadable through
  ``transformers.AutoModelForCausalLM``.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   api
