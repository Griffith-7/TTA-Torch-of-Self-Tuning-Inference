Usage
=====

Quick Start
-----------

Load a HuggingFace model, wrap it with ``TTAModel``, and run inference with
confidence-gated test-time adaptation:

.. code-block:: python

   import torch
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from tta_torch import TTAModel

   model_id = "meta-llama/Llama-2-7b-hf"

   tokenizer = AutoTokenizer.from_pretrained(model_id)
   base_model = AutoModelForCausalLM.from_pretrained(
       model_id,
       load_in_8bit=True,
       device_map="auto",
   )

   tta_model = TTAModel(
       model=base_model,
       tokenizer=tokenizer,
       entropy_threshold=0.8,
       num_candidates=3,
   )

   inputs = tokenizer("The capital of France is", return_tensors="pt").to(base_model.device)
   output = tta_model.generate(**inputs, max_new_tokens=64)
   print(tokenizer.decode(output[0], skip_special_tokens=True))

CLI Usage
---------

TTA-Torch ships with a command-line interface for common tasks.

Generate text
~~~~~~~~~~~~~

.. code-block:: bash

   tta-torch generate \
       --model meta-llama/Llama-2-7b-hf \
       --prompt "The capital of France is" \
       --max-new-tokens 64 \
       --entropy-threshold 0.8

Run a benchmark
~~~~~~~~~~~~~~~

.. code-block:: bash

   tta-torch benchmark \
       --model meta-llama/Llama-2-7b-hf \
       --dataset lambada \
       --num-samples 200

Clean cached adapters
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   tta-torch clean --all
