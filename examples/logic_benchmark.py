from tta_torch.engine import TTAModel
from tta_torch.loader import load_tta_model
import torch

def benchmark_tta():
    """
    Test-Time Adaptation Benchmark (Logic Olympics).
    Compares Standard Inference vs. Phase 2 TTA-Enhanced Inference.
    """
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model, tokenizer = load_tta_model(model_id, lora_rank=2)
    
    # Wrap in our Refined TTA Engine (Phase 2)
    tta_model = TTAModel(model, tta_config={
        "entropy_threshold": 0.35, # Aggressive focus on accuracy
        "learning_rate": 2e-5,
        "kl_weight": 0.15,         # The 'Consistency Shield' strength
        "max_new_tokens": 100
    })

    # The Logic Traps (Puzzles that usually trick 0.5B models)
    puzzles = [
        {
            "name": "Sally's Brothers",
            "prompt": "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?"
        },
        {
            "name": "The Bat and Ball",
            "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? (Answer with just the price)"
        },
        {
            "name": "The Lily Pad Riddle",
            "prompt": "In a lake, there's a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how many days does it take for the patch to cover half the lake?"
        }
    ]

    print("\n" + "="*60)
    print("TTA-TORCH PHASE 2: THE LOGIC OLYMPICS")
    print("="*60)

    for i, puzzle in enumerate(puzzles):
        print(f"\nPUZZLE {i+1}: {puzzle['name']}")
        print(f"QUESTION: {puzzle['prompt']}")
        
        # Format for Qwen-Instruct
        formatted_prompt = f"<|im_start|>system\nYou are a logical assistant. Focus on precision.<|im_end|>\n<|im_start|>user\n{puzzle['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to("cuda")

        # 1. Baseline Generation (Disable Adapters)
        print("\n[SCENARIO A] Standard Inference (Static Brain)")
        with model.disable_adapter():
            baseline_ids = model.generate(input_ids, max_new_tokens=64)
            baseline_reply = tokenizer.decode(baseline_ids[0], skip_special_tokens=True).split("assistant\n")[-1]
            print(f"BASE RESULT: {baseline_reply.strip()}")

        # 2. TTA Generation (Phase 2 Consistency Shield)
        print("\n[SCENARIO B] TTA-Torch Phase 2 (Adaptive Reasoning)")
        adaptive_ids = tta_model.generate_adaptive(input_ids)
        adaptive_reply = tokenizer.decode(adaptive_ids[0], skip_special_tokens=True).split("assistant\n")[-1]
        print(f"TTA RESULT: {adaptive_reply.strip()}")
        print("-" * 30)

if __name__ == "__main__":
    benchmark_tta()
