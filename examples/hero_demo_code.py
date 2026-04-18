from tta_torch.engine import TTAModel
from tta_torch.loader import load_tta_model
import torch

def run_coding_hero_demo():
    """
    The 'Hero Demo' for TTA-Torch: Self-Correcting Python Syntax.
    Shows the model detecting a potential logic/syntax error and tuning itself.
    """
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model, tokenizer = load_tta_model(model_id, lora_rank=2)
    
    # Phase 2 Config: High KL-weight to keep code valid
    tta_model = TTAModel(model, tta_config={
        "entropy_threshold": 0.3, 
        "learning_rate": 3e-5,
        "kl_weight": 0.2,
        "max_new_tokens": 80
    })

    # A prompt designed to make small models trip up on nested structures
    coding_task = "Write a one-line Python list comprehension that filters a list of dictionaries for 'id' and 'value', but only if 'value' is greater than 10, and handles missing 'value' keys using a default of 0."
    
    formatted_prompt = f"<|im_start|>system\nYou are an expert Python coder.<|im_end|>\n<|im_start|>user\n{coding_task}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to("cuda")

    print("\n" + "-"*60)
    print("DEMO: SELF-CORRECTING CODE SYNTAX (TTA-TORCH)")
    print("-"*60)

    print("\n[STEP 1] Running Standard Code Generation...")
    with model.disable_adapter():
        baseline_ids = model.generate(input_ids, max_new_tokens=80)
        baseline_reply = tokenizer.decode(baseline_ids[0], skip_special_tokens=True).split("assistant\n")[-1]
        print(f"BASELINE CODE: {baseline_reply.strip()}")

    # Reset weights for fair contest
    tta_model.reset_weights()

    print("\n[STEP 2] Running TTA-Enhanced Generation...")
    print("(Model is monitoring logit entropy for mid-sentence self-correction...)\n")
    adaptive_ids = tta_model.generate_adaptive(input_ids)
    adaptive_reply = tokenizer.decode(adaptive_ids[0], skip_special_tokens=True).split("assistant\n")[-1]
    print(f"TTA-OPTIMIZED CODE: {adaptive_reply.strip()}")
    print("-" * 60)

if __name__ == "__main__":
    run_coding_hero_demo()
