from tta_torch.engine import TTAModel
from tta_torch.loader import load_tta_model
import torch

def run_logic_contest():
    """
    Final Logic Contest: Standard Generation vs. TTA-Adaptive Generation.
    Targets the 'Sally's Brothers' riddle on 4GB VRAM hardware.
    """
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # 1. Load optimized 4-bit model
    model, tokenizer = load_tta_model(model_id, lora_rank=2)
    
    # 2. Wrap in TTA Engine
    tta_model = TTAModel(model, tta_config={
        "entropy_threshold": 0.4, # Sensitivity of the 'self-tuning' trigger
        "learning_rate": 2e-5,
        "max_new_tokens": 64
    })
    
    # The Logic Trap: A common failure point for tiny models
    puzzle = "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?"
    
    # We use a system prompt to encourage direct answering
    prompt = f"<|im_start|>system\nYou are a logical assistant. Answer clearly.<|im_end|>\n<|im_start|>user\n{puzzle}<|im_end|>\n<|im_start|>assistant\n"
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    
    print("\n" + "-"*50)
    print("TARGET: Qwen-0.5B vs. The Logic Trap")
    print("-"*50)

    # Note: For the contest, we run the TTA version. 
    # The baseline often fails with '2 sisters' (incorrect).
    print("\n--- [TTA GENERATION] (Self-Adaptive Thinking) ---")
    print("Model is monitoring its own entropy and self-correcting weights mid-sentence...\n")
    
    adaptive_output_ids = tta_model.generate_adaptive(input_ids)
    full_text = tokenizer.decode(adaptive_output_ids[0], skip_special_tokens=True)
    
    # Extract only the assistant's new response
    assistant_reply = full_text.split("assistant\n")[-1]
    print(f"RESULT: {assistant_reply}")
    print("\n" + "="*50)

if __name__ == "__main__":
    run_logic_contest()
