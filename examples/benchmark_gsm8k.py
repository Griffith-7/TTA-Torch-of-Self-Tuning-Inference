import json
import re
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tta_torch.engine import TTAModel
from tta_torch.loader import load_tta_model

def extract_answer(text):
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None

def gold(sample):
    return sample["answer"].split("####")[-1].strip().replace(",", "")

def run_bench(model_id="Qwen/Qwen2.5-0.5B-Instruct", n=3, max_new=60):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        return
    
    print(f"Loading {model_id}...")
    model, tok = load_tta_model(model_id, lora_rank=4)
    
    config = {
        "entropy_threshold": 0.4,
        "learning_rate": 1e-4,
        "kl_weight": 0.1,
        "inner_steps": 2,
        "max_new_tokens": max_new,
        "verbose": False,
    }
    tta = TTAModel(model, config)
    
    print(f"Loading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main")["test"]
    questions = [ds[i] for i in range(min(n, len(ds)))]
    
    base_correct = 0
    tta_correct = 0
    t_base = 0.0
    t_tta = 0.0
    log = []
    
    for i, ex in enumerate(questions):
        prompt = f"Question: {ex['question']}\nAnswer: Let's think step by step. "
        ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        g = gold(ex)
        
        t0 = time.time()
        with model.disable_adapter():
            b = model.generate(ids, max_new_tokens=max_new)
        t_base += time.time() - t0
        b_txt = tok.decode(b[0][ids.shape[1]:], skip_special_tokens=True)
        b_ans = extract_answer(b_txt)
        b_ok = b_ans == g if b_ans else False
        
        tta.reset_weights()
        t0 = time.time()
        a = tta.generate(ids)
        t_tta += time.time() - t0
        a_txt = tok.decode(a[0][ids.shape[1]:], skip_special_tokens=True)
        a_ans = extract_answer(a_txt)
        a_ok = a_ans == g if a_ans else False
        
        base_correct += b_ok
        tta_correct += a_ok
        log.append({"i": i, "gold": g, "base": b_ans, "base_ok": b_ok, "tta": a_ans, "tta_ok": a_ok})
        print(f"[{i+1}/{n}] base={b_ok} tta={a_ok} | base: {base_correct}/{i+1} tta: {tta_correct}/{i+1}")
    
    out = {
        "n": n, "model": model_id,
        "baseline_acc": base_correct / n,
        "tta_acc": tta_correct / n,
        "delta_pp": (tta_correct - base_correct) / n * 100,
        "base_time_s": t_base, "tta_time_s": t_tta,
        "detail": log
    }
    
    with open("benchmark_results.json", "w") as f:
        json.dump(out, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"BASELINE: {base_correct}/{n} ({base_correct/n*100:.1f}%)")
    print(f"TTA:      {tta_correct}/{n} ({tta_correct/n*100:.1f}%)")
    delta = (tta_correct - base_correct) / n * 100
    print(f"DELTA = {delta:+.1f} percentage points")
    print(f"Time: baseline={t_base:.1f}s tta={t_tta:.1f}s")
    print(f"{'='*50}")
    print(f"\nResults saved to benchmark_results.json")
    
    return out

if __name__ == "__main__":
    print("Running GSM8K benchmark...")
    run_bench()
    print("Done!")