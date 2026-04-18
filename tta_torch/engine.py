import torch
import torch.nn as nn
from transformers import PreTrainedModel
import torch.nn.functional as F

class TTAModel(nn.Module):
    """
    TTA-Torch - Dynamic Test-Time Adaptation
    Features: Weight updates during inference + Multi-pass selection
    """
    def __init__(self, model: PreTrainedModel, tta_config=None):
        super().__init__()
        self.model = model
        defaults = {
            "entropy_threshold": 0.5,
            "learning_rate": 1e-5,
            "kl_weight": 0.1,
            "max_new_tokens": 128,
            "grad_clip": 1.0,
            "n_passes": 3,
            "verbose": False,
        }
        if tta_config:
            defaults.update(tta_config)
        self.tta_config = defaults
        self.init_state = {k: v.clone() for k, v in self.model.named_parameters() if v.requires_grad}
        self.stats = {'updates': 0, 'generations': 0}

    def reset_weights(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.init_state:
                    p.copy_(self.init_state[n])
        torch.cuda.empty_cache()

    def _entropy(self, logits):
        p = F.softmax(logits, dim=-1)
        return -(p * torch.log(p + 1e-9)).sum(-1)

    @torch.enable_grad()
    def generate(self, input_ids, **kwargs):
        """Main generation with TTA"""
        current = input_ids.clone()
        lr = self.tta_config["learning_rate"]
        kl_w = self.tta_config.get("kl_weight", 0.1)
        grad_clip = self.tta_config.get("grad_clip", 1.0)
        opt = torch.optim.SGD(self.model.parameters(), lr=lr)
        max_tok = kwargs.get('max_tokens', self.tta_config.get("max_new_tokens", 128))
        ent_thresh = self.tta_config["entropy_threshold"]
        
        self.current_entropy = []
        verbose = self.tta_config.get("verbose", False)
        
        for idx in range(max_tok):
            out = self.model(current)
            logits = out.logits[:, -1, :]
            
            with self.model.disable_adapter():
                with torch.no_grad():
                    frozen = self.model(current)
                    f_logits = frozen.logits[:, -1, :]
            
            ent = self._entropy(logits).mean()
            
            # Track entropy for best-of-n selection
            if hasattr(self, 'current_entropy'):
                self.current_entropy.append(ent.item())
            
            if ent > ent_thresh:
                if verbose:
                    print(f"  Token {idx}: entropy={ent.item():.4f} > {ent_thresh} -> TTA update")
                
                # LEARNING signal: minimize entropy (make model more confident)
                ent_loss = ent
                # STABILITY signal: prevent drift from base model
                kl = F.kl_div(F.log_softmax(f_logits, dim=-1), F.softmax(logits, dim=-1), reduction='batchmean')
                # Combined: learn but stay anchored
                total_loss = ent_loss + (kl_w * kl)
                
                if total_loss > 0:
                    opt.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    opt.step()
                    self.stats['updates'] += 1
                    
                    # Re-run forward and log entropy change
                    out = self.model(current)
                    new_logits = out.logits[:, -1, :]
                    new_ent = self._entropy(new_logits).mean()
                    self.current_entropy[-1] = new_ent.item()
                    
                    if verbose:
                        change = ((ent.item() - new_ent.item()) / ent.item()) * 100
                        print(f"    -> entropy={new_ent.item():.4f} (change: {change:+.1f}%)")
                    
                    # Use the updated logits for sampling
                    logits = new_logits
            
            next_tok = torch.argmax(logits, dim=-1).unsqueeze(-1)
            current = torch.cat([current, next_tok], dim=-1)
            
            eos = getattr(self.model.config, 'eos_token_id', None)
            if eos and next_tok.item() == eos:
                break
            torch.cuda.empty_cache()
        
        return current

    @torch.enable_grad()
    def generate_best_of_n(self, input_ids, **kwargs):
        """Multi-pass: Generate N times, pick most confident (lowest entropy)"""
        n = self.tta_config.get("n_passes", 3)
        results = []
        entropies = []
        
        for i in range(n):
            self.reset_weights()
            out = self.generate(input_ids, **kwargs)
            results.append(out)
            self.stats['generations'] += 1
            
            # Collect average entropy from this generation
            if hasattr(self, 'current_entropy') and self.current_entropy:
                avg_ent = sum(self.current_entropy) / len(self.current_entropy)
                entropies.append(avg_ent)
        
        # Pick result with lowest entropy (most confident)
        if entropies:
            best_idx = min(range(len(entropies)), key=lambda i: entropies[i])
            return results[best_idx]
        
        return results[0]

    def generate_adaptive(self, input_ids, **kwargs):
        return self.generate(input_ids, **kwargs)

    def generate_best(self, input_ids, **kwargs):
        return self.generate_best_of_n(input_ids, **kwargs)