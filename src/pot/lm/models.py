from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LM:
    def __init__(self, name: str, device: str = "cuda", seed: int = 0):
        torch.manual_seed(seed)
        self.tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        self.m = AutoModelForCausalLM.from_pretrained(name).eval().to(device)
        self.device = device

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 64):
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.m.generate(
            **ids, 
            do_sample=False, 
            temperature=0.0, 
            top_k=1,
            max_new_tokens=max_new_tokens, 
            pad_token_id=self.tok.eos_token_id
        )
        return self.tok.decode(out[0], skip_special_tokens=True)