from __future__ import annotations
from typing import Optional

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None


class HFLocalModel:
    """
    Minimal HF wrapper (optional dependency).
    Deterministic decode (greedy) for repeatability.
    """

    def __init__(self, model_name: str, device: Optional[str] = None, max_new_tokens: int = 128):
        if AutoModelForCausalLM is None:
            raise RuntimeError("Transformers not installed. Install with `pip install '.[hf]'`.")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # Greedy decode
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, do_sample=False, temperature=None, num_beams=1, max_new_tokens=self.max_new_tokens
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Return only the continuation beyond the prompt tokens
        # (or return full text—choose one policy; we keep full for simplicity)
        return text

    def name(self) -> str:
        return f"hf:{self.model_name}"