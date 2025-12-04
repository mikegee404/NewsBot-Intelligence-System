from typing import Optional

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


class TextGenerator:
    """
    Optional text generator using a small HF model.
    For class project, you can call this for:
    - headline suggestions
    - follow-up questions
    """
    def __init__(self, model_name: str = "gpt2"):
        if HF_AVAILABLE:
            self.generator = pipeline("text-generation", model=model_name)
        else:
            self.generator = None

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        if self.generator is None:
            return f"[GENERATION DISABLED] Prompt: {prompt}"
        out = self.generator(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1)
        return out[0]["generated_text"]
