from typing import List

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


class Summarizer:
    """
    Uses Hugging Face summarization pipeline if available,
    otherwise falls back to a simple naive truncation.
    """
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        if HF_AVAILABLE:
            self.summarizer = pipeline("summarization", model=model_name)
        else:
            self.summarizer = None

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        if self.summarizer is None:
            # naive fallback
            words = text.split()
            return " ".join(words[:max_length])
        out = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return out[0]["summary_text"]
