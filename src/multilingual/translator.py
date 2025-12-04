from typing import Optional
from googletrans import Translator


class NewsTranslator:
    def __init__(self):
        self.translator = Translator()

    def translate(self, text: str, dest: str = "en", src: Optional[str] = None) -> str:
        if not text:
            return ""
        res = self.translator.translate(text, dest=dest, src=src)
        return res.text
