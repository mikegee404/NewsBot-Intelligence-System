from typing import List, Dict, Tuple

import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError as e:
    raise RuntimeError("spaCy model 'en_core_web_sm' not found. "
                       "Run: python -m spacy download en_core_web_sm") from e


class NERExtractor:
    def extract(self, text: str) -> List[Tuple[str, str]]:
        doc = nlp(text or "")
        return [(ent.text, ent.label_) for ent in doc.ents]

    def extract_filtered(self, text: str,
                         whitelist=("PERSON", "ORG", "GPE", "DATE", "MONEY")) -> List[Tuple[str, str]]:
        doc = nlp(text or "")
        return [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in whitelist]

    def entity_stats(self, texts: List[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for t in texts:
            for ent, label in self.extract(t):
                key = f"{label}:{ent}"
                counts[key] = counts.get(key, 0) + 1
        return counts
