from typing import Tuple, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


class TfidfFeatureExtractor:
    def __init__(self, max_features: int = 20000, ngram_range=(1, 2),
                 min_df: int = 3, max_df: float = 0.9):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True
        )

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: List[str]) -> np.ndarray:
        return self.vectorizer.transform(texts)

    def get_feature_names(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()


class EmbeddingExtractor:
    """
    Wrapper around SentenceTransformer. If model download fails,
    you can swap this for a lighter local model or TF-IDF average.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)
